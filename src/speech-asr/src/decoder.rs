//! Independent scalar Qwen3-ASR causal decoder and transactional session state.
//!
//! The reviewed graph is described in `rust-decoder.json`. This implementation
//! exposes token/audio prefill, GQA, per-head Q/K `RMSNorm`, `NeoX` `RoPE`, `SwiGLU`,
//! tied output projection and complete cache reset. Its explicitly separate
//! scalar arithmetic is not a claim of native quantized/flash-attention parity.
//! Production continues to select the native engine until encoder, weight
//! intake, independent numerical and actual-model acceptance are complete.
#![forbid(unsafe_code)]
#![allow(
    clippy::suboptimal_flops,
    reason = "Separate scalar operations define this independent reference's accumulation order."
)]

use crate::{
    packed::{self, PreparedBatch, WeightMatrix, WeightSlice},
    protocol::Prompt,
    row_workers, threads,
};
use std::{
    io::{self, Read, Seek},
    sync::Arc,
};

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid Rust Qwen3-ASR decoder input, weights or state.",
    )
}

fn check(cancel: &mut impl FnMut() -> bool) -> io::Result<()> {
    if cancel() {
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "Rust Qwen3-ASR decoder cancelled.",
        ))
    } else {
        Ok(())
    }
}

fn zeros(count: usize) -> io::Result<Vec<f32>> {
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(io::Error::other)?;
    result.resize(count, 0.0);
    Ok(result)
}

fn finite(values: &[f32]) -> io::Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid())
    }
}

/// GGUF row-major F32 or `Q8_0` projection, retaining compressed Q8 weights.
pub struct Projection {
    input: usize,
    output: usize,
    data: Storage,
}

enum Storage {
    Float(Vec<f32>),
    Q8(Vec<u8>),
}

impl Projection {
    // ggml_conv_2d uses the F16 im2col dot provider for admitted F32/F16
    // kernels. Its F32 kernel rows are converted to that provider's F16
    // operand type too. Do this once when admitting the encoder, rather than
    // converting the same immutable kernel at every spatial position.
    pub(super) fn round_convolution_weights(&mut self) -> io::Result<()> {
        let Storage::Float(values) = &mut self.data else {
            return Err(invalid());
        };
        for value in values {
            *value = binary16_operand(*value)?;
        }
        Ok(())
    }

    /// Return `(input width, output rows)` for checked graph construction.
    #[must_use]
    pub const fn dimensions(&self) -> (usize, usize) {
        (self.input, self.output)
    }
    /// Admit little-endian IEEE binary16 rows, expanding only this projection.
    ///
    /// # Errors
    /// Rejects invalid shape, incomplete bytes, allocation failure or nonfinite weights.
    pub fn f16(input: usize, output: usize, bytes: &[u8]) -> io::Result<Self> {
        let count = extent(input, output)?;
        if bytes.len() != count * 2 {
            return Err(invalid());
        }
        let mut values = zeros(count)?;
        for (value, bytes) in values.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *value = half(u16::from_le_bytes([bytes[0], bytes[1]]))?;
        }
        Self::f32(input, output, values)
    }
    /// Admit finite F32 values in `[output, input]` row order.
    ///
    /// # Errors
    /// Rejects invalid shape, nonfinite data or more than 8 GiB of values.
    pub fn f32(input: usize, output: usize, data: Vec<f32>) -> io::Result<Self> {
        let count = extent(input, output)?;
        if count != data.len() {
            return Err(invalid());
        }
        finite(&data)?;
        Ok(Self {
            input,
            output,
            data: Storage::Float(data),
        })
    }

    /// Admit `Q8_0` blocks: little-endian half scale followed by 32 signed bytes.
    ///
    /// Activations stay F32; this is a separately identified scalar path rather
    /// than GGML's activation-requantizing optimized matrix multiplication.
    ///
    /// # Errors
    /// Rejects incomplete blocks, wrong byte length or nonfinite half scales.
    pub fn q8(input: usize, output: usize, data: Vec<u8>) -> io::Result<Self> {
        let count = extent(input, output)?;
        if !input.is_multiple_of(32) || data.len() != count / 32 * 34 {
            return Err(invalid());
        }
        for block in data.as_chunks::<34>().0 {
            let _ = half(u16::from_le_bytes([block[0], block[1]]))?;
        }
        Ok(Self {
            input,
            output,
            data: Storage::Q8(data),
        })
    }

    fn row(&self, row: usize) -> io::Result<Vec<f32>> {
        let mut output = zeros(self.input)?;
        self.row_into(row, &mut output)?;
        Ok(output)
    }

    fn row_into(&self, row: usize, output: &mut [f32]) -> io::Result<()> {
        if row >= self.output || output.len() != self.input {
            return Err(invalid());
        }
        match &self.data {
            Storage::Float(values) => {
                output.copy_from_slice(&values[row * self.input..(row + 1) * self.input]);
            }
            Storage::Q8(bytes) => {
                let row_bytes = self.input / 32 * 34;
                for (block, target) in bytes[row * row_bytes..(row + 1) * row_bytes]
                    .as_chunks::<34>()
                    .0
                    .iter()
                    .zip(output.as_chunks_mut::<32>().0)
                {
                    let [s0, s1, ref qs @ ..] = *block;
                    let scale = half(u16::from_le_bytes([s0, s1]))?;
                    for (value, &byte) in target.iter_mut().zip(qs) {
                        *value = scale * f32::from(byte.cast_signed());
                    }
                }
            }
        }
        finite(output)
    }

    pub(super) fn as_weight_matrix(&self) -> WeightMatrix<'_> {
        let data = match &self.data {
            Storage::Float(values) => WeightSlice::Float(values),
            Storage::Q8(bytes) => WeightSlice::Q8(bytes),
        };
        WeightMatrix {
            input: self.input,
            output: self.output,
            data,
        }
    }

    /// Compute every output row into `out` across `1x4` SIMD tiles and bounded workers.
    pub(super) fn multiply_into(
        &self,
        input: &[f32],
        workers: usize,
        out: &mut [f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        if input.len() != self.input || out.len() != self.output {
            return Err(invalid());
        }
        finite(input)?;
        for _ in 0..self.output {
            check(cancel)?;
        }
        packed::multiply_single_into(self.as_weight_matrix(), input, workers, out);
        finite(out)
    }

    /// Compute every output row with bounded workers, allocating only the result vector.
    pub(super) fn multiply_with_threads(
        &self,
        input: &[f32],
        workers: usize,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Vec<f32>> {
        let mut output = zeros(self.output)?;
        self.multiply_into(input, workers, &mut output, cancel)?;
        Ok(output)
    }

    /// Compute batched `[batch, input]` rows using a shared `PreparedBatch` and `4x8` SIMD tiles.
    pub(super) fn multiply_batch_prepared_into(
        &self,
        prepared: &PreparedBatch,
        raw_input: &[f32],
        workers: usize,
        scratch: &mut Vec<f32>,
        out: &mut [f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        let batch = prepared.batch();
        if prepared.input() != self.input
            || raw_input.len() != batch.checked_mul(self.input).ok_or_else(invalid)?
            || out.len() != batch.checked_mul(self.output).ok_or_else(invalid)?
        {
            return Err(invalid());
        }
        finite(raw_input)?;
        for _ in 0..(batch + self.output) {
            check(cancel)?;
        }
        packed::multiply_batch_prepared_into(
            self.as_weight_matrix(),
            prepared,
            raw_input,
            workers,
            scratch,
            out,
        );
        finite(out)
    }

    /// Compute every output row with a cancellation check before each row.
    ///
    /// # Errors
    /// Rejects bad/nonfinite input, nonfinite output, allocation failure or cancellation.
    pub fn multiply(
        &self,
        input: &[f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Vec<f32>> {
        self.multiply_with_threads(input, 1, cancel)
    }

    const fn shape(&self, input: usize, output: usize) -> bool {
        self.input == input && self.output == output
    }
}

fn extent(input: usize, output: usize) -> io::Result<usize> {
    input
        .checked_mul(output)
        .filter(|count| input > 0 && output > 0 && *count <= 2 * 1024 * 1024 * 1024)
        .ok_or_else(invalid)
}

fn half(bits: u16) -> io::Result<f32> {
    let value = half::f16::from_bits(bits);
    if !value.is_finite() {
        return Err(invalid());
    }
    Ok(value.to_f32())
}

/// Cache scalar policy, selected explicitly when creating a decoder model.
#[derive(Clone, Copy, Debug)]
pub enum CachePrecision {
    /// Preserve F32 K/V bits.
    F32,
    /// Round K/V to IEEE binary16 with nearest-even ties before attention.
    F16,
}

fn cache_value(value: f32, precision: CachePrecision) -> io::Result<f32> {
    if !value.is_finite() {
        return Err(invalid());
    }
    if matches!(precision, CachePrecision::F32) {
        return Ok(value);
    }
    binary16_operand(value)
}

// Shared exact nearest-even conversion for F16 cache and convolution operands.
// Reject finite values that would narrow to infinity before graph publication.
pub(super) fn binary16_operand(value: f32) -> io::Result<f32> {
    if !value.is_finite() {
        return Err(invalid());
    }
    let narrowed = half::f16::from_f32(value);
    if !narrowed.is_finite() {
        Err(invalid())
    } else {
        Ok(narrowed.to_f32())
    }
}

/// Reviewed single-stream geometry for a Qwen3 causal decoder.
#[derive(Clone, Copy)]
pub struct Config {
    /// Residual-stream and audio-row width.
    pub hidden: usize,
    /// Number of query heads.
    pub heads: usize,
    /// Number of shared key/value heads.
    pub kv_heads: usize,
    /// Even number of values per attention head.
    pub head_dim: usize,
    /// `SwiGLU` intermediate width.
    pub intermediate: usize,
    /// Bounded session context, at most 8,192 positions.
    pub context: usize,
    /// Positive finite RMS normalization epsilon.
    pub epsilon: f32,
    /// Positive finite `NeoX` rotation base.
    pub rope_theta: f32,
    /// Explicit K/V storage rounding policy.
    pub cache_precision: CachePrecision,
}

/// One complete bias-free Qwen3 block with per-head Q/K normalization.
pub struct Block {
    /// Input `RMSNorm` weights, `[hidden]`.
    pub attention_norm: Vec<f32>,
    /// Post-attention `RMSNorm` weights, `[hidden]`.
    pub ffn_norm: Vec<f32>,
    /// Q projection, `[heads * head_dim, hidden]`.
    pub query: Projection,
    /// K projection, `[kv_heads * head_dim, hidden]`.
    pub key: Projection,
    /// V projection, `[kv_heads * head_dim, hidden]`.
    pub value: Projection,
    /// Attention output projection, `[hidden, heads * head_dim]`.
    pub attention_out: Projection,
    /// Q `RMSNorm` weights, shared across heads, `[head_dim]`.
    pub query_norm: Vec<f32>,
    /// K `RMSNorm` weights, shared across heads, `[head_dim]`.
    pub key_norm: Vec<f32>,
    /// `SwiGLU` gate projection, `[intermediate, hidden]`.
    pub gate: Projection,
    /// `SwiGLU` up projection, `[intermediate, hidden]`.
    pub up: Projection,
    /// `SwiGLU` down projection, `[hidden, intermediate]`.
    pub down: Projection,
}

/// Immutable admitted decoder weights, sharing the embedding and output head.
pub struct Model {
    config: Config,
    embedding: Projection,
    blocks: Vec<Block>,
    output_norm: Vec<f32>,
    inv_freq: Vec<f32>,
    decode_threads: usize,
    batch_threads: usize,
}

impl Model {
    /// Return the immutable residual width, vocabulary and bounded context.
    #[must_use]
    pub const fn dimensions(&self) -> (usize, usize, usize) {
        (
            self.config.hidden,
            self.embedding.output,
            self.config.context,
        )
    }

    /// Configure bounded worker counts for single-token decode and batched prefill.
    ///
    /// # Errors
    /// Rejects thread counts outside `1..=256`.
    pub fn with_threads(mut self, decode_threads: usize, batch_threads: usize) -> io::Result<Self> {
        self.decode_threads = threads::validate_thread_count(decode_threads)?;
        self.batch_threads = threads::validate_thread_count(batch_threads)?;
        Ok(self)
    }

    /// Load the complete decoder from an admitted Qwen3-ASR GGUF index.
    ///
    /// The caller owns digest verification and immutable artifact guards. This
    /// reads decoder weights only, accepting `F32/F16/Q8_0` projections and F32
    /// normalization vectors. It does not execute inference or an encoder.
    ///
    /// # Errors
    /// Rejects malformed/unsupported tensors, scalar geometry, nonfinite weights,
    /// cancellation and I/O/allocation failures.
    pub fn load(
        source: &mut (impl Read + Seek),
        length: u64,
        precision: CachePrecision,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Self> {
        crate::intake::decoder(source, length, precision, cancel)
    }
    /// Validate all layer dimensions before admitting a decoder session.
    ///
    /// # Errors
    /// Rejects unsupported/overflowing geometry, nonfinite normalization weights,
    /// invalid scalar parameters, untied/incorrect embedding or incomplete layers.
    pub fn new(
        config: Config,
        embedding: Projection,
        blocks: Vec<Block>,
        output_norm: Vec<f32>,
    ) -> io::Result<Self> {
        let c = config;
        if !(1..=8192).contains(&c.hidden)
            || !(1..=128).contains(&c.heads)
            || c.kv_heads == 0
            || !c.heads.is_multiple_of(c.kv_heads)
            || c.head_dim == 0
            || c.head_dim > 256
            || !c.head_dim.is_multiple_of(2)
            || !(1..=65_536).contains(&c.intermediate)
            || !(1..=8192).contains(&c.context)
            || !(1..=128).contains(&blocks.len())
            || embedding.input != c.hidden
            || embedding.output > 262_144
            || !c.epsilon.is_finite()
            || c.epsilon <= 0.0
            || !c.rope_theta.is_finite()
            || c.rope_theta <= 0.0
        {
            return Err(invalid());
        }
        norm_weights(&output_norm, c.hidden)?;
        for block in &blocks {
            norm_weights(&block.attention_norm, c.hidden)?;
            norm_weights(&block.ffn_norm, c.hidden)?;
            norm_weights(&block.query_norm, c.head_dim)?;
            norm_weights(&block.key_norm, c.head_dim)?;
            if !block.query.shape(c.hidden, c.heads * c.head_dim)
                || !block.key.shape(c.hidden, c.kv_heads * c.head_dim)
                || !block.value.shape(c.hidden, c.kv_heads * c.head_dim)
                || !block.attention_out.shape(c.heads * c.head_dim, c.hidden)
                || !block.gate.shape(c.hidden, c.intermediate)
                || !block.up.shape(c.hidden, c.intermediate)
                || !block.down.shape(c.intermediate, c.hidden)
            {
                return Err(invalid());
            }
        }
        let half_dim = c.head_dim / 2;
        let mut inv_freq = zeros(half_dim)?;
        for (index, freq) in inv_freq.iter_mut().enumerate() {
            *freq = c.rope_theta.powf(-scalar(2 * index) / scalar(c.head_dim));
        }
        finite(&inv_freq)?;
        Ok(Self {
            config,
            embedding,
            blocks,
            output_norm,
            inv_freq,
            decode_threads: 1,
            batch_threads: 1,
        })
    }

    /// Create an empty session bound to this exact immutable model owner.
    #[must_use]
    pub fn session(self: &Arc<Self>) -> Session {
        Session {
            model: Arc::clone(self),
            position: 0,
            layers: (0..self.blocks.len())
                .map(|_| LayerState {
                    keys: Vec::new(),
                    values: Vec::new(),
                })
                .collect(),
            workspace: DecoderWorkspace::default(),
        }
    }
}

struct LayerState {
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Default)]
struct DecoderWorkspace {
    single_hidden: Vec<f32>,
    single_norm: Vec<f32>,
    single_q: Vec<f32>,
    single_k: Vec<f32>,
    single_v: Vec<f32>,
    single_attn: Vec<f32>,
    single_proj: Vec<f32>,
    single_gate: Vec<f32>,
    single_up: Vec<f32>,
    single_scores: Vec<f32>,
    single_provisional_keys: Vec<f32>,
    single_provisional_values: Vec<f32>,
    batch_a: Vec<f32>,
    batch_norm: Vec<f32>,
    batch_q: Vec<f32>,
    batch_k: Vec<f32>,
    batch_v: Vec<f32>,
    batch_attn: Vec<f32>,
    batch_proj: Vec<f32>,
    batch_gate: Vec<f32>,
    batch_up: Vec<f32>,
    transposed_scratch: Vec<f32>,
}

impl DecoderWorkspace {
    fn clear(&mut self) {
        self.single_hidden.clear();
        self.single_norm.clear();
        self.single_q.clear();
        self.single_k.clear();
        self.single_v.clear();
        self.single_attn.clear();
        self.single_proj.clear();
        self.single_gate.clear();
        self.single_up.clear();
        self.single_scores.clear();
        self.single_provisional_keys.clear();
        self.single_provisional_values.clear();
        self.batch_a.clear();
        self.batch_norm.clear();
        self.batch_q.clear();
        self.batch_k.clear();
        self.batch_v.clear();
        self.batch_attn.clear();
        self.batch_proj.clear();
        self.batch_gate.clear();
        self.batch_up.clear();
        self.transposed_scratch.clear();
    }
}

fn ensure_len(buffer: &mut Vec<f32>, len: usize) -> io::Result<()> {
    if buffer.capacity() < len {
        buffer
            .try_reserve_exact(len - buffer.len())
            .map_err(io::Error::other)?;
    }
    buffer.resize(len, 0.0);
    Ok(())
}

/// One isolated mutable decoder session; no cache is shared between sessions.
pub struct Session {
    model: Arc<Model>,
    position: usize,
    layers: Vec<LayerState>,
    workspace: DecoderWorkspace,
}

impl Session {
    /// Number of immutable decoder layers whose complete K/V must be observed.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Number of successfully published token/audio positions.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Borrow complete layer K/V in `[position, kv_head, head_dim]` order.
    #[must_use]
    pub fn cache(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        self.layers
            .get(layer)
            .map(|state| (state.keys.as_slice(), state.values.as_slice()))
    }

    /// Discard every layer's K/V and the position, preserving reusable capacity.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.keys.clear();
            layer.values.clear();
        }
        self.workspace.clear();
        self.position = 0;
    }

    /// Decode a vocabulary embedding and publish one complete position.
    ///
    /// # Errors
    /// Rejects invalid token IDs or the conditions from [`Self::step_audio`].
    pub fn step(&mut self, id: u32, cancel: &mut impl FnMut() -> bool) -> io::Result<Vec<f32>> {
        self.step_token(id, true, cancel)
    }

    fn step_token(
        &mut self,
        id: u32,
        project_logits: bool,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Vec<f32>> {
        check(cancel)?;
        let row = self
            .model
            .embedding
            .row(usize::try_from(id).map_err(|_| invalid())?)?;
        self.advance(&row, project_logits, cancel)
    }

    /// Decode one encoder row or token embedding, then return tied-head logits.
    ///
    /// Every layer's K/V remains provisional until all layers and logits are
    /// finite and cancellation is checked. Errors and unwinding preserve the
    /// exact previously committed cache and position; no partial append escapes.
    ///
    /// # Errors
    /// Rejects wrong/nonfinite rows, full context, allocation faults, arithmetic
    /// overflow/nonfinite output and cancellation.
    pub fn step_audio(
        &mut self,
        row: &[f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Vec<f32>> {
        self.advance(row, true, cancel)
    }

    fn advance(
        &mut self,
        row: &[f32],
        project_logits: bool,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Vec<f32>> {
        let model = &self.model;
        let c = model.config;
        if self.position >= c.context || row.len() != c.hidden {
            return Err(invalid());
        }
        finite(row)?;
        check(cancel)?;
        let kv_width = c.kv_heads * c.head_dim;
        let q_width = c.heads * c.head_dim;
        let ws = &mut self.workspace;
        ensure_len(&mut ws.single_hidden, c.hidden)?;
        ws.single_hidden.copy_from_slice(row);
        ensure_len(&mut ws.single_norm, c.hidden)?;
        ensure_len(&mut ws.single_q, q_width)?;
        ensure_len(&mut ws.single_k, kv_width)?;
        ensure_len(&mut ws.single_v, kv_width)?;
        ensure_len(&mut ws.single_attn, q_width)?;
        ensure_len(&mut ws.single_proj, c.hidden)?;
        ensure_len(&mut ws.single_gate, c.intermediate)?;
        ensure_len(&mut ws.single_up, c.intermediate)?;
        ensure_len(&mut ws.single_scores, self.position + 1)?;
        let prov_total = self.layers.len() * kv_width;
        ensure_len(&mut ws.single_provisional_keys, prov_total)?;
        ensure_len(&mut ws.single_provisional_values, prov_total)?;

        for (layer_idx, (block, state)) in model.blocks.iter().zip(&self.layers).enumerate() {
            check(cancel)?;
            forward_into(block, state, model, self.position, ws, cancel)?;
            ws.single_provisional_keys[layer_idx * kv_width..(layer_idx + 1) * kv_width]
                .copy_from_slice(&ws.single_k);
            ws.single_provisional_values[layer_idx * kv_width..(layer_idx + 1) * kv_width]
                .copy_from_slice(&ws.single_v);
        }
        rms_into(
            &ws.single_hidden,
            &model.output_norm,
            c.epsilon,
            &mut ws.single_norm,
        )?;
        let logits = if project_logits {
            model
                .embedding
                .multiply_with_threads(&ws.single_norm, model.decode_threads, cancel)?
        } else {
            Vec::new()
        };
        check(cancel)?;
        for state in &mut self.layers {
            state
                .keys
                .try_reserve_exact(kv_width)
                .map_err(io::Error::other)?;
            state
                .values
                .try_reserve_exact(kv_width)
                .map_err(io::Error::other)?;
        }
        for (layer_idx, state) in self.layers.iter_mut().enumerate() {
            let k_slice =
                &ws.single_provisional_keys[layer_idx * kv_width..(layer_idx + 1) * kv_width];
            let v_slice =
                &ws.single_provisional_values[layer_idx * kv_width..(layer_idx + 1) * kv_width];
            state.keys.extend_from_slice(k_slice);
            state.values.extend_from_slice(v_slice);
        }
        self.position += 1;
        Ok(logits)
    }

    /// Start a fresh prompt, replacing only its contiguous audio placeholder rows.
    ///
    /// On failure or unwind the entire attempted prompt is reset, including
    /// successful earlier rows. On success the complete prompt remains cached.
    /// Only the final prompt row projects to vocabulary logits; earlier rows
    /// retain the complete block, normalization and transactional cache checks.
    ///
    /// # Errors
    /// Rejects audio geometry, context or any token/step failure.
    pub fn prefill(
        &mut self,
        prompt: &Prompt,
        audio: &[f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Vec<f32>> {
        self.reset();
        let mut guard = ResetGuard {
            session: self,
            success: false,
        };
        let total = prompt.ids().len();
        let model = Arc::clone(&guard.session.model);
        let c = model.config;
        let range = prompt.audio_rows();
        let hidden = c.hidden;
        if audio.len() != range.len().checked_mul(hidden).ok_or_else(invalid)? || total > c.context
        {
            return Err(invalid());
        }
        finite(audio)?;
        if total == 0 {
            guard.success = true;
            return Ok(Vec::new());
        }
        let logits = prefill_inner(
            &model,
            &mut guard.session.layers,
            &mut guard.session.workspace,
            prompt,
            audio,
            cancel,
        )?;
        guard.session.position = total;
        guard.success = true;
        Ok(logits)
    }
}

fn prefill_inner(
    model: &Model,
    layers: &mut [LayerState],
    ws: &mut DecoderWorkspace,
    prompt: &Prompt,
    audio: &[f32],
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Vec<f32>> {
    let c = model.config;
    let total = prompt.ids().len();
    let hidden = c.hidden;
    let kv_width = c.kv_heads * c.head_dim;
    let q_width = c.heads * c.head_dim;
    let workers = model.batch_threads;
    let range = prompt.audio_rows();

    ensure_len(&mut ws.batch_a, total * hidden)?;
    ensure_len(&mut ws.batch_norm, total * hidden)?;
    ensure_len(&mut ws.batch_q, total * q_width)?;
    ensure_len(&mut ws.batch_k, total * kv_width)?;
    ensure_len(&mut ws.batch_v, total * kv_width)?;
    ensure_len(&mut ws.batch_attn, total * q_width)?;
    ensure_len(&mut ws.batch_proj, total * hidden)?;
    ensure_len(&mut ws.batch_gate, total * c.intermediate)?;
    ensure_len(&mut ws.batch_up, total * c.intermediate)?;

    for (pos, &id) in prompt.ids().iter().enumerate() {
        check(cancel)?;
        let dst = &mut ws.batch_a[pos * hidden..(pos + 1) * hidden];
        if range.contains(&pos) {
            let start = (pos - range.start) * hidden;
            dst.copy_from_slice(&audio[start..start + hidden]);
        } else {
            let token_id = usize::try_from(id).map_err(|_| invalid())?;
            model.embedding.row_into(token_id, dst)?;
        }
    }
    finite(&ws.batch_a)?;

    let kv_total = total * kv_width;
    for state in &mut *layers {
        state.keys.clear();
        state.values.clear();
        state
            .keys
            .try_reserve_exact(kv_total)
            .map_err(io::Error::other)?;
        state
            .values
            .try_reserve_exact(kv_total)
            .map_err(io::Error::other)?;
        state.keys.resize(kv_total, 0.0);
        state.values.resize(kv_total, 0.0);
    }

    let divisor = scalar(c.head_dim).sqrt();
    let gqa_group = c.heads / c.kv_heads;
    for (block, state) in model.blocks.iter().zip(layers.iter_mut()) {
        check(cancel)?;
        rms_batch(
            &ws.batch_a,
            &mut ws.batch_norm,
            &block.attention_norm,
            c.epsilon,
            hidden,
            workers,
        )?;
        let qkv_prepared = PreparedBatch::new(&ws.batch_norm, total, hidden);
        block.query.multiply_batch_prepared_into(
            &qkv_prepared,
            &ws.batch_norm,
            workers,
            &mut ws.transposed_scratch,
            &mut ws.batch_q,
            cancel,
        )?;
        block.key.multiply_batch_prepared_into(
            &qkv_prepared,
            &ws.batch_norm,
            workers,
            &mut ws.transposed_scratch,
            &mut ws.batch_k,
            cancel,
        )?;
        block.value.multiply_batch_prepared_into(
            &qkv_prepared,
            &ws.batch_norm,
            workers,
            &mut ws.transposed_scratch,
            &mut ws.batch_v,
            cancel,
        )?;

        row_workers::try_run_aligned(&mut ws.batch_q, workers, q_width, &|elem_off, q_slice| {
            let start_pos = elem_off / q_width;
            for (local_pos, q_row) in q_slice.chunks_exact_mut(q_width).enumerate() {
                normalize_heads(q_row, &block.query_norm, c)?;
                rope_with_inv_freq(q_row, &model.inv_freq, c, start_pos + local_pos)?;
            }
            Ok(())
        })?;

        let raw_k = &ws.batch_k;
        row_workers::try_run_aligned(&mut state.keys, workers, kv_width, &|elem_off, k_slice| {
            let start_pos = elem_off / kv_width;
            let end = elem_off + k_slice.len();
            k_slice.copy_from_slice(&raw_k[elem_off..end]);
            for (local_pos, k_row) in k_slice.chunks_exact_mut(kv_width).enumerate() {
                normalize_heads(k_row, &block.key_norm, c)?;
                rope_with_inv_freq(k_row, &model.inv_freq, c, start_pos + local_pos)?;
                for value in k_row {
                    *value = cache_value(*value, c.cache_precision)?;
                }
            }
            Ok(())
        })?;

        let raw_v = &ws.batch_v;
        row_workers::try_run_aligned(
            &mut state.values,
            workers,
            kv_width,
            &|elem_off, v_slice| {
                let end = elem_off + v_slice.len();
                for (dst, &src) in v_slice.iter_mut().zip(&raw_v[elem_off..end]) {
                    *dst = cache_value(src, c.cache_precision)?;
                }
                Ok(())
            },
        )?;

        for _ in 0..(total * c.heads) {
            check(cancel)?;
        }
        let queries = &ws.batch_q;
        let cached_keys = &state.keys;
        let cached_values = &state.values;
        row_workers::try_run_aligned(
            &mut ws.batch_attn,
            workers,
            q_width,
            &|elem_off, attn_slice| {
                let start_pos = elem_off / q_width;
                let rows = attn_slice.len() / q_width;
                let max_pos = start_pos + rows - 1;
                let mut scores = zeros(max_pos + 1)?;
                for (local_pos, out_row) in attn_slice.chunks_exact_mut(q_width).enumerate() {
                    let pos = start_pos + local_pos;
                    let q_row = &queries[pos * q_width..(pos + 1) * q_width];
                    let history_rows = pos + 1;
                    let row_scores = &mut scores[..history_rows];
                    for head in 0..c.heads {
                        let kv_head = head / gqa_group;
                        let offset = kv_head * c.head_dim;
                        let q_head = &q_row[head * c.head_dim..(head + 1) * c.head_dim];
                        let out_head = &mut out_row[head * c.head_dim..(head + 1) * c.head_dim];
                        attend_head_causal(
                            q_head,
                            cached_keys,
                            cached_values,
                            None,
                            None,
                            history_rows,
                            kv_width,
                            offset,
                            c.head_dim,
                            divisor,
                            row_scores,
                            out_head,
                        )?;
                    }
                    finite(out_row)?;
                }
                Ok(())
            },
        )?;

        let attn_prepared = PreparedBatch::new(&ws.batch_attn, total, q_width);
        block.attention_out.multiply_batch_prepared_into(
            &attn_prepared,
            &ws.batch_attn,
            workers,
            &mut ws.transposed_scratch,
            &mut ws.batch_proj,
            cancel,
        )?;
        add_residual_batch(&mut ws.batch_a, &ws.batch_proj, workers, hidden)?;

        rms_batch(
            &ws.batch_a,
            &mut ws.batch_norm,
            &block.ffn_norm,
            c.epsilon,
            hidden,
            workers,
        )?;
        let ffn_prepared = PreparedBatch::new(&ws.batch_norm, total, hidden);
        block.gate.multiply_batch_prepared_into(
            &ffn_prepared,
            &ws.batch_norm,
            workers,
            &mut ws.transposed_scratch,
            &mut ws.batch_gate,
            cancel,
        )?;
        block.up.multiply_batch_prepared_into(
            &ffn_prepared,
            &ws.batch_norm,
            workers,
            &mut ws.transposed_scratch,
            &mut ws.batch_up,
            cancel,
        )?;
        let up_all = &ws.batch_up;
        row_workers::try_run_aligned(
            &mut ws.batch_gate,
            workers,
            c.intermediate,
            &|elem_off, gate_slice| {
                let end = elem_off + gate_slice.len();
                for (gate, &up) in gate_slice.iter_mut().zip(&up_all[elem_off..end]) {
                    *gate = (*gate / (1.0 + (-*gate).exp())) * up;
                }
                finite(gate_slice)
            },
        )?;
        let down_prepared = PreparedBatch::new(&ws.batch_gate, total, c.intermediate);
        block.down.multiply_batch_prepared_into(
            &down_prepared,
            &ws.batch_gate,
            workers,
            &mut ws.transposed_scratch,
            &mut ws.batch_proj,
            cancel,
        )?;
        add_residual_batch(&mut ws.batch_a, &ws.batch_proj, workers, hidden)?;
    }

    rms_batch(
        &ws.batch_a,
        &mut ws.batch_norm,
        &model.output_norm,
        c.epsilon,
        hidden,
        workers,
    )?;
    let last_norm = &ws.batch_norm[(total - 1) * hidden..total * hidden];
    let logits = model
        .embedding
        .multiply_with_threads(last_norm, workers, cancel)?;
    check(cancel)?;
    Ok(logits)
}

struct ResetGuard<'a> {
    session: &'a mut Session,
    success: bool,
}
impl Drop for ResetGuard<'_> {
    fn drop(&mut self) {
        if !self.success {
            self.session.reset();
        }
    }
}

fn norm_weights(weights: &[f32], width: usize) -> io::Result<()> {
    if weights.len() != width {
        return Err(invalid());
    }
    finite(weights)
}

#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Admitted decoder dimensions and positions are at most 65,536 and exactly representable in f32."
)]
const fn scalar(value: usize) -> f32 {
    value as f32
}

fn rms_into(input: &[f32], weights: &[f32], epsilon: f32, output: &mut [f32]) -> io::Result<()> {
    let mut squares = 0.0_f32;
    for value in input {
        squares += *value * *value;
    }
    if !squares.is_finite() || !(squares / scalar(input.len()) + epsilon).is_finite() {
        return Err(invalid());
    }
    let scale = 1.0 / (squares / scalar(input.len()) + epsilon).sqrt();
    for ((dst, value), weight) in output.iter_mut().zip(input).zip(weights) {
        *dst = (*value * scale) * *weight;
    }
    finite(output)
}

fn rms_batch(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    epsilon: f32,
    width: usize,
    workers: usize,
) -> io::Result<()> {
    row_workers::try_run_aligned(output, workers, width, &|elem_off, slice| {
        for (src_row, dst_row) in input[elem_off..elem_off + slice.len()]
            .chunks_exact(width)
            .zip(slice.chunks_exact_mut(width))
        {
            rms_into(src_row, weights, epsilon, dst_row)?;
        }
        Ok(())
    })
}

fn add_residual_batch(
    dst: &mut [f32],
    add: &[f32],
    workers: usize,
    width: usize,
) -> io::Result<()> {
    row_workers::try_run_aligned(dst, workers, width, &|elem_off, dst_slice| {
        let end = elem_off + dst_slice.len();
        for (out, &delta) in dst_slice.iter_mut().zip(&add[elem_off..end]) {
            *out += delta;
        }
        finite(dst_slice)
    })
}

fn normalize_heads(values: &mut [f32], weights: &[f32], c: Config) -> io::Result<()> {
    let mut normalized = [0.0_f32; 256];
    let head_buf = &mut normalized[..c.head_dim];
    for head in values.chunks_exact_mut(c.head_dim) {
        rms_into(head, weights, c.epsilon, head_buf)?;
        head.copy_from_slice(head_buf);
    }
    Ok(())
}

fn rope_with_inv_freq(
    values: &mut [f32],
    inv_freq: &[f32],
    c: Config,
    position: usize,
) -> io::Result<()> {
    let half = c.head_dim / 2;
    let mut trig = [(0.0_f32, 0.0_f32); 128];
    let pos = scalar(position);
    for (slot, &freq) in trig[..half].iter_mut().zip(inv_freq) {
        *slot = (pos * freq).sin_cos();
    }
    for head in values.chunks_exact_mut(c.head_dim) {
        for (index, &(sin, cos)) in trig[..half].iter().enumerate() {
            let (a, b) = (head[index], head[index + half]);
            head[index] = a * cos - b * sin;
            head[index + half] = a * sin + b * cos;
        }
    }
    finite(values)
}

fn forward_into(
    block: &Block,
    state: &LayerState,
    model: &Model,
    position: usize,
    ws: &mut DecoderWorkspace,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<()> {
    let c = model.config;
    let workers = model.decode_threads;
    rms_into(
        &ws.single_hidden,
        &block.attention_norm,
        c.epsilon,
        &mut ws.single_norm,
    )?;
    block
        .query
        .multiply_into(&ws.single_norm, workers, &mut ws.single_q, cancel)?;
    block
        .key
        .multiply_into(&ws.single_norm, workers, &mut ws.single_k, cancel)?;
    block
        .value
        .multiply_into(&ws.single_norm, workers, &mut ws.single_v, cancel)?;
    normalize_heads(&mut ws.single_q, &block.query_norm, c)?;
    normalize_heads(&mut ws.single_k, &block.key_norm, c)?;
    rope_with_inv_freq(&mut ws.single_q, &model.inv_freq, c, position)?;
    rope_with_inv_freq(&mut ws.single_k, &model.inv_freq, c, position)?;
    for value in &mut ws.single_k {
        *value = cache_value(*value, c.cache_precision)?;
    }
    for value in &mut ws.single_v {
        *value = cache_value(*value, c.cache_precision)?;
    }
    attend_into(
        &ws.single_q,
        &ws.single_k,
        &ws.single_v,
        state,
        c,
        position,
        &mut ws.single_scores,
        &mut ws.single_attn,
        cancel,
    )?;
    block
        .attention_out
        .multiply_into(&ws.single_attn, workers, &mut ws.single_proj, cancel)?;
    for (dst, &add) in ws.single_hidden.iter_mut().zip(&ws.single_proj) {
        *dst += add;
    }
    finite(&ws.single_hidden)?;
    rms_into(
        &ws.single_hidden,
        &block.ffn_norm,
        c.epsilon,
        &mut ws.single_norm,
    )?;
    block
        .gate
        .multiply_into(&ws.single_norm, workers, &mut ws.single_gate, cancel)?;
    block
        .up
        .multiply_into(&ws.single_norm, workers, &mut ws.single_up, cancel)?;
    for (gate, &up) in ws.single_gate.iter_mut().zip(&ws.single_up) {
        *gate = (*gate / (1.0 + (-*gate).exp())) * up;
    }
    finite(&ws.single_gate)?;
    block
        .down
        .multiply_into(&ws.single_gate, workers, &mut ws.single_proj, cancel)?;
    for (out, &value) in ws.single_hidden.iter_mut().zip(&ws.single_proj) {
        *out += value;
    }
    finite(&ws.single_hidden)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Causal attention kernel shares state/current slices and reusable score buffer without allocations."
)]
fn attend_head_causal(
    q_head: &[f32],
    state_keys: &[f32],
    state_values: &[f32],
    cur_key: Option<&[f32]>,
    cur_val: Option<&[f32]>,
    history_rows: usize,
    kv_width: usize,
    offset: usize,
    head_dim: usize,
    divisor: f32,
    scores: &mut [f32],
    out_head: &mut [f32],
) -> io::Result<()> {
    let full_quads = history_rows / 4;
    for quad in 0..full_quads {
        let r = quad * 4;
        let k0 = &state_keys[r * kv_width + offset..r * kv_width + offset + head_dim];
        let k1 = &state_keys[(r + 1) * kv_width + offset..(r + 1) * kv_width + offset + head_dim];
        let k2 = &state_keys[(r + 2) * kv_width + offset..(r + 2) * kv_width + offset + head_dim];
        let k3 = &state_keys[(r + 3) * kv_width + offset..(r + 3) * kv_width + offset + head_dim];
        scores[r..r + 4].copy_from_slice(&packed::dot_4keys_seq(q_head, [k0, k1, k2, k3], divisor));
    }
    for (row, score) in scores
        .iter_mut()
        .enumerate()
        .take(history_rows)
        .skip(full_quads * 4)
    {
        let k = &state_keys[row * kv_width + offset..row * kv_width + offset + head_dim];
        let mut dot = 0.0_f32;
        for (q_elem, k_elem) in q_head.iter().zip(k) {
            dot += *q_elem * *k_elem;
        }
        *score = dot / divisor;
    }
    if let Some(k) = cur_key {
        let mut dot = 0.0_f32;
        for (q_elem, k_elem) in q_head.iter().zip(k) {
            dot += *q_elem * *k_elem;
        }
        scores[history_rows] = dot / divisor;
    }
    finite(scores)?;
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for score in &mut *scores {
        *score = (*score - maximum).exp();
        sum += *score;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(invalid());
    }
    out_head.fill(0.0);
    for quad in 0..full_quads {
        let r = quad * 4;
        let v0 = &state_values[r * kv_width + offset..r * kv_width + offset + head_dim];
        let v1 = &state_values[(r + 1) * kv_width + offset..(r + 1) * kv_width + offset + head_dim];
        let v2 = &state_values[(r + 2) * kv_width + offset..(r + 2) * kv_width + offset + head_dim];
        let v3 = &state_values[(r + 3) * kv_width + offset..(r + 3) * kv_width + offset + head_dim];
        let weights = [
            scores[r] / sum,
            scores[r + 1] / sum,
            scores[r + 2] / sum,
            scores[r + 3] / sum,
        ];
        packed::weighted_value_accum_4(out_head, [v0, v1, v2, v3], weights);
    }
    for (row, &score) in scores
        .iter()
        .enumerate()
        .take(history_rows)
        .skip(full_quads * 4)
    {
        let v = &state_values[row * kv_width + offset..row * kv_width + offset + head_dim];
        packed::weighted_value_accum(out_head, v, score / sum);
    }
    if let Some(v) = cur_val {
        packed::weighted_value_accum(out_head, v, scores[history_rows] / sum);
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Maintains direct single-position attend contract alongside workspace-reusing attend_into."
)]
fn attend_into(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    state: &LayerState,
    c: Config,
    position: usize,
    scores: &mut [f32],
    output: &mut [f32],
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<()> {
    let width = c.kv_heads * c.head_dim;
    let divisor = scalar(c.head_dim).sqrt();
    let gqa_group = c.heads / c.kv_heads;
    let row_scores = &mut scores[..=position];
    for head in 0..c.heads {
        check(cancel)?;
        let kv_head = head / gqa_group;
        let offset = kv_head * c.head_dim;
        let q_head = &query[head * c.head_dim..(head + 1) * c.head_dim];
        let out_head = &mut output[head * c.head_dim..(head + 1) * c.head_dim];
        attend_head_causal(
            q_head,
            &state.keys,
            &state.values,
            Some(&keys[offset..offset + c.head_dim]),
            Some(&values[offset..offset + c.head_dim]),
            position,
            width,
            offset,
            c.head_dim,
            divisor,
            row_scores,
            out_head,
        )?;
    }
    finite(output)
}

#[cfg(test)]
fn attend(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    state: &LayerState,
    c: Config,
    position: usize,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Vec<f32>> {
    let mut output = zeros(c.heads * c.head_dim)?;
    let mut scores = zeros(position + 1)?;
    attend_into(
        query,
        keys,
        values,
        state,
        c,
        position,
        &mut scores,
        &mut output,
        cancel,
    )?;
    Ok(output)
}

#[cfg(test)]
#[path = "decoder/tests.rs"]
mod tests;
