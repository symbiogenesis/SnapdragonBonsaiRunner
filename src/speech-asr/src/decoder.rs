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
    replay::{CacheValues, LayerCache},
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
    packed::ensure_finite(values, invalid)
}

/// Borrowed owned-or-mmap byte backing for quantized projections.
pub(super) enum WeightBytes {
    Owned(Vec<u8>),
    #[cfg(all(windows, feature = "rust-qualification"))]
    Mapped {
        _mapping: Arc<bonsai_windows_process::MappedFile>,
        offset: usize,
        len: usize,
    },
}

impl WeightBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes.as_slice(),
            #[cfg(all(windows, feature = "rust-qualification"))]
            Self::Mapped {
                _mapping,
                offset,
                len,
            } => &_mapping.as_slice()[*offset..*offset + *len],
        }
    }
}

/// GGUF row-major F32, `Q8_0`, `Q4_K`, or `Q6_K` projection, retaining compressed weights.
pub struct Projection {
    input: usize,
    output: usize,
    data: Storage,
}

enum Storage {
    Float(Vec<f32>),
    Q8(WeightBytes),
    Q4K(WeightBytes),
    Q6K(WeightBytes),
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
        let (chunks8, tail) = bytes.as_chunks::<16>();
        let mut offset = 0_usize;
        for raw_chunk in chunks8 {
            let pairs = raw_chunk.as_chunks::<2>().0;
            let bits: [u16; 8] = std::array::from_fn(|idx| u16::from_le_bytes(pairs[idx]));
            let [lo, hi] = bonsai_neon_dot::widen_f16_eight(&bits);
            for (slot, value) in values[offset..offset + 8]
                .iter_mut()
                .zip(lo.into_iter().chain(hi))
            {
                if !value.is_finite() {
                    return Err(invalid());
                }
                *slot = value;
            }
            offset += 8;
        }
        for (value, pair) in values[offset..].iter_mut().zip(tail.as_chunks::<2>().0) {
            *value = half(u16::from_le_bytes(*pair))?;
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

    fn admit_quant<const BLOCK_BYTES: usize>(
        input: usize,
        output: usize,
        data: WeightBytes,
        block_elems: usize,
        check: fn(&[u8; BLOCK_BYTES]) -> bool,
        wrap: fn(WeightBytes) -> Storage,
    ) -> io::Result<Self> {
        let count = extent(input, output)?;
        let bytes = data.as_slice();
        if !input.is_multiple_of(block_elems) || bytes.len() != count / block_elems * BLOCK_BYTES {
            return Err(invalid());
        }
        validate_quant_blocks::<BLOCK_BYTES>(bytes, check)?;
        Ok(Self {
            input,
            output,
            data: wrap(data),
        })
    }

    /// Admit `Q8_0` blocks: little-endian half scale followed by 32 signed bytes.
    ///
    /// # Errors
    /// Rejects incomplete blocks, wrong byte length or nonfinite half scales.
    pub fn q8(input: usize, output: usize, data: Vec<u8>) -> io::Result<Self> {
        Self::q8_bytes(input, output, WeightBytes::Owned(data))
    }

    pub(super) fn q8_bytes(input: usize, output: usize, data: WeightBytes) -> io::Result<Self> {
        Self::admit_quant::<34>(
            input,
            output,
            data,
            32,
            |block| f16_bits_finite(block[0], block[1]),
            Storage::Q8,
        )
    }

    /// Admit `Q4_K` super-blocks (`144` bytes per `256` elements).
    ///
    /// # Errors
    /// Rejects misaligned dimensions, incomplete blocks, wrong byte length or nonfinite half scales.
    pub fn q4_k(input: usize, output: usize, data: Vec<u8>) -> io::Result<Self> {
        Self::q4_k_bytes(input, output, WeightBytes::Owned(data))
    }

    pub(super) fn q4_k_bytes(input: usize, output: usize, data: WeightBytes) -> io::Result<Self> {
        Self::admit_quant::<144>(
            input,
            output,
            data,
            256,
            |block| f16_bits_finite(block[0], block[1]) && f16_bits_finite(block[2], block[3]),
            Storage::Q4K,
        )
    }

    /// Admit `Q6_K` super-blocks (`210` bytes per `256` elements).
    ///
    /// # Errors
    /// Rejects misaligned dimensions, incomplete blocks, wrong byte length or nonfinite half scales.
    pub fn q6_k(input: usize, output: usize, data: Vec<u8>) -> io::Result<Self> {
        Self::q6_k_bytes(input, output, WeightBytes::Owned(data))
    }

    pub(super) fn q6_k_bytes(input: usize, output: usize, data: WeightBytes) -> io::Result<Self> {
        Self::admit_quant::<210>(
            input,
            output,
            data,
            256,
            |block| f16_bits_finite(block[208], block[209]),
            Storage::Q6K,
        )
    }

    fn row_into(&self, row: usize, output: &mut [f32]) -> io::Result<()> {
        if row >= self.output || output.len() != self.input {
            return Err(invalid());
        }
        match &self.data {
            Storage::Float(values) => {
                output.copy_from_slice(&values[row * self.input..(row + 1) * self.input]);
            }
            Storage::Q8(raw) => {
                let bytes = raw.as_slice();
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
            Storage::Q4K(raw) => {
                let bytes = raw.as_slice();
                let row_bytes = self.input / 256 * 144;
                packed::decode_q4_k_row(&bytes[row * row_bytes..(row + 1) * row_bytes], output);
            }
            Storage::Q6K(raw) => {
                let bytes = raw.as_slice();
                let row_bytes = self.input / 256 * 210;
                packed::decode_q6_k_row(&bytes[row * row_bytes..(row + 1) * row_bytes], output);
            }
        }
        finite(output)
    }

    pub(super) fn as_weight_matrix(&self) -> WeightMatrix<'_> {
        let data = match &self.data {
            Storage::Float(values) => WeightSlice::Float(values),
            Storage::Q8(bytes) => WeightSlice::Q8(bytes.as_slice()),
            Storage::Q4K(bytes) => WeightSlice::Q4K(bytes.as_slice()),
            Storage::Q6K(bytes) => WeightSlice::Q6K(bytes.as_slice()),
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

    pub(super) fn multiply_group_into<const N: usize>(
        pairs: [(&Self, &mut [f32]); N],
        input: &[f32],
        workers: usize,
        scratch: &mut Vec<f32>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        let mut checks = 0_usize;
        for (p, out) in &pairs {
            if input.len() != p.input || out.len() != p.output {
                return Err(invalid());
            }
            checks += p.output;
        }
        finite(input)?;
        for _ in 0..checks {
            check(cancel)?;
        }
        let mut targets = pairs.map(|(p, out)| (p.as_weight_matrix(), out));
        packed::multiply_single_group_into(&mut targets, input, workers, scratch);
        for (_, out) in &targets {
            finite(out)?;
        }
        Ok(())
    }

    pub(super) fn multiply_batch_group_into<const N: usize>(
        pairs: [(&Self, &mut [f32]); N],
        prepared: &PreparedBatch,
        raw_input: &[f32],
        workers: usize,
        scratch: &mut Vec<f32>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        let batch = prepared.batch();
        if raw_input.len() != batch.checked_mul(prepared.input()).ok_or_else(invalid)? {
            return Err(invalid());
        }
        let mut checks = 0_usize;
        for (p, out) in &pairs {
            if prepared.input() != p.input
                || out.len() != batch.checked_mul(p.output).ok_or_else(invalid)?
            {
                return Err(invalid());
            }
            checks += batch + p.output;
        }
        finite(raw_input)?;
        for _ in 0..checks {
            check(cancel)?;
        }
        let mut targets = pairs.map(|(p, out)| (p.as_weight_matrix(), out));
        packed::multiply_batch_group_into(&mut targets, prepared, raw_input, workers, scratch);
        for (_, out) in &targets {
            finite(out)?;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Fuses three sibling batched projections (`q+k+v`) sharing `prepared` and quantized activations."
    )]
    pub(super) fn multiply_batch_triple_into(
        p0: &Self,
        p1: &Self,
        p2: &Self,
        prepared: &PreparedBatch,
        raw_input: &[f32],
        workers: usize,
        scratch: &mut Vec<f32>,
        out0: &mut [f32],
        out1: &mut [f32],
        out2: &mut [f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        Self::multiply_batch_group_into(
            [(p0, out0), (p1, out1), (p2, out2)],
            prepared,
            raw_input,
            workers,
            scratch,
            cancel,
        )
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

#[inline]
const fn f16_bits_finite(lo: u8, hi: u8) -> bool {
    (u16::from_le_bytes([lo, hi]) & 0x7C00) != 0x7C00
}

fn validate_quant_blocks<const BLOCK_BYTES: usize>(
    bytes: &[u8],
    check: fn(&[u8; BLOCK_BYTES]) -> bool,
) -> io::Result<()> {
    const PARALLEL_THRESHOLD_BYTES: usize = 1024 * 1024;
    if bytes.len() < PARALLEL_THRESHOLD_BYTES {
        let valid = bytes.as_chunks::<BLOCK_BYTES>().0.iter().all(check);
        return if valid { Ok(()) } else { Err(invalid()) };
    }
    use rayon::prelude::*;
    let chunk_bytes = 16_384 * BLOCK_BYTES;
    let valid = bytes
        .par_chunks(chunk_bytes)
        .all(|chunk| chunk.as_chunks::<BLOCK_BYTES>().0.iter().all(check));
    if valid { Ok(()) } else { Err(invalid()) }
}

fn half(bits: u16) -> io::Result<f32> {
    let value = bonsai_ternary::half_to_float(bits);
    if !value.is_finite() {
        return Err(invalid());
    }
    Ok(value)
}

/// Cache scalar policy, selected explicitly when creating a decoder model.
#[derive(Clone, Copy, Debug)]
pub enum CachePrecision {
    /// Preserve F32 K/V bits.
    F32,
    /// Round K/V to IEEE binary16 with nearest-even ties before attention.
    F16,
}

#[cfg(test)]
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
    let bits = packed::checked_float_to_half(value).ok_or_else(invalid)?;
    Ok(bonsai_ternary::half_to_float(bits))
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
    /// reads decoder weights only, accepting `F32/F16/Q8_0/Q4_K/Q6_K` projections and F32
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
        let precision = self.config.cache_precision;
        Session {
            model: Arc::clone(self),
            position: 0,
            layers: (0..self.blocks.len())
                .map(|_| LayerState::new(precision))
                .collect(),
            workspace: DecoderWorkspace::default(),
        }
    }
}

struct KvPair<T> {
    keys: Vec<T>,
    values: Vec<T>,
}

impl<T: Default + Clone> KvPair<T> {
    const fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    fn clear_bounded(&mut self, retain_cap: usize) {
        clear_bounded(&mut self.keys, retain_cap);
        clear_bounded(&mut self.values, retain_cap);
    }

    fn reserve_extra(&mut self, extra_kv: usize) -> io::Result<()> {
        for buf in [&mut self.keys, &mut self.values] {
            if buf.capacity() < buf.len().saturating_add(extra_kv) {
                buf.try_reserve(extra_kv).map_err(io::Error::other)?;
            }
        }
        Ok(())
    }

    fn prepare_append(&mut self, start_kv: usize, total: usize) -> io::Result<()> {
        for buf in [&mut self.keys, &mut self.values] {
            buf.truncate(start_kv);
            if buf.capacity() < total {
                buf.try_reserve_exact(total - buf.len())
                    .map_err(io::Error::other)?;
            }
            buf.resize(total, T::default());
        }
        Ok(())
    }

    fn truncate_to(&mut self, keep: usize) {
        self.keys.truncate(keep);
        self.values.truncate(keep);
    }

    fn extend(&mut self, src_k: &[T], src_v: &[T])
    where
        T: Copy,
    {
        self.keys.extend_from_slice(src_k);
        self.values.extend_from_slice(src_v);
    }
}

enum LayerKv {
    F32(KvPair<f32>),
    F16(KvPair<u16>),
}

struct LayerState {
    data: LayerKv,
}

impl LayerState {
    const fn new(precision: CachePrecision) -> Self {
        Self {
            data: match precision {
                CachePrecision::F32 => LayerKv::F32(KvPair::new()),
                CachePrecision::F16 => LayerKv::F16(KvPair::new()),
            },
        }
    }

    #[cfg(test)]
    fn from_f32(keys: Vec<f32>, values: Vec<f32>) -> Self {
        Self {
            data: LayerKv::F32(KvPair { keys, values }),
        }
    }

    fn clear_bounded(&mut self, retain_cap: usize) {
        match &mut self.data {
            LayerKv::F32(pair) => pair.clear_bounded(retain_cap),
            LayerKv::F16(pair) => pair.clear_bounded(retain_cap),
        }
    }

    fn reserve_extra(&mut self, extra_kv: usize) -> io::Result<()> {
        match &mut self.data {
            LayerKv::F32(pair) => pair.reserve_extra(extra_kv),
            LayerKv::F16(pair) => pair.reserve_extra(extra_kv),
        }
    }

    fn prepare_prefill(&mut self, kv_total: usize) -> io::Result<()> {
        self.prepare_append(0, kv_total)
    }

    fn prepare_append(&mut self, start_kv: usize, add_kv: usize) -> io::Result<()> {
        let total = start_kv.checked_add(add_kv).ok_or_else(invalid)?;
        match &mut self.data {
            LayerKv::F32(pair) => pair.prepare_append(start_kv, total),
            LayerKv::F16(pair) => pair.prepare_append(start_kv, total),
        }
    }

    fn write_append(&mut self, start_kv: usize, src_k: &[f32], src_v: &[f32]) -> io::Result<()> {
        match &mut self.data {
            LayerKv::F32(pair) => {
                finite(src_k)?;
                finite(src_v)?;
                pair.keys[start_kv..start_kv + src_k.len()].copy_from_slice(src_k);
                pair.values[start_kv..start_kv + src_v.len()].copy_from_slice(src_v);
            }
            LayerKv::F16(pair) => {
                if !packed::narrow_f32_to_f16_slice(
                    src_k,
                    &mut pair.keys[start_kv..start_kv + src_k.len()],
                ) || !packed::narrow_f32_to_f16_slice(
                    src_v,
                    &mut pair.values[start_kv..start_kv + src_v.len()],
                ) {
                    return Err(invalid());
                }
            }
        }
        Ok(())
    }

    fn truncate_to(&mut self, keep: usize) {
        match &mut self.data {
            LayerKv::F32(pair) => pair.truncate_to(keep),
            LayerKv::F16(pair) => pair.truncate_to(keep),
        }
    }

    fn as_layer_cache(&self) -> LayerCache<'_> {
        match &self.data {
            LayerKv::F32(pair) => LayerCache {
                keys: CacheValues::F32(pair.keys.as_slice()),
                values: CacheValues::F32(pair.values.as_slice()),
            },
            LayerKv::F16(pair) => LayerCache {
                keys: CacheValues::F16(pair.keys.as_slice()),
                values: CacheValues::F16(pair.values.as_slice()),
            },
        }
    }

    #[cfg(test)]
    fn key_capacity(&self) -> usize {
        match &self.data {
            LayerKv::F32(pair) => pair.keys.capacity(),
            LayerKv::F16(pair) => pair.keys.capacity(),
        }
    }

    #[cfg(test)]
    fn value_capacity(&self) -> usize {
        match &self.data {
            LayerKv::F32(pair) => pair.values.capacity(),
            LayerKv::F16(pair) => pair.values.capacity(),
        }
    }
}

#[derive(Default)]
struct DecoderWorkspace {
    single_hidden: Vec<f32>,
    single_norm: Vec<f32>,
    single_q: Vec<f32>,
    single_k: Vec<f32>,
    single_v: Vec<f32>,
    single_k_f16: Vec<u16>,
    single_v_f16: Vec<u16>,
    single_attn: Vec<f32>,
    single_proj: Vec<f32>,
    single_gate: Vec<f32>,
    single_up: Vec<f32>,
    single_scores: Vec<f32>,
    single_provisional_keys: Vec<f32>,
    single_provisional_values: Vec<f32>,
    single_provisional_keys_f16: Vec<u16>,
    single_provisional_values_f16: Vec<u16>,
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
    prepared: PreparedBatch,
}

const RETAIN_POSITIONS: usize = 128;

fn clear_bounded<T>(values: &mut Vec<T>, retain_cap: usize) {
    values.clear();
    if values.capacity() > retain_cap.saturating_mul(2) {
        values.shrink_to(retain_cap);
    }
}

impl DecoderWorkspace {
    fn clear_bounded(&mut self, c: Config) {
        let retain_pos = RETAIN_POSITIONS.min(c.context.max(1));
        let q_width = c.heads.saturating_mul(c.head_dim);
        let kv_width = c.kv_heads.saturating_mul(c.head_dim);
        let max_proj_out = c.intermediate.max(q_width).max(c.hidden);
        let max_proj_in = c.intermediate.max(q_width).max(c.hidden);
        self.single_hidden.clear();
        self.single_norm.clear();
        self.single_q.clear();
        self.single_k.clear();
        self.single_v.clear();
        self.single_k_f16.clear();
        self.single_v_f16.clear();
        self.single_attn.clear();
        self.single_proj.clear();
        self.single_gate.clear();
        self.single_up.clear();
        clear_bounded(&mut self.single_scores, retain_pos);
        self.single_provisional_keys.clear();
        self.single_provisional_values.clear();
        self.single_provisional_keys_f16.clear();
        self.single_provisional_values_f16.clear();
        clear_bounded(&mut self.batch_a, retain_pos.saturating_mul(c.hidden));
        clear_bounded(&mut self.batch_norm, retain_pos.saturating_mul(c.hidden));
        clear_bounded(&mut self.batch_q, retain_pos.saturating_mul(q_width));
        clear_bounded(&mut self.batch_k, retain_pos.saturating_mul(kv_width));
        clear_bounded(&mut self.batch_v, retain_pos.saturating_mul(kv_width));
        clear_bounded(&mut self.batch_attn, retain_pos.saturating_mul(q_width));
        clear_bounded(&mut self.batch_proj, retain_pos.saturating_mul(c.hidden));
        clear_bounded(
            &mut self.batch_gate,
            retain_pos.saturating_mul(c.intermediate),
        );
        clear_bounded(
            &mut self.batch_up,
            retain_pos.saturating_mul(c.intermediate),
        );
        clear_bounded(
            &mut self.transposed_scratch,
            retain_pos.saturating_mul(max_proj_out),
        );
        self.prepared
            .clear_bounded((retain_pos / 4).max(1).saturating_mul(max_proj_in));
    }
}

fn ensure_len_t<T: Default + Clone>(buffer: &mut Vec<T>, len: usize) -> io::Result<()> {
    if buffer.capacity() < len {
        buffer
            .try_reserve(len - buffer.len())
            .map_err(io::Error::other)?;
    }
    buffer.resize(len, T::default());
    Ok(())
}

fn ensure_len(buffer: &mut Vec<f32>, len: usize) -> io::Result<()> {
    ensure_len_t(buffer, len)
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

    /// Borrow complete layer K/V in `[position, kv_head, head_dim]` order (`F16` or `F32`).
    #[must_use]
    pub fn cache(&self, layer: usize) -> Option<LayerCache<'_>> {
        self.layers.get(layer).map(LayerState::as_layer_cache)
    }

    /// Discard every layer's K/V and the position, retaining bounded reusable
    /// capacity while shrinking oversized buffers from high-water bursts.
    pub fn reset(&mut self) {
        let c = self.model.config;
        let kv_retain = RETAIN_POSITIONS
            .min(c.context.max(1))
            .saturating_mul(c.kv_heads)
            .saturating_mul(c.head_dim);
        for layer in &mut self.layers {
            layer.clear_bounded(kv_retain);
        }
        self.workspace.clear_bounded(c);
        self.position = 0;
    }

    /// Pre-reserve capacity for `additional` decode tokens without exact per-step reallocations.
    pub(crate) fn reserve_decode(&mut self, additional: usize) -> io::Result<()> {
        let c = self.model.config;
        let kv_width = c.kv_heads.checked_mul(c.head_dim).ok_or_else(invalid)?;
        let extra_kv = additional.checked_mul(kv_width).ok_or_else(invalid)?;
        for state in &mut self.layers {
            state.reserve_extra(extra_kv)?;
        }
        let scores_len = self.position.saturating_add(additional).max(1);
        if self.workspace.single_scores.capacity() < scores_len {
            let needed = scores_len - self.workspace.single_scores.len();
            self.workspace
                .single_scores
                .try_reserve(needed)
                .map_err(io::Error::other)?;
        }
        Ok(())
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
        let c = self.model.config;
        if self.position >= c.context {
            return Err(invalid());
        }
        ensure_len(&mut self.workspace.single_hidden, c.hidden)?;
        let token_id = usize::try_from(id).map_err(|_| invalid())?;
        self.model
            .embedding
            .row_into(token_id, &mut self.workspace.single_hidden)?;
        self.advance_prepared(project_logits, cancel)
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
        let c = self.model.config;
        if self.position >= c.context || row.len() != c.hidden {
            return Err(invalid());
        }
        finite(row)?;
        ensure_len(&mut self.workspace.single_hidden, c.hidden)?;
        self.workspace.single_hidden.copy_from_slice(row);
        self.advance_prepared(project_logits, cancel)
    }

    fn advance_prepared(
        &mut self,
        project_logits: bool,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Vec<f32>> {
        let model = &self.model;
        let c = model.config;
        check(cancel)?;
        let kv_width = c.kv_heads * c.head_dim;
        let q_width = c.heads * c.head_dim;
        let ws = &mut self.workspace;
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
        if matches!(c.cache_precision, CachePrecision::F16) {
            ensure_len_t(&mut ws.single_k_f16, kv_width)?;
            ensure_len_t(&mut ws.single_v_f16, kv_width)?;
            ensure_len_t(&mut ws.single_provisional_keys_f16, prov_total)?;
            ensure_len_t(&mut ws.single_provisional_values_f16, prov_total)?;
        } else {
            ensure_len(&mut ws.single_provisional_keys, prov_total)?;
            ensure_len(&mut ws.single_provisional_values, prov_total)?;
        }

        for (layer_idx, (block, state)) in model.blocks.iter().zip(&self.layers).enumerate() {
            check(cancel)?;
            forward_into(block, state, model, self.position, ws, cancel)?;
            if matches!(c.cache_precision, CachePrecision::F16) {
                ws.single_provisional_keys_f16[layer_idx * kv_width..(layer_idx + 1) * kv_width]
                    .copy_from_slice(&ws.single_k_f16);
                ws.single_provisional_values_f16[layer_idx * kv_width..(layer_idx + 1) * kv_width]
                    .copy_from_slice(&ws.single_v_f16);
            } else {
                ws.single_provisional_keys[layer_idx * kv_width..(layer_idx + 1) * kv_width]
                    .copy_from_slice(&ws.single_k);
                ws.single_provisional_values[layer_idx * kv_width..(layer_idx + 1) * kv_width]
                    .copy_from_slice(&ws.single_v);
            }
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
            state.reserve_extra(kv_width)?;
        }
        for (layer_idx, state) in self.layers.iter_mut().enumerate() {
            let range = layer_idx * kv_width..(layer_idx + 1) * kv_width;
            match &mut state.data {
                LayerKv::F32(pair) => pair.extend(
                    &ws.single_provisional_keys[range.clone()],
                    &ws.single_provisional_values[range],
                ),
                LayerKv::F16(pair) => pair.extend(
                    &ws.single_provisional_keys_f16[range.clone()],
                    &ws.single_provisional_values_f16[range],
                ),
            }
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
        let logits = prefill_rows_inner(
            &model,
            &mut guard.session.layers,
            &mut guard.session.workspace,
            0,
            PrefillSource::Prompt { prompt, audio },
            true,
            cancel,
        )?
        .ok_or_else(invalid)?;
        guard.session.position = total;
        guard.success = true;
        Ok(logits)
    }

    /// Incrementally append prompt or control tokens at the current position,
    /// skipping the final vocabulary projection when `need_logits` is `false`.
    ///
    /// Failures, cancellations, or unwinds roll every layer back to the previous position.
    ///
    /// # Errors
    /// Rejects empty token slices when `need_logits` is `true`, context overflow, or invalid IDs.
    pub fn prefill_tokens(
        &mut self,
        ids: &[u32],
        need_logits: bool,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Option<Vec<f32>>> {
        check(cancel)?;
        let saved_pos = self.position;
        let model = Arc::clone(&self.model);
        let c = model.config;
        let kv_width = c.kv_heads * c.head_dim;
        let end_pos = saved_pos.checked_add(ids.len()).ok_or_else(invalid)?;
        if end_pos > c.context || (ids.is_empty() && need_logits) {
            return Err(invalid());
        }
        if ids.is_empty() {
            return Ok(None);
        }
        let mut guard = TruncateGuard {
            session: self,
            saved_pos,
            kv_width,
            success: false,
        };
        let logits = prefill_rows_inner(
            &model,
            &mut guard.session.layers,
            &mut guard.session.workspace,
            saved_pos,
            PrefillSource::Tokens(ids),
            need_logits,
            cancel,
        )?;
        guard.session.position = end_pos;
        guard.success = true;
        Ok(logits)
    }

    /// Incrementally append encoded audio feature rows during live microphone capture.
    ///
    /// Failures, cancellations, or unwinds roll every layer back to the previous position.
    ///
    /// # Errors
    /// Rejects mismatched audio dimensions, non-finite inputs, context overflow, or cancellation.
    pub fn prefill_audio_chunk(
        &mut self,
        audio: &[f32],
        rows: usize,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        check(cancel)?;
        let saved_pos = self.position;
        let model = Arc::clone(&self.model);
        let c = model.config;
        let hidden = c.hidden;
        let kv_width = c.kv_heads * c.head_dim;
        if audio.len() != rows.checked_mul(hidden).ok_or_else(invalid)? {
            return Err(invalid());
        }
        let end_pos = saved_pos.checked_add(rows).ok_or_else(invalid)?;
        if end_pos > c.context {
            return Err(invalid());
        }
        finite(audio)?;
        if rows == 0 {
            return Ok(());
        }
        let mut guard = TruncateGuard {
            session: self,
            saved_pos,
            kv_width,
            success: false,
        };
        let _ = prefill_rows_inner(
            &model,
            &mut guard.session.layers,
            &mut guard.session.workspace,
            saved_pos,
            PrefillSource::Audio { audio, rows },
            false,
            cancel,
        )?;
        guard.session.position = end_pos;
        guard.success = true;
        Ok(())
    }
}

enum PrefillSource<'a> {
    Prompt {
        prompt: &'a Prompt,
        audio: &'a [f32],
    },
    Tokens(&'a [u32]),
    Audio {
        audio: &'a [f32],
        rows: usize,
    },
}

impl PrefillSource<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Prompt { prompt, .. } => prompt.ids().len(),
            Self::Tokens(ids) => ids.len(),
            Self::Audio { rows, .. } => *rows,
        }
    }
}

fn prefill_rows_inner(
    model: &Model,
    layers: &mut [LayerState],
    ws: &mut DecoderWorkspace,
    start_pos: usize,
    source: PrefillSource<'_>,
    need_logits: bool,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Option<Vec<f32>>> {
    let c = model.config;
    let total = source.len();
    let hidden = c.hidden;
    let kv_width = c.kv_heads * c.head_dim;
    let q_width = c.heads * c.head_dim;
    let workers = model.batch_threads;

    ensure_len(&mut ws.batch_a, total * hidden)?;
    ensure_len(&mut ws.batch_norm, total * hidden)?;
    ensure_len(&mut ws.batch_q, total * q_width)?;
    ensure_len(&mut ws.batch_k, total * kv_width)?;
    ensure_len(&mut ws.batch_v, total * kv_width)?;
    ensure_len(&mut ws.batch_attn, total * q_width)?;
    ensure_len(&mut ws.batch_proj, total * hidden)?;
    ensure_len(&mut ws.batch_gate, total * c.intermediate)?;
    ensure_len(&mut ws.batch_up, total * c.intermediate)?;

    match source {
        PrefillSource::Prompt { prompt, audio } => {
            let range = prompt.audio_rows();
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
        }
        PrefillSource::Tokens(ids) => {
            for (pos, &id) in ids.iter().enumerate() {
                check(cancel)?;
                let dst = &mut ws.batch_a[pos * hidden..(pos + 1) * hidden];
                let token_id = usize::try_from(id).map_err(|_| invalid())?;
                model.embedding.row_into(token_id, dst)?;
            }
        }
        PrefillSource::Audio { audio, .. } => {
            check(cancel)?;
            ws.batch_a[..total * hidden].copy_from_slice(audio);
        }
    }
    finite(&ws.batch_a)?;

    let start_kv = start_pos * kv_width;
    let add_kv = total * kv_width;
    for state in &mut *layers {
        if start_pos == 0 {
            state.prepare_prefill(add_kv)?;
        } else {
            state.prepare_append(start_kv, add_kv)?;
        }
    }

    let divisor = scalar(c.head_dim).sqrt();
    let gqa_group = c.heads / c.kv_heads;
    let last_layer = model.blocks.len() - 1;
    let last_offset = (total - 1) * hidden;
    for (layer_idx, (block, state)) in model.blocks.iter().zip(layers.iter_mut()).enumerate() {
        check(cancel)?;
        rms_batch(
            &ws.batch_a,
            &mut ws.batch_norm,
            &block.attention_norm,
            c.epsilon,
            hidden,
            workers,
        )?;
        ws.prepared.prepare_into(&ws.batch_norm, total, hidden);
        Projection::multiply_batch_group_into(
            [
                (&block.query, &mut ws.batch_q),
                (&block.key, &mut ws.batch_k),
                (&block.value, &mut ws.batch_v),
            ],
            &ws.prepared,
            &ws.batch_norm,
            workers,
            &mut ws.transposed_scratch,
            cancel,
        )?;

        normalize_and_rope_batch(
            &mut ws.batch_q,
            workers,
            q_width,
            &block.query_norm,
            &model.inv_freq,
            c,
            start_pos,
        )?;
        normalize_and_rope_batch(
            &mut ws.batch_k,
            workers,
            kv_width,
            &block.key_norm,
            &model.inv_freq,
            c,
            start_pos,
        )?;

        state.write_append(start_kv, &ws.batch_k, &ws.batch_v)?;

        for _ in 0..(total * c.heads) {
            check(cancel)?;
        }
        let queries = &ws.batch_q;
        row_workers::try_run_aligned(
            &mut ws.batch_attn,
            workers,
            q_width,
            &|elem_off, attn_slice| {
                let local_start = elem_off / q_width;
                let rows = attn_slice.len() / q_width;
                let max_history = start_pos + local_start + rows;
                let mut scores = zeros(max_history)?;
                for (local_pos, out_row) in attn_slice.chunks_exact_mut(q_width).enumerate() {
                    let local_row = local_start + local_pos;
                    let pos = start_pos + local_row;
                    let q_row = &queries[local_row * q_width..(local_row + 1) * q_width];
                    let history_rows = pos + 1;
                    let row_scores = &mut scores[..history_rows];
                    for head in 0..c.heads {
                        let kv_head = head / gqa_group;
                        let offset = kv_head * c.head_dim;
                        let q_head = &q_row[head * c.head_dim..(head + 1) * c.head_dim];
                        let out_head = &mut out_row[head * c.head_dim..(head + 1) * c.head_dim];
                        attend_head_causal(
                            q_head,
                            state,
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

        ws.prepared.prepare_into(&ws.batch_attn, total, q_width);
        block.attention_out.multiply_batch_prepared_into(
            &ws.prepared,
            &ws.batch_attn,
            workers,
            &mut ws.transposed_scratch,
            &mut ws.batch_proj,
            cancel,
        )?;
        add_residual_batch(&mut ws.batch_a, &ws.batch_proj, workers, hidden)?;

        if layer_idx == last_layer && !need_logits {
            return Ok(None);
        }

        if layer_idx == last_layer {
            // Only the final prompt token feeds output_norm and tied lm_head;
            // every layer's K/V cache is already committed above (slice_last_before_ffn).
            ensure_len(&mut ws.single_norm, hidden)?;
            ensure_len(&mut ws.single_gate, c.intermediate)?;
            ensure_len(&mut ws.single_up, c.intermediate)?;
            ensure_len(&mut ws.single_proj, hidden)?;
            ffn_single_row_into(
                block,
                &mut ws.batch_a[last_offset..last_offset + hidden],
                &mut ws.single_norm,
                &mut ws.single_gate,
                &mut ws.single_up,
                &mut ws.single_proj,
                &mut ws.transposed_scratch,
                c,
                workers,
                cancel,
            )?;
        } else {
            rms_batch(
                &ws.batch_a,
                &mut ws.batch_norm,
                &block.ffn_norm,
                c.epsilon,
                hidden,
                workers,
            )?;
            ws.prepared.prepare_into(&ws.batch_norm, total, hidden);
            Projection::multiply_batch_group_into(
                [
                    (&block.gate, &mut ws.batch_gate),
                    (&block.up, &mut ws.batch_up),
                ],
                &ws.prepared,
                &ws.batch_norm,
                workers,
                &mut ws.transposed_scratch,
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
            ws.prepared
                .prepare_into(&ws.batch_gate, total, c.intermediate);
            block.down.multiply_batch_prepared_into(
                &ws.prepared,
                &ws.batch_gate,
                workers,
                &mut ws.transposed_scratch,
                &mut ws.batch_proj,
                cancel,
            )?;
            add_residual_batch(&mut ws.batch_a, &ws.batch_proj, workers, hidden)?;
        }
    }

    ensure_len(&mut ws.single_norm, hidden)?;
    rms_into(
        &ws.batch_a[last_offset..last_offset + hidden],
        &model.output_norm,
        c.epsilon,
        &mut ws.single_norm,
    )?;
    let logits = model
        .embedding
        .multiply_with_threads(&ws.single_norm, workers, cancel)?;
    check(cancel)?;
    Ok(Some(logits))
}

struct TruncateGuard<'a> {
    session: &'a mut Session,
    saved_pos: usize,
    kv_width: usize,
    success: bool,
}

impl Drop for TruncateGuard<'_> {
    fn drop(&mut self) {
        if !self.success {
            let keep = self.saved_pos * self.kv_width;
            for layer in &mut self.session.layers {
                layer.truncate_to(keep);
            }
            self.session.position = self.saved_pos;
        }
    }
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

const fn scalar(value: usize) -> f32 {
    packed::scalar_f32(value)
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
    packed::add_residual_batch(dst, add, workers, width, invalid)
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

fn normalize_and_rope_batch(
    values: &mut [f32],
    workers: usize,
    width: usize,
    norm: &[f32],
    inv_freq: &[f32],
    c: Config,
    start_pos: usize,
) -> io::Result<()> {
    row_workers::try_run_aligned(values, workers, width, &|elem_off, slice| {
        let local_start = elem_off / width;
        for (local_pos, row) in slice.chunks_exact_mut(width).enumerate() {
            normalize_heads(row, norm, c)?;
            rope_with_inv_freq(row, inv_freq, c, start_pos + local_start + local_pos)?;
        }
        Ok(())
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shares single-row SwiGLU FFN across decode steps and prefill tails with caller scratch."
)]
fn ffn_single_row_into(
    block: &Block,
    row: &mut [f32],
    norm: &mut [f32],
    gate: &mut [f32],
    up: &mut [f32],
    proj: &mut [f32],
    transposed_scratch: &mut Vec<f32>,
    c: Config,
    workers: usize,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<()> {
    rms_into(row, &block.ffn_norm, c.epsilon, norm)?;
    Projection::multiply_group_into(
        [(&block.gate, gate), (&block.up, up)],
        norm,
        workers,
        transposed_scratch,
        cancel,
    )?;
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = (*g / (1.0 + (-*g).exp())) * u;
    }
    finite(gate)?;
    block.down.multiply_into(gate, workers, proj, cancel)?;
    for (out, &value) in row.iter_mut().zip(proj.iter()) {
        *out += value;
    }
    finite(row)
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
    Projection::multiply_group_into(
        [
            (&block.query, &mut ws.single_q),
            (&block.key, &mut ws.single_k),
            (&block.value, &mut ws.single_v),
        ],
        &ws.single_norm,
        workers,
        &mut ws.transposed_scratch,
        cancel,
    )?;
    normalize_heads(&mut ws.single_q, &block.query_norm, c)?;
    normalize_heads(&mut ws.single_k, &block.key_norm, c)?;
    rope_with_inv_freq(&mut ws.single_q, &model.inv_freq, c, position)?;
    rope_with_inv_freq(&mut ws.single_k, &model.inv_freq, c, position)?;
    if matches!(c.cache_precision, CachePrecision::F16) {
        if !packed::narrow_f32_to_f16_slice(&ws.single_k, &mut ws.single_k_f16)
            || !packed::narrow_f32_to_f16_slice(&ws.single_v, &mut ws.single_v_f16)
        {
            return Err(invalid());
        }
        packed::widen_f16_slice(&ws.single_k_f16, &mut ws.single_k);
        packed::widen_f16_slice(&ws.single_v_f16, &mut ws.single_v);
    } else {
        finite(&ws.single_k)?;
        finite(&ws.single_v)?;
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
    ffn_single_row_into(
        block,
        &mut ws.single_hidden,
        &mut ws.single_norm,
        &mut ws.single_gate,
        &mut ws.single_up,
        &mut ws.single_proj,
        &mut ws.transposed_scratch,
        c,
        workers,
        cancel,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Causal attention kernel shares state/current slices and reusable score buffer without allocations."
)]
fn attend_head_causal(
    q_head: &[f32],
    state: &LayerState,
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
    match &state.data {
        LayerKv::F32(pair) => packed::attend_head(
            q_head,
            &pair.keys,
            &pair.values,
            history_rows,
            kv_width,
            offset,
            head_dim,
            divisor,
            cur_key.zip(cur_val),
            scores,
            out_head,
            invalid,
        ),
        LayerKv::F16(pair) => packed::attend_head_f16(
            q_head,
            &pair.keys,
            &pair.values,
            history_rows,
            kv_width,
            offset,
            head_dim,
            divisor,
            cur_key.zip(cur_val),
            scores,
            out_head,
            invalid,
        ),
    }
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
            state,
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
