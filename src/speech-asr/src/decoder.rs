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
use fearless_simd::{dispatch, f32x4, prelude::*};
use std::{
    cell::RefCell,
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
    #[cfg(all(windows, feature = "rust-qualification"))]
    RepackedQ4K8(Vec<u8>),
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
            #[cfg(all(windows, feature = "rust-qualification"))]
            Storage::RepackedQ4K8(_) => return Err(invalid()),
            Storage::Q6K(raw) => {
                let bytes = raw.as_slice();
                let row_bytes = self.input / 256 * 210;
                packed::decode_q6_k_row(&bytes[row * row_bytes..(row + 1) * row_bytes], output);
            }
        }
        finite(output)
    }

    /// Repack eligible `Q4_K` linear projection matrices (`input % 256 == 0`,
    /// `output % 8 == 0`) into 8-row strips tuned for ARM64 `SDOT` / `I8MM` kernels.
    ///
    /// Never call on tied/embedding matrices queried via `row_into`.
    pub(super) fn repack_q4_k_if_aligned(&mut self) {
        #[cfg(all(windows, feature = "rust-qualification"))]
        {
            if !bonsai_windows_process::quant_kernels::Arm64QuantCaps::detect().dotprod {
                return;
            }
            self.repack_q4_k_if_aligned_unchecked();
        }
    }

    #[cfg(all(windows, feature = "rust-qualification"))]
    pub(super) fn repack_q4_k_if_aligned_unchecked(&mut self) {
        if !self.output.is_multiple_of(8) || !self.input.is_multiple_of(256) {
            return;
        }
        let Storage::Q4K(raw) = &self.data else {
            return;
        };
        if let Some(repacked) = bonsai_windows_process::quant_kernels::repack_q4_k_8rows(
            raw.as_slice(),
            self.input,
            self.output,
        ) {
            self.data = Storage::RepackedQ4K8(repacked);
        }
    }

    #[cfg(all(test, windows, feature = "rust-qualification"))]
    pub(super) const fn is_raw_q4_k(&self) -> bool {
        matches!(self.data, Storage::Q4K(_))
    }

    #[cfg(all(test, windows, feature = "rust-qualification"))]
    pub(super) const fn is_repacked_q4_k(&self) -> bool {
        matches!(self.data, Storage::RepackedQ4K8(_))
    }

    pub(super) fn as_weight_matrix(&self) -> WeightMatrix<'_> {
        let data = match &self.data {
            Storage::Float(values) => WeightSlice::Float(values),
            Storage::Q8(bytes) => WeightSlice::Q8(bytes.as_slice()),
            Storage::Q4K(bytes) => WeightSlice::Q4K(bytes.as_slice()),
            #[cfg(all(windows, feature = "rust-qualification"))]
            Storage::RepackedQ4K8(bytes) => WeightSlice::RepackedQ4K8(bytes.as_slice()),
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

    /// Compute batched `[batch, input]` rows with optional bias on pre-validated internal inputs,
    /// verifying output finiteness in-register during scatter.
    #[expect(
        clippy::too_many_arguments,
        reason = "Fuses optional bias and in-register finiteness checks into batch projection scatter."
    )]
    pub(super) fn multiply_batch_prepared_biased_prevalidated_into(
        &self,
        prepared: &mut PreparedBatch,
        raw_input: &[f32],
        bias: Option<&[f32]>,
        workers: usize,
        scratch: &mut Vec<f32>,
        out: &mut [f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        self.multiply_batch_prepared_biased_checked_into(
            prepared, raw_input, bias, false, workers, scratch, out, cancel,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Shared batch projection helper parameterizes optional bias and input validation."
    )]
    fn multiply_batch_prepared_biased_checked_into(
        &self,
        prepared: &mut PreparedBatch,
        raw_input: &[f32],
        bias: Option<&[f32]>,
        check_input: bool,
        workers: usize,
        scratch: &mut Vec<f32>,
        out: &mut [f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        let batch = prepared.batch();
        if prepared.input() != self.input
            || raw_input.len() != batch.checked_mul(self.input).ok_or_else(invalid)?
            || out.len() != batch.checked_mul(self.output).ok_or_else(invalid)?
            || bias.is_some_and(|b| b.len() != self.output)
        {
            return Err(invalid());
        }
        if check_input {
            finite(raw_input)?;
        }
        for _ in 0..(batch + self.output) {
            check(cancel)?;
        }
        if !packed::multiply_batch_prepared_biased_into(
            self.as_weight_matrix(),
            prepared,
            raw_input,
            bias,
            workers,
            scratch,
            out,
        ) {
            return Err(invalid());
        }
        Ok(())
    }

    pub(super) const fn is_float(&self) -> bool {
        matches!(self.data, Storage::Float(_))
    }

    /// Compute batched channel-major `[output, batch]` rows from pre-packed finite `quads`,
    /// omitting redundant input validation when the caller guarantees finite quads and checks `out`.
    pub(super) fn multiply_batch_channel_major_prepacked_into(
        &self,
        prepared: &mut PreparedBatch,
        workers: usize,
        scratch: &mut Vec<f32>,
        out: &mut [f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        let batch = prepared.batch();
        if !self.is_float()
            || prepared.input() != self.input
            || out.len() != batch.checked_mul(self.output).ok_or_else(invalid)?
        {
            return Err(invalid());
        }
        for _ in 0..(batch + self.output) {
            check(cancel)?;
        }
        packed::multiply_batch_channel_major_into(
            self.as_weight_matrix(),
            prepared,
            &[],
            workers,
            scratch,
            out,
        );
        Ok(())
    }

    /// Compute batched channel-major `[output, batch]` rows directly using a shared `PreparedBatch`.
    pub(super) fn multiply_batch_channel_major_prepared_into(
        &self,
        prepared: &mut PreparedBatch,
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
        packed::multiply_batch_channel_major_into(
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

    pub(super) fn multiply_batch_group_prevalidated_into<const N: usize>(
        pairs: [(&Self, Option<&[f32]>, &mut [f32]); N],
        prepared: &mut PreparedBatch,
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
        for (p, bias, out) in &pairs {
            if prepared.input() != p.input
                || out.len() != batch.checked_mul(p.output).ok_or_else(invalid)?
                || bias.is_some_and(|b| b.len() != p.output)
            {
                return Err(invalid());
            }
            checks += batch + p.output;
        }
        for _ in 0..checks {
            check(cancel)?;
        }
        let mut targets = pairs.map(|(p, bias, out)| (p.as_weight_matrix(), bias, out));
        if !packed::multiply_batch_group_biased_into(
            &mut targets,
            prepared,
            raw_input,
            workers,
            scratch,
        ) {
            return Err(invalid());
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Fuses three sibling batched projections (`q+k+v`) with biases and unified scatter."
    )]
    pub(super) fn multiply_batch_triple_biased_prevalidated_into(
        p0: &Self,
        p1: &Self,
        p2: &Self,
        b0: &[f32],
        b1: &[f32],
        b2: &[f32],
        prepared: &mut PreparedBatch,
        raw_input: &[f32],
        workers: usize,
        scratch: &mut Vec<f32>,
        out0: &mut [f32],
        out1: &mut [f32],
        out2: &mut [f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        Self::multiply_batch_group_prevalidated_into(
            [
                (p0, Some(b0), out0),
                (p1, Some(b1), out1),
                (p2, Some(b2), out2),
            ],
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
    use std::sync::atomic::{AtomicBool, Ordering};

    const PARALLEL_THRESHOLD_BYTES: usize = 1024 * 1024;
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, threads::MAX_WORKERS);
    let total_blocks = bytes.len() / BLOCK_BYTES;
    if bytes.len() < PARALLEL_THRESHOLD_BYTES || workers <= 1 || total_blocks < 2 {
        let valid = bytes.as_chunks::<BLOCK_BYTES>().0.iter().all(check);
        return if valid { Ok(()) } else { Err(invalid()) };
    }
    let blocks_per_worker = total_blocks.div_ceil(workers);
    let chunk_bytes = blocks_per_worker * BLOCK_BYTES;
    let valid = AtomicBool::new(true);
    std::thread::scope(|scope| {
        let mut chunks = bytes.chunks(chunk_bytes);
        let Some(first) = chunks.next() else {
            return;
        };
        for chunk in chunks {
            scope.spawn(|| {
                if !valid.load(Ordering::Acquire) {
                    return;
                }
                if !chunk.as_chunks::<BLOCK_BYTES>().0.iter().all(check) {
                    valid.store(false, Ordering::Release);
                }
            });
        }
        if valid.load(Ordering::Acquire) && !first.as_chunks::<BLOCK_BYTES>().0.iter().all(check) {
            valid.store(false, Ordering::Release);
        }
    });
    if valid.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err(invalid())
    }
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

impl Block {
    pub(super) fn repack_q4_k_if_aligned(&mut self) {
        self.query.repack_q4_k_if_aligned();
        self.key.repack_q4_k_if_aligned();
        self.value.repack_q4_k_if_aligned();
        self.attention_out.repack_q4_k_if_aligned();
        self.gate.repack_q4_k_if_aligned();
        self.up.repack_q4_k_if_aligned();
        self.down.repack_q4_k_if_aligned();
    }
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
        mut blocks: Vec<Block>,
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
        for block in &mut blocks {
            block.repack_q4_k_if_aligned();
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

    fn write_append(
        &mut self,
        start_kv: usize,
        src_k: &[f32],
        src_v: &[f32],
        workers: usize,
    ) -> io::Result<()> {
        let len = src_k.len();
        if len != src_v.len() {
            return Err(invalid());
        }
        let active_workers = workers.clamp(1, threads::MAX_WORKERS);
        match &mut self.data {
            LayerKv::F32(pair) => {
                if active_workers > 1 && len >= 4_096 {
                    let chunk_len = len.div_ceil(active_workers).next_multiple_of(4);
                    let num_tasks = len.div_ceil(chunk_len);
                    let mut slots: [Option<std::sync::Mutex<(&mut [f32], &mut [f32])>>;
                        threads::MAX_WORKERS] = [const { None }; threads::MAX_WORKERS];
                    for (task_idx, (k_chunk, v_chunk)) in pair.keys[start_kv..start_kv + len]
                        .chunks_mut(chunk_len)
                        .zip(pair.values[start_kv..start_kv + len].chunks_mut(chunk_len))
                        .enumerate()
                    {
                        slots[task_idx] = Some(std::sync::Mutex::new((k_chunk, v_chunk)));
                    }
                    let mut task_tokens = [0.0_f32; threads::MAX_WORKERS];
                    row_workers::try_run_aligned(
                        &mut task_tokens[..num_tasks],
                        active_workers,
                        1,
                        &|start_task, slice| {
                            for task_idx in start_task..(start_task + slice.len()) {
                                let Some(slot) = slots.get(task_idx).and_then(Option::as_ref)
                                else {
                                    continue;
                                };
                                let mut guard = slot
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                let off = task_idx * chunk_len;
                                let take = guard.0.len();
                                let ks = &src_k[off..off + take];
                                let vs = &src_v[off..off + take];
                                finite(ks)?;
                                finite(vs)?;
                                guard.0.copy_from_slice(ks);
                                guard.1.copy_from_slice(vs);
                            }
                            Ok(())
                        },
                    )?;
                } else {
                    finite(src_k)?;
                    finite(src_v)?;
                    pair.keys[start_kv..start_kv + len].copy_from_slice(src_k);
                    pair.values[start_kv..start_kv + len].copy_from_slice(src_v);
                }
            }
            LayerKv::F16(pair) => {
                if active_workers > 1 && len >= 4_096 {
                    let chunk_len = len.div_ceil(active_workers).next_multiple_of(8);
                    let num_tasks = len.div_ceil(chunk_len);
                    let mut slots: [Option<std::sync::Mutex<(&mut [u16], &mut [u16])>>;
                        threads::MAX_WORKERS] = [const { None }; threads::MAX_WORKERS];
                    for (task_idx, (k_chunk, v_chunk)) in pair.keys[start_kv..start_kv + len]
                        .chunks_mut(chunk_len)
                        .zip(pair.values[start_kv..start_kv + len].chunks_mut(chunk_len))
                        .enumerate()
                    {
                        slots[task_idx] = Some(std::sync::Mutex::new((k_chunk, v_chunk)));
                    }
                    let mut task_tokens = [0.0_f32; threads::MAX_WORKERS];
                    row_workers::try_run_aligned(
                        &mut task_tokens[..num_tasks],
                        active_workers,
                        1,
                        &|start_task, slice| {
                            for task_idx in start_task..(start_task + slice.len()) {
                                let Some(slot) = slots.get(task_idx).and_then(Option::as_ref)
                                else {
                                    continue;
                                };
                                let mut guard = slot
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                let off = task_idx * chunk_len;
                                let take = guard.0.len();
                                let ks = &src_k[off..off + take];
                                let vs = &src_v[off..off + take];
                                let (k_dst, v_dst) = &mut *guard;
                                if !packed::narrow_f32_to_f16_slice(ks, k_dst)
                                    || !packed::narrow_f32_to_f16_slice(vs, v_dst)
                                {
                                    return Err(invalid());
                                }
                            }
                            Ok(())
                        },
                    )?;
                } else if !packed::narrow_f32_to_f16_slice(
                    src_k,
                    &mut pair.keys[start_kv..start_kv + len],
                ) || !packed::narrow_f32_to_f16_slice(
                    src_v,
                    &mut pair.values[start_kv..start_kv + len],
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
    rope_sin: Vec<f32>,
    rope_cos: Vec<f32>,
    prefill_k_hm: Vec<f32>,
    prefill_v_hm: Vec<f32>,
    transposed_scratch: Vec<f32>,
    prepared: PreparedBatch,
}

const RETAIN_POSITIONS: usize = 128;
const MAX_SCORE_SCRATCH_CAP: usize = 65_536;

thread_local! {
    static SCORE_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn with_score_scratch<R>(
    len: usize,
    action: impl FnOnce(&mut [f32]) -> io::Result<R>,
) -> io::Result<R> {
    SCORE_SCRATCH.with(|cell| {
        if let Ok(mut buf) = cell.try_borrow_mut() {
            buf.clear();
            if buf.capacity() < len {
                buf.try_reserve(len).map_err(io::Error::other)?;
            }
            buf.resize(len, 0.0);
            let res = action(&mut buf[..len]);
            if buf.capacity() > MAX_SCORE_SCRATCH_CAP {
                buf.clear();
                buf.shrink_to(MAX_SCORE_SCRATCH_CAP);
            }
            res
        } else {
            let mut fallback = zeros(len)?;
            action(&mut fallback)
        }
    })
}

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
        let gqa_group = (c.heads / c.kv_heads.max(1)).max(1);
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
        clear_bounded(
            &mut self.single_scores,
            retain_pos.saturating_mul(gqa_group),
        );
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
        let retain_rope = retain_pos.saturating_mul((c.head_dim / 2).max(1));
        clear_bounded(&mut self.rope_sin, retain_rope);
        clear_bounded(&mut self.rope_cos, retain_rope);
        clear_bounded(&mut self.prefill_k_hm, retain_pos.saturating_mul(kv_width));
        clear_bounded(&mut self.prefill_v_hm, retain_pos.saturating_mul(kv_width));
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
        let gqa_group = (c.heads / c.kv_heads.max(1)).max(1);
        let scores_len = self
            .position
            .saturating_add(additional)
            .max(1)
            .saturating_mul(gqa_group);
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
        let mut logits = Vec::new();
        self.step_into(id, &mut logits, cancel)?;
        Ok(logits)
    }

    /// Decode a vocabulary embedding into reusable `logits` storage and publish one position.
    ///
    /// # Errors
    /// Rejects invalid token IDs, allocation faults, or the conditions from [`Self::step_audio`].
    pub(crate) fn step_into(
        &mut self,
        id: u32,
        logits: &mut Vec<f32>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        self.step_token_into(id, Some(logits), cancel)
    }

    fn step_token_into(
        &mut self,
        id: u32,
        logits: Option<&mut Vec<f32>>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
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
        self.advance_prepared_into(logits, cancel)
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
        let mut logits = Vec::new();
        self.advance_into(row, Some(&mut logits), cancel)?;
        Ok(logits)
    }

    fn advance_into(
        &mut self,
        row: &[f32],
        logits: Option<&mut Vec<f32>>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        let c = self.model.config;
        if self.position >= c.context || row.len() != c.hidden {
            return Err(invalid());
        }
        finite(row)?;
        ensure_len(&mut self.workspace.single_hidden, c.hidden)?;
        self.workspace.single_hidden.copy_from_slice(row);
        self.advance_prepared_into(logits, cancel)
    }

    fn advance_prepared_into(
        &mut self,
        logits_out: Option<&mut Vec<f32>>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
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
        let gqa_group = c.heads / c.kv_heads;
        ensure_len(&mut ws.single_scores, gqa_group * (self.position + 1))?;
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

        let half_dim = c.head_dim / 2;
        let mut rope_sin = [0.0_f32; 128];
        let mut rope_cos = [0.0_f32; 128];
        fill_rope_tables(
            &model.inv_freq,
            self.position,
            1,
            &mut rope_sin[..half_dim],
            &mut rope_cos[..half_dim],
        );

        for (layer_idx, (block, state)) in model.blocks.iter().zip(&self.layers).enumerate() {
            check(cancel)?;
            forward_into(
                block,
                state,
                model,
                self.position,
                &rope_sin[..half_dim],
                &rope_cos[..half_dim],
                ws,
                cancel,
            )?;
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
        if let Some(logits) = logits_out {
            ensure_len(logits, model.embedding.output)?;
            model
                .embedding
                .multiply_into(&ws.single_norm, model.decode_threads, logits, cancel)?;
        }
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
        Ok(())
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
        let mut logits = Vec::new();
        self.prefill_into(prompt, audio, &mut logits, cancel)?;
        Ok(logits)
    }

    /// Start a fresh prompt and project the final row into reusable `logits` storage.
    ///
    /// # Errors
    /// Rejects audio geometry, context, allocation faults, or any token/step failure.
    pub(crate) fn prefill_into(
        &mut self,
        prompt: &Prompt,
        audio: &[f32],
        logits: &mut Vec<f32>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
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
            logits.clear();
            guard.success = true;
            return Ok(());
        }
        prefill_rows_inner(
            &model,
            &mut guard.session.layers,
            &mut guard.session.workspace,
            0,
            PrefillSource::Prompt { prompt, audio },
            Some(logits),
            cancel,
        )?;
        guard.session.position = total;
        guard.success = true;
        Ok(())
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
        if need_logits {
            let mut logits = Vec::new();
            self.prefill_tokens_into(ids, Some(&mut logits), cancel)?;
            Ok(Some(logits))
        } else {
            self.prefill_tokens_into(ids, None, cancel)?;
            Ok(None)
        }
    }

    /// Incrementally append tokens into optional reusable `logits` storage.
    ///
    /// # Errors
    /// Rejects empty slices when `logits` is `Some`, context overflow, or invalid IDs.
    pub(crate) fn prefill_tokens_into(
        &mut self,
        ids: &[u32],
        logits: Option<&mut Vec<f32>>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        check(cancel)?;
        let saved_pos = self.position;
        let model = Arc::clone(&self.model);
        let c = model.config;
        let kv_width = c.kv_heads * c.head_dim;
        let end_pos = saved_pos.checked_add(ids.len()).ok_or_else(invalid)?;
        if end_pos > c.context || (ids.is_empty() && logits.is_some()) {
            return Err(invalid());
        }
        if ids.is_empty() {
            return Ok(());
        }
        let mut guard = TruncateGuard {
            session: self,
            saved_pos,
            kv_width,
            success: false,
        };
        prefill_rows_inner(
            &model,
            &mut guard.session.layers,
            &mut guard.session.workspace,
            saved_pos,
            PrefillSource::Tokens(ids),
            logits,
            cancel,
        )?;
        guard.session.position = end_pos;
        guard.success = true;
        Ok(())
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
        prefill_rows_inner(
            &model,
            &mut guard.session.layers,
            &mut guard.session.workspace,
            saved_pos,
            PrefillSource::Audio { audio, rows },
            None,
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
    logits_out: Option<&mut Vec<f32>>,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<()> {
    let c = model.config;
    let total = source.len();
    let hidden = c.hidden;
    let kv_width = c.kv_heads * c.head_dim;
    let q_width = c.heads * c.head_dim;
    let half_dim = c.head_dim / 2;
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
    ensure_len(&mut ws.rope_sin, total * half_dim)?;
    ensure_len(&mut ws.rope_cos, total * half_dim)?;

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

    let start_kv = start_pos * kv_width;
    let add_kv = total * kv_width;
    let total_kv_rows = start_pos + total;
    for state in &mut *layers {
        if start_pos == 0 {
            state.prepare_prefill(add_kv)?;
        } else {
            state.prepare_append(start_kv, add_kv)?;
        }
    }
    ensure_len(&mut ws.prefill_k_hm, total_kv_rows * kv_width)?;
    ensure_len(&mut ws.prefill_v_hm, total_kv_rows * kv_width)?;

    let divisor = scalar(c.head_dim).sqrt();
    let gqa_group = c.heads / c.kv_heads;
    let last_layer = model.blocks.len() - 1;
    let last_offset = (total - 1) * hidden;
    let need_logits = logits_out.is_some();
    fill_rope_tables(
        &model.inv_freq,
        start_pos,
        total,
        &mut ws.rope_sin,
        &mut ws.rope_cos,
    );
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
        ws.prepared.prepare_meta(total, hidden);
        Projection::multiply_batch_group_prevalidated_into(
            [
                (&block.query, None, &mut ws.batch_q),
                (&block.key, None, &mut ws.batch_k),
                (&block.value, None, &mut ws.batch_v),
            ],
            &mut ws.prepared,
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
            &ws.rope_sin,
            &ws.rope_cos,
            c,
        )?;
        normalize_and_rope_batch(
            &mut ws.batch_k,
            workers,
            kv_width,
            &block.key_norm,
            &ws.rope_sin,
            &ws.rope_cos,
            c,
        )?;

        state.write_append(start_kv, &ws.batch_k, &ws.batch_v, workers)?;
        widen_transpose_kv_to_head_major(
            state,
            total_kv_rows,
            c.kv_heads,
            c.head_dim,
            workers,
            &mut ws.prefill_k_hm,
            &mut ws.prefill_v_hm,
        );

        for _ in 0..(total * c.heads) {
            check(cancel)?;
        }
        prefill_causal_attention_hm(
            &ws.batch_q,
            &ws.prefill_k_hm,
            &ws.prefill_v_hm,
            start_pos,
            total,
            total_kv_rows,
            c,
            gqa_group,
            divisor,
            workers,
            &mut ws.batch_attn,
        )?;

        ws.prepared.prepare_meta(total, q_width);
        block
            .attention_out
            .multiply_batch_prepared_biased_prevalidated_into(
                &mut ws.prepared,
                &ws.batch_attn,
                None,
                workers,
                &mut ws.transposed_scratch,
                &mut ws.batch_proj,
                cancel,
            )?;
        add_residual_batch(&mut ws.batch_a, &ws.batch_proj, workers, hidden)?;

        if layer_idx == last_layer && !need_logits {
            return Ok(());
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
            ws.prepared.prepare_meta(total, hidden);
            Projection::multiply_batch_group_prevalidated_into(
                [
                    (&block.gate, None, &mut ws.batch_gate),
                    (&block.up, None, &mut ws.batch_up),
                ],
                &mut ws.prepared,
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
                    if !packed::swiglu_in_place(gate_slice, &up_all[elem_off..end]) {
                        return Err(invalid());
                    }
                    Ok(())
                },
            )?;
            ws.prepared.prepare_meta(total, c.intermediate);
            block.down.multiply_batch_prepared_biased_prevalidated_into(
                &mut ws.prepared,
                &ws.batch_gate,
                None,
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
    if let Some(logits) = logits_out {
        ensure_len(logits, model.embedding.output)?;
        model
            .embedding
            .multiply_into(&ws.single_norm, workers, logits, cancel)?;
    }
    check(cancel)?;
    Ok(())
}

fn widen_transpose_kv_to_head_major(
    state: &LayerState,
    total_kv_rows: usize,
    kv_heads: usize,
    head_dim: usize,
    workers: usize,
    k_hm: &mut [f32],
    v_hm: &mut [f32],
) {
    let total_slots = kv_heads * total_kv_rows;
    if total_slots == 0 || head_dim == 0 {
        return;
    }
    let kv_width = kv_heads * head_dim;
    let copy_range = |start_slot: usize, k_dst: &mut [f32], v_dst: &mut [f32]| {
        let slots_here = k_dst.len() / head_dim;
        match &state.data {
            LayerKv::F16(pair) => {
                for local_slot in 0..slots_here {
                    let slot = start_slot + local_slot;
                    let kv_head = slot / total_kv_rows;
                    let pos = slot % total_kv_rows;
                    let src_off = pos * kv_width + kv_head * head_dim;
                    let dst_off = local_slot * head_dim;
                    packed::widen_f16_slice(
                        &pair.keys[src_off..src_off + head_dim],
                        &mut k_dst[dst_off..dst_off + head_dim],
                    );
                    packed::widen_f16_slice(
                        &pair.values[src_off..src_off + head_dim],
                        &mut v_dst[dst_off..dst_off + head_dim],
                    );
                }
            }
            LayerKv::F32(pair) => {
                for local_slot in 0..slots_here {
                    let slot = start_slot + local_slot;
                    let kv_head = slot / total_kv_rows;
                    let pos = slot % total_kv_rows;
                    let src_off = pos * kv_width + kv_head * head_dim;
                    let dst_off = local_slot * head_dim;
                    k_dst[dst_off..dst_off + head_dim]
                        .copy_from_slice(&pair.keys[src_off..src_off + head_dim]);
                    v_dst[dst_off..dst_off + head_dim]
                        .copy_from_slice(&pair.values[src_off..src_off + head_dim]);
                }
            }
        }
    };

    let active_workers = workers.clamp(1, threads::MAX_WORKERS).min(total_slots);
    if active_workers > 1 && total_kv_rows * kv_width >= 4_096 {
        let chunk_slots = total_slots.div_ceil(active_workers);
        let num_tasks = total_slots.div_ceil(chunk_slots);
        let chunk_elems = chunk_slots * head_dim;
        let mut slots: [Option<std::sync::Mutex<(&mut [f32], &mut [f32])>>; threads::MAX_WORKERS] =
            [const { None }; threads::MAX_WORKERS];
        for (task_idx, (k_chunk, v_chunk)) in k_hm
            .chunks_mut(chunk_elems)
            .zip(v_hm.chunks_mut(chunk_elems))
            .enumerate()
        {
            slots[task_idx] = Some(std::sync::Mutex::new((k_chunk, v_chunk)));
        }
        let mut task_tokens = [0.0_f32; threads::MAX_WORKERS];
        row_workers::run_aligned(
            &mut task_tokens[..num_tasks],
            active_workers,
            1,
            &|start_task, slice| {
                for task_idx in start_task..(start_task + slice.len()) {
                    let Some(slot) = slots.get(task_idx).and_then(Option::as_ref) else {
                        continue;
                    };
                    let mut guard = slot
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let (k_dst, v_dst) = &mut *guard;
                    copy_range(task_idx * chunk_slots, k_dst, v_dst);
                }
            },
        );
    } else {
        copy_range(0, k_hm, v_hm);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Runs 2-position GQA causal attention over head-major F32 KV with symmetric load balancing."
)]
fn prefill_causal_attention_hm(
    queries: &[f32],
    k_hm: &[f32],
    v_hm: &[f32],
    start_pos: usize,
    total: usize,
    total_kv_rows: usize,
    c: Config,
    gqa_group: usize,
    divisor: f32,
    workers: usize,
    batch_attn: &mut [f32],
) -> io::Result<()> {
    if total == 0 {
        return Ok(());
    }
    let q_width = c.heads * c.head_dim;
    let group_width = gqa_group * c.head_dim;
    let kv_head_stride = total_kv_rows * c.head_dim;
    let num_row_tiles = total.div_ceil(2);
    let num_pairs = num_row_tiles.div_ceil(2);
    let total_tasks = c.kv_heads * num_pairs;
    let max_history = start_pos + total;
    let score_needed = 2 * gqa_group * max_history;

    let run_tile = |kv_head: usize,
                    tile_idx: usize,
                    scores: &mut [f32],
                    out_a: &mut [f32],
                    out_b: Option<&mut [f32]>|
     -> io::Result<()> {
        let r0 = tile_idx * 2;
        let k_slab = &k_hm[kv_head * kv_head_stride..(kv_head + 1) * kv_head_stride];
        let v_slab = &v_hm[kv_head * kv_head_stride..(kv_head + 1) * kv_head_stride];
        let q_a = &queries[r0 * q_width + kv_head * group_width
            ..r0 * q_width + (kv_head + 1) * group_width];
        let len_a = start_pos + r0 + 1;
        if let Some(out_b_slice) = out_b {
            let r1 = r0 + 1;
            let len_b = len_a + 1;
            let q_b = &queries[r1 * q_width + kv_head * group_width
                ..r1 * q_width + (kv_head + 1) * group_width];
            let (sa, sb_rest) = scores.split_at_mut(gqa_group * len_a);
            let sb = &mut sb_rest[..gqa_group * len_b];
            packed::attend_kv_group_2pos_f32(
                q_a,
                q_b,
                k_slab,
                v_slab,
                len_a,
                len_b,
                c.head_dim,
                gqa_group,
                divisor,
                sa,
                sb,
                out_a,
                out_b_slice,
                invalid,
            )
        } else {
            packed::attend_kv_group_f32(
                q_a,
                k_slab,
                v_slab,
                len_a,
                c.head_dim,
                0,
                c.head_dim,
                gqa_group,
                divisor,
                None,
                &mut scores[..gqa_group * len_a],
                out_a,
                invalid,
            )?;
            finite(out_a)
        }
    };

    let active_workers = workers.clamp(1, threads::MAX_WORKERS).min(total_tasks);
    if active_workers > 1 && total_tasks >= 2 {
        let mut attn_slots = Vec::new();
        attn_slots
            .try_reserve_exact(total * c.kv_heads)
            .map_err(io::Error::other)?;
        for chunk in batch_attn.chunks_exact_mut(group_width) {
            attn_slots.push(std::sync::Mutex::new(chunk));
        }
        let next_task = std::sync::atomic::AtomicUsize::new(0);
        let mut task_tokens = [0.0_f32; threads::MAX_WORKERS];
        row_workers::try_run_aligned(
            &mut task_tokens[..active_workers],
            active_workers,
            1,
            &|_, _| {
                with_score_scratch(score_needed, |scores| {
                    loop {
                        let task_id = next_task.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if task_id >= total_tasks {
                            break;
                        }
                        let kv_head = task_id / num_pairs;
                        let pair_idx = task_id % num_pairs;
                        let t_lo = pair_idx;
                        let t_hi = num_row_tiles - 1 - pair_idx;
                        for tile_idx in [Some(t_lo), (t_hi != t_lo).then_some(t_hi)]
                            .into_iter()
                            .flatten()
                        {
                            let r0 = tile_idx * 2;
                            let mut out_a = attn_slots[r0 * c.kv_heads + kv_head]
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            if r0 + 1 < total {
                                let mut out_b = attn_slots[(r0 + 1) * c.kv_heads + kv_head]
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                run_tile(kv_head, tile_idx, scores, &mut out_a, Some(&mut out_b))?;
                            } else {
                                run_tile(kv_head, tile_idx, scores, &mut out_a, None)?;
                            }
                        }
                    }
                    Ok(())
                })
            },
        )?;
        return Ok(());
    }

    with_score_scratch(score_needed, |scores| {
        for (tile_idx, rows_chunk) in batch_attn.chunks_mut(2 * q_width).enumerate() {
            let r0 = tile_idx * 2;
            if r0 + 1 < total {
                let (row_a, row_b) = rows_chunk.split_at_mut(q_width);
                for (kv_head, (out_a, out_b)) in row_a
                    .chunks_exact_mut(group_width)
                    .zip(row_b.chunks_exact_mut(group_width))
                    .enumerate()
                {
                    run_tile(kv_head, tile_idx, scores, out_a, Some(out_b))?;
                }
            } else {
                for (kv_head, out_a) in rows_chunk.chunks_exact_mut(group_width).enumerate() {
                    run_tile(kv_head, tile_idx, scores, out_a, None)?;
                }
            }
        }
        Ok(())
    })
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

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn rms_into_simd<S: Simd>(
    simd: S,
    input: &[f32],
    weights: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> io::Result<()> {
    if input.is_empty() || output.len() != input.len() || weights.len() != input.len() {
        return Err(invalid());
    }
    let (chunks, tail) = input.as_chunks::<4>();
    let mut sum_v = f32x4::splat(simd, 0.0_f32);
    for chunk in chunks {
        let x = f32x4::from_slice(simd, chunk);
        sum_v += x * x;
    }
    let lanes: [f32; 4] = sum_v.into();
    let mut squares = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for &value in tail {
        squares += value * value;
    }
    let denom = scalar(input.len());
    if !squares.is_finite() || !(squares / denom + epsilon).is_finite() {
        return Err(invalid());
    }
    let scale = 1.0 / (squares / denom + epsilon).sqrt();
    let scale_v = f32x4::splat(simd, scale);
    let inf_v = f32x4::splat(simd, f32::INFINITY);
    let mut all_finite = true;
    let (dst_chunks, dst_tail) = output.as_chunks_mut::<4>();
    let (w_chunks, w_tail) = weights.as_chunks::<4>();
    for ((dst_chunk, in_chunk), w_chunk) in dst_chunks.iter_mut().zip(chunks).zip(w_chunks) {
        let x = f32x4::from_slice(simd, in_chunk);
        let w = f32x4::from_slice(simd, w_chunk);
        let res = (x * scale_v) * w;
        if !res.abs().simd_lt(inf_v).all_true() {
            all_finite = false;
        }
        res.store_slice(dst_chunk);
    }
    for ((dst, &value), &w) in dst_tail.iter_mut().zip(tail).zip(w_tail) {
        let res = (value * scale) * w;
        if !res.is_finite() {
            all_finite = false;
        }
        *dst = res;
    }
    if all_finite { Ok(()) } else { Err(invalid()) }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn rms_in_place_simd<S: Simd>(
    simd: S,
    values: &mut [f32],
    weights: &[f32],
    epsilon: f32,
) -> io::Result<()> {
    if values.is_empty() || weights.len() != values.len() {
        return Err(invalid());
    }
    let mut sum_v = f32x4::splat(simd, 0.0_f32);
    {
        let (chunks, tail) = values.as_chunks::<4>();
        for chunk in chunks {
            let x = f32x4::from_slice(simd, chunk);
            sum_v += x * x;
        }
        let lanes: [f32; 4] = sum_v.into();
        let mut squares = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        for &value in tail {
            squares += value * value;
        }
        let denom = scalar(values.len());
        if !squares.is_finite() || !(squares / denom + epsilon).is_finite() {
            return Err(invalid());
        }
        let scale = 1.0 / (squares / denom + epsilon).sqrt();
        let scale_v = f32x4::splat(simd, scale);
        let inf_v = f32x4::splat(simd, f32::INFINITY);
        let mut all_finite = true;
        let (val_chunks, val_tail) = values.as_chunks_mut::<4>();
        let (w_chunks, w_tail) = weights.as_chunks::<4>();
        for (val_chunk, w_chunk) in val_chunks.iter_mut().zip(w_chunks) {
            let x = f32x4::from_slice(simd, val_chunk);
            let w = f32x4::from_slice(simd, w_chunk);
            let res = (x * scale_v) * w;
            if !res.abs().simd_lt(inf_v).all_true() {
                all_finite = false;
            }
            res.store_slice(val_chunk);
        }
        for (value, &w) in val_tail.iter_mut().zip(w_tail) {
            let res = (*value * scale) * w;
            if !res.is_finite() {
                all_finite = false;
            }
            *value = res;
        }
        if all_finite { Ok(()) } else { Err(invalid()) }
    }
}

fn rms_into(input: &[f32], weights: &[f32], epsilon: f32, output: &mut [f32]) -> io::Result<()> {
    let level = packed::simd_level();
    dispatch!(level, simd => rms_into_simd(simd, input, weights, epsilon, output))
}

fn rms_batch(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    epsilon: f32,
    width: usize,
    workers: usize,
) -> io::Result<()> {
    let level = packed::simd_level();
    row_workers::try_run_aligned(output, workers, width, &|elem_off, slice| {
        dispatch!(level, simd => {
            for (src_row, dst_row) in input[elem_off..elem_off + slice.len()]
                .chunks_exact(width)
                .zip(slice.chunks_exact_mut(width))
            {
                rms_into_simd(simd, src_row, weights, epsilon, dst_row)?;
            }
            Ok(())
        })
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

fn fill_rope_tables(
    inv_freq: &[f32],
    start_pos: usize,
    rows: usize,
    sin_out: &mut [f32],
    cos_out: &mut [f32],
) {
    let half = inv_freq.len();
    if half == 0 || rows == 0 {
        return;
    }
    for (local_pos, (sin_row, cos_row)) in sin_out
        .chunks_exact_mut(half)
        .zip(cos_out.chunks_exact_mut(half))
        .enumerate()
    {
        let pos = scalar(start_pos + local_pos);
        for ((s, c), &freq) in sin_row.iter_mut().zip(cos_row.iter_mut()).zip(inv_freq) {
            (*s, *c) = (pos * freq).sin_cos();
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn rope_head_simd<S: Simd>(
    simd: S,
    lo: &mut [f32],
    hi: &mut [f32],
    sin: &[f32],
    cos: &[f32],
) -> bool {
    let inf_v = f32x4::splat(simd, f32::INFINITY);
    let mut all_finite = true;
    let (lo_chunks, lo_tail) = lo.as_chunks_mut::<4>();
    let (hi_chunks, hi_tail) = hi.as_chunks_mut::<4>();
    let (sin_chunks, sin_tail) = sin.as_chunks::<4>();
    let (cos_chunks, cos_tail) = cos.as_chunks::<4>();
    for (((lc, hc), sc), cc) in lo_chunks
        .iter_mut()
        .zip(hi_chunks)
        .zip(sin_chunks)
        .zip(cos_chunks)
    {
        let a = f32x4::from_slice(simd, lc);
        let b = f32x4::from_slice(simd, hc);
        let s = f32x4::from_slice(simd, sc);
        let c = f32x4::from_slice(simd, cc);
        let r0 = a * c - b * s;
        let r1 = a * s + b * c;
        if !r0.abs().simd_lt(inf_v).all_true() || !r1.abs().simd_lt(inf_v).all_true() {
            all_finite = false;
        }
        r0.store_slice(lc);
        r1.store_slice(hc);
    }
    for (((a_ref, b_ref), &s), &c) in lo_tail.iter_mut().zip(hi_tail).zip(sin_tail).zip(cos_tail) {
        let (a, b) = (*a_ref, *b_ref);
        let r0 = a * c - b * s;
        let r1 = a * s + b * c;
        if !r0.is_finite() || !r1.is_finite() {
            all_finite = false;
        }
        *a_ref = r0;
        *b_ref = r1;
    }
    all_finite
}

fn normalize_and_rope_heads(
    values: &mut [f32],
    weights: &[f32],
    sin: &[f32],
    cos: &[f32],
    c: Config,
) -> io::Result<()> {
    let half = c.head_dim / 2;
    let level = packed::simd_level();
    dispatch!(level, simd => {
        for head in values.chunks_exact_mut(c.head_dim) {
            rms_in_place_simd(simd, head, weights, c.epsilon)?;
            let (lo, hi) = head.split_at_mut(half);
            if !rope_head_simd(simd, lo, hi, sin, cos) {
                return Err(invalid());
            }
        }
        Ok(())
    })
}

fn normalize_and_rope_batch(
    values: &mut [f32],
    workers: usize,
    width: usize,
    norm: &[f32],
    rope_sin: &[f32],
    rope_cos: &[f32],
    c: Config,
) -> io::Result<()> {
    let half = c.head_dim / 2;
    let level = packed::simd_level();
    row_workers::try_run_aligned(values, workers, width, &|elem_off, slice| {
        let local_start = elem_off / width;
        dispatch!(level, simd => {
            for (local_pos, row) in slice.chunks_exact_mut(width).enumerate() {
                let pos_idx = local_start + local_pos;
                let sin = &rope_sin[pos_idx * half..(pos_idx + 1) * half];
                let cos = &rope_cos[pos_idx * half..(pos_idx + 1) * half];
                for head in row.chunks_exact_mut(c.head_dim) {
                    rms_in_place_simd(simd, head, norm, c.epsilon)?;
                    let (lo, hi) = head.split_at_mut(half);
                    if !rope_head_simd(simd, lo, hi, sin, cos) {
                        return Err(invalid());
                    }
                }
            }
            Ok(())
        })
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
    if !packed::swiglu_in_place(gate, up) {
        return Err(invalid());
    }
    block.down.multiply_into(gate, workers, proj, cancel)?;
    for (out, &value) in row.iter_mut().zip(proj.iter()) {
        *out += value;
    }
    finite(row)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Single-position decoder step shares workspace and precomputed RoPE tables across layers."
)]
fn forward_into(
    block: &Block,
    state: &LayerState,
    model: &Model,
    position: usize,
    rope_sin: &[f32],
    rope_cos: &[f32],
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
    normalize_and_rope_heads(&mut ws.single_q, &block.query_norm, rope_sin, rope_cos, c)?;
    normalize_and_rope_heads(&mut ws.single_k, &block.key_norm, rope_sin, rope_cos, c)?;
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
        workers,
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

#[cfg(test)]
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
    reason = "Grouped causal attention shares state/current slices and reusable score buffer without allocations."
)]
fn attend_kv_group_causal(
    q_group: &[f32],
    state: &LayerState,
    cur_key: Option<&[f32]>,
    cur_val: Option<&[f32]>,
    history_rows: usize,
    kv_width: usize,
    offset: usize,
    head_dim: usize,
    gqa_group: usize,
    divisor: f32,
    scores: &mut [f32],
    out_group: &mut [f32],
) -> io::Result<()> {
    match &state.data {
        LayerKv::F32(pair) => packed::attend_kv_group_f32(
            q_group,
            &pair.keys,
            &pair.values,
            history_rows,
            kv_width,
            offset,
            head_dim,
            gqa_group,
            divisor,
            cur_key.zip(cur_val),
            scores,
            out_group,
            invalid,
        ),
        LayerKv::F16(pair) => packed::attend_kv_group_f16(
            q_group,
            &pair.keys,
            &pair.values,
            history_rows,
            kv_width,
            offset,
            head_dim,
            gqa_group,
            divisor,
            cur_key.zip(cur_val),
            scores,
            out_group,
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
    workers: usize,
    scores: &mut [f32],
    output: &mut [f32],
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<()> {
    let width = c.kv_heads * c.head_dim;
    let divisor = scalar(c.head_dim).sqrt();
    let gqa_group = c.heads / c.kv_heads;
    let group_width = gqa_group * c.head_dim;
    let total_rows = position + 1;
    let group_scores_len = gqa_group * total_rows;
    if output.len() != c.heads * c.head_dim || query.len() != c.heads * c.head_dim {
        return Err(invalid());
    }
    if workers > 1 && c.kv_heads > 1 && position >= 32 {
        for _ in 0..c.heads {
            check(cancel)?;
        }
        row_workers::try_run_aligned(output, workers, group_width, &|elem_off, out_chunk| {
            let kv_start = elem_off / group_width;
            let kv_count = out_chunk.len() / group_width;
            with_score_scratch(group_scores_len, |local_scores| {
                for local_kv in 0..kv_count {
                    let kv_head = kv_start + local_kv;
                    let offset = kv_head * c.head_dim;
                    let q_group = &query[kv_head * group_width..(kv_head + 1) * group_width];
                    let out_group =
                        &mut out_chunk[local_kv * group_width..(local_kv + 1) * group_width];
                    attend_kv_group_causal(
                        q_group,
                        state,
                        Some(&keys[offset..offset + c.head_dim]),
                        Some(&values[offset..offset + c.head_dim]),
                        position,
                        width,
                        offset,
                        c.head_dim,
                        gqa_group,
                        divisor,
                        local_scores,
                        out_group,
                    )?;
                }
                Ok(())
            })
        })?;
        check(cancel)?;
    } else {
        let row_scores = scores.get_mut(..group_scores_len).ok_or_else(invalid)?;
        for kv_head in 0..c.kv_heads {
            for _ in 0..gqa_group {
                check(cancel)?;
            }
            let offset = kv_head * c.head_dim;
            let q_group = &query[kv_head * group_width..(kv_head + 1) * group_width];
            let out_group = &mut output[kv_head * group_width..(kv_head + 1) * group_width];
            attend_kv_group_causal(
                q_group,
                state,
                Some(&keys[offset..offset + c.head_dim]),
                Some(&values[offset..offset + c.head_dim]),
                position,
                width,
                offset,
                c.head_dim,
                gqa_group,
                divisor,
                row_scores,
                out_group,
            )?;
        }
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
    attend_with_workers(query, keys, values, state, c, position, 1, cancel)
}

#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "Tests parallel and sequential GQA attention parity against single-head attend_head_causal."
)]
fn attend_with_workers(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    state: &LayerState,
    c: Config,
    position: usize,
    workers: usize,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Vec<f32>> {
    let gqa_group = c.heads / c.kv_heads;
    let mut output = zeros(c.heads * c.head_dim)?;
    let mut scores = zeros(gqa_group * (position + 1))?;
    attend_into(
        query,
        keys,
        values,
        state,
        c,
        position,
        workers,
        &mut scores,
        &mut output,
        cancel,
    )?;
    Ok(output)
}

#[cfg(test)]
mod gqa_attention_tests {
    use super::{
        CachePrecision, Config, LayerKv, LayerState, attend_head_causal, attend_into,
        attend_with_workers, packed, scalar, zeros,
    };
    use std::io;

    fn make_inputs(
        c: Config,
        position: usize,
    ) -> io::Result<(Vec<f32>, Vec<f32>, Vec<f32>, LayerState)> {
        let q_len = c.heads * c.head_dim;
        let kv_width = c.kv_heads * c.head_dim;
        let query: Vec<f32> = (0..q_len)
            .map(|i| {
                let v = i32::try_from((i * 17 + 3) % 29).unwrap_or(0) - 14;
                f32::from(i8::try_from(v).unwrap_or(0)) * 0.0625
            })
            .collect();
        let cur_k: Vec<f32> = (0..kv_width)
            .map(|i| {
                let v = i32::try_from((i * 11 + 5) % 23).unwrap_or(0) - 11;
                f32::from(i8::try_from(v).unwrap_or(0)) * 0.05
            })
            .collect();
        let cur_v: Vec<f32> = (0..kv_width)
            .map(|i| {
                let v = i32::try_from((i * 13 + 7) % 19).unwrap_or(0) - 9;
                f32::from(i8::try_from(v).unwrap_or(0)) * 0.03125
            })
            .collect();
        let hist_k: Vec<f32> = (0..position * kv_width)
            .map(|i| {
                let v = i32::try_from((i * 7 + 11) % 31).unwrap_or(0) - 15;
                f32::from(i8::try_from(v).unwrap_or(0)) * 0.05
            })
            .collect();
        let hist_v: Vec<f32> = (0..position * kv_width)
            .map(|i| {
                let v = i32::try_from((i * 23 + 1) % 25).unwrap_or(0) - 12;
                f32::from(i8::try_from(v).unwrap_or(0)) * 0.0625
            })
            .collect();

        let state = match c.cache_precision {
            CachePrecision::F32 => LayerState::from_f32(hist_k, hist_v),
            CachePrecision::F16 => {
                let mut hk_u16 = vec![0_u16; hist_k.len()];
                let mut hv_u16 = vec![0_u16; hist_v.len()];
                assert!(packed::narrow_f32_to_f16_slice(&hist_k, &mut hk_u16));
                assert!(packed::narrow_f32_to_f16_slice(&hist_v, &mut hv_u16));
                LayerState {
                    data: LayerKv::F16(super::KvPair {
                        keys: hk_u16,
                        values: hv_u16,
                    }),
                }
            }
        };
        let (final_k, final_v) = if matches!(c.cache_precision, CachePrecision::F16) {
            let mut k16 = vec![0_u16; kv_width];
            let mut v16 = vec![0_u16; kv_width];
            assert!(packed::narrow_f32_to_f16_slice(&cur_k, &mut k16));
            assert!(packed::narrow_f32_to_f16_slice(&cur_v, &mut v16));
            let mut kf = vec![0.0_f32; kv_width];
            let mut vf = vec![0.0_f32; kv_width];
            packed::widen_f16_slice(&k16, &mut kf);
            packed::widen_f16_slice(&v16, &mut vf);
            (kf, vf)
        } else {
            (cur_k, cur_v)
        };
        Ok((query, final_k, final_v, state))
    }

    #[test]
    fn parallel_gqa_grouped_attend_into_matches_single_head_bit_for_bit() -> io::Result<()> {
        let combos = [
            (CachePrecision::F32, 6_usize),
            (CachePrecision::F32, 64_usize),
            (CachePrecision::F16, 6_usize),
            (CachePrecision::F16, 64_usize),
        ];
        std::thread::scope(|scope| -> io::Result<()> {
            let handles: Vec<_> = combos
                .into_iter()
                .map(|(precision, head_dim)| {
                    scope.spawn(move || -> io::Result<()> {
                        for gqa_ratio in [1_usize, 2, 4, 8] {
                            for position in [32_usize, 67] {
                                let kv_heads = 4_usize;
                                let c = Config {
                                    hidden: 64,
                                    heads: kv_heads * gqa_ratio,
                                    kv_heads,
                                    head_dim,
                                    intermediate: 128,
                                    context: 256,
                                    epsilon: 1e-6,
                                    rope_theta: 10_000.0,
                                    cache_precision: precision,
                                };
                                let (query, cur_k, cur_v, state) = make_inputs(c, position)?;
                                let kv_width = c.kv_heads * c.head_dim;
                                let divisor = scalar(c.head_dim).sqrt();
                                let mut expected = zeros(c.heads * c.head_dim)?;
                                let mut head_scores = zeros(position + 1)?;
                                for head in 0..c.heads {
                                    let kv_head = head / gqa_ratio;
                                    let offset = kv_head * c.head_dim;
                                    attend_head_causal(
                                        &query[head * c.head_dim..(head + 1) * c.head_dim],
                                        &state,
                                        Some(&cur_k[offset..offset + c.head_dim]),
                                        Some(&cur_v[offset..offset + c.head_dim]),
                                        position,
                                        kv_width,
                                        offset,
                                        c.head_dim,
                                        divisor,
                                        &mut head_scores,
                                        &mut expected[head * c.head_dim..(head + 1) * c.head_dim],
                                    )?;
                                }
                                let seq_grouped = attend_with_workers(
                                    &query,
                                    &cur_k,
                                    &cur_v,
                                    &state,
                                    c,
                                    position,
                                    1,
                                    &mut || false,
                                )?;
                                let par_grouped = attend_with_workers(
                                    &query,
                                    &cur_k,
                                    &cur_v,
                                    &state,
                                    c,
                                    position,
                                    4,
                                    &mut || false,
                                )?;
                                let expected_bits: Vec<u32> =
                                    expected.iter().map(|v| v.to_bits()).collect();
                                let seq_bits: Vec<u32> =
                                    seq_grouped.iter().map(|v| v.to_bits()).collect();
                                let par_bits: Vec<u32> =
                                    par_grouped.iter().map(|v| v.to_bits()).collect();
                                assert_eq!(
                                    seq_bits, expected_bits,
                                    "seq mismatch gqa={gqa_ratio} pos={position} dim={head_dim}"
                                );
                                assert_eq!(
                                    par_bits, expected_bits,
                                    "par mismatch gqa={gqa_ratio} pos={position} dim={head_dim}"
                                );
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| io::Error::other("GQA test worker panicked"))??;
            }
            Ok(())
        })
    }

    #[test]
    fn parallel_attend_into_checks_cancellation_before_and_after_dispatch() -> io::Result<()> {
        let c = Config {
            hidden: 64,
            heads: 8,
            kv_heads: 4,
            head_dim: 8,
            intermediate: 128,
            context: 256,
            epsilon: 1e-6,
            rope_theta: 10_000.0,
            cache_precision: CachePrecision::F16,
        };
        let position = 35;
        let (query, cur_k, cur_v, state) = make_inputs(c, position)?;
        let mut scores = zeros(2 * (position + 1))?;
        let mut output = zeros(c.heads * c.head_dim)?;

        let mut pre_polls = 0_usize;
        let pre_err = attend_into(
            &query,
            &cur_k,
            &cur_v,
            &state,
            c,
            position,
            4,
            &mut scores,
            &mut output,
            &mut || {
                pre_polls += 1;
                pre_polls == 3
            },
        );
        assert_eq!(
            pre_err.err().map(|e| e.kind()),
            Some(io::ErrorKind::Interrupted)
        );
        assert_eq!(pre_polls, 3);

        let mut post_polls = 0_usize;
        let post_err = attend_into(
            &query,
            &cur_k,
            &cur_v,
            &state,
            c,
            position,
            4,
            &mut scores,
            &mut output,
            &mut || {
                post_polls += 1;
                post_polls > c.heads
            },
        );
        assert_eq!(
            post_err.err().map(|e| e.kind()),
            Some(io::ErrorKind::Interrupted)
        );
        assert_eq!(post_polls, c.heads + 1);
        Ok(())
    }
}

#[cfg(test)]
#[path = "decoder/tests.rs"]
mod tests;
