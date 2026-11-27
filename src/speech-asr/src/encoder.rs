//! Independent scalar Qwen3-ASR CNN and bidirectional transformer encoder.
//!
//! Three stride-two convolutions pad each chunk separately. Positional sinusoids
//! restart per chunk, padded tail rows are removed before full self-attention,
//! and the post-normalization/two-projection head emits decoder audio rows.
//! `rust-encoder.json` identifies this arithmetic and its outstanding parity gates.
#![forbid(unsafe_code)]
#![allow(
    clippy::suboptimal_flops,
    reason = "Separate scalar operations define this independent encoder reference."
)]

#[cfg(test)]
use crate::decoder::binary16_operand;
use crate::{
    decoder::Projection,
    frontend::{Frontend, MelSpectrogram},
    packed::{self, PreparedBatch},
    protocol::EncoderLayout,
    row_workers, threads,
};
use fearless_simd::{dispatch, f32x4, prelude::*};
use std::{
    cell::RefCell,
    io::{self, Read, Seek},
};

#[path = "encoder/numerics.rs"]
mod numerics;

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid Rust Qwen3-ASR encoder input, weights or geometry.",
    )
}
fn finite(values: &[f32]) -> io::Result<()> {
    packed::ensure_finite(values, invalid)
}
fn check(cancel: &mut impl FnMut() -> bool) -> io::Result<()> {
    if cancel() {
        Err(io::Error::from(io::ErrorKind::Interrupted))
    } else {
        Ok(())
    }
}
fn ensure_buffer(values: &mut Vec<f32>, count: usize) -> io::Result<()> {
    if count > 128 * 1024 * 1024 {
        return Err(invalid());
    }
    if values.capacity() < count {
        values
            .try_reserve(count - values.len())
            .map_err(io::Error::other)?;
    }
    values.resize(count, 0.0);
    Ok(())
}
fn zeros(count: usize) -> io::Result<Vec<f32>> {
    let mut values = Vec::new();
    ensure_buffer(&mut values, count)?;
    Ok(values)
}
const fn scalar(value: usize) -> f32 {
    packed::scalar_f32(value)
}

/// A biased row-major linear projection.
pub struct Linear {
    /// Weights in `[output, input]` row order.
    pub weights: Projection,
    /// Per-output bias.
    pub bias: Vec<f32>,
}
impl Linear {
    fn validate(&self, input: usize, output: usize) -> io::Result<()> {
        if self.weights.dimensions() != (input, output) || self.bias.len() != output {
            return Err(invalid());
        }
        finite(&self.bias)
    }

    fn apply_batch_prepared(
        &self,
        prepared: &mut PreparedBatch,
        raw_input: &[f32],
        workers: usize,
        scratch: &mut Vec<f32>,
        output: &mut [f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        self.weights
            .multiply_batch_prepared_into(prepared, raw_input, workers, scratch, output, cancel)?;
        add_batch_bias(output, &self.bias, workers)
    }
}

/// Layer normalization's finite per-channel scale and bias.
pub struct Norm {
    /// Channel scale.
    pub weight: Vec<f32>,
    /// Channel bias.
    pub bias: Vec<f32>,
}
impl Norm {
    fn validate(&self, width: usize) -> io::Result<()> {
        if self.weight.len() != width || self.bias.len() != width {
            return Err(invalid());
        }
        finite(&self.weight)?;
        finite(&self.bias)
    }

    #[cfg(test)]
    fn apply_into(&self, row: &[f32], dst: &mut [f32]) -> io::Result<()> {
        let level = packed::simd_level();
        dispatch!(
            level,
            simd => norm_apply_into_simd(simd, &self.weight, &self.bias, row, dst)
        )
    }

    #[cfg(test)]
    fn apply(&self, row: &[f32]) -> io::Result<Vec<f32>> {
        let mut output = zeros(row.len())?;
        self.apply_into(row, &mut output)?;
        Ok(output)
    }

    fn apply_batch(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        workers: usize,
    ) -> io::Result<()> {
        if width == 0 || input.len() != output.len() || !output.len().is_multiple_of(width) {
            return Err(invalid());
        }
        let level = packed::simd_level();
        row_workers::try_run_aligned(output, workers, width, &|elem_off, slice| {
            dispatch!(level, simd => {
                for (src_row, dst_row) in input[elem_off..elem_off + slice.len()]
                    .chunks_exact(width)
                    .zip(slice.chunks_exact_mut(width))
                {
                    norm_apply_into_simd(simd, &self.weight, &self.bias, src_row, dst_row)?;
                }
                Ok(())
            })
        })
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn norm_apply_into_simd<S: Simd>(
    simd: S,
    weight: &[f32],
    bias: &[f32],
    row: &[f32],
    dst: &mut [f32],
) -> io::Result<()> {
    let width = row.len();
    if width == 0 || dst.len() != width || weight.len() != width || bias.len() != width {
        return Err(invalid());
    }
    let (chunks, tail) = row.as_chunks::<4>();
    let mut sum_v = f32x4::splat(simd, 0.0_f32);
    for chunk in chunks {
        sum_v += f32x4::from_slice(simd, chunk);
    }
    let sum_lanes: [f32; 4] = sum_v.into();
    let mut sum = (sum_lanes[0] + sum_lanes[1]) + (sum_lanes[2] + sum_lanes[3]);
    for &value in tail {
        sum += value;
    }
    let denom = scalar(width);
    let mean = sum / denom;
    let mean_v = f32x4::splat(simd, mean);
    let mut var_v = f32x4::splat(simd, 0.0_f32);
    for chunk in chunks {
        let delta = f32x4::from_slice(simd, chunk) - mean_v;
        var_v += delta * delta;
    }
    let var_lanes: [f32; 4] = var_v.into();
    let mut var_sum = (var_lanes[0] + var_lanes[1]) + (var_lanes[2] + var_lanes[3]);
    for &value in tail {
        let delta = value - mean;
        var_sum += delta * delta;
    }
    let variance = var_sum / denom;
    if !mean.is_finite() || !variance.is_finite() {
        return Err(invalid());
    }
    let scale = 1.0 / (variance + 0.000_01).sqrt();
    let scale_v = f32x4::splat(simd, scale);
    let inf_v = f32x4::splat(simd, f32::INFINITY);
    let mut all_finite = true;
    let (dst_chunks, dst_tail) = dst.as_chunks_mut::<4>();
    let (w_chunks, w_tail) = weight.as_chunks::<4>();
    let (b_chunks, b_tail) = bias.as_chunks::<4>();
    for (((out_chunk, in_chunk), w_chunk), b_chunk) in dst_chunks
        .iter_mut()
        .zip(chunks)
        .zip(w_chunks)
        .zip(b_chunks)
    {
        let x = f32x4::from_slice(simd, in_chunk);
        let w = f32x4::from_slice(simd, w_chunk);
        let b = f32x4::from_slice(simd, b_chunk);
        let res = (((x - mean_v) * scale_v) * w) + b;
        if !res.abs().simd_lt(inf_v).all_true() {
            all_finite = false;
        }
        res.store_slice(out_chunk);
    }
    for (((out, &value), &w), &b) in dst_tail.iter_mut().zip(tail).zip(w_tail).zip(b_tail) {
        let res = (((value - mean) * scale) * w) + b;
        if !res.is_finite() {
            all_finite = false;
        }
        *out = res;
    }
    if all_finite { Ok(()) } else { Err(invalid()) }
}

/// One pre-normalized, biased, bidirectional encoder transformer block.
pub struct Block {
    /// Pre-attention layer normalization.
    pub attention_norm: Norm,
    /// Pre-FFN layer normalization.
    pub ffn_norm: Norm,
    /// Query projection.
    pub query: Linear,
    /// Key projection.
    pub key: Linear,
    /// Value projection.
    pub value: Linear,
    /// Attention output projection.
    pub attention_out: Linear,
    /// FFN input projection before erf-based GELU.
    pub ffn_in: Linear,
    /// FFN output projection.
    pub ffn_out: Linear,
}

impl Block {
    pub(super) fn repack_q4_k_if_aligned(&mut self) {
        self.query.weights.repack_q4_k_if_aligned();
        self.key.weights.repack_q4_k_if_aligned();
        self.value.weights.repack_q4_k_if_aligned();
        self.attention_out.weights.repack_q4_k_if_aligned();
        self.ffn_in.weights.repack_q4_k_if_aligned();
        self.ffn_out.weights.repack_q4_k_if_aligned();
    }
}

/// Fixed geometry from the verified encoder metadata.
#[derive(Clone, Copy)]
pub struct Config {
    /// Number of mel-frequency bins.
    pub mel_bins: usize,
    /// Output channels of each of the three convolutions.
    pub channels: usize,
    /// Transformer width, even and at least four.
    pub hidden: usize,
    /// Number of attention heads, dividing `hidden`.
    pub heads: usize,
    /// FFN intermediate width.
    pub intermediate: usize,
    /// Decoder audio-row width.
    pub output: usize,
    /// Half of the independently padded convolution chunk width.
    pub window: usize,
    /// Maximum supported sinusoidal positions per chunk.
    pub max_positions: usize,
}

/// Complete immutable encoder graph weights.
pub struct Weights {
    /// Three convolution projections, rows `[output_channel, input_channel, ky, kx]`.
    pub convolutions: [Linear; 3],
    /// Projection after channel/frequency flattening, without a bias.
    pub convolution_out: Projection,
    /// Ordered bidirectional transformer blocks.
    pub blocks: Vec<Block>,
    /// Post-transformer layer normalization.
    pub output_norm: Norm,
    /// First output projection before erf-based GELU.
    pub output_in: Linear,
    /// Final decoder-width output projection.
    pub output_out: Linear,
}

/// One admitted immutable CNN/transformer encoder.
pub struct Encoder {
    config: Config,
    weights: Weights,
    sinusoids: Vec<f32>,
    threads: usize,
}

impl Encoder {
    /// Configure bounded worker threads for encoder convolutions, projections, and attention.
    ///
    /// # Errors
    /// Rejects thread counts outside `1..=256`.
    pub fn with_threads(mut self, threads: usize) -> io::Result<Self> {
        self.threads = threads::validate_thread_count(threads)?;
        Ok(self)
    }

    /// Decoder width of each complete audio row.
    #[must_use]
    pub const fn output_width(&self) -> usize {
        self.config.output
    }
    /// Read encoder weights and matching frontend tables from a verified GGUF.
    ///
    /// The caller owns SHA-256 verification and immutable artifact guards.
    ///
    /// # Errors
    /// Rejects bad/unsupported metadata, tensor storage, nonfinite values,
    /// cancellation or I/O/allocation failures.
    pub fn load(
        source: &mut (impl Read + Seek),
        length: u64,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<(Self, Frontend)> {
        crate::intake::encoder(source, length, cancel)
    }

    /// Admit every convolution, transformer and output-head shape before execution.
    ///
    /// Convolution kernels are rounded once to nearest-even binary16, matching
    /// the admitted native F16 dot operands. Biases and scalar sums stay F32.
    ///
    /// # Errors
    /// Rejects missing/incompatible weights, unsupported geometry, nonfinite
    /// biases or convolution weights that overflow binary16.
    pub fn new(config: Config, mut weights: Weights) -> io::Result<Self> {
        let c = config;
        if !(1..=128).contains(&c.mel_bins)
            || !(1..=1024).contains(&c.channels)
            || !(4..=8192).contains(&c.hidden)
            || !c.hidden.is_multiple_of(2)
            || c.heads == 0
            || c.heads > 128
            || !c.hidden.is_multiple_of(c.heads)
            || !(1..=65_536).contains(&c.intermediate)
            || !(1..=8192).contains(&c.output)
            || !(1..=256).contains(&weights.blocks.len())
        {
            return Err(invalid());
        }
        let layout = EncoderLayout::new(1, c.window)?;
        if layout.rows_per_chunk > c.max_positions {
            return Err(invalid());
        }
        for (layer, convolution) in weights.convolutions.iter_mut().enumerate() {
            convolution.validate(9 * if layer == 0 { 1 } else { c.channels }, c.channels)?;
            convolution.weights.round_convolution_weights()?;
        }
        if weights.convolution_out.dimensions() != (c.channels * c.mel_bins.div_ceil(8), c.hidden) {
            return Err(invalid());
        }
        for block in &weights.blocks {
            block.attention_norm.validate(c.hidden)?;
            block.ffn_norm.validate(c.hidden)?;
            for linear in [&block.query, &block.key, &block.value, &block.attention_out] {
                linear.validate(c.hidden, c.hidden)?;
            }
            block.ffn_in.validate(c.hidden, c.intermediate)?;
            block.ffn_out.validate(c.intermediate, c.hidden)?;
        }
        weights.output_norm.validate(c.hidden)?;
        weights.output_in.validate(c.hidden, c.hidden)?;
        weights.output_out.validate(c.hidden, c.output)?;
        weights.convolution_out.repack_q4_k_if_aligned();
        for block in &mut weights.blocks {
            block.repack_q4_k_if_aligned();
        }
        weights.output_in.weights.repack_q4_k_if_aligned();
        weights.output_out.weights.repack_q4_k_if_aligned();
        let sinusoids = build_sinusoids(layout.rows_per_chunk, c.hidden)?;
        Ok(Self {
            config,
            weights,
            sinusoids,
            threads: 1,
        })
    }

    /// Compute frontend features and every encoder row for one bounded clip.
    ///
    /// # Errors
    /// Returns frontend validation/cancellation errors or [`Self::encode`] errors.
    pub fn encode_pcm(
        &self,
        frontend: &Frontend,
        pcm: &[f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Encoded> {
        let mel = frontend.compute_with_threads(pcm, self.threads, &mut *cancel)?;
        self.encode(&mel, cancel)
    }

    /// Encode a complete bin-major normalized mel matrix from the Rust frontend.
    ///
    /// # Errors
    /// Rejects wrong/nonfinite input, unsupported geometry, cancellation,
    /// allocation failure or nonfinite intermediate output. No partial rows escape.
    pub fn encode(
        &self,
        mel: &MelSpectrogram,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Encoded> {
        if self.config.mel_bins != crate::frontend::MEL_BINS {
            return Err(invalid());
        }
        self.encode_mel(mel.values(), mel.frames(), cancel)
    }

    /// Encode bin-major features under the already admitted model geometry.
    ///
    /// This low-level entry point also admits miniature mathematical controls;
    /// production callers use [`Self::encode_pcm`] with the matching frontend.
    ///
    /// # Errors
    /// Has the same bounded transactional output contract as [`Self::encode`].
    pub fn encode_mel(
        &self,
        mel: &[f32],
        frames: usize,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Encoded> {
        let c = self.config;
        let workers = self.threads;
        let layout = EncoderLayout::new(frames, c.window)?;
        if mel.len() != c.mel_bins * frames {
            return Err(invalid());
        }
        finite(mel)?;
        check(cancel)?;
        let mut rows = zeros(layout.audio_rows * c.hidden)?;
        let mut conv_scratch = Vec::new();
        let mut prepared = PreparedBatch::default();
        let mut image_a = Vec::new();
        let mut image_b = Vec::new();
        let mut flat_out = Vec::new();
        let mut flat_in = Vec::new();
        let flat_dim = c.channels * c.mel_bins.div_ceil(8);
        if layout.audio_rows > 0 {
            ensure_buffer(&mut flat_in, layout.audio_rows * flat_dim)?;
        }
        let mut row_offset = 0;
        for chunk in 0..layout.chunks {
            check(cancel)?;
            let real = if chunk + 1 == layout.chunks {
                layout.last_chunk_mel
            } else {
                layout.mel_per_chunk
            };
            ensure_buffer(&mut image_a, c.mel_bins * layout.mel_per_chunk)?;
            for frequency in 0..c.mel_bins {
                let start = frequency * frames + chunk * layout.mel_per_chunk;
                let dst_start = frequency * layout.mel_per_chunk;
                image_a[dst_start..dst_start + real].copy_from_slice(&mel[start..start + real]);
                if real < layout.mel_per_chunk {
                    image_a[dst_start + real..dst_start + layout.mel_per_chunk].fill(0.0);
                }
            }
            let (mut width, mut height, mut channels) = (layout.mel_per_chunk, c.mel_bins, 1);
            for convolution in &self.weights.convolutions {
                convolve_into_with_threads(
                    convolution,
                    &image_a,
                    width,
                    height,
                    channels,
                    workers,
                    &mut prepared,
                    &mut flat_out,
                    &mut conv_scratch,
                    &mut image_b,
                    cancel,
                )?;
                std::mem::swap(&mut image_a, &mut image_b);
                width = width.div_ceil(2);
                height = height.div_ceil(2);
                channels = c.channels;
            }
            let retained = if chunk + 1 == layout.chunks {
                layout.last_chunk_rows
            } else {
                layout.rows_per_chunk
            };
            if retained > 0 {
                let chunk_flat_dim = channels * height;
                if chunk_flat_dim != flat_dim {
                    return Err(invalid());
                }
                let chunk_flat =
                    &mut flat_in[row_offset * flat_dim..(row_offset + retained) * flat_dim];
                for time in 0..retained {
                    check(cancel)?;
                    let dst = &mut chunk_flat[time * flat_dim..(time + 1) * flat_dim];
                    for cf in 0..flat_dim {
                        dst[cf] = image_a[cf * width + time];
                    }
                }
                row_offset += retained;
            }
        }
        if layout.audio_rows > 0 {
            prepared.prepare_meta(layout.audio_rows, flat_dim);
            self.weights.convolution_out.multiply_batch_prepared_into(
                &mut prepared,
                &flat_in[..layout.audio_rows * flat_dim],
                workers,
                &mut conv_scratch,
                &mut rows,
                cancel,
            )?;
            add_chunk_sinusoids_parallel(
                &mut rows,
                &self.sinusoids,
                layout.rows_per_chunk,
                c.hidden,
                workers,
            )?;
        }

        let num_rows = layout.audio_rows;
        let window_rows = (8 * layout.rows_per_chunk).max(1);
        let mut norm_buf = zeros(num_rows * c.hidden)?;
        let mut qkv = zeros(3 * num_rows * c.hidden)?;
        let mut attn_out = zeros(num_rows * c.hidden)?;
        let mut proj = zeros(num_rows * c.hidden)?;
        let mut ffn_buf = zeros(num_rows * c.intermediate)?;
        for block in &self.weights.blocks {
            transform_block(
                block,
                &mut rows,
                &mut norm_buf,
                &mut qkv,
                &mut attn_out,
                &mut proj,
                &mut ffn_buf,
                &mut prepared,
                &mut conv_scratch,
                num_rows,
                window_rows,
                c,
                workers,
                cancel,
            )?;
        }
        self.weights
            .output_norm
            .apply_batch(&rows, &mut norm_buf, c.hidden, workers)?;
        prepared.prepare_meta(num_rows, c.hidden);
        self.weights.output_in.apply_batch_prepared(
            &mut prepared,
            &norm_buf,
            workers,
            &mut conv_scratch,
            &mut proj,
            cancel,
        )?;
        gelu_parallel(&mut proj, workers, c.hidden)?;
        let mut values = zeros(num_rows * c.output)?;
        prepared.prepare_meta(num_rows, c.hidden);
        self.weights.output_out.apply_batch_prepared(
            &mut prepared,
            &proj,
            workers,
            &mut conv_scratch,
            &mut values,
            cancel,
        )?;
        Ok(Encoded {
            rows: num_rows,
            width: c.output,
            values,
        })
    }
}

/// Complete time-major audio embeddings for decoder placeholder replacement.
pub struct Encoded {
    rows: usize,
    width: usize,
    values: Vec<f32>,
}
impl Encoded {
    /// Number of real encoder time rows, after independent chunk-tail trimming.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Number of features in each decoder audio row.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }
    /// Borrow `[time, feature]` values without transferring mutable ownership.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

fn round_chunk_f16(src: &[f32], dst: &mut [f32]) -> io::Result<()> {
    let mut f16_buf = [0_u16; 1024];
    for (s, d) in src.chunks(1024).zip(dst.chunks_mut(1024)) {
        let f16_slice = &mut f16_buf[..s.len()];
        if !packed::narrow_f32_to_f16_slice(s, f16_slice) {
            return Err(invalid());
        }
        packed::widen_f16_slice(f16_slice, d);
    }
    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn add_scalar_bias_simd<S: Simd>(simd: S, values: &mut [f32], bias: f32) -> bool {
    let b = f32x4::splat(simd, bias);
    let inf = f32x4::splat(simd, f32::INFINITY);
    let mut all_finite = true;
    let (chunks, tail) = values.as_chunks_mut::<4>();
    for chunk in chunks {
        let sum = f32x4::from_slice(simd, chunk) + b;
        if !sum.abs().simd_lt(inf).all_true() {
            all_finite = false;
        }
        sum.store_slice(chunk);
    }
    for value in tail {
        *value += bias;
        if !value.is_finite() {
            all_finite = false;
        }
    }
    all_finite
}

fn channel_major_bias_and_gelu(
    out: &mut [f32],
    bias: &[f32],
    total_patches: usize,
    workers: usize,
) -> io::Result<()> {
    if total_patches == 0 || bias.is_empty() {
        return finite(out);
    }
    let level = packed::simd_level();
    row_workers::try_run_aligned(out, workers, total_patches, &|elem_off, slice| {
        let first_ch = elem_off / total_patches;
        for (local_ch, ch_slice) in slice.chunks_exact_mut(total_patches).enumerate() {
            let b = bias[first_ch + local_ch];
            if !dispatch!(level, simd => add_scalar_bias_simd(simd, ch_slice, b)) {
                return Err(invalid());
            }
            numerics::gelu_slice_in_place(ch_slice);
            finite(ch_slice)?;
        }
        Ok(())
    })
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn add_chunk_sinusoids_simd<S: Simd>(
    simd: S,
    slice: &mut [f32],
    sinusoids: &[f32],
    first_row: usize,
    rows_per_chunk: usize,
    hidden: usize,
) -> bool {
    let inf = f32x4::splat(simd, f32::INFINITY);
    let mut all_finite = true;
    for (local_r, row) in slice.chunks_exact_mut(hidden).enumerate() {
        let pos_in_chunk = (first_row + local_r) % rows_per_chunk;
        let sin_row = &sinusoids[pos_in_chunk * hidden..(pos_in_chunk + 1) * hidden];
        let (r_chunks, r_tail) = row.as_chunks_mut::<4>();
        let (s_chunks, s_tail) = sin_row.as_chunks::<4>();
        for (rc, sc) in r_chunks.iter_mut().zip(s_chunks) {
            let sum = f32x4::from_slice(simd, rc) + f32x4::from_slice(simd, sc);
            if !sum.abs().simd_lt(inf).all_true() {
                all_finite = false;
            }
            sum.store_slice(rc);
        }
        for (dst, &sin_val) in r_tail.iter_mut().zip(s_tail) {
            *dst += sin_val;
            if !dst.is_finite() {
                all_finite = false;
            }
        }
    }
    all_finite
}

fn add_chunk_sinusoids_parallel(
    rows: &mut [f32],
    sinusoids: &[f32],
    rows_per_chunk: usize,
    hidden: usize,
    workers: usize,
) -> io::Result<()> {
    if rows.is_empty() || rows_per_chunk == 0 {
        return Ok(());
    }
    let level = packed::simd_level();
    row_workers::try_run_aligned(rows, workers, hidden, &|elem_off, slice| {
        let first_row = elem_off / hidden;
        let ok = dispatch!(
            level,
            simd => add_chunk_sinusoids_simd(simd, slice, sinusoids, first_row, rows_per_chunk, hidden)
        );
        if ok { Ok(()) } else { Err(invalid()) }
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Direct-to-quads Conv2d im2col writes 4-position quads without an intermediate patch buffer."
)]
fn im2col_quads_slice(
    rounded: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    out_width: usize,
    total_patches: usize,
    first_quad: usize,
    dst_quads: &mut [[f32; 4]],
) {
    let patch_width = 9 * channels;
    for (local_q, quad_slice) in dst_quads.chunks_exact_mut(patch_width).enumerate() {
        let q = first_quad + local_q;
        let base_pos = q * 4;
        if base_pos + 3 < total_patches {
            let y0 = base_pos / out_width;
            let x0 = base_pos % out_width;
            if x0 + 3 < out_width && x0 > 0 && 2 * (x0 + 3) + 1 < width {
                let py_opts: [Option<usize>; 3] =
                    std::array::from_fn(|ky| (2 * y0 + ky).checked_sub(1).filter(|&py| py < height));
                for ch in 0..channels {
                    let ch_in_base = ch * height;
                    let ch_p_base = ch * 9;
                    for (ky, py_opt) in py_opts.iter().enumerate() {
                        if let Some(py) = *py_opt {
                            let in_row =
                                &rounded[(ch_in_base + py) * width..(ch_in_base + py + 1) * width];
                            for kx in 0..3 {
                                let bx = 2 * x0 + kx - 1;
                                quad_slice[ch_p_base + ky * 3 + kx] = [
                                    in_row[bx],
                                    in_row[bx + 2],
                                    in_row[bx + 4],
                                    in_row[bx + 6],
                                ];
                            }
                        } else {
                            quad_slice[ch_p_base + ky * 3..ch_p_base + ky * 3 + 3].fill([0.0; 4]);
                        }
                    }
                }
                continue;
            }
        }
        quad_slice.fill([0.0; 4]);
        for lane in 0..4 {
            let pos = base_pos + lane;
            if pos >= total_patches {
                break;
            }
            let y = pos / out_width;
            let x = pos % out_width;
            let py_opts: [Option<usize>; 3] =
                std::array::from_fn(|ky| (2 * y + ky).checked_sub(1).filter(|&py| py < height));
            let px_opts: [Option<usize>; 3] =
                std::array::from_fn(|kx| (2 * x + kx).checked_sub(1).filter(|&px| px < width));
            for ch in 0..channels {
                let ch_in_base = ch * height;
                let ch_p_base = ch * 9;
                for (ky, py_opt) in py_opts.iter().enumerate() {
                    if let Some(py) = *py_opt {
                        let in_row =
                            &rounded[(ch_in_base + py) * width..(ch_in_base + py + 1) * width];
                        for (kx, px_opt) in px_opts.iter().enumerate() {
                            if let Some(px) = *px_opt {
                                quad_slice[ch_p_base + ky * 3 + kx][lane] = in_row[px];
                            }
                        }
                    }
                }
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Reuses batch im2col, rounded input, and channel-major buffers across chunked encoder stages."
)]
fn convolve_into_with_threads(
    linear: &Linear,
    input: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    workers: usize,
    prepared: &mut PreparedBatch,
    rounded_input: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
    output: &mut Vec<f32>,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<()> {
    let (out_width, out_height) = (width.div_ceil(2), height.div_ceil(2));
    let out_channels = linear.bias.len();
    let total_patches = out_width * out_height;
    let patch_width = 9 * channels;
    for _ in 0..total_patches {
        check(cancel)?;
    }
    let input_len = channels
        .checked_mul(height)
        .and_then(|v| v.checked_mul(width))
        .ok_or_else(invalid)?;
    let input_view = input.get(..input_len).ok_or_else(invalid)?;
    if total_patches > 0 && patch_width > 0 {
        ensure_buffer(rounded_input, input_len)?;
        let rounded_dst = &mut rounded_input[..input_len];
        if workers > 1 && input_len >= 4_096 {
            row_workers::try_run_aligned(rounded_dst, workers, 1024, &|elem_off, dst_slice| {
                round_chunk_f16(&input_view[elem_off..elem_off + dst_slice.len()], dst_slice)
            })?;
        } else {
            round_chunk_f16(input_view, rounded_dst)?;
        }
    }

    ensure_buffer(output, out_channels * total_patches)?;
    if linear.weights.is_float() {
        let rounded = &rounded_input[..input_len];
        prepared.prepare_quads_direct(total_patches, patch_width, workers, &|first_quad, dst| {
            im2col_quads_slice(
                rounded,
                width,
                height,
                channels,
                out_width,
                total_patches,
                first_quad,
                dst,
            );
        });
        linear.weights.multiply_batch_channel_major_prepacked_into(
            prepared,
            workers,
            scratch,
            &mut output[..out_channels * total_patches],
            cancel,
        )?;
    } else {
        let mut patches = Vec::new();
        ensure_buffer(&mut patches, total_patches * patch_width)?;
        if total_patches > 0 && patch_width > 0 {
            let rounded = &rounded_input[..input_len];
            let row_stride = out_width * patch_width;
            row_workers::run_aligned(&mut patches, workers, row_stride, &|elem_off, slice| {
                let first_y = elem_off / row_stride;
                for (local_y, y_slice) in slice.chunks_exact_mut(row_stride).enumerate() {
                    let y = first_y + local_y;
                    let py_opts: [Option<usize>; 3] = std::array::from_fn(|ky| {
                        (2 * y + ky).checked_sub(1).filter(|&py| py < height)
                    });
                    for (x, patch) in y_slice.chunks_exact_mut(patch_width).enumerate() {
                        let interior_x = x > 0 && 2 * x + 1 < width;
                        for channel in 0..channels {
                            let ch_in_base = channel * height;
                            let ch_p_base = channel * 9;
                            for (ky, py_opt) in py_opts.iter().enumerate() {
                                let row_dst =
                                    &mut patch[ch_p_base + ky * 3..ch_p_base + (ky + 1) * 3];
                                if let Some(py) = *py_opt {
                                    let in_row = &rounded
                                        [(ch_in_base + py) * width..(ch_in_base + py + 1) * width];
                                    if interior_x {
                                        row_dst.copy_from_slice(&in_row[2 * x - 1..2 * x + 2]);
                                    } else {
                                        for (kx, dst) in row_dst.iter_mut().enumerate() {
                                            *dst = (2 * x + kx)
                                                .checked_sub(1)
                                                .filter(|&px| px < width)
                                                .map_or(0.0, |px| in_row[px]);
                                        }
                                    }
                                } else {
                                    row_dst.fill(0.0);
                                }
                            }
                        }
                    }
                }
            });
        }
        prepared.prepare_meta(total_patches, patch_width);
        linear.weights.multiply_batch_channel_major_prepared_into(
            prepared,
            &patches[..total_patches * patch_width],
            workers,
            scratch,
            &mut output[..out_channels * total_patches],
            cancel,
        )?;
    }
    channel_major_bias_and_gelu(
        &mut output[..out_channels * total_patches],
        &linear.bias,
        total_patches,
        workers,
    )
}

#[cfg(test)]
fn convolve(
    linear: &Linear,
    input: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Vec<f32>> {
    convolve_with_threads(linear, input, width, height, channels, 1, cancel)
}

#[cfg(test)]
fn convolve_with_threads(
    linear: &Linear,
    input: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    workers: usize,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Vec<f32>> {
    let mut prepared = PreparedBatch::default();
    let mut flat_out = Vec::new();
    let mut scratch = Vec::new();
    let mut output = Vec::new();
    convolve_into_with_threads(
        linear,
        input,
        width,
        height,
        channels,
        workers,
        &mut prepared,
        &mut flat_out,
        &mut scratch,
        &mut output,
        cancel,
    )?;
    Ok(output)
}

fn gelu_parallel(values: &mut [f32], workers: usize, alignment: usize) -> io::Result<()> {
    row_workers::try_run_aligned(values, workers, alignment, &|_, slice| {
        numerics::gelu_slice_in_place(slice);
        finite(slice)
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

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "The pinned sinusoid constructs f64 angles and explicitly rounds sin/cos to f32; bounded dimensions convert exactly."
)]
fn build_sinusoids(rows: usize, hidden: usize) -> io::Result<Vec<f32>> {
    let half = hidden / 2;
    let logarithm = 10_000.0_f64.ln() / (half - 1) as f64;
    let mut table = zeros(rows * hidden)?;
    for position in 0..rows {
        let row = &mut table[position * hidden..(position + 1) * hidden];
        for index in 0..half {
            let angle = position as f64 * (-logarithm * index as f64).exp();
            row[index] = angle.sin() as f32;
            row[index + half] = angle.cos() as f32;
        }
    }
    finite(&table)?;
    Ok(table)
}

#[cfg(test)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "The pinned sinusoid constructs f64 angles and explicitly rounds sin/cos to f32; bounded dimensions convert exactly."
)]
fn position_add(row: &mut [f32], position: usize) -> io::Result<()> {
    let half = row.len() / 2;
    let logarithm = 10_000.0_f64.ln() / (half - 1) as f64;
    for index in 0..half {
        let angle = position as f64 * (-logarithm * index as f64).exp();
        row[index] += angle.sin() as f32;
        row[index + half] += angle.cos() as f32;
    }
    finite(row)
}

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

#[expect(
    clippy::too_many_arguments,
    reason = "Reuses flat encoder transformer buffers across layers with zero per-layer allocations."
)]
fn transform_block(
    block: &Block,
    rows: &mut [f32],
    norm_buf: &mut [f32],
    qkv: &mut [f32],
    attn_out: &mut [f32],
    proj: &mut [f32],
    ffn_buf: &mut [f32],
    prepared: &mut PreparedBatch,
    scratch: &mut Vec<f32>,
    num_rows: usize,
    window_rows: usize,
    c: Config,
    workers: usize,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<()> {
    check(cancel)?;
    block
        .attention_norm
        .apply_batch(rows, norm_buf, c.hidden, workers)?;
    let part = num_rows * c.hidden;
    let (q, kv_mut) = qkv.split_at_mut(part);
    let (k, v) = kv_mut.split_at_mut(part);
    prepared.prepare_meta(num_rows, c.hidden);
    Projection::multiply_batch_triple_into(
        &block.query.weights,
        &block.key.weights,
        &block.value.weights,
        prepared,
        norm_buf,
        workers,
        scratch,
        q,
        k,
        v,
        cancel,
    )?;
    add_batch_bias_triple(
        qkv,
        &block.query.bias,
        &block.key.bias,
        &block.value.bias,
        part,
        workers,
    )?;

    for _ in 0..(num_rows * c.heads) {
        check(cancel)?;
    }
    let (q_ref, kv_ref) = qkv.split_at(part);
    let (k_ref, v_ref) = kv_ref.split_at(part);
    let head_dim = c.hidden / c.heads;
    let divisor = scalar(head_dim).sqrt();
    let window_rows = window_rows.max(1);
    let score_needed = 4 * window_rows.min(num_rows).max(1);
    row_workers::try_run_aligned(attn_out, workers, c.hidden, &|elem_off, slice| {
        let start_row = elem_off / c.hidden;
        let rows_here = slice.len() / c.hidden;
        with_score_scratch(score_needed, |scores| {
            let mut local_idx = 0_usize;
            while local_idx < rows_here {
                let row_idx = start_row + local_idx;
                let win_start = (row_idx / window_rows) * window_rows;
                let win_end = (win_start + window_rows).min(num_rows);
                let win_len = win_end - win_start;
                let tile_rows = (win_end - row_idx).min(rows_here - local_idx).min(4);
                let q_tile = &q_ref[row_idx * c.hidden..(row_idx + tile_rows) * c.hidden];
                let k_win = &k_ref[win_start * c.hidden..win_end * c.hidden];
                let v_win = &v_ref[win_start * c.hidden..win_end * c.hidden];
                let out_tile = &mut slice[local_idx * c.hidden..(local_idx + tile_rows) * c.hidden];
                packed::attend_q_rows_bidirectional_f32(
                    q_tile,
                    k_win,
                    v_win,
                    win_len,
                    c.hidden,
                    c.heads,
                    head_dim,
                    divisor,
                    &mut scores[..tile_rows * win_len],
                    out_tile,
                    invalid,
                )?;
                finite(out_tile)?;
                local_idx += tile_rows;
            }
            Ok(())
        })
    })?;

    prepared.prepare_meta(num_rows, c.hidden);
    block
        .attention_out
        .apply_batch_prepared(prepared, attn_out, workers, scratch, proj, cancel)?;
    add_residual_batch(rows, proj, workers, c.hidden)?;

    block
        .ffn_norm
        .apply_batch(rows, norm_buf, c.hidden, workers)?;
    prepared.prepare_meta(num_rows, c.hidden);
    block
        .ffn_in
        .apply_batch_prepared(prepared, norm_buf, workers, scratch, ffn_buf, cancel)?;
    gelu_parallel(ffn_buf, workers, c.intermediate)?;
    prepared.prepare_meta(num_rows, c.intermediate);
    block
        .ffn_out
        .apply_batch_prepared(prepared, ffn_buf, workers, scratch, proj, cancel)?;
    add_residual_batch(rows, proj, workers, c.hidden)
}

#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "Shares SIMD bidirectional score/value accumulation across transformer blocks and tests."
)]
fn attention_head_bidirectional(
    q_head: &[f32],
    keys: &[f32],
    values: &[f32],
    num_rows: usize,
    row_width: usize,
    offset: usize,
    head_dim: usize,
    divisor: f32,
    scores: &mut [f32],
    out_head: &mut [f32],
) -> io::Result<()> {
    packed::attend_head(
        q_head, keys, values, num_rows, row_width, offset, head_dim, divisor, None, scores,
        out_head, invalid,
    )
}

#[cfg(test)]
fn attention(
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    heads: usize,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Vec<f32>> {
    let row_width = query.len();
    let head_dim = row_width / heads;
    let num_rows = keys.len();
    let divisor = scalar(head_dim).sqrt();
    let mut flat_keys = zeros(num_rows * row_width)?;
    let mut flat_values = zeros(num_rows * row_width)?;
    for (r, (k_row, v_row)) in keys.iter().zip(values).enumerate() {
        flat_keys[r * row_width..(r + 1) * row_width].copy_from_slice(k_row);
        flat_values[r * row_width..(r + 1) * row_width].copy_from_slice(v_row);
    }
    let mut output = zeros(row_width)?;
    let mut scores = zeros(num_rows)?;
    for head in 0..heads {
        check(cancel)?;
        let offset = head * head_dim;
        attention_head_bidirectional(
            &query[offset..offset + head_dim],
            &flat_keys,
            &flat_values,
            num_rows,
            row_width,
            offset,
            head_dim,
            divisor,
            &mut scores,
            &mut output[offset..offset + head_dim],
        )?;
    }
    finite(&output)?;
    Ok(output)
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn add_bias_rows_simd<S: Simd>(simd: S, rows: &mut [f32], bias: &[f32]) -> bool {
    let out_width = bias.len();
    let inf = f32x4::splat(simd, f32::INFINITY);
    let (b_chunks, b_tail) = bias.as_chunks::<4>();
    let mut all_finite = true;
    for row in rows.chunks_exact_mut(out_width) {
        let (r_chunks, r_tail) = row.as_chunks_mut::<4>();
        for (r_chunk, b_chunk) in r_chunks.iter_mut().zip(b_chunks) {
            let sum = f32x4::from_slice(simd, r_chunk) + f32x4::from_slice(simd, b_chunk);
            if !sum.abs().simd_lt(inf).all_true() {
                all_finite = false;
            }
            sum.store_slice(r_chunk);
        }
        for (value, &b) in r_tail.iter_mut().zip(b_tail) {
            *value += b;
            if !value.is_finite() {
                all_finite = false;
            }
        }
    }
    all_finite
}

fn add_batch_bias(out: &mut [f32], bias: &[f32], workers: usize) -> io::Result<()> {
    let out_width = bias.len();
    if out_width == 0 {
        return finite(out);
    }
    if !out.len().is_multiple_of(out_width) {
        return Err(invalid());
    }
    let level = packed::simd_level();
    row_workers::try_run_aligned(out, workers, out_width, &|_, slice| {
        let ok = dispatch!(level, simd => add_bias_rows_simd(simd, slice, bias));
        if ok { Ok(()) } else { Err(invalid()) }
    })
}

fn add_batch_bias_triple(
    qkv: &mut [f32],
    q_bias: &[f32],
    k_bias: &[f32],
    v_bias: &[f32],
    part: usize,
    workers: usize,
) -> io::Result<()> {
    let width = q_bias.len();
    if k_bias.len() != width || v_bias.len() != width {
        return Err(invalid());
    }
    if width == 0 {
        return finite(qkv);
    }
    if !part.is_multiple_of(width) || qkv.len() != part.checked_mul(3).ok_or_else(invalid)? {
        return Err(invalid());
    }
    let level = packed::simd_level();
    let biases = [q_bias, k_bias, v_bias];
    row_workers::try_run_aligned(qkv, workers, width, &|elem_off, slice| {
        let mut cursor = elem_off;
        let mut local_off = 0_usize;
        while local_off < slice.len() {
            let section = cursor / part;
            let section_end = (section + 1) * part;
            let take = (section_end - cursor).min(slice.len() - local_off);
            let subslice = &mut slice[local_off..local_off + take];
            let ok = dispatch!(level, simd => add_bias_rows_simd(simd, subslice, biases[section]));
            if !ok {
                return Err(invalid());
            }
            cursor += take;
            local_off += take;
        }
        Ok(())
    })
}

#[cfg(test)]
#[path = "encoder/tests.rs"]
mod tests;

#[cfg(test)]
mod wave5_tests {
    use super::*;

    fn scalar_conv2d_reference(
        weights: &[f32],
        bias: &[f32],
        input: &[f32],
        width: usize,
        height: usize,
        channels: usize,
    ) -> io::Result<Vec<f32>> {
        let out_width = width.div_ceil(2);
        let out_height = height.div_ceil(2);
        let out_channels = bias.len();
        let total_patches = out_width * out_height;
        let patch_width = 9 * channels;
        let mut out = vec![0.0_f32; out_channels * total_patches];
        for y in 0..out_height {
            for x in 0..out_width {
                let mut patch = vec![0.0_f32; patch_width];
                for ch in 0..channels {
                    for ky in 0..3 {
                        let py_opt = (2 * y + ky).checked_sub(1).filter(|&py| py < height);
                        for kx in 0..3 {
                            let idx = ch * 9 + ky * 3 + kx;
                            if let Some(py) = py_opt
                                && let Some(px) =
                                    (2 * x + kx).checked_sub(1).filter(|&px| px < width)
                            {
                                patch[idx] =
                                    binary16_operand(input[(ch * height + py) * width + px])?;
                            }
                        }
                    }
                }
                let pos = y * out_width + x;
                for out_ch in 0..out_channels {
                    let row = &weights[out_ch * patch_width..(out_ch + 1) * patch_width];
                    let mut sum = 0.0_f32;
                    for (&w, &p) in row.iter().zip(&patch) {
                        sum += w * p;
                    }
                    out[out_ch * total_patches + pos] = numerics::gelu(sum + bias[out_ch]);
                }
            }
        }
        Ok(out)
    }

    #[test]
    fn convolve_channel_major_f16_stem_matches_scalar_bit_exact() -> io::Result<()> {
        for (width, height, channels, out_channels) in [
            (1_usize, 1_usize, 1_usize, 3_usize),
            (3, 2, 1, 5),
            (7, 5, 4, 9),
            (17, 13, 8, 13),
            (70, 64, 1, 6),
        ] {
            let patch_width = 9 * channels;
            let weights: Vec<f32> = (0..out_channels * patch_width)
                .map(|i| {
                    if i % 13 == 0 {
                        -0.0
                    } else {
                        (f32::from(u16::try_from((i * 7) % 41).unwrap_or(0)) - 20.0) * 0.03125
                    }
                })
                .collect();
            let bias: Vec<f32> = (0..out_channels)
                .map(|i| {
                    if i % 5 == 0 {
                        -0.0
                    } else {
                        (f32::from(u16::try_from((i * 11) % 29).unwrap_or(0)) - 14.0) * 0.0625
                    }
                })
                .collect();
            let input: Vec<f32> = (0..channels * height * width)
                .map(|i| {
                    if i % 17 == 0 {
                        -0.0
                    } else {
                        (f32::from(u16::try_from((i * 13) % 97).unwrap_or(0)) - 48.0) * 0.025
                    }
                })
                .collect();
            let linear = Linear {
                weights: Projection::f32(patch_width, out_channels, weights.clone())?,
                bias: bias.clone(),
            };
            let expected =
                scalar_conv2d_reference(&weights, &bias, &input, width, height, channels)?;
            let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();

            for workers in [1_usize, 3] {
                let actual = convolve_with_threads(
                    &linear,
                    &input,
                    width,
                    height,
                    channels,
                    workers,
                    &mut || false,
                )?;
                let actual_bits: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
                assert_eq!(
                    actual_bits, expected_bits,
                    "conv2d mismatch w={width} h={height} c={channels} oc={out_channels} workers={workers}"
                );
            }
        }

        // Fail-closed on binary16 overflow (> 65504.0) and non-finite pixels, including parallel path.
        let linear = Linear {
            weights: Projection::f32(9, 4, vec![0.25_f32; 36])?,
            bias: vec![0.0_f32; 4],
        };
        let mut overflow_in = vec![0.5_f32; 70 * 64];
        overflow_in[70 * 64 - 1] = 65_520.0;
        assert!(convolve_with_threads(&linear, &overflow_in, 70, 64, 1, 3, &mut || false).is_err());
        let mut nan_in = vec![0.5_f32; 70 * 64];
        nan_in[0] = f32::NAN;
        assert!(convolve_with_threads(&linear, &nan_in, 70, 64, 1, 3, &mut || false).is_err());
        Ok(())
    }
}
