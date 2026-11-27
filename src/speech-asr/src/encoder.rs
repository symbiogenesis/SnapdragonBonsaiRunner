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
    packed::{self, PreparedBatch, WeightSlice},
    protocol::EncoderLayout,
    row_workers, threads,
};
#[cfg(windows)]
use bonsai_windows_process::conv_kernels;
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
            .multiply_batch_prepared_biased_prevalidated_into(
                prepared,
                raw_input,
                Some(&self.bias),
                workers,
                scratch,
                output,
                cancel,
            )
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

struct PackedConvWeights {
    out_channels: usize,
    patch_width: usize,
    blocks16: Vec<f32>,
}

impl PackedConvWeights {
    fn from_projection(proj: &Projection) -> io::Result<Self> {
        let (patch_width, out_channels) = proj.dimensions();
        let WeightSlice::Float(weights) = proj.as_weight_matrix().data else {
            return Err(invalid());
        };
        if weights.len() != out_channels.checked_mul(patch_width).ok_or_else(invalid)? {
            return Err(invalid());
        }
        let oc_blocks = out_channels.div_ceil(16);
        let packed_len = oc_blocks
            .checked_mul(patch_width)
            .and_then(|v| v.checked_mul(16))
            .ok_or_else(invalid)?;
        let mut blocks16 = zeros(packed_len)?;
        let block_stride = patch_width * 16;
        for blk in 0..oc_blocks {
            let oc_base = blk * 16;
            let lane_w = 16.min(out_channels - oc_base);
            let dst_blk = &mut blocks16[blk * block_stride..(blk + 1) * block_stride];
            for (k, dst_16) in dst_blk.chunks_exact_mut(16).enumerate() {
                for (lane, slot) in dst_16[..lane_w].iter_mut().enumerate() {
                    *slot = weights[(oc_base + lane) * patch_width + k];
                }
            }
        }
        Ok(Self {
            out_channels,
            patch_width,
            blocks16,
        })
    }

    fn block(&self, oc_blk: usize) -> &[f32] {
        let stride = self.patch_width * 16;
        &self.blocks16[oc_blk * stride..(oc_blk + 1) * stride]
    }
}

/// One admitted immutable CNN/transformer encoder.
pub struct Encoder {
    config: Config,
    weights: Weights,
    packed_convolutions: [PackedConvWeights; 3],
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
        let packed_convolutions = [
            PackedConvWeights::from_projection(&weights.convolutions[0].weights)?,
            PackedConvWeights::from_projection(&weights.convolutions[1].weights)?,
            PackedConvWeights::from_projection(&weights.convolutions[2].weights)?,
        ];
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
            packed_convolutions,
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
        let per_chunk_conv_work =
            (c.channels * c.mel_bins.div_ceil(2) * layout.mel_per_chunk.div_ceil(2))
                .max(9 * c.channels * c.mel_bins.div_ceil(4) * layout.mel_per_chunk.div_ceil(4));
        let chunk_batch = (28_000_000 / per_chunk_conv_work.max(1))
            .clamp(1, 8)
            .min(layout.chunks);
        for chunk_start in (0..layout.chunks).step_by(chunk_batch) {
            check(cancel)?;
            let g_chunks = (layout.chunks - chunk_start).min(chunk_batch);
            ensure_buffer(&mut image_a, g_chunks * c.mel_bins * layout.mel_per_chunk)?;
            for g in 0..g_chunks {
                let chunk = chunk_start + g;
                let real = if chunk + 1 == layout.chunks {
                    layout.last_chunk_mel
                } else {
                    layout.mel_per_chunk
                };
                let chunk_base = g * c.mel_bins * layout.mel_per_chunk;
                for frequency in 0..c.mel_bins {
                    let start = frequency * frames + chunk * layout.mel_per_chunk;
                    let dst_start = chunk_base + frequency * layout.mel_per_chunk;
                    image_a[dst_start..dst_start + real].copy_from_slice(&mel[start..start + real]);
                    if real < layout.mel_per_chunk {
                        image_a[dst_start + real..dst_start + layout.mel_per_chunk].fill(0.0);
                    }
                }
            }
            let (mut width, mut height, mut channels) = (layout.mel_per_chunk, c.mel_bins, 1);
            for (convolution, packed_conv) in self
                .weights
                .convolutions
                .iter()
                .zip(&self.packed_convolutions)
            {
                convolve_batch_into_with_threads(
                    convolution,
                    Some(packed_conv),
                    &image_a,
                    g_chunks,
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
            let chunk_flat_dim = channels * height;
            if chunk_flat_dim != flat_dim {
                return Err(invalid());
            }
            for g in 0..g_chunks {
                let chunk = chunk_start + g;
                let retained = if chunk + 1 == layout.chunks {
                    layout.last_chunk_rows
                } else {
                    layout.rows_per_chunk
                };
                if retained > 0 {
                    let chunk_flat =
                        &mut flat_in[row_offset * flat_dim..(row_offset + retained) * flat_dim];
                    for time in 0..retained {
                        check(cancel)?;
                        let dst = &mut chunk_flat[time * flat_dim..(time + 1) * flat_dim];
                        for (cf, slot) in dst.iter_mut().enumerate() {
                            let ch = cf / height;
                            let freq = cf % height;
                            *slot = image_a[((ch * g_chunks + g) * height + freq) * width + time];
                        }
                    }
                    row_offset += retained;
                }
            }
        }
        if layout.audio_rows > 0 {
            prepared.prepare_meta(layout.audio_rows, flat_dim);
            self.weights
                .convolution_out
                .multiply_batch_prepared_biased_prevalidated_into(
                    &mut prepared,
                    &flat_in[..layout.audio_rows * flat_dim],
                    None,
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

const CONV_TILE_PATCHES: usize = 360;
const CONV_SUB_FRAMES: usize = 60;

#[inline]
fn gemm_16oc_strided_dispatch(
    w_ic_16: &[f32],
    x_base: &[f32],
    x_frame_stride: usize,
    in_c: usize,
    frames: usize,
    acc_tile: &mut [[f32; 16]],
) {
    #[cfg(windows)]
    {
        conv_kernels::gemm_f32_fma_16oc_strided(
            w_ic_16,
            x_base,
            x_frame_stride,
            in_c,
            frames,
            acc_tile,
        );
    }
    #[cfg(not(windows))]
    {
        let (w_steps, _) = w_ic_16[..in_c * 16].as_chunks::<16>();
        for (f, acc_row) in acc_tile[..frames].iter_mut().enumerate() {
            let x_row = &x_base[f * x_frame_stride..f * x_frame_stride + in_c];
            for (ic, w_16) in w_steps.iter().enumerate() {
                let x_val = x_row[ic];
                for (oc, &w_val) in w_16.iter().enumerate() {
                    acc_row[oc] = x_val.mul_add(w_val, acc_row[oc]);
                }
            }
        }
    }
}

#[inline]
fn fill_im2col_edge_x_ch(rows: [Option<&[f32]>; 3], width: usize, x: usize, dst: &mut [f32]) {
    let px_opts: [Option<usize>; 3] =
        std::array::from_fn(|kx| (2 * x + kx).checked_sub(1).filter(|&px| px < width));
    for (ky, row_opt) in rows.iter().enumerate() {
        let row_dst = &mut dst[ky * 3..(ky + 1) * 3];
        if let Some(in_row) = *row_opt {
            for (kx, px_opt) in px_opts.iter().enumerate() {
                row_dst[kx] = px_opt.map_or(0.0, |px| in_row[px]);
            }
        } else {
            row_dst.fill(0.0);
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Row-oriented stride-2 im2col helper shares channel-plane geometry across one output row segment."
)]
fn fill_im2col_row_segment(
    rounded: &[f32],
    chunk_count: usize,
    width: usize,
    height: usize,
    channels: usize,
    chunk_local: usize,
    y: usize,
    x_begin: usize,
    x_end: usize,
    seg_dst: &mut [f32],
) {
    let hw = height * width;
    let patch_width = 9 * channels;
    let py_opts: [Option<usize>; 3] =
        std::array::from_fn(|ky| (2 * y + ky).checked_sub(1).filter(|&py| py < height));
    let has_left = x_begin == 0 && x_end > 0;
    let int_start = x_begin.max(1);
    let int_end = x_end.min(width / 2).max(int_start);

    for ch in 0..channels {
        let ch_in_base = (ch * chunk_count + chunk_local) * hw;
        let ch_p_base = ch * 9;
        let rows: [Option<&[f32]>; 3] = py_opts.map(|py_opt| {
            py_opt.map(|py| &rounded[ch_in_base + py * width..ch_in_base + (py + 1) * width])
        });

        if has_left {
            fill_im2col_edge_x_ch(rows, width, 0, &mut seg_dst[ch_p_base..ch_p_base + 9]);
        }

        if int_start < int_end {
            if let [Some(r0), Some(r1), Some(r2)] = rows {
                for x in int_start..int_end {
                    let off = (x - x_begin) * patch_width + ch_p_base;
                    let px0 = 2 * x - 1;
                    let dst = &mut seg_dst[off..off + 9];
                    dst[0..3].copy_from_slice(&r0[px0..px0 + 3]);
                    dst[3..6].copy_from_slice(&r1[px0..px0 + 3]);
                    dst[6..9].copy_from_slice(&r2[px0..px0 + 3]);
                }
            } else {
                for x in int_start..int_end {
                    let off = (x - x_begin) * patch_width + ch_p_base;
                    let px0 = 2 * x - 1;
                    let dst = &mut seg_dst[off..off + 9];
                    for (ky, row_opt) in rows.iter().enumerate() {
                        let row_dst = &mut dst[ky * 3..(ky + 1) * 3];
                        if let Some(in_row) = *row_opt {
                            row_dst.copy_from_slice(&in_row[px0..px0 + 3]);
                        } else {
                            row_dst.fill(0.0);
                        }
                    }
                }
            }
        }

        for x in int_end..x_end {
            let off = (x - x_begin) * patch_width + ch_p_base;
            fill_im2col_edge_x_ch(rows, width, x, &mut seg_dst[off..off + 9]);
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Walks contiguous output-row segments across one L2-resident im2col patch tile."
)]
fn im2col_tile_slice(
    rounded: &[f32],
    chunk_count: usize,
    width: usize,
    height: usize,
    channels: usize,
    out_width: usize,
    patches_per_chunk: usize,
    tile_start: usize,
    first_local: usize,
    dst: &mut [f32],
) {
    let patch_width = 9 * channels;
    let local_count = dst.len() / patch_width;
    let mut done = 0_usize;
    while done < local_count {
        let pos = tile_start + first_local + done;
        let chunk_local = pos / patches_per_chunk;
        let local_pos = pos % patches_per_chunk;
        let y = local_pos / out_width;
        let x_begin = local_pos % out_width;
        let seg_len = (out_width - x_begin).min(local_count - done);
        let x_end = x_begin + seg_len;
        fill_im2col_row_segment(
            rounded,
            chunk_count,
            width,
            height,
            channels,
            chunk_local,
            y,
            x_begin,
            x_end,
            &mut dst[done * patch_width..(done + seg_len) * patch_width],
        );
        done += seg_len;
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn transpose_4x4_simd<S: Simd>(simd: S, rows: [f32x4<S>; 4]) -> [f32x4<S>; 4] {
    let t0 = simd.zip_low_f32x4(rows[0], rows[1]);
    let t1 = simd.zip_high_f32x4(rows[0], rows[1]);
    let t2 = simd.zip_low_f32x4(rows[2], rows[3]);
    let t3 = simd.zip_high_f32x4(rows[2], rows[3]);
    let c0: f32x4<S> = simd.zip_low_u64x2(t0.bitcast(), t2.bitcast()).bitcast();
    let c1: f32x4<S> = simd.zip_high_u64x2(t0.bitcast(), t2.bitcast()).bitcast();
    let c2: f32x4<S> = simd.zip_low_u64x2(t1.bitcast(), t3.bitcast()).bitcast();
    let c3: f32x4<S> = simd.zip_high_u64x2(t1.bitcast(), t3.bitcast()).bitcast();
    [c0, c1, c2, c3]
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn store_16oc_bias_gelu_tile_simd<S: Simd>(
    simd: S,
    acc_tile: &mut [[f32; 16]],
    bias_blk: &[f32; 16],
    lane_w: usize,
    total_patches: usize,
    dst_col: usize,
    out_slice: &mut [f32],
) -> io::Result<()> {
    let bv = [
        f32x4::from_slice(simd, &bias_blk[0..4]),
        f32x4::from_slice(simd, &bias_blk[4..8]),
        f32x4::from_slice(simd, &bias_blk[8..12]),
        f32x4::from_slice(simd, &bias_blk[12..16]),
    ];
    let inf = f32x4::splat(simd, f32::INFINITY);
    let mut all_finite = true;
    for acc_row in acc_tile.iter_mut() {
        let (aq, _) = acc_row.as_chunks_mut::<4>();
        for rq in 0..4 {
            let sum = f32x4::from_slice(simd, &aq[rq]) + bv[rq];
            if !sum.abs().simd_lt(inf).all_true() {
                all_finite = false;
            }
            sum.store_slice(&mut aq[rq]);
        }
    }
    if !all_finite {
        return Err(invalid());
    }
    numerics::gelu_slice_in_place(acc_tile.as_flattened_mut());
    finite(acc_tile.as_flattened())?;

    if lane_w == 16 {
        let (acc_quads, acc_tail) = acc_tile.as_chunks::<4>();
        for (q_idx, aq) in acc_quads.iter().enumerate() {
            let col = dst_col + q_idx * 4;
            for rq in 0..4 {
                let r_off = rq * 4;
                let cols = transpose_4x4_simd(
                    simd,
                    [
                        f32x4::from_slice(simd, &aq[0][r_off..r_off + 4]),
                        f32x4::from_slice(simd, &aq[1][r_off..r_off + 4]),
                        f32x4::from_slice(simd, &aq[2][r_off..r_off + 4]),
                        f32x4::from_slice(simd, &aq[3][r_off..r_off + 4]),
                    ],
                );
                for (lane, col_v) in cols.into_iter().enumerate() {
                    let off = (r_off + lane) * total_patches + col;
                    col_v.store_slice(&mut out_slice[off..off + 4]);
                }
            }
        }
        let col_base = dst_col + acc_quads.len() * 4;
        for (df, acc_row) in acc_tail.iter().enumerate() {
            for (lane, &val) in acc_row.iter().enumerate() {
                out_slice[lane * total_patches + col_base + df] = val;
            }
        }
    } else {
        for (df, acc_row) in acc_tile.iter().enumerate() {
            for lane in 0..lane_w {
                out_slice[lane * total_patches + dst_col + df] = acc_row[lane];
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Reuses batch im2col, rounded input, and channel-major buffers across multi-chunk encoder stages."
)]
fn convolve_batch_into_with_threads(
    linear: &Linear,
    packed_weights: Option<&PackedConvWeights>,
    input: &[f32],
    chunk_count: usize,
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
    let patches_per_chunk = out_width * out_height;
    let total_patches = chunk_count
        .checked_mul(patches_per_chunk)
        .ok_or_else(invalid)?;
    let patch_width = 9 * channels;
    for _ in 0..total_patches {
        check(cancel)?;
    }
    let hw = height.checked_mul(width).ok_or_else(invalid)?;
    let input_len = channels
        .checked_mul(chunk_count)
        .and_then(|v| v.checked_mul(hw))
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
        let owned_packed;
        let packed_conv = match packed_weights {
            Some(p) => p,
            None => {
                owned_packed = PackedConvWeights::from_projection(&linear.weights)?;
                &owned_packed
            }
        };
        if packed_conv.out_channels != out_channels || packed_conv.patch_width != patch_width {
            return Err(invalid());
        }
        if out_channels == 0 {
            return Ok(());
        }
        if total_patches == 0 {
            prepared.prepare_panels16_direct(0, patch_width, workers, &|_, _| {});
            return linear.weights.multiply_batch_channel_major_prepacked_into(
                prepared,
                workers,
                scratch,
                &mut output[..0],
                cancel,
            );
        }
        let rounded = &rounded_input[..input_len];
        let tile_cap = total_patches.min(CONV_TILE_PATCHES);
        ensure_buffer(scratch, tile_cap * patch_width)?;
        let level = packed::simd_level();
        let out_used = &mut output[..out_channels * total_patches];

        for tile_start in (0..total_patches).step_by(CONV_TILE_PATCHES) {
            check(cancel)?;
            let tile_len = (total_patches - tile_start).min(CONV_TILE_PATCHES);
            let tile_buf = &mut scratch[..tile_len * patch_width];
            if patch_width > 0 {
                if workers > 1 && tile_len >= 16 && tile_buf.len() >= 8_192 {
                    row_workers::run_aligned(tile_buf, workers, patch_width, &|elem_off, slice| {
                        let first_local = elem_off / patch_width;
                        im2col_tile_slice(
                            rounded,
                            chunk_count,
                            width,
                            height,
                            channels,
                            out_width,
                            patches_per_chunk,
                            tile_start,
                            first_local,
                            slice,
                        );
                    });
                } else {
                    im2col_tile_slice(
                        rounded,
                        chunk_count,
                        width,
                        height,
                        channels,
                        out_width,
                        patches_per_chunk,
                        tile_start,
                        0,
                        tile_buf,
                    );
                }
            }

            row_workers::try_run_aligned(
                out_used,
                workers,
                16 * total_patches,
                &|elem_off, slice| {
                    let first_ch = elem_off / total_patches;
                    let ch_count = slice.len() / total_patches;
                    let oc_blocks_here = ch_count.div_ceil(16);
                    for local_blk in 0..oc_blocks_here {
                        let local_ch = local_blk * 16;
                        let lane_w = 16.min(ch_count - local_ch);
                        let oc_base = first_ch + local_ch;
                        let w_blk = packed_conv.block(oc_base / 16);
                        let mut bias_blk = [0.0_f32; 16];
                        bias_blk[..lane_w].copy_from_slice(&linear.bias[oc_base..oc_base + lane_w]);
                        let oc_out_slice = &mut slice
                            [local_ch * total_patches..(local_ch + lane_w) * total_patches];

                        for f_sub in (0..tile_len).step_by(CONV_SUB_FRAMES) {
                            let f_len = (tile_len - f_sub).min(CONV_SUB_FRAMES);
                            let mut acc_tile = [[0.0_f32; 16]; CONV_SUB_FRAMES];
                            if patch_width > 0 {
                                gemm_16oc_strided_dispatch(
                                    w_blk,
                                    &tile_buf[f_sub * patch_width..],
                                    patch_width,
                                    patch_width,
                                    f_len,
                                    &mut acc_tile[..f_len],
                                );
                            }
                            dispatch!(
                                level,
                                simd => store_16oc_bias_gelu_tile_simd(
                                    simd,
                                    &mut acc_tile[..f_len],
                                    &bias_blk,
                                    lane_w,
                                    total_patches,
                                    tile_start + f_sub,
                                    oc_out_slice,
                                )
                            )?;
                        }
                    }
                    Ok(())
                },
            )?;
        }
        return Ok(());
    }

    let mut patches = Vec::new();
    ensure_buffer(&mut patches, total_patches * patch_width)?;
    if total_patches > 0 && patch_width > 0 {
        let rounded = &rounded_input[..input_len];
        im2col_tile_slice(
            rounded,
            chunk_count,
            width,
            height,
            channels,
            out_width,
            patches_per_chunk,
            0,
            0,
            &mut patches[..total_patches * patch_width],
        );
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
    channel_major_bias_and_gelu(
        &mut output[..out_channels * total_patches],
        &linear.bias,
        total_patches,
        workers,
    )
}

#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "Test convenience wrapper for single-chunk convolve."
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
    convolve_batch_into_with_threads(
        linear,
        None,
        input,
        1,
        width,
        height,
        channels,
        workers,
        prepared,
        rounded_input,
        scratch,
        output,
        cancel,
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
    Projection::multiply_batch_triple_biased_prevalidated_into(
        &block.query.weights,
        &block.key.weights,
        &block.value.weights,
        &block.query.bias,
        &block.key.bias,
        &block.value.bias,
        prepared,
        norm_buf,
        workers,
        scratch,
        q,
        k,
        v,
        cancel,
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

#[cfg(test)]
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

#[cfg(test)]
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
            (25, 16, 4, 16),
            (50, 32, 4, 25),
            (13, 11, 2, 33),
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

        // Multi-chunk encode_mel bit-exact invariance across workers (exercising multi-chunk Conv1/Conv2 batching).
        let cfg = Config {
            mel_bins: 16,
            channels: 8,
            hidden: 16,
            heads: 2,
            intermediate: 32,
            output: 8,
            window: 5,
            max_positions: 2,
        };
        let make_linear = |in_w: usize, out_w: usize, seed: usize| -> io::Result<Linear> {
            let weights = (0..in_w * out_w)
                .map(|i| {
                    (f32::from(u16::try_from((i * 7 + seed) % 37).unwrap_or(0)) - 18.0) * 0.015625
                })
                .collect();
            let bias = (0..out_w)
                .map(|i| {
                    (f32::from(u16::try_from((i * 5 + seed) % 19).unwrap_or(0)) - 9.0) * 0.03125
                })
                .collect();
            Ok(Linear {
                weights: Projection::f32(in_w, out_w, weights)?,
                bias,
            })
        };
        let make_norm = |w: usize, seed: usize| Norm {
            weight: (0..w)
                .map(|i| 0.5 + f32::from(u16::try_from((i + seed) % 7).unwrap_or(0)) * 0.125)
                .collect(),
            bias: vec![0.0; w],
        };
        let enc_weights = Weights {
            convolutions: [
                make_linear(9, 8, 1)?,
                make_linear(9 * 8, 8, 2)?,
                make_linear(9 * 8, 8, 3)?,
            ],
            convolution_out: Projection::f32(16, 16, vec![0.0625; 256])?,
            blocks: vec![Block {
                attention_norm: make_norm(16, 1),
                ffn_norm: make_norm(16, 2),
                query: make_linear(16, 16, 4)?,
                key: make_linear(16, 16, 5)?,
                value: make_linear(16, 16, 6)?,
                attention_out: make_linear(16, 16, 7)?,
                ffn_in: make_linear(16, 32, 8)?,
                ffn_out: make_linear(32, 16, 9)?,
            }],
            output_norm: make_norm(16, 3),
            output_in: make_linear(16, 16, 10)?,
            output_out: make_linear(16, 8, 11)?,
        };
        let enc_seq = Encoder::new(cfg, enc_weights)?;
        let frames = 105_usize; // 11 chunks -> 8 + 3 multi-chunk batches with partial tail chunk!
        let mel: Vec<f32> = (0..cfg.mel_bins * frames)
            .map(|i| (f32::from(u16::try_from((i * 11) % 59).unwrap_or(0)) - 29.0) * 0.03125)
            .collect();
        let out_1 = enc_seq.encode_mel(&mel, frames, &mut || false)?;
        let out_4 = enc_seq
            .with_threads(4)?
            .encode_mel(&mel, frames, &mut || false)?;
        assert_eq!(
            out_1
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            out_4
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        Ok(())
    }
}
