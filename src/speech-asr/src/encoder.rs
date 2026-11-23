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

use crate::{
    decoder::{Projection, binary16_operand},
    frontend::{Frontend, MelSpectrogram},
    packed::{self, PreparedBatch},
    protocol::EncoderLayout,
    row_workers, threads,
};
use std::io::{self, Read, Seek};

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
        prepared: &PreparedBatch,
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

    fn apply_into(&self, row: &[f32], dst: &mut [f32]) -> io::Result<()> {
        let mean = row.iter().sum::<f32>() / scalar(row.len());
        let variance = row
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / scalar(row.len());
        if !mean.is_finite() || !variance.is_finite() {
            return Err(invalid());
        }
        let scale = 1.0 / (variance + 0.000_01).sqrt();
        for ((out, value), (weight, bias)) in dst
            .iter_mut()
            .zip(row)
            .zip(self.weight.iter().zip(&self.bias))
        {
            *out = (((*value - mean) * scale) * *weight) + *bias;
        }
        finite(dst)
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
        row_workers::try_run_aligned(output, workers, width, &|elem_off, slice| {
            for (src_row, dst_row) in input[elem_off..elem_off + slice.len()]
                .chunks_exact(width)
                .zip(slice.chunks_exact_mut(width))
            {
                self.apply_into(src_row, dst_row)?;
            }
            Ok(())
        })
    }
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
        let mut patches = Vec::new();
        let mut flat_out = Vec::new();
        let mut flat_in = Vec::new();
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
                    &mut patches,
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
                let flat_dim = channels * height;
                ensure_buffer(&mut flat_in, retained * flat_dim)?;
                for time in 0..retained {
                    check(cancel)?;
                    let dst = &mut flat_in[time * flat_dim..(time + 1) * flat_dim];
                    for cf in 0..flat_dim {
                        dst[cf] = image_a[cf * width + time];
                    }
                }
                prepared.prepare_into(&flat_in, retained, flat_dim);
                let chunk_rows =
                    &mut rows[row_offset * c.hidden..(row_offset + retained) * c.hidden];
                self.weights.convolution_out.multiply_batch_prepared_into(
                    &prepared,
                    &flat_in,
                    workers,
                    &mut conv_scratch,
                    chunk_rows,
                    cancel,
                )?;
                for (row, sin_row) in chunk_rows
                    .chunks_exact_mut(c.hidden)
                    .zip(self.sinusoids.chunks_exact(c.hidden))
                {
                    for (dst, &sin_val) in row.iter_mut().zip(sin_row) {
                        *dst += sin_val;
                    }
                    finite(row)?;
                }
                row_offset += retained;
            }
        }

        let num_rows = layout.audio_rows;
        let window_rows = (8 * layout.rows_per_chunk).max(1);
        let mut norm_buf = zeros(num_rows * c.hidden)?;
        let mut q = zeros(num_rows * c.hidden)?;
        let mut k = zeros(num_rows * c.hidden)?;
        let mut v = zeros(num_rows * c.hidden)?;
        let mut attn_out = zeros(num_rows * c.hidden)?;
        let mut proj = zeros(num_rows * c.hidden)?;
        let mut ffn_buf = zeros(num_rows * c.intermediate)?;
        for block in &self.weights.blocks {
            transform_block(
                block,
                &mut rows,
                &mut norm_buf,
                &mut q,
                &mut k,
                &mut v,
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
        prepared.prepare_into(&norm_buf, num_rows, c.hidden);
        self.weights.output_in.apply_batch_prepared(
            &prepared,
            &norm_buf,
            workers,
            &mut conv_scratch,
            &mut proj,
            cancel,
        )?;
        gelu_parallel(&mut proj, workers, c.hidden)?;
        let mut values = zeros(num_rows * c.output)?;
        prepared.prepare_into(&proj, num_rows, c.hidden);
        self.weights.output_out.apply_batch_prepared(
            &prepared,
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

#[expect(
    clippy::too_many_arguments,
    reason = "Reuses batch im2col, flat output, and channel-major buffers across chunked encoder stages."
)]
fn convolve_into_with_threads(
    linear: &Linear,
    input: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    workers: usize,
    prepared: &mut PreparedBatch,
    patches: &mut Vec<f32>,
    flat_out: &mut Vec<f32>,
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
    ensure_buffer(patches, total_patches * patch_width)?;
    let row_stride = out_width * patch_width;
    row_workers::try_run_aligned(patches, workers, row_stride, &|elem_off, slice| {
        let first_y = elem_off / row_stride;
        for (local_y, y_slice) in slice.chunks_exact_mut(row_stride).enumerate() {
            let y = first_y + local_y;
            for (x, patch) in y_slice.chunks_exact_mut(patch_width).enumerate() {
                for channel in 0..channels {
                    let ch_in_base = channel * height;
                    let ch_p_base = channel * 9;
                    for ky in 0..3 {
                        let py_opt = (2 * y + ky).checked_sub(1).filter(|&py| py < height);
                        for kx in 0..3 {
                            let dst = &mut patch[ch_p_base + ky * 3 + kx];
                            if let Some(py) = py_opt
                                && let Some(px) =
                                    (2 * x + kx).checked_sub(1).filter(|&px| px < width)
                            {
                                // The pinned native im2col stage narrows every
                                // activation before its convolution dot product.
                                *dst = binary16_operand(input[(ch_in_base + py) * width + px])?;
                            } else {
                                *dst = 0.0;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    prepared.prepare_into(patches, total_patches, patch_width);
    ensure_buffer(flat_out, total_patches * out_channels)?;
    linear.apply_batch_prepared(prepared, patches, workers, scratch, flat_out, cancel)?;
    gelu_parallel(flat_out, workers, out_channels)?;
    ensure_buffer(output, total_patches * out_channels)?;
    row_workers::run_aligned(output, workers, total_patches, &|elem_off, out_slice| {
        let first_ch = elem_off / total_patches;
        for (local_ch, ch_dst) in out_slice.chunks_exact_mut(total_patches).enumerate() {
            let channel = first_ch + local_ch;
            for (pos, dst) in ch_dst.iter_mut().enumerate() {
                *dst = flat_out[pos * out_channels + channel];
            }
        }
    });
    Ok(())
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
    let mut patches = Vec::new();
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
        &mut patches,
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

#[expect(
    clippy::too_many_arguments,
    reason = "Reuses flat encoder transformer buffers across layers with zero per-layer allocations."
)]
fn transform_block(
    block: &Block,
    rows: &mut [f32],
    norm_buf: &mut [f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
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
    prepared.prepare_into(norm_buf, num_rows, c.hidden);
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
    for (dst, bias) in [
        (&mut *q, &block.query.bias),
        (&mut *k, &block.key.bias),
        (&mut *v, &block.value.bias),
    ] {
        add_batch_bias(dst, bias, workers)?;
    }

    for _ in 0..(num_rows * c.heads) {
        check(cancel)?;
    }
    let q_ref = &*q;
    let k_ref = &*k;
    let v_ref = &*v;
    let head_dim = c.hidden / c.heads;
    let divisor = scalar(head_dim).sqrt();
    let window_rows = window_rows.max(1);
    row_workers::try_run_aligned(attn_out, workers, c.hidden, &|elem_off, slice| {
        let start_row = elem_off / c.hidden;
        let mut scores = zeros(window_rows.min(num_rows).max(1))?;
        for (local_idx, out_row) in slice.chunks_exact_mut(c.hidden).enumerate() {
            let row_idx = start_row + local_idx;
            let win_start = (row_idx / window_rows) * window_rows;
            let win_end = (win_start + window_rows).min(num_rows);
            let win_len = win_end - win_start;
            let q_row = &q_ref[row_idx * c.hidden..(row_idx + 1) * c.hidden];
            let k_win = &k_ref[win_start * c.hidden..win_end * c.hidden];
            let v_win = &v_ref[win_start * c.hidden..win_end * c.hidden];
            let win_scores = &mut scores[..win_len];
            for head in 0..c.heads {
                let offset = head * head_dim;
                let q_head = &q_row[offset..offset + head_dim];
                let out_head = &mut out_row[offset..offset + head_dim];
                attention_head_bidirectional(
                    q_head, k_win, v_win, win_len, c.hidden, offset, head_dim, divisor, win_scores,
                    out_head,
                )?;
            }
            finite(out_row)?;
        }
        Ok(())
    })?;

    prepared.prepare_into(attn_out, num_rows, c.hidden);
    block
        .attention_out
        .apply_batch_prepared(prepared, attn_out, workers, scratch, proj, cancel)?;
    add_residual_batch(rows, proj, workers, c.hidden)?;

    block
        .ffn_norm
        .apply_batch(rows, norm_buf, c.hidden, workers)?;
    prepared.prepare_into(norm_buf, num_rows, c.hidden);
    block
        .ffn_in
        .apply_batch_prepared(prepared, norm_buf, workers, scratch, ffn_buf, cancel)?;
    gelu_parallel(ffn_buf, workers, c.intermediate)?;
    prepared.prepare_into(ffn_buf, num_rows, c.intermediate);
    block
        .ffn_out
        .apply_batch_prepared(prepared, ffn_buf, workers, scratch, proj, cancel)?;
    add_residual_batch(rows, proj, workers, c.hidden)
}

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

fn add_batch_bias(out: &mut [f32], bias: &[f32], workers: usize) -> io::Result<()> {
    let out_width = bias.len();
    if out_width == 0 {
        return finite(out);
    }
    row_workers::try_run_aligned(out, workers, out_width, &|_, slice| {
        for row in slice.chunks_exact_mut(out_width) {
            for (value, &b) in row.iter_mut().zip(bias) {
                *value += b;
            }
        }
        finite(slice)
    })
}

#[cfg(test)]
#[path = "encoder/tests.rs"]
mod tests;
