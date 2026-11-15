//! Rust log-mel frontend for the pinned Qwen3-ASR Whisper subset.
//!
//! This is replacement preparation; native serving still owns inference. The
//! supported configuration is 16 kHz, 128 Slaney mel bins, a periodic 400-sample
//! Hann window, a 160-sample hop, reflect padding and per-utterance normalization.
//! Preserve checkpoint-provided window/filterbank tensors when present. Reviewed
//! source identities and the limits of mathematical controls are in
//! `rust-frontend.json`; MIT attribution is retained in `LICENSE.transcribe`.
#![forbid(unsafe_code)]
#![allow(
    clippy::suboptimal_flops,
    reason = "Separate floating operations preserve the reviewed frontend's rounding and accumulation order."
)]

use crate::{packed, row_workers, threads};
use fearless_simd::{Level, dispatch, f32x4, prelude::*};
use std::io;

/// Required mono PCM sample rate.
pub const SAMPLE_RATE: usize = 16_000;
/// Number of mel-frequency rows.
pub const MEL_BINS: usize = 128;
/// Periodic Hann window and Fourier transform length.
pub const FFT_SIZE: usize = 400;
/// Samples between consecutive centered frames.
pub const HOP_LENGTH: usize = 160;
const FREQUENCIES: usize = FFT_SIZE / 2 + 1;
const PAD: usize = FFT_SIZE / 2;
const MAX_SAMPLES: usize = SAMPLE_RATE * 30;
const FILTER_ELEMENTS: usize = MEL_BINS * FREQUENCIES;

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid Qwen3-ASR frontend input or tables.",
    )
}

fn cancelled(check: &mut impl FnMut() -> bool) -> io::Result<()> {
    if check() {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "Qwen3-ASR frontend was cancelled.",
        ));
    }
    Ok(())
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The reviewed numerical contract explicitly rounds f64 table/log/normalization results to f32."
)]
const fn rounded(value: f64) -> f32 {
    value as f32
}

fn small_integer(value: usize) -> f64 {
    // All callers index fixed tables no larger than 400 entries.
    f64::from(u16::try_from(value).unwrap_or(u16::MAX))
}

fn zeroes(length: usize) -> io::Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
    values.resize(length, 0.0);
    Ok(values)
}

fn hz_to_mel(hz: f64) -> f64 {
    if hz < 1000.0 {
        hz / (200.0 / 3.0)
    } else {
        1000.0 / (200.0 / 3.0) + (hz / 1000.0).ln() / (6.4_f64.ln() / 27.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let min_log_mel = 1000.0 / (200.0 / 3.0);
    if mel < min_log_mel {
        mel * (200.0 / 3.0)
    } else {
        1000.0 * ((6.4_f64.ln() / 27.0) * (mel - min_log_mel)).exp()
    }
}

fn slaney_filterbank() -> io::Result<Vec<f32>> {
    let mut bank = zeroes(FILTER_ELEMENTS)?;
    let mut edges = [0.0; MEL_BINS + 2];
    let mel_max = hz_to_mel(8000.0);
    for (index, edge) in edges.iter_mut().enumerate() {
        *edge = mel_to_hz(mel_max * small_integer(index) / small_integer(MEL_BINS + 1));
    }
    for (mel, row) in bank.as_chunks_mut::<FREQUENCIES>().0.iter_mut().enumerate() {
        let [left, center, right] = [edges[mel], edges[mel + 1], edges[mel + 2]];
        let area = 2.0 / (right - left);
        for (bin, weight) in row.iter_mut().enumerate() {
            let hz = 16_000.0 * small_integer(bin) / small_integer(FFT_SIZE);
            let lower = (hz - left) / (center - left);
            let upper = (right - hz) / (right - center);
            *weight = rounded(lower.min(upper).max(0.0) * area);
        }
    }
    Ok(bank)
}

/// Immutable bounded frontend tables, reusable across requests without audio state.
pub struct Frontend {
    window: [f32; FFT_SIZE],
    filterbank: Vec<f32>,
    extents: [(usize, usize); MEL_BINS],
    cos: [f32; FFT_SIZE],
    sin: [f32; FFT_SIZE],
}

impl Frontend {
    /// Construct the supported frontend, preserving supplied checkpoint tensors.
    ///
    /// `window` is `[400]`; `filterbank` is row-major `[128, 201]`. `None`
    /// selects the reviewed periodic Hann / Slaney formula. The caller must
    /// verify the immutable model and reject different frontend metadata before
    /// using this fixed subset; this constructor establishes no artifact identity.
    ///
    /// # Errors
    /// Rejects malformed, nonfinite, negative or empty-support tables and failed
    /// allocation. No supplied tensor is silently replaced by a generated table.
    pub fn from_checkpoint(window: Option<&[f32]>, filterbank: Option<&[f32]>) -> io::Result<Self> {
        let mut resolved_window = [0.0; FFT_SIZE];
        if let Some(window) = window {
            if window.len() != FFT_SIZE
                || window
                    .iter()
                    .any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0)
                || !window.iter().any(|x| *x > 0.0)
            {
                return Err(invalid());
            }
            resolved_window.copy_from_slice(window);
        } else {
            for (index, value) in resolved_window.iter_mut().enumerate() {
                let phase =
                    2.0 * std::f64::consts::PI * small_integer(index) / small_integer(FFT_SIZE);
                *value = rounded(0.5 - 0.5 * phase.cos());
            }
        }
        let bank = if let Some(bank) = filterbank {
            if bank.len() != FILTER_ELEMENTS || bank.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(invalid());
            }
            let mut retained = zeroes(FILTER_ELEMENTS)?;
            retained.copy_from_slice(bank);
            retained
        } else {
            slaney_filterbank()?
        };
        let mut extents = [(0, 0); MEL_BINS];
        for (mel, row) in bank.as_chunks::<FREQUENCIES>().0.iter().enumerate() {
            let first = row
                .iter()
                .position(|weight| *weight != 0.0)
                .ok_or_else(invalid)?;
            let last = row
                .iter()
                .rposition(|weight| *weight != 0.0)
                .ok_or_else(invalid)?;
            extents[mel] = (first - first % 4, last + 1);
        }
        let mut cos = [0.0; FFT_SIZE];
        let mut sin = [0.0; FFT_SIZE];
        for index in 0..FFT_SIZE {
            let phase = 2.0 * std::f64::consts::PI * small_integer(index) / small_integer(FFT_SIZE);
            cos[index] = rounded(phase.cos());
            sin[index] = rounded(phase.sin());
        }
        Ok(Self {
            window: resolved_window,
            filterbank: bank,
            extents,
            cos,
            sin,
        })
    }

    /// Compute the complete normalized log-mel matrix for one bounded clip.
    ///
    /// Cancellation is checked during validation, padding, every frame and final
    /// normalization. Results are returned only after complete success. There is
    /// no mutable cross-request scratch or partial output to reuse after failure.
    ///
    /// # Errors
    /// Rejects fewer than 201 or more than 480,000 samples, nonfinite or out-of-
    /// range PCM, allocation failure and cancellation (`Interrupted`). The minimum
    /// is required by the native reflect-padding contract.
    pub fn compute(
        &self,
        pcm: &[f32],
        is_cancelled: impl FnMut() -> bool,
    ) -> io::Result<MelSpectrogram> {
        self.compute_with_threads(pcm, 1, is_cancelled)
    }

    /// Compute the complete normalized log-mel matrix with bounded worker threads.
    ///
    /// Uses bit-exact 4-frame `fearless_simd` FFT + power + Slaney filterbank tiles
    /// across `threads` workers, producing identical `f32` bits to [`Self::compute`].
    ///
    /// # Errors
    /// Same contract as [`Self::compute`], plus `InvalidInput` for thread counts
    /// outside `1..=256`.
    pub(crate) fn compute_with_threads(
        &self,
        pcm: &[f32],
        threads: usize,
        mut is_cancelled: impl FnMut() -> bool,
    ) -> io::Result<MelSpectrogram> {
        cancelled(&mut is_cancelled)?;
        let threads = threads::validate_thread_count(threads)?;
        if !(PAD + 1..=MAX_SAMPLES).contains(&pcm.len()) {
            return Err(invalid());
        }
        for chunk in pcm.chunks(4096) {
            cancelled(&mut is_cancelled)?;
            if !packed::pcm_in_unit_range(chunk) {
                return Err(invalid());
            }
        }
        let mut padded = zeroes(pcm.len() + 2 * PAD)?;
        for (part, chunk) in pcm.chunks(4096).enumerate() {
            cancelled(&mut is_cancelled)?;
            let start = PAD + part * 4096;
            padded[start..start + chunk.len()].copy_from_slice(chunk);
        }
        reflect_edges(&mut padded, pcm.len());
        let frames = pcm.len() / HOP_LENGTH;
        // The native center-padding transform computes one additional frame and
        // discards it before finding the utterance maximum. Omitting that unused
        // computation preserves every emitted frame and the maximum's scope.
        for _ in 0..frames {
            cancelled(&mut is_cancelled)?;
        }
        let mut frame_major = zeroes(frames * MEL_BINS)?;
        let level = Level::new();
        row_workers::run_aligned(
            &mut frame_major,
            threads,
            4 * MEL_BINS,
            &|elem_off, slice| {
                let start_frame = elem_off / MEL_BINS;
                dispatch!(
                    level,
                    simd => compute_frame_slice_simd(simd, self, &padded, start_frame, slice)
                );
            },
        );
        let mut values = zeroes(MEL_BINS * frames)?;
        for frame in 0..frames {
            let src = &frame_major[frame * MEL_BINS..(frame + 1) * MEL_BINS];
            for mel in 0..MEL_BINS {
                values[mel * frames + frame] = src[mel];
            }
        }
        normalize(&mut values, &mut is_cancelled)?;
        Ok(MelSpectrogram { frames, values })
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn compute_frame_slice_simd<S: Simd>(
    simd: S,
    frontend: &Frontend,
    padded: &[f32],
    start_frame: usize,
    slice: &mut [f32],
) {
    let quad_stride = 4 * MEL_BINS;
    let quad_count = slice.len() / quad_stride;
    if quad_count > 0 {
        let mut input4 = [f32x4::splat(simd, 0.0); 2 * FFT_SIZE];
        let mut output4 = [f32x4::splat(simd, 0.0); 8 * FFT_SIZE];
        let mut power4 = [[0.0_f32; 4]; FREQUENCIES];
        for (quad_idx, group_out) in slice[..quad_count * quad_stride]
            .chunks_exact_mut(quad_stride)
            .enumerate()
        {
            compute_four_frames_simd(
                simd,
                frontend,
                padded,
                start_frame + quad_idx * 4,
                &mut input4,
                &mut output4,
                &mut power4,
                group_out,
            );
        }
    }
    let tail_slice = &mut slice[quad_count * quad_stride..];
    if !tail_slice.is_empty() {
        let mut input = [0.0_f32; 2 * FFT_SIZE];
        let mut output = [0.0_f32; 8 * FFT_SIZE];
        let mut power = [0.0_f32; FREQUENCIES];
        let tail_start_frame = start_frame + quad_count * 4;
        for (tail_idx, frame_out) in tail_slice.chunks_exact_mut(MEL_BINS).enumerate() {
            compute_single_frame(
                frontend,
                padded,
                tail_start_frame + tail_idx,
                &mut input,
                &mut output,
                &mut power,
                frame_out,
            );
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd helper called within dispatch! target_feature boundary."
)]
#[expect(
    clippy::too_many_arguments,
    reason = "Reuses stack/worker scratch buffers across 4-frame FFT chunks with zero heap allocations."
)]
#[inline(always)]
fn compute_four_frames_simd<S: Simd>(
    simd: S,
    frontend: &Frontend,
    padded: &[f32],
    base_frame: usize,
    input: &mut [f32x4<S>; 2 * FFT_SIZE],
    output: &mut [f32x4<S>; 8 * FFT_SIZE],
    power4: &mut [[f32; 4]; FREQUENCIES],
    group_out: &mut [f32],
) {
    let s0 = base_frame * HOP_LENGTH;
    let s1 = (base_frame + 1) * HOP_LENGTH;
    let s2 = (base_frame + 2) * HOP_LENGTH;
    let s3 = (base_frame + 3) * HOP_LENGTH;
    for (index, slot) in input[..FFT_SIZE].iter_mut().enumerate() {
        let samples = f32x4::from_slice(
            simd,
            &[
                padded[s0 + index],
                padded[s1 + index],
                padded[s2 + index],
                padded[s3 + index],
            ],
        );
        *slot = samples * f32x4::splat(simd, frontend.window[index]);
    }
    mixed_fft_4(simd, input, FFT_SIZE, &frontend.cos, &frontend.sin, output);
    for (index, slot) in power4.iter_mut().enumerate() {
        let real = output[2 * index];
        let imaginary = output[2 * index + 1];
        *slot = (real * real + imaginary * imaginary).into();
    }
    for mel in 0..MEL_BINS {
        let bank = &frontend.filterbank[mel * FREQUENCIES..(mel + 1) * FREQUENCIES];
        let (mut bin, end) = frontend.extents[mel];
        let mut sum = [0.0_f64; 4];
        while bin < end && bin < FREQUENCIES - 3 {
            let b0 = f64::from(bank[bin]);
            let b1 = f64::from(bank[bin + 1]);
            let b2 = f64::from(bank[bin + 2]);
            let b3 = f64::from(bank[bin + 3]);
            let p0 = power4[bin];
            let p1 = power4[bin + 1];
            let p2 = power4[bin + 2];
            let p3 = power4[bin + 3];
            for lane in 0..4 {
                sum[lane] += b0 * f64::from(p0[lane])
                    + b1 * f64::from(p1[lane])
                    + b2 * f64::from(p2[lane])
                    + b3 * f64::from(p3[lane]);
            }
            bin += 4;
        }
        while bin < end {
            let b = f64::from(bank[bin]);
            let p = power4[bin];
            for lane in 0..4 {
                sum[lane] += b * f64::from(p[lane]);
            }
            bin += 1;
        }
        for lane in 0..4 {
            group_out[lane * MEL_BINS + mel] = rounded(sum[lane].max(1.0e-10).log10());
        }
    }
}

fn compute_single_frame(
    frontend: &Frontend,
    padded: &[f32],
    frame: usize,
    input: &mut [f32; 2 * FFT_SIZE],
    output: &mut [f32; 8 * FFT_SIZE],
    power: &mut [f32; FREQUENCIES],
    frame_out: &mut [f32],
) {
    let start = frame * HOP_LENGTH;
    for (index, sample) in input[..FFT_SIZE].iter_mut().enumerate() {
        *sample = padded[start + index] * frontend.window[index];
    }
    mixed_fft(input, FFT_SIZE, &frontend.cos, &frontend.sin, output);
    for (index, value) in power.iter_mut().enumerate() {
        let real = output[2 * index];
        let imaginary = output[2 * index + 1];
        *value = real * real + imaginary * imaginary;
    }
    for mel in 0..MEL_BINS {
        let bank = &frontend.filterbank[mel * FREQUENCIES..(mel + 1) * FREQUENCIES];
        let (mut bin, end) = frontend.extents[mel];
        let mut sum = 0.0;
        while bin < end && bin < FREQUENCIES - 3 {
            sum += f64::from(bank[bin]) * f64::from(power[bin])
                + f64::from(bank[bin + 1]) * f64::from(power[bin + 1])
                + f64::from(bank[bin + 2]) * f64::from(power[bin + 2])
                + f64::from(bank[bin + 3]) * f64::from(power[bin + 3]);
            bin += 4;
        }
        while bin < end {
            sum += f64::from(bank[bin]) * f64::from(power[bin]);
            bin += 1;
        }
        frame_out[mel] = rounded(sum.max(1.0e-10).log10());
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd recursive FFT helper inherits target_feature from dispatch!."
)]
#[inline(always)]
fn mixed_fft_4<S: Simd>(
    simd: S,
    input: &mut [f32x4<S>],
    size: usize,
    cos: &[f32; FFT_SIZE],
    sin: &[f32; FFT_SIZE],
    output: &mut [f32x4<S>],
) {
    if size == 1 {
        output[0] = input[0];
        output[1] = f32x4::splat(simd, 0.0);
    } else if !size.is_multiple_of(2) {
        let stride = FFT_SIZE / size;
        for bin in 0..size {
            let mut real = f32x4::splat(simd, 0.0);
            let mut imaginary = f32x4::splat(simd, 0.0);
            for (index, sample) in input[..size].iter().enumerate() {
                let phase = (bin * index * stride) % FFT_SIZE;
                real += *sample * f32x4::splat(simd, cos[phase]);
                imaginary -= *sample * f32x4::splat(simd, sin[phase]);
            }
            output[2 * bin] = real;
            output[2 * bin + 1] = imaginary;
        }
    } else {
        let half = size / 2;
        let (original, input_scratch) = input.split_at_mut(size);
        for index in 0..half {
            input_scratch[index] = original[2 * index];
        }
        let (result, output_scratch) = output.split_at_mut(2 * size);
        mixed_fft_4(simd, input_scratch, half, cos, sin, output_scratch);
        for index in 0..half {
            input_scratch[index] = original[2 * index + 1];
        }
        let (even, odd_scratch) = output_scratch.split_at_mut(size);
        mixed_fft_4(simd, input_scratch, half, cos, sin, odd_scratch);
        let step = FFT_SIZE / size;
        for bin in 0..half {
            let phase = bin * step;
            let wr = f32x4::splat(simd, cos[phase]);
            let wi = f32x4::splat(simd, -sin[phase]);
            let real = odd_scratch[2 * bin];
            let imaginary = odd_scratch[2 * bin + 1];
            let wr_real = wr * real;
            let wi_imag = wi * imaginary;
            let wr_imag = wr * imaginary;
            let wi_real = wi * real;
            result[2 * bin] = even[2 * bin] + wr_real - wi_imag;
            result[2 * bin + 1] = even[2 * bin + 1] + wr_imag + wi_real;
            result[2 * (bin + half)] = even[2 * bin] - wr_real + wi_imag;
            result[2 * (bin + half) + 1] = even[2 * bin + 1] - wr_imag - wi_real;
        }
    }
}

/// Complete normalized row-major `[128, frames]` log-mel features.
#[derive(Debug)]
pub struct MelSpectrogram {
    frames: usize,
    values: Vec<f32>,
}

impl MelSpectrogram {
    /// Number of frames, exactly `floor(input_samples / 160)`.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Normalized row-major matrix, with all frames of each mel bin contiguous.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

fn reflect_edges(padded: &mut [f32], samples: usize) {
    for index in 0..PAD {
        padded[index] = padded[2 * PAD - index];
        padded[PAD + samples + index] = padded[PAD + samples - 2 - index];
    }
}

fn normalize(values: &mut [f32], check: &mut impl FnMut() -> bool) -> io::Result<()> {
    let mut maximum = f64::NEG_INFINITY;
    for chunk in values.chunks(4096) {
        cancelled(check)?;
        for value in chunk {
            if !value.is_finite() {
                return Err(invalid());
            }
            maximum = maximum.max(f64::from(*value));
        }
    }
    let floor = maximum - 8.0;
    for chunk in values.chunks_mut(4096) {
        cancelled(check)?;
        for value in chunk {
            *value = rounded((f64::from(*value).max(floor) + 4.0) / 4.0);
        }
    }
    cancelled(check)
}

// Exact mixed-radix split, odd-leaf DFT and butterfly order of the reviewed
// fp32 frontend. Buffers follow its 2*N input / 8*N output scratch contract.
fn mixed_fft(
    input: &mut [f32],
    size: usize,
    cos: &[f32; FFT_SIZE],
    sin: &[f32; FFT_SIZE],
    output: &mut [f32],
) {
    if size == 1 {
        output[0] = input[0];
        output[1] = 0.0;
    } else if size % 2 == 1 {
        let stride = FFT_SIZE / size;
        for bin in 0..size {
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for (index, sample) in input[..size].iter().enumerate() {
                let phase = (bin * index * stride) % FFT_SIZE;
                real += *sample * cos[phase];
                imaginary -= *sample * sin[phase];
            }
            output[2 * bin] = real;
            output[2 * bin + 1] = imaginary;
        }
    } else {
        let half = size / 2;
        let (original, input_scratch) = input.split_at_mut(size);
        for index in 0..half {
            input_scratch[index] = original[2 * index];
        }
        let (result, output_scratch) = output.split_at_mut(2 * size);
        mixed_fft(input_scratch, half, cos, sin, output_scratch);
        for index in 0..half {
            input_scratch[index] = original[2 * index + 1];
        }
        let (even, odd_scratch) = output_scratch.split_at_mut(size);
        mixed_fft(input_scratch, half, cos, sin, odd_scratch);
        let step = FFT_SIZE / size;
        for bin in 0..half {
            let phase = bin * step;
            let wr = cos[phase];
            let wi = -sin[phase];
            let real = odd_scratch[2 * bin];
            let imaginary = odd_scratch[2 * bin + 1];
            result[2 * bin] = even[2 * bin] + wr * real - wi * imaginary;
            result[2 * bin + 1] = even[2 * bin + 1] + wr * imaginary + wi * real;
            result[2 * (bin + half)] = even[2 * bin] - wr * real + wi * imaginary;
            result[2 * (bin + half) + 1] = even[2 * bin + 1] - wr * imaginary - wi * real;
        }
    }
}

#[cfg(test)]
#[path = "frontend/tests.rs"]
mod tests;
