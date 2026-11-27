//! Pure-Rust ECAPA-TDNN speaker encoder and Slaney log-mel spectrogram extractor.
//!
//! Mirrors `qwen_tts.core.models.modeling_qwen3_tts.Qwen3TTSSpeakerEncoder`
//! for 24 kHz mono reference waveforms:
//! 1. Reflect-pad by `(n_fft - hop) / 2 = 384` samples on each side.
//! 2. Periodic Hann window (`n_fft = 1024, hop = 256`) + real/imaginary DFT
//!    power spectrum (`mag = sqrt(re^2 + im^2 + 1e-9)`).
//! 3. Slaney-normalized 128-bin triangular mel filterbank (`0..12000` Hz)
//!    followed by `log(max(mel, 1e-5))`.
//! 4. Initial TDNN (`k = 5, 128 -> 512`, reflect-padded) + `ReLU`.
//! 5. Three SE-Res2Net blocks (`dilations = [2, 3, 4], scale = 8, se_channels = 128`).
//! 6. Multi-layer feature aggregation (`MFA`, `k = 1, 1536 -> 1536`) + `ReLU`.
//! 7. Attentive Statistical Pooling (`ASP`, `1536 -> 3072`) + final `k = 1` FC
//!    projection (`3072 -> 2048` x-vector speaker embedding).
#![forbid(unsafe_code)]
#![expect(
    clippy::missing_errors_doc,
    reason = "Internal spectrogram and TDNN shape checks return io::ErrorKind::InvalidInput."
)]

use std::io;

use fearless_simd::{Level, dispatch, f32x4, prelude::*};

use crate::{
    engine::row_workers,
    numerics::{f32_dot_simd, usize_to_f32},
};

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The pinned audio-mel.h explicitly rounds double-precision sine/cosine coefficients to binary32 before any subsequent arithmetic."
)]
const fn coefficient(value: f64) -> f32 {
    value as f32
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "Register-blocked 4x4 TDNN convolution kernel keeps 16 accumulators live across k and ic."
)]
fn tdnn_4oc_row_simd<S: Simd>(
    simd: S,
    padded: &[f32],
    padded_frames: usize,
    weight: &[f32],
    bias: [f32; 4],
    in_c: usize,
    kernel: usize,
    dilation: usize,
    frames: usize,
    apply_relu: bool,
    out_quad: &mut [f32],
) {
    let zero_v = f32x4::splat(simd, 0.0_f32);
    let (r0_r1, r2_r3) = out_quad.split_at_mut(2 * frames);
    let (r0, r1) = r0_r1.split_at_mut(frames);
    let (r2, r3) = r2_r3.split_at_mut(frames);
    let b0 = f32x4::splat(simd, bias[0]);
    let b1 = f32x4::splat(simd, bias[1]);
    let b2 = f32x4::splat(simd, bias[2]);
    let b3 = f32x4::splat(simd, bias[3]);
    let w_stride = in_c * kernel;

    let mut f = 0_usize;
    while f + 16 <= frames {
        let mut a00 = b0;
        let mut a01 = b0;
        let mut a02 = b0;
        let mut a03 = b0;
        let mut a10 = b1;
        let mut a11 = b1;
        let mut a12 = b1;
        let mut a13 = b1;
        let mut a20 = b2;
        let mut a21 = b2;
        let mut a22 = b2;
        let mut a23 = b2;
        let mut a30 = b3;
        let mut a31 = b3;
        let mut a32 = b3;
        let mut a33 = b3;
        for k in 0..kernel {
            let tap_off = k * dilation + f;
            for ic in 0..in_c {
                let x_off = ic * padded_frames + tap_off;
                let x0 = f32x4::from_slice(simd, &padded[x_off..x_off + 4]);
                let x1 = f32x4::from_slice(simd, &padded[x_off + 4..x_off + 8]);
                let x2 = f32x4::from_slice(simd, &padded[x_off + 8..x_off + 12]);
                let x3 = f32x4::from_slice(simd, &padded[x_off + 12..x_off + 16]);
                let w_off = ic * kernel + k;
                let w0 = f32x4::splat(simd, weight[w_off]);
                let w1 = f32x4::splat(simd, weight[w_stride + w_off]);
                let w2 = f32x4::splat(simd, weight[2 * w_stride + w_off]);
                let w3 = f32x4::splat(simd, weight[3 * w_stride + w_off]);
                a00 = w0.mul_add(x0, a00);
                a01 = w0.mul_add(x1, a01);
                a02 = w0.mul_add(x2, a02);
                a03 = w0.mul_add(x3, a03);
                a10 = w1.mul_add(x0, a10);
                a11 = w1.mul_add(x1, a11);
                a12 = w1.mul_add(x2, a12);
                a13 = w1.mul_add(x3, a13);
                a20 = w2.mul_add(x0, a20);
                a21 = w2.mul_add(x1, a21);
                a22 = w2.mul_add(x2, a22);
                a23 = w2.mul_add(x3, a23);
                a30 = w3.mul_add(x0, a30);
                a31 = w3.mul_add(x1, a31);
                a32 = w3.mul_add(x2, a32);
                a33 = w3.mul_add(x3, a33);
            }
        }
        if apply_relu {
            a00 = a00.max(zero_v);
            a01 = a01.max(zero_v);
            a02 = a02.max(zero_v);
            a03 = a03.max(zero_v);
            a10 = a10.max(zero_v);
            a11 = a11.max(zero_v);
            a12 = a12.max(zero_v);
            a13 = a13.max(zero_v);
            a20 = a20.max(zero_v);
            a21 = a21.max(zero_v);
            a22 = a22.max(zero_v);
            a23 = a23.max(zero_v);
            a30 = a30.max(zero_v);
            a31 = a31.max(zero_v);
            a32 = a32.max(zero_v);
            a33 = a33.max(zero_v);
        }
        a00.store_slice(&mut r0[f..f + 4]);
        a01.store_slice(&mut r0[f + 4..f + 8]);
        a02.store_slice(&mut r0[f + 8..f + 12]);
        a03.store_slice(&mut r0[f + 12..f + 16]);
        a10.store_slice(&mut r1[f..f + 4]);
        a11.store_slice(&mut r1[f + 4..f + 8]);
        a12.store_slice(&mut r1[f + 8..f + 12]);
        a13.store_slice(&mut r1[f + 12..f + 16]);
        a20.store_slice(&mut r2[f..f + 4]);
        a21.store_slice(&mut r2[f + 4..f + 8]);
        a22.store_slice(&mut r2[f + 8..f + 12]);
        a23.store_slice(&mut r2[f + 12..f + 16]);
        a30.store_slice(&mut r3[f..f + 4]);
        a31.store_slice(&mut r3[f + 4..f + 8]);
        a32.store_slice(&mut r3[f + 8..f + 12]);
        a33.store_slice(&mut r3[f + 12..f + 16]);
        f += 16;
    }
    while f + 4 <= frames {
        let mut a0 = b0;
        let mut a1 = b1;
        let mut a2 = b2;
        let mut a3 = b3;
        for k in 0..kernel {
            let tap_off = k * dilation + f;
            for ic in 0..in_c {
                let x_off = ic * padded_frames + tap_off;
                let xv = f32x4::from_slice(simd, &padded[x_off..x_off + 4]);
                let w_off = ic * kernel + k;
                a0 = f32x4::splat(simd, weight[w_off]).mul_add(xv, a0);
                a1 = f32x4::splat(simd, weight[w_stride + w_off]).mul_add(xv, a1);
                a2 = f32x4::splat(simd, weight[2 * w_stride + w_off]).mul_add(xv, a2);
                a3 = f32x4::splat(simd, weight[3 * w_stride + w_off]).mul_add(xv, a3);
            }
        }
        if apply_relu {
            a0 = a0.max(zero_v);
            a1 = a1.max(zero_v);
            a2 = a2.max(zero_v);
            a3 = a3.max(zero_v);
        }
        a0.store_slice(&mut r0[f..f + 4]);
        a1.store_slice(&mut r1[f..f + 4]);
        a2.store_slice(&mut r2[f..f + 4]);
        a3.store_slice(&mut r3[f..f + 4]);
        f += 4;
    }
    while f < frames {
        let mut a0 = bias[0];
        let mut a1 = bias[1];
        let mut a2 = bias[2];
        let mut a3 = bias[3];
        for k in 0..kernel {
            let tap_off = k * dilation + f;
            for ic in 0..in_c {
                let xv = padded[ic * padded_frames + tap_off];
                let w_off = ic * kernel + k;
                a0 = weight[w_off].mul_add(xv, a0);
                a1 = weight[w_stride + w_off].mul_add(xv, a1);
                a2 = weight[2 * w_stride + w_off].mul_add(xv, a2);
                a3 = weight[3 * w_stride + w_off].mul_add(xv, a3);
            }
        }
        if apply_relu {
            a0 = a0.max(0.0_f32);
            a1 = a1.max(0.0_f32);
            a2 = a2.max(0.0_f32);
            a3 = a3.max(0.0_f32);
        }
        r0[f] = a0;
        r1[f] = a1;
        r2[f] = a2;
        r3[f] = a3;
        f += 1;
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "Register-blocked 1x4 TDNN tail convolution kernel keeps 4 accumulators live across k and ic."
)]
fn tdnn_1oc_row_simd<S: Simd>(
    simd: S,
    padded: &[f32],
    padded_frames: usize,
    weight_row: &[f32],
    bias: f32,
    in_c: usize,
    kernel: usize,
    dilation: usize,
    frames: usize,
    apply_relu: bool,
    out_row: &mut [f32],
) {
    let zero_v = f32x4::splat(simd, 0.0_f32);
    let b_v = f32x4::splat(simd, bias);
    let mut f = 0_usize;
    while f + 16 <= frames {
        let mut a0 = b_v;
        let mut a1 = b_v;
        let mut a2 = b_v;
        let mut a3 = b_v;
        for k in 0..kernel {
            let tap_off = k * dilation + f;
            for ic in 0..in_c {
                let x_off = ic * padded_frames + tap_off;
                let w = f32x4::splat(simd, weight_row[ic * kernel + k]);
                a0 = w.mul_add(f32x4::from_slice(simd, &padded[x_off..x_off + 4]), a0);
                a1 = w.mul_add(f32x4::from_slice(simd, &padded[x_off + 4..x_off + 8]), a1);
                a2 = w.mul_add(f32x4::from_slice(simd, &padded[x_off + 8..x_off + 12]), a2);
                a3 = w.mul_add(f32x4::from_slice(simd, &padded[x_off + 12..x_off + 16]), a3);
            }
        }
        if apply_relu {
            a0 = a0.max(zero_v);
            a1 = a1.max(zero_v);
            a2 = a2.max(zero_v);
            a3 = a3.max(zero_v);
        }
        a0.store_slice(&mut out_row[f..f + 4]);
        a1.store_slice(&mut out_row[f + 4..f + 8]);
        a2.store_slice(&mut out_row[f + 8..f + 12]);
        a3.store_slice(&mut out_row[f + 12..f + 16]);
        f += 16;
    }
    while f + 4 <= frames {
        let mut acc = b_v;
        for k in 0..kernel {
            let tap_off = k * dilation + f;
            for ic in 0..in_c {
                let x_off = ic * padded_frames + tap_off;
                let w = f32x4::splat(simd, weight_row[ic * kernel + k]);
                acc = w.mul_add(f32x4::from_slice(simd, &padded[x_off..x_off + 4]), acc);
            }
        }
        if apply_relu {
            acc = acc.max(zero_v);
        }
        acc.store_slice(&mut out_row[f..f + 4]);
        f += 4;
    }
    while f < frames {
        let mut acc = bias;
        for k in 0..kernel {
            let tap_off = k * dilation + f;
            for ic in 0..in_c {
                acc =
                    weight_row[ic * kernel + k].mul_add(padded[ic * padded_frames + tap_off], acc);
            }
        }
        out_row[f] = if apply_relu { acc.max(0.0_f32) } else { acc };
        f += 1;
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn window_frame_simd<S: Simd>(simd: S, frame: &[f32], hann: &[f32], dst: &mut [f32]) {
    let (f_chunks, f_tail) = frame.as_chunks::<4>();
    let (h_chunks, h_tail) = hann.as_chunks::<4>();
    let (d_chunks, d_tail) = dst.as_chunks_mut::<4>();
    for ((fc, hc), dc) in f_chunks
        .iter()
        .zip(h_chunks.iter())
        .zip(d_chunks.iter_mut())
    {
        let fv = f32x4::from_slice(simd, fc);
        let hv = f32x4::from_slice(simd, hc);
        (fv * hv).store_slice(dc);
    }
    for ((&f, &h), d) in f_tail.iter().zip(h_tail.iter()).zip(d_tail.iter_mut()) {
        *d = f * h;
    }
}

#[expect(
    clippy::inline_always,
    reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
)]
#[inline(always)]
fn f32_dot_pair_simd<S: Simd>(simd: S, x: &[f32], y0: &[f32], y1: &[f32]) -> (f32, f32) {
    let (x_chunks, x_tail) = x.as_chunks::<4>();
    let (y0_chunks, y0_tail) = y0.as_chunks::<4>();
    let (y1_chunks, y1_tail) = y1.as_chunks::<4>();
    let mut acc0_a = f32x4::splat(simd, 0.0_f32);
    let mut acc0_b = f32x4::splat(simd, 0.0_f32);
    let mut acc1_a = f32x4::splat(simd, 0.0_f32);
    let mut acc1_b = f32x4::splat(simd, 0.0_f32);
    let (x_pairs, x_rem) = x_chunks.as_chunks::<2>();
    let (y0_pairs, y0_rem) = y0_chunks.as_chunks::<2>();
    let (y1_pairs, y1_rem) = y1_chunks.as_chunks::<2>();
    for ((xp, y0p), y1p) in x_pairs.iter().zip(y0_pairs.iter()).zip(y1_pairs.iter()) {
        let x_a = f32x4::from_slice(simd, &xp[0]);
        let x_b = f32x4::from_slice(simd, &xp[1]);
        acc0_a = x_a.mul_add(f32x4::from_slice(simd, &y0p[0]), acc0_a);
        acc0_b = x_b.mul_add(f32x4::from_slice(simd, &y0p[1]), acc0_b);
        acc1_a = x_a.mul_add(f32x4::from_slice(simd, &y1p[0]), acc1_a);
        acc1_b = x_b.mul_add(f32x4::from_slice(simd, &y1p[1]), acc1_b);
    }
    for ((xc, y0c), y1c) in x_rem.iter().zip(y0_rem.iter()).zip(y1_rem.iter()) {
        let xv = f32x4::from_slice(simd, xc);
        acc0_a = xv.mul_add(f32x4::from_slice(simd, y0c), acc0_a);
        acc1_a = xv.mul_add(f32x4::from_slice(simd, y1c), acc1_a);
    }
    let mut l0 = [0.0_f32; 4];
    let mut l1 = [0.0_f32; 4];
    (acc0_a + acc0_b).store_slice(&mut l0);
    (acc1_a + acc1_b).store_slice(&mut l1);
    let mut s0 = (l0[0] + l0[1]) + (l0[2] + l0[3]);
    let mut s1 = (l1[0] + l1[1]) + (l1[2] + l1[3]);
    for ((&xv, &v0), &v1) in x_tail.iter().zip(y0_tail.iter()).zip(y1_tail.iter()) {
        s0 = xv.mul_add(v0, s0);
        s1 = xv.mul_add(v1, s1);
    }
    (s0, s1)
}

/// Expected input sample rate for the Qwen3-TTS speaker encoder (`24_000` Hz).
pub const SPEAKER_ENCODER_SAMPLE_RATE: u32 = 24_000;
/// Default STFT FFT size (`1024`).
pub const MEL_N_FFT: usize = 1024;
/// Default STFT hop size (`256`).
pub const MEL_HOP: usize = 256;
/// Default number of mel filterbank channels (`128`).
pub const MEL_CHANNELS: usize = 128;
/// Default speaker x-vector embedding dimension (`2048`).
pub const SPEAKER_EMBEDDING_DIM: usize = 2048;

/// Configuration for the Slaney log-mel spectrogram frontend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioMelConfig {
    /// Audio sample rate in Hz (`24_000`).
    pub sample_rate: u32,
    /// FFT window length (`1024`).
    pub n_fft: usize,
    /// Frame hop length (`256`).
    pub hop: usize,
    /// Number of mel bins (`128`).
    pub n_mels: usize,
    /// Minimum filterbank frequency in Hz (`0.0`).
    pub fmin: f32,
    /// Maximum filterbank frequency in Hz (`12_000.0`).
    pub fmax: f32,
}

impl Default for AudioMelConfig {
    fn default() -> Self {
        Self {
            sample_rate: SPEAKER_ENCODER_SAMPLE_RATE,
            n_fft: MEL_N_FFT,
            hop: MEL_HOP,
            n_mels: MEL_CHANNELS,
            fmin: 0.0,
            fmax: 12_000.0,
        }
    }
}

/// Slaney mel-scale forward conversion (`librosa.filters.mel(htk=False)`).
#[must_use]
pub fn hz_to_mel_slaney(hz: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = 6.4_f32.ln() / 27.0_f32;
    if hz < MIN_LOG_HZ {
        (hz - F_MIN) / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    }
}

/// Slaney mel-scale inverse conversion (`librosa.filters.mel(htk=False)`).
#[must_use]
pub fn mel_to_hz_slaney(mel: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = 6.4_f32.ln() / 27.0_f32;
    if mel < MIN_LOG_MEL {
        F_SP.mul_add(mel, F_MIN)
    } else {
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    }
}

/// Precomputed periodic Hann window, DFT basis matrices, and Slaney mel filterbank.
#[derive(Clone, Debug)]
pub struct MelSpectrogramExtractor {
    cfg: AudioMelConfig,
    n_freq: usize,
    hann: Vec<f32>,
    dft_real: Vec<f32>,
    dft_imag: Vec<f32>,
    fft_bitrev: Vec<usize>,
    fft_cos: Vec<f32>,
    fft_sin: Vec<f32>,
    mel_basis: Vec<f32>,
    mel_extents: Vec<(usize, usize)>,
}

impl MelSpectrogramExtractor {
    fn use_fft(&self) -> bool {
        !self.fft_bitrev.is_empty()
    }

    /// Bake the periodic Hann window, DFT/FFT basis, and Slaney triangular mel filterbank.
    ///
    /// # Errors
    /// Rejects zero `n_fft`, `hop`, `n_mels`, or `n_fft < hop`.
    pub fn new(cfg: AudioMelConfig) -> io::Result<Self> {
        let u32_to_f64 = |value: usize, msg: &str| -> io::Result<f64> {
            u32::try_from(value)
                .map(f64::from)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, msg))
        };
        if cfg.n_fft == 0 || cfg.hop == 0 || cfg.n_mels == 0 || cfg.n_fft < cfg.hop {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid mel spectrogram configuration.",
            ));
        }
        let n_freq = cfg.n_fft / 2 + 1;
        let n_fft_f = usize_to_f32(cfg.n_fft);
        let n_fft_double = u32_to_f64(cfg.n_fft, "FFT length exceeds u32.")?;
        let two_pi = 2.0_f64 * core::f64::consts::PI;

        let mut hann = Vec::with_capacity(cfg.n_fft);
        for i in 0..cfg.n_fft {
            let phase = two_pi * u32_to_f64(i, "Hann index exceeds u32.")? / n_fft_double;
            // The original narrows cos(), then subtracts/multiplies in f32.
            hann.push(0.5_f32 * (1.0_f32 - coefficient(phase.cos())));
        }

        let use_fft = cfg.n_fft.is_power_of_two() && cfg.n_fft >= 32;
        let (dft_real, dft_imag, fft_bitrev, fft_cos, fft_sin) = if use_fft {
            let half_n = cfg.n_fft / 2;
            let bits = half_n.trailing_zeros();
            let mut bitrev = Vec::with_capacity(half_n);
            for i in 0..half_n {
                let mut v = i;
                let mut r = 0_usize;
                for _ in 0..bits {
                    r = (r << 1) | (v & 1);
                    v >>= 1;
                }
                bitrev.push(r);
            }
            let mut cos_t = Vec::with_capacity(half_n);
            let mut sin_t = Vec::with_capacity(half_n);
            for k in 0..half_n {
                let th = two_pi * u32_to_f64(k, "FFT twiddle index exceeds u32.")? / n_fft_double;
                cos_t.push(coefficient(th.cos()));
                sin_t.push(coefficient(-th.sin()));
            }
            (Vec::new(), Vec::new(), bitrev, cos_t, sin_t)
        } else {
            let mut dft_real = vec![0.0_f32; n_freq * cfg.n_fft];
            let mut dft_imag = vec![0.0_f32; n_freq * cfg.n_fft];
            for k in 0..n_freq {
                let kf = u32_to_f64(k, "DFT row exceeds u32.")?;
                for n in 0..cfg.n_fft {
                    let nf = u32_to_f64(n, "DFT column exceeds u32.")?;
                    let th = two_pi * kf * nf / n_fft_double;
                    dft_real[k * cfg.n_fft + n] = coefficient(th.cos());
                    dft_imag[k * cfg.n_fft + n] = coefficient(-th.sin());
                }
            }
            (dft_real, dft_imag, Vec::new(), Vec::new(), Vec::new())
        };

        let sr_f = usize_to_f32(usize::try_from(cfg.sample_rate).unwrap_or(24_000));
        let fmin = cfg.fmin.max(0.0);
        let fmax = if cfg.fmax <= 0.0 {
            sr_f * 0.5
        } else {
            cfg.fmax
        };
        let mmin = hz_to_mel_slaney(fmin);
        let mmax = hz_to_mel_slaney(fmax);

        let mut hz_pts = Vec::with_capacity(cfg.n_mels + 2);
        let denom = usize_to_f32(cfg.n_mels + 1);
        for i in 0..(cfg.n_mels + 2) {
            let mel = mmin + (mmax - mmin) * usize_to_f32(i) / denom;
            hz_pts.push(mel_to_hz_slaney(mel));
        }

        let mut fft_freqs = Vec::with_capacity(n_freq);
        for k in 0..n_freq {
            fft_freqs.push(usize_to_f32(k) * sr_f / n_fft_f);
        }

        let mut mel_basis = vec![0.0_f32; cfg.n_mels * n_freq];
        let mut mel_extents = Vec::with_capacity(cfg.n_mels);
        for m in 0..cfg.n_mels {
            let lo = hz_pts[m];
            let md = hz_pts[m + 1];
            let hi = hz_pts[m + 2];
            let enorm = 2.0_f32 / (hi - lo).max(1e-9_f32);
            let row_start = m * n_freq;
            for (k, &f) in fft_freqs.iter().enumerate() {
                let up = (f - lo) / (md - lo).max(1e-9_f32);
                let down = (hi - f) / (hi - md).max(1e-9_f32);
                let w = up.min(down).max(0.0_f32);
                mel_basis[row_start + k] = w * enorm;
            }
            let row = &mel_basis[row_start..row_start + n_freq];
            let extent = match (
                row.iter().position(|&w| w != 0.0),
                row.iter().rposition(|&w| w != 0.0),
            ) {
                (Some(first), Some(last)) => (first, last + 1),
                _ => (0, 0),
            };
            mel_extents.push(extent);
        }

        Ok(Self {
            cfg,
            n_freq,
            hann,
            dft_real,
            dft_imag,
            fft_bitrev,
            fft_cos,
            fft_sin,
            mel_basis,
            mel_extents,
        })
    }

    /// Extract a `[n_mels, n_frames]` channel-major (`m * n_frames + t`) log-mel spectrogram.
    ///
    /// # Errors
    /// Rejects waveforms shorter than `(n_fft - hop) / 2 + 1` or containing non-finite samples.
    pub fn extract(&self, audio: &[f32]) -> io::Result<(Vec<f32>, usize)> {
        self.extract_parallel(audio, 1)
    }

    /// Extract a `[n_mels, n_frames]` log-mel spectrogram using up to `threads` frame workers
    /// with 0-ULP bit parity against [`Self::extract`].
    ///
    /// # Errors
    /// Rejects waveforms shorter than `(n_fft - hop) / 2 + 1` or containing non-finite samples.
    pub fn extract_parallel(&self, audio: &[f32], threads: usize) -> io::Result<(Vec<f32>, usize)> {
        let pad = (self.cfg.n_fft - self.cfg.hop) / 2;
        if audio.len() <= pad || !audio.iter().all(|s| s.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Reference audio is too short or non-finite for mel extraction.",
            ));
        }
        let padded = reflect_pad_1d(audio, pad, pad)?;
        if padded.len() < self.cfg.n_fft {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Padded reference audio is shorter than n_fft.",
            ));
        }
        let n_frames = (padded.len() - self.cfg.n_fft) / self.cfg.hop + 1;
        let frame_work = if self.use_fft() {
            self.cfg.n_fft * 12
        } else {
            self.n_freq.saturating_mul(self.cfg.n_fft)
        };
        let work = n_frames.saturating_mul(frame_work);
        let workers = if n_frames < 4 || work < 65_536 {
            1
        } else {
            threads.clamp(1, 18)
        };
        let mut log_mel = vec![0.0_f32; self.cfg.n_mels * n_frames];
        let level = Level::new();
        let half_n = if self.use_fft() {
            self.cfg.n_fft / 2
        } else {
            0
        };

        let make_scratch = || {
            (
                vec![0.0_f32; self.cfg.n_fft],
                vec![0.0_f32; half_n],
                vec![0.0_f32; half_n],
                vec![0.0_f32; self.n_freq],
            )
        };
        if workers <= 1 {
            let (mut windowed, mut fft_re, mut fft_im, mut mag_frame) = make_scratch();
            dispatch!(level, simd => {
                for t in 0..n_frames {
                    let start = t * self.cfg.hop;
                    self.compute_log_mel_frame_simd(
                        simd,
                        &padded[start..start + self.cfg.n_fft],
                        &mut windowed,
                        &mut fft_re,
                        &mut fft_im,
                        &mut mag_frame,
                        |m, val| log_mel[m * n_frames + t] = val,
                    );
                }
            });
            return Ok((log_mel, n_frames));
        }

        let mut frame_major = vec![0.0_f32; n_frames * self.cfg.n_mels];
        row_workers::run_aligned(
            &mut frame_major,
            workers,
            self.cfg.n_mels,
            &|offset, slice| {
                let first_frame = offset / self.cfg.n_mels;
                let (mut windowed, mut fft_re, mut fft_im, mut mag_frame) = make_scratch();
                dispatch!(level, simd => {
                    for (local_t, frame_out) in slice.chunks_exact_mut(self.cfg.n_mels).enumerate() {
                        let start = (first_frame + local_t) * self.cfg.hop;
                        self.compute_log_mel_frame_simd(
                            simd,
                            &padded[start..start + self.cfg.n_fft],
                            &mut windowed,
                            &mut fft_re,
                            &mut fft_im,
                            &mut mag_frame,
                            |m, val| frame_out[m] = val,
                        );
                    }
                });
            },
        );
        for (t, frame_row) in frame_major.chunks_exact(self.cfg.n_mels).enumerate() {
            for (m, &value) in frame_row.iter().enumerate() {
                log_mel[m * n_frames + t] = value;
            }
        }
        Ok((log_mel, n_frames))
    }

    #[expect(
        clippy::inline_always,
        clippy::too_many_arguments,
        reason = "fearless_simd dispatch! requires inline(always) to inherit target_feature attributes."
    )]
    #[inline(always)]
    fn compute_log_mel_frame_simd<S: Simd>(
        &self,
        simd: S,
        frame: &[f32],
        windowed: &mut [f32],
        fft_re: &mut [f32],
        fft_im: &mut [f32],
        mag_frame: &mut [f32],
        mut write_bin: impl FnMut(usize, f32),
    ) {
        window_frame_simd(simd, frame, &self.hann, windowed);
        if self.use_fft() {
            self.real_fft_magnitudes(windowed, fft_re, fft_im, mag_frame);
        } else {
            for (k, mag_slot) in mag_frame.iter_mut().enumerate() {
                let re_row = &self.dft_real[k * self.cfg.n_fft..(k + 1) * self.cfg.n_fft];
                let im_row = &self.dft_imag[k * self.cfg.n_fft..(k + 1) * self.cfg.n_fft];
                let (re, im) = f32_dot_pair_simd(simd, windowed, re_row, im_row);
                *mag_slot = re.mul_add(re, im.mul_add(im, 1e-9_f32)).sqrt();
            }
        }
        for (m, &(b_start, b_end)) in self.mel_extents.iter().enumerate() {
            let basis_row = &self.mel_basis[m * self.n_freq + b_start..m * self.n_freq + b_end];
            let sum = f32_dot_simd(simd, basis_row, &mag_frame[b_start..b_end]);
            write_bin(m, sum.max(1e-5_f32).ln());
        }
    }

    fn real_fft_magnitudes(
        &self,
        windowed: &[f32],
        z_re: &mut [f32],
        z_im: &mut [f32],
        mag_frame: &mut [f32],
    ) {
        let half_n = z_re.len();
        for (dst_idx, &src_idx) in self.fft_bitrev.iter().enumerate() {
            z_re[dst_idx] = windowed[2 * src_idx];
            z_im[dst_idx] = windowed[2 * src_idx + 1];
        }
        let mut size = 2_usize;
        while size <= half_n {
            let half = size / 2;
            let tw_step = half_n / half;
            for block in (0..half_n).step_by(size) {
                for j in 0..half {
                    let tw = j * tw_step;
                    let wr = self.fft_cos[tw];
                    let wi = self.fft_sin[tw];
                    let even = block + j;
                    let odd = even + half;
                    let xr = z_re[odd];
                    let xi = z_im[odd];
                    let tr = wr.mul_add(xr, -wi * xi);
                    let ti = wr.mul_add(xi, wi * xr);
                    let er = z_re[even];
                    let ei = z_im[even];
                    z_re[even] = er + tr;
                    z_im[even] = ei + ti;
                    z_re[odd] = er - tr;
                    z_im[odd] = ei - ti;
                }
            }
            size <<= 1;
        }
        let mag = |re: f32, im: f32| re.mul_add(re, im.mul_add(im, 1e-9_f32)).sqrt();
        mag_frame[0] = mag(z_re[0] + z_im[0], 0.0);
        mag_frame[half_n] = mag(z_re[0] - z_im[0], 0.0);
        let q = half_n / 2;
        mag_frame[q] = mag(z_re[q], -z_im[q]);
        for k in 1..q {
            let rk = z_re[k];
            let ik = z_im[k];
            let rm = z_re[half_n - k];
            let im = z_im[half_n - k];
            let e_re = 0.5_f32 * (rk + rm);
            let e_im = 0.5_f32 * (ik - im);
            let o_re = 0.5_f32 * (ik + im);
            let o_im = -0.5_f32 * (rk - rm);
            let wr = self.fft_cos[k];
            let wi = self.fft_sin[k];
            let tr = wr.mul_add(o_re, -wi * o_im);
            let ti = wr.mul_add(o_im, wi * o_re);
            mag_frame[k] = mag(e_re + tr, e_im + ti);
            mag_frame[half_n - k] = mag(e_re - tr, ti - e_im);
        }
    }
}

fn fill_reflect_padded(src: &[f32], left: usize, right: usize, dst: &mut [f32]) {
    let len = src.len();
    for i in 0..left {
        dst[i] = src[left - i];
    }
    dst[left..left + len].copy_from_slice(src);
    for i in 0..right {
        dst[left + len + i] = src[len - 2 - i];
    }
}

/// 1D reflect padding matching `torch.nn.functional.pad(mode="reflect")`.
pub fn reflect_pad_1d(input: &[f32], left: usize, right: usize) -> io::Result<Vec<f32>> {
    let len = input.len();
    if len == 0 || (left > 0 && len <= left) || (right > 0 && len <= right) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Reflect padding width must be strictly less than signal length.",
        ));
    }
    let mut out = vec![0.0_f32; left + len + right];
    fill_reflect_padded(input, left, right, &mut out);
    Ok(out)
}

/// Reflect-padded `Conv1d(padding="same", mode="reflect")` weight and bias.
///
/// Weights are stored in `[out_c, in_c, kernel]` row-major order.
#[derive(Clone, Debug)]
pub struct TdnnConv1d {
    /// Output channels.
    pub out_c: usize,
    /// Input channels.
    pub in_c: usize,
    /// Odd kernel width (`1`, `3`, or `5`).
    pub kernel: usize,
    /// Dilation factor (`1`, `2`, `3`, or `4`).
    pub dilation: usize,
    /// Flattened weights `[out_c * in_c * kernel]`.
    pub weight: Vec<f32>,
    /// Bias vector `[out_c]`.
    pub bias: Vec<f32>,
}

impl TdnnConv1d {
    /// Execute reflect-padded same-length 1D convolution on `[in_c, frames]` channel-major input.
    pub fn forward(&self, input: &[f32], frames: usize, apply_relu: bool) -> io::Result<Vec<f32>> {
        self.forward_parallel(input, frames, apply_relu, 1)
    }

    /// Execute reflect-padded same-length 1D convolution across up to `threads` output-channel
    /// workers with 0-ULP bit parity against [`Self::forward`].
    pub fn forward_parallel(
        &self,
        input: &[f32],
        frames: usize,
        apply_relu: bool,
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        if input.len() != self.in_c * frames
            || self.weight.len() != self.out_c * self.in_c * self.kernel
            || self.bias.len() != self.out_c
            || self.kernel.is_multiple_of(2)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid TDNN Conv1d dimensions.",
            ));
        }
        let pad = ((self.kernel - 1) * self.dilation) / 2;
        let padded_frames = frames + 2 * pad;
        let owned_padded;
        let padded: &[f32] = if pad == 0 {
            input
        } else {
            if frames == 0 || pad >= frames {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Reflect padding width must be strictly less than signal length.",
                ));
            }
            let mut buf = vec![0.0_f32; self.in_c * padded_frames];
            let pad_workers = if self.in_c < 2 || buf.len() < 65_536 {
                1
            } else {
                threads.clamp(1, 18)
            };
            row_workers::run_aligned(&mut buf, pad_workers, padded_frames, &|offset, slice| {
                let first_c = offset / padded_frames;
                for (local_c, dst) in slice.chunks_exact_mut(padded_frames).enumerate() {
                    let c = first_c + local_c;
                    fill_reflect_padded(&input[c * frames..(c + 1) * frames], pad, pad, dst);
                }
            });
            owned_padded = buf;
            &owned_padded
        };

        let mut out = vec![0.0_f32; self.out_c * frames];
        if self.out_c == 0 || frames == 0 {
            return Ok(out);
        }
        let work = self
            .out_c
            .saturating_mul(self.in_c)
            .saturating_mul(self.kernel)
            .saturating_mul(frames);
        let workers = if self.out_c <= 1 || work < 65_536 {
            1
        } else {
            threads.clamp(1, 18)
        };
        let level = Level::new();
        let align = if self.out_c >= 4 { 4 * frames } else { frames };
        let w_stride = self.in_c * self.kernel;
        row_workers::run_aligned(&mut out, workers, align, &|offset, slice| {
            let first_oc = offset / frames;
            let oc_count = slice.len() / frames;
            let oc_quads = oc_count / 4;
            dispatch!(level, simd => {
                for q_oc in 0..oc_quads {
                    let oc = first_oc + q_oc * 4;
                    let w_quad = &self.weight[oc * w_stride..(oc + 4) * w_stride];
                    let bias = [
                        self.bias[oc],
                        self.bias[oc + 1],
                        self.bias[oc + 2],
                        self.bias[oc + 3],
                    ];
                    let out_quad = &mut slice[q_oc * 4 * frames..(q_oc + 1) * 4 * frames];
                    tdnn_4oc_row_simd(
                        simd,
                        padded,
                        padded_frames,
                        w_quad,
                        bias,
                        self.in_c,
                        self.kernel,
                        self.dilation,
                        frames,
                        apply_relu,
                        out_quad,
                    );
                }
                for local_oc in (oc_quads * 4)..oc_count {
                    let oc = first_oc + local_oc;
                    let w_row = &self.weight[oc * w_stride..(oc + 1) * w_stride];
                    let out_row = &mut slice[local_oc * frames..(local_oc + 1) * frames];
                    tdnn_1oc_row_simd(
                        simd,
                        padded,
                        padded_frames,
                        w_row,
                        self.bias[oc],
                        self.in_c,
                        self.kernel,
                        self.dilation,
                        frames,
                        apply_relu,
                        out_row,
                    );
                }
            });
        });
        Ok(out)
    }
}

/// Squeeze-and-Excitation channel attention block (`conv1 + ReLU -> conv2 + Sigmoid`).
#[derive(Clone, Debug)]
pub struct SqueezeExcitation {
    /// Reduction 1x1 conv (`channels -> se_channels`).
    pub conv1: TdnnConv1d,
    /// Expansion 1x1 conv (`se_channels -> channels`).
    pub conv2: TdnnConv1d,
}

impl SqueezeExcitation {
    /// Apply channel-wise Squeeze-and-Excitation scaling to `[channels, frames]`.
    pub fn forward(&self, input: &[f32], frames: usize) -> io::Result<Vec<f32>> {
        let channels = self.conv1.in_c;
        if frames == 0 || input.len() != channels * frames || self.conv2.out_c != channels {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid Squeeze-and-Excitation input dimensions.",
            ));
        }
        let inv_t = 1.0_f32 / usize_to_f32(frames);
        let mut mean = vec![0.0_f32; channels];
        for c in 0..channels {
            let row = &input[c * frames..(c + 1) * frames];
            let sum: f32 = row.iter().copied().sum();
            mean[c] = sum * inv_t;
        }
        let h = self.conv1.forward(&mean, 1, true)?;
        let gate_logits = self.conv2.forward(&h, 1, false)?;
        let mut out = input.to_vec();
        for c in 0..channels {
            let gate = 1.0_f32 / (1.0_f32 + (-gate_logits[c]).exp());
            for val in &mut out[c * frames..(c + 1) * frames] {
                *val *= gate;
            }
        }
        Ok(out)
    }
}

/// SE-Res2Net block (`tdnn1 -> 8-branch dilated Res2Net -> tdnn2 -> SE + residual`).
#[derive(Clone, Debug)]
pub struct SeRes2NetBlock {
    /// Initial 1x1 TDNN.
    pub tdnn1: TdnnConv1d,
    /// `scale - 1` (`7`) dilated `k = 3` TDNN branches, each over `channels / scale` channels.
    pub branches: Vec<TdnnConv1d>,
    /// Output 1x1 TDNN.
    pub tdnn2: TdnnConv1d,
    /// Squeeze-and-Excitation gate.
    pub se: SqueezeExcitation,
    /// `Res2Net` scale factor (`8`).
    pub scale: usize,
}

impl SeRes2NetBlock {
    /// Run one SE-Res2Net block on `[channels, frames]`.
    pub fn forward(&self, input: &[f32], frames: usize) -> io::Result<Vec<f32>> {
        self.forward_parallel(input, frames, 1)
    }

    /// Run one SE-Res2Net block across up to `threads` channel workers.
    pub fn forward_parallel(
        &self,
        input: &[f32],
        frames: usize,
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        let channels = self.tdnn1.in_c;
        if self.scale < 2
            || !channels.is_multiple_of(self.scale)
            || self.branches.len() != self.scale - 1
            || input.len() != channels * frames
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid SE-Res2Net block configuration.",
            ));
        }
        let h = self.tdnn1.forward_parallel(input, frames, true, threads)?;
        let chunk_c = channels / self.scale;
        let chunk_len = chunk_c * frames;
        let mut res2net_out = vec![0.0_f32; channels * frames];
        res2net_out[..chunk_len].copy_from_slice(&h[..chunk_len]);

        let mut prev = Vec::new();
        for i in 1..self.scale {
            let slice = &h[i * chunk_len..(i + 1) * chunk_len];
            let branch_in: &[f32] = if i == 1 {
                slice
            } else {
                for (dst, &src) in prev.iter_mut().zip(slice.iter()) {
                    *dst += src;
                }
                &prev
            };
            let y = self.branches[i - 1].forward_parallel(branch_in, frames, true, threads)?;
            res2net_out[i * chunk_len..(i + 1) * chunk_len].copy_from_slice(&y);
            prev = y;
        }

        let h2 = self
            .tdnn2
            .forward_parallel(&res2net_out, frames, true, threads)?;
        let mut gated = self.se.forward(&h2, frames)?;
        for (dst, &res) in gated.iter_mut().zip(input.iter()) {
            *dst += res;
        }
        Ok(gated)
    }
}

/// Attentive Statistical Pooling (`[C, T] -> [2*C, 1]`).
#[derive(Clone, Debug)]
pub struct AttentiveStatPool {
    /// Attention TDNN (`3*C -> attn_dim`) with `ReLU`.
    pub tdnn: TdnnConv1d,
    /// Attention 1x1 projection (`attn_dim -> C`).
    pub conv: TdnnConv1d,
}

impl AttentiveStatPool {
    /// Pool `[channels, frames]` into `[2 * channels]` weighted mean and standard deviation.
    pub fn forward(&self, input: &[f32], channels: usize, frames: usize) -> io::Result<Vec<f32>> {
        self.forward_parallel(input, channels, frames, 1)
    }

    /// Pool `[channels, frames]` across up to `threads` workers.
    pub fn forward_parallel(
        &self,
        input: &[f32],
        channels: usize,
        frames: usize,
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        if frames == 0
            || input.len() != channels * frames
            || self.tdnn.in_c != channels * 3
            || self.conv.out_c != channels
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid AttentiveStatisticalPooling dimensions.",
            ));
        }
        let inv_t = 1.0_f32 / usize_to_f32(frames);
        let mut means = vec![0.0_f32; channels];
        let mut stds = vec![0.0_f32; channels];
        for c in 0..channels {
            let row = &input[c * frames..(c + 1) * frames];
            let mean = row.iter().copied().sum::<f32>() * inv_t;
            let var = row
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                * inv_t;
            means[c] = mean;
            stds[c] = (var + 1e-12_f32).sqrt();
        }

        // Upstream applies TDNN (with ReLU), then tanh, then 1x1 conv.
        // For k=1 TDNN over [input, broadcast(mean), broadcast(std)], fold the
        // constant mean/std contributions into per-channel bias instead of allocating
        // a 3*C*T concatenated activation buffer.
        let mut attn = if self.tdnn.kernel == 1
            && self.tdnn.weight.len() == self.tdnn.out_c * channels * 3
            && self.tdnn.bias.len() == self.tdnn.out_c
        {
            let hidden = self.tdnn.out_c;
            let mut folded_w = Vec::with_capacity(hidden * channels);
            let mut folded_b = Vec::with_capacity(hidden);
            for oc in 0..hidden {
                let row = &self.tdnn.weight[oc * channels * 3..(oc + 1) * channels * 3];
                let w_x = &row[..channels];
                let w_mean = &row[channels..2 * channels];
                let w_std = &row[2 * channels..];
                folded_w.extend_from_slice(w_x);
                let mut b = self.tdnn.bias[oc];
                for (&w, &m) in w_mean.iter().zip(&means) {
                    b = w.mul_add(m, b);
                }
                for (&w, &s) in w_std.iter().zip(&stds) {
                    b = w.mul_add(s, b);
                }
                folded_b.push(b);
            }
            let x_proj = TdnnConv1d {
                out_c: hidden,
                in_c: channels,
                kernel: 1,
                dilation: 1,
                weight: folded_w,
                bias: folded_b,
            };
            x_proj.forward_parallel(input, frames, true, threads)?
        } else {
            let mut cat = vec![0.0_f32; channels * 3 * frames];
            cat[..channels * frames].copy_from_slice(input);
            for c in 0..channels {
                cat[(channels + c) * frames..(channels + c + 1) * frames].fill(means[c]);
                cat[(2 * channels + c) * frames..(2 * channels + c + 1) * frames].fill(stds[c]);
            }
            self.tdnn.forward_parallel(&cat, frames, true, threads)?
        };
        for v in &mut attn {
            *v = v.tanh();
        }
        let scores = self.conv.forward_parallel(&attn, frames, false, threads)?;

        let mut stats = vec![0.0_f32; channels * 2];
        let workers = if channels <= 1 || channels.saturating_mul(frames) < 16_384 {
            1
        } else {
            threads.clamp(1, 18)
        };
        let pool_channel = |c: usize, s_row: &[f32], w_row: &mut [f32]| -> (f32, f32) {
            let x_row = &input[c * frames..(c + 1) * frames];
            let max_s = s_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut w_sum = 0.0_f32;
            for (dst, &s) in w_row.iter_mut().zip(s_row) {
                let w = (s - max_s).exp();
                *dst = w;
                w_sum += w;
            }
            let inv_w = 1.0_f32 / w_sum.max(1e-12_f32);
            for w in &mut *w_row {
                *w *= inv_w;
            }
            let mut w_mean = 0.0_f32;
            for (&w, &x) in w_row.iter().zip(x_row.iter()) {
                w_mean = w.mul_add(x, w_mean);
            }
            let mut w_var = 0.0_f32;
            for (&w, &x) in w_row.iter().zip(x_row.iter()) {
                let d = x - w_mean;
                w_var = w.mul_add(d * d, w_var);
            }
            (w_mean, (w_var + 1e-12_f32).sqrt())
        };
        if workers <= 1 {
            let mut w_row = vec![0.0_f32; frames];
            for (c, s_row) in scores.chunks_exact(frames).enumerate() {
                let (w_mean, w_std) = pool_channel(c, s_row, &mut w_row);
                stats[c] = w_mean;
                stats[channels + c] = w_std;
            }
        } else {
            let mut channel_means_vars = vec![0.0_f32; channels * 2];
            row_workers::run_aligned(
                &mut channel_means_vars,
                workers,
                2,
                &|stat_off, out_pairs| {
                    let first_c = stat_off / 2;
                    let c_count = out_pairs.len() / 2;
                    let sc = &scores[first_c * frames..(first_c + c_count) * frames];
                    let mut w_row = vec![0.0_f32; frames];
                    for (local_c, s_row) in sc.chunks_exact(frames).enumerate() {
                        let c = first_c + local_c;
                        let (w_mean, w_std) = pool_channel(c, s_row, &mut w_row);
                        out_pairs[local_c * 2] = w_mean;
                        out_pairs[local_c * 2 + 1] = w_std;
                    }
                },
            );
            for c in 0..channels {
                stats[c] = channel_means_vars[c * 2];
                stats[channels + c] = channel_means_vars[c * 2 + 1];
            }
        }
        Ok(stats)
    }
}

/// Complete pure-Rust ECAPA-TDNN speaker encoder.
#[derive(Clone, Debug)]
pub struct EcapaSpeakerEncoder {
    /// Mel spectrogram frontend.
    pub mel: MelSpectrogramExtractor,
    /// Initial `k = 5` TDNN (`128 -> 512`).
    pub conv0: TdnnConv1d,
    /// Three SE-Res2Net blocks at dilations `2, 3, 4`.
    pub blocks: [SeRes2NetBlock; 3],
    /// Multi-layer feature aggregation `k = 1` TDNN (`1536 -> 1536`).
    pub mfa: TdnnConv1d,
    /// Attentive statistical pooling (`1536 -> 3072`).
    pub asp: AttentiveStatPool,
    /// Final 1x1 projection (`3072 -> 2048`).
    pub fc: TdnnConv1d,
}

impl EcapaSpeakerEncoder {
    /// Extract the 2048-dimensional speaker x-vector from a 24 kHz mono waveform.
    pub fn extract_xvector(&self, audio_24k: &[f32]) -> io::Result<Vec<f32>> {
        self.extract_xvector_parallel(audio_24k, 1)
    }

    /// Extract the 2048-dimensional speaker x-vector using up to `threads` workers
    /// with 0-ULP bit parity against [`Self::extract_xvector`].
    pub fn extract_xvector_parallel(
        &self,
        audio_24k: &[f32],
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        let (mel, frames) = self.mel.extract_parallel(audio_24k, threads)?;
        let h0 = self.conv0.forward_parallel(&mel, frames, true, threads)?;
        let b1 = self.blocks[0].forward_parallel(&h0, frames, threads)?;
        let b2 = self.blocks[1].forward_parallel(&b1, frames, threads)?;
        let b3 = self.blocks[2].forward_parallel(&b2, frames, threads)?;

        let hidden = self.conv0.out_c;
        let mut cat = Vec::with_capacity(hidden * 3 * frames);
        cat.extend_from_slice(&b1);
        cat.extend_from_slice(&b2);
        cat.extend_from_slice(&b3);

        let mfa_out = self.mfa.forward_parallel(&cat, frames, true, threads)?;
        let pooled = self
            .asp
            .forward_parallel(&mfa_out, hidden * 3, frames, threads)?;
        self.fc.forward_parallel(&pooled, 1, false, threads)
    }
}

#[cfg(test)]
mod independent_tests;

#[cfg(test)]
mod tests {
    use super::{
        AttentiveStatPool, AudioMelConfig, EcapaSpeakerEncoder, MelSpectrogramExtractor,
        SeRes2NetBlock, SqueezeExcitation, TdnnConv1d, hz_to_mel_slaney, mel_to_hz_slaney,
        reflect_pad_1d,
    };

    #[test]
    fn slaney_mel_round_trips_across_linear_and_log_regions() {
        for hz in [0.0_f32, 250.0, 999.5, 1000.0, 2400.0, 8000.0, 12_000.0] {
            let mel = hz_to_mel_slaney(hz);
            let round = mel_to_hz_slaney(mel);
            assert!((round - hz).abs() < 1e-2, "hz={hz} round={round}");
        }
    }

    #[test]
    fn reflect_padding_and_mel_extractor_produce_finite_frames() -> std::io::Result<()> {
        let padded = reflect_pad_1d(&[1.0, 2.0, 3.0, 4.0], 2, 2)?;
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);

        let cfg = AudioMelConfig {
            sample_rate: 24_000,
            n_fft: 64,
            hop: 16,
            n_mels: 8,
            fmin: 0.0,
            fmax: 12_000.0,
        };
        let extractor = MelSpectrogramExtractor::new(cfg)?;
        let audio: Vec<f32> = (0..128)
            .map(|i| (f32::from(u8::try_from(i).unwrap_or(0)) * 0.1).sin())
            .collect();
        let (mel, frames) = extractor.extract(&audio)?;
        assert_eq!(frames, 8);
        assert_eq!(mel.len(), 8 * 8);
        assert!(mel.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Constructs synthetic TDNN, SE-Res2Net, MFA, ASP, and FC layers in one end-to-end test."
    )]
    fn ecapa_tdnn_speaker_encoder_extracts_finite_xvector() -> std::io::Result<()> {
        let cfg = AudioMelConfig {
            sample_rate: 24_000,
            n_fft: 32,
            hop: 8,
            n_mels: 4,
            fmin: 0.0,
            fmax: 12_000.0,
        };
        let mel = MelSpectrogramExtractor::new(cfg)?;
        let hidden = 4_usize;
        let scale = 2_usize;
        let make_block = |dilation: usize| SeRes2NetBlock {
            tdnn1: TdnnConv1d {
                out_c: hidden,
                in_c: hidden,
                kernel: 1,
                dilation: 1,
                weight: vec![0.1; hidden * hidden],
                bias: vec![0.01; hidden],
            },
            branches: vec![TdnnConv1d {
                out_c: hidden / scale,
                in_c: hidden / scale,
                kernel: 3,
                dilation,
                weight: vec![0.05; (hidden / scale) * (hidden / scale) * 3],
                bias: vec![0.0; hidden / scale],
            }],
            tdnn2: TdnnConv1d {
                out_c: hidden,
                in_c: hidden,
                kernel: 1,
                dilation: 1,
                weight: vec![0.1; hidden * hidden],
                bias: vec![0.0; hidden],
            },
            se: SqueezeExcitation {
                conv1: TdnnConv1d {
                    out_c: 2,
                    in_c: hidden,
                    kernel: 1,
                    dilation: 1,
                    weight: vec![0.1; 2 * hidden],
                    bias: vec![0.0; 2],
                },
                conv2: TdnnConv1d {
                    out_c: hidden,
                    in_c: 2,
                    kernel: 1,
                    dilation: 1,
                    weight: vec![0.1; hidden * 2],
                    bias: vec![0.0; hidden],
                },
            },
            scale,
        };
        let mfa_c = hidden * 3;
        let encoder = EcapaSpeakerEncoder {
            mel,
            conv0: TdnnConv1d {
                out_c: hidden,
                in_c: 4,
                kernel: 5,
                dilation: 1,
                weight: vec![0.05; hidden * 4 * 5],
                bias: vec![0.01; hidden],
            },
            blocks: [make_block(2), make_block(3), make_block(4)],
            mfa: TdnnConv1d {
                out_c: mfa_c,
                in_c: mfa_c,
                kernel: 1,
                dilation: 1,
                weight: vec![0.05; mfa_c * mfa_c],
                bias: vec![0.0; mfa_c],
            },
            asp: AttentiveStatPool {
                tdnn: TdnnConv1d {
                    out_c: 4,
                    in_c: mfa_c * 3,
                    kernel: 1,
                    dilation: 1,
                    weight: vec![0.02; 4 * mfa_c * 3],
                    bias: vec![0.0; 4],
                },
                conv: TdnnConv1d {
                    out_c: mfa_c,
                    in_c: 4,
                    kernel: 1,
                    dilation: 1,
                    weight: vec![0.02; mfa_c * 4],
                    bias: vec![0.0; mfa_c],
                },
            },
            fc: TdnnConv1d {
                out_c: 8,
                in_c: mfa_c * 2,
                kernel: 1,
                dilation: 1,
                weight: vec![0.05; 8 * mfa_c * 2],
                bias: vec![0.01; 8],
            },
        };
        let audio: Vec<f32> = (0..64)
            .map(|i| (f32::from(u8::try_from(i).unwrap_or(0)) * 0.2).cos())
            .collect();
        let xvec = encoder.extract_xvector(&audio)?;
        assert_eq!(xvec.len(), 8);
        assert!(xvec.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn speaker_encoder_parallel_matches_sequential_bits() -> std::io::Result<()> {
        let cfg = AudioMelConfig {
            sample_rate: 24_000,
            n_fft: 64,
            hop: 8,
            n_mels: 16,
            fmin: 0.0,
            fmax: 12_000.0,
        };
        let mel = MelSpectrogramExtractor::new(cfg)?;
        let audio: Vec<f32> = (0..1024)
            .map(|i| (f32::from(u16::try_from(i % 73).unwrap_or(0)) * 0.05).sin())
            .collect();
        let (seq_mel, frames) = mel.extract(&audio)?;
        assert!(frames >= 64);

        let hidden = 24_usize;
        let scale = 3_usize;
        let make_block = |dilation: usize| SeRes2NetBlock {
            tdnn1: TdnnConv1d {
                out_c: hidden,
                in_c: hidden,
                kernel: 1,
                dilation: 1,
                weight: (0..hidden * hidden)
                    .map(|i| (f32::from(u8::try_from(i % 13).unwrap_or(0)) - 6.0) * 0.01)
                    .collect(),
                bias: vec![0.01; hidden],
            },
            branches: (0..(scale - 1))
                .map(|b| TdnnConv1d {
                    out_c: hidden / scale,
                    in_c: hidden / scale,
                    kernel: 3,
                    dilation,
                    weight: (0..(hidden / scale) * (hidden / scale) * 3)
                        .map(|i| (f32::from(u8::try_from((i + b) % 11).unwrap_or(0)) - 5.0) * 0.01)
                        .collect(),
                    bias: vec![0.0; hidden / scale],
                })
                .collect(),
            tdnn2: TdnnConv1d {
                out_c: hidden,
                in_c: hidden,
                kernel: 1,
                dilation: 1,
                weight: vec![0.02; hidden * hidden],
                bias: vec![0.0; hidden],
            },
            se: SqueezeExcitation {
                conv1: TdnnConv1d {
                    out_c: 6,
                    in_c: hidden,
                    kernel: 1,
                    dilation: 1,
                    weight: vec![0.05; 6 * hidden],
                    bias: vec![0.0; 6],
                },
                conv2: TdnnConv1d {
                    out_c: hidden,
                    in_c: 6,
                    kernel: 1,
                    dilation: 1,
                    weight: vec![0.05; hidden * 6],
                    bias: vec![0.0; hidden],
                },
            },
            scale,
        };
        let mfa_c = hidden * 3;
        let encoder = EcapaSpeakerEncoder {
            mel,
            conv0: TdnnConv1d {
                out_c: hidden,
                in_c: 16,
                kernel: 5,
                dilation: 1,
                weight: (0..hidden * 16 * 5)
                    .map(|i| (f32::from(u16::try_from(i % 19).unwrap_or(0)) - 9.0) * 0.01)
                    .collect(),
                bias: vec![0.01; hidden],
            },
            blocks: [make_block(2), make_block(3), make_block(4)],
            mfa: TdnnConv1d {
                out_c: mfa_c,
                in_c: mfa_c,
                kernel: 1,
                dilation: 1,
                weight: vec![0.01; mfa_c * mfa_c],
                bias: vec![0.0; mfa_c],
            },
            asp: AttentiveStatPool {
                tdnn: TdnnConv1d {
                    out_c: 8,
                    in_c: mfa_c * 3,
                    kernel: 1,
                    dilation: 1,
                    weight: vec![0.01; 8 * mfa_c * 3],
                    bias: vec![0.0; 8],
                },
                conv: TdnnConv1d {
                    out_c: mfa_c,
                    in_c: 8,
                    kernel: 1,
                    dilation: 1,
                    weight: vec![0.01; mfa_c * 8],
                    bias: vec![0.0; mfa_c],
                },
            },
            fc: TdnnConv1d {
                out_c: 16,
                in_c: mfa_c * 2,
                kernel: 1,
                dilation: 1,
                weight: vec![0.02; 16 * mfa_c * 2],
                bias: vec![0.01; 16],
            },
        };
        let seq_xvec = encoder.extract_xvector(&audio)?;
        for threads in [2_usize, 4, 8] {
            let (par_mel, par_frames) = encoder.mel.extract_parallel(&audio, threads)?;
            assert_eq!(par_frames, frames);
            assert_eq!(
                par_mel.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                seq_mel.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "mel threads {threads}"
            );
            let par_xvec = encoder.extract_xvector_parallel(&audio, threads)?;
            assert_eq!(
                par_xvec.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                seq_xvec.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "xvec threads {threads}"
            );
        }
        Ok(())
    }

    #[test]
    fn mel_real_fft_matches_direct_dft_and_omits_matrix_for_1024() -> std::io::Result<()> {
        for n_fft in [64_usize, 1024] {
            let extractor = MelSpectrogramExtractor::new(AudioMelConfig {
                sample_rate: 24_000,
                n_fft,
                hop: 256.min(n_fft / 2),
                n_mels: 32,
                fmin: 0.0,
                fmax: 12_000.0,
            })?;
            assert!(extractor.dft_real.is_empty());
            assert!(extractor.dft_imag.is_empty());

            let frame: Vec<f32> = (0..n_fft)
                .map(|i| {
                    let t = f32::from(u16::try_from(i % 137).unwrap_or(0));
                    (t * 0.17).sin() * 0.6 + (t * 0.43).cos() * 0.35
                })
                .collect();
            let mut windowed = vec![0.0_f32; n_fft];
            for (dst, (&s, &w)) in windowed.iter_mut().zip(frame.iter().zip(&extractor.hann)) {
                *dst = s * w;
            }
            let mut fft_re = vec![0.0_f32; n_fft / 2];
            let mut fft_im = vec![0.0_f32; n_fft / 2];
            let mut mag = vec![0.0_f32; extractor.n_freq];
            extractor.real_fft_magnitudes(&windowed, &mut fft_re, &mut fft_im, &mut mag);

            let n_fft_f64 = f64::from(u32::try_from(n_fft).unwrap_or(0));
            let two_pi = 2.0_f64 * core::f64::consts::PI;
            let twiddle: Vec<(f64, f64)> = (0..n_fft)
                .map(|m| {
                    let th = two_pi * f64::from(u32::try_from(m).unwrap_or(0)) / n_fft_f64;
                    th.sin_cos()
                })
                .collect();
            for (k, &got_mag) in mag.iter().enumerate() {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (n, &w) in windowed.iter().enumerate() {
                    let (s, c) = twiddle[(k * n) & (n_fft - 1)];
                    re += f64::from(w) * c;
                    im -= f64::from(w) * s;
                }
                let ref_mag = (re * re + im * im + 1e-9_f64).sqrt();
                assert!(
                    (f64::from(got_mag) - ref_mag).abs() <= 1e-4,
                    "n_fft={n_fft} k={k} got={got_mag} ref={ref_mag}"
                );
            }
        }
        Ok(())
    }
}
