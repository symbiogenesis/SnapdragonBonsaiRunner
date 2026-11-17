//! ITU-R BS.1770-4 / EBU R128 loudness, EBU Tech 3342 LRA, 4x true-peak metering,
//! and attested loudness-matched listening copies.

use crate::{WavError, decode_pcm16le_into, mono_pcm16_data_checked, mono_pcm16_wav_bytes};
use std::{error::Error, fmt};

/// Minimum sample rate admitted by the K-weighting bilinear filter design.
pub const MIN_LOUDNESS_SAMPLE_RATE: u32 = 8_000;
/// Maximum sample rate admitted by the K-weighting bilinear filter design.
pub const MAX_LOUDNESS_SAMPLE_RATE: u32 = 192_000;

const BS1770_OFFSET_DB: f64 = -0.691;
const ABSOLUTE_GATE_LUFS: f64 = -70.0;
const INTEGRATED_RELATIVE_GATE_LU: f64 = -10.0;
const LRA_RELATIVE_GATE_LU: f64 = -20.0;
const LRA_LOW_PERCENTILE: f64 = 0.10;
const LRA_HIGH_PERCENTILE: f64 = 0.95;

// Stage 1: ITU-R BS.1770-4 pre-filter high-shelf prototype parameters.
const STAGE1_F0_HZ: f64 = 1681.974_450_955_533;
const STAGE1_GAIN_DB: f64 = 3.999_843_853_973_347;
const STAGE1_Q: f64 = 0.707_175_236_955_419_6;
const STAGE1_VB_EXPONENT: f64 = 0.499_666_774_154_541_6;

// Stage 2: ITU-R BS.1770-4 revised low-frequency B-weighting (RLB) high-pass parameters.
const STAGE2_F0_HZ: f64 = 38.135_470_876_024_44;
const STAGE2_Q: f64 = 0.500_327_037_323_877_3;

// 4x oversampling Kaiser-windowed sinc FIR interpolator: 12 taps per phase (48 taps total).
const TRUE_PEAK_HALF_TAPS: usize = 6;
const TRUE_PEAK_PHASE_TAPS: usize = TRUE_PEAK_HALF_TAPS * 2;
const TRUE_PEAK_KAISER_BETA: f64 = 7.0;

/// Loudness, loudness-range, and peak measurements over a finite mono audio buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoudnessObservation {
    /// Gated integrated loudness in LUFS (`-70 LUFS` absolute, `-10 LU` relative), or `None` when
    /// below gate, silent, or shorter than a single 400 ms gating block.
    pub integrated_lufs: Option<f64>,
    /// Maximum ungated 400 ms momentary loudness in LUFS, or `None` for silence or clips under 400 ms.
    pub max_momentary_lufs: Option<f64>,
    /// Maximum ungated 3.0 s short-term loudness in LUFS, or `None` for silence or clips under 3.0 s.
    pub max_short_term_lufs: Option<f64>,
    /// EBU Tech 3342 loudness range (`P95 - P10` in LU) over 3.0 s / 100 ms blocks with `-70 LUFS`
    /// absolute and `-20 LU` relative gating, or `None` if no blocks survive gating.
    pub loudness_range_lu: Option<f64>,
    /// Maximum sample magnitude in dBFS, or `None` for all-zero digital silence.
    pub sample_peak_dbfs: Option<f64>,
    /// Maximum 4x polyphase Kaiser-sinc oversampled true peak in dBTP, or `None` for digital silence.
    pub true_peak_dbtp: Option<f64>,
}

/// Separate canonical mono PCM16 listening copy with attested gain and ceiling metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct LoudnessMatchedListeningWav {
    /// Loudness observation measured on the unmodified input samples prior to gain application.
    pub observation: LoudnessObservation,
    /// Canonical 44-byte-header mono PCM16 WAV bytes with `applied_gain_db` applied.
    pub wav_bytes: Vec<u8>,
    /// Exact gain in dB applied to produce `wav_bytes`.
    pub applied_gain_db: f64,
    /// `true` when `target_lufs - integrated_lufs` was reduced to respect `true_peak_ceiling_dbtp`.
    pub true_peak_clamped: bool,
}

/// Descriptive error returned by loudness metering and listening-copy generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoudnessError {
    /// Input audio slice is empty.
    EmptyInput,
    /// Sample rate is outside `8_000..=192_000 Hz`.
    UnsupportedSampleRate(u32),
    /// Input audio contains at least one `NaN` or infinite sample.
    NonFiniteSample,
    /// Target loudness in LUFS is non-finite.
    InvalidTargetLufs,
    /// True-peak ceiling in dBTP is non-finite.
    InvalidTruePeakCeiling,
    /// Input audio fell below the `-70 LUFS` integrated gate or was shorter than 400 ms.
    BelowLoudnessGate,
    /// Working buffer allocation failed.
    AllocationFailed,
    /// Underlying WAV validation, decoding, or framing failed.
    Wav(WavError),
}

impl From<WavError> for LoudnessError {
    fn from(error: WavError) -> Self {
        Self::Wav(error)
    }
}

impl fmt::Display for LoudnessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => formatter.write_str("audio input must contain at least one sample"),
            Self::UnsupportedSampleRate(rate) => write!(
                formatter,
                "sample rate {rate} Hz is outside the supported {MIN_LOUDNESS_SAMPLE_RATE}..={MAX_LOUDNESS_SAMPLE_RATE} Hz range"
            ),
            Self::NonFiniteSample => {
                formatter.write_str("audio input contains non-finite (NaN or infinite) samples")
            }
            Self::InvalidTargetLufs => {
                formatter.write_str("target loudness in LUFS must be finite")
            }
            Self::InvalidTruePeakCeiling => {
                formatter.write_str("true-peak ceiling in dBTP must be finite")
            }
            Self::BelowLoudnessGate => formatter
                .write_str("audio has no measurable integrated loudness above the -70 LUFS gate"),
            Self::AllocationFailed => {
                formatter.write_str("failed to allocate loudness analysis buffer")
            }
            Self::Wav(error) => error.fmt(formatter),
        }
    }
}

impl Error for LoudnessError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Wav(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn k_weight_stage1(sample_rate_hz: f64) -> Self {
        let k = (std::f64::consts::PI * STAGE1_F0_HZ / sample_rate_hz).tan();
        let vh = 10.0_f64.powf(STAGE1_GAIN_DB / 20.0);
        let vb = vh.powf(STAGE1_VB_EXPONENT);
        let k2 = k * k;
        let k_over_q = k / STAGE1_Q;
        let a0 = 1.0 + k_over_q + k2;
        Self {
            b0: (vh + vb * k_over_q + k2) / a0,
            b1: 2.0 * (k2 - vh) / a0,
            b2: (vh - vb * k_over_q + k2) / a0,
            a1: 2.0 * (k2 - 1.0) / a0,
            a2: (1.0 - k_over_q + k2) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn k_weight_stage2(sample_rate_hz: f64) -> Self {
        let k = (std::f64::consts::PI * STAGE2_F0_HZ / sample_rate_hz).tan();
        let k2 = k * k;
        let k_over_q = k / STAGE2_Q;
        let a0 = 1.0 + k_over_q + k2;
        // BS.1770-4 Stage 2 preserves unnormalized unity numerator [1, -2, 1] so the
        // -0.691 dB offset yields exact 0 dB net K-weighting alignment at 997 Hz / 48 kHz.
        Self {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: 2.0 * (k2 - 1.0) / a0,
            a2: (1.0 - k_over_q + k2) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

struct KWeightFilter {
    stage1: Biquad,
    stage2: Biquad,
}

impl KWeightFilter {
    fn new(sample_rate: u32) -> Self {
        let rate_hz = f64::from(sample_rate);
        Self {
            stage1: Biquad::k_weight_stage1(rate_hz),
            stage2: Biquad::k_weight_stage2(rate_hz),
        }
    }

    #[inline]
    fn process(&mut self, sample: f32) -> f64 {
        let s1 = self.stage1.process(f64::from(sample));
        self.stage2.process(s1)
    }
}

#[inline]
fn energy_to_lufs(mean_square: f64) -> Option<f64> {
    if mean_square > 0.0 && mean_square.is_finite() {
        let lufs = BS1770_OFFSET_DB + 10.0 * mean_square.log10();
        lufs.is_finite().then_some(lufs)
    } else {
        None
    }
}

#[inline]
fn amplitude_to_db(peak: f64) -> Option<f64> {
    if peak > 0.0 && peak.is_finite() {
        let db = 20.0 * peak.log10();
        db.is_finite().then_some(db)
    } else {
        None
    }
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x * 0.5;
    for k in 1_u16..=25 {
        let denom = f64::from(k);
        let ratio = half / denom;
        term *= ratio * ratio;
        sum += term;
    }
    sum
}

fn kaiser_sinc_phase_taps(mu: f64) -> [f64; TRUE_PEAK_PHASE_TAPS] {
    let mut taps = [0.0_f64; TRUE_PEAK_PHASE_TAPS];
    let radius = f64::from(u16::try_from(TRUE_PEAK_HALF_TAPS).unwrap_or(6));
    let i0_beta = bessel_i0(TRUE_PEAK_KAISER_BETA);
    let mut sum = 0.0;
    let offset_base = i32::try_from(TRUE_PEAK_HALF_TAPS).unwrap_or(6) - 1;
    for (idx, slot) in taps.iter_mut().enumerate() {
        let m = i32::try_from(idx).unwrap_or(0) - offset_base;
        let d = f64::from(m) - mu;
        let sinc = if d.abs() < 1.0e-12 {
            1.0
        } else {
            let pi_d = std::f64::consts::PI * d;
            pi_d.sin() / pi_d
        };
        let u = d / radius;
        let window = if u.abs() >= 1.0 {
            0.0
        } else {
            bessel_i0(TRUE_PEAK_KAISER_BETA * (1.0 - u * u).sqrt()) / i0_beta
        };
        let coeff = sinc * window;
        *slot = coeff;
        sum += coeff;
    }
    if sum != 0.0 {
        for slot in &mut taps {
            *slot /= sum;
        }
    }
    taps
}

fn measure_true_peak_linear(samples: &[f32], sample_peak: f64) -> f64 {
    if sample_peak <= 0.0 || samples.is_empty() {
        return 0.0;
    }
    let phases = [
        kaiser_sinc_phase_taps(0.25),
        kaiser_sinc_phase_taps(0.50),
        kaiser_sinc_phase_taps(0.75),
    ];
    let mut true_peak = sample_peak;
    let len = samples.len();
    let half_before = TRUE_PEAK_HALF_TAPS - 1;
    let half_after = TRUE_PEAK_HALF_TAPS;

    let interior_start = half_before.min(len);
    let interior_end = len.saturating_sub(half_after);

    let eval_boundary = |center: isize, true_peak: &mut f64| {
        for taps in &phases {
            let mut acc = 0.0_f64;
            for (k, &coeff) in taps.iter().enumerate() {
                let offset =
                    isize::try_from(k).unwrap_or(0) - isize::try_from(half_before).unwrap_or(5);
                let sample_idx = center + offset;
                if sample_idx >= 0
                    && let Ok(idx) = usize::try_from(sample_idx)
                    && let Some(&sample) = samples.get(idx)
                {
                    acc += f64::from(sample) * coeff;
                }
            }
            let mag = acc.abs();
            if mag > *true_peak {
                *true_peak = mag;
            }
        }
    };

    let lead_start = -isize::try_from(half_after).unwrap_or(6);
    let lead_end = isize::try_from(interior_start).unwrap_or(0);
    for center in lead_start..lead_end {
        eval_boundary(center, &mut true_peak);
    }

    for center in interior_start..interior_end {
        let window = &samples[center - half_before..=center + half_after];
        for taps in &phases {
            let mut acc = 0.0_f64;
            for (&sample, &coeff) in window.iter().zip(taps.iter()) {
                acc += f64::from(sample) * coeff;
            }
            let mag = acc.abs();
            if mag > true_peak {
                true_peak = mag;
            }
        }
    }

    let tail_start = isize::try_from(interior_end.max(interior_start)).unwrap_or(0);
    let tail_end = isize::try_from(len + half_before).unwrap_or(0);
    for center in tail_start..=tail_end {
        eval_boundary(center, &mut true_peak);
    }

    true_peak
}

struct WindowGeometry {
    step_samples: usize,
    momentary_samples: usize,
    short_term_samples: usize,
}

impl WindowGeometry {
    fn new(sample_rate: u32) -> Option<Self> {
        let rate = u64::from(sample_rate);
        let step = ((rate * 100 + 500) / 1_000).max(1);
        let momentary = ((rate * 400 + 500) / 1_000).max(1);
        let short_term = rate * 3;
        Some(Self {
            step_samples: usize::try_from(step).ok()?,
            momentary_samples: usize::try_from(momentary).ok()?,
            short_term_samples: usize::try_from(short_term).ok()?,
        })
    }

    const fn exact_step_grid(&self) -> bool {
        self.momentary_samples == self.step_samples * 4
            && self.short_term_samples == self.step_samples * 30
    }
}

fn sum_squares(values: &[f64]) -> f64 {
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let (chunks, tail) = values.as_chunks::<4>();
    for &[a, b, c, d] in chunks {
        s0 += a * a;
        s1 += b * b;
        s2 += c * c;
        s3 += d * d;
    }
    let mut total = (s0 + s1) + (s2 + s3);
    for &x in tail {
        total += x * x;
    }
    total
}

fn collect_block_energies_grid(
    samples: &[f32],
    sample_rate: u32,
    geometry: &WindowGeometry,
) -> Result<(Vec<f64>, Vec<f64>), LoudnessError> {
    let chunk_count = samples.len() / geometry.step_samples;
    if chunk_count < 4 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut step_energies = Vec::new();
    step_energies
        .try_reserve_exact(chunk_count)
        .map_err(|_| LoudnessError::AllocationFailed)?;

    let mut filter = KWeightFilter::new(sample_rate);
    for chunk in samples.chunks_exact(geometry.step_samples) {
        let mut s0 = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        let mut s3 = 0.0_f64;
        let (quads, tail) = chunk.as_chunks::<4>();
        for &[a, b, c, d] in quads {
            let y0 = filter.process(a);
            let y1 = filter.process(b);
            let y2 = filter.process(c);
            let y3 = filter.process(d);
            s0 += y0 * y0;
            s1 += y1 * y1;
            s2 += y2 * y2;
            s3 += y3 * y3;
        }
        let mut energy = (s0 + s1) + (s2 + s3);
        for &s in tail {
            let y = filter.process(s);
            energy += y * y;
        }
        step_energies.push(energy);
    }

    let inv_momentary = 1.0 / f64::from(u32::try_from(geometry.momentary_samples).unwrap_or(1));
    let momentary_count = chunk_count - 3;
    let mut momentary = Vec::new();
    momentary
        .try_reserve_exact(momentary_count)
        .map_err(|_| LoudnessError::AllocationFailed)?;
    for win in step_energies.windows(4) {
        momentary.push((win[0] + win[1] + win[2] + win[3]) * inv_momentary);
    }

    let mut short_term = Vec::new();
    if chunk_count >= 30 {
        let inv_short = 1.0 / f64::from(u32::try_from(geometry.short_term_samples).unwrap_or(1));
        let short_count = chunk_count - 29;
        short_term
            .try_reserve_exact(short_count)
            .map_err(|_| LoudnessError::AllocationFailed)?;
        for win in step_energies.windows(30) {
            let sum: f64 = win.iter().sum();
            short_term.push(sum * inv_short);
        }
    }

    Ok((momentary, short_term))
}

fn collect_block_energies_general(
    samples: &[f32],
    sample_rate: u32,
    geometry: &WindowGeometry,
) -> Result<(Vec<f64>, Vec<f64>), LoudnessError> {
    if samples.len() < geometry.momentary_samples {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut k_weighted = Vec::new();
    k_weighted
        .try_reserve_exact(samples.len())
        .map_err(|_| LoudnessError::AllocationFailed)?;
    let mut filter = KWeightFilter::new(sample_rate);
    for &sample in samples {
        k_weighted.push(filter.process(sample));
    }

    let momentary_count = (samples.len() - geometry.momentary_samples) / geometry.step_samples + 1;
    let inv_momentary = 1.0 / f64::from(u32::try_from(geometry.momentary_samples).unwrap_or(1));
    let mut momentary = Vec::new();
    momentary
        .try_reserve_exact(momentary_count)
        .map_err(|_| LoudnessError::AllocationFailed)?;
    for idx in 0..momentary_count {
        let start = idx * geometry.step_samples;
        let end = start + geometry.momentary_samples;
        momentary.push(sum_squares(&k_weighted[start..end]) * inv_momentary);
    }

    let mut short_term = Vec::new();
    if samples.len() >= geometry.short_term_samples {
        let short_count = (samples.len() - geometry.short_term_samples) / geometry.step_samples + 1;
        let inv_short = 1.0 / f64::from(u32::try_from(geometry.short_term_samples).unwrap_or(1));
        short_term
            .try_reserve_exact(short_count)
            .map_err(|_| LoudnessError::AllocationFailed)?;
        for idx in 0..short_count {
            let start = idx * geometry.step_samples;
            let end = start + geometry.short_term_samples;
            short_term.push(sum_squares(&k_weighted[start..end]) * inv_short);
        }
    }

    Ok((momentary, short_term))
}

fn compute_integrated_lufs(momentary_energies: &[f64]) -> (Option<f64>, Option<f64>) {
    let mut max_momentary: Option<f64> = None;
    let mut abs_sum = 0.0_f64;
    let mut abs_count = 0_usize;

    for &energy in momentary_energies {
        let Some(lufs) = energy_to_lufs(energy) else {
            continue;
        };
        max_momentary = Some(match max_momentary {
            Some(prev) => prev.max(lufs),
            None => lufs,
        });
        if lufs > ABSOLUTE_GATE_LUFS {
            abs_sum += energy;
            abs_count += 1;
        }
    }

    let Some(abs_count_u32) = u32::try_from(abs_count).ok().filter(|&n| n > 0) else {
        return (None, max_momentary);
    };
    let abs_mean = abs_sum / f64::from(abs_count_u32);
    let Some(abs_lufs) = energy_to_lufs(abs_mean) else {
        return (None, max_momentary);
    };
    let relative_gate = abs_lufs + INTEGRATED_RELATIVE_GATE_LU;

    let mut rel_sum = 0.0_f64;
    let mut rel_count = 0_usize;
    for &energy in momentary_energies {
        let Some(lufs) = energy_to_lufs(energy) else {
            continue;
        };
        if lufs > ABSOLUTE_GATE_LUFS && lufs > relative_gate {
            rel_sum += energy;
            rel_count += 1;
        }
    }

    let Some(rel_count_u32) = u32::try_from(rel_count).ok().filter(|&n| n > 0) else {
        return (None, max_momentary);
    };
    let integrated = energy_to_lufs(rel_sum / f64::from(rel_count_u32));
    (integrated, max_momentary)
}

fn interpolated_percentile(sorted: &[f64], percentile: f64) -> Option<f64> {
    let count = u32::try_from(sorted.len()).ok().filter(|&n| n > 0)?;
    if count == 1 {
        return Some(sorted[0]);
    }
    let n = f64::from(count);
    let max_idx = f64::from(count - 1);
    let pos = (percentile * n - 0.5).clamp(0.0, max_idx);
    let lower_f = pos.floor();
    let frac = pos - lower_f;
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Non-negative floored position is bounded by sorted.len() - 1."
    )]
    let lower = (lower_f as usize).min(sorted.len() - 1);
    let upper = (lower + 1).min(sorted.len() - 1);
    Some(sorted[lower] + frac * (sorted[upper] - sorted[lower]))
}

fn compute_short_term_and_lra(short_term_energies: &[f64]) -> (Option<f64>, Option<f64>) {
    let mut max_short_term: Option<f64> = None;
    let mut abs_sum = 0.0_f64;
    let mut abs_count = 0_usize;

    for &energy in short_term_energies {
        let Some(lufs) = energy_to_lufs(energy) else {
            continue;
        };
        max_short_term = Some(match max_short_term {
            Some(prev) => prev.max(lufs),
            None => lufs,
        });
        if lufs > ABSOLUTE_GATE_LUFS {
            abs_sum += energy;
            abs_count += 1;
        }
    }

    let Some(abs_count_u32) = u32::try_from(abs_count).ok().filter(|&n| n > 0) else {
        return (max_short_term, None);
    };
    let abs_mean = abs_sum / f64::from(abs_count_u32);
    let Some(abs_lufs) = energy_to_lufs(abs_mean) else {
        return (max_short_term, None);
    };
    let relative_gate = abs_lufs + LRA_RELATIVE_GATE_LU;

    let mut gated_lufs = Vec::new();
    if gated_lufs.try_reserve(abs_count).is_err() {
        return (max_short_term, None);
    }
    for &energy in short_term_energies {
        if let Some(lufs) = energy_to_lufs(energy)
            && lufs > ABSOLUTE_GATE_LUFS
            && lufs >= relative_gate
        {
            gated_lufs.push(lufs);
        }
    }

    if gated_lufs.is_empty() {
        return (max_short_term, None);
    }
    gated_lufs.sort_by(f64::total_cmp);
    let lra = match (
        interpolated_percentile(&gated_lufs, LRA_LOW_PERCENTILE),
        interpolated_percentile(&gated_lufs, LRA_HIGH_PERCENTILE),
    ) {
        (Some(low), Some(high)) => Some((high - low).max(0.0)),
        _ => None,
    };
    (max_short_term, lra)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Input is explicitly clamped to finite [-1.0, 1.0] prior to narrowing to f32."
)]
fn clamp_unit_f64_to_f32(value: f64) -> f32 {
    value.clamp(-1.0, 1.0) as f32
}

/// Measures ITU-R BS.1770-4 / EBU R128 loudness, EBU Tech 3342 LRA, sample peak,
/// and 4x oversampled true peak over finite mono `f32` audio samples.
///
/// # Errors
/// Returns [`LoudnessError::EmptyInput`] if `samples` is empty,
/// [`LoudnessError::UnsupportedSampleRate`] if `sample_rate` is outside `8_000..=192_000 Hz`,
/// [`LoudnessError::NonFiniteSample`] if any sample is `NaN` or infinite, or
/// [`LoudnessError::AllocationFailed`] if internal scratch allocation fails.
pub fn measure_loudness_f32(
    samples: &[f32],
    sample_rate: u32,
) -> Result<LoudnessObservation, LoudnessError> {
    if samples.is_empty() {
        return Err(LoudnessError::EmptyInput);
    }
    if !(MIN_LOUDNESS_SAMPLE_RATE..=MAX_LOUDNESS_SAMPLE_RATE).contains(&sample_rate) {
        return Err(LoudnessError::UnsupportedSampleRate(sample_rate));
    }

    let mut sample_peak = 0.0_f64;
    for &sample in samples {
        if !sample.is_finite() {
            return Err(LoudnessError::NonFiniteSample);
        }
        let abs = f64::from(sample).abs();
        if abs > sample_peak {
            sample_peak = abs;
        }
    }

    let sample_peak_dbfs = amplitude_to_db(sample_peak);
    let true_peak_linear = measure_true_peak_linear(samples, sample_peak);
    let true_peak_dbtp = amplitude_to_db(true_peak_linear);

    if sample_peak <= 0.0 {
        return Ok(LoudnessObservation {
            integrated_lufs: None,
            max_momentary_lufs: None,
            max_short_term_lufs: None,
            loudness_range_lu: None,
            sample_peak_dbfs: None,
            true_peak_dbtp: None,
        });
    }

    let geometry = WindowGeometry::new(sample_rate)
        .ok_or(LoudnessError::UnsupportedSampleRate(sample_rate))?;
    let (momentary_energies, short_term_energies) = if geometry.exact_step_grid() {
        collect_block_energies_grid(samples, sample_rate, &geometry)?
    } else {
        collect_block_energies_general(samples, sample_rate, &geometry)?
    };

    let (integrated_lufs, max_momentary_lufs) = compute_integrated_lufs(&momentary_energies);
    let (max_short_term_lufs, loudness_range_lu) = compute_short_term_and_lra(&short_term_energies);

    Ok(LoudnessObservation {
        integrated_lufs,
        max_momentary_lufs,
        max_short_term_lufs,
        loudness_range_lu,
        sample_peak_dbfs,
        true_peak_dbtp,
    })
}

/// Validates a canonical mono PCM16 WAV payload and measures its loudness and true peak.
///
/// # Errors
/// Returns [`LoudnessError::Wav`] if WAV validation or decoding fails, or any
/// [`LoudnessError`] reported by [`measure_loudness_f32`].
pub fn measure_loudness_mono_pcm16_wav(
    wave: &[u8],
    sample_rate: u32,
    maximum_samples: u32,
) -> Result<LoudnessObservation, LoudnessError> {
    let pcm = mono_pcm16_data_checked(wave, sample_rate, maximum_samples)?;
    if !(MIN_LOUDNESS_SAMPLE_RATE..=MAX_LOUDNESS_SAMPLE_RATE).contains(&sample_rate) {
        return Err(LoudnessError::UnsupportedSampleRate(sample_rate));
    }
    let mut samples = Vec::new();
    decode_pcm16le_into(pcm, &mut samples)?;
    measure_loudness_f32(&samples, sample_rate)
}

/// Builds a separate, attested canonical mono PCM16 listening WAV matched to
/// `target_lufs` while clamping gain so 4x true peak never exceeds `true_peak_ceiling_dbtp`.
///
/// The original `samples` buffer is borrowed immutably and never modified.
///
/// # Errors
/// Returns [`LoudnessError::InvalidTargetLufs`] or [`LoudnessError::InvalidTruePeakCeiling`]
/// for non-finite parameters, [`LoudnessError::BelowLoudnessGate`] when integrated loudness
/// is `None` (silence, sub-threshold, or shorter than 400 ms), or any [`LoudnessError`]
/// from measurement, allocation, or canonical WAV framing.
pub fn build_loudness_matched_listening_wav(
    samples: &[f32],
    sample_rate: u32,
    target_lufs: f64,
    true_peak_ceiling_dbtp: f64,
) -> Result<LoudnessMatchedListeningWav, LoudnessError> {
    if !target_lufs.is_finite() {
        return Err(LoudnessError::InvalidTargetLufs);
    }
    if !true_peak_ceiling_dbtp.is_finite() {
        return Err(LoudnessError::InvalidTruePeakCeiling);
    }

    let observation = measure_loudness_f32(samples, sample_rate)?;
    let integrated_lufs = observation
        .integrated_lufs
        .ok_or(LoudnessError::BelowLoudnessGate)?;
    let true_peak_dbtp = observation
        .true_peak_dbtp
        .ok_or(LoudnessError::BelowLoudnessGate)?;

    let desired_gain_db = target_lufs - integrated_lufs;
    let max_safe_gain_db = true_peak_ceiling_dbtp - true_peak_dbtp;
    let (applied_gain_db, true_peak_clamped) = if desired_gain_db > max_safe_gain_db {
        (max_safe_gain_db, true)
    } else {
        (desired_gain_db, false)
    };
    if !applied_gain_db.is_finite() {
        return Err(LoudnessError::InvalidTruePeakCeiling);
    }
    let linear_gain = 10.0_f64.powf(applied_gain_db / 20.0);
    if !linear_gain.is_finite() {
        return Err(LoudnessError::InvalidTruePeakCeiling);
    }

    let mut scaled = Vec::new();
    scaled
        .try_reserve_exact(samples.len())
        .map_err(|_| LoudnessError::AllocationFailed)?;
    for &sample in samples {
        let boosted = f64::from(sample) * linear_gain;
        if !boosted.is_finite() {
            return Err(LoudnessError::InvalidTruePeakCeiling);
        }
        scaled.push(clamp_unit_f64_to_f32(boosted));
    }

    let wav_bytes = mono_pcm16_wav_bytes(&scaled, sample_rate, scaled.len())?;
    Ok(LoudnessMatchedListeningWav {
        observation,
        wav_bytes,
        applied_gain_db,
        true_peak_clamped,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Biquad, LoudnessError, build_loudness_matched_listening_wav, measure_loudness_f32,
        measure_loudness_mono_pcm16_wav,
    };
    use crate::{WavError, mono_pcm16_wav_bytes};

    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Synthetic sine test generator safely narrows bounded f64 values in [-1.0, 1.0] to f32."
    )]
    fn sine_wave(freq_hz: f64, peak_amp: f64, sample_rate: u32, seconds: f64) -> Vec<f32> {
        let len_f = (f64::from(sample_rate) * seconds).round();
        let len = (len_f as usize).max(1);
        let rate_f = f64::from(sample_rate);
        (0..len)
            .map(|idx| {
                let idx_f = f64::from(u32::try_from(idx).unwrap_or(u32::MAX));
                (peak_amp * (2.0 * std::f64::consts::PI * freq_hz * idx_f / rate_f).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn bs1770_48khz_coefficients_match_annex1_reference_table() {
        let s1 = Biquad::k_weight_stage1(48_000.0);
        assert!((s1.b0 - 1.535_124_859_586_97).abs() < 1.0e-12);
        assert!((s1.b1 - (-2.691_696_189_406_38)).abs() < 1.0e-12);
        assert!((s1.b2 - 1.198_392_810_852_85).abs() < 1.0e-12);
        assert!((s1.a1 - (-1.690_659_293_182_41)).abs() < 1.0e-12);
        assert!((s1.a2 - 0.732_480_774_215_85).abs() < 1.0e-12);

        let s2 = Biquad::k_weight_stage2(48_000.0);
        assert!((s2.b0 - 1.0).abs() < 1.0e-12);
        assert!((s2.b1 - (-2.0)).abs() < 1.0e-12);
        assert!((s2.b2 - 1.0).abs() < 1.0e-12);
        assert!((s2.a1 - (-1.990_047_454_833_98)).abs() < 1.0e-12);
        assert!((s2.a2 - 0.990_072_250_366_21).abs() < 1.0e-12);
    }

    #[test]
    fn sine_calibration_at_48khz_24khz_16khz_and_non_grid_rate_is_within_0_1_lu()
    -> Result<(), LoudnessError> {
        // Peak amplitude corresponding to -19.99 dBFS peak (0 dBFS RMS sine -> -3.01 dB, so -23.00 LUFS).
        let amp_minus_23_lufs = 10.0_f64.powf((-23.0 + 3.010_299_956_639_812) / 20.0);
        for rate in [48_000_u32, 24_000_u32, 16_000_u32, 11_025_u32] {
            for freq in [997.0, 1_000.0] {
                let samples = sine_wave(freq, amp_minus_23_lufs, rate, 4.0);
                let obs = measure_loudness_f32(&samples, rate)?;
                let lufs = obs
                    .integrated_lufs
                    .ok_or(LoudnessError::BelowLoudnessGate)?;
                assert!(
                    (lufs - (-23.0)).abs() <= 0.1,
                    "rate={rate} freq={freq} measured {lufs} LUFS"
                );
                let mom = obs
                    .max_momentary_lufs
                    .ok_or(LoudnessError::BelowLoudnessGate)?;
                assert!((mom - (-23.0)).abs() <= 0.1);
                let st = obs
                    .max_short_term_lufs
                    .ok_or(LoudnessError::BelowLoudnessGate)?;
                assert!((st - (-23.0)).abs() <= 0.1);
                let lra = obs
                    .loudness_range_lu
                    .ok_or(LoudnessError::BelowLoudnessGate)?;
                assert!(lra <= 0.05, "steady sine LRA should be ~0 LU, got {lra}");
            }
        }
        Ok(())
    }

    #[test]
    fn true_peak_detects_plus_3_dbtp_intersample_overshoot_above_0_dbfs()
    -> Result<(), LoudnessError> {
        // Repeating [+1, +1, -1, -1] represents sqrt(2)*sin(pi/2 * n + pi/4), which hits
        // sqrt(2) (+3.01 dBTP) midway between consecutive +1 samples while sample peak is 0 dBFS.
        let mut samples = Vec::with_capacity(256);
        for idx in 0..256 {
            samples.push(if idx % 4 < 2 { 1.0_f32 } else { -1.0_f32 });
        }
        let obs = measure_loudness_f32(&samples, 48_000)?;
        let sample_peak = obs
            .sample_peak_dbfs
            .ok_or(LoudnessError::BelowLoudnessGate)?;
        let true_peak = obs.true_peak_dbtp.ok_or(LoudnessError::BelowLoudnessGate)?;
        assert!(sample_peak.abs() < 1.0e-6);
        assert!(
            true_peak >= 2.99 && true_peak <= 3.15,
            "expected ~+3.01 dBTP inter-sample peak, got {true_peak}"
        );
        Ok(())
    }

    #[test]
    fn lra_measures_stepped_dynamic_range_correctly() -> Result<(), LoudnessError> {
        // 5 seconds at -20 dBFS RMS (-19.3 LUFS) followed by 5 seconds at -30 dBFS RMS (-29.3 LUFS)
        // produces ~10.0 LU loudness range (both segments well above -70 LUFS and -20 LU rel gate).
        let high_amp = 10.0_f64.powf((-20.0 + 3.010_299_956_639_812) / 20.0);
        let low_amp = 10.0_f64.powf((-30.0 + 3.010_299_956_639_812) / 20.0);
        let mut samples = sine_wave(997.0, high_amp, 24_000, 5.0);
        samples.extend(sine_wave(997.0, low_amp, 24_000, 5.0));

        let obs = measure_loudness_f32(&samples, 24_000)?;
        let lra = obs
            .loudness_range_lu
            .ok_or(LoudnessError::BelowLoudnessGate)?;
        assert!(
            (lra - 10.0).abs() <= 0.3,
            "expected ~10.0 LU for 10 dB stepped tone, got {lra}"
        );
        Ok(())
    }

    #[test]
    fn silence_subthreshold_short_clips_and_invalid_inputs_behave_safely()
    -> Result<(), LoudnessError> {
        // Pure silence: all Option<f64> fields are None without NaN/Inf.
        let silence = vec![0.0_f32; 24_000 * 4];
        let silent_obs = measure_loudness_f32(&silence, 24_000)?;
        assert_eq!(silent_obs.integrated_lufs, None);
        assert_eq!(silent_obs.max_momentary_lufs, None);
        assert_eq!(silent_obs.max_short_term_lufs, None);
        assert_eq!(silent_obs.loudness_range_lu, None);
        assert_eq!(silent_obs.sample_peak_dbfs, None);
        assert_eq!(silent_obs.true_peak_dbtp, None);

        // Sub-threshold (-80 dBFS peak) sine: rejected by -70 LUFS gate -> integrated & LRA None.
        let quiet = sine_wave(997.0, 1.0e-4, 24_000, 4.0);
        let quiet_obs = measure_loudness_f32(&quiet, 24_000)?;
        assert_eq!(quiet_obs.integrated_lufs, None);
        assert_eq!(quiet_obs.loudness_range_lu, None);
        assert!(quiet_obs.max_momentary_lufs.is_some_and(f64::is_finite));
        assert!(quiet_obs.true_peak_dbtp.is_some_and(f64::is_finite));

        // Short clip (200 ms < 400 ms): peaks exist, loudness windows are None.
        let short = sine_wave(997.0, 0.25, 24_000, 0.2);
        let short_obs = measure_loudness_f32(&short, 24_000)?;
        assert_eq!(short_obs.integrated_lufs, None);
        assert_eq!(short_obs.max_momentary_lufs, None);
        assert_eq!(short_obs.max_short_term_lufs, None);
        assert_eq!(short_obs.loudness_range_lu, None);
        assert!(short_obs.sample_peak_dbfs.is_some_and(f64::is_finite));
        assert!(short_obs.true_peak_dbtp.is_some_and(f64::is_finite));

        // Validation errors.
        assert_eq!(
            measure_loudness_f32(&[], 24_000),
            Err(LoudnessError::EmptyInput)
        );
        assert_eq!(
            measure_loudness_f32(&[0.0], 7_999),
            Err(LoudnessError::UnsupportedSampleRate(7_999))
        );
        assert_eq!(
            measure_loudness_f32(&[0.0], 192_001),
            Err(LoudnessError::UnsupportedSampleRate(192_001))
        );
        assert_eq!(
            measure_loudness_f32(&[0.1, f32::NAN], 24_000),
            Err(LoudnessError::NonFiniteSample)
        );
        assert_eq!(
            measure_loudness_f32(&[f32::INFINITY], 24_000),
            Err(LoudnessError::NonFiniteSample)
        );
        Ok(())
    }

    #[test]
    fn listening_copy_clamps_gain_to_true_peak_ceiling_and_preserves_input()
    -> Result<(), LoudnessError> {
        // Create a signal whose peak-to-loudness ratio is high: quiet -30 LUFS sine plus a -3 dBTP burst
        // using [+0.5, +0.5, -0.5, -0.5] inter-sample peak (+0.5 * sqrt(2) -> ~ -3.0 dBTP).
        let quiet_amp = 10.0_f64.powf((-30.0 + 3.010_299_956_639_812) / 20.0);
        let mut samples = sine_wave(997.0, quiet_amp, 24_000, 1.5);
        let before = samples.clone();
        // Inject a short 32-sample [+0.5, +0.5, -0.5, -0.5] transient (~ -3.0 dBTP true peak).
        for idx in 0..32 {
            samples[12_000 + idx] = if idx % 4 < 2 { 0.5 } else { -0.5 };
        }
        let original = samples.clone();

        // Requesting -14 LUFS (+16 dB) with a -1.0 dBTP ceiling must clamp to ~ +2.0 dB gain!
        let listening = build_loudness_matched_listening_wav(&samples, 24_000, -14.0, -1.0)?;
        assert_eq!(samples, original);
        assert_ne!(before, original);
        assert!(listening.true_peak_clamped);
        let input_tp = listening
            .observation
            .true_peak_dbtp
            .ok_or(LoudnessError::BelowLoudnessGate)?;
        assert!(
            (input_tp + listening.applied_gain_db - (-1.0)).abs() <= 1.0e-9,
            "clamped gain must land exactly on true-peak ceiling"
        );

        let output_obs = measure_loudness_mono_pcm16_wav(
            &listening.wav_bytes,
            24_000,
            u32::try_from(samples.len()).unwrap_or(u32::MAX),
        )?;
        let out_tp = output_obs
            .true_peak_dbtp
            .ok_or(LoudnessError::BelowLoudnessGate)?;
        assert!(
            out_tp <= -0.95,
            "decoded listening WAV true peak {out_tp} dBTP exceeded -1.0 dBTP ceiling"
        );

        // Now test unclamped matching on a quiet -28 LUFS sine targeting -18 LUFS with -1.0 dBTP ceiling.
        let clean_quiet = sine_wave(
            997.0,
            10.0_f64.powf((-28.0 + 3.010_299_956_639_812) / 20.0),
            16_000,
            1.0,
        );
        let unmatched = build_loudness_matched_listening_wav(&clean_quiet, 16_000, -18.0, -1.0)?;
        assert!(!unmatched.true_peak_clamped);
        let matched_obs = measure_loudness_mono_pcm16_wav(&unmatched.wav_bytes, 16_000, 16_000)?;
        let matched_lufs = matched_obs
            .integrated_lufs
            .ok_or(LoudnessError::BelowLoudnessGate)?;
        assert!(
            (matched_lufs - (-18.0)).abs() <= 0.1,
            "unclamped matched copy measured {matched_lufs} LUFS"
        );

        // Verify error guards.
        assert_eq!(
            build_loudness_matched_listening_wav(&clean_quiet, 16_000, f64::NAN, -1.0),
            Err(LoudnessError::InvalidTargetLufs)
        );
        assert_eq!(
            build_loudness_matched_listening_wav(&clean_quiet, 16_000, -18.0, f64::INFINITY),
            Err(LoudnessError::InvalidTruePeakCeiling)
        );
        let silence = vec![0.0_f32; 16_000];
        assert_eq!(
            build_loudness_matched_listening_wav(&silence, 16_000, -18.0, -1.0),
            Err(LoudnessError::BelowLoudnessGate)
        );
        Ok(())
    }

    #[test]
    fn wav_entrypoint_rejects_malformed_or_unsupported_headers() -> Result<(), LoudnessError> {
        let wave = mono_pcm16_wav_bytes(&sine_wave(997.0, 0.1, 16_000, 1.0), 16_000, 16_000)?;
        let obs = measure_loudness_mono_pcm16_wav(&wave, 16_000, 16_000)?;
        assert!(obs.integrated_lufs.is_some());

        assert_eq!(
            measure_loudness_mono_pcm16_wav(&wave[..20], 16_000, 16_000),
            Err(LoudnessError::Wav(WavError::InvalidRiffHeader))
        );
        assert_eq!(
            measure_loudness_mono_pcm16_wav(&wave, 24_000, 16_000),
            Err(LoudnessError::Wav(WavError::UnsupportedFormat))
        );
        Ok(())
    }
}
