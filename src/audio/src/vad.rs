//! Sub-band energy and adaptive minimum-statistics voice activity detector.
//!
//! Inspired by WebRTC/Earshot sub-band splitting across `80-250 Hz`,
//! `250-500 Hz`, `500-1000 Hz`, and `1000-4000 Hz`, pairing DC-blocking
//! high-pass filtering and zero-crossing bounds with stationary noise tracking.

const MIN_BAND_FLOOR: f32 = 1.0e-8;
const PCM16_INV_SCALE: f32 = 1.0 / 32768.0;

/// Configuration for [`VoiceActivityDetector`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VadConfig {
    /// PCM sample rate in Hz (`8_000`, `16_000`, `32_000`, or `48_000`).
    pub sample_rate: u32,
    /// Frame duration in milliseconds (`10`, `20`, or `30`).
    pub frame_ms: u16,
    /// Aggressiveness mode `0..=3` (higher requires stronger speech SNR).
    pub aggressiveness: u8,
    /// Trailing active frames held as [`VadFrameStatus::Hangover`] after speech.
    pub hangover_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_ms: 20,
            aggressiveness: 2,
            hangover_frames: 8,
        }
    }
}

/// Frame classification produced by [`VoiceActivityDetector::process_frame`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VadFrameStatus {
    /// Active voiced or unvoiced speech detected over background noise.
    Speech,
    /// Trailing hangover frame preserving consonant offsets after active speech.
    Hangover,
    /// Background noise or silence below speech thresholds.
    Silence,
}

impl VadFrameStatus {
    /// Returns `true` when the frame carries speech or hangover continuity.
    #[must_use]
    pub const fn is_active(self) -> bool {
        matches!(self, Self::Speech | Self::Hangover)
    }
}

/// Streaming sub-band + minimum-statistics voice activity detector.
#[derive(Clone, Debug)]
pub struct VoiceActivityDetector {
    frame_samples: usize,
    aggressiveness: u8,
    hangover_limit: usize,
    hp_alpha: f32,
    lp_alphas: [f32; 4],
    hp_prev_in: f32,
    hp_prev_out: f32,
    lp_states: [f32; 4],
    noise_floor: [f32; 4],
    prev_energies: [f32; 4],
    frames_seen: usize,
    hangover_remaining: usize,
}

impl VoiceActivityDetector {
    /// Construct a detector for supported sample rates (`8/16/32/48 kHz`) and
    /// frame durations (`10/20/30 ms`), returning `None` if unsupported.
    #[must_use]
    pub fn new(config: VadConfig) -> Option<Self> {
        let rate = match config.sample_rate {
            8_000 | 16_000 | 32_000 | 48_000 => config.sample_rate,
            _ => return None,
        };
        if !matches!(config.frame_ms, 10 | 20 | 30) {
            return None;
        }
        let samples_u32 = (rate / 1_000) * u32::from(config.frame_ms);
        let frame_samples = usize::try_from(samples_u32).ok()?;
        let fs = f64::from(rate);
        let hp_alpha = pole_alpha(80.0, fs);
        let nyquist_cap = (fs * 0.45).min(4_000.0);
        let lp_alphas = [
            pole_alpha(250.0, fs),
            pole_alpha(500.0, fs),
            pole_alpha(1_000.0, fs),
            pole_alpha(nyquist_cap, fs),
        ];

        Some(Self {
            frame_samples,
            aggressiveness: config.aggressiveness.min(3),
            hangover_limit: config.hangover_frames,
            hp_alpha,
            lp_alphas,
            hp_prev_in: 0.0,
            hp_prev_out: 0.0,
            lp_states: [0.0; 4],
            noise_floor: [MIN_BAND_FLOOR; 4],
            prev_energies: [MIN_BAND_FLOOR; 4],
            frames_seen: 0,
            hangover_remaining: 0,
        })
    }

    /// Expected PCM16 sample count per `process_frame` call.
    #[must_use]
    pub const fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    /// Clear filter states, hangover counters, and noise estimates.
    pub fn reset(&mut self) {
        self.hp_prev_in = 0.0;
        self.hp_prev_out = 0.0;
        self.lp_states = [0.0; 4];
        self.noise_floor = [MIN_BAND_FLOOR; 4];
        self.prev_energies = [MIN_BAND_FLOOR; 4];
        self.frames_seen = 0;
        self.hangover_remaining = 0;
    }

    /// Classify one PCM16 frame, returning `None` if `frame.len() != frame_samples()`.
    pub fn process_frame(&mut self, frame: &[i16]) -> Option<VadFrameStatus> {
        if frame.len() != self.frame_samples {
            return None;
        }

        let mut band_sum_sq = [0.0_f32; 4];
        let mut total_sq = 0.0_f32;
        let mut zero_crossings = 0_usize;
        let mut prev_sign_pos = self.hp_prev_out >= 0.0;

        for &sample in frame {
            let x = f32::from(sample) * PCM16_INV_SCALE;
            let hp = (x - self.hp_prev_in) + self.hp_alpha * self.hp_prev_out;
            self.hp_prev_in = x;
            self.hp_prev_out = hp;
            total_sq += hp * hp;

            let sign_pos = hp >= 0.0;
            if sign_pos != prev_sign_pos && hp.abs() > 1.0e-5 {
                zero_crossings += 1;
            }
            prev_sign_pos = sign_pos;

            for (state, alpha) in self.lp_states.iter_mut().zip(self.lp_alphas) {
                *state = alpha * (*state) + (1.0 - alpha) * hp;
            }
            let b0 = self.lp_states[0];
            let b1 = self.lp_states[1] - self.lp_states[0];
            let b2 = self.lp_states[2] - self.lp_states[1];
            let b3 = self.lp_states[3] - self.lp_states[2];
            band_sum_sq[0] += b0 * b0;
            band_sum_sq[1] += b1 * b1;
            band_sum_sq[2] += b2 * b2;
            band_sum_sq[3] += b3 * b3;
        }

        let inv_n = 1.0 / u32_to_f32(u32::try_from(frame.len()).unwrap_or(1));
        let band_energies = band_sum_sq.map(|e| (e * inv_n).max(MIN_BAND_FLOOR));
        let total_energy = total_sq * inv_n;
        let zcr = u32_to_f32(u32::try_from(zero_crossings).unwrap_or(0)) * inv_n;

        if self.frames_seen == 0 {
            for (floor, &energy) in self.noise_floor.iter_mut().zip(band_energies.iter()) {
                *floor = energy.clamp(MIN_BAND_FLOOR, 5.0e-3);
            }
            self.prev_energies = band_energies;
        }

        let mut stationary = true;
        for (cur, prev) in band_energies.iter().zip(self.prev_energies.iter()) {
            let rel_delta = (*cur - *prev).abs() / (*cur + *prev + 1.0e-9);
            if rel_delta > 0.14 {
                stationary = false;
                break;
            }
        }

        let (snr_thresh_db, abs_energy_floor, max_zcr) = match self.aggressiveness {
            0 => (3.5_f32, 3.0e-6_f32, 0.55_f32),
            1 => (4.8_f32, 5.0e-6_f32, 0.48_f32),
            2 => (6.2_f32, 8.0e-6_f32, 0.42_f32),
            _ => (7.8_f32, 1.4e-5_f32, 0.36_f32),
        };

        let mut snr_sum = 0.0_f32;
        let mut max_speech_band_snr = 0.0_f32;
        for b in 0..4 {
            let snr_db = 10.0 * (band_energies[b] / self.noise_floor[b]).log10();
            let weight = if b == 0 { 0.35 } else { 1.0 };
            snr_sum += weight * snr_db.max(0.0);
            if b >= 1 {
                max_speech_band_snr = max_speech_band_snr.max(snr_db);
            }
        }

        let formants_active = band_energies[1] + band_energies[2] + band_energies[3] > 2.5e-6;
        let zcr_plausible = zcr <= max_zcr || formants_active;
        let raw_speech = total_energy >= abs_energy_floor
            && zcr_plausible
            && (snr_sum >= snr_thresh_db || max_speech_band_snr >= snr_thresh_db + 1.0);

        let is_speech =
            raw_speech && !(stationary && self.frames_seen >= 2 && total_energy < 5.0e-3);

        for b in 0..4 {
            let energy = band_energies[b];
            let floor = &mut self.noise_floor[b];
            if energy < *floor {
                *floor = (*floor * 0.60 + energy * 0.40).max(MIN_BAND_FLOOR);
            } else if !is_speech {
                let attack = if stationary { 0.25 } else { 0.015 };
                *floor = (*floor * (1.0 - attack) + energy * attack).max(MIN_BAND_FLOOR);
            } else {
                *floor = (*floor * 0.998 + energy * 0.002).max(MIN_BAND_FLOOR);
            }
        }

        self.prev_energies = band_energies;
        self.frames_seen += 1;

        if is_speech {
            self.hangover_remaining = self.hangover_limit;
            Some(VadFrameStatus::Speech)
        } else if self.hangover_remaining > 0 {
            self.hangover_remaining -= 1;
            Some(VadFrameStatus::Hangover)
        } else {
            Some(VadFrameStatus::Silence)
        }
    }
}

fn pole_alpha(cutoff_hz: f64, sample_rate_hz: f64) -> f32 {
    let ratio = (-2.0 * std::f64::consts::PI * cutoff_hz / sample_rate_hz).exp();
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "1-pole IIR coefficient in (0, 1) fits in f32 without overflow."
    )]
    {
        ratio.clamp(0.0, 0.9999) as f32
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Bounded frame counts (<= 1440) convert losslessly to f32."
)]
const fn u32_to_f32(value: u32) -> f32 {
    value as f32
}

#[cfg(test)]
mod tests {
    use super::{VadConfig, VadFrameStatus, VoiceActivityDetector};

    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "Synthetic test oscillator converts f64 sine samples into clamped i16."
    )]
    fn synth_tone(freq_hz: f64, amp: f64, len: usize, phase_offset: usize, rate: f64) -> Vec<i16> {
        (0..len)
            .map(|i| {
                let t = (phase_offset + i) as f64 / rate;
                let v = (amp * (2.0 * std::f64::consts::PI * freq_hz * t).sin() * 32767.0)
                    .clamp(-32767.0, 32767.0);
                v as i16
            })
            .collect()
    }

    #[test]
    fn vad_rejects_invalid_rate_and_frame_length() -> Result<(), &'static str> {
        assert!(
            VoiceActivityDetector::new(VadConfig {
                sample_rate: 44_100,
                ..VadConfig::default()
            })
            .is_none()
        );
        assert!(
            VoiceActivityDetector::new(VadConfig {
                frame_ms: 25,
                ..VadConfig::default()
            })
            .is_none()
        );
        let mut vad = VoiceActivityDetector::new(VadConfig::default()).ok_or("default config")?;
        assert_eq!(vad.frame_samples(), 320);
        assert!(vad.process_frame(&[0; 160]).is_none());
        Ok(())
    }

    #[test]
    fn vad_detects_speech_with_hangover_over_silence() -> Result<(), &'static str> {
        let mut vad = VoiceActivityDetector::new(VadConfig {
            hangover_frames: 3,
            ..VadConfig::default()
        })
        .ok_or("vad config")?;
        let silence = vec![0_i16; 320];
        for _ in 0..5 {
            assert_eq!(vad.process_frame(&silence), Some(VadFrameStatus::Silence));
        }

        let mut speech_hits = 0_usize;
        for frame_idx in 0..10 {
            let f1 = synth_tone(380.0, 0.16, 320, frame_idx * 320, 16_000.0);
            let f2 = synth_tone(1_250.0, 0.10, 320, frame_idx * 320, 16_000.0);
            let mixed: Vec<i16> = f1
                .iter()
                .zip(f2.iter())
                .map(|(&a, &b)| a.saturating_add(b))
                .collect();
            if vad.process_frame(&mixed) == Some(VadFrameStatus::Speech) {
                speech_hits += 1;
            }
        }
        assert!(speech_hits >= 8);

        assert_eq!(vad.process_frame(&silence), Some(VadFrameStatus::Hangover));
        assert_eq!(vad.process_frame(&silence), Some(VadFrameStatus::Hangover));
        assert_eq!(vad.process_frame(&silence), Some(VadFrameStatus::Hangover));
        assert_eq!(vad.process_frame(&silence), Some(VadFrameStatus::Silence));
        Ok(())
    }

    #[test]
    fn vad_adapts_to_steady_low_frequency_hum_and_still_detects_speech() -> Result<(), &'static str>
    {
        let mut vad = VoiceActivityDetector::new(VadConfig {
            hangover_frames: 2,
            ..VadConfig::default()
        })
        .ok_or("vad config")?;
        // -26 dBFS ~= 0.05 linear amplitude at 120 Hz steady hum.
        for frame_idx in 0..40 {
            let hum = synth_tone(120.0, 0.05, 320, frame_idx * 320, 16_000.0);
            let status = vad.process_frame(&hum);
            if frame_idx >= 25 {
                assert_eq!(status, Some(VadFrameStatus::Silence));
            }
        }

        let mut active_during_speech = 0_usize;
        for frame_idx in 40..55 {
            let hum = synth_tone(120.0, 0.05, 320, frame_idx * 320, 16_000.0);
            let f1 = synth_tone(420.0, 0.08, 320, frame_idx * 320, 16_000.0);
            let f2 = synth_tone(1_650.0, 0.05, 320, frame_idx * 320, 16_000.0);
            let mixed: Vec<i16> = hum
                .iter()
                .zip(f1.iter())
                .zip(f2.iter())
                .map(|((&h, &a), &b)| h.saturating_add(a).saturating_add(b))
                .collect();
            if vad.process_frame(&mixed) == Some(VadFrameStatus::Speech) {
                active_during_speech += 1;
            }
        }
        assert!(active_during_speech >= 8);
        Ok(())
    }
}
