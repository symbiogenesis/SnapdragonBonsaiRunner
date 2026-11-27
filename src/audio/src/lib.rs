//! Canonical PCM sample admission and mono PCM16 WAV framing.
//!
//! This codec accepts only a 44-byte PCM header followed by nonempty, complete
//! little-endian samples. Callers select the sample rate and maximum sample
//! count; optional chunks and extensible formats need explicit normalization
//! before entering this boundary. Audio stays borrowed and no devices are opened.
#![forbid(unsafe_code)]

use std::{error::Error, fmt};

mod loudness;
mod segmentation;
mod vad;

pub use loudness::{
    LoudnessError, LoudnessMatchedListeningWav, LoudnessObservation, MAX_LOUDNESS_SAMPLE_RATE,
    MIN_LOUDNESS_SAMPLE_RATE, build_loudness_matched_listening_wav, measure_loudness_f32,
    measure_loudness_mono_pcm16_wav,
};
pub use segmentation::{pcm16_pause_boundary, pcm16_pause_boundary_adaptive};
pub use vad::{VadConfig, VadFrameStatus, VoiceActivityDetector};

/// Descriptive error returned by checked WAV framing and PCM16 conversion helpers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WavError {
    /// Input does not begin with a valid `RIFF`..`WAVE` header matching its byte length.
    InvalidRiffHeader,
    /// A RIFF chunk header or payload is truncated or overflows bounds.
    TruncatedChunk,
    /// WAV input contains more than one `fmt ` chunk.
    DuplicateFormatChunk,
    /// WAV input contains more than one `data` chunk.
    DuplicateDataChunk,
    /// WAV input is missing a required `fmt ` chunk.
    MissingFormatChunk,
    /// WAV input is missing a required `data` chunk.
    MissingDataChunk,
    /// WAV input contains non-canonical chunks or chunk ordering.
    NonCanonicalChunkLayout,
    /// The `fmt ` chunk does not describe canonical 16-bit mono PCM at the expected sample rate.
    UnsupportedFormat,
    /// The `data` chunk is empty, unaligned, or exceeds the caller's sample limit.
    InvalidDataLength,
    /// PCM output contains non-finite samples, zero length, or an unsupported rate.
    InvalidSamples,
    /// Destination PCM byte slice is smaller than `samples.len() * 2`.
    BufferTooSmall,
    /// Sample destination vector capacity reservation failed.
    AllocationFailed,
}

impl fmt::Display for WavError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRiffHeader => {
                "WAV input must start with a valid RIFF/WAVE header matching its byte length"
            }
            Self::TruncatedChunk => "WAV RIFF chunk header or payload is truncated",
            Self::DuplicateFormatChunk => "WAV input contains duplicate 'fmt ' chunks",
            Self::DuplicateDataChunk => "WAV input contains duplicate 'data' chunks",
            Self::MissingFormatChunk => "WAV input is missing a required 'fmt ' chunk",
            Self::MissingDataChunk => "WAV input is missing a required 'data' chunk",
            Self::NonCanonicalChunkLayout => {
                "WAV input must contain only canonical 'fmt ' followed by 'data' chunks"
            }
            Self::UnsupportedFormat => {
                "WAV 'fmt ' chunk is not canonical 16-bit mono PCM at the expected sample rate"
            }
            Self::InvalidDataLength => {
                "WAV 'data' chunk is empty, unaligned, or exceeds the maximum sample count"
            }
            Self::InvalidSamples => {
                "PCM output contains non-finite samples, zero length, or an invalid rate"
            }
            Self::BufferTooSmall => "destination PCM buffer is smaller than required",
            Self::AllocationFailed => "failed to reserve sample buffer capacity",
        })
    }
}

impl Error for WavError {}

/// Admits a nonempty finite PCM chunk and returns its cumulative sample count.
///
/// `previous` counts samples already admitted; `maximum` is the caller's clip
/// or stream bound. Returns `None` for nonfinite samples, overflow or a count
/// exceeding that bound. Finite values outside -1 through 1 remain valid for
/// callers that clip during PCM16 encoding. No samples are copied or changed.
#[must_use]
pub fn finite_sample_count(samples: &[f32], previous: usize, maximum: usize) -> Option<usize> {
    let total = previous.checked_add(samples.len())?;
    if samples.is_empty() || total > maximum {
        return None;
    }
    let (chunks, tail) = samples.as_chunks::<8>();
    for c in chunks {
        if !(c[0].is_finite()
            & c[1].is_finite()
            & c[2].is_finite()
            & c[3].is_finite()
            & c[4].is_finite()
            & c[5].is_finite()
            & c[6].is_finite()
            & c[7].is_finite())
        {
            return None;
        }
    }
    if tail.iter().all(|sample| sample.is_finite()) {
        Some(total)
    } else {
        None
    }
}

fn validate_canonical_pcm16_fmt(
    body: &[u8],
    sample_rate: u32,
    byte_rate: u32,
) -> Result<(), WavError> {
    let Some((
        &[
            a0,
            a1,
            c0,
            c1,
            r0,
            r1,
            r2,
            r3,
            b0,
            b1,
            b2,
            b3,
            ba0,
            ba1,
            bps0,
            bps1,
        ],
        [],
    )) = body.split_first_chunk::<16>()
    else {
        return Err(WavError::UnsupportedFormat);
    };
    let audio_format = u16::from_le_bytes([a0, a1]);
    let channels = u16::from_le_bytes([c0, c1]);
    let rate = u32::from_le_bytes([r0, r1, r2, r3]);
    let actual_byte_rate = u32::from_le_bytes([b0, b1, b2, b3]);
    let block_align = u16::from_le_bytes([ba0, ba1]);
    let bits_per_sample = u16::from_le_bytes([bps0, bps1]);
    if audio_format != 1
        || channels != 1
        || rate != sample_rate
        || actual_byte_rate != byte_rate
        || block_align != 2
        || bits_per_sample != 16
    {
        return Err(WavError::UnsupportedFormat);
    }
    Ok(())
}

/// Validates a canonical, bounded mono PCM16 WAV and returns its borrowed payload.
///
/// Rejects duplicate `fmt ` or `data` chunks, missing chunks, mismatched rates,
/// lengths, channel/sample formats, empty or partial samples, extra chunks,
/// and trailing data with a descriptive [`WavError`].
///
/// # Errors
/// Returns [`WavError`] describing the first structural, format, or bound violation.
pub fn mono_pcm16_data_checked(
    wave: &[u8],
    sample_rate: u32,
    maximum_samples: u32,
) -> Result<&[u8], WavError> {
    if sample_rate == 0 {
        return Err(WavError::UnsupportedFormat);
    }
    let byte_rate = sample_rate
        .checked_mul(2)
        .ok_or(WavError::UnsupportedFormat)?;
    let Some((
        &[
            b'R',
            b'I',
            b'F',
            b'F',
            r0,
            r1,
            r2,
            r3,
            b'W',
            b'A',
            b'V',
            b'E',
        ],
        _,
    )) = wave.split_first_chunk::<12>()
    else {
        return Err(WavError::InvalidRiffHeader);
    };
    let riff_size = u32::from_le_bytes([r0, r1, r2, r3]);
    if usize::try_from(riff_size).ok() != Some(wave.len() - 8) {
        return Err(WavError::InvalidRiffHeader);
    }

    let mut offset = 12;
    let mut fmt_valid = false;
    let mut data_payload: Option<&[u8]> = None;
    let mut non_canonical = false;
    let mut chunks = 0_usize;

    while offset < wave.len() {
        chunks += 1;
        if chunks > 128 {
            return Err(WavError::NonCanonicalChunkLayout);
        }
        let Some((chunk_header, _)) = wave[offset..].split_first_chunk::<8>() else {
            return Err(WavError::TruncatedChunk);
        };
        let [id0, id1, id2, id3, s0, s1, s2, s3] = *chunk_header;
        let id = [id0, id1, id2, id3];
        let length = usize::try_from(u32::from_le_bytes([s0, s1, s2, s3]))
            .map_err(|_| WavError::TruncatedChunk)?;
        let body_start = offset + 8;
        let body_end = body_start
            .checked_add(length)
            .ok_or(WavError::TruncatedChunk)?;
        let padded_end = body_end
            .checked_add(length % 2)
            .ok_or(WavError::TruncatedChunk)?;
        if padded_end > wave.len() {
            return Err(WavError::TruncatedChunk);
        }
        match &id {
            b"fmt " => {
                if fmt_valid {
                    return Err(WavError::DuplicateFormatChunk);
                }
                if data_payload.is_some() {
                    non_canonical = true;
                }
                validate_canonical_pcm16_fmt(&wave[body_start..body_end], sample_rate, byte_rate)?;
                fmt_valid = true;
            }
            b"data" => {
                if data_payload.is_some() {
                    return Err(WavError::DuplicateDataChunk);
                }
                if !fmt_valid {
                    non_canonical = true;
                }
                data_payload = Some(&wave[body_start..body_end]);
            }
            _ => {
                non_canonical = true;
            }
        }
        offset = padded_end;
    }

    if !fmt_valid {
        return Err(WavError::MissingFormatChunk);
    }
    let payload = data_payload.ok_or(WavError::MissingDataChunk)?;
    if non_canonical {
        return Err(WavError::NonCanonicalChunkLayout);
    }
    if payload.is_empty() || !payload.len().is_multiple_of(2) {
        return Err(WavError::InvalidDataLength);
    }
    let samples = payload.len() / 2;
    if u32::try_from(samples).map_err(|_| WavError::InvalidDataLength)? > maximum_samples {
        return Err(WavError::InvalidDataLength);
    }
    Ok(payload)
}

/// Returns the borrowed PCM payload of a canonical, bounded mono PCM16 WAV.
///
/// Rejects mismatched rates, lengths, channel/sample formats, empty or partial
/// samples, extra chunks and trailing data. `maximum_samples` counts individual
/// mono samples, independently of their two-byte representation.
#[must_use]
pub fn mono_pcm16_data(wave: &[u8], sample_rate: u32, maximum_samples: u32) -> Option<&[u8]> {
    mono_pcm16_data_checked(wave, sample_rate, maximum_samples).ok()
}

/// Validates finite samples against `maximum_samples` and builds a canonical 44-byte WAV header.
///
/// # Errors
/// Returns [`WavError::InvalidSamples`] when samples are empty, non-finite, exceed
/// `maximum_samples`, or cannot be framed at `sample_rate`.
pub fn mono_pcm16_header_checked(
    samples: &[f32],
    sample_rate: u32,
    maximum_samples: usize,
) -> Result<[u8; 44], WavError> {
    if finite_sample_count(samples, 0, maximum_samples).is_none() {
        return Err(WavError::InvalidSamples);
    }
    mono_pcm16_header(sample_rate, samples.len()).ok_or(WavError::InvalidSamples)
}

/// Frames finite normalized `f32` samples into a canonical mono PCM16 WAV byte vector.
///
/// # Errors
/// Returns [`WavError::InvalidSamples`] when samples are empty, non-finite, exceed
/// `maximum_samples`, or cannot be framed at `sample_rate`, or
/// [`WavError::AllocationFailed`] / [`WavError::BufferTooSmall`] if the buffer cannot be reserved.
pub fn mono_pcm16_wav_bytes(
    samples: &[f32],
    sample_rate: u32,
    maximum_samples: usize,
) -> Result<Vec<u8>, WavError> {
    let header = mono_pcm16_header_checked(samples, sample_rate, maximum_samples)?;
    let pcm_bytes = samples
        .len()
        .checked_mul(2)
        .ok_or(WavError::BufferTooSmall)?;
    let total_bytes = header
        .len()
        .checked_add(pcm_bytes)
        .ok_or(WavError::BufferTooSmall)?;
    let mut wave = Vec::new();
    wave.try_reserve_exact(total_bytes)
        .map_err(|_| WavError::AllocationFailed)?;
    wave.extend_from_slice(&header);
    let (chunks, tail) = samples.as_chunks::<8>();
    for c in chunks {
        let mut block = [0_u8; 16];
        for (dst, &sample) in block.as_chunks_mut::<2>().0.iter_mut().zip(c.iter()) {
            *dst = encode_pcm16le_sample(sample).to_le_bytes();
        }
        wave.extend_from_slice(&block);
    }
    for &sample in tail {
        wave.extend_from_slice(&encode_pcm16le_sample(sample).to_le_bytes());
    }
    Ok(wave)
}

/// Decodes little-endian signed 16-bit PCM bytes into normalized `f32` samples.
///
/// Clears `samples` before validation and reserves exact capacity fallibly.
///
/// # Errors
/// Returns [`WavError::InvalidDataLength`] for empty or odd-length PCM payloads,
/// or [`WavError::AllocationFailed`] if reserving sample capacity fails.
pub fn decode_pcm16le_into<'a>(
    pcm: &[u8],
    samples: &'a mut Vec<f32>,
) -> Result<&'a [f32], WavError> {
    samples.clear();
    let (pairs, remainder) = pcm.as_chunks::<2>();
    if pairs.is_empty() || !remainder.is_empty() {
        return Err(WavError::InvalidDataLength);
    }
    samples
        .try_reserve_exact(pairs.len())
        .map_err(|_| WavError::AllocationFailed)?;
    // 1.0 / 32_768.0 is exact binary 2^-15, preserving bit-for-bit equality
    // with division by 32_768.0 across every i16 sample without FP division.
    let scale = 1.0 / 32_768.0;
    let (chunks, tail) = pairs.as_chunks::<8>();
    for c in chunks {
        let decoded = [
            f32::from(i16::from_le_bytes(c[0])) * scale,
            f32::from(i16::from_le_bytes(c[1])) * scale,
            f32::from(i16::from_le_bytes(c[2])) * scale,
            f32::from(i16::from_le_bytes(c[3])) * scale,
            f32::from(i16::from_le_bytes(c[4])) * scale,
            f32::from(i16::from_le_bytes(c[5])) * scale,
            f32::from(i16::from_le_bytes(c[6])) * scale,
            f32::from(i16::from_le_bytes(c[7])) * scale,
        ];
        samples.extend_from_slice(&decoded);
    }
    for &sample in tail {
        samples.push(f32::from(i16::from_le_bytes(sample)) * scale);
    }
    Ok(samples)
}

/// Converts one normalized floating-point sample into a clamped signed 16-bit PCM integer.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Finite samples are explicitly clamped and rounded into the representable PCM16 range."
)]
#[must_use]
pub fn encode_pcm16le_sample(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// Encodes normalized floating-point samples into little-endian signed 16-bit PCM bytes.
///
/// # Errors
/// Returns [`WavError::BufferTooSmall`] when `bytes` cannot hold `samples.len() * 2` bytes.
pub fn encode_pcm16le_into<'a>(samples: &[f32], bytes: &'a mut [u8]) -> Result<&'a [u8], WavError> {
    let byte_len = samples
        .len()
        .checked_mul(2)
        .ok_or(WavError::BufferTooSmall)?;
    let target = bytes.get_mut(..byte_len).ok_or(WavError::BufferTooSmall)?;
    let (sample_chunks, sample_tail) = samples.as_chunks::<8>();
    let (dst_chunks, dst_tail) = target.as_chunks_mut::<16>();
    for (s, d) in sample_chunks.iter().zip(dst_chunks.iter_mut()) {
        let mut idx = 0_usize;
        for slot in d.as_chunks_mut::<2>().0 {
            *slot = encode_pcm16le_sample(s[idx]).to_le_bytes();
            idx += 1;
        }
    }
    for (&sample, slot) in sample_tail.iter().zip(dst_tail.as_chunks_mut::<2>().0) {
        *slot = encode_pcm16le_sample(sample).to_le_bytes();
    }
    Ok(target)
}

/// Builds the exact 44-byte header for a nonempty mono PCM16 WAV clip.
///
/// Returns `None` for a zero rate/count, an unrepresentable byte rate or a clip
/// whose RIFF length cannot fit the format's 32-bit field. The caller owns
/// duration admission and validation of sample values before sending the header.
#[must_use]
pub fn mono_pcm16_header(sample_rate: u32, samples: usize) -> Option<[u8; 44]> {
    if sample_rate == 0 || samples == 0 {
        return None;
    }
    let byte_rate = sample_rate.checked_mul(2)?;
    let length = u32::try_from(samples).ok()?.checked_mul(2)?;
    let riff_length = length.checked_add(36)?;
    let mut bytes = [0; 44];
    bytes[..4].copy_from_slice(b"RIFF");
    bytes[4..8].copy_from_slice(&riff_length.to_le_bytes());
    bytes[8..16].copy_from_slice(b"WAVEfmt ");
    bytes[16..20].copy_from_slice(&16_u32.to_le_bytes());
    bytes[20..22].copy_from_slice(&1_u16.to_le_bytes());
    bytes[22..24].copy_from_slice(&1_u16.to_le_bytes());
    bytes[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    bytes[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    bytes[32..34].copy_from_slice(&2_u16.to_le_bytes());
    bytes[34..36].copy_from_slice(&16_u16.to_le_bytes());
    bytes[36..40].copy_from_slice(b"data");
    bytes[40..44].copy_from_slice(&length.to_le_bytes());
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::{
        WavError, decode_pcm16le_into, encode_pcm16le_into, encode_pcm16le_sample,
        finite_sample_count, mono_pcm16_data, mono_pcm16_data_checked, mono_pcm16_header,
        mono_pcm16_header_checked,
    };

    #[test]
    fn complete_and_streamed_samples_share_finite_count_and_overflow_bounds() {
        assert_eq!(finite_sample_count(&[-2.0, 2.0], 0, 2), Some(2));
        assert_eq!(finite_sample_count(&[0.0], 1, 2), Some(2));
        assert_eq!(finite_sample_count(&[0.0], 2, 2), None);
        assert_eq!(finite_sample_count(&[0.0], usize::MAX, usize::MAX), None);
        for samples in [&[][..], &[f32::NAN], &[f32::INFINITY], &[f32::NEG_INFINITY]] {
            assert_eq!(finite_sample_count(samples, 0, 2), None);
            assert_eq!(finite_sample_count(samples, 1, 2), None);
        }
    }

    // Independent wire fixture keeps header generation and validation honest.
    fn fixture(rate: u32) -> Vec<u8> {
        let mut wave = b"RIFF\x28\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0".to_vec();
        wave.extend_from_slice(&rate.to_le_bytes());
        wave.extend_from_slice(&(rate * 2).to_le_bytes());
        wave.extend_from_slice(b"\x02\0\x10\0data\x04\0\0\0\0\x80\xff\x7f");
        wave
    }

    #[test]
    fn speech_formats_preserve_exact_headers_payloads_and_bounds() {
        for (rate, maximum) in [(16_000, 480_000), (24_000, 2_880_000)] {
            let wave = fixture(rate);
            assert_eq!(
                mono_pcm16_header(rate, 2).as_ref(),
                wave.first_chunk::<44>()
            );
            assert_eq!(mono_pcm16_data(&wave, rate, maximum), Some(&wave[44..]));
            assert!(mono_pcm16_data(&wave, rate, 1).is_none());
            assert!(mono_pcm16_data(&wave, rate + 1, maximum).is_none());
            let count = usize::try_from(maximum).unwrap_or_default();
            let mut maximum_wave = Vec::with_capacity(count * 2 + 44);
            maximum_wave.extend_from_slice(&mono_pcm16_header(rate, count).unwrap_or([0; 44]));
            maximum_wave.resize(count * 2 + 44, 0);
            assert!(mono_pcm16_data(&maximum_wave, rate, maximum).is_some());
            assert!(mono_pcm16_data(&maximum_wave, rate, maximum - 1).is_none());
        }
    }

    #[test]
    fn malformed_wave_headers_truncation_and_trailing_data_are_rejected() {
        let wave = fixture(16_000);
        for length in 0..wave.len() {
            assert!(mono_pcm16_data(&wave[..length], 16_000, 480_000).is_none());
        }
        for offset in 0..44 {
            let mut invalid = wave.clone();
            invalid[offset] ^= 1;
            assert!(mono_pcm16_data(&invalid, 16_000, 480_000).is_none());
        }
        let mut invalid = wave;
        invalid.extend_from_slice(b"JUNK\0\0\0\0");
        assert!(mono_pcm16_data(&invalid, 16_000, 480_000).is_none());
        let wave = fixture(16_000);
        let mut empty = wave[..44].to_vec();
        empty[4..8].copy_from_slice(&36_u32.to_le_bytes());
        empty[40..44].copy_from_slice(&0_u32.to_le_bytes());
        let mut duplicated_data = wave.clone();
        duplicated_data.extend_from_slice(b"data\x02\0\0\0\0\0");
        duplicated_data[4..8].copy_from_slice(&50_u32.to_le_bytes());
        let mut reordered = b"RIFF\x28\0\0\0WAVEdata\x04\0\0\0".to_vec();
        reordered.extend_from_slice(&wave[44..]);
        reordered.extend_from_slice(&wave[12..36]);
        let mut padded_chunk = wave[..36].to_vec();
        padded_chunk.extend_from_slice(b"JUNK\x01\0\0\0x\0");
        padded_chunk.extend_from_slice(&wave[36..]);
        padded_chunk[4..8].copy_from_slice(&50_u32.to_le_bytes());
        let mut odd_data = wave;
        odd_data[4..8].copy_from_slice(&39_u32.to_le_bytes());
        odd_data[40..44].copy_from_slice(&3_u32.to_le_bytes());
        odd_data.pop();
        for invalid in [empty, duplicated_data, reordered, padded_chunk, odd_data] {
            assert!(mono_pcm16_data(&invalid, 16_000, 480_000).is_none());
        }
        assert!(mono_pcm16_header(0, 1).is_none());
        assert!(mono_pcm16_header(16_000, 0).is_none());
        assert!(mono_pcm16_header(u32::MAX, 1).is_none());
        assert!(mono_pcm16_header(16_000, usize::MAX).is_none());
        assert!(mono_pcm16_header(16_000, 2_147_483_630).is_none());
    }

    #[test]
    fn duplicate_fmt_chunks_are_rejected_with_descriptive_error() {
        let wave = fixture(16_000);
        let fmt_chunk = &wave[12..36];
        let mut duplicated_fmt_before_data = wave[..36].to_vec();
        duplicated_fmt_before_data.extend_from_slice(fmt_chunk);
        duplicated_fmt_before_data.extend_from_slice(&wave[36..]);
        let riff_len = u32::try_from(duplicated_fmt_before_data.len() - 8).unwrap_or_default();
        duplicated_fmt_before_data[4..8].copy_from_slice(&riff_len.to_le_bytes());

        let error = mono_pcm16_data_checked(&duplicated_fmt_before_data, 16_000, 480_000)
            .err()
            .unwrap_or(WavError::InvalidRiffHeader);
        assert_eq!(error, WavError::DuplicateFormatChunk);
        assert!(error.to_string().contains("duplicate 'fmt '"));
        assert!(mono_pcm16_data(&duplicated_fmt_before_data, 16_000, 480_000).is_none());

        let mut duplicated_fmt_after_data = wave.clone();
        duplicated_fmt_after_data.extend_from_slice(fmt_chunk);
        let riff_len = u32::try_from(duplicated_fmt_after_data.len() - 8).unwrap_or_default();
        duplicated_fmt_after_data[4..8].copy_from_slice(&riff_len.to_le_bytes());
        assert_eq!(
            mono_pcm16_data_checked(&duplicated_fmt_after_data, 16_000, 480_000),
            Err(WavError::DuplicateFormatChunk)
        );
    }

    #[test]
    fn pcm16le_encoding_and_decoding_preserve_exact_samples_and_bounds() {
        let samples = [-2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0];
        let mut bytes = [0_u8; 16];
        assert_eq!(
            encode_pcm16le_into(&samples, &mut bytes),
            Ok([
                1, 128, 1, 128, 0, 192, 0, 0, 0, 0, 0, 64, 255, 127, 255, 127
            ]
            .as_slice())
        );
        assert_eq!(encode_pcm16le_sample(-2.0), -32767);
        assert_eq!(encode_pcm16le_sample(2.0), 32767);
        assert_eq!(
            encode_pcm16le_into(&[0.0], &mut [0_u8; 1]),
            Err(WavError::BufferTooSmall)
        );

        let mut decoded = Vec::new();
        let pcm = [0_u8, 0x80, 0xff, 0xff, 0, 0, 0xff, 0x7f];
        assert_eq!(
            decode_pcm16le_into(&pcm, &mut decoded),
            Ok([-1.0, -1.0 / 32768.0, 0.0, 32767.0 / 32768.0].as_slice())
        );
        assert_eq!(
            decode_pcm16le_into(&[0_u8; 3], &mut decoded),
            Err(WavError::InvalidDataLength)
        );
        assert!(decoded.is_empty());

        assert!(mono_pcm16_header_checked(&[0.0, 1.0], 24_000, 10).is_ok());
        assert_eq!(
            mono_pcm16_header_checked(&[f32::NAN], 24_000, 10),
            Err(WavError::InvalidSamples)
        );
    }
}
