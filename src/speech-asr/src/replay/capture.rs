//! Provider-neutral writing of original complete ASR observation packets.

use std::{
    io::{self, BufWriter, Write},
    ops::Range,
};

use super::{Binding, MAGIC, Provider, Stage, Tolerance, Tolerances, check_cancel, invalid};
use crate::rust_engine::ARITHMETIC_IDENTITY;

/// Independently reviewed model geometry, supplied before any observation.
#[derive(Clone, Copy)]
pub struct CaptureLayout {
    /// Width of every trimmed encoder row.
    pub encoder_width: usize,
    /// Complete decoder vocabulary size.
    pub vocabulary: usize,
    /// Complete decoder layer count; omitted layers cannot establish a capture.
    pub decoder_layers: usize,
    /// Key/value scalars per position (`kv_heads * head_dim`).
    pub kv_width: usize,
    /// Verified model's EOS token ID, which must be the final decision.
    pub eos_token: u32,
}

/// Borrowed original cache storage, widened one scalar at a time on output.
#[derive(Clone, Copy)]
pub enum CacheValues<'a> {
    /// Original F32 values, retaining every bit.
    F32(&'a [f32]),
    /// Original binary16 words, widened exactly without allocating an F32 cache.
    F16(&'a [u16]),
}

impl CacheValues<'_> {
    /// Number of scalar elements retained by this cache view.
    #[must_use]
    pub const fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) => values.len(),
        }
    }

    /// Whether this cache view contains zero scalars.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Widen committed cache scalars into a caller-owned `f32` vector.
    #[must_use]
    pub fn to_f32_vec(self) -> Vec<f32> {
        match self {
            Self::F32(values) => values.to_vec(),
            Self::F16(words) => {
                let mut out = vec![0.0_f32; words.len()];
                crate::packed::widen_f16_slice(words, &mut out);
                out
            }
        }
    }

    fn bits(self, index: usize) -> u32 {
        match self {
            Self::F32(values) => values[index].to_bits(),
            Self::F16(values) => bonsai_ternary::half_to_float(values[index]).to_bits(),
        }
    }
}

/// One original committed layer in `[position, kv_head, head_dim]` order.
pub struct LayerCache<'a> {
    /// Every committed key, including the current position.
    pub keys: CacheValues<'a>,
    /// Every committed value, including the current position.
    pub values: CacheValues<'a>,
}

/// One original generation observation, without a Rust decoder dependency.
pub struct DecoderSnapshot<'a> {
    /// Zero for prefill, then contiguous generation indices through EOS.
    pub generation_index: usize,
    /// Complete raw logits in vocabulary order, before greedy selection.
    pub logits: &'a [f32],
    /// Lowest-ID finite greedy decision made by the original provider.
    pub token: u32,
    /// Complete committed position count, including the current position.
    pub positions: usize,
    /// Every layer in original ascending layer order.
    pub layers: &'a [LayerCache<'a>],
}

/// Single-use, bounded writer for independent original `ASRREPL1` observations.
///
/// This adapter accepts borrowed slices from any reviewed provider, including a
/// future native observation ABI. It does not execute an engine, authenticate
/// provenance or make a native API expose otherwise unavailable logits/KV. The
/// caller verifies bindings/layout, holds source/model/input guards and owns
/// bounded output I/O. Cancellation and all writes share the original absolute
/// workload allowance. A blocking `Write` cannot be interrupted cooperatively.
///
/// Only [`Self::finish`] emits a terminal record, after the independently
/// successful result and EOS. Refusal, write failure or cancellation is sticky;
/// retained partial bytes must never be admitted as a completed capture. The
/// writer owns one fixed 16 KiB buffer and copies no tensor or complete cache.
#[must_use]
pub struct CaptureWriter<W: Write> {
    output: BufWriter<W>,
    layout: CaptureLayout,
    stage: Stage,
    audio_rows: usize,
    prompt_positions: usize,
    generation: usize,
    eos: bool,
}

impl<W: Write> CaptureWriter<W> {
    /// Write a source/model/PCM/tolerance-bound header, without engine execution.
    ///
    /// # Errors
    /// Rejects invalid model geometry, empty identities, unsupported candidate
    /// arithmetic, invalid tolerances or output I/O failure.
    pub fn new(
        output: W,
        binding: &Binding<'_>,
        tolerance: Tolerances,
        layout: CaptureLayout,
    ) -> io::Result<Self> {
        if !(1..=8192).contains(&layout.encoder_width)
            || !(1..=262_144).contains(&layout.vocabulary)
            || !(1..=128).contains(&layout.decoder_layers)
            || !(1..=32_768).contains(&layout.kv_width)
            || usize::try_from(layout.eos_token).map_err(|_| invalid())? >= layout.vocabulary
            || binding.candidate.arithmetic != ARITHMETIC_IDENTITY
            || ![
                tolerance.frontend,
                tolerance.encoder,
                tolerance.logits,
                tolerance.cache,
            ]
            .into_iter()
            .all(Tolerance::valid)
        {
            return Err(invalid());
        }
        let mut capture = Self {
            output: BufWriter::with_capacity(16 * 1024, output),
            layout,
            stage: Stage::Frontend,
            audio_rows: 0,
            prompt_positions: 0,
            generation: 0,
            eos: false,
        };
        capture.output.write_all(MAGIC)?;
        capture.provider(&binding.reference)?;
        capture.provider(&binding.candidate)?;
        capture.digest(&binding.model)?;
        capture.digest(&binding.pcm)?;
        for rule in [
            tolerance.frontend,
            tolerance.encoder,
            tolerance.logits,
            tolerance.cache,
        ] {
            match rule {
                Tolerance::Bits => capture.output.write_all(&[0])?,
                Tolerance::Finite { absolute, relative } => {
                    capture.output.write_all(&[1])?;
                    capture.output.write_all(&absolute.to_le_bytes())?;
                    capture.output.write_all(&relative.to_le_bytes())?;
                }
            }
        }
        Ok(capture)
    }

    /// Write all normalized bin-major mel features before encoder observations.
    ///
    /// # Errors
    /// Rejects wrong order/geometry, cancellation or output I/O failure.
    pub fn frontend(
        &mut self,
        frames: usize,
        values: &[f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        self.record(Stage::Frontend, |capture| {
            check_cancel(cancel)?;
            if !(1..=3000).contains(&frames) || values.len() != frames * crate::frontend::MEL_BINS {
                return Err(invalid());
            }
            capture.output.write_all(&[1])?;
            capture.number(frames)?;
            capture.floats(values, cancel)?;
            capture.stage = Stage::Encoder;
            Ok(())
        })
    }

    /// Write every trimmed encoder row in time-major order.
    ///
    /// # Errors
    /// Rejects wrong order/geometry, cancellation or output I/O failure.
    pub fn encoder(
        &mut self,
        rows: usize,
        values: &[f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        self.record(Stage::Encoder, |capture| {
            check_cancel(cancel)?;
            if !(1..=8192).contains(&rows) || values.len() != rows * capture.layout.encoder_width {
                return Err(invalid());
            }
            capture.output.write_all(&[2])?;
            capture.number(rows)?;
            capture.number(capture.layout.encoder_width)?;
            capture.floats(values, cancel)?;
            capture.audio_rows = rows;
            capture.stage = Stage::Prompt;
            Ok(())
        })
    }

    /// Write the complete original prompt and its sole audio-injection range.
    ///
    /// # Errors
    /// Rejects wrong order, absent/mismatched audio rows, out-of-vocabulary IDs,
    /// a prompt without the 256-position output reserve, cancellation or I/O.
    pub fn prompt(
        &mut self,
        ids: &[u32],
        audio: Range<usize>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        self.record(Stage::Prompt, |capture| {
            check_cancel(cancel)?;
            if ids.is_empty()
                || ids.len() > 8192 - 256
                || audio.start >= audio.end
                || audio.end > ids.len()
                || audio.len() != capture.audio_rows
                || ids.iter().any(|id| {
                    usize::try_from(*id).map_or(true, |id| id >= capture.layout.vocabulary)
                })
            {
                return Err(invalid());
            }
            capture.output.write_all(&[3])?;
            capture.number(ids.len())?;
            for chunk in ids.chunks(4096) {
                check_cancel(cancel)?;
                for id in chunk {
                    capture.output.write_all(&id.to_le_bytes())?;
                }
            }
            capture.number(audio.start)?;
            capture.number(audio.end)?;
            capture.prompt_positions = ids.len();
            capture.stage = Stage::Decoder;
            Ok(())
        })
    }

    /// Write every raw logit and every committed layer at one generation step.
    ///
    /// # Errors
    /// Rejects missing/extra layers or values, skipped positions, nonfinite
    /// logits, an incorrect greedy ID, observations after EOS, cancellation or I/O.
    pub fn decoder(
        &mut self,
        snapshot: &DecoderSnapshot<'_>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        self.record(Stage::Decoder, |capture| {
            check_cancel(cancel)?;
            let positions = capture
                .prompt_positions
                .checked_add(capture.generation)
                .ok_or_else(invalid)?;
            let count = positions
                .checked_mul(capture.layout.kv_width)
                .ok_or_else(invalid)?;
            if capture.eos
                || snapshot.generation_index != capture.generation
                || snapshot.positions != positions
                || positions > 8192
                || snapshot.logits.len() != capture.layout.vocabulary
                || snapshot.layers.len() != capture.layout.decoder_layers
                || snapshot
                    .layers
                    .iter()
                    .any(|layer| layer.keys.len() != count || layer.values.len() != count)
                || snapshot.logits.iter().any(|value| !value.is_finite())
            {
                return Err(invalid());
            }
            let mut best = 0;
            for index in 1..snapshot.logits.len() {
                if snapshot.logits[index] > snapshot.logits[best] {
                    best = index;
                }
            }
            if usize::try_from(snapshot.token).map_err(|_| invalid())? != best {
                return Err(invalid());
            }
            capture.output.write_all(&[4])?;
            capture.number(snapshot.generation_index)?;
            capture.floats(snapshot.logits, cancel)?;
            capture.output.write_all(&snapshot.token.to_le_bytes())?;
            capture.number(positions)?;
            capture.number(snapshot.layers.len())?;
            for (index, layer) in snapshot.layers.iter().enumerate() {
                capture.number(index)?;
                capture.cache_values(layer.keys, cancel)?;
                capture.cache_values(layer.values, cancel)?;
            }
            check_cancel(cancel)?;
            capture.eos = snapshot.token == capture.layout.eos_token;
            capture.generation += 1;
            Ok(())
        })
    }

    /// Finalize only an independently successful complete provider result.
    ///
    /// Pass the actual engine result, after its cleanup/reset, rather than a
    /// partial transcript or a fabricated success. This adapter authenticates
    /// neither the result nor EOS identity. Flush must also succeed before the
    /// caller retains/publishes the original packet as complete.
    ///
    /// # Errors
    /// Propagates provider failure, prior refusal, missing EOS, invalid complete
    /// transcript, cancellation and terminal-write/flush failure.
    pub fn finish(
        mut self,
        result: io::Result<&str>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<W> {
        let transcript = result?;
        self.record(Stage::Decoder, |capture| {
            check_cancel(cancel)?;
            if !capture.eos || capture.generation == 0 {
                return Err(invalid());
            }
            crate::protocol::validate_transcript(transcript)?;
            capture.output.write_all(&[5])?;
            capture.number(capture.generation)?;
            capture.number(transcript.len())?;
            for chunk in transcript.as_bytes().chunks(1024) {
                check_cancel(cancel)?;
                capture.output.write_all(chunk)?;
            }
            capture.output.flush()?;
            check_cancel(cancel)
        })?;
        self.output
            .into_inner()
            .map_err(io::IntoInnerError::into_error)
    }

    fn record(
        &mut self,
        expected: Stage,
        write: impl FnOnce(&mut Self) -> io::Result<()>,
    ) -> io::Result<()> {
        if self.stage != expected {
            self.stage = Stage::Failed;
            return Err(invalid());
        }
        // Mark failure before entering caller-owned I/O/cancellation so caught
        // callback unwinds cannot restore authority to complete this packet.
        self.stage = Stage::Failed;
        let result = write(self);
        if result.is_ok() && self.stage == Stage::Failed {
            self.stage = expected;
        }
        if result.is_err() {
            self.stage = Stage::Failed;
        }
        result
    }

    fn number(&mut self, value: usize) -> io::Result<()> {
        self.output
            .write_all(&u64::try_from(value).map_err(|_| invalid())?.to_le_bytes())
    }

    fn digest(&mut self, value: &[u8]) -> io::Result<()> {
        if value.iter().all(|byte| *byte == 0) {
            return Err(invalid());
        }
        self.output.write_all(value)
    }

    fn text(&mut self, value: &str) -> io::Result<()> {
        if value.is_empty() || value.len() > 1024 || value.chars().any(char::is_control) {
            return Err(invalid());
        }
        self.number(value.len())?;
        self.output.write_all(value.as_bytes())
    }

    fn provider(&mut self, value: &Provider<'_>) -> io::Result<()> {
        self.digest(&value.revision)?;
        self.text(value.compiler)?;
        self.text(value.target)?;
        self.digest(&value.executable)?;
        self.text(value.arithmetic)
    }

    fn floats(&mut self, values: &[f32], cancel: &mut impl FnMut() -> bool) -> io::Result<()> {
        self.cache_values(CacheValues::F32(values), cancel)
    }

    fn cache_values(
        &mut self,
        values: CacheValues<'_>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        self.number(values.len())?;
        for start in (0..values.len()).step_by(4096) {
            check_cancel(cancel)?;
            for index in start..(start + 4096).min(values.len()) {
                self.output.write_all(&values.bits(index).to_le_bytes())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CacheValues;

    #[test]
    fn rust_asr_capture_half_cache_preserves_zero_subnormal_limits_and_exceptional_bits() {
        let raw = [
            0, 0x8000, 1, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7d01, 0xfe01,
        ];
        let expected = [
            0,
            0x8000_0000,
            0x3380_0000,
            0x387f_c000,
            0x3880_0000,
            0x3f80_0000,
            0x477f_e000,
            0x7f80_0000,
            0xff80_0000,
            0x7fa0_2000,
            0xffc0_2000,
        ];
        let values = CacheValues::F16(&raw);
        for (index, expected) in expected.into_iter().enumerate() {
            assert_eq!(values.bits(index), expected);
        }
    }
}
