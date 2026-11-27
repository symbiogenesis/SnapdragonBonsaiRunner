//! Explicit qualification consumer for the complete independent Rust ASR graph.
//!
//! This opt-in API loads no native inference library and never falls back to one.
//! Default application serving remains native until the source-bound numerical,
//! transcript, quality, cancellation and resource gates pass. See `rust-engine.json`.
#![forbid(unsafe_code)]

use crate::{
    MAX_CLIP_SECONDS, SAMPLE_RATE,
    decoder::{CachePrecision, Session},
    encoder::{Encoded, Encoder},
    frontend::{Frontend, FrontendScratch, MelSpectrogram},
    protocol::{
        DecodeStop, Prompt, PromptTokens, TranscribeOptions, completed_transcript,
        completed_transcript_with_options,
    },
    tokenizer::Tokenizer,
};
use std::{fs::OpenOptions, io, path::Path, sync::Arc};

const LIVE_CHUNK_SAMPLES: usize = 8 * SAMPLE_RATE;
const MIN_TAIL_SAMPLES: usize = 201;
const SHORT_CLIP_HINT_MAX_SAMPLES: usize = (SAMPLE_RATE * 3) / 2;
const CHAT_FRAME_TOKENS: usize = 15;

/// Independent arithmetic identity; not the native GGML/flash-attention identity.
pub const ARITHMETIC_IDENTITY: &str = "rust-asr-sdot-i8mm-v4-f16-kv-minimax-gelu-exp";

/// Borrowed numerical observations for an independently retained replay packet.
///
/// These are observations of this Rust provider only. The caller binds the
/// exact reviewed source/compiler/executable, verified model and PCM identities
/// before collecting them, and supplies independent expected observations and
/// tolerances. Neither an event nor a successful request establishes parity.
/// Values are borrowed for one synchronous callback; the engine retains no
/// capture, copies no cache and exposes no mutable model or session authority.
pub enum Observation<'a> {
    /// Complete normalized bin-major mel features, before encoder execution.
    Frontend(&'a MelSpectrogram),
    /// Complete trimmed time-major encoder output, before prompt injection.
    Encoder(&'a Encoded),
    /// Exact model-specific token IDs and audio-placeholder injection range.
    Prompt(&'a Prompt),
    /// One complete decoder state and tied-head logit vector, including EOS.
    Decoder {
        /// Zero for prefill, then the number of previously generated text IDs.
        generation_index: usize,
        /// Every vocabulary logit, in token-ID order.
        logits: &'a [f32],
        /// Lowest-ID finite greedy selection; EOS is observed but not decoded.
        token: u32,
        /// Complete committed K/V for every layer, before the request reset.
        state: &'a Session,
    },
}

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid or incomplete Rust Qwen3-ASR recognition.",
    )
}
fn check(cancel: &mut impl FnMut() -> bool) -> io::Result<()> {
    if cancel() {
        Err(io::Error::from(io::ErrorKind::Interrupted))
    } else {
        Ok(())
    }
}

/// Loaded immutable frontend/encoder/tokenizer and one isolated Rust decoder session.
pub struct RustRecognizer {
    frontend: Frontend,
    frontend_scratch: FrontendScratch,
    encoder: Encoder,
    tokenizer: Tokenizer,
    decoder: Session,
    logits_scratch: Vec<f32>,
    prompt_tokens: PromptTokens,
    context: usize,
    batch_threads: usize,
}

impl RustRecognizer {
    /// Load the verified pinned Qwen3-ASR GGUF using Rust tensor intake only.
    ///
    /// The caller verifies the model lock/digest and retains ancestor guards.
    /// All component loads share one immutable read-only file handle. F16 KV
    /// matches the native default's storage policy, with separately identified
    /// scalar arithmetic. No download, driver or native fallback is attempted.
    ///
    /// # Errors
    /// Rejects nonabsolute/reparse/nonregular files, unsupported metadata/storage,
    /// model geometry, nonfinite weights and I/O/allocation failures.
    pub fn load(path: &Path) -> io::Result<Self> {
        Self::load_with_threads_cancellable(path, None, &mut || false)
    }

    /// Load with cooperative checks during tensor intake and stage transitions.
    ///
    /// # Errors
    /// Returns [`Self::load`] errors or `Interrupted` on cancellation.
    pub fn load_cancellable(path: &Path, cancel: &mut impl FnMut() -> bool) -> io::Result<Self> {
        Self::load_with_threads_cancellable(path, None, cancel)
    }

    /// Load the verified pinned Qwen3-ASR GGUF with an optional CPU thread budget.
    ///
    /// `None` uses the Snapdragon X1/X2 cluster-aware ASR policy (`(8, 10)` on
    /// 12-core X1, `(8, 16)` on 18-core X2); `Some(n)` validates `1..=256`.
    ///
    /// # Errors
    /// Returns [`Self::load`] errors or `InvalidInput` for out-of-range thread counts.
    pub fn load_with_threads(path: &Path, threads: Option<usize>) -> io::Result<Self> {
        Self::load_with_threads_cancellable(path, threads, &mut || false)
    }

    /// Load with an explicit optional thread budget and cooperative cancellation checks.
    ///
    /// # Errors
    /// Returns [`Self::load_with_threads`] errors or `Interrupted` on cancellation.
    pub fn load_with_threads_cancellable(
        path: &Path,
        threads: Option<usize>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Self> {
        check(cancel)?;
        if !path.is_absolute() {
            return Err(invalid());
        }
        let (decode_threads, batch_threads) = crate::threads::select_asr_threads(
            threads,
            std::thread::available_parallelism()?.get(),
        )?;
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt as _;
            // Combine FILE_FLAG_OPEN_REPARSE_POINT with FILE_FLAG_SEQUENTIAL_SCAN.
            options
                .share_mode(1)
                .custom_flags(0x0020_0000 | 0x0800_0000);
        }
        #[cfg(not(all(windows, feature = "rust-qualification")))]
        let mut file = options.open(path)?;
        #[cfg(all(windows, feature = "rust-qualification"))]
        let file = options.open(path)?;
        let properties = file.metadata()?;
        if !properties.is_file() {
            return Err(invalid());
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt as _;
            if properties.file_attributes() & 0x400 != 0 {
                return Err(invalid());
            }
        }
        #[cfg(all(windows, feature = "rust-qualification"))]
        let (tokenizer, prompt_tokens, encoder, frontend, model) = {
            let mapping =
                Arc::new(bonsai_windows_process::MappedFile::from_file_without_prefetch(&file)?);
            drop(file);
            crate::intake::load_artifacts_mapped(&mapping, CachePrecision::F16, cancel)?
        };
        #[cfg(not(all(windows, feature = "rust-qualification")))]
        let (tokenizer, prompt_tokens, encoder, frontend, model) = crate::intake::load_artifacts(
            &mut file,
            properties.len(),
            CachePrecision::F16,
            cancel,
        )?;
        let encoder = encoder.with_threads(batch_threads)?;
        check(cancel)?;
        let model = model.with_threads(decode_threads, batch_threads)?;
        let (hidden, vocabulary, context) = model.dimensions();
        if encoder.output_width() != hidden || tokenizer.vocabulary_size() != vocabulary {
            return Err(invalid());
        }
        check(cancel)?;
        Ok(Self {
            frontend,
            frontend_scratch: FrontendScratch::default(),
            encoder,
            tokenizer,
            decoder: Arc::new(model).session(),
            logits_scratch: Vec::new(),
            prompt_tokens,
            context,
            batch_threads,
        })
    }

    /// Recognize one normalized mono 16 kHz clip, without native inference.
    ///
    /// # Errors
    /// Rejects invalid PCM, incomplete/repeating output, malformed text and graph faults.
    pub fn transcribe(&mut self, pcm: &[f32]) -> io::Result<String> {
        self.transcribe_cancellable(pcm, &mut || false)
    }

    /// Recognize complete text with cooperative cancellation and unconditional reset.
    ///
    /// PCM is finite `[-1,1]`, at least 201 samples (reflect padding), at most
    /// thirty seconds at 16 kHz. A request starts with empty session state and
    /// clears every layer and position after success, failure, cancellation or
    /// unwind. Only EOS-terminated, valid complete text is returned.
    ///
    /// # Errors
    /// Returns the validation conditions of [`Self::transcribe`], frontend and
    /// decoder failures, or `Interrupted`; no partial transcript is returned.
    pub fn transcribe_cancellable(
        &mut self,
        pcm: &[f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<String> {
        self.transcribe_observed(pcm, cancel, &mut |_| Ok(()))
    }

    /// Recognize with synchronous observations of every numerical boundary.
    ///
    /// The observer may stream original float bits, complete cache contents and
    /// IDs to caller-owned replay storage without accumulating duplicate state
    /// here. The caller must bind source, model, input and comparison identities
    /// as described by [`Observation`]. Captures from failed, cancelled or
    /// unwound requests remain incomplete; no partial transcript is returned.
    /// Callback work shares the caller's original absolute workload allowance.
    /// Ordinary serving neither installs an observer nor retains these values.
    ///
    /// # Errors
    /// Returns [`Self::transcribe_cancellable`] failures or an observer error.
    /// Every exit, including observer failure or unwind, clears all K/V and
    /// positions. Observation callbacks grant no native inference fallback.
    pub fn transcribe_observed(
        &mut self,
        pcm: &[f32],
        cancel: &mut impl FnMut() -> bool,
        observe: &mut impl FnMut(Observation<'_>) -> io::Result<()>,
    ) -> io::Result<String> {
        self.decoder.reset();
        let guard = ClearSession(&mut self.decoder);
        check(cancel)?;
        let mel = self.frontend.compute_with_scratch(
            pcm,
            self.batch_threads,
            &mut self.frontend_scratch,
            &mut *cancel,
        )?;
        observe(Observation::Frontend(&mel))?;
        check(cancel)?;
        let audio = self.encoder.encode(&mel, cancel)?;
        drop(mel);
        observe(Observation::Encoder(&audio))?;
        check(cancel)?;
        decode_audio(
            guard.0,
            &self.tokenizer,
            self.prompt_tokens,
            self.context,
            &audio,
            &mut self.logits_scratch,
            cancel,
            observe,
        )
    }

    /// Recognize one bounded clip using optional short-clip language hints (`<= 1.5s`)
    /// and repetition-suffix trimming when hitting the generation budget or repetition bar.
    ///
    /// # Errors
    /// Rejects invalid PCM, invalid language hint strings, graph faults, cancellation, or
    /// unrecovered incomplete generation.
    pub fn transcribe_with_options(
        &mut self,
        pcm: &[f32],
        options: &TranscribeOptions<'_>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<String> {
        self.decoder.reset();
        let guard = ClearSession(&mut self.decoder);
        check(cancel)?;
        let system_tokens = if pcm.len() <= SHORT_CLIP_HINT_MAX_SAMPLES
            && let Some(lang) = options.short_clip_language
        {
            let trimmed = lang.trim();
            if trimmed.is_empty() || trimmed.chars().any(char::is_control) {
                return Err(invalid());
            }
            self.tokenizer.encode(&format!("Languages: {trimmed}."))?
        } else {
            Vec::new()
        };
        let mel = self.frontend.compute_with_scratch(
            pcm,
            self.batch_threads,
            &mut self.frontend_scratch,
            &mut *cancel,
        )?;
        check(cancel)?;
        let audio = self.encoder.encode(&mel, cancel)?;
        drop(mel);
        check(cancel)?;
        let prompt =
            self.prompt_tokens
                .prompt_with_system(&system_tokens, audio.rows(), self.context)?;
        let budget = decode_budget(audio.rows(), prompt.ids().len(), self.context)?;
        guard
            .0
            .prefill_into(&prompt, audio.values(), &mut self.logits_scratch, cancel)?;
        decode_from_logits(
            guard.0,
            &self.tokenizer,
            &mut self.logits_scratch,
            budget,
            options.trim_repetition_on_stop,
            cancel,
            &mut |_| Ok(()),
        )
    }

    /// Start an incremental live-microphone capture session that prefills 8-second
    /// (`128,000`-sample) windows on disk/stream arrival before [`LiveCapture::finish`].
    ///
    /// Dropping [`LiveCapture`] at any stage unconditionally resets decoder KV state.
    ///
    /// # Errors
    /// Rejects cancellation or failure while prefilling the 9-token prompt prefix.
    pub fn begin_live_capture(
        &mut self,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<LiveCapture<'_>> {
        self.decoder.reset();
        let prefix = self.prompt_tokens.prefix_ids();
        if let Err(err) = self.decoder.prefill_tokens(&prefix, false, cancel) {
            self.decoder.reset();
            return Err(err);
        }
        Ok(LiveCapture {
            recognizer: self,
            pending_pcm: Vec::new(),
            total_samples: 0,
            prefilled_audio_rows: 0,
        })
    }
}

/// Incremental live-audio capture session holding exclusive access to one [`RustRecognizer`].
pub struct LiveCapture<'a> {
    recognizer: &'a mut RustRecognizer,
    pending_pcm: Vec<f32>,
    total_samples: usize,
    prefilled_audio_rows: usize,
}

impl LiveCapture<'_> {
    /// Append validated finite `[-1.0, 1.0]` 16 kHz mono samples and immediately prefill
    /// any completed 8-second window while retaining at least `201` samples for reflect padding.
    ///
    /// # Errors
    /// Rejects non-unit/non-finite PCM, clips exceeding `30` seconds, context exhaustion, or cancellation.
    pub fn push_pcm(
        &mut self,
        samples: &[f32],
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<usize> {
        check(cancel)?;
        let max_samples = SAMPLE_RATE * MAX_CLIP_SECONDS;
        let next_total = self
            .total_samples
            .checked_add(samples.len())
            .ok_or_else(invalid)?;
        if next_total > max_samples {
            return Err(invalid());
        }
        for chunk in samples.chunks(4096) {
            check(cancel)?;
            if !crate::packed::pcm_in_unit_range(chunk) {
                return Err(invalid());
            }
        }
        if !samples.is_empty() {
            self.pending_pcm
                .try_reserve(samples.len())
                .map_err(io::Error::other)?;
            self.pending_pcm.extend_from_slice(samples);
            self.total_samples = next_total;
        }
        let mut added_rows = 0_usize;
        while self.pending_pcm.len() >= LIVE_CHUNK_SAMPLES + MIN_TAIL_SAMPLES {
            let mel = self.recognizer.frontend.compute_with_scratch(
                &self.pending_pcm[..LIVE_CHUNK_SAMPLES],
                self.recognizer.batch_threads,
                &mut self.recognizer.frontend_scratch,
                &mut *cancel,
            )?;
            let encoded = self.recognizer.encoder.encode(&mel, cancel)?;
            drop(mel);
            let rows = encoded.rows();
            let projected_rows = self
                .prefilled_audio_rows
                .checked_add(rows)
                .ok_or_else(invalid)?;
            let projected_prompt = CHAT_FRAME_TOKENS
                .checked_add(projected_rows)
                .ok_or_else(invalid)?;
            let _ = decode_budget(projected_rows, projected_prompt, self.recognizer.context)?;
            self.recognizer
                .decoder
                .prefill_audio_chunk(encoded.values(), rows, cancel)?;
            self.prefilled_audio_rows = projected_rows;
            added_rows += rows;
            self.pending_pcm.drain(..LIVE_CHUNK_SAMPLES);
        }
        Ok(added_rows)
    }

    /// Encode the final tail audio, prefill the 6-token suffix, and decode the complete transcript.
    ///
    /// # Errors
    /// Rejects clips under `201` tail samples, decoding faults, repetition, or cancellation.
    pub fn finish(self, cancel: &mut impl FnMut() -> bool) -> io::Result<String> {
        check(cancel)?;
        if self.pending_pcm.len() < MIN_TAIL_SAMPLES {
            return Err(invalid());
        }
        let mel = self.recognizer.frontend.compute_with_scratch(
            &self.pending_pcm,
            self.recognizer.batch_threads,
            &mut self.recognizer.frontend_scratch,
            &mut *cancel,
        )?;
        let encoded = self.recognizer.encoder.encode(&mel, cancel)?;
        drop(mel);
        let total_audio = self
            .prefilled_audio_rows
            .checked_add(encoded.rows())
            .ok_or_else(invalid)?;
        let total_prompt = CHAT_FRAME_TOKENS
            .checked_add(total_audio)
            .ok_or_else(invalid)?;
        let budget = decode_budget(total_audio, total_prompt, self.recognizer.context)?;
        self.recognizer
            .decoder
            .prefill_audio_chunk(encoded.values(), encoded.rows(), cancel)?;
        let suffix = self.recognizer.prompt_tokens.suffix_ids();
        self.recognizer.decoder.prefill_tokens_into(
            &suffix,
            Some(&mut self.recognizer.logits_scratch),
            cancel,
        )?;
        decode_from_logits(
            &mut self.recognizer.decoder,
            &self.recognizer.tokenizer,
            &mut self.recognizer.logits_scratch,
            budget,
            false,
            cancel,
            &mut |_| Ok(()),
        )
    }
}

impl Drop for LiveCapture<'_> {
    fn drop(&mut self) {
        self.recognizer.decoder.reset();
    }
}

struct ClearSession<'a>(&'a mut Session);
impl Drop for ClearSession<'_> {
    fn drop(&mut self) {
        self.0.reset();
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shares prefill and decode loop with caller-owned reusable logits scratch."
)]
fn decode_audio(
    session: &mut Session,
    tokenizer: &Tokenizer,
    tokens: PromptTokens,
    context: usize,
    audio: &Encoded,
    logits: &mut Vec<f32>,
    cancel: &mut impl FnMut() -> bool,
    observe: &mut impl FnMut(Observation<'_>) -> io::Result<()>,
) -> io::Result<String> {
    let prompt = tokens.automatic_prompt(audio.rows(), context)?;
    let budget = decode_budget(audio.rows(), prompt.ids().len(), context)?;
    observe(Observation::Prompt(&prompt))?;
    check(cancel)?;
    session.prefill_into(&prompt, audio.values(), logits, cancel)?;
    decode_from_logits(session, tokenizer, logits, budget, false, cancel, observe)
}

fn decode_from_logits(
    session: &mut Session,
    tokenizer: &Tokenizer,
    logits: &mut Vec<f32>,
    budget: usize,
    trim_on_stop: bool,
    cancel: &mut impl FnMut() -> bool,
    observe: &mut impl FnMut(Observation<'_>) -> io::Result<()>,
) -> io::Result<String> {
    session.reserve_decode(budget)?;
    let mut generated = Vec::new();
    generated
        .try_reserve_exact(budget)
        .map_err(io::Error::other)?;
    for index in 0..budget {
        check(cancel)?;
        let token = argmax(logits)?;
        observe(Observation::Decoder {
            generation_index: index,
            logits,
            token,
            state: session,
        })?;
        check(cancel)?;
        if token == tokenizer.eos() {
            let raw = tokenizer.decode(&generated)?;
            let transcript = if trim_on_stop {
                completed_transcript_with_options(&raw, DecodeStop::EndOfSequence, true)?
            } else {
                completed_transcript(&raw, DecodeStop::EndOfSequence)?.to_owned()
            };
            check(cancel)?;
            return Ok(transcript);
        }
        generated.push(token);
        if repeating(&generated) {
            if trim_on_stop {
                let raw = tokenizer.decode(&generated)?;
                let transcript =
                    completed_transcript_with_options(&raw, DecodeStop::Repetition, true)?;
                check(cancel)?;
                return Ok(transcript);
            }
            return Err(invalid());
        }
        if index + 1 < budget {
            session.step_into(token, logits, cancel)?;
        }
    }
    if trim_on_stop {
        let raw = tokenizer.decode(&generated)?;
        let transcript = completed_transcript_with_options(&raw, DecodeStop::TokenLimit, true)?;
        check(cancel)?;
        return Ok(transcript);
    }
    Err(invalid())
}

fn decode_budget(audio_rows: usize, prompt: usize, context: usize) -> io::Result<usize> {
    // The pinned frontend has 80ms per encoder token and predicts 12 text
    // tokens/second. Integer 24/25 is the same positive floor, without a host
    // floating conversion. Independent chunk rounding is retained in audio_rows.
    let predicted = audio_rows.checked_mul(24).ok_or_else(invalid)? / 25;
    let room = context.checked_sub(prompt).ok_or_else(invalid)?;
    if room < 256 {
        return Err(invalid());
    }
    Ok(predicted.max(256).min(room))
}

fn argmax(logits: &[f32]) -> io::Result<u32> {
    let best = crate::packed::argmax_finite_f32(logits).ok_or_else(invalid)?;
    u32::try_from(best).map_err(|_| invalid())
}

fn repeating(ids: &[u32]) -> bool {
    for block in 1_usize..=128 {
        let span = (8 * block).clamp(64, 192);
        let copies = span.div_ceil(block).max(4);
        let extent = block * copies;
        if extent <= ids.len()
            && ids[ids.len() - extent + block..]
                .iter()
                .zip(&ids[ids.len() - extent..ids.len() - block])
                .all(|(a, b)| a == b)
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
#[path = "rust_engine/tests.rs"]
pub(crate) mod tests;
