//! Explicit qualification consumer for the complete independent Rust ASR graph.
//!
//! This opt-in API loads no native inference library and never falls back to one.
//! Default application serving remains native until the source-bound numerical,
//! transcript, quality, cancellation and resource gates pass. See `rust-engine.json`.
#![forbid(unsafe_code)]

use crate::{
    decoder::{CachePrecision, Session},
    encoder::{Encoded, Encoder},
    frontend::{Frontend, MelSpectrogram},
    protocol::{DecodeStop, Prompt, PromptTokens, completed_transcript},
    tokenizer::Tokenizer,
};
use std::{fs::OpenOptions, io, path::Path, sync::Arc};

/// Independent arithmetic identity; not the native GGML/flash-attention identity.
pub const ARITHMETIC_IDENTITY: &str = "rust-asr-scalar-f32-v2-f16-conv-kv-f64-erfc-gelu";

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
    encoder: Encoder,
    tokenizer: Tokenizer,
    decoder: Session,
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
        let mut file = options.open(path)?;
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
            encoder,
            tokenizer,
            decoder: Arc::new(model).session(),
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
        let mel = self
            .frontend
            .compute_with_threads(pcm, self.batch_threads, &mut *cancel)?;
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
            cancel,
            observe,
        )
    }
}

struct ClearSession<'a>(&'a mut Session);
impl Drop for ClearSession<'_> {
    fn drop(&mut self) {
        self.0.reset();
    }
}

fn decode_audio(
    session: &mut Session,
    tokenizer: &Tokenizer,
    tokens: PromptTokens,
    context: usize,
    audio: &Encoded,
    cancel: &mut impl FnMut() -> bool,
    observe: &mut impl FnMut(Observation<'_>) -> io::Result<()>,
) -> io::Result<String> {
    let prompt = tokens.automatic_prompt(audio.rows(), context)?;
    let budget = decode_budget(audio.rows(), prompt.ids().len(), context)?;
    observe(Observation::Prompt(&prompt))?;
    check(cancel)?;
    let mut logits = session.prefill(&prompt, audio.values(), cancel)?;
    session.reserve_decode(budget)?;
    let mut generated = Vec::new();
    generated
        .try_reserve_exact(budget)
        .map_err(io::Error::other)?;
    for index in 0..budget {
        check(cancel)?;
        let token = argmax(&logits)?;
        observe(Observation::Decoder {
            generation_index: index,
            logits: &logits,
            token,
            state: session,
        })?;
        check(cancel)?;
        if token == tokenizer.eos() {
            let raw = tokenizer.decode(&generated)?;
            let transcript = completed_transcript(&raw, DecodeStop::EndOfSequence)?.to_owned();
            check(cancel)?;
            return Ok(transcript);
        }
        generated.push(token);
        if repeating(&generated) {
            return Err(invalid());
        }
        if index + 1 < budget {
            logits = session.step(token, cancel)?;
        }
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
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(invalid());
    }
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
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
