//! Native-independent Qwen3-ASR prompt, audio-layout and completion contracts.
//!
//! These helpers prepare the automatic-language, empty-system-message subset
//! exposed by Bonsai. They do not execute the audio encoder or decoder. Model
//! parity and replacement of the current native engine remain separate gates.
//! Source identity and MIT attribution are recorded in `rust-protocol.json`
//! and `LICENSE.transcribe` beside this crate's README.

use std::{io, ops::Range};

const CHAT_FRAME_TOKENS: usize = 15;
const GENERATION_RESERVE: usize = 256;
const SESSION_CONTEXT: usize = 8192;
const MAX_MEL_FRAMES: usize = 3000;
const MAX_TEXT_BYTES: usize = 32 * 1024;

/// Resolved model-specific chat and audio token IDs.
///
/// Resolve these from the verified tokenizer and GGUF audio-token metadata;
/// numeric IDs are deliberately not copied from another Qwen vocabulary.
#[derive(Clone, Copy, Debug)]
pub struct PromptTokens {
    roles: [u32; 6],
    audio: [u32; 3],
}

impl PromptTokens {
    /// Validate IDs in `[im_start, im_end, newline, system, user, assistant]`
    /// and `[audio_start, audio_pad, audio_end]` order.
    ///
    /// `newline` names the byte-level spelling U+010A, rather than a literal
    /// newline spelling in the tokenizer's encoded vocabulary.
    ///
    /// # Errors
    /// Rejects an empty vocabulary or an ID outside the verified vocabulary.
    pub fn new(roles: [u32; 6], audio: [u32; 3], vocabulary_size: u32) -> io::Result<Self> {
        if vocabulary_size == 0 || roles.iter().chain(&audio).any(|id| *id >= vocabulary_size) {
            return Err(invalid("Qwen3-ASR prompt token is outside the vocabulary."));
        }
        Ok(Self { roles, audio })
    }

    /// Build the automatic-language prompt and its encoder replacement range.
    ///
    /// Audio embeddings replace only the `audio_pad` rows. The caller must
    /// retain the complete generated prefix and reserve at least 256 tokens
    /// before executing either the encoder or decoder.
    ///
    /// # Errors
    /// Rejects empty audio, arithmetic overflow or a prompt that cannot fit the
    /// selected model/session context with its generation reserve. The session
    /// ceiling cannot exceed Bonsai's current 8,192-token ASR context.
    pub fn automatic_prompt(self, audio_tokens: usize, model_context: usize) -> io::Result<Prompt> {
        self.prompt_with_system(&[], audio_tokens, model_context)
    }

    /// Return the 9-token automatic-language prefix preceding audio feature injection.
    #[must_use]
    pub const fn prefix_ids(self) -> [u32; 9] {
        let [start, end, newline, system, user, _assistant] = self.roles;
        let [audio_start, _, _] = self.audio;
        [
            start,
            system,
            newline,
            end,
            newline,
            start,
            user,
            newline,
            audio_start,
        ]
    }

    /// Return the 6-token suffix terminating audio injection and opening assistant generation.
    #[must_use]
    pub const fn suffix_ids(self) -> [u32; 6] {
        let [start, end, newline, _system, _user, assistant] = self.roles;
        let [_, _, audio_end] = self.audio;
        [audio_end, end, newline, start, assistant, newline]
    }

    /// Build a prompt with optional system-message token IDs (`Languages: <Name>.`)
    /// inserted inside the system turn before audio placeholders.
    ///
    /// # Errors
    /// Rejects zero audio tokens, arithmetic overflow, or context+reserve exhaustion.
    pub fn prompt_with_system(
        self,
        system_tokens: &[u32],
        audio_tokens: usize,
        model_context: usize,
    ) -> io::Result<Prompt> {
        let context = model_context.min(SESSION_CONTEXT);
        let total = audio_tokens
            .checked_add(CHAT_FRAME_TOKENS)
            .and_then(|n| n.checked_add(system_tokens.len()))
            .filter(|total| {
                total
                    .checked_add(GENERATION_RESERVE)
                    .is_some_and(|n| n <= context)
            })
            .ok_or_else(|| invalid("Qwen3-ASR audio and output reserve exceed the context."))?;
        if audio_tokens == 0 {
            return Err(invalid(
                "Qwen3-ASR requires at least one encoded audio row.",
            ));
        }
        let [start, end, newline, system, user, assistant] = self.roles;
        let [audio_start, audio_pad, audio_end] = self.audio;
        let mut ids = Vec::new();
        ids.try_reserve_exact(total)
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        ids.extend([start, system, newline]);
        ids.extend_from_slice(system_tokens);
        ids.extend([end, newline]);
        ids.extend([start, user, newline, audio_start]);
        let first_audio = ids.len();
        ids.resize(first_audio + audio_tokens, audio_pad);
        let audio_rows = first_audio..ids.len();
        ids.extend([audio_end, end, newline, start, assistant, newline]);
        Ok(Prompt { ids, audio_rows })
    }
}

/// Complete prompt tokens and the exact rows replaced with encoded audio.
#[derive(Debug)]
pub struct Prompt {
    ids: Vec<u32>,
    audio_rows: Range<usize>,
}

impl Prompt {
    /// Borrow all prompt token IDs, including framing and audio placeholders.
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Return the half-open audio-placeholder range in [`Self::ids`].
    #[must_use]
    pub fn audio_rows(&self) -> Range<usize> {
        self.audio_rows.clone()
    }
}

/// Shape plan for three stride-two, kernel-three, pad-one convolutions.
///
/// Every chunk is padded independently; only the last chunk's trailing output
/// rows are removed. Applying `ceil(total_frames / 8)` to the whole utterance
/// would be incorrect when chunk width is not a multiple of eight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncoderLayout {
    /// Real mel frames in the complete input.
    pub mel_frames: usize,
    /// Padded mel frames per independently convolved chunk.
    pub mel_per_chunk: usize,
    /// Number of padded input chunks.
    pub chunks: usize,
    /// Real mel frames in the final chunk.
    pub last_chunk_mel: usize,
    /// Output rows in every complete chunk.
    pub rows_per_chunk: usize,
    /// Real output rows retained from the final chunk.
    pub last_chunk_rows: usize,
    /// Padded output extent before the last chunk's padding is removed.
    pub padded_rows: usize,
    /// Complete pad-trimmed encoder output rows, used as prompt audio tokens.
    pub audio_rows: usize,
}

impl EncoderLayout {
    /// Compute bounded chunk and prompt geometry from verified model metadata.
    ///
    /// # Errors
    /// Rejects zero or more than 3,000 mel frames (the thirty-second input
    /// contract), zero window size and overflow in the doubled chunk width.
    pub fn new(mel_frames: usize, encoder_window: usize) -> io::Result<Self> {
        let mel_per_chunk = encoder_window
            .checked_mul(2)
            .filter(|width| *width > 0 && *width <= MAX_MEL_FRAMES)
            .ok_or_else(|| invalid("Qwen3-ASR encoder window is invalid."))?;
        if mel_frames == 0 || mel_frames > MAX_MEL_FRAMES {
            return Err(invalid(
                "Qwen3-ASR mel extent is outside the clip contract.",
            ));
        }
        let chunks = mel_frames.div_ceil(mel_per_chunk);
        let last_chunk_mel = mel_frames - (chunks - 1) * mel_per_chunk;
        let rows_per_chunk = mel_per_chunk.div_ceil(8);
        let last_chunk_rows = last_chunk_mel.div_ceil(8);
        let padded_rows = chunks * rows_per_chunk;
        let audio_rows = (chunks - 1) * rows_per_chunk + last_chunk_rows;
        Ok(Self {
            mel_frames,
            mel_per_chunk,
            chunks,
            last_chunk_mel,
            rows_per_chunk,
            last_chunk_rows,
            padded_rows,
            audio_rows,
        })
    }
}

/// Why generation stopped; only EOS establishes complete recognition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeStop {
    /// The selected model's end-of-sequence token was observed.
    EndOfSequence,
    /// Generation exhausted its reserved token budget.
    TokenLimit,
    /// A repetition guard stopped decoding.
    Repetition,
    /// The request was cancelled or its owner disconnected.
    Cancelled,
}

/// Borrow completed transcript bytes after stripping the first ASR envelope.
///
/// Automatic language detection emits `language X<asr_text>text`. As in the
/// pinned provider, absence of the separator preserves the complete output;
/// whitespace and additional separators inside the transcript are preserved.
/// The decoder must remove its trailing EOS token before detokenizing.
///
/// # Errors
/// Rejects incomplete/cancelled decoding, NUL bytes and oversized final text.
pub fn completed_transcript(raw_text: &str, stop: DecodeStop) -> io::Result<&str> {
    match stop {
        DecodeStop::Cancelled => return Err(io::Error::from(io::ErrorKind::Interrupted)),
        DecodeStop::TokenLimit | DecodeStop::Repetition => {
            return Err(invalid("Qwen3-ASR did not produce a complete transcript."));
        }
        DecodeStop::EndOfSequence => {}
    }
    let text = raw_text
        .split_once("<asr_text>")
        .map_or(raw_text, |(_, text)| text);
    validate_transcript(text)?;
    Ok(text)
}

/// Recognition policy options extending the baseline automatic-language contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TranscribeOptions<'a> {
    /// Optional language name (`"English"`, etc.) injected as `Languages: <Name>.` on clips `<= 1.5s`.
    pub short_clip_language: Option<&'a str>,
    /// Recover trimmed non-repeating prefix text on `TokenLimit`/`Repetition` stops instead of failing.
    pub trim_repetition_on_stop: bool,
}

/// Trim consecutive repeating suffix patterns (`>= 3` multi-char loops or `>= 6` short loops),
/// retaining the first occurrence and preserving UTF-8 char boundaries (`transcribe.cpp` guard).
#[must_use]
pub fn trim_repeating_text_suffix(text: &str) -> &str {
    let mut char_bounds: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let total_chars = char_bounds.len();
    if total_chars < 6 {
        return text;
    }
    char_bounds.push(text.len());
    let max_unit = (total_chars / 3).min(128);
    for unit in 1..=max_unit {
        let min_reps = if unit >= 4 { 3 } else { 6 };
        let min_span = unit * min_reps;
        if min_span > total_chars {
            continue;
        }
        let unit_start = char_bounds[total_chars - unit];
        let pattern = &text[unit_start..];
        let mut reps = 1_usize;
        let mut cur_end = total_chars - unit;
        while cur_end >= unit {
            let prev_start = cur_end - unit;
            if &text[char_bounds[prev_start]..char_bounds[cur_end]] == pattern {
                reps += 1;
                cur_end = prev_start;
            } else {
                break;
            }
        }
        if reps >= min_reps && reps * unit >= 6 {
            let keep_end = char_bounds[cur_end + unit];
            return text[..keep_end].trim_end();
        }
    }
    text
}

/// Join and normalize multiple transcript segments (stripping any leading `language X<asr_text>` tag).
///
/// # Errors
/// Rejects NUL bytes or combined text exceeding 32 KiB.
pub fn normalize_multi_segment_transcript<'a, I: IntoIterator<Item = &'a str>>(
    segments: I,
) -> io::Result<String> {
    let mut joined = String::new();
    for raw in segments {
        let body = raw.split_once("<asr_text>").map_or(raw, |(_, text)| text);
        let trimmed = body.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !joined.is_empty() {
            joined.push(' ');
        }
        joined.push_str(trimmed);
        if joined.len() > MAX_TEXT_BYTES {
            return Err(invalid("Qwen3-ASR returned an oversized transcript."));
        }
    }
    validate_transcript(&joined)?;
    Ok(joined)
}

/// Normalize completed or budget-stopped ASR output according to `trim_on_stop`.
///
/// # Errors
/// Rejects cancellation, NUL bytes, oversized output, or non-EOS stop when `trim_on_stop` is `false`.
pub fn completed_transcript_with_options(
    raw_text: &str,
    stop: DecodeStop,
    trim_on_stop: bool,
) -> io::Result<String> {
    match stop {
        DecodeStop::Cancelled => return Err(io::Error::from(io::ErrorKind::Interrupted)),
        DecodeStop::TokenLimit | DecodeStop::Repetition => {
            if !trim_on_stop {
                return Err(invalid("Qwen3-ASR did not produce a complete transcript."));
            }
            let stripped = raw_text
                .split_once("<asr_text>")
                .map_or(raw_text, |(_, text)| text);
            let trimmed = trim_repeating_text_suffix(stripped).trim_end();
            validate_transcript(trimmed)?;
            return Ok(trimmed.to_owned());
        }
        DecodeStop::EndOfSequence => {}
    }
    if let Some((_, first)) = raw_text.split_once("<asr_text>") {
        if first.contains("<asr_text>") {
            return normalize_multi_segment_transcript(first.split("<asr_text>"));
        }
        validate_transcript(first)?;
        Ok(first.to_owned())
    } else {
        validate_transcript(raw_text)?;
        Ok(raw_text.to_owned())
    }
}

pub(crate) fn validate_transcript(text: &str) -> io::Result<()> {
    if text.len() > MAX_TEXT_BYTES || text.contains('\0') {
        return Err(invalid(
            "Qwen3-ASR returned an invalid or oversized transcript.",
        ));
    }
    Ok(())
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::{
        DecodeStop, EncoderLayout, PromptTokens, TranscribeOptions, completed_transcript,
        completed_transcript_with_options, normalize_multi_segment_transcript,
        trim_repeating_text_suffix,
    };
    use std::io;

    #[test]
    fn rust_asr_protocol_prompt_matches_symbolic_chat_template() -> io::Result<()> {
        let tokens = PromptTokens::new([0, 1, 2, 3, 4, 5], [6, 7, 8], 9)?;
        let prompt = tokens.automatic_prompt(3, 8192)?;
        // Literal expansion of the documented template, independently of the
        // implementation's slices, resize and range arithmetic.
        assert_eq!(
            prompt.ids(),
            [0, 3, 2, 1, 2, 0, 4, 2, 6, 7, 7, 7, 8, 1, 2, 0, 5, 2]
        );
        assert_eq!(prompt.audio_rows(), 9..12);
        assert!(PromptTokens::new([0, 1, 2, 3, 4, 9], [6, 7, 8], 9).is_err());
        assert!(tokens.automatic_prompt(0, 8192).is_err());
        assert!(tokens.automatic_prompt(usize::MAX, 8192).is_err());
        assert!(tokens.automatic_prompt(7921, 8192).is_ok());
        assert!(tokens.automatic_prompt(7922, 8192).is_err());
        assert!(tokens.automatic_prompt(7922, usize::MAX).is_err());
        assert!(tokens.automatic_prompt(3, 273).is_err());
        assert!(tokens.automatic_prompt(3, 274).is_ok());
        Ok(())
    }

    #[test]
    fn rust_asr_protocol_chunks_retain_ragged_audio_rows() -> io::Result<()> {
        for (frames, window, chunks, last, per, last_rows, padded, rows) in [
            (1, 50, 1, 1, 13, 1, 13, 1),
            (100, 50, 1, 100, 13, 13, 13, 13),
            (101, 50, 2, 1, 13, 1, 26, 14),
            (200, 50, 2, 100, 13, 13, 26, 26),
            (3000, 50, 30, 100, 13, 13, 390, 390),
            (17, 4, 3, 1, 1, 1, 3, 3),
        ] {
            let layout = EncoderLayout::new(frames, window)?;
            assert_eq!(layout.chunks, chunks);
            assert_eq!(layout.last_chunk_mel, last);
            assert_eq!(layout.rows_per_chunk, per);
            assert_eq!(layout.last_chunk_rows, last_rows);
            assert_eq!(layout.padded_rows, padded);
            assert_eq!(layout.audio_rows, rows);
            let tokens = PromptTokens::new([0, 1, 2, 3, 4, 5], [6, 7, 8], 9)?;
            assert_eq!(
                tokens.automatic_prompt(rows, 8192)?.audio_rows().len(),
                rows
            );
        }
        for (frames, window) in [(0, 50), (3001, 50), (10, 0), (1, usize::MAX)] {
            assert!(EncoderLayout::new(frames, window).is_err());
        }
        Ok(())
    }

    #[test]
    fn rust_asr_protocol_only_complete_text_is_delivered() -> io::Result<()> {
        for (raw, expected) in [
            ("language English<asr_text>Hello.", "Hello."),
            ("language Chinese<asr_text>你好，世界。", "你好，世界。"),
            ("  unchanged\n", "  unchanged\n"),
            ("<asr_text>a<asr_text>b", "a<asr_text>b"),
            ("language English<asr_text>", ""),
        ] {
            assert_eq!(
                completed_transcript(raw, DecodeStop::EndOfSequence)?,
                expected
            );
        }
        for stop in [
            DecodeStop::TokenLimit,
            DecodeStop::Repetition,
            DecodeStop::Cancelled,
        ] {
            let result = completed_transcript("language English<asr_text>partial", stop);
            let expected = if stop == DecodeStop::Cancelled {
                io::ErrorKind::Interrupted
            } else {
                io::ErrorKind::InvalidData
            };
            assert_eq!(result.err().map(|error| error.kind()), Some(expected));
        }
        assert!(completed_transcript("bad\0text", DecodeStop::EndOfSequence).is_err());
        assert!(completed_transcript(&"x".repeat(32 * 1024), DecodeStop::EndOfSequence).is_ok());
        assert!(
            completed_transcript(&"x".repeat(32 * 1024 + 1), DecodeStop::EndOfSequence).is_err()
        );
        assert!(completed_transcript(&"界".repeat(11_000), DecodeStop::EndOfSequence).is_err());
        Ok(())
    }

    #[test]
    fn rust_asr_protocol_live_framing_system_hint_and_repetition_trimming() -> io::Result<()> {
        let tokens = PromptTokens::new([0, 1, 2, 3, 4, 5], [6, 7, 8], 9)?;
        let auto = tokens.automatic_prompt(3, 8192)?;
        let mut framed = tokens.prefix_ids().to_vec();
        framed.extend([7, 7, 7]);
        framed.extend(tokens.suffix_ids());
        assert_eq!(framed, auto.ids());

        let hinted = tokens.prompt_with_system(&[4, 5], 2, 8192)?;
        assert_eq!(hinted.audio_rows(), 11..13);
        assert_eq!(hinted.ids()[..5], [0, 3, 2, 4, 5]);
        assert!(tokens.prompt_with_system(&[1; 300], 2, 274).is_err());

        assert_eq!(
            trim_repeating_text_suffix("hello world yes yes yes yes"),
            "hello world yes"
        );
        assert_eq!(trim_repeating_text_suffix("谢谢谢谢谢谢"), "谢");
        assert_eq!(
            trim_repeating_text_suffix("normal sentence."),
            "normal sentence."
        );
        assert_eq!(
            normalize_multi_segment_transcript([
                "language English<asr_text>First chunk.",
                "language English<asr_text>   ",
                "Second chunk.",
            ])?,
            "First chunk. Second chunk."
        );
        assert_eq!(
            completed_transcript_with_options(
                "language English<asr_text>ok again again again again",
                DecodeStop::Repetition,
                true,
            )?,
            "ok again"
        );
        assert_eq!(
            completed_transcript_with_options(
                "language English<asr_text>one<asr_text>two",
                DecodeStop::EndOfSequence,
                false,
            )?,
            "one two"
        );
        assert!(
            completed_transcript_with_options("partial", DecodeStop::TokenLimit, false).is_err()
        );
        assert_eq!(
            TranscribeOptions::default(),
            TranscribeOptions {
                short_clip_language: None,
                trim_repetition_on_stop: false,
            }
        );
        Ok(())
    }
}
