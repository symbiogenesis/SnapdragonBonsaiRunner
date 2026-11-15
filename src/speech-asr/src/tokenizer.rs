//! Bounded byte-level BPE for the pinned Qwen3-ASR tokenizer.
//!
//! This implements the Qwen2 pretokenizer, first-occurrence vocabulary/merge
//! policy and literal byte decoding from the source in `rust-tokenizer.json`.
//! It is independent of the currently selected native inference engine.
#![forbid(unsafe_code)]

use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    io::{self, Read, Seek},
};

#[path = "tokenizer/scanner.rs"]
mod scanner;
#[path = "tokenizer/unicode.rs"]
mod unicode;

const MAX_INPUT: usize = 32 * 1024;
const MAX_IDS: usize = 8192;
const MAX_ENTRIES: usize = 262_144;
const MAX_SPELLING: usize = 4096;
const MAX_TABLE: usize = 64 * 1024 * 1024;
const MAX_PROBES: usize = 64 * 1024 * 1024;

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid Qwen3-ASR tokenizer input or tables.",
    )
}

fn byte_glyph(byte: u8) -> char {
    if matches!(byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff) {
        return char::from(byte);
    }
    let offset = match byte {
        0..=32 => u32::from(byte),
        127..=160 => u32::from(byte) - 94,
        173 => 67,
        _ => 0,
    };
    char::from_u32(256 + offset).unwrap_or('\0')
}

fn raw_byte(glyph: char) -> Option<u8> {
    let cp = u32::from(glyph);
    match cp {
        0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff => u8::try_from(cp).ok(),
        256..=288 => u8::try_from(cp - 256).ok(),
        289..=322 => u8::try_from(cp - 256 + 94).ok(),
        323 => Some(173),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct TokenSpan {
    start: u32,
    len: u16,
}

#[derive(Clone, Copy)]
struct MergeEntry {
    start: u32,
    left_len: u16,
    total_len: u16,
    rank: usize,
}

fn span_bytes(arena: &[u8], span: TokenSpan) -> &[u8] {
    let start = usize::try_from(span.start).unwrap_or(arena.len());
    let len = usize::from(span.len);
    arena.get(start..start.saturating_add(len)).unwrap_or(&[])
}

fn merge_parts(arena: &[u8], entry: MergeEntry) -> (&[u8], &[u8]) {
    let start = usize::try_from(entry.start).unwrap_or(arena.len());
    let left_end = start.saturating_add(usize::from(entry.left_len));
    let total_end = start.saturating_add(usize::from(entry.total_len));
    let left = arena.get(start..left_end).unwrap_or(&[]);
    let right = arena.get(left_end..total_end).unwrap_or(&[]);
    (left, right)
}

fn cmp_merge_parts(
    arena: &[u8],
    entry: MergeEntry,
    query_left: &[u8],
    query_right: &[u8],
) -> Ordering {
    let (left, right) = merge_parts(arena, entry);
    left.cmp(query_left).then_with(|| right.cmp(query_right))
}

#[derive(Default)]
struct WordScratch {
    encoded: Vec<u8>,
    symbols: Vec<Symbol>,
    queue: Queue,
}

impl WordScratch {
    fn clear(&mut self) {
        self.encoded.clear();
        self.symbols.clear();
        self.queue.clear();
    }
}

/// Immutable Qwen2/GPT-2 byte vocabulary and ranked merge table.
pub struct Tokenizer {
    token_arena: Vec<u8>,
    token_spans: Vec<TokenSpan>,
    vocab_sorted: Vec<u32>,
    merge_arena: Vec<u8>,
    merges: Vec<MergeEntry>,
    eos: u32,
}

impl Tokenizer {
    /// Number of original GGUF vocabulary slots, including duplicate spellings.
    #[must_use]
    pub const fn vocabulary_size(&self) -> usize {
        self.token_spans.len()
    }
    /// Admit the complete verified GGUF index and extract its tokenizer.
    ///
    /// The caller retains artifact digest verification and the immutable file
    /// guard. Only absent, empty or `qwen2` pretokenizer metadata is supported.
    /// No model weights are read and no inference is performed.
    ///
    /// # Errors
    /// Rejects malformed GGUF, unsupported metadata, invalid tables or bounds.
    pub fn load(source: &mut (impl Read + Seek), length: u64) -> io::Result<Self> {
        crate::intake::tokenizer(source, length, &mut || false).map(|(tokenizer, _)| tokenizer)
    }

    /// Construct from GGUF-order spellings and `left<space>right` merge lines.
    ///
    /// Duplicate spellings and merge pairs retain their first occurrence, as in
    /// the pinned native loader. Special-looking text is encoded as ordinary
    /// text; prompt construction resolves special IDs separately.
    ///
    /// # Errors
    /// Rejects empty/oversized tables, invalid EOS, malformed merges, spelling
    /// limits and aggregate table bytes exceeding 64 MiB.
    pub fn from_tables(tokens: Vec<String>, lines: Vec<String>, eos: u32) -> io::Result<Self> {
        if tokens.is_empty()
            || tokens.len() > MAX_ENTRIES
            || lines.is_empty()
            || lines.len() > MAX_ENTRIES
            || usize::try_from(eos).map_err(|_| invalid())? >= tokens.len()
        {
            return Err(invalid());
        }
        let mut charged = 0_usize;
        let token_bytes_hint: usize = tokens.iter().map(String::len).sum();
        let mut token_arena = Vec::new();
        token_arena
            .try_reserve_exact(token_bytes_hint.min(MAX_TABLE))
            .map_err(io::Error::other)?;
        let mut token_spans = Vec::new();
        token_spans
            .try_reserve_exact(tokens.len())
            .map_err(io::Error::other)?;
        let mut vocab_sorted = Vec::new();
        vocab_sorted
            .try_reserve_exact(tokens.len())
            .map_err(io::Error::other)?;
        for (id, piece) in tokens.iter().enumerate() {
            charge(&mut charged, piece)?;
            let start = u32::try_from(token_arena.len()).map_err(|_| invalid())?;
            let len = u16::try_from(piece.len()).map_err(|_| invalid())?;
            token_arena.extend_from_slice(piece.as_bytes());
            token_spans.push(TokenSpan { start, len });
            vocab_sorted.push(u32::try_from(id).map_err(|_| invalid())?);
        }
        drop(tokens);
        vocab_sorted.sort_unstable_by(|&a, &b| {
            let a_bytes = span_bytes(&token_arena, token_spans[usize::try_from(a).unwrap_or(0)]);
            let b_bytes = span_bytes(&token_arena, token_spans[usize::try_from(b).unwrap_or(0)]);
            a_bytes.cmp(b_bytes).then_with(|| a.cmp(&b))
        });
        vocab_sorted.dedup_by(|later, earlier| {
            let earlier_bytes = span_bytes(
                &token_arena,
                token_spans[usize::try_from(*earlier).unwrap_or(0)],
            );
            let later_bytes = span_bytes(
                &token_arena,
                token_spans[usize::try_from(*later).unwrap_or(0)],
            );
            earlier_bytes == later_bytes
        });

        let mut merge_arena = Vec::new();
        let merge_bytes_hint: usize = lines.iter().map(String::len).sum();
        merge_arena
            .try_reserve_exact(merge_bytes_hint.min(MAX_TABLE))
            .map_err(io::Error::other)?;
        let mut merges = Vec::new();
        merges
            .try_reserve_exact(lines.len())
            .map_err(io::Error::other)?;
        for (rank, line) in lines.into_iter().enumerate() {
            charge(&mut charged, &line)?;
            let split = line
                .as_bytes()
                .get(1..)
                .and_then(|rest| rest.iter().position(|&byte| byte == b' '))
                .map(|offset| offset + 1)
                .ok_or_else(invalid)?;
            if line.contains('\u{1f}') {
                return Err(invalid());
            }
            let bytes = line.as_bytes();
            let left = &bytes[..split];
            let right = &bytes[split + 1..];
            let start = u32::try_from(merge_arena.len()).map_err(|_| invalid())?;
            let left_len = u16::try_from(left.len()).map_err(|_| invalid())?;
            let total_len = u16::try_from(left.len() + right.len()).map_err(|_| invalid())?;
            merge_arena.extend_from_slice(left);
            merge_arena.extend_from_slice(right);
            merges.push(MergeEntry {
                start,
                left_len,
                total_len,
                rank,
            });
        }
        merges.sort_unstable_by(|&a, &b| {
            let (a_left, a_right) = merge_parts(&merge_arena, a);
            let (b_left, b_right) = merge_parts(&merge_arena, b);
            a_left
                .cmp(b_left)
                .then_with(|| a_right.cmp(b_right))
                .then_with(|| a.rank.cmp(&b.rank))
        });
        merges.dedup_by(|later, earlier| {
            merge_parts(&merge_arena, *earlier) == merge_parts(&merge_arena, *later)
        });
        Ok(Self {
            token_arena,
            token_spans,
            vocab_sorted,
            merge_arena,
            merges,
            eos,
        })
    }

    fn find_bytes(&self, piece: &[u8]) -> Option<u32> {
        self.vocab_sorted
            .binary_search_by(|&id| {
                let span = self.token_spans[usize::try_from(id).unwrap_or(0)];
                span_bytes(&self.token_arena, span).cmp(piece)
            })
            .ok()
            .map(|idx| self.vocab_sorted[idx])
    }

    /// Resolve a literal encoded vocabulary spelling, retaining first occurrence.
    #[must_use]
    pub fn find(&self, piece: &str) -> Option<u32> {
        self.find_bytes(piece.as_bytes())
    }

    /// The model's validated end-of-sequence token.
    #[must_use]
    pub const fn eos(&self) -> u32 {
        self.eos
    }

    /// Encode at most 32 KiB of UTF-8 with ranked, leftmost-tie BPE.
    ///
    /// # Errors
    /// Rejects missing fallback bytes, more than 8,192 output tokens, exhausted
    /// merge-work allowance or input limits. No partial IDs are published.
    pub fn encode(&self, text: &str) -> io::Result<Vec<u32>> {
        if text.len() > MAX_INPUT {
            return Err(invalid());
        }
        let mut output = Vec::new();
        output
            .try_reserve(text.len().min(MAX_IDS))
            .map_err(io::Error::other)?;
        let mut scratch = WordScratch::default();
        let mut probes = MAX_PROBES;
        for word in scanner::pretokens(text) {
            self.encode_word(word.as_bytes(), &mut output, &mut probes, &mut scratch)?;
        }
        Ok(output)
    }

    fn encode_word(
        &self,
        word: &[u8],
        output: &mut Vec<u32>,
        probes: &mut usize,
        scratch: &mut WordScratch,
    ) -> io::Result<()> {
        scratch.clear();
        scratch
            .encoded
            .try_reserve(word.len().saturating_mul(2))
            .map_err(io::Error::other)?;
        scratch
            .symbols
            .try_reserve(word.len())
            .map_err(io::Error::other)?;
        let mut glyph_buf = [0_u8; 4];
        for (start, &byte) in word.iter().enumerate() {
            let enc_start = scratch.encoded.len();
            scratch
                .encoded
                .extend_from_slice(byte_glyph(byte).encode_utf8(&mut glyph_buf).as_bytes());
            scratch.symbols.push(Symbol {
                raw_start: start,
                raw_end: start + 1,
                enc_start,
                enc_end: scratch.encoded.len(),
                previous: start.checked_sub(1),
                next: (start + 1 < word.len()).then_some(start + 1),
            });
        }
        for left in 0..word.len().saturating_sub(1) {
            self.enqueue(
                &scratch.encoded,
                &scratch.symbols,
                left,
                &mut scratch.queue,
                probes,
            )?;
        }
        while let Some(Reverse((_, left, right, middle, end))) = scratch.queue.pop() {
            if scratch.symbols[left].raw_end != middle
                || scratch.symbols[right].raw_start != middle
                || scratch.symbols[right].raw_end != end
                || scratch.symbols[left].next != Some(right)
            {
                continue;
            }
            scratch.symbols[left].raw_end = end;
            scratch.symbols[left].enc_end = scratch.symbols[right].enc_end;
            scratch.symbols[left].next = scratch.symbols[right].next;
            scratch.symbols[right].raw_end = scratch.symbols[right].raw_start;
            if let Some(next) = scratch.symbols[left].next {
                scratch.symbols[next].previous = Some(left);
            }
            if let Some(previous) = scratch.symbols[left].previous {
                self.enqueue(
                    &scratch.encoded,
                    &scratch.symbols,
                    previous,
                    &mut scratch.queue,
                    probes,
                )?;
            }
            self.enqueue(
                &scratch.encoded,
                &scratch.symbols,
                left,
                &mut scratch.queue,
                probes,
            )?;
        }
        for symbol in scratch
            .symbols
            .iter()
            .filter(|symbol| symbol.raw_end > symbol.raw_start)
        {
            if let Some(id) = self.find_bytes(&scratch.encoded[symbol.enc_start..symbol.enc_end]) {
                push_id(output, id)?;
            } else {
                for &byte in &word[symbol.raw_start..symbol.raw_end] {
                    let glyph = byte_glyph(byte);
                    let id = self
                        .find_bytes(glyph.encode_utf8(&mut glyph_buf).as_bytes())
                        .ok_or_else(invalid)?;
                    push_id(output, id)?;
                }
            }
        }
        Ok(())
    }

    fn enqueue(
        &self,
        encoded: &[u8],
        symbols: &[Symbol],
        left: usize,
        queue: &mut Queue,
        probes: &mut usize,
    ) -> io::Result<()> {
        let Some(right) = symbols[left].next else {
            return Ok(());
        };
        let (a, b) = (&symbols[left], &symbols[right]);
        *probes = probes
            .checked_sub(b.raw_end - a.raw_start)
            .ok_or_else(invalid)?;
        let query_left = &encoded[a.enc_start..a.enc_end];
        let query_right = &encoded[b.enc_start..b.enc_end];
        if let Ok(idx) = self.merges.binary_search_by(|&entry| {
            cmp_merge_parts(&self.merge_arena, entry, query_left, query_right)
        }) {
            queue.push(Reverse((
                self.merges[idx].rank,
                left,
                right,
                a.raw_end,
                b.raw_end,
            )));
        }
        Ok(())
    }

    /// Decode literal bytes without stripping control/EOS spellings.
    ///
    /// Out-of-range IDs are skipped to preserve the pinned native decoder.
    /// Unknown glyphs pass through as their original UTF-8. Byte fragments may
    /// span token boundaries and need not independently form valid UTF-8.
    ///
    /// # Errors
    /// Rejects more than 8,192 IDs or decoded bytes exceeding 32 KiB.
    pub fn decode_bytes(&self, ids: &[u32]) -> io::Result<Vec<u8>> {
        if ids.len() > MAX_IDS {
            return Err(invalid());
        }
        let mut output = Vec::new();
        for id in ids {
            let Some(&span) = usize::try_from(*id)
                .ok()
                .and_then(|id| self.token_spans.get(id))
            else {
                continue;
            };
            let piece =
                std::str::from_utf8(span_bytes(&self.token_arena, span)).map_err(|_| invalid())?;
            // Decoding never expands a spelling. Reserve once per token,
            // bounded by the remaining output allowance; byte glyphs can
            // still fit when their UTF-8 spelling exceeds that allowance.
            output
                .try_reserve(piece.len().min(MAX_INPUT - output.len()))
                .map_err(io::Error::other)?;
            for glyph in piece.chars() {
                let byte = raw_byte(glyph);
                let addition = byte.map_or_else(|| glyph.len_utf8(), |_| 1);
                if output.len() + addition > MAX_INPUT {
                    return Err(invalid());
                }
                if let Some(byte) = byte {
                    output.push(byte);
                } else {
                    output.extend_from_slice(glyph.encode_utf8(&mut [0; 4]).as_bytes());
                }
            }
        }
        Ok(output)
    }

    /// Decode a complete UTF-8 sequence, rejecting incomplete byte fragments.
    ///
    /// # Errors
    /// Returns the limits from [`Self::decode_bytes`] or invalid UTF-8.
    pub fn decode(&self, ids: &[u32]) -> io::Result<String> {
        String::from_utf8(self.decode_bytes(ids)?).map_err(|_| invalid())
    }
}

struct Symbol {
    raw_start: usize,
    raw_end: usize,
    enc_start: usize,
    enc_end: usize,
    previous: Option<usize>,
    next: Option<usize>,
}
type Queue = BinaryHeap<Reverse<(usize, usize, usize, usize, usize)>>;

fn charge(total: &mut usize, piece: &str) -> io::Result<()> {
    if piece.len() > MAX_SPELLING {
        return Err(invalid());
    }
    *total = total
        .checked_add(piece.len())
        .filter(|bytes| *bytes <= MAX_TABLE)
        .ok_or_else(invalid)?;
    Ok(())
}

fn push_id(output: &mut Vec<u32>, id: u32) -> io::Result<()> {
    if output.len() >= MAX_IDS {
        return Err(invalid());
    }
    output.push(id);
    Ok(())
}

#[cfg(test)]
#[path = "tokenizer/tests.rs"]
mod tests;
