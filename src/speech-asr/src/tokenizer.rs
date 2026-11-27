//! Bounded byte-level BPE for the pinned Qwen3-ASR tokenizer.
//!
//! This implements the Qwen2 pretokenizer, first-occurrence vocabulary/merge
//! policy and literal byte decoding from the source in `rust-tokenizer.json`.
//! It is independent of the currently selected native inference engine.
#![forbid(unsafe_code)]

use std::{
    cmp::Reverse,
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
const UNMAPPED_SYMBOL: u32 = u32::MAX;
const NO_MERGE_RANK: u32 = u32::MAX;

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

fn span_bytes(arena: &[u8], span: TokenSpan) -> &[u8] {
    let start = usize::try_from(span.start).unwrap_or(arena.len());
    let len = usize::from(span.len);
    arena.get(start..start.saturating_add(len)).unwrap_or(&[])
}

fn span_matches_concat(piece: &[u8], left: &[u8], right: &[u8]) -> bool {
    let total = match left.len().checked_add(right.len()) {
        Some(len) => len,
        None => return false,
    };
    piece.len() == total && &piece[..left.len()] == left && &piece[left.len()..] == right
}

fn hash_pair(left: u32, right: u32) -> usize {
    let mut x = (u64::from(left) << 32) | u64::from(right);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    let folded = (x ^ (x >> 32)) & 0xffff_ffff;
    let [b0, b1, b2, b3, ..] = folded.to_le_bytes();
    usize::from(b0) | (usize::from(b1) << 8) | (usize::from(b2) << 16) | (usize::from(b3) << 24)
}

#[derive(Clone, Copy)]
struct PairSlot {
    left: u32,
    right: u32,
    rank: u32,
    merged: u32,
}

impl Default for PairSlot {
    fn default() -> Self {
        Self {
            left: UNMAPPED_SYMBOL,
            right: UNMAPPED_SYMBOL,
            rank: NO_MERGE_RANK,
            merged: UNMAPPED_SYMBOL,
        }
    }
}

struct PairTable {
    slots: Vec<PairSlot>,
    mask: usize,
}

impl PairTable {
    fn with_capacity(entries: usize) -> io::Result<Self> {
        if entries == 0 {
            return Ok(Self {
                slots: Vec::new(),
                mask: 0,
            });
        }
        let cap = entries
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(invalid)?
            .max(16);
        let mut slots = Vec::new();
        slots.try_reserve_exact(cap).map_err(io::Error::other)?;
        slots.resize(cap, PairSlot::default());
        Ok(Self {
            slots,
            mask: cap - 1,
        })
    }

    fn insert_if_absent_with(
        &mut self,
        left: u32,
        right: u32,
        rank: u32,
        merged: impl FnOnce() -> u32,
    ) {
        let mut idx = hash_pair(left, right) & self.mask;
        loop {
            let slot = &mut self.slots[idx];
            if slot.left == UNMAPPED_SYMBOL {
                *slot = PairSlot {
                    left,
                    right,
                    rank,
                    merged: merged(),
                };
                return;
            }
            if slot.left == left && slot.right == right {
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn get(&self, left: u32, right: u32) -> (u32, u32) {
        if self.slots.is_empty() {
            return (NO_MERGE_RANK, UNMAPPED_SYMBOL);
        }
        let mut idx = hash_pair(left, right) & self.mask;
        loop {
            let slot = self.slots[idx];
            if slot.left == UNMAPPED_SYMBOL {
                return (NO_MERGE_RANK, UNMAPPED_SYMBOL);
            }
            if slot.left == left && slot.right == right {
                return (slot.rank, slot.merged);
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

#[derive(Default)]
struct WordScratch {
    symbols: Vec<Symbol>,
    queue: Queue,
}

impl WordScratch {
    fn clear(&mut self) {
        self.symbols.clear();
        self.queue.clear();
    }
}

/// Immutable Qwen2/GPT-2 byte vocabulary and ranked merge table.
pub struct Tokenizer {
    token_arena: Vec<u8>,
    token_spans: Vec<TokenSpan>,
    vocab_sorted: Vec<u32>,
    vocab_cap: u32,
    pair_table: PairTable,
    byte_symbols: [u32; 256],
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
        let (token_arena, token_spans, vocab_sorted) = build_vocab_tables(tokens, &mut charged)?;
        let vocab_cap = u32::try_from(token_spans.len()).map_err(|_| invalid())?;
        let (pair_table, byte_symbols) = build_pair_index(
            &token_arena,
            &token_spans,
            &vocab_sorted,
            vocab_cap,
            &lines,
            &mut charged,
        )?;
        Ok(Self {
            token_arena,
            token_spans,
            vocab_sorted,
            vocab_cap,
            pair_table,
            byte_symbols,
            eos,
        })
    }

    fn find_bytes(&self, piece: &[u8]) -> Option<u32> {
        find_vocab_bytes(
            &self.token_arena,
            &self.token_spans,
            &self.vocab_sorted,
            piece,
        )
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
            .symbols
            .try_reserve(word.len())
            .map_err(io::Error::other)?;
        for (start, &byte) in word.iter().enumerate() {
            scratch.symbols.push(Symbol {
                raw_start: start,
                raw_end: start + 1,
                sym: self.byte_symbols[usize::from(byte)],
                previous: start.checked_sub(1),
                next: (start + 1 < word.len()).then_some(start + 1),
            });
        }
        for left in 0..word.len().saturating_sub(1) {
            self.enqueue(&scratch.symbols, left, &mut scratch.queue, probes)?;
        }
        while let Some(Reverse((_, left, right, middle, end, merged_sym))) = scratch.queue.pop() {
            if scratch.symbols[left].raw_end != middle
                || scratch.symbols[right].raw_start != middle
                || scratch.symbols[right].raw_end != end
                || scratch.symbols[left].next != Some(right)
            {
                continue;
            }
            scratch.symbols[left].raw_end = end;
            scratch.symbols[left].sym = merged_sym;
            scratch.symbols[left].next = scratch.symbols[right].next;
            scratch.symbols[right].raw_end = scratch.symbols[right].raw_start;
            if let Some(next) = scratch.symbols[left].next {
                scratch.symbols[next].previous = Some(left);
            }
            if let Some(previous) = scratch.symbols[left].previous {
                self.enqueue(&scratch.symbols, previous, &mut scratch.queue, probes)?;
            }
            self.enqueue(&scratch.symbols, left, &mut scratch.queue, probes)?;
        }
        for symbol in scratch
            .symbols
            .iter()
            .filter(|symbol| symbol.raw_end > symbol.raw_start)
        {
            if symbol.sym < self.vocab_cap {
                push_id(output, symbol.sym)?;
            } else {
                for &byte in &word[symbol.raw_start..symbol.raw_end] {
                    let byte_sym = self.byte_symbols[usize::from(byte)];
                    if byte_sym < self.vocab_cap {
                        push_id(output, byte_sym)?;
                    } else {
                        return Err(invalid());
                    }
                }
            }
        }
        Ok(())
    }

    fn enqueue(
        &self,
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
        if a.sym != UNMAPPED_SYMBOL && b.sym != UNMAPPED_SYMBOL {
            let (rank, merged) = self.pair_table.get(a.sym, b.sym);
            if rank != NO_MERGE_RANK {
                queue.push(Reverse((rank, left, right, a.raw_end, b.raw_end, merged)));
            }
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
    sym: u32,
    previous: Option<usize>,
    next: Option<usize>,
}
type Queue = BinaryHeap<Reverse<(u32, usize, usize, usize, usize, u32)>>;

fn find_vocab_bytes(
    token_arena: &[u8],
    token_spans: &[TokenSpan],
    vocab_sorted: &[u32],
    piece: &[u8],
) -> Option<u32> {
    vocab_sorted
        .binary_search_by(|&id| {
            let span = token_spans[usize::try_from(id).unwrap_or(0)];
            span_bytes(token_arena, span).cmp(piece)
        })
        .ok()
        .map(|idx| vocab_sorted[idx])
}

fn build_vocab_tables(
    tokens: Vec<String>,
    charged: &mut usize,
) -> io::Result<(Vec<u8>, Vec<TokenSpan>, Vec<u32>)> {
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
        charge(charged, piece)?;
        let start = u32::try_from(token_arena.len()).map_err(|_| invalid())?;
        let len = u16::try_from(piece.len()).map_err(|_| invalid())?;
        token_arena.extend_from_slice(piece.as_bytes());
        token_spans.push(TokenSpan { start, len });
        vocab_sorted.push(u32::try_from(id).map_err(|_| invalid())?);
    }
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
    Ok((token_arena, token_spans, vocab_sorted))
}

const FNV_OFFSET_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0100_0000_01b3;

#[inline]
fn fnv1a_step(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }
    hash
}

#[inline]
fn hash_slot(hash: u64, mask: usize) -> usize {
    let mixed = hash ^ (hash >> 33);
    let bytes = mixed.to_le_bytes();
    let low = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    usize::try_from(low).unwrap_or(0) & mask
}

struct VocabHashIndex {
    slots: Vec<u32>,
    mask: usize,
}

impl VocabHashIndex {
    fn build(
        token_arena: &[u8],
        token_spans: &[TokenSpan],
        vocab_sorted: &[u32],
    ) -> io::Result<Self> {
        let cap = (vocab_sorted.len().saturating_mul(2))
            .next_power_of_two()
            .max(16);
        let mut slots = Vec::new();
        slots.try_reserve_exact(cap).map_err(io::Error::other)?;
        slots.resize(cap, UNMAPPED_SYMBOL);
        let mask = cap - 1;
        for &vocab_id in vocab_sorted {
            let span = token_spans[usize::try_from(vocab_id).unwrap_or(0)];
            let hash = fnv1a_step(FNV_OFFSET_64, span_bytes(token_arena, span));
            let mut pos = hash_slot(hash, mask);
            while slots[pos] != UNMAPPED_SYMBOL {
                pos = (pos + 1) & mask;
            }
            slots[pos] = vocab_id;
        }
        Ok(Self { slots, mask })
    }

    fn get(&self, token_arena: &[u8], token_spans: &[TokenSpan], piece: &[u8]) -> Option<u32> {
        let hash = fnv1a_step(FNV_OFFSET_64, piece);
        let mut pos = hash_slot(hash, self.mask);
        loop {
            let id = self.slots[pos];
            if id == UNMAPPED_SYMBOL {
                return None;
            }
            let span = token_spans[usize::try_from(id).unwrap_or(0)];
            if span_bytes(token_arena, span) == piece {
                return Some(id);
            }
            pos = (pos + 1) & self.mask;
        }
    }

    fn get_concat(
        &self,
        token_arena: &[u8],
        token_spans: &[TokenSpan],
        left: &[u8],
        right: &[u8],
    ) -> Option<u32> {
        let hash = fnv1a_step(fnv1a_step(FNV_OFFSET_64, left), right);
        let mut pos = hash_slot(hash, self.mask);
        loop {
            let id = self.slots[pos];
            if id == UNMAPPED_SYMBOL {
                return None;
            }
            let span = token_spans[usize::try_from(id).unwrap_or(0)];
            if span_matches_concat(span_bytes(token_arena, span), left, right) {
                return Some(id);
            }
            pos = (pos + 1) & self.mask;
        }
    }
}

struct ExtraTable {
    arena: Vec<u8>,
    spans: Vec<TokenSpan>,
    slots: Vec<u32>,
    mask: usize,
}

impl ExtraTable {
    fn with_capacity(initial_slots: usize) -> io::Result<Self> {
        let cap = initial_slots.next_power_of_two().max(16);
        let mut slots = Vec::new();
        slots.try_reserve_exact(cap).map_err(io::Error::other)?;
        slots.resize(cap, UNMAPPED_SYMBOL);
        Ok(Self {
            arena: Vec::new(),
            spans: Vec::new(),
            slots,
            mask: cap - 1,
        })
    }

    fn rehash(&mut self) -> io::Result<()> {
        let new_cap = self.slots.len().checked_mul(2).ok_or_else(invalid)?;
        let mut new_slots = Vec::new();
        new_slots
            .try_reserve_exact(new_cap)
            .map_err(io::Error::other)?;
        new_slots.resize(new_cap, UNMAPPED_SYMBOL);
        let new_mask = new_cap - 1;
        for (idx, &span) in self.spans.iter().enumerate() {
            let hash = fnv1a_step(FNV_OFFSET_64, span_bytes(&self.arena, span));
            let mut pos = hash_slot(hash, new_mask);
            while new_slots[pos] != UNMAPPED_SYMBOL {
                pos = (pos + 1) & new_mask;
            }
            new_slots[pos] = u32::try_from(idx).map_err(|_| invalid())?;
        }
        self.slots = new_slots;
        self.mask = new_mask;
        Ok(())
    }

    fn insert_if_absent(&mut self, piece: &[u8]) -> io::Result<()> {
        let hash = fnv1a_step(FNV_OFFSET_64, piece);
        let mut pos = hash_slot(hash, self.mask);
        loop {
            let existing = self.slots[pos];
            if existing == UNMAPPED_SYMBOL {
                break;
            }
            let span = self.spans[usize::try_from(existing).unwrap_or(0)];
            if span_bytes(&self.arena, span) == piece {
                return Ok(());
            }
            pos = (pos + 1) & self.mask;
        }
        if (self.spans.len() + 1).saturating_mul(2) > self.slots.len() {
            self.rehash()?;
            pos = hash_slot(hash, self.mask);
            while self.slots[pos] != UNMAPPED_SYMBOL {
                pos = (pos + 1) & self.mask;
            }
        }
        let start = u32::try_from(self.arena.len()).map_err(|_| invalid())?;
        let len = u16::try_from(piece.len()).map_err(|_| invalid())?;
        self.arena
            .try_reserve(piece.len())
            .map_err(io::Error::other)?;
        self.arena.extend_from_slice(piece);
        let idx = u32::try_from(self.spans.len()).map_err(|_| invalid())?;
        self.spans.try_reserve(1).map_err(io::Error::other)?;
        self.spans.push(TokenSpan { start, len });
        self.slots[pos] = idx;
        Ok(())
    }

    fn get(&self, piece: &[u8]) -> Option<u32> {
        if self.spans.is_empty() {
            return None;
        }
        let hash = fnv1a_step(FNV_OFFSET_64, piece);
        let mut pos = hash_slot(hash, self.mask);
        loop {
            let idx = self.slots[pos];
            if idx == UNMAPPED_SYMBOL {
                return None;
            }
            let span = self.spans[usize::try_from(idx).unwrap_or(0)];
            if span_bytes(&self.arena, span) == piece {
                return Some(idx);
            }
            pos = (pos + 1) & self.mask;
        }
    }

    fn get_concat(&self, left: &[u8], right: &[u8]) -> Option<u32> {
        if self.spans.is_empty() {
            return None;
        }
        let hash = fnv1a_step(fnv1a_step(FNV_OFFSET_64, left), right);
        let mut pos = hash_slot(hash, self.mask);
        loop {
            let idx = self.slots[pos];
            if idx == UNMAPPED_SYMBOL {
                return None;
            }
            let span = self.spans[usize::try_from(idx).unwrap_or(0)];
            if span_matches_concat(span_bytes(&self.arena, span), left, right) {
                return Some(idx);
            }
            pos = (pos + 1) & self.mask;
        }
    }
}

fn resolve_symbol_indexed(
    vocab_index: &VocabHashIndex,
    token_arena: &[u8],
    token_spans: &[TokenSpan],
    extras: &ExtraTable,
    vocab_cap: u32,
    key: &[u8],
) -> u32 {
    vocab_index
        .get(token_arena, token_spans, key)
        .unwrap_or_else(|| {
            extras
                .get(key)
                .and_then(|idx| vocab_cap.checked_add(idx))
                .unwrap_or(UNMAPPED_SYMBOL)
        })
}

fn resolve_concat_indexed(
    vocab_index: &VocabHashIndex,
    token_arena: &[u8],
    token_spans: &[TokenSpan],
    extras: &ExtraTable,
    vocab_cap: u32,
    left: &[u8],
    right: &[u8],
) -> u32 {
    vocab_index
        .get_concat(token_arena, token_spans, left, right)
        .unwrap_or_else(|| {
            extras
                .get_concat(left, right)
                .and_then(|idx| vocab_cap.checked_add(idx))
                .unwrap_or(UNMAPPED_SYMBOL)
        })
}

fn build_pair_index(
    token_arena: &[u8],
    token_spans: &[TokenSpan],
    vocab_sorted: &[u32],
    vocab_cap: u32,
    lines: &[String],
    charged: &mut usize,
) -> io::Result<(PairTable, [u32; 256])> {
    let vocab_index = VocabHashIndex::build(token_arena, token_spans, vocab_sorted)?;
    let mut extras = ExtraTable::with_capacity((lines.len() / 8).max(16))?;
    let mut splits = Vec::new();
    splits
        .try_reserve_exact(lines.len())
        .map_err(io::Error::other)?;
    for (rank, line) in lines.iter().enumerate() {
        charge(charged, line)?;
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
        let _ = u16::try_from(left.len()).map_err(|_| invalid())?;
        let _ = u16::try_from(left.len() + right.len()).map_err(|_| invalid())?;
        let _ = u32::try_from(rank).map_err(|_| invalid())?;
        let split_u16 = u16::try_from(split).map_err(|_| invalid())?;
        splits.push(split_u16);
        if vocab_index.get(token_arena, token_spans, left).is_none() {
            extras.insert_if_absent(left)?;
        }
        if vocab_index.get(token_arena, token_spans, right).is_none() {
            extras.insert_if_absent(right)?;
        }
    }
    let total_symbols = token_spans
        .len()
        .checked_add(extras.spans.len())
        .and_then(|count| u32::try_from(count).ok())
        .filter(|&count| count < UNMAPPED_SYMBOL);
    if total_symbols.is_none() {
        return Err(invalid());
    }
    let mut pair_table = PairTable::with_capacity(lines.len())?;
    for (rank, (line, &split)) in lines.iter().zip(&splits).enumerate() {
        let split_idx = usize::from(split);
        let bytes = line.as_bytes();
        let left = &bytes[..split_idx];
        let right = &bytes[split_idx + 1..];
        let left_sym = resolve_symbol_indexed(
            &vocab_index,
            token_arena,
            token_spans,
            &extras,
            vocab_cap,
            left,
        );
        let right_sym = resolve_symbol_indexed(
            &vocab_index,
            token_arena,
            token_spans,
            &extras,
            vocab_cap,
            right,
        );
        let rank_u32 = u32::try_from(rank).map_err(|_| invalid())?;
        pair_table.insert_if_absent_with(left_sym, right_sym, rank_u32, || {
            resolve_concat_indexed(
                &vocab_index,
                token_arena,
                token_spans,
                &extras,
                vocab_cap,
                left,
                right,
            )
        });
    }
    let mut byte_symbols = [UNMAPPED_SYMBOL; 256];
    let mut glyph_buf = [0_u8; 4];
    for (byte, slot) in byte_symbols.iter_mut().enumerate() {
        let byte_val = u8::try_from(byte).unwrap_or(0);
        let encoded = byte_glyph(byte_val).encode_utf8(&mut glyph_buf);
        *slot = resolve_symbol_indexed(
            &vocab_index,
            token_arena,
            token_spans,
            &extras,
            vocab_cap,
            encoded.as_bytes(),
        );
    }
    Ok((pair_table, byte_symbols))
}

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
