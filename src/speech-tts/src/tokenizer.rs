//! Owned byte-level tokenizer policy from pinned Qwen `src/bpe.h`.
//!
//! Ported from qwentts.cpp `5916bf7c79dbb4e66fc059cdc507a79a23a28bb4`,
//! under the MIT license in `LICENSE.qwen-core`. GGUF access, owned tables,
//! special matching, segmentation and merges are safe Rust.
#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep the tokenizer and model-free boundary observations private to this crate."
)]

use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    io,
    ops::Range,
};

mod loading;
#[path = "tokenizer/scanner.rs"]
mod scanner;

const MAX_INPUT_BYTES: usize = 64 * 1024;
const MAX_ENTRIES: usize = 1_000_000;
const MAX_TABLE_BYTES: usize = 64 * 1024 * 1024;
const MAX_SPELLING_BYTES: usize = 4096;
const MAX_SPECIALS: usize = 64;
const MAX_MERGE_PROBES: usize = 64 * 1024 * 1024;
const UNASSIGNED_REVERSE: u32 = u32::MAX;
const NO_MERGE_RANK: i32 = i32::MAX;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn copy_bytes(bytes: &[u8]) -> io::Result<Vec<u8>> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(bytes.len())
        .map_err(io::Error::other)?;
    owned.extend_from_slice(bytes);
    Ok(owned)
}

#[derive(Clone, Copy)]
struct TableEntry {
    start: u32,
    len: u16,
    value: i32,
}

#[derive(Default)]
struct FlatTable {
    arena: Vec<u8>,
    entries: Vec<TableEntry>,
    bytes: usize,
}

impl FlatTable {
    fn span_bytes(arena: &[u8], start: u32, len: u16) -> &[u8] {
        let begin = usize::try_from(start).unwrap_or(arena.len());
        let width = usize::from(len);
        arena.get(begin..begin.saturating_add(width)).unwrap_or(&[])
    }

    fn entry_bytes(&self, entry: TableEntry) -> &[u8] {
        Self::span_bytes(&self.arena, entry.start, entry.len)
    }

    fn push(&mut self, bytes: &[u8], value: i32) -> io::Result<()> {
        let start = u32::try_from(self.arena.len())
            .map_err(|_| invalid("Qwen tokenizer table exceeds its aggregate bound."))?;
        let len = u16::try_from(bytes.len())
            .map_err(|_| invalid("Qwen tokenizer table entry is outside its bounds."))?;
        self.arena
            .try_reserve(bytes.len())
            .map_err(io::Error::other)?;
        self.entries.try_reserve(1).map_err(io::Error::other)?;
        self.arena.extend_from_slice(bytes);
        self.entries.push(TableEntry { start, len, value });
        self.bytes += bytes.len();
        Ok(())
    }

    fn sort_and_dedup_last_wins(&mut self) -> io::Result<()> {
        if self.entries.is_empty() {
            return Ok(());
        }
        let mut order = Vec::new();
        order
            .try_reserve_exact(self.entries.len())
            .map_err(io::Error::other)?;
        order.extend(0..self.entries.len());
        let arena = &self.arena;
        let entries = &self.entries;
        order.sort_unstable_by(|&a, &b| {
            Self::span_bytes(arena, entries[a].start, entries[a].len)
                .cmp(Self::span_bytes(arena, entries[b].start, entries[b].len))
                .then_with(|| b.cmp(&a))
        });
        let mut sorted_entries = Vec::new();
        sorted_entries
            .try_reserve_exact(order.len())
            .map_err(io::Error::other)?;
        let mut prev_span: Option<(u32, u16)> = None;
        let mut unique_bytes = 0_usize;
        for idx in order {
            let entry = entries[idx];
            if let Some((prev_start, prev_len)) = prev_span
                && Self::span_bytes(arena, entry.start, entry.len)
                    == Self::span_bytes(arena, prev_start, prev_len)
            {
                continue;
            }
            prev_span = Some((entry.start, entry.len));
            unique_bytes += usize::from(entry.len);
            sorted_entries.push(entry);
        }
        if sorted_entries.len() < entries.len() {
            let mut compact_arena = Vec::new();
            compact_arena
                .try_reserve_exact(unique_bytes)
                .map_err(io::Error::other)?;
            for entry in &mut sorted_entries {
                let new_start = u32::try_from(compact_arena.len())
                    .map_err(|_| invalid("Qwen tokenizer table exceeds its aggregate bound."))?;
                compact_arena.extend_from_slice(Self::span_bytes(arena, entry.start, entry.len));
                entry.start = new_start;
            }
            self.arena = compact_arena;
        }
        self.entries = sorted_entries;
        self.bytes = unique_bytes;
        Ok(())
    }

    fn get_sorted(&self, key: &[u8]) -> Option<i32> {
        self.entries
            .binary_search_by(|entry| self.entry_bytes(*entry).cmp(key))
            .ok()
            .map(|idx| self.entries[idx].value)
    }

    fn get_pair_sorted(&self, left: &[u8], right: &[u8]) -> Option<i32> {
        self.entries
            .binary_search_by(|entry| cmp_pair_key(self.entry_bytes(*entry), left, right))
            .ok()
            .map(|idx| self.entries[idx].value)
    }

    #[cfg(test)]
    fn get_unsorted_rev(&self, key: &[u8]) -> Option<i32> {
        self.entries
            .iter()
            .rev()
            .find(|entry| self.entry_bytes(**entry) == key)
            .map(|entry| entry.value)
    }
}

fn cmp_pair_key(stored: &[u8], left: &[u8], right: &[u8]) -> Ordering {
    let prefix = left.len().min(stored.len());
    match stored[..prefix].cmp(&left[..prefix]) {
        Ordering::Equal if prefix < left.len() || stored.len() == left.len() => Ordering::Less,
        Ordering::Equal => match stored[left.len()].cmp(&b' ') {
            Ordering::Equal => stored[left.len() + 1..].cmp(right),
            other => other,
        },
        other => other,
    }
}

#[derive(Clone, Copy)]
struct Symbol {
    start: usize,
    end: usize,
    prev: usize,
    next: usize,
    active: bool,
}

impl Symbol {
    fn span(self) -> Range<usize> {
        self.start..self.end
    }
}

#[derive(Default)]
struct ChunkScratch {
    encoded: Vec<u8>,
    symbols: Vec<Symbol>,
    ranks: Vec<i32>,
    heap: BinaryHeap<Reverse<(i32, usize, usize, usize)>>,
}

impl ChunkScratch {
    fn clear(&mut self) {
        self.encoded.clear();
        self.symbols.clear();
        self.ranks.clear();
        self.heap.clear();
    }
}

/// The scanner's UTF-8-scalar-aligned pretokenization chunks.
#[cfg(test)]
pub(super) fn pretokenize(text: &[u8]) -> io::Result<Vec<&[u8]>> {
    if text.len() > MAX_INPUT_BYTES || std::str::from_utf8(text).is_err() {
        return Err(invalid(
            "Qwen tokenizer input is oversized or invalid UTF-8.",
        ));
    }
    scanner::pretokenize(text)
}

/// One context's immutable finalized vocabulary and merge policy.
pub(super) struct Tokenizer {
    vocabulary: FlatTable,
    merges: FlatTable,
    reverse: Vec<u32>,
    specials: Vec<(Vec<u8>, i32)>,
    eos: i32,
    table_bytes: usize,
    max_merge_bytes: usize,
    finished: bool,
}

impl Tokenizer {
    /// Load bounded pinned tokenizer tables from an already opened GGUF.
    pub(super) fn from_weights(weights: &crate::weights::Weights) -> io::Result<Self> {
        loading::from_weights(weights)
    }

    /// Load bounded pinned tokenizer tables without a native metadata API.
    #[cfg(test)]
    pub(super) fn load(path: &std::path::Path) -> io::Result<Self> {
        loading::load(path)
    }

    #[cfg(test)]
    pub(super) fn token_id_for_test(&self, spelling: &[u8]) -> Option<i32> {
        if self.finished {
            self.vocabulary.get_sorted(spelling)
        } else {
            self.vocabulary.get_unsorted_rev(spelling)
        }
    }

    #[cfg(test)]
    pub(super) fn merge_rank_for_test(&self, spelling: &[u8]) -> Option<i32> {
        if self.finished {
            self.merges.get_sorted(spelling)
        } else {
            self.merges.get_unsorted_rev(spelling)
        }
    }

    /// Create an empty owner without allocating model tables.
    pub(super) fn new() -> Self {
        #[cfg(test)]
        loading::owner_created();
        Self {
            vocabulary: FlatTable::default(),
            merges: FlatTable::default(),
            reverse: Vec::new(),
            specials: Vec::new(),
            eos: -1,
            table_bytes: 0,
            max_merge_bytes: 0,
            finished: false,
        }
    }

    fn compact_tables(&mut self) -> io::Result<()> {
        self.vocabulary.sort_and_dedup_last_wins()?;
        self.merges.sort_and_dedup_last_wins()?;
        self.table_bytes = self.vocabulary.bytes + self.merges.bytes;
        Ok(())
    }

    fn insert(&mut self, bytes: &[u8], value: i32, merge: bool) -> io::Result<()> {
        if self.finished
            || bytes.len() > MAX_SPELLING_BYTES
            || value < 0
            || !usize::try_from(value).is_ok_and(|value| value < MAX_ENTRIES)
        {
            return Err(invalid("Qwen tokenizer table entry is outside its bounds."));
        }
        let table_len = if merge {
            self.merges.entries.len()
        } else {
            self.vocabulary.entries.len()
        };
        if table_len >= MAX_ENTRIES || bytes.len() > MAX_TABLE_BYTES - self.table_bytes {
            self.compact_tables()?;
            let table_len = if merge {
                self.merges.entries.len()
            } else {
                self.vocabulary.entries.len()
            };
            if table_len >= MAX_ENTRIES || bytes.len() > MAX_TABLE_BYTES - self.table_bytes {
                return Err(invalid("Qwen tokenizer table exceeds its aggregate bound."));
            }
        }
        if merge {
            self.merges.push(bytes, value)?;
            self.max_merge_bytes = self.max_merge_bytes.max(bytes.len());
        } else {
            self.vocabulary.push(bytes, value)?;
        }
        self.table_bytes += bytes.len();
        Ok(())
    }

    /// Add a raw vocabulary spelling; repeated spellings replace their ID.
    pub(super) fn add_token(&mut self, bytes: &[u8], id: i32) -> io::Result<()> {
        self.insert(bytes, id, false)
    }

    /// Add the complete `left + space + right` key and its native rank.
    pub(super) fn add_merge(&mut self, bytes: &[u8], rank: i32) -> io::Result<()> {
        self.insert(bytes, rank, true)
    }

    /// Build native-compatible reverse slots and register the end-of-text token.
    pub(super) fn finish(&mut self) -> io::Result<()> {
        if self.finished {
            return Err(invalid("Qwen tokenizer tables were already finalized."));
        }
        self.compact_tables()?;
        let mut reverse = Vec::new();
        reverse
            .try_reserve_exact(self.vocabulary.entries.len())
            .map_err(io::Error::other)?;
        reverse.resize(self.vocabulary.entries.len(), UNASSIGNED_REVERSE);
        for (vocab_idx, entry) in self.vocabulary.entries.iter().enumerate() {
            let id =
                usize::try_from(entry.value).map_err(|_| invalid("Invalid Qwen vocabulary ID."))?;
            if let Some(slot) = reverse.get_mut(id) {
                if *slot != UNASSIGNED_REVERSE {
                    return Err(invalid("Qwen vocabulary contains ambiguous reverse IDs."));
                }
                *slot = u32::try_from(vocab_idx)
                    .map_err(|_| invalid("Invalid Qwen vocabulary index."))?;
            }
        }
        self.reverse = reverse;
        self.eos = self.vocabulary.get_sorted(b"<|endoftext|>").unwrap_or(-1);
        if self.eos >= 0 {
            self.add_special(b"<|endoftext|>", self.eos)?;
        }
        self.finished = true;
        Ok(())
    }

    /// Register a nonempty spelling; the first registration of each spelling wins.
    pub(super) fn add_special(&mut self, bytes: &[u8], id: i32) -> io::Result<()> {
        Self::insert_special(&mut self.specials, bytes, id)
    }

    fn insert_special(specials: &mut Vec<(Vec<u8>, i32)>, bytes: &[u8], id: i32) -> io::Result<()> {
        if specials.iter().any(|(spelling, _)| spelling == bytes) {
            return Ok(());
        }
        if bytes.is_empty() || bytes.len() > MAX_SPELLING_BYTES || specials.len() >= MAX_SPECIALS {
            return Err(invalid("Qwen tokenizer special exceeds its bounds."));
        }
        let owned = copy_bytes(bytes)?;
        specials.try_reserve(1).map_err(io::Error::other)?;
        specials.push((owned, id));
        Ok(())
    }

    /// Register a speech special by reverse ID, skipping invalid or empty slots.
    pub(super) fn add_special_id(&mut self, id: i32) -> io::Result<()> {
        if !self.finished {
            return Err(invalid("Qwen tokenizer tables are not finalized."));
        }
        let Some(&vocab_idx) = usize::try_from(id).ok().and_then(|id| self.reverse.get(id)) else {
            return Ok(());
        };
        if vocab_idx == UNASSIGNED_REVERSE {
            return Ok(());
        }
        let Some(entry) = usize::try_from(vocab_idx)
            .ok()
            .and_then(|idx| self.vocabulary.entries.get(idx).copied())
        else {
            return Ok(());
        };
        let spelling = self.vocabulary.entry_bytes(entry);
        if spelling.is_empty() {
            return Ok(());
        }
        // Specials copy into independent owned storage from the immutable vocabulary arena.
        Self::insert_special(&mut self.specials, spelling, id)
    }

    fn encode_segment(
        &self,
        bytes: &[u8],
        ids: &mut Vec<i32>,
        probes: &mut usize,
        scratch: &mut ChunkScratch,
    ) -> io::Result<()> {
        if bytes.len() > MAX_INPUT_BYTES || std::str::from_utf8(bytes).is_err() {
            return Err(invalid(
                "Qwen tokenizer input is oversized or invalid UTF-8.",
            ));
        }
        for chunk in scanner::pretokenize_iter(bytes) {
            self.encode_chunk(chunk?, ids, probes, scratch)?;
        }
        Ok(())
    }

    /// Encode valid UTF-8 using the pinned byte scanner, special order and merges.
    ///
    /// Tables, input, output and merge work have explicit limits. Invalid or
    /// truncated UTF-8 is rejected rather than reproducing unchecked native reads.
    pub(super) fn encode(&self, text: &[u8], add_eos: bool) -> io::Result<Vec<i32>> {
        if !self.finished || text.len() > MAX_INPUT_BYTES || std::str::from_utf8(text).is_err() {
            return Err(invalid(
                "Qwen tokenizer input or table state is unsupported.",
            ));
        }
        let mut ids = Vec::new();
        // A byte expands to at most two encoded bytes; even the native raw-byte
        // fallback cannot emit more than this many IDs, plus the optional EOS.
        ids.try_reserve_exact(text.len() * 2 + 1)
            .map_err(io::Error::other)?;
        let mut scratch = ChunkScratch::default();
        let mut position = 0;
        let mut probes = 0;
        while position < text.len() {
            let mut best: Option<(usize, usize)> = None;
            for (index, (spelling, _)) in self.specials.iter().enumerate() {
                let search_end = if let Some((previous, _)) = best {
                    if previous == position {
                        break;
                    }
                    previous
                        .saturating_sub(1)
                        .saturating_add(spelling.len())
                        .min(text.len())
                } else {
                    text.len()
                };
                if let Some(offset) = text[position..search_end]
                    .windows(spelling.len())
                    .position(|window| window == spelling)
                {
                    let found = position + offset;
                    if best.is_none_or(|(previous, _)| found < previous) {
                        best = Some((found, index));
                    }
                }
            }
            let Some((found, index)) = best else {
                self.encode_segment(&text[position..], &mut ids, &mut probes, &mut scratch)?;
                break;
            };
            self.encode_segment(&text[position..found], &mut ids, &mut probes, &mut scratch)?;
            let (spelling, id) = &self.specials[index];
            ids.push(*id);
            position = found + spelling.len();
        }
        if add_eos && self.eos >= 0 {
            ids.push(self.eos);
        }
        Ok(ids)
    }

    fn lookup_merge_rank(&self, left: &[u8], right: &[u8], probes: &mut usize) -> io::Result<i32> {
        *probes += 1;
        if *probes > MAX_MERGE_PROBES {
            return Err(invalid("Qwen tokenizer merge work exceeds its bound."));
        }
        if left.len() + right.len() + 1 > self.max_merge_bytes {
            return Ok(NO_MERGE_RANK);
        }
        Ok(self
            .merges
            .get_pair_sorted(left, right)
            .unwrap_or(NO_MERGE_RANK))
    }

    fn encode_chunk(
        &self,
        chunk: &[u8],
        ids: &mut Vec<i32>,
        probes: &mut usize,
        scratch: &mut ChunkScratch,
    ) -> io::Result<()> {
        if chunk.is_empty() {
            return Ok(());
        }
        if let [byte] = chunk {
            let spelling = BYTE_ENCODER[usize::from(*byte)];
            let width = if spelling[1] == 0 { 1 } else { 2 };
            self.emit_piece(&spelling[..width], ids);
            return Ok(());
        }
        scratch.clear();
        let count = chunk.len();
        scratch
            .encoded
            .try_reserve(count * 2)
            .map_err(io::Error::other)?;
        scratch
            .symbols
            .try_reserve(count)
            .map_err(io::Error::other)?;
        scratch.ranks.try_reserve(count).map_err(io::Error::other)?;
        for (index, &byte) in chunk.iter().enumerate() {
            let start = scratch.encoded.len();
            let bytes = BYTE_ENCODER[usize::from(byte)];
            scratch.encoded.push(bytes[0]);
            if bytes[1] != 0 {
                scratch.encoded.push(bytes[1]);
            }
            scratch.symbols.push(Symbol {
                start,
                end: scratch.encoded.len(),
                prev: index.wrapping_sub(1),
                next: if index + 1 < count {
                    index + 1
                } else {
                    usize::MAX
                },
                active: true,
            });
            scratch.ranks.push(NO_MERGE_RANK);
        }
        for index in 0..count - 1 {
            let right_idx = index + 1;
            let rank = self.lookup_merge_rank(
                &scratch.encoded[scratch.symbols[index].span()],
                &scratch.encoded[scratch.symbols[right_idx].span()],
                probes,
            )?;
            scratch.ranks[index] = rank;
            if rank != NO_MERGE_RANK {
                scratch.heap.push(Reverse((
                    rank,
                    index,
                    scratch.symbols[index].end,
                    scratch.symbols[right_idx].end,
                )));
            }
        }
        while let Some(Reverse((rank, left, exp_left_end, exp_right_end))) = scratch.heap.pop() {
            if !scratch.symbols[left].active
                || scratch.symbols[left].end != exp_left_end
                || scratch.ranks[left] != rank
            {
                continue;
            }
            let right = scratch.symbols[left].next;
            if right == usize::MAX || scratch.symbols[right].end != exp_right_end {
                continue;
            }
            let after = scratch.symbols[right].next;
            scratch.symbols[left].end = exp_right_end;
            scratch.symbols[left].next = after;
            if after != usize::MAX {
                scratch.symbols[after].prev = left;
            }
            scratch.symbols[right].active = false;
            scratch.symbols[right].prev = usize::MAX;
            scratch.symbols[right].next = usize::MAX;
            scratch.ranks[right] = NO_MERGE_RANK;

            let prev = scratch.symbols[left].prev;
            if prev != usize::MAX {
                let prev_rank = self.lookup_merge_rank(
                    &scratch.encoded[scratch.symbols[prev].span()],
                    &scratch.encoded[scratch.symbols[left].span()],
                    probes,
                )?;
                scratch.ranks[prev] = prev_rank;
                if prev_rank != NO_MERGE_RANK {
                    scratch.heap.push(Reverse((
                        prev_rank,
                        prev,
                        scratch.symbols[prev].end,
                        scratch.symbols[left].end,
                    )));
                }
            }
            if after != usize::MAX {
                let next_rank = self.lookup_merge_rank(
                    &scratch.encoded[scratch.symbols[left].span()],
                    &scratch.encoded[scratch.symbols[after].span()],
                    probes,
                )?;
                scratch.ranks[left] = next_rank;
                if next_rank != NO_MERGE_RANK {
                    scratch.heap.push(Reverse((
                        next_rank,
                        left,
                        scratch.symbols[left].end,
                        scratch.symbols[after].end,
                    )));
                }
            } else {
                scratch.ranks[left] = NO_MERGE_RANK;
            }
        }
        let mut cursor = 0;
        while cursor != usize::MAX {
            let symbol = scratch.symbols[cursor];
            self.emit_piece(&scratch.encoded[symbol.span()], ids);
            cursor = symbol.next;
        }
        Ok(())
    }

    fn emit_piece(&self, piece: &[u8], ids: &mut Vec<i32>) {
        if let Some(id) = self.vocabulary.get_sorted(piece) {
            ids.push(id);
        } else {
            // Native fallback probes the raw bytes of the encoded spelling,
            // not the original input bytes or the GPT-2 byte encoder again.
            for byte in piece {
                if let Some(id) = self.vocabulary.get_sorted(std::slice::from_ref(byte)) {
                    ids.push(id);
                }
            }
        }
    }
}

const fn visible(byte: u16) -> bool {
    matches!(byte, 33..=126 | 161..=172 | 174..=255)
}

const fn byte_encoder() -> [[u8; 2]; 256] {
    let mut result = [[0; 2]; 256];
    let mut byte = 0_usize;
    let mut omitted = 0_u16;
    while byte < 256 {
        let byte_value = u16::from_le_bytes([byte.to_le_bytes()[0], 0]);
        let codepoint = if visible(byte_value) {
            byte_value
        } else {
            let codepoint = 256 + omitted;
            omitted += 1;
            codepoint
        };
        // Every GPT-2 byte spelling is an ASCII or two-byte UTF-8 codepoint.
        let [low, high] = codepoint.to_le_bytes();
        if let Some(character) = char::from_u32(u32::from_le_bytes([low, high, 0, 0])) {
            let mut utf8 = [0_u8; 4];
            let _ = character.encode_utf8(&mut utf8);
            let [first, second, ..] = utf8;
            result[byte] = [first, second];
        }
        byte += 1;
    }
    result
}

const BYTE_ENCODER: [[u8; 2]; 256] = byte_encoder();
