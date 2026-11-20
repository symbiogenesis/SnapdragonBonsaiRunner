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
const UNMAPPED_SYMBOL: u32 = u32::MAX;
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

    fn reserve(&mut self, entries: usize, bytes: usize) -> io::Result<()> {
        let entry_cap = entries.min(MAX_ENTRIES);
        let arena_cap = bytes.min(MAX_TABLE_BYTES);
        if entry_cap > self.entries.capacity() {
            self.entries
                .try_reserve(entry_cap - self.entries.len())
                .map_err(io::Error::other)?;
        }
        if arena_cap > self.arena.capacity() {
            self.arena
                .try_reserve(arena_cap - self.arena.len())
                .map_err(io::Error::other)?;
        }
        Ok(())
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

    #[cfg(test)]
    fn get_unsorted_rev(&self, key: &[u8]) -> Option<i32> {
        self.entries
            .iter()
            .rev()
            .find(|entry| self.entry_bytes(**entry) == key)
            .map(|entry| entry.value)
    }
}

fn cmp_concat(candidate: &[u8], left: &[u8], right: &[u8]) -> Ordering {
    let prefix = candidate.len().min(left.len());
    match candidate[..prefix].cmp(&left[..prefix]) {
        Ordering::Equal => {
            if candidate.len() < left.len() {
                Ordering::Less
            } else {
                candidate[left.len()..].cmp(right)
            }
        }
        other => other,
    }
}

fn split_merge_pair(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let split = bytes.iter().position(|&byte| byte == b' ')?;
    let left = &bytes[..split];
    let right = &bytes[split + 1..];
    (!left.is_empty() && !right.is_empty() && !right.contains(&b' ')).then_some((left, right))
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
    let low = u32::from_le_bytes([
        mixed.to_le_bytes()[0],
        mixed.to_le_bytes()[1],
        mixed.to_le_bytes()[2],
        mixed.to_le_bytes()[3],
    ]);
    usize::try_from(low).unwrap_or(0) & mask
}

struct VocabHashIndex {
    slots: Vec<u32>,
    mask: usize,
}

impl VocabHashIndex {
    fn build(vocabulary: &FlatTable) -> io::Result<Self> {
        let count = vocabulary.entries.len();
        let cap = (count.saturating_mul(2)).next_power_of_two().max(16);
        let mut slots = Vec::new();
        slots.try_reserve_exact(cap).map_err(io::Error::other)?;
        slots.resize(cap, UNMAPPED_SYMBOL);
        let mask = cap - 1;
        for (idx, entry) in vocabulary.entries.iter().enumerate() {
            let idx_u32 =
                u32::try_from(idx).map_err(|_| invalid("Invalid Qwen vocabulary index."))?;
            let hash = fnv1a_step(FNV_OFFSET_64, vocabulary.entry_bytes(*entry));
            let mut pos = hash_slot(hash, mask);
            while slots[pos] != UNMAPPED_SYMBOL {
                pos = (pos + 1) & mask;
            }
            slots[pos] = idx_u32;
        }
        Ok(Self { slots, mask })
    }

    fn get(&self, vocabulary: &FlatTable, key: &[u8]) -> Option<u32> {
        if vocabulary.entries.is_empty() {
            return None;
        }
        let hash = fnv1a_step(FNV_OFFSET_64, key);
        let mut pos = hash_slot(hash, self.mask);
        loop {
            let slot = self.slots[pos];
            if slot == UNMAPPED_SYMBOL {
                return None;
            }
            let entry = vocabulary.entries[usize::try_from(slot).unwrap_or(0)];
            if vocabulary.entry_bytes(entry) == key {
                return Some(slot);
            }
            pos = (pos + 1) & self.mask;
        }
    }

    fn get_concat(&self, vocabulary: &FlatTable, left: &[u8], right: &[u8]) -> Option<u32> {
        if vocabulary.entries.is_empty() {
            return None;
        }
        let target_len = left.len().checked_add(right.len())?;
        let hash = fnv1a_step(fnv1a_step(FNV_OFFSET_64, left), right);
        let mut pos = hash_slot(hash, self.mask);
        loop {
            let slot = self.slots[pos];
            if slot == UNMAPPED_SYMBOL {
                return None;
            }
            let entry = vocabulary.entries[usize::try_from(slot).unwrap_or(0)];
            let candidate = vocabulary.entry_bytes(entry);
            if candidate.len() == target_len
                && candidate[..left.len()] == *left
                && candidate[left.len()..] == *right
            {
                return Some(slot);
            }
            pos = (pos + 1) & self.mask;
        }
    }
}

#[derive(Clone, Copy)]
struct Symbol {
    start: usize,
    end: usize,
    prev: usize,
    next: usize,
    active: bool,
    sym: u32,
    next_merged: u32,
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

#[derive(Clone, Copy)]
struct PairSlot {
    left: u32,
    right: u32,
    rank: i32,
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

#[derive(Default)]
struct PairTable {
    slots: Vec<PairSlot>,
    mask: usize,
}

impl PairTable {
    fn with_capacity(entries: usize) -> io::Result<Self> {
        if entries == 0 {
            return Ok(Self::default());
        }
        let cap = entries
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| invalid("Qwen tokenizer pair table exceeds its bound."))?
            .max(16);
        let mut slots = Vec::new();
        slots.try_reserve_exact(cap).map_err(io::Error::other)?;
        slots.resize(cap, PairSlot::default());
        Ok(Self {
            slots,
            mask: cap - 1,
        })
    }

    fn insert(&mut self, left: u32, right: u32, rank: i32, merged: u32) {
        let mut idx = hash_pair(left, right) & self.mask;
        loop {
            let slot = &mut self.slots[idx];
            if slot.left == UNMAPPED_SYMBOL || (slot.left == left && slot.right == right) {
                *slot = PairSlot {
                    left,
                    right,
                    rank,
                    merged,
                };
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn get(&self, left: u32, right: u32) -> (i32, u32) {
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

fn mark_special_first_byte(bitmap: &mut [u64; 4], byte: u8) {
    bitmap[usize::from(byte >> 6)] |= 1_u64 << (byte & 63);
}

fn has_special_first_byte(bitmap: &[u64; 4], byte: u8) -> bool {
    (bitmap[usize::from(byte >> 6)] & (1_u64 << (byte & 63))) != 0
}

/// One context's immutable finalized vocabulary and merge policy.
pub(super) struct Tokenizer {
    vocabulary: FlatTable,
    merges: FlatTable,
    pair_table: PairTable,
    byte_symbols: [u32; 256],
    vocab_cap: u32,
    reverse: Vec<u32>,
    specials: Vec<(Vec<u8>, i32)>,
    special_first_bytes: [u64; 4],
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
            pair_table: PairTable::default(),
            byte_symbols: [UNMAPPED_SYMBOL; 256],
            vocab_cap: 0,
            reverse: Vec::new(),
            specials: Vec::new(),
            special_first_bytes: [0; 4],
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

    /// Pre-reserve vocabulary table storage for bulk metadata intake.
    pub(super) fn reserve_tokens(&mut self, count: usize, bytes: usize) -> io::Result<()> {
        self.vocabulary.reserve(count, bytes)
    }

    /// Pre-reserve merge table storage for bulk metadata intake.
    pub(super) fn reserve_merges(&mut self, count: usize, bytes: usize) -> io::Result<()> {
        self.merges.reserve(count, bytes)
    }

    /// Add a raw vocabulary spelling; repeated spellings replace their ID.
    pub(super) fn add_token(&mut self, bytes: &[u8], id: i32) -> io::Result<()> {
        self.insert(bytes, id, false)
    }

    /// Add the complete `left + space + right` key and its native rank.
    pub(super) fn add_merge(&mut self, bytes: &[u8], rank: i32) -> io::Result<()> {
        self.insert(bytes, rank, true)
    }

    /// Build native-compatible reverse slots, pair lookup index, and register end-of-text.
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
        let (pair_table, byte_symbols, vocab_cap) = self.build_pair_index()?;
        self.reverse = reverse;
        self.pair_table = pair_table;
        self.byte_symbols = byte_symbols;
        self.vocab_cap = vocab_cap;
        self.eos = self.vocabulary.get_sorted(b"<|endoftext|>").unwrap_or(-1);
        if self.eos >= 0 {
            self.add_special(b"<|endoftext|>", self.eos)?;
        }
        self.finished = true;
        Ok(())
    }

    fn resolve_symbol(
        &self,
        vocab_index: &VocabHashIndex,
        key: &[u8],
        extras: &[&[u8]],
        vocab_cap: u32,
    ) -> u32 {
        vocab_index.get(&self.vocabulary, key).unwrap_or_else(|| {
            if extras.is_empty() {
                return UNMAPPED_SYMBOL;
            }
            extras
                .binary_search(&key)
                .ok()
                .and_then(|idx| u32::try_from(idx).ok())
                .and_then(|offset| vocab_cap.checked_add(offset))
                .unwrap_or(UNMAPPED_SYMBOL)
        })
    }

    fn resolve_concat_symbol(
        &self,
        vocab_index: &VocabHashIndex,
        left: &[u8],
        right: &[u8],
        extras: &[&[u8]],
        vocab_cap: u32,
    ) -> u32 {
        if let Some(idx) = vocab_index.get_concat(&self.vocabulary, left, right) {
            return idx;
        }
        if !extras.is_empty()
            && let Ok(idx) = extras.binary_search_by(|candidate| cmp_concat(candidate, left, right))
        {
            return u32::try_from(idx)
                .ok()
                .and_then(|offset| vocab_cap.checked_add(offset))
                .unwrap_or(UNMAPPED_SYMBOL);
        }
        UNMAPPED_SYMBOL
    }

    fn build_pair_index(&self) -> io::Result<(PairTable, [u32; 256], u32)> {
        let vocab_cap = u32::try_from(self.vocabulary.entries.len())
            .map_err(|_| invalid("Invalid Qwen vocabulary index."))?;
        let vocab_index = VocabHashIndex::build(&self.vocabulary)?;
        let mut extras = Vec::new();
        let mut valid_merges = 0_usize;
        for entry in &self.merges.entries {
            if let Some((left, right)) = split_merge_pair(self.merges.entry_bytes(*entry)) {
                valid_merges += 1;
                if vocab_index.get(&self.vocabulary, left).is_none() {
                    extras.try_reserve(1).map_err(io::Error::other)?;
                    extras.push(left);
                }
                if vocab_index.get(&self.vocabulary, right).is_none() {
                    extras.try_reserve(1).map_err(io::Error::other)?;
                    extras.push(right);
                }
            }
        }
        extras.sort_unstable();
        extras.dedup();
        let total_symbols = self
            .vocabulary
            .entries
            .len()
            .checked_add(extras.len())
            .and_then(|count| u32::try_from(count).ok())
            .filter(|&count| count < UNMAPPED_SYMBOL);
        if total_symbols.is_none() {
            return Err(invalid("Qwen tokenizer symbol space exceeded."));
        }
        let mut pair_table = PairTable::with_capacity(valid_merges)?;
        for entry in &self.merges.entries {
            if let Some((left, right)) = split_merge_pair(self.merges.entry_bytes(*entry)) {
                let left_sym = self.resolve_symbol(&vocab_index, left, &extras, vocab_cap);
                let right_sym = self.resolve_symbol(&vocab_index, right, &extras, vocab_cap);
                let merged =
                    self.resolve_concat_symbol(&vocab_index, left, right, &extras, vocab_cap);
                pair_table.insert(left_sym, right_sym, entry.value, merged);
            }
        }
        let mut byte_symbols = [UNMAPPED_SYMBOL; 256];
        for (byte, slot) in byte_symbols.iter_mut().enumerate() {
            let spelling = BYTE_ENCODER[byte];
            let width = if spelling[1] == 0 { 1 } else { 2 };
            *slot = self.resolve_symbol(&vocab_index, &spelling[..width], &extras, vocab_cap);
        }
        Ok((pair_table, byte_symbols, vocab_cap))
    }

    /// Register a nonempty spelling; the first registration of each spelling wins.
    pub(super) fn add_special(&mut self, bytes: &[u8], id: i32) -> io::Result<()> {
        Self::insert_special(&mut self.specials, &mut self.special_first_bytes, bytes, id)
    }

    fn insert_special(
        specials: &mut Vec<(Vec<u8>, i32)>,
        first_bytes: &mut [u64; 4],
        bytes: &[u8],
        id: i32,
    ) -> io::Result<()> {
        if specials.iter().any(|(spelling, _)| spelling == bytes) {
            return Ok(());
        }
        if bytes.is_empty() || bytes.len() > MAX_SPELLING_BYTES || specials.len() >= MAX_SPECIALS {
            return Err(invalid("Qwen tokenizer special exceeds its bounds."));
        }
        let owned = copy_bytes(bytes)?;
        specials.try_reserve(1).map_err(io::Error::other)?;
        mark_special_first_byte(first_bytes, bytes[0]);
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
        Self::insert_special(
            &mut self.specials,
            &mut self.special_first_bytes,
            spelling,
            id,
        )
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

    fn find_next_special(&self, text: &[u8], position: usize) -> Option<(usize, usize)> {
        let first_offset = text[position..]
            .iter()
            .position(|&byte| has_special_first_byte(&self.special_first_bytes, byte))?;
        let scan_start = position + first_offset;
        let mut best: Option<(usize, usize)> = None;
        for (index, (spelling, _)) in self.specials.iter().enumerate() {
            let search_end = if let Some((previous, _)) = best {
                if previous == scan_start {
                    break;
                }
                previous
                    .saturating_sub(1)
                    .saturating_add(spelling.len())
                    .min(text.len())
            } else {
                text.len()
            };
            if let Some(offset) = text[scan_start..search_end]
                .windows(spelling.len())
                .position(|window| window == spelling)
            {
                let found = scan_start + offset;
                if best.is_none_or(|(previous, _)| found < previous) {
                    best = Some((found, index));
                }
            }
        }
        best
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
        let mut probes = 0;
        if self.specials.is_empty() {
            self.encode_segment(text, &mut ids, &mut probes, &mut scratch)?;
        } else {
            let mut position = 0;
            while position < text.len() {
                let Some((found, index)) = self.find_next_special(text, position) else {
                    self.encode_segment(&text[position..], &mut ids, &mut probes, &mut scratch)?;
                    break;
                };
                self.encode_segment(&text[position..found], &mut ids, &mut probes, &mut scratch)?;
                let (spelling, id) = &self.specials[index];
                ids.push(*id);
                position = found + spelling.len();
            }
        }
        if add_eos && self.eos >= 0 {
            ids.push(self.eos);
        }
        Ok(ids)
    }

    fn lookup_merge_pair(
        &self,
        left_sym: u32,
        right_sym: u32,
        left_len: usize,
        right_len: usize,
        probes: &mut usize,
    ) -> io::Result<(i32, u32)> {
        *probes += 1;
        if *probes > MAX_MERGE_PROBES {
            return Err(invalid("Qwen tokenizer merge work exceeds its bound."));
        }
        if left_sym == UNMAPPED_SYMBOL
            || right_sym == UNMAPPED_SYMBOL
            || left_len + right_len + 1 > self.max_merge_bytes
        {
            return Ok((NO_MERGE_RANK, UNMAPPED_SYMBOL));
        }
        Ok(self.pair_table.get(left_sym, right_sym))
    }

    fn init_chunk_symbols(
        &self,
        chunk: &[u8],
        probes: &mut usize,
        scratch: &mut ChunkScratch,
    ) -> io::Result<()> {
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
                sym: self.byte_symbols[usize::from(byte)],
                next_merged: UNMAPPED_SYMBOL,
            });
            scratch.ranks.push(NO_MERGE_RANK);
        }
        for index in 0..count - 1 {
            let right_idx = index + 1;
            let (left_sym, left_span) = (scratch.symbols[index].sym, scratch.symbols[index].span());
            let (right_sym, right_span) = (
                scratch.symbols[right_idx].sym,
                scratch.symbols[right_idx].span(),
            );
            let (rank, merged) = self.lookup_merge_pair(
                left_sym,
                right_sym,
                left_span.len(),
                right_span.len(),
                probes,
            )?;
            scratch.ranks[index] = rank;
            scratch.symbols[index].next_merged = merged;
            if rank != NO_MERGE_RANK {
                scratch.heap.push(Reverse((
                    rank,
                    index,
                    scratch.symbols[index].end,
                    scratch.symbols[right_idx].end,
                )));
            }
        }
        Ok(())
    }

    fn refresh_neighbor_pair(
        &self,
        left_idx: usize,
        right_idx: usize,
        probes: &mut usize,
        scratch: &mut ChunkScratch,
    ) -> io::Result<()> {
        let (left_sym, left_span) = (
            scratch.symbols[left_idx].sym,
            scratch.symbols[left_idx].span(),
        );
        let (right_sym, right_span) = (
            scratch.symbols[right_idx].sym,
            scratch.symbols[right_idx].span(),
        );
        let (rank, merged) = self.lookup_merge_pair(
            left_sym,
            right_sym,
            left_span.len(),
            right_span.len(),
            probes,
        )?;
        scratch.ranks[left_idx] = rank;
        scratch.symbols[left_idx].next_merged = merged;
        if rank != NO_MERGE_RANK {
            scratch.heap.push(Reverse((
                rank,
                left_idx,
                scratch.symbols[left_idx].end,
                scratch.symbols[right_idx].end,
            )));
        }
        Ok(())
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
            let sym = self.byte_symbols[usize::from(*byte)];
            let spelling = BYTE_ENCODER[usize::from(*byte)];
            let width = if spelling[1] == 0 { 1 } else { 2 };
            self.emit_symbol(sym, &spelling[..width], ids);
            return Ok(());
        }
        self.init_chunk_symbols(chunk, probes, scratch)?;
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
            let merged_sym = scratch.symbols[left].next_merged;
            scratch.symbols[left].end = exp_right_end;
            scratch.symbols[left].next = after;
            scratch.symbols[left].sym = merged_sym;
            if after != usize::MAX {
                scratch.symbols[after].prev = left;
            }
            scratch.symbols[right].active = false;
            scratch.symbols[right].prev = usize::MAX;
            scratch.symbols[right].next = usize::MAX;
            scratch.ranks[right] = NO_MERGE_RANK;
            scratch.symbols[right].next_merged = UNMAPPED_SYMBOL;

            let prev = scratch.symbols[left].prev;
            if prev != usize::MAX {
                self.refresh_neighbor_pair(prev, left, probes, scratch)?;
            }
            if after != usize::MAX {
                self.refresh_neighbor_pair(left, after, probes, scratch)?;
            } else {
                scratch.ranks[left] = NO_MERGE_RANK;
                scratch.symbols[left].next_merged = UNMAPPED_SYMBOL;
            }
        }
        let mut cursor = 0;
        while cursor != usize::MAX {
            let symbol = scratch.symbols[cursor];
            self.emit_symbol(symbol.sym, &scratch.encoded[symbol.span()], ids);
            cursor = symbol.next;
        }
        Ok(())
    }

    fn emit_symbol(&self, sym: u32, piece: &[u8], ids: &mut Vec<i32>) {
        if sym < self.vocab_cap
            && let Some(entry) = usize::try_from(sym)
                .ok()
                .and_then(|idx| self.vocabulary.entries.get(idx))
        {
            ids.push(entry.value);
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

#[cfg(test)]
mod tests {
    use super::Tokenizer;
    use std::io;

    #[test]
    fn pair_index_chains_non_vocab_intermediate_merges() -> io::Result<()> {
        let mut tokenizer = Tokenizer::new();
        for (idx, piece) in [b"a".as_slice(), b"b", b"c", b"abc", b"\xC4\xA0"]
            .into_iter()
            .enumerate()
        {
            tokenizer.add_token(piece, i32::try_from(idx).map_err(io::Error::other)?)?;
        }
        // "ab" is not in the vocabulary yet must still chain to "abc".
        tokenizer.add_merge(b"a b", 0)?;
        tokenizer.add_merge(b"ab c", 1)?;
        tokenizer.finish()?;

        assert_eq!(tokenizer.encode(b"abc abc", false)?, [3, 4, 3]);
        Ok(())
    }

    #[test]
    fn pair_index_honors_last_wins_merges_and_raw_byte_fallback() -> io::Result<()> {
        let mut tokenizer = Tokenizer::new();
        for (idx, piece) in [b"a".as_slice(), b"b", b"c", b"ab"].into_iter().enumerate() {
            tokenizer.add_token(piece, i32::try_from(idx).map_err(io::Error::other)?)?;
        }
        // "b c" overrides an earlier worse rank and wins before "a b"; because "bc" is not in
        // vocab, raw-byte fallback decomposes the merged piece into ['b', 'c'].
        tokenizer.add_merge(b"b c", 5)?;
        tokenizer.add_merge(b"a b", 1)?;
        tokenizer.add_merge(b"b c", 0)?;
        tokenizer.finish()?;

        assert_eq!(tokenizer.encode(b"abc", false)?, [0, 1, 2]);
        assert_eq!(tokenizer.encode(b"ab", false)?, [3]);
        Ok(())
    }

    #[test]
    fn special_first_byte_bitmap_skips_prose_and_tracks_post_finish_additions() -> io::Result<()> {
        let mut tokenizer = Tokenizer::new();
        for (idx, piece) in [
            b"h".as_slice(),
            b"i",
            b"hi",
            b"<|endoftext|>",
            b"<|talk|>",
            b"!tag",
        ]
        .into_iter()
        .enumerate()
        {
            tokenizer.add_token(piece, i32::try_from(idx).map_err(io::Error::other)?)?;
        }
        tokenizer.add_merge(b"h i", 0)?;
        tokenizer.finish()?;
        tokenizer.add_special_id(4)?;
        tokenizer.add_special(b"!tag", 5)?;

        assert_eq!(tokenizer.encode(b"hi!tag<|talk|>", true)?, [2, 5, 4, 3]);
        Ok(())
    }

    #[test]
    fn reserved_tables_and_hash_pair_index_handle_collisions_and_extras() -> io::Result<()> {
        let mut tokenizer = Tokenizer::new();
        let pad_count = 512_usize;
        tokenizer.reserve_tokens(pad_count + 5, pad_count * 6 + 64)?;
        tokenizer.reserve_merges(16, 256)?;
        for idx in 0..pad_count {
            let piece = format!("t{idx:03}");
            tokenizer.add_token(
                piece.as_bytes(),
                i32::try_from(idx).map_err(io::Error::other)?,
            )?;
        }
        let base = i32::try_from(pad_count).map_err(io::Error::other)?;
        tokenizer.add_token(b"a", base)?;
        tokenizer.add_token(b"b", base + 1)?;
        tokenizer.add_token(b"c", base + 2)?;
        tokenizer.add_token(b"abc", base + 3)?;
        tokenizer.add_token(b"unseen_x", base + 4)?;
        // Also test non-vocab intermediate merge ("ab") chaining + duplicate last-wins override.
        tokenizer.add_merge(b"a b", 5)?;
        tokenizer.add_merge(b"a b", 0)?;
        tokenizer.add_merge(b"ab c", 1)?;
        tokenizer.finish()?;
        assert!(tokenizer.pair_table.slots.len().is_power_of_two());
        assert_eq!(tokenizer.encode(b"abc", false)?, [base + 3]);
        Ok(())
    }
}
