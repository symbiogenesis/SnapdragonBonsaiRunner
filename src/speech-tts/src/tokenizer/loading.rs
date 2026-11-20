//! Safe metadata intake shared by serving and retained tokenizer controls.
#![forbid(unsafe_code)]

use std::io;
#[cfg(test)]
use std::path::Path;

use crate::weights::{Weights, invalid};

use super::Tokenizer;

const TOKENS: &[u8] = b"tokenizer.ggml.tokens";
const MERGES: &[u8] = b"tokenizer.ggml.merges";
const SPECIAL_KEYS: [&[u8]; 5] = [
    b"qwen3-tts.text.im_start_id",
    b"qwen3-tts.text.im_end_id",
    b"qwen3-tts.text.tts_pad_id",
    b"qwen3-tts.text.tts_bos_id",
    b"qwen3-tts.text.tts_eos_id",
];
const ALL_KEYS: [&[u8]; 7] = [
    TOKENS,
    MERGES,
    SPECIAL_KEYS[0],
    SPECIAL_KEYS[1],
    SPECIAL_KEYS[2],
    SPECIAL_KEYS[3],
    SPECIAL_KEYS[4],
];

trait Metadata {
    fn find(&self, name: &[u8]) -> Option<usize>;
    fn string_count(&self, key: usize) -> io::Result<usize>;
    fn string(&self, key: usize, index: usize) -> io::Result<&[u8]>;
    fn string_table(&self, _key: usize) -> Option<&[String]> {
        None
    }
    fn u32_value(&self, key: usize) -> io::Result<u32>;
}

fn key_name(key: usize) -> io::Result<&'static str> {
    ALL_KEYS
        .get(key)
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .ok_or_else(invalid)
}

impl Metadata for Weights {
    fn find(&self, name: &[u8]) -> Option<usize> {
        let index = ALL_KEYS.iter().position(|key| *key == name)?;
        let key = std::str::from_utf8(name).ok()?;
        self.metadata_contains(key).then_some(index)
    }

    fn string_count(&self, key: usize) -> io::Result<usize> {
        Ok(self.string_array(key_name(key)?)?.len())
    }

    fn string(&self, key: usize, index: usize) -> io::Result<&[u8]> {
        self.string_array(key_name(key)?)?
            .get(index)
            .map(String::as_bytes)
            .ok_or_else(invalid)
    }

    fn string_table(&self, key: usize) -> Option<&[String]> {
        self.string_array(key_name(key).ok()?).ok()
    }

    fn u32_value(&self, key: usize) -> io::Result<u32> {
        self.u32(key_name(key)?)
    }
}

#[cfg(test)]
fn reject_stage(stage: u8) -> io::Result<()> {
    if tests::REJECT.with(|reject| reject.get() == stage) {
        return Err(io::Error::other("Injected tokenizer construction failure."));
    }
    Ok(())
}

#[cfg(not(test))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "Keep test-only construction faults on the serving intake path."
)]
const fn reject_stage(_: u8) -> io::Result<()> {
    Ok(())
}

fn tables(metadata: &impl Metadata) -> io::Result<Tokenizer> {
    let token_key = metadata.find(TOKENS).ok_or_else(invalid)?;
    let merge_key = metadata.find(MERGES).ok_or_else(invalid)?;
    let tokens = metadata.string_count(token_key)?;
    let merges = metadata.string_count(merge_key)?;
    if tokens > super::MAX_ENTRIES || merges > super::MAX_ENTRIES {
        return Err(invalid());
    }
    reject_stage(1)?;
    let mut owner = Tokenizer::new();
    if let Some(table) = metadata.string_table(token_key) {
        let total_bytes: usize = table.iter().map(String::len).sum();
        owner.reserve_tokens(table.len(), total_bytes)?;
        for (index, entry) in table.iter().enumerate() {
            reject_stage(2)?;
            owner.add_token(
                entry.as_bytes(),
                i32::try_from(index).map_err(io::Error::other)?,
            )?;
        }
    } else {
        for index in 0..tokens {
            reject_stage(2)?;
            owner.add_token(
                metadata.string(token_key, index)?,
                i32::try_from(index).map_err(io::Error::other)?,
            )?;
        }
    }
    if let Some(table) = metadata.string_table(merge_key) {
        let total_bytes: usize = table.iter().map(String::len).sum();
        owner.reserve_merges(table.len(), total_bytes)?;
        for (index, entry) in table.iter().enumerate() {
            reject_stage(3)?;
            owner.add_merge(
                entry.as_bytes(),
                i32::try_from(index).map_err(io::Error::other)?,
            )?;
        }
    } else {
        for index in 0..merges {
            reject_stage(3)?;
            owner.add_merge(
                metadata.string(merge_key, index)?,
                i32::try_from(index).map_err(io::Error::other)?,
            )?;
        }
    }
    reject_stage(4)?;
    owner.finish()?;
    Ok(owner)
}

fn candidate(metadata: &impl Metadata) -> io::Result<Tokenizer> {
    let mut owner = tables(metadata)?;
    for name in SPECIAL_KEYS {
        let key = metadata.find(name).ok_or_else(invalid)?;
        reject_stage(5)?;
        let id = i32::try_from(metadata.u32_value(key)?).map_err(io::Error::other)?;
        owner.add_special_id(id)?;
    }
    Ok(owner)
}

pub(super) fn from_weights(weights: &Weights) -> io::Result<Tokenizer> {
    candidate(weights)
}

#[cfg(test)]
pub(super) fn load(path: &Path) -> io::Result<Tokenizer> {
    let metadata = Weights::open_metadata(path)?;
    candidate(&metadata)
}

#[cfg(test)]
pub(super) fn owner_created() {
    tests::OWNER_COUNTS.with(|counts| {
        let (created, freed) = counts.get();
        counts.set((created + 1, freed));
    });
}

#[cfg(test)]
impl Drop for Tokenizer {
    fn drop(&mut self) {
        tests::OWNER_COUNTS.with(|counts| {
            let (created, freed) = counts.get();
            counts.set((created, freed + 1));
        });
    }
}

#[cfg(test)]
#[path = "loading_tests.rs"]
mod tests;
