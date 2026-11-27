//! Bounded, native-independent GGUF v3 index and tokenizer admission.
//!
//! This checks structure, not artifact authenticity or inference equivalence.
//! The service must retain its digest-verified file guard through recognition.
#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep the source-bound GGUF preflight private to the ASR crate."
)]

use std::{
    collections::BTreeMap,
    io::{self, Read, Seek},
};

#[path = "intake_decoder.rs"]
mod decoder;
#[path = "intake_encoder.rs"]
mod encoder;
#[path = "intake_model.rs"]
mod model;

const MAX_ITEMS: u64 = 262_144;
const MAX_METADATA: usize = 64 * 1024 * 1024;
const MAX_FILE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid Qwen3-ASR GGUF structure.",
    )
}

#[derive(Debug)]
enum Value {
    Integer { kind: u32, value: u64 },
    Float { kind: u32, value: f64 },
    Boolean(bool),
    Text(String),
    Strings(Vec<String>),
    Array { kind: u32, count: usize },
    UnsetToken,
    Other,
}

#[derive(Debug)]
struct Tensor {
    shape: Vec<u64>,
    kind: u32,
    offset: u64,
    bytes: u64,
}

struct Index {
    metadata: BTreeMap<String, Value>,
    tensors: BTreeMap<String, Tensor>,
}

const LINEAR_TYPES: &[u32] = &[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 30];

fn check_cancel(cancel: &mut (impl FnMut() -> bool + ?Sized)) -> io::Result<()> {
    if cancel() {
        Err(io::Error::from(io::ErrorKind::Interrupted))
    } else {
        Ok(())
    }
}

const fn asr_index_policy(metadata_budget: usize) -> bonsai_gguf::IndexPolicy<'static> {
    bonsai_gguf::IndexPolicy {
        allow_v2: false,
        allow_zero_tensors: false,
        allow_zero_metadata: false,
        max_file_bytes: MAX_FILE_BYTES,
        max_tensors: 8192,
        max_metadata: 65_536,
        metadata_budget,
        max_key_bytes: 1024,
        max_string_bytes: 65_536,
        max_array_string_bytes: 4096,
        max_retained_string_array_bytes: None,
        max_array_items: MAX_ITEMS,
        max_alignment: 4096,
        allow_positive_i32_alignment: false,
        reject_nonfinite_floats: false,
        reject_interior_nul_keys: true,
        reject_interior_nul_strings: false,
        kv_overhead_bytes: 128,
        string_overhead_bytes: 32,
        tensor_overhead_bytes: 512,
        charge_string_array_slots: true,
        charge_skipped_scalar_arrays: true,
        prism_dialect: false,
        allowed_tensor_types: Some(LINEAR_TYPES),
        ..bonsai_gguf::IndexPolicy::new()
    }
}

fn signed(kind: u32, value: i64) -> Value {
    if kind == 5 && value == -1 {
        return Value::UnsetToken;
    }
    u64::try_from(value).map_or(Value::Other, |value| Value::Integer { kind, value })
}

fn map_index_value(entry: bonsai_gguf::IndexValue) -> Value {
    use bonsai_gguf::{IndexValue, MetadataArray, MetadataValue};
    match entry {
        IndexValue::Metadata(meta) => match meta {
            MetadataValue::U8(v) => Value::Integer {
                kind: 0,
                value: u64::from(v),
            },
            MetadataValue::I8(v) => signed(1, i64::from(v)),
            MetadataValue::U16(v) => Value::Integer {
                kind: 2,
                value: u64::from(v),
            },
            MetadataValue::I16(v) => signed(3, i64::from(v)),
            MetadataValue::U32(v) => Value::Integer {
                kind: 4,
                value: u64::from(v),
            },
            MetadataValue::I32(v) => signed(5, i64::from(v)),
            MetadataValue::F32(v) => Value::Float {
                kind: 6,
                value: f64::from(v),
            },
            MetadataValue::Bool(v) => Value::Boolean(v),
            MetadataValue::String(v) => Value::Text(v),
            MetadataValue::U64(v) => Value::Integer { kind: 10, value: v },
            MetadataValue::I64(v) => signed(11, v),
            MetadataValue::F64(v) => Value::Float { kind: 12, value: v },
            MetadataValue::Array(MetadataArray::String(v)) => Value::Strings(v),
            MetadataValue::Array(_) => Value::Other,
        },
        IndexValue::SkippedArray {
            element_type_id,
            count,
        } => Value::Array {
            kind: element_type_id,
            count,
        },
    }
}

#[cfg(any(feature = "native-engine", test))]
pub(super) fn validate(source: &mut (impl Read + Seek), length: u64) -> io::Result<()> {
    model::validate(&read_index(source, length, &mut || false)?)
}

fn read_index(
    source: &mut (impl Read + Seek),
    length: u64,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Index> {
    read_index_with_budget(source, length, MAX_METADATA, cancel)
}

fn read_index_with_budget(
    source: &mut (impl Read + Seek),
    length: u64,
    budget: usize,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Index> {
    let parsed =
        bonsai_gguf::read_index_cancellable(source, length, &asr_index_policy(budget), cancel)
            .map_err(|err| {
                if err.kind() == io::ErrorKind::InvalidData {
                    invalid()
                } else {
                    err
                }
            })?;
    let metadata = parsed
        .metadata
        .into_iter()
        .map(|(key, value)| (key, map_index_value(value)))
        .collect();
    let tensors = parsed
        .tensors
        .into_iter()
        .map(|(name, tensor)| {
            (
                name,
                Tensor {
                    shape: tensor.dimensions,
                    kind: tensor.type_id,
                    offset: tensor.offset,
                    bytes: tensor.bytes,
                },
            )
        })
        .collect();
    Ok(Index { metadata, tensors })
}

pub(super) fn decoder(
    source: &mut (impl Read + Seek),
    length: u64,
    precision: crate::decoder::CachePrecision,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<crate::decoder::Model> {
    let index = read_index(source, length, cancel)?;
    model::validate(&index)?;
    check_cancel(cancel)?;
    decoder::load(&index, source, precision, cancel)
}

pub(super) fn encoder(
    source: &mut (impl Read + Seek),
    length: u64,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<(crate::encoder::Encoder, crate::frontend::Frontend)> {
    let index = read_index(source, length, cancel)?;
    model::validate(&index)?;
    check_cancel(cancel)?;
    encoder::load(&index, source, cancel)
}

pub(super) fn tokenizer(
    source: &mut (impl Read + Seek),
    length: u64,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<(crate::tokenizer::Tokenizer, crate::protocol::PromptTokens)> {
    let mut index = read_index(source, length, cancel)?;
    model::validate(&index)?;
    check_cancel(cancel)?;
    load_tokenizer(&mut index, cancel)
}

#[cfg(not(all(windows, feature = "rust-qualification")))]
pub(super) fn load_artifacts(
    source: &mut (impl Read + Seek),
    length: u64,
    precision: crate::decoder::CachePrecision,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<(
    crate::tokenizer::Tokenizer,
    crate::protocol::PromptTokens,
    crate::encoder::Encoder,
    crate::frontend::Frontend,
    crate::decoder::Model,
)> {
    let mut index = read_index(source, length, cancel)?;
    model::validate(&index)?;
    check_cancel(cancel)?;
    let (tokenizer, prompt_tokens) = load_tokenizer(&mut index, cancel)?;
    check_cancel(cancel)?;
    let (encoder, frontend) = encoder::load(&index, source, cancel)?;
    check_cancel(cancel)?;
    let decoder = decoder::load(&index, source, precision, cancel)?;
    Ok((tokenizer, prompt_tokens, encoder, frontend, decoder))
}

#[cfg(all(windows, feature = "rust-qualification"))]
pub(super) fn load_artifacts_mapped(
    mapping: &std::sync::Arc<bonsai_windows_process::MappedFile>,
    precision: crate::decoder::CachePrecision,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<(
    crate::tokenizer::Tokenizer,
    crate::protocol::PromptTokens,
    crate::encoder::Encoder,
    crate::frontend::Frontend,
    crate::decoder::Model,
)> {
    let length = u64::try_from(mapping.len()).map_err(|_| invalid())?;
    let mut cursor = io::Cursor::new(mapping.as_slice());
    let mut index = read_index(&mut cursor, length, cancel)?;
    model::validate(&index)?;
    check_cancel(cancel)?;
    let (tokenizer, prompt_tokens) = load_tokenizer(&mut index, cancel)?;
    check_cancel(cancel)?;
    let (encoder_res, decoder_res) = std::thread::scope(|scope| {
        let encoder_handle = scope.spawn(|| encoder::load_mapped(&index, mapping, &mut || false));
        let decoder_res = decoder::load_mapped(&index, mapping, precision, &mut || false);
        let encoder_res = encoder_handle
            .join()
            .unwrap_or_else(|_| Err(io::Error::other("ASR encoder intake thread panicked.")));
        (encoder_res, decoder_res)
    });
    check_cancel(cancel)?;
    let (encoder, frontend) = encoder_res?;
    let decoder = decoder_res?;
    Ok((tokenizer, prompt_tokens, encoder, frontend, decoder))
}

fn load_tokenizer(
    index: &mut Index,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<(crate::tokenizer::Tokenizer, crate::protocol::PromptTokens)> {
    if let Some(Value::Text(pre)) = index.metadata.get("tokenizer.ggml.pre")
        && !matches!(pre.as_str(), "" | "qwen2")
    {
        return Err(invalid());
    }
    let Some(Value::Strings(tokens)) = index.metadata.remove("tokenizer.ggml.tokens") else {
        return Err(invalid());
    };
    let Some(Value::Strings(merges)) = index.metadata.remove("tokenizer.ggml.merges") else {
        return Err(invalid());
    };
    let Some(Value::Integer { value: eos, .. }) =
        index.metadata.remove("tokenizer.ggml.eos_token_id")
    else {
        return Err(invalid());
    };
    let tokenizer = crate::tokenizer::Tokenizer::from_tables(
        tokens,
        merges,
        u32::try_from(eos).map_err(|_| invalid())?,
    )?;
    check_cancel(cancel)?;
    let prompt = model::prompt_tokens(index, &tokenizer)?;
    Ok((tokenizer, prompt))
}

#[cfg(test)]
#[path = "intake_tests.rs"]
mod tests;
