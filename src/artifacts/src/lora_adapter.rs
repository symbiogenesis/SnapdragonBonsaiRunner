//! Offline, bounded verification for stackable multi-rank `LoRA` GGUF adapters.

use crate::{ArtifactError, hold_ancestors, open_locked_file_with_access, verification_buffer};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const DEFAULT_GGUF_ALIGNMENT: u32 = 32;
const MAX_GGUF_TENSOR_COUNT: usize = 4_096;
const MAX_GGUF_METADATA_COUNT: usize = 4_096;
const MAX_GGUF_STRING_BYTES: usize = 65_536;
const MAX_GGUF_ARRAY_LEN: usize = 65_536;
const FINITE_SCAN_CHUNK_BYTES: usize = 65_536;

/// Target admission policy for multi-rank `LoRA` GGUF verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiRankEnvelope {
    /// Excludes FFN gating (`ffn_gate`, `ffn_up`) and quantized recurrent outputs (`ssm_out`).
    NpuFriendly,
    /// Permits all valid linear projections except token/output embeddings and `ssm_out`/norms.
    FullMatrix,
}

/// Verified multi-rank `LoRA` GGUF adapter holding deny-write/deny-delete Windows read locks.
pub struct VerifiedMultiRankAdapter {
    path: PathBuf,
    bytes: u64,
    sha256: String,
    max_rank: u16,
    target_count: usize,
    payload_bytes: u64,
    envelope: MultiRankEnvelope,
    _handles: Vec<File>,
}

impl VerifiedMultiRankAdapter {
    /// Canonical filesystem path of the verified multi-rank adapter.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Exact byte length of the verified GGUF file.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Lowercase 64-character hexadecimal SHA-256 digest of the complete file.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Maximum `LoRA` rank observed across all target `(lora_a, lora_b)` pairs.
    #[must_use]
    pub const fn max_rank(&self) -> u16 {
        self.max_rank
    }

    /// Number of verified target matrix pairs.
    #[must_use]
    pub const fn target_count(&self) -> usize {
        self.target_count
    }

    /// Total unpadded factor tensor payload bytes across all verified tensors.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Policy envelope enforced during verification.
    #[must_use]
    pub const fn envelope(&self) -> MultiRankEnvelope {
        self.envelope
    }

    /// Read-only locked leaf file handle held until this guard drops.
    #[must_use]
    pub fn file(&self) -> Option<&File> {
        self._handles.last()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FactorDtype {
    F32,
    F16,
}

impl FactorDtype {
    const fn from_gguf(kind: u32) -> Result<Self, ArtifactError> {
        match kind {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            _ => Err(ArtifactError::Identity),
        }
    }

    const fn byte_width(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

struct TensorDesc {
    name: String,
    dims: [u64; 2],
    dtype: FactorDtype,
    offset: u64,
    byte_len: u64,
}

#[derive(Default)]
struct PairEntry {
    a: Option<(u64, u64, FactorDtype)>,
    b: Option<(u64, u64, FactorDtype)>,
}

struct BoundedReader<R> {
    reader: R,
    position: u64,
    limit: u64,
}

impl<R: Read> BoundedReader<R> {
    const fn new(reader: R, limit: u64) -> Self {
        Self {
            reader,
            position: 0,
            limit,
        }
    }

    fn read_into(&mut self, buf: &mut [u8]) -> Result<(), ArtifactError> {
        let len_u64 = u64::try_from(buf.len()).map_err(|_| ArtifactError::Identity)?;
        let next = self
            .position
            .checked_add(len_u64)
            .filter(|&p| p <= self.limit)
            .ok_or(ArtifactError::Identity)?;
        self.reader
            .read_exact(buf)
            .map_err(|_| ArtifactError::Identity)?;
        self.position = next;
        Ok(())
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], ArtifactError> {
        let mut array = [0_u8; N];
        self.read_into(&mut array)?;
        Ok(array)
    }

    fn read_exact_bytes(&mut self, len: usize) -> Result<Vec<u8>, ArtifactError> {
        let mut buf = Vec::new();
        buf.try_reserve_exact(len)
            .map_err(|_| ArtifactError::Identity)?;
        buf.resize(len, 0);
        self.read_into(&mut buf)?;
        Ok(buf)
    }

    fn read_u32(&mut self) -> Result<u32, ArtifactError> {
        self.read_array::<4>().map(u32::from_le_bytes)
    }

    fn read_u64(&mut self) -> Result<u64, ArtifactError> {
        self.read_array::<8>().map(u64::from_le_bytes)
    }

    fn read_f32(&mut self) -> Result<f32, ArtifactError> {
        self.read_array::<4>().map(f32::from_le_bytes)
    }

    fn read_count(&mut self, maximum: usize) -> Result<usize, ArtifactError> {
        let count = usize::try_from(self.read_u64()?).map_err(|_| ArtifactError::Identity)?;
        if count > maximum {
            return Err(ArtifactError::Identity);
        }
        Ok(count)
    }

    fn read_string(&mut self) -> Result<String, ArtifactError> {
        let len = self.read_count(MAX_GGUF_STRING_BYTES)?;
        let bytes = self.read_exact_bytes(len)?;
        String::from_utf8(bytes).map_err(|_| ArtifactError::Identity)
    }

    fn skip_value(&mut self, kind: u32, depth: u32) -> Result<(), ArtifactError> {
        if depth > 4 {
            return Err(ArtifactError::Identity);
        }
        let scalar_bytes = match kind {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            8 => {
                let _ = self.read_string()?;
                return Ok(());
            }
            9 => {
                let elem_kind = self.read_u32()?;
                let count = self.read_count(MAX_GGUF_ARRAY_LEN)?;
                for _ in 0..count {
                    self.skip_value(elem_kind, depth + 1)?;
                }
                return Ok(());
            }
            _ => return Err(ArtifactError::Identity),
        };
        let mut scratch = [0_u8; 8];
        self.read_into(&mut scratch[..scalar_bytes])
    }
}

fn hash_locked_file(file: &mut File, expected_len: u64) -> Result<String, ArtifactError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|_| ArtifactError::Identity)?;
    let mut buffer = verification_buffer(expected_len);
    let digest = crate::hash_reader(file, expected_len, &mut buffer, &mut || Ok(()))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|_| ArtifactError::Identity)?;
    Ok(digest)
}

fn validate_target_name(
    base_weight_name: &str,
    envelope: MultiRankEnvelope,
) -> Result<(), ArtifactError> {
    if !base_weight_name.ends_with(".weight") {
        return Err(ArtifactError::Identity);
    }
    if base_weight_name == "token_embd.weight"
        || base_weight_name == "output.weight"
        || base_weight_name == "output_norm.weight"
        || base_weight_name.ends_with(".attn_norm.weight")
        || base_weight_name.ends_with(".post_attention_norm.weight")
        || base_weight_name.ends_with(".ssm_conv1d.weight")
        || base_weight_name.ends_with(".ssm_a.weight")
        || base_weight_name.ends_with(".ssm_dt.weight")
        || base_weight_name.ends_with(".ssm_norm.weight")
        || base_weight_name.ends_with(".ssm_out.weight")
    {
        return Err(ArtifactError::Identity);
    }
    if envelope == MultiRankEnvelope::NpuFriendly
        && (base_weight_name.ends_with(".ffn_gate.weight")
            || base_weight_name.ends_with(".ffn_up.weight"))
    {
        return Err(ArtifactError::Identity);
    }
    Ok(())
}

fn check_f32_finite_slice(bytes: &[u8]) -> Result<(), ArtifactError> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(ArtifactError::Identity);
    }
    let (blocks, tail) = chunks.as_chunks::<64>();
    for block in blocks {
        let mut nonfinite = false;
        for &raw in block {
            nonfinite |= (u32::from_le_bytes(raw) & 0x7f80_0000) == 0x7f80_0000;
        }
        if nonfinite {
            return Err(ArtifactError::Identity);
        }
    }
    for &raw in tail {
        if (u32::from_le_bytes(raw) & 0x7f80_0000) == 0x7f80_0000 {
            return Err(ArtifactError::Identity);
        }
    }
    Ok(())
}

fn check_f16_finite_slice(bytes: &[u8]) -> Result<(), ArtifactError> {
    let (chunks, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(ArtifactError::Identity);
    }
    let (blocks, tail) = chunks.as_chunks::<64>();
    for block in blocks {
        let mut nonfinite = false;
        for &raw in block {
            nonfinite |= (u16::from_le_bytes(raw) & 0x7c00) == 0x7c00;
        }
        if nonfinite {
            return Err(ArtifactError::Identity);
        }
    }
    for &raw in tail {
        if (u16::from_le_bytes(raw) & 0x7c00) == 0x7c00 {
            return Err(ArtifactError::Identity);
        }
    }
    Ok(())
}

fn check_finite_slice(bytes: &[u8], dtype: FactorDtype) -> Result<(), ArtifactError> {
    match dtype {
        FactorDtype::F32 => check_f32_finite_slice(bytes),
        FactorDtype::F16 => check_f16_finite_slice(bytes),
    }
}

fn scan_tensor_finite(
    file: &mut File,
    abs_offset: u64,
    byte_len: u64,
    dtype: FactorDtype,
    scan_buf: &mut [u8],
) -> Result<(), ArtifactError> {
    file.seek(SeekFrom::Start(abs_offset))
        .map_err(|_| ArtifactError::Identity)?;
    let width = dtype.byte_width();
    let max_take = u64::try_from(scan_buf.len())
        .map_err(|_| ArtifactError::Identity)?
        .checked_div(width)
        .and_then(|elements| elements.checked_mul(width))
        .filter(|&bytes| bytes >= width)
        .ok_or(ArtifactError::Identity)?;
    let mut remaining = byte_len;
    while remaining > 0 {
        let take = usize::try_from(remaining.min(max_take)).map_err(|_| ArtifactError::Identity)?;
        file.read_exact(&mut scan_buf[..take])
            .map_err(|_| ArtifactError::Identity)?;
        check_finite_slice(&scan_buf[..take], dtype)?;
        remaining -= u64::try_from(take).map_err(|_| ArtifactError::Identity)?;
    }
    Ok(())
}

struct ParsedHeader {
    data_start: u64,
    tensors: Vec<TensorDesc>,
}

struct TensorSpan {
    abs_start: u64,
    byte_len: u64,
    dtype: FactorDtype,
}

struct ValidatedAdapterPlan {
    max_rank: u16,
    target_count: usize,
    payload_bytes: u64,
    spans: Vec<TensorSpan>,
}

fn parse_gguf_header<R: Read>(reader: R, file_len: u64) -> Result<ParsedHeader, ArtifactError> {
    let mut reader = BoundedReader::new(reader, file_len);
    let magic = reader.read_array::<4>()?;
    if &magic != GGUF_MAGIC {
        return Err(ArtifactError::Identity);
    }
    if reader.read_u32()? != GGUF_VERSION {
        return Err(ArtifactError::Identity);
    }
    let tensor_count = reader.read_count(MAX_GGUF_TENSOR_COUNT)?;
    if tensor_count == 0 || !tensor_count.is_multiple_of(2) {
        return Err(ArtifactError::Identity);
    }
    let kv_count = reader.read_count(MAX_GGUF_METADATA_COUNT)?;

    let mut seen_keys = BTreeSet::new();
    let mut general_type: Option<String> = None;
    let mut adapter_type: Option<String> = None;
    let mut primary_alpha: Option<f32> = None;
    let mut legacy_alpha: Option<f32> = None;
    let mut alt_alpha: Option<f32> = None;
    let mut alignment: Option<u32> = None;

    for _ in 0..kv_count {
        let key = reader.read_string()?;
        if !seen_keys.insert(key.clone()) {
            return Err(ArtifactError::Identity);
        }
        let kind = reader.read_u32()?;
        match (key.as_str(), kind) {
            ("general.type", 8) => general_type = Some(reader.read_string()?),
            ("adapter.type", 8) => adapter_type = Some(reader.read_string()?),
            ("adapter.lora.alpha", 6) => primary_alpha = Some(reader.read_f32()?),
            ("adapter.lora_alpha", 6) => legacy_alpha = Some(reader.read_f32()?),
            ("lora.alpha", 6) => alt_alpha = Some(reader.read_f32()?),
            ("general.alignment", 4) => alignment = Some(reader.read_u32()?),
            (
                "general.type" | "adapter.type" | "adapter.lora.alpha" | "adapter.lora_alpha"
                | "lora.alpha" | "general.alignment",
                _,
            ) => return Err(ArtifactError::Identity),
            _ => reader.skip_value(kind, 0)?,
        }
    }

    if general_type.as_deref() != Some("adapter") || adapter_type.as_deref() != Some("lora") {
        return Err(ArtifactError::Identity);
    }

    let resolved_alpha = primary_alpha
        .or(legacy_alpha)
        .ok_or(ArtifactError::Identity)?;
    if !resolved_alpha.is_finite() || resolved_alpha <= 0.0 {
        return Err(ArtifactError::Identity);
    }
    for extra in [legacy_alpha, alt_alpha].into_iter().flatten() {
        if !extra.is_finite() || extra <= 0.0 || extra.to_bits() != resolved_alpha.to_bits() {
            return Err(ArtifactError::Identity);
        }
    }

    let align_u32 = alignment.unwrap_or(DEFAULT_GGUF_ALIGNMENT);
    if !align_u32.is_power_of_two() || !(1..=4_096).contains(&align_u32) {
        return Err(ArtifactError::Identity);
    }
    let align_u64 = u64::from(align_u32);

    let mut seen_tensors = BTreeSet::new();
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        if !seen_tensors.insert(name.clone()) {
            return Err(ArtifactError::Identity);
        }
        if reader.read_u32()? != 2 {
            return Err(ArtifactError::Identity);
        }
        let dim0 = reader.read_u64()?;
        let dim1 = reader.read_u64()?;
        if dim0 == 0 || dim1 == 0 {
            return Err(ArtifactError::Identity);
        }
        let dtype = FactorDtype::from_gguf(reader.read_u32()?)?;
        let offset = reader.read_u64()?;
        if !offset.is_multiple_of(align_u64) {
            return Err(ArtifactError::Identity);
        }
        let elems = dim0.checked_mul(dim1).ok_or(ArtifactError::Identity)?;
        let byte_len = elems
            .checked_mul(dtype.byte_width())
            .ok_or(ArtifactError::Identity)?;
        tensors.push(TensorDesc {
            name,
            dims: [dim0, dim1],
            dtype,
            offset,
            byte_len,
        });
    }

    let rem = reader.position % align_u64;
    let pad = if rem == 0 { 0 } else { align_u64 - rem };
    let data_start = reader
        .position
        .checked_add(pad)
        .filter(|&start| start <= file_len)
        .ok_or(ArtifactError::Identity)?;

    Ok(ParsedHeader {
        data_start,
        tensors,
    })
}

fn validate_header_and_pairs(
    parsed: &ParsedHeader,
    file_len: u64,
    max_rank_limit: u16,
    envelope: MultiRankEnvelope,
) -> Result<ValidatedAdapterPlan, ArtifactError> {
    let mut pairs: BTreeMap<String, PairEntry> = BTreeMap::new();
    let mut payload_bytes = 0_u64;
    let mut spans = Vec::with_capacity(parsed.tensors.len());

    for tensor in &parsed.tensors {
        let abs_start = parsed
            .data_start
            .checked_add(tensor.offset)
            .ok_or(ArtifactError::Identity)?;
        abs_start
            .checked_add(tensor.byte_len)
            .filter(|&end| end <= file_len)
            .ok_or(ArtifactError::Identity)?;
        payload_bytes = payload_bytes
            .checked_add(tensor.byte_len)
            .ok_or(ArtifactError::Identity)?;

        let (base_name, is_a) = if let Some(name) = tensor.name.strip_suffix(".lora_a") {
            (name, true)
        } else if let Some(name) = tensor.name.strip_suffix(".lora_b") {
            (name, false)
        } else {
            return Err(ArtifactError::Identity);
        };
        validate_target_name(base_name, envelope)?;
        let entry = pairs.entry(base_name.to_owned()).or_default();
        let slot = if is_a { &mut entry.a } else { &mut entry.b };
        if slot.is_some() {
            return Err(ArtifactError::Identity);
        }
        *slot = Some((tensor.dims[0], tensor.dims[1], tensor.dtype));
        spans.push(TensorSpan {
            abs_start,
            byte_len: tensor.byte_len,
            dtype: tensor.dtype,
        });
    }

    if pairs.is_empty() {
        return Err(ArtifactError::Identity);
    }

    let mut max_rank = 0_u16;
    for entry in pairs.values() {
        let (Some((_, a_rank, a_dtype)), Some((b_rank, _, b_dtype))) = (entry.a, entry.b) else {
            return Err(ArtifactError::Identity);
        };
        if a_dtype != b_dtype || a_rank != b_rank {
            return Err(ArtifactError::Identity);
        }
        let rank_u16 = u16::try_from(a_rank).map_err(|_| ArtifactError::Identity)?;
        if rank_u16 == 0 || rank_u16 > max_rank_limit {
            return Err(ArtifactError::Identity);
        }
        max_rank = max_rank.max(rank_u16);
    }

    Ok(ValidatedAdapterPlan {
        max_rank,
        target_count: pairs.len(),
        payload_bytes,
        spans,
    })
}

#[cfg(windows)]
fn scan_mapped_spans_finite(bytes: &[u8], spans: &[TensorSpan]) -> Result<(), ArtifactError> {
    for span in spans {
        let start = usize::try_from(span.abs_start).map_err(|_| ArtifactError::Identity)?;
        let len = usize::try_from(span.byte_len).map_err(|_| ArtifactError::Identity)?;
        let end = start.checked_add(len).ok_or(ArtifactError::Identity)?;
        let slice = bytes.get(start..end).ok_or(ArtifactError::Identity)?;
        check_finite_slice(slice, span.dtype)?;
    }
    Ok(())
}

/// Offline verify a stackable multi-rank `LoRA` GGUF adapter while retaining Windows
/// ancestor and leaf file locks until the returned guard is dropped.
///
/// # Errors
/// Rejects unsafe/non-Windows paths, files larger than `max_bytes`, invalid GGUF v3
/// headers or alignments, disallowed targets, unpaired/mismatched factor tensors,
/// ranks above `max_rank_limit`, or nonfinite/overflowed `F16`/`F32` weights.
pub fn verify_multi_rank_lora_gguf(
    path: &Path,
    max_rank_limit: u16,
    max_bytes: u64,
    envelope: MultiRankEnvelope,
) -> Result<VerifiedMultiRankAdapter, ArtifactError> {
    if !cfg!(windows) {
        return Err(ArtifactError::Configuration);
    }
    if max_rank_limit == 0 || max_bytes == 0 {
        return Err(ArtifactError::Configuration);
    }
    let mut checkpoint = || Ok(());
    let mut handles = Vec::new();
    let mut held_ancestors = BTreeSet::new();
    hold_ancestors(path, &mut handles, &mut held_ancestors, &mut checkpoint)?;
    let mut file = open_locked_file_with_access(path, true)?;
    let file_len = file.metadata().map_err(|_| ArtifactError::Path)?.len();
    if file_len == 0 || file_len > max_bytes {
        return Err(ArtifactError::Identity);
    }

    #[cfg(windows)]
    let (sha256, plan) = if let Some(mapped) =
        bonsai_windows_credentials::MappedLockedFile::try_open(&file, file_len)
    {
        let mapped_bytes = mapped.as_bytes();
        let parsed = parse_gguf_header(mapped_bytes, file_len)?;
        let plan = validate_header_and_pairs(&parsed, file_len, max_rank_limit, envelope)?;
        let (sha_result, scan_result) = std::thread::scope(|scope| {
            let hash_task = || {
                bonsai_windows_credentials::hash_mapped_bytes(
                    mapped_bytes,
                    &mut || Ok(()),
                    |_| ArtifactError::Identity,
                )
            };
            if let Ok(hash_handle) = std::thread::Builder::new()
                .name("lora-sha256".into())
                .spawn_scoped(scope, hash_task)
            {
                let scan_result = scan_mapped_spans_finite(mapped_bytes, &plan.spans);
                let hash_result = hash_handle.join().unwrap_or(Err(ArtifactError::Identity));
                (hash_result, scan_result)
            } else {
                let scan_result = scan_mapped_spans_finite(mapped_bytes, &plan.spans);
                let hash_result = hash_task();
                (hash_result, scan_result)
            }
        });
        drop(mapped);
        scan_result?;
        let sha256 = crate::digest_hex(sha_result?);
        (sha256, plan)
    } else {
        file.seek(SeekFrom::Start(0))
            .map_err(|_| ArtifactError::Identity)?;
        let parsed = parse_gguf_header(BufReader::new(&mut file), file_len)?;
        let plan = validate_header_and_pairs(&parsed, file_len, max_rank_limit, envelope)?;
        let max_span = plan
            .spans
            .iter()
            .map(|span| span.byte_len)
            .max()
            .unwrap_or(0);
        let buf_len = FINITE_SCAN_CHUNK_BYTES
            .min(usize::try_from(max_span).unwrap_or(FINITE_SCAN_CHUNK_BYTES))
            .max(4);
        let mut scan_buf = vec![0_u8; buf_len];
        for span in &plan.spans {
            scan_tensor_finite(
                &mut file,
                span.abs_start,
                span.byte_len,
                span.dtype,
                &mut scan_buf,
            )?;
        }
        let sha256 = hash_locked_file(&mut file, file_len)?;
        (sha256, plan)
    };

    #[cfg(not(windows))]
    let (sha256, plan) = {
        let _ = file;
        return Err(ArtifactError::Configuration);
    };

    file.seek(SeekFrom::Start(0))
        .map_err(|_| ArtifactError::Identity)?;
    handles.push(file);

    Ok(VerifiedMultiRankAdapter {
        path: path.to_owned(),
        bytes: file_len,
        sha256,
        max_rank: plan.max_rank,
        target_count: plan.target_count,
        payload_bytes: plan.payload_bytes,
        envelope,
        _handles: handles,
    })
}

#[cfg(test)]
mod tests {
    use super::{MultiRankEnvelope, verify_multi_rank_lora_gguf};
    use crate::ArtifactError;
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::path::PathBuf;

    struct TempDir(PathBuf);

    impl TempDir {
        fn create(label: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "SwiftBonsai-lora-adapter-{label}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&path)?;
            Ok(Self(path))
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn push_str(out: &mut Vec<u8>, s: &str) {
        let len = u64::try_from(s.len()).unwrap_or(0);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn push_kv_str(out: &mut Vec<u8>, key: &str, val: &str) {
        push_str(out, key);
        out.extend_from_slice(&8_u32.to_le_bytes());
        push_str(out, val);
    }

    fn push_kv_f32(out: &mut Vec<u8>, key: &str, val: f32) {
        push_str(out, key);
        out.extend_from_slice(&6_u32.to_le_bytes());
        out.extend_from_slice(&val.to_le_bytes());
    }

    fn push_kv_u32(out: &mut Vec<u8>, key: &str, val: u32) {
        push_str(out, key);
        out.extend_from_slice(&4_u32.to_le_bytes());
        out.extend_from_slice(&val.to_le_bytes());
    }

    struct TensorBuildSpec<'a> {
        name: &'a str,
        dims: [u64; 2],
        gguf_type: u32,
        data: Vec<u8>,
    }

    fn build_gguf(
        alpha: f32,
        legacy_alpha: Option<f32>,
        alignment: u32,
        tensors: Vec<TensorBuildSpec<'_>>,
    ) -> Vec<u8> {
        let mut kv = Vec::new();
        let mut kv_count = 4_u64;
        push_kv_str(&mut kv, "general.type", "adapter");
        push_kv_str(&mut kv, "adapter.type", "lora");
        push_kv_f32(&mut kv, "adapter.lora.alpha", alpha);
        push_kv_u32(&mut kv, "general.alignment", alignment);
        if let Some(leg) = legacy_alpha {
            kv_count += 1;
            push_kv_f32(&mut kv, "adapter.lora_alpha", leg);
        }

        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3_u32.to_le_bytes());
        out.extend_from_slice(&(u64::try_from(tensors.len()).unwrap_or(0)).to_le_bytes());
        out.extend_from_slice(&kv_count.to_le_bytes());
        out.extend_from_slice(&kv);

        let align_usize = usize::try_from(alignment.max(1)).unwrap_or(32);
        let mut offset = 0_usize;
        for t in &tensors {
            push_str(&mut out, t.name);
            out.extend_from_slice(&2_u32.to_le_bytes());
            out.extend_from_slice(&t.dims[0].to_le_bytes());
            out.extend_from_slice(&t.dims[1].to_le_bytes());
            out.extend_from_slice(&t.gguf_type.to_le_bytes());
            out.extend_from_slice(&(u64::try_from(offset).unwrap_or(0)).to_le_bytes());
            offset += t.data.len().next_multiple_of(align_usize);
        }
        let header_padded = out.len().next_multiple_of(align_usize);
        out.resize(header_padded, 0);
        for t in &tensors {
            out.extend_from_slice(&t.data);
            let padded = t.data.len().next_multiple_of(align_usize);
            out.resize(out.len() + (padded - t.data.len()), 0);
        }
        out
    }

    fn f16_bytes(bits: &[u16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(bits.len() * 2);
        for &b in bits {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[cfg(windows)]
    #[test]
    fn valid_npu_friendly_f16_and_f32_adapters_verify_and_retain_lock()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::create("valid")?;
        let f16_path = dir.0.join("daisy-f16.gguf");
        let f16_gguf = build_gguf(
            16.0,
            Some(16.0),
            32,
            vec![
                TensorBuildSpec {
                    name: "blk.0.ffn_down.weight.lora_a",
                    dims: [4, 2],
                    gguf_type: 1,
                    data: f16_bytes(&[0x3c00; 8]),
                },
                TensorBuildSpec {
                    name: "blk.0.ffn_down.weight.lora_b",
                    dims: [2, 3],
                    gguf_type: 1,
                    data: f16_bytes(&[0x3800; 6]),
                },
            ],
        );
        fs::write(&f16_path, &f16_gguf)?;
        let expected_sha = hex::encode(Sha256::digest(&f16_gguf));

        let verified = verify_multi_rank_lora_gguf(
            &f16_path,
            32,
            64 * 1024 * 1024,
            MultiRankEnvelope::NpuFriendly,
        )?;
        assert_eq!(verified.path(), f16_path);
        assert_eq!(verified.bytes(), u64::try_from(f16_gguf.len())?);
        assert_eq!(verified.sha256(), expected_sha);
        assert_eq!(verified.max_rank(), 2);
        assert_eq!(verified.target_count(), 1);
        assert_eq!(verified.payload_bytes(), 28);
        assert_eq!(verified.envelope(), MultiRankEnvelope::NpuFriendly);
        assert!(verified.file().is_some());

        // Write attempt must fail while the locked guard is alive.
        assert!(fs::OpenOptions::new().write(true).open(&f16_path).is_err());
        drop(verified);

        let f32_path = dir.0.join("daisy-f32.gguf");
        let f32_gguf = build_gguf(
            8.0,
            None,
            32,
            vec![
                TensorBuildSpec {
                    name: "blk.3.attn_q.weight.lora_a",
                    dims: [3, 4],
                    gguf_type: 0,
                    data: f32_bytes(&[0.25; 12]),
                },
                TensorBuildSpec {
                    name: "blk.3.attn_q.weight.lora_b",
                    dims: [4, 5],
                    gguf_type: 0,
                    data: f32_bytes(&[-0.5; 20]),
                },
            ],
        );
        fs::write(&f32_path, &f32_gguf)?;
        let f32_verified = verify_multi_rank_lora_gguf(
            &f32_path,
            16,
            64 * 1024 * 1024,
            MultiRankEnvelope::NpuFriendly,
        )?;
        assert_eq!(f32_verified.max_rank(), 4);
        assert_eq!(f32_verified.payload_bytes(), 128);
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn rejects_policy_rank_nan_unpaired_and_alignment_violations()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::create("rejects")?;

        // Rank > limit
        let p = dir.0.join("rank.gguf");
        fs::write(
            &p,
            build_gguf(
                16.0,
                None,
                32,
                vec![
                    TensorBuildSpec {
                        name: "blk.0.ffn_down.weight.lora_a",
                        dims: [2, 8],
                        gguf_type: 1,
                        data: f16_bytes(&[0x3c00; 16]),
                    },
                    TensorBuildSpec {
                        name: "blk.0.ffn_down.weight.lora_b",
                        dims: [8, 2],
                        gguf_type: 1,
                        data: f16_bytes(&[0x3c00; 16]),
                    },
                ],
            ),
        )?;
        assert_eq!(
            verify_multi_rank_lora_gguf(&p, 4, 64 * 1024 * 1024, MultiRankEnvelope::NpuFriendly)
                .err(),
            Some(ArtifactError::Identity)
        );

        // Forbidden targets under NpuFriendly vs always-forbidden
        for (file, target, env) in [
            (
                "ffn_gate.gguf",
                "blk.0.ffn_gate.weight",
                MultiRankEnvelope::NpuFriendly,
            ),
            (
                "ssm_out.gguf",
                "blk.0.ssm_out.weight",
                MultiRankEnvelope::FullMatrix,
            ),
            (
                "embd.gguf",
                "token_embd.weight",
                MultiRankEnvelope::FullMatrix,
            ),
        ] {
            let path = dir.0.join(file);
            fs::write(
                &path,
                build_gguf(
                    4.0,
                    None,
                    32,
                    vec![
                        TensorBuildSpec {
                            name: &format!("{target}.lora_a"),
                            dims: [2, 2],
                            gguf_type: 1,
                            data: f16_bytes(&[0x3c00; 4]),
                        },
                        TensorBuildSpec {
                            name: &format!("{target}.lora_b"),
                            dims: [2, 2],
                            gguf_type: 1,
                            data: f16_bytes(&[0x3c00; 4]),
                        },
                    ],
                ),
            )?;
            assert_eq!(
                verify_multi_rank_lora_gguf(&path, 16, 64 * 1024 * 1024, env).err(),
                Some(ArtifactError::Identity)
            );
        }

        // Non-finite F16 (Inf = 0x7c00)
        let nan_path = dir.0.join("nan.gguf");
        fs::write(
            &nan_path,
            build_gguf(
                4.0,
                None,
                32,
                vec![
                    TensorBuildSpec {
                        name: "blk.0.ffn_down.weight.lora_a",
                        dims: [2, 2],
                        gguf_type: 1,
                        data: f16_bytes(&[0x3c00, 0x7c00, 0x3c00, 0x3c00]),
                    },
                    TensorBuildSpec {
                        name: "blk.0.ffn_down.weight.lora_b",
                        dims: [2, 2],
                        gguf_type: 1,
                        data: f16_bytes(&[0x3c00; 4]),
                    },
                ],
            ),
        )?;
        assert_eq!(
            verify_multi_rank_lora_gguf(
                &nan_path,
                16,
                64 * 1024 * 1024,
                MultiRankEnvelope::NpuFriendly
            )
            .err(),
            Some(ArtifactError::Identity)
        );

        // Non-power-of-two alignment
        let align_path = dir.0.join("bad-align.gguf");
        fs::write(
            &align_path,
            build_gguf(
                4.0,
                None,
                24,
                vec![
                    TensorBuildSpec {
                        name: "blk.0.ffn_down.weight.lora_a",
                        dims: [2, 2],
                        gguf_type: 1,
                        data: f16_bytes(&[0x3c00; 4]),
                    },
                    TensorBuildSpec {
                        name: "blk.0.ffn_down.weight.lora_b",
                        dims: [2, 2],
                        gguf_type: 1,
                        data: f16_bytes(&[0x3c00; 4]),
                    },
                ],
            ),
        )?;
        assert_eq!(
            verify_multi_rank_lora_gguf(
                &align_path,
                16,
                64 * 1024 * 1024,
                MultiRankEnvelope::NpuFriendly
            )
            .err(),
            Some(ArtifactError::Identity)
        );

        // Mismatched alpha pair
        let alpha_path = dir.0.join("bad-alpha.gguf");
        fs::write(
            &alpha_path,
            build_gguf(
                4.0,
                Some(8.0),
                32,
                vec![
                    TensorBuildSpec {
                        name: "blk.0.ffn_down.weight.lora_a",
                        dims: [2, 2],
                        gguf_type: 1,
                        data: f16_bytes(&[0x3c00; 4]),
                    },
                    TensorBuildSpec {
                        name: "blk.0.ffn_down.weight.lora_b",
                        dims: [2, 2],
                        gguf_type: 1,
                        data: f16_bytes(&[0x3c00; 4]),
                    },
                ],
            ),
        )?;
        assert_eq!(
            verify_multi_rank_lora_gguf(
                &alpha_path,
                16,
                64 * 1024 * 1024,
                MultiRankEnvelope::NpuFriendly
            )
            .err(),
            Some(ArtifactError::Identity)
        );

        // Truncated payload
        let trunc_path = dir.0.join("trunc.gguf");
        let mut trunc = build_gguf(
            4.0,
            None,
            32,
            vec![
                TensorBuildSpec {
                    name: "blk.0.ffn_down.weight.lora_a",
                    dims: [2, 2],
                    gguf_type: 1,
                    data: f16_bytes(&[0x3c00; 4]),
                },
                TensorBuildSpec {
                    name: "blk.0.ffn_down.weight.lora_b",
                    dims: [2, 2],
                    gguf_type: 1,
                    data: f16_bytes(&[0x3c00; 4]),
                },
            ],
        );
        // Truncated payload cutting into the last tensor's actual data bytes
        trunc.truncate(trunc.len().saturating_sub(25));
        fs::write(&trunc_path, &trunc)?;
        assert_eq!(
            verify_multi_rank_lora_gguf(
                &trunc_path,
                16,
                64 * 1024 * 1024,
                MultiRankEnvelope::NpuFriendly
            )
            .err(),
            Some(ArtifactError::Identity)
        );

        // Unpaired A/B targets
        let unpaired_path = dir.0.join("unpaired.gguf");
        fs::write(
            &unpaired_path,
            build_gguf(
                4.0,
                None,
                32,
                vec![
                    TensorBuildSpec {
                        name: "blk.0.ffn_down.weight.lora_a",
                        dims: [2, 2],
                        gguf_type: 1,
                        data: f16_bytes(&[0x3c00; 4]),
                    },
                    TensorBuildSpec {
                        name: "blk.1.ffn_down.weight.lora_b",
                        dims: [2, 2],
                        gguf_type: 1,
                        data: f16_bytes(&[0x3c00; 4]),
                    },
                ],
            ),
        )?;
        assert_eq!(
            verify_multi_rank_lora_gguf(
                &unpaired_path,
                16,
                64 * 1024 * 1024,
                MultiRankEnvelope::NpuFriendly
            )
            .err(),
            Some(ArtifactError::Identity)
        );

        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn mapped_single_pass_verifies_large_f16_and_f32_adapters_and_catches_tail_nan()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::create("mapped-lora")?;
        // >= MIN_MAPPED_HASH_BYTES (4 MiB) exercises the MappedLockedFile single-pass path
        // plus 64-element block scanning and tail handling.
        let f16_elems = 1_100_005_usize;
        let f32_elems = 550_003_usize;
        let f16_data = f16_bytes(&vec![0x3c00_u16; f16_elems]);
        let f32_data = f32_bytes(&vec![0.5_f32; f32_elems]);
        let path = dir.0.join("large-valid.gguf");
        let gguf = build_gguf(
            16.0,
            None,
            32,
            vec![
                TensorBuildSpec {
                    name: "blk.0.ffn_down.weight.lora_a",
                    dims: [u64::try_from(f16_elems)?, 1],
                    gguf_type: 1,
                    data: f16_data,
                },
                TensorBuildSpec {
                    name: "blk.0.ffn_down.weight.lora_b",
                    dims: [1, 4],
                    gguf_type: 1,
                    data: f16_bytes(&[0x3800; 4]),
                },
                TensorBuildSpec {
                    name: "blk.1.attn_q.weight.lora_a",
                    dims: [u64::try_from(f32_elems)?, 1],
                    gguf_type: 0,
                    data: f32_data.clone(),
                },
                TensorBuildSpec {
                    name: "blk.1.attn_q.weight.lora_b",
                    dims: [1, 4],
                    gguf_type: 0,
                    data: f32_bytes(&[0.25; 4]),
                },
            ],
        );
        assert!(u64::try_from(gguf.len())? >= bonsai_windows_credentials::MIN_MAPPED_HASH_BYTES);
        fs::write(&path, &gguf)?;
        let expected_sha = hex::encode(bonsai_windows_credentials::Sha256::digest_bytes(&gguf)?);
        let verified = verify_multi_rank_lora_gguf(
            &path,
            16,
            64 * 1024 * 1024,
            MultiRankEnvelope::NpuFriendly,
        )?;
        assert_eq!(verified.sha256(), expected_sha);
        assert_eq!(verified.max_rank(), 1);
        assert_eq!(verified.target_count(), 2);
        drop(verified);

        // Corrupt the final tail element of the F32 tensor in place with NaN.
        use std::io::{Seek as _, SeekFrom, Write as _};
        let padded_last_b = (4 * 4_usize).next_multiple_of(32);
        let f32_a_pad = f32_data.len().next_multiple_of(32) - f32_data.len();
        let tail_offset = gguf.len() - padded_last_b - f32_a_pad - 4;
        assert_eq!(&gguf[tail_offset..tail_offset + 4], &0.5_f32.to_le_bytes());
        let mut file = fs::OpenOptions::new().write(true).open(&path)?;
        file.seek(SeekFrom::Start(u64::try_from(tail_offset)?))?;
        file.write_all(&f32::NAN.to_le_bytes())?;
        file.flush()?;
        drop(file);
        assert_eq!(
            verify_multi_rank_lora_gguf(
                &path,
                16,
                64 * 1024 * 1024,
                MultiRankEnvelope::NpuFriendly
            )
            .err(),
            Some(ArtifactError::Identity)
        );
        Ok(())
    }
}
