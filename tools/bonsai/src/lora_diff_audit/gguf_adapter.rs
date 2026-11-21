//! Pure-Rust GGUF v3 LoRA adapter writer and structural/numerical verifier.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read as _, Seek as _, SeekFrom, Write as _};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::safetensors::{FloatDtype, decode_scalar_f32, swift_tensor_inventory};
use crate::identity::Hasher;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGUF_ALIGNMENT: usize = 32;
const GGUF_ALIGNMENT_U32: u32 = 32;
const GGUF_TYPE_F32: u32 = 0;
const GGUF_TYPE_F16: u32 = 1;
const MAX_ADAPTER_FILE_BYTES: u64 = 1_073_741_824;
const MAX_GGUF_METADATA_COUNT: usize = 512;
const MAX_GGUF_TENSOR_COUNT: usize = 1_024;
const MAX_GGUF_STRING_BYTES: usize = 4_096;
const MAX_GGUF_ARRAY_LEN: usize = 65_536;
const MAX_LORA_RANK: usize = 64;
const STREAM_CHUNK_BYTES: usize = 262_144;

/// Tensor serialization precision for exported GGUF LoRA adapters.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum ExportDtype {
    #[default]
    F16,
    F32,
}

impl ExportDtype {
    const fn gguf_type(self) -> u32 {
        match self {
            Self::F16 => GGUF_TYPE_F16,
            Self::F32 => GGUF_TYPE_F32,
        }
    }

    const fn byte_width(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Optional verification policy gates for `verify-adapter`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct VerifyPolicy {
    pub(super) expected_sha256: Option<String>,
    pub(super) expected_envelope: Option<String>,
    pub(super) require_npu_friendly: bool,
    pub(super) require_no_grouped_ssm: bool,
    pub(super) max_allowed_rank: Option<usize>,
}

impl VerifyPolicy {
    pub(super) fn is_default(&self) -> bool {
        self.expected_sha256.is_none()
            && self.expected_envelope.is_none()
            && !self.require_npu_friendly
            && !self.require_no_grouped_ssm
            && self.max_allowed_rank.is_none()
    }
}

/// One exported LoRA `(lora_a, lora_b)` matrix pair in canonical GGUF dimensions.
pub(super) struct ExportTensorPair {
    pub(super) gguf_base_name: String,
    pub(super) in_features: usize,
    pub(super) out_features: usize,
    pub(super) rank: usize,
    /// Row-major `[rank x in_features]` (GGUF shape `[in_features, rank]`).
    pub(super) lora_a_f32: Vec<f32>,
    /// Row-major `[out_features x rank]` (GGUF shape `[rank, out_features]`).
    pub(super) lora_b_f32: Vec<f32>,
}

/// Metadata summary returned after atomically publishing a GGUF adapter.
pub(super) struct WrittenAdapter {
    pub(super) bytes: u64,
    pub(super) sha256: String,
    pub(super) tensor_count: usize,
}

#[expect(
    clippy::as_conversions,
    reason = "Scalar integer rank/index values convert directly to f64 for LoRA factor scaling."
)]
#[inline]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[expect(
    clippy::as_conversions,
    reason = "Computed f64 LoRA factor products narrow cleanly to f32 before optional F16 quantization."
)]
#[inline]
const fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

/// Encodes one finite IEEE-754 `f32` to little-endian IEEE-754 binary16 using
/// round-ties-to-even, preserving signed zero and rejecting overflow (`|v| >= 65520`).
pub(super) fn encode_f16_le(value: f32) -> crate::Result<[u8; 2]> {
    if !value.is_finite() {
        return Err("Nonfinite value cannot be encoded as F16.".into());
    }
    if value.abs() >= 65_520.0 {
        return Err(format!(
            "Value {value} overflows IEEE-754 binary16 finite range (< 65520.0)."
        ));
    }
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let mag_bits = bits & 0x7fff_ffff;
    if mag_bits == 0 {
        let half = u16::try_from(sign).map_err(|e| e.to_string())?;
        return Ok(half.to_le_bytes());
    }
    let exp = (bits >> 23) & 0xff;
    let mant = bits & 0x007f_ffff;
    let h_mag = if exp > 112 {
        let h_exp = exp - 112;
        let truncated = mant >> 13;
        let rem = mant & 0x1fff;
        let round_up = rem > 0x1000 || (rem == 0x1000 && (truncated & 1) == 1);
        let combined = ((h_exp << 10) | truncated) + u32::from(round_up);
        if combined >= 0x7c00 {
            return Err("Value rounded to binary16 infinity.".into());
        }
        combined
    } else if exp >= 102 {
        let sig = 0x0080_0000_u32 | mant;
        let shift = 126 - exp;
        let q = sig >> shift;
        let mask = (1_u32 << shift) - 1;
        let rem = sig & mask;
        let half = 1_u32 << (shift - 1);
        let round_up = rem > half || (rem == half && (q & 1) == 1);
        q + u32::from(round_up)
    } else {
        0
    };
    let word = u16::try_from(sign | h_mag).map_err(|e| e.to_string())?;
    Ok(word.to_le_bytes())
}

/// Constructs balanced `(lora_a, lora_b)` F32 factors such that
/// `(global_alpha / rank) * B * A == sum_{k < rank} shrunk_sigmas[k] * u_k * v_k^T`.
pub(super) fn build_balanced_factors(
    u_cols: &[Vec<f32>],
    v_cols: &[Vec<f32>],
    shrunk_sigmas: &[f64],
    global_alpha: f32,
    dtype: ExportDtype,
) -> crate::Result<(Vec<f32>, Vec<f32>)> {
    let rank = shrunk_sigmas.len();
    if rank == 0 || u_cols.len() != rank || v_cols.len() != rank {
        return Err(
            "Balanced LoRA factor construction requires matching non-empty SVD factors.".into(),
        );
    }
    if !global_alpha.is_finite() || global_alpha <= 0.0 {
        return Err("LoRA global alpha must be finite and positive.".into());
    }
    let out_features = u_cols[0].len();
    let in_features = v_cols[0].len();
    if out_features == 0 || in_features == 0 {
        return Err("LoRA tensor dimensions must be positive.".into());
    }
    for (u, v) in u_cols.iter().zip(v_cols) {
        if u.len() != out_features || v.len() != in_features {
            return Err("Inconsistent singular vector dimensions.".into());
        }
    }

    let mut lora_a = Vec::new();
    lora_a
        .try_reserve_exact(
            rank.checked_mul(in_features)
                .ok_or("lora_a size overflow.")?,
        )
        .map_err(|e| format!("Cannot allocate lora_a buffer: {e}"))?;
    lora_a.resize(rank * in_features, 0.0_f32);

    let mut lora_b = Vec::new();
    lora_b
        .try_reserve_exact(
            out_features
                .checked_mul(rank)
                .ok_or("lora_b size overflow.")?,
        )
        .map_err(|e| format!("Cannot allocate lora_b buffer: {e}"))?;
    lora_b.resize(out_features * rank, 0.0_f32);

    let rank_over_alpha = usize_to_f64(rank) / f64::from(global_alpha);
    for k in 0..rank {
        let sigma = shrunk_sigmas[k];
        if !sigma.is_finite() || sigma < 0.0 {
            return Err("Shrunk singular value must be finite and nonnegative.".into());
        }
        let scale_k = (sigma * rank_over_alpha).sqrt();
        let a_row = &mut lora_a[k * in_features..(k + 1) * in_features];
        for (dst, &vc) in a_row.iter_mut().zip(&v_cols[k]) {
            let val = f64_to_f32(scale_k * f64::from(vc));
            if !val.is_finite() || (dtype == ExportDtype::F16 && val.abs() >= 65_520.0) {
                return Err("Balanced lora_a coefficient overflowed target dtype.".into());
            }
            *dst = val;
        }
        for j in 0..out_features {
            let val = f64_to_f32(scale_k * f64::from(u_cols[k][j]));
            if !val.is_finite() || (dtype == ExportDtype::F16 && val.abs() >= 65_520.0) {
                return Err("Balanced lora_b coefficient overflowed target dtype.".into());
            }
            lora_b[j * rank + k] = val;
        }
    }
    Ok((lora_a, lora_b))
}

fn push_gguf_string(buf: &mut Vec<u8>, text: &str) -> crate::Result<()> {
    let len = u64::try_from(text.len()).map_err(|e| e.to_string())?;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(text.as_bytes());
    Ok(())
}

fn push_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) -> crate::Result<()> {
    push_gguf_string(buf, key)?;
    buf.extend_from_slice(&8_u32.to_le_bytes());
    push_gguf_string(buf, value)
}

fn push_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) -> crate::Result<()> {
    push_gguf_string(buf, key)?;
    buf.extend_from_slice(&4_u32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) -> crate::Result<()> {
    if !value.is_finite() {
        return Err(format!("Nonfinite metadata F32 value for '{key}'."));
    }
    push_gguf_string(buf, key)?;
    buf.extend_from_slice(&6_u32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn encode_tensor_bytes(values: &[f32], dtype: ExportDtype) -> crate::Result<Vec<u8>> {
    let width = dtype.byte_width();
    let total_bytes = values
        .len()
        .checked_mul(width)
        .ok_or("Tensor byte count overflow.")?;
    let mut out = Vec::new();
    out.try_reserve_exact(total_bytes)
        .map_err(|e| format!("Cannot allocate tensor serialization buffer: {e}"))?;
    match dtype {
        ExportDtype::F32 => {
            for &v in values {
                if !v.is_finite() {
                    return Err("Nonfinite F32 tensor element during export.".into());
                }
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        ExportDtype::F16 => {
            for &v in values {
                out.extend_from_slice(&encode_f16_le(v)?);
            }
        }
    }
    Ok(out)
}

struct PendingGuard(std::path::PathBuf);

impl Drop for PendingGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Atomically writes a GGUF v3 LoRA adapter with both `adapter.lora.alpha` and
/// `adapter.lora_alpha` set identically for `--publisher-engine` and `prism-inference`.
pub(super) fn write_lora_gguf(
    path: &Path,
    dtype: ExportDtype,
    global_alpha: f32,
    metadata_extra: &[(&str, &str)],
    pairs: &[ExportTensorPair],
) -> crate::Result<WrittenAdapter> {
    bonsai_artifacts::validate_external_path(path).map_err(|e| e.to_string())?;
    if fs::exists(path).map_err(|e| e.to_string())? {
        return Err("Adapter outputPath must name a new file that does not already exist.".into());
    }
    if path.parent().is_none_or(|parent| !parent.is_dir()) {
        return Err("Adapter outputPath parent directory must exist.".into());
    }
    if pairs.is_empty() || pairs.len() * 2 > MAX_GGUF_TENSOR_COUNT {
        return Err("Adapter tensor pair count out of bounds.".into());
    }
    if !global_alpha.is_finite() || global_alpha <= 0.0 {
        return Err("Adapter global alpha must be finite and positive.".into());
    }

    let mut seen_bases = BTreeSet::new();
    let mut tensor_entries: Vec<(String, [u64; 2], Vec<u8>)> = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        if !seen_bases.insert(pair.gguf_base_name.as_str()) {
            return Err(format!(
                "Duplicate adapter target '{}'.",
                pair.gguf_base_name
            ));
        }
        if !(1..=MAX_LORA_RANK).contains(&pair.rank) {
            return Err(format!(
                "Adapter rank {} out of 1..={MAX_LORA_RANK} bounds for '{}'.",
                pair.rank, pair.gguf_base_name
            ));
        }
        if pair.lora_a_f32.len() != pair.rank.saturating_mul(pair.in_features)
            || pair.lora_b_f32.len() != pair.out_features.saturating_mul(pair.rank)
        {
            return Err(format!(
                "Adapter factor shape mismatch for '{}'.",
                pair.gguf_base_name
            ));
        }
        let in_u64 = u64::try_from(pair.in_features).map_err(|e| e.to_string())?;
        let out_u64 = u64::try_from(pair.out_features).map_err(|e| e.to_string())?;
        let r_u64 = u64::try_from(pair.rank).map_err(|e| e.to_string())?;
        let a_bytes = encode_tensor_bytes(&pair.lora_a_f32, dtype)?;
        let b_bytes = encode_tensor_bytes(&pair.lora_b_f32, dtype)?;
        tensor_entries.push((
            format!("{}.lora_a", pair.gguf_base_name),
            [in_u64, r_u64],
            a_bytes,
        ));
        tensor_entries.push((
            format!("{}.lora_b", pair.gguf_base_name),
            [r_u64, out_u64],
            b_bytes,
        ));
    }

    let tensor_count = tensor_entries.len();
    let mut seen_meta = BTreeSet::new();
    let mut metadata_pairs: Vec<(&str, &str)> = Vec::new();
    for &(key, val) in metadata_extra {
        if !seen_meta.insert(key) {
            return Err(format!("Duplicate GGUF metadata key '{key}'."));
        }
        metadata_pairs.push((key, val));
    }
    let metadata_count = 6_usize.saturating_add(metadata_pairs.len());
    if metadata_count > MAX_GGUF_METADATA_COUNT {
        return Err("GGUF metadata count exceeds bound.".into());
    }

    let mut header = Vec::new();
    header.extend_from_slice(GGUF_MAGIC);
    header.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    header.extend_from_slice(
        &u64::try_from(tensor_count)
            .map_err(|e| e.to_string())?
            .to_le_bytes(),
    );
    header.extend_from_slice(
        &u64::try_from(metadata_count)
            .map_err(|e| e.to_string())?
            .to_le_bytes(),
    );

    push_kv_string(&mut header, "general.architecture", "qwen35")?;
    push_kv_string(&mut header, "general.type", "adapter")?;
    push_kv_string(&mut header, "adapter.type", "lora")?;
    push_kv_f32(&mut header, "adapter.lora.alpha", global_alpha)?;
    push_kv_f32(&mut header, "adapter.lora_alpha", global_alpha)?;
    push_kv_u32(&mut header, "general.alignment", GGUF_ALIGNMENT_U32)?;
    for (key, val) in metadata_pairs {
        push_kv_string(&mut header, key, val)?;
    }

    let mut data_offset = 0_usize;
    for (name, dims, payload) in &tensor_entries {
        push_gguf_string(&mut header, name)?;
        header.extend_from_slice(&2_u32.to_le_bytes());
        header.extend_from_slice(&dims[0].to_le_bytes());
        header.extend_from_slice(&dims[1].to_le_bytes());
        header.extend_from_slice(&dtype.gguf_type().to_le_bytes());
        let off_u64 = u64::try_from(data_offset).map_err(|e| e.to_string())?;
        header.extend_from_slice(&off_u64.to_le_bytes());
        let padded_len = payload
            .len()
            .checked_next_multiple_of(GGUF_ALIGNMENT)
            .ok_or("Tensor payload alignment overflow.")?;
        data_offset = data_offset
            .checked_add(padded_len)
            .ok_or("Adapter GGUF total data size overflow.")?;
    }

    let header_padded = header
        .len()
        .checked_next_multiple_of(GGUF_ALIGNMENT)
        .ok_or("GGUF header alignment overflow.")?;
    header.resize(header_padded, 0);

    let file_name = path
        .file_name()
        .ok_or("Adapter outputPath requires a file name.")?
        .to_string_lossy();
    let staging = path.with_file_name(format!(
        "{file_name}.pending-{}",
        crate::startup_process::unique()
    ));
    bonsai_artifacts::validate_external_path(&staging).map_err(|e| e.to_string())?;
    let _cleanup = PendingGuard(staging.clone());

    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        options.share_mode(0).custom_flags(0x0020_0000);
    }
    let mut file = options
        .open(&staging)
        .map_err(|e| format!("Cannot create staged adapter GGUF: {e}"))?;

    let mut hasher = Hasher::new()?;
    let mut written_bytes = 0_u64;
    let write_hashed =
        |f: &mut File, h: &mut Hasher, total: &mut u64, chunk: &[u8]| -> crate::Result<()> {
            f.write_all(chunk).map_err(|e| e.to_string())?;
            h.update(chunk)?;
            *total = total
                .checked_add(u64::try_from(chunk.len()).map_err(|e| e.to_string())?)
                .ok_or("Written GGUF byte count overflow.")?;
            Ok(())
        };

    write_hashed(&mut file, &mut hasher, &mut written_bytes, &header)?;
    let zeros = [0_u8; GGUF_ALIGNMENT];
    for (_, _, payload) in &tensor_entries {
        write_hashed(&mut file, &mut hasher, &mut written_bytes, payload)?;
        let rem = payload.len() % GGUF_ALIGNMENT;
        if rem != 0 {
            let pad = GGUF_ALIGNMENT - rem;
            write_hashed(&mut file, &mut hasher, &mut written_bytes, &zeros[..pad])?;
        }
    }
    file.sync_all().map_err(|e| e.to_string())?;
    drop(file);
    fs::hard_link(&staging, path).map_err(|e| format!("Cannot publish adapter GGUF: {e}"))?;
    let sha256 = hasher.finalize_hex()?;

    Ok(WrittenAdapter {
        bytes: written_bytes,
        sha256,
        tensor_count,
    })
}

struct StreamReader {
    reader: BufReader<File>,
    position: u64,
    limit: u64,
}

impl StreamReader {
    fn new(file: File, limit: u64) -> Self {
        Self {
            reader: BufReader::new(file),
            position: 0,
            limit,
        }
    }

    fn read_exact_bounded(&mut self, len: usize) -> crate::Result<Vec<u8>> {
        let len_u64 = u64::try_from(len).map_err(|e| e.to_string())?;
        let next = self
            .position
            .checked_add(len_u64)
            .filter(|&p| p <= self.limit)
            .ok_or("GGUF header extended past EOF.")?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(len)
            .map_err(|e| format!("Cannot allocate GGUF header buffer: {e}"))?;
        buf.resize(len, 0);
        self.reader
            .read_exact(&mut buf)
            .map_err(|e| e.to_string())?;
        self.position = next;
        Ok(buf)
    }

    fn read_u32(&mut self) -> crate::Result<u32> {
        let b = self.read_exact_bounded(4)?;
        Ok(u32::from_le_bytes(
            b.try_into().map_err(|_| "Invalid GGUF U32.")?,
        ))
    }

    fn read_u64(&mut self) -> crate::Result<u64> {
        let b = self.read_exact_bounded(8)?;
        Ok(u64::from_le_bytes(
            b.try_into().map_err(|_| "Invalid GGUF U64.")?,
        ))
    }

    fn read_f32(&mut self) -> crate::Result<f32> {
        let b = self.read_exact_bounded(4)?;
        Ok(f32::from_le_bytes(
            b.try_into().map_err(|_| "Invalid GGUF F32.")?,
        ))
    }

    fn read_count(&mut self, maximum: usize, what: &str) -> crate::Result<usize> {
        let count = usize::try_from(self.read_u64()?).map_err(|e| e.to_string())?;
        if count > maximum {
            return Err(format!(
                "GGUF {what} count {count} exceeds bound {maximum}."
            ));
        }
        Ok(count)
    }

    fn read_string(&mut self) -> crate::Result<String> {
        let len = self.read_count(MAX_GGUF_STRING_BYTES, "string")?;
        let bytes = self.read_exact_bounded(len)?;
        String::from_utf8(bytes).map_err(|e| format!("Invalid UTF-8 in GGUF string: {e}"))
    }

    fn skip_value(&mut self, kind: u32, depth: u32) -> crate::Result<()> {
        if depth > 4 {
            return Err("GGUF metadata exceeds nesting bound.".into());
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
                let count = self.read_count(MAX_GGUF_ARRAY_LEN, "array")?;
                for _ in 0..count {
                    self.skip_value(elem_kind, depth + 1)?;
                }
                return Ok(());
            }
            _ => return Err(format!("Unsupported GGUF metadata type {kind}.")),
        };
        let _ = self.read_exact_bounded(scalar_bytes)?;
        Ok(())
    }
}

struct ParsedTensorDesc {
    name: String,
    dims: [usize; 2],
    dtype: FloatDtype,
    offset: u64,
    byte_len: u64,
}

pub(super) fn swift_base_dims_map() -> BTreeMap<String, (usize, usize, &'static str)> {
    let mut map = BTreeMap::new();
    for spec in swift_tensor_inventory() {
        let dims = match spec.family {
            "token_embd" | "output" => (5_120, 248_320),
            "ffn_gate" | "ffn_up" => (5_120, 17_408),
            "ffn_down" => (17_408, 5_120),
            "attn_q" => (5_120, 12_288),
            "attn_k" | "attn_v" => (5_120, 1_024),
            "attn_output" | "ssm_out" => (6_144, 5_120),
            "attn_gate" => (5_120, 6_144),
            "attn_qkv" => (5_120, 10_240),
            "ssm_alpha" | "ssm_beta" => (5_120, 48),
            _ => continue,
        };
        map.insert(spec.gguf_name, (dims.0, dims.1, spec.family));
    }
    map
}

#[derive(Default)]
struct FamilyVerifyAccum {
    pairs: usize,
    min_rank: Option<usize>,
    max_rank: usize,
    payload_bytes: u64,
    projected_f16_bytes: u64,
    frobenius_energy: f64,
}

impl FamilyVerifyAccum {
    fn record(&mut self, rank: usize, bytes: u64, f16_bytes: u64, energy: f64) {
        self.pairs += 1;
        self.min_rank = Some(self.min_rank.map_or(rank, |m| m.min(rank)));
        self.max_rank = self.max_rank.max(rank);
        self.payload_bytes += bytes;
        self.projected_f16_bytes += f16_bytes;
        self.frobenius_energy += energy;
    }
}

fn verify_and_sum_tensor(
    file: &mut File,
    abs_offset: u64,
    byte_len: u64,
    dtype: FloatDtype,
) -> crate::Result<f64> {
    file.seek(SeekFrom::Start(abs_offset))
        .map_err(|e| e.to_string())?;
    let width = dtype.byte_width();
    let chunk_cap = STREAM_CHUNK_BYTES - (STREAM_CHUNK_BYTES % width);
    let mut buf = vec![0_u8; chunk_cap.min(usize::try_from(byte_len).unwrap_or(chunk_cap))];
    let mut remaining = byte_len;
    let mut energy = 0.0_f64;
    while remaining > 0 {
        crate::startup_cancel::checkpoint()?;
        let take =
            usize::try_from(remaining.min(u64::try_from(buf.len()).map_err(|e| e.to_string())?))
                .map_err(|e| e.to_string())?;
        file.read_exact(&mut buf[..take])
            .map_err(|e| e.to_string())?;
        for elem in buf[..take].chunks_exact(width) {
            let v = decode_scalar_f32(dtype, elem)?;
            let vf = f64::from(v);
            energy += vf * vf;
        }
        remaining -= u64::try_from(take).map_err(|e| e.to_string())?;
    }
    Ok(energy)
}

#[expect(
    clippy::as_conversions,
    reason = "Byte count converts directly to f64 for MiB reporting."
)]
fn bytes_to_mib(bytes: u64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0)
}

/// Exported or loaded multi-rank GGUF LoRA adapter payload.
pub(super) struct LoadedLoraAdapter {
    pub(super) sha256: String,
    pub(super) alpha: f32,
    pub(super) dtype: ExportDtype,
    pub(super) detected_envelope: String,
    pub(super) provenance_meta: BTreeMap<String, String>,
    pub(super) pairs: Vec<ExportTensorPair>,
}

struct ParsedGgufHeader {
    file: File,
    file_len: u64,
    file_sha256: String,
    architecture: Option<String>,
    general_type: Option<String>,
    adapter_type: Option<String>,
    alpha_dot: Option<f32>,
    alpha_underscore: Option<f32>,
    effective_alpha: f32,
    provenance_meta: BTreeMap<String, String>,
    data_start: u64,
    by_base: BTreeMap<String, (Option<ParsedTensorDesc>, Option<ParsedTensorDesc>)>,
    tensor_count: usize,
}

fn canonical_target_order() -> BTreeMap<String, usize> {
    swift_tensor_inventory()
        .into_iter()
        .enumerate()
        .map(|(idx, spec)| (spec.gguf_name, idx))
        .collect()
}

pub(super) fn sort_pairs_canonically(pairs: &mut [ExportTensorPair]) {
    let order = canonical_target_order();
    pairs.sort_by(|left, right| {
        let li = order
            .get(&left.gguf_base_name)
            .copied()
            .unwrap_or(usize::MAX);
        let ri = order
            .get(&right.gguf_base_name)
            .copied()
            .unwrap_or(usize::MAX);
        li.cmp(&ri)
            .then_with(|| left.gguf_base_name.cmp(&right.gguf_base_name))
    });
}

fn validate_pair_dims(
    base_name: &str,
    a: &ParsedTensorDesc,
    b: &ParsedTensorDesc,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<(usize, usize, usize, &'static str)> {
    let &(in_dim, out_dim, family) = base_dims
        .get(base_name)
        .ok_or_else(|| format!("Unknown Swift-Bonsai-2 adapter target '{base_name}'."))?;
    let rank = if base_name == "token_embd.weight" {
        let r = a.dims[0];
        if a.dims[1] != out_dim || b.dims != [r, in_dim] {
            return Err(format!(
                "Embedding LoRA shape mismatch on '{base_name}': A={:?}, B={:?}.",
                a.dims, b.dims
            ));
        }
        r
    } else {
        let r = a.dims[1];
        if a.dims[0] != in_dim || b.dims != [r, out_dim] {
            return Err(format!(
                "Linear LoRA shape mismatch on '{base_name}': expected A=[{in_dim}, r] B=[r, {out_dim}], got A={:?}, B={:?}.",
                a.dims, b.dims
            ));
        }
        r
    };
    if !(1..=MAX_LORA_RANK).contains(&rank) {
        return Err(format!(
            "Rank {rank} out of 1..={MAX_LORA_RANK} on '{base_name}'."
        ));
    }
    Ok((in_dim, out_dim, rank, family))
}

pub(super) fn classify_families(
    families_used: &BTreeSet<&'static str>,
) -> (&'static str, bool, bool) {
    let excludes_grouped_ssm_out = !families_used.contains("ssm_out");
    let preserves_hexagon_ffn_chains =
        !families_used.contains("ffn_gate") && !families_used.contains("ffn_up");
    let has_only_residual = families_used
        .iter()
        .all(|&f| matches!(f, "ffn_down" | "attn_output"));
    let has_only_npu_friendly = families_used.iter().all(|&f| {
        matches!(
            f,
            "ffn_down" | "attn_output" | "attn_q" | "attn_k" | "attn_v"
        )
    });
    let has_only_full_linear = families_used.iter().all(|&f| {
        matches!(
            f,
            "ffn_gate" | "ffn_up" | "ffn_down" | "attn_output" | "attn_q" | "attn_k" | "attn_v"
        )
    });
    let detected_envelope = if has_only_residual {
        "residual-writers"
    } else if has_only_npu_friendly {
        "npu-friendly"
    } else if has_only_full_linear {
        "full-linear"
    } else {
        "custom"
    };
    (
        detected_envelope,
        preserves_hexagon_ffn_chains,
        excludes_grouped_ssm_out,
    )
}

#[expect(
    clippy::too_many_lines,
    reason = "Single-pass bounded GGUF parser validating magic, metadata, alignment, non-overlapping extents, and A/B pairing."
)]
fn parse_gguf_header(
    path: &Path,
    expected_sha256: Option<&str>,
) -> crate::Result<ParsedGgufHeader> {
    bonsai_artifacts::validate_external_path(path).map_err(|e| e.to_string())?;
    if let Some(expected) = expected_sha256
        && !crate::identity::is_lower_hex(expected, 64)
    {
        return Err(
            "verifyPolicy.expectedSha256 must be a 64-character lowercase hex digest.".into(),
        );
    }

    let mut file = bonsai_artifacts::open_locked_file(path).map_err(|e| e.to_string())?;
    let file_len = file.metadata().map_err(|e| e.to_string())?.len();
    if file_len == 0 || file_len > MAX_ADAPTER_FILE_BYTES {
        return Err(format!(
            "Adapter GGUF length {file_len} outside 1..={MAX_ADAPTER_FILE_BYTES} bytes."
        ));
    }
    let (_, file_sha256) = crate::identity::hash_reader_checked(
        &mut file,
        Some(file_len),
        expected_sha256,
        &mut crate::startup_cancel::checkpoint,
    )?;
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;

    let mut reader = StreamReader::new(file, file_len);
    let magic = reader.read_exact_bounded(4)?;
    if magic.as_slice() != GGUF_MAGIC {
        return Err("File is not a GGUF archive.".into());
    }
    if reader.read_u32()? != GGUF_VERSION {
        return Err("Adapter verification requires GGUF version 3.".into());
    }
    let tensor_count = reader.read_count(MAX_GGUF_TENSOR_COUNT, "tensor")?;
    if tensor_count == 0 || !tensor_count.is_multiple_of(2) {
        return Err("Adapter GGUF must contain an even positive number of tensors.".into());
    }
    let metadata_count = reader.read_count(MAX_GGUF_METADATA_COUNT, "metadata")?;

    let mut meta_keys = BTreeSet::new();
    let mut architecture: Option<String> = None;
    let mut general_type: Option<String> = None;
    let mut adapter_type: Option<String> = None;
    let mut alpha_dot: Option<f32> = None;
    let mut alpha_underscore: Option<f32> = None;
    let mut alignment: usize = GGUF_ALIGNMENT;
    let mut provenance_meta = BTreeMap::new();

    for _ in 0..metadata_count {
        let key = reader.read_string()?;
        if !meta_keys.insert(key.clone()) {
            return Err(format!("Duplicate GGUF metadata key '{key}'."));
        }
        let kind = reader.read_u32()?;
        match (key.as_str(), kind) {
            ("general.architecture", 8) => architecture = Some(reader.read_string()?),
            ("general.type", 8) => general_type = Some(reader.read_string()?),
            ("adapter.type", 8) => adapter_type = Some(reader.read_string()?),
            ("adapter.lora.alpha", 6) => alpha_dot = Some(reader.read_f32()?),
            ("adapter.lora_alpha", 6) => alpha_underscore = Some(reader.read_f32()?),
            ("general.alignment", 4) => {
                let align = usize::try_from(reader.read_u32()?).map_err(|e| e.to_string())?;
                if align != GGUF_ALIGNMENT {
                    return Err(format!(
                        "Adapter alignment must be {GGUF_ALIGNMENT}, got {align}."
                    ));
                }
                alignment = align;
            }
            (k, 8) if k.starts_with("bonsai.") => {
                provenance_meta.insert(k.to_owned(), reader.read_string()?);
            }
            _ => reader.skip_value(kind, 0)?,
        }
    }

    if architecture.as_deref() != Some("qwen35")
        || general_type.as_deref() != Some("adapter")
        || adapter_type.as_deref() != Some("lora")
    {
        return Err(
            "Adapter metadata must declare qwen35 / general.type=adapter / adapter.type=lora."
                .into(),
        );
    }
    let effective_alpha = match (alpha_dot, alpha_underscore) {
        (Some(a), Some(b)) => {
            if !a.is_finite() || a <= 0.0 || a.to_bits() != b.to_bits() {
                return Err("Dual alpha keys adapter.lora.alpha and adapter.lora_alpha disagree or are invalid.".into());
            }
            a
        }
        (Some(a), None) | (None, Some(a)) => {
            if !a.is_finite() || a <= 0.0 {
                return Err("Adapter alpha must be finite and positive.".into());
            }
            a
        }
        (None, None) => return Err("Adapter GGUF is missing alpha metadata.".into()),
    };

    let mut tensor_names = BTreeSet::new();
    let mut parsed_tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        if !tensor_names.insert(name.clone()) {
            return Err(format!("Duplicate GGUF tensor name '{name}'."));
        }
        let rank_dims = reader.read_u32()?;
        if rank_dims != 2 {
            return Err(format!(
                "Adapter tensor '{name}' must be 2D, got rank {rank_dims}."
            ));
        }
        let d0 = reader.read_count(248_320, "dimension")?;
        let d1 = reader.read_count(248_320, "dimension")?;
        if d0 == 0 || d1 == 0 {
            return Err(format!("Adapter tensor '{name}' has zero dimension."));
        }
        let gguf_dtype = reader.read_u32()?;
        let dtype = match gguf_dtype {
            GGUF_TYPE_F32 => FloatDtype::F32,
            GGUF_TYPE_F16 => FloatDtype::F16,
            _ => {
                return Err(format!(
                    "Adapter tensor '{name}' uses unsupported dtype {gguf_dtype}; expected F16 or F32."
                ));
            }
        };
        let offset = reader.read_u64()?;
        let align_u64 = u64::try_from(alignment).map_err(|e| e.to_string())?;
        if !offset.is_multiple_of(align_u64) {
            return Err(format!("Tensor '{name}' offset {offset} is misaligned."));
        }
        let elem_u64 = u64::try_from(d0)
            .map_err(|e| e.to_string())?
            .checked_mul(u64::try_from(d1).map_err(|e| e.to_string())?)
            .ok_or("Tensor element count overflow.")?;
        let byte_len = elem_u64
            .checked_mul(u64::try_from(dtype.byte_width()).map_err(|e| e.to_string())?)
            .ok_or("Tensor byte extent overflow.")?;
        parsed_tensors.push(ParsedTensorDesc {
            name,
            dims: [d0, d1],
            dtype,
            offset,
            byte_len,
        });
    }

    let align_u64 = u64::try_from(alignment).map_err(|e| e.to_string())?;
    let data_start = reader
        .position
        .checked_next_multiple_of(align_u64)
        .filter(|&d| d <= file_len)
        .ok_or("GGUF header alignment extended past EOF.")?;

    let mut extents: Vec<(u64, u64)> = Vec::with_capacity(parsed_tensors.len());
    for t in &parsed_tensors {
        let abs_begin = data_start
            .checked_add(t.offset)
            .ok_or("Tensor data offset overflow.")?;
        let abs_end = abs_begin
            .checked_add(t.byte_len)
            .filter(|&end| end <= file_len)
            .ok_or_else(|| format!("Tensor '{}' payload extends past EOF.", t.name))?;
        extents.push((abs_begin, abs_end));
    }
    extents.sort_unstable();
    for pair in extents.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err("Adapter GGUF tensor payloads overlap.".into());
        }
    }

    let mut by_base: BTreeMap<String, (Option<ParsedTensorDesc>, Option<ParsedTensorDesc>)> =
        BTreeMap::new();
    for t in parsed_tensors {
        if let Some(base) = t.name.strip_suffix(".lora_a") {
            let entry = by_base.entry(base.to_owned()).or_insert((None, None));
            entry.0 = Some(t);
        } else if let Some(base) = t.name.strip_suffix(".lora_b") {
            let entry = by_base.entry(base.to_owned()).or_insert((None, None));
            entry.1 = Some(t);
        } else {
            return Err(format!(
                "Unsupported adapter tensor suffix on '{}'.",
                t.name
            ));
        }
    }

    Ok(ParsedGgufHeader {
        file: reader.reader.into_inner(),
        file_len,
        file_sha256,
        architecture,
        general_type,
        adapter_type,
        alpha_dot,
        alpha_underscore,
        effective_alpha,
        provenance_meta,
        data_start,
        by_base,
        tensor_count,
    })
}

fn read_tensor_f32(
    file: &mut File,
    abs_offset: u64,
    byte_len: u64,
    elem_count: usize,
    dtype: FloatDtype,
) -> crate::Result<Vec<f32>> {
    let width = dtype.byte_width();
    let expected_bytes = u64::try_from(elem_count)
        .map_err(|e| e.to_string())?
        .checked_mul(u64::try_from(width).map_err(|e| e.to_string())?)
        .ok_or("Tensor byte length overflow.")?;
    if expected_bytes != byte_len {
        return Err("GGUF tensor byte extent mismatches element count.".into());
    }
    file.seek(SeekFrom::Start(abs_offset))
        .map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(usize::try_from(byte_len).map_err(|e| e.to_string())?)
        .map_err(|e| format!("Cannot allocate GGUF tensor read buffer: {e}"))?;
    raw.resize(usize::try_from(byte_len).map_err(|e| e.to_string())?, 0);
    file.read_exact(&mut raw).map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    out.try_reserve_exact(elem_count)
        .map_err(|e| format!("Cannot allocate F32 LoRA factor buffer: {e}"))?;
    for elem in raw.chunks_exact(width) {
        crate::startup_cancel::checkpoint()?;
        out.push(decode_scalar_f32(dtype, elem)?);
    }
    Ok(out)
}

/// Loads all linear LoRA `(lora_a, lora_b)` pairs from a GGUF v3 adapter file.
#[cfg(test)]
pub(super) fn read_lora_gguf_pairs(path: &Path) -> crate::Result<LoadedLoraAdapter> {
    read_lora_gguf_pairs_with_dims(path, &swift_base_dims_map())
}

pub(super) fn read_lora_gguf_pairs_with_dims(
    path: &Path,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<LoadedLoraAdapter> {
    let mut parsed = parse_gguf_header(path, None)?;
    if parsed.alpha_dot.is_none() || parsed.alpha_underscore.is_none() {
        return Err(
            "Adapter GGUF must define matching adapter.lora.alpha and adapter.lora_alpha metadata."
                .into(),
        );
    }
    let mut families_used = BTreeSet::new();
    let mut uniform_dtype: Option<FloatDtype> = None;
    let mut pairs = Vec::with_capacity(parsed.by_base.len());

    for (base_name, (maybe_a, maybe_b)) in &parsed.by_base {
        if base_name == "token_embd.weight" {
            return Err(
                "Multi-rank merge-adapters/edit-adapter does not admit token_embd.weight.".into(),
            );
        }
        let (Some(a), Some(b)) = (maybe_a, maybe_b) else {
            return Err(format!("Incomplete LoRA A/B pair for '{base_name}'."));
        };
        if a.dtype != b.dtype {
            return Err(format!("Mismatched factor dtypes on '{base_name}'."));
        }
        match uniform_dtype {
            None => uniform_dtype = Some(a.dtype),
            Some(prev) if prev == a.dtype => {}
            Some(_) => {
                return Err(
                    "Adapter GGUF mixes tensor dtypes; expected uniform F16 or F32.".into(),
                );
            }
        }
        let (in_features, out_features, rank, family) =
            validate_pair_dims(base_name, a, b, base_dims)?;
        families_used.insert(family);

        let a_elems = rank
            .checked_mul(in_features)
            .ok_or("lora_a element count overflow.")?;
        let b_elems = out_features
            .checked_mul(rank)
            .ok_or("lora_b element count overflow.")?;
        let lora_a_f32 = read_tensor_f32(
            &mut parsed.file,
            parsed.data_start + a.offset,
            a.byte_len,
            a_elems,
            a.dtype,
        )?;
        let lora_b_f32 = read_tensor_f32(
            &mut parsed.file,
            parsed.data_start + b.offset,
            b.byte_len,
            b_elems,
            b.dtype,
        )?;
        pairs.push(ExportTensorPair {
            gguf_base_name: base_name.clone(),
            in_features,
            out_features,
            rank,
            lora_a_f32,
            lora_b_f32,
        });
    }

    let dtype = match uniform_dtype {
        Some(FloatDtype::F16) => ExportDtype::F16,
        Some(FloatDtype::F32) => ExportDtype::F32,
        _ => return Err("Adapter GGUF must use uniform F16 or F32 factor dtype.".into()),
    };
    sort_pairs_canonically(&mut pairs);
    let (detected_envelope, _, _) = classify_families(&families_used);
    Ok(LoadedLoraAdapter {
        sha256: parsed.file_sha256,
        alpha: parsed.effective_alpha,
        dtype,
        detected_envelope: detected_envelope.to_owned(),
        provenance_meta: parsed.provenance_meta,
        pairs,
    })
}

/// Reads unit-normalized output (`lora_b`) directions from a rank-1 reference GGUF adapter.
pub(super) fn read_rank_one_b_directions(path: &Path) -> crate::Result<BTreeMap<String, Vec<f64>>> {
    let base_dims = swift_base_dims_map();
    let mut parsed = parse_gguf_header(path, None)?;
    let mut directions = BTreeMap::new();
    for (base_name, (maybe_a, maybe_b)) in &parsed.by_base {
        if base_name == "token_embd.weight" {
            continue;
        }
        let (Some(a), Some(b)) = (maybe_a, maybe_b) else {
            return Err(format!("Incomplete rank-1 LoRA pair for '{base_name}'."));
        };
        let (_in_dim, out_dim, rank, _family) = validate_pair_dims(base_name, a, b, &base_dims)?;
        if rank != 1 {
            return Err(format!(
                "orthogonalizeAgainstRankOnePath requires rank-1 factors, got rank {rank} on '{base_name}'."
            ));
        }
        let b_vals = read_tensor_f32(
            &mut parsed.file,
            parsed.data_start + b.offset,
            b.byte_len,
            out_dim,
            b.dtype,
        )?;
        let b_f64: Vec<f64> = b_vals.into_iter().map(f64::from).collect();
        let norm_sq: f64 = b_f64.iter().map(|&v| v * v).sum();
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(format!(
                "Rank-1 reference direction '{base_name}' has zero or nonfinite norm."
            ));
        }
        let unit: Vec<f64> = b_f64.into_iter().map(|v| v / norm).collect();
        directions.insert(base_name.clone(), unit);
    }
    Ok(directions)
}

/// Verifies a GGUF v3 LoRA adapter file against `Swift-Bonsai-2` shapes, finite scalars,
/// dual alpha metadata compatibility, and optional policy gates without leaking local paths.
pub(super) fn verify_lora_gguf(path: &Path, policy: &VerifyPolicy) -> crate::Result<Value> {
    verify_lora_gguf_with_dims(path, policy, &swift_base_dims_map())
}

#[expect(
    clippy::too_many_lines,
    reason = "Single-pass bounded verifier checking Swift dimensions, finiteness, and NPU/SSM policy gates."
)]
pub(super) fn verify_lora_gguf_with_dims(
    path: &Path,
    policy: &VerifyPolicy,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<Value> {
    if let Some(max_r) = policy.max_allowed_rank
        && !(1..=MAX_LORA_RANK).contains(&max_r)
    {
        return Err(format!(
            "verifyPolicy.maxAllowedRank must be in 1..={MAX_LORA_RANK}."
        ));
    }

    let mut parsed = parse_gguf_header(path, policy.expected_sha256.as_deref())?;
    let mut families_used = BTreeSet::new();
    let mut dtypes_used = BTreeSet::new();
    let mut by_family: BTreeMap<&'static str, FamilyVerifyAccum> = BTreeMap::new();
    let mut min_rank = usize::MAX;
    let mut max_rank = 0_usize;
    let mut total_payload_bytes = 0_u64;
    let mut projected_f16_payload_bytes = 0_u64;

    for (base_name, (maybe_a, maybe_b)) in &parsed.by_base {
        let (Some(a), Some(b)) = (maybe_a, maybe_b) else {
            return Err(format!("Incomplete LoRA A/B pair for '{base_name}'."));
        };
        let (_in_dim, _out_dim, rank, family) = validate_pair_dims(base_name, a, b, base_dims)?;

        let e_a = verify_and_sum_tensor(
            &mut parsed.file,
            parsed.data_start + a.offset,
            a.byte_len,
            a.dtype,
        )?;
        let e_b = verify_and_sum_tensor(
            &mut parsed.file,
            parsed.data_start + b.offset,
            b.byte_len,
            b.dtype,
        )?;
        let pair_bytes = a.byte_len + b.byte_len;
        let f16_elems = (u64::try_from(a.dims[0]).map_err(|e| e.to_string())?
            * u64::try_from(a.dims[1]).map_err(|e| e.to_string())?)
            + (u64::try_from(b.dims[0]).map_err(|e| e.to_string())?
                * u64::try_from(b.dims[1]).map_err(|e| e.to_string())?);
        let f16_bytes = f16_elems * 2;

        families_used.insert(family);
        dtypes_used.insert(a.dtype.as_str());
        dtypes_used.insert(b.dtype.as_str());
        min_rank = min_rank.min(rank);
        max_rank = max_rank.max(rank);
        total_payload_bytes += pair_bytes;
        projected_f16_payload_bytes += f16_bytes;
        by_family
            .entry(family)
            .or_default()
            .record(rank, pair_bytes, f16_bytes, e_a + e_b);
    }

    let (detected_envelope, preserves_hexagon_ffn_chains, excludes_grouped_ssm_out) =
        classify_families(&families_used);

    let mut policy_failures = Vec::new();
    if policy.require_no_grouped_ssm && !excludes_grouped_ssm_out {
        policy_failures.push(
            "Adapter targets grouped ssm_out.weight while P1 SSM input order remains unresolved."
                .to_owned(),
        );
    }
    if policy.require_npu_friendly && !preserves_hexagon_ffn_chains {
        policy_failures.push(
            "Adapter targets ffn_gate/ffn_up, breaking Hexagon NPU fused FFN chains.".to_owned(),
        );
    }
    if let Some(max_allowed) = policy.max_allowed_rank
        && max_rank > max_allowed
    {
        policy_failures.push(format!(
            "Adapter max rank {max_rank} exceeds verifyPolicy.maxAllowedRank {max_allowed}."
        ));
    }
    if let Some(expected_env) = &policy.expected_envelope
        && expected_env != detected_envelope
    {
        policy_failures.push(format!(
            "Detected target envelope '{detected_envelope}' differs from expected '{expected_env}'."
        ));
    }
    let verified = policy_failures.is_empty();

    let mut family_json = serde_json::Map::new();
    for (family, acc) in by_family {
        family_json.insert(
            family.to_owned(),
            json!({
                "pairCount": acc.pairs,
                "minRank": acc.min_rank.unwrap_or(0),
                "maxRank": acc.max_rank,
                "payloadBytes": acc.payload_bytes,
                "projectedF16Bytes": acc.projected_f16_bytes,
                "factorFrobeniusEnergy": acc.frobenius_energy,
            }),
        );
    }
    let dtype_list: Vec<&str> = dtypes_used.into_iter().collect();

    Ok(json!({
        "schemaVersion": 1,
        "operation": "verify-adapter",
        "verified": verified,
        "status": if verified { "verified" } else { "policy-rejected" },
        "bytes": parsed.file_len,
        "sha256": parsed.file_sha256,
        "architecture": parsed.architecture,
        "generalType": parsed.general_type,
        "adapterType": parsed.adapter_type,
        "alpha": parsed.effective_alpha,
        "adapterLoraDotAlpha": parsed.alpha_dot,
        "adapterLoraUnderscoreAlpha": parsed.alpha_underscore,
        "publisherEngineAlphaCompatible": parsed.alpha_dot.is_some(),
        "prismInferenceAlphaCompatible": parsed.alpha_underscore.is_some(),
        "tensorCount": parsed.tensor_count,
        "pairCount": parsed.by_base.len(),
        "dtypes": dtype_list,
        "minRank": min_rank,
        "maxRank": max_rank,
        "uniformRank": min_rank == max_rank,
        "detectedEnvelope": detected_envelope,
        "preservesHexagonFfnChains": preserves_hexagon_ffn_chains,
        "excludesGroupedSsmOut": excludes_grouped_ssm_out,
        "totalPayloadBytes": total_payload_bytes,
        "projectedF16PayloadBytes": projected_f16_payload_bytes,
        "projectedF16PayloadMiB": bytes_to_mib(projected_f16_payload_bytes),
        "byFamily": family_json,
        "provenanceMetadata": parsed.provenance_meta,
        "policyFailures": policy_failures,
    }))
}
