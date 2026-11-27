//! Strict safetensors index/header parsing, HF-to-GGUF mapping, and bounded range I/O.

use std::collections::BTreeMap;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use serde::Deserialize;
use serde_json::Value;

const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_RANGE_CHUNK_BYTES: u64 = 32 * 1024 * 1024;

/// High-level parameter role used for Gate 1 admissibility classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum TensorCategory {
    NonMatrix,
    GdnSmallMatrix,
    LinearMatrix,
    VocabMatrix,
}

impl TensorCategory {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::NonMatrix => "non-matrix",
            Self::GdnSmallMatrix => "gdn-small-matrix",
            Self::LinearMatrix => "linear-matrix",
            Self::VocabMatrix => "vocab-matrix",
        }
    }
}

/// Canonical tensor pair metadata mapping HuggingFace safetensors keys to Swift GGUF names.
#[derive(Clone, Debug)]
pub(super) struct TensorSpec {
    pub(super) hf_name: String,
    pub(super) gguf_name: String,
    pub(super) layer: Option<u32>,
    pub(super) family: &'static str,
    pub(super) category: TensorCategory,
}

/// Generates the exact 851-tensor language-model mapping for Qwen3.8 / Swift-Bonsai-2 (`qwen35`).
pub(super) fn swift_tensor_inventory() -> Vec<TensorSpec> {
    let mut specs = Vec::with_capacity(851);
    specs.push(TensorSpec {
        hf_name: "model.language_model.embed_tokens.weight".to_owned(),
        gguf_name: "token_embd.weight".to_owned(),
        layer: None,
        family: "token_embd",
        category: TensorCategory::VocabMatrix,
    });
    specs.push(TensorSpec {
        hf_name: "lm_head.weight".to_owned(),
        gguf_name: "output.weight".to_owned(),
        layer: None,
        family: "output",
        category: TensorCategory::VocabMatrix,
    });
    specs.push(TensorSpec {
        hf_name: "model.language_model.norm.weight".to_owned(),
        gguf_name: "output_norm.weight".to_owned(),
        layer: None,
        family: "norm",
        category: TensorCategory::NonMatrix,
    });

    for layer in 0_u32..64 {
        let mut push = |hf_suffix: &str, gguf_suffix: &str, family: &'static str, category| {
            specs.push(TensorSpec {
                hf_name: format!("model.language_model.layers.{layer}.{hf_suffix}"),
                gguf_name: format!("blk.{layer}.{gguf_suffix}"),
                layer: Some(layer),
                family,
                category,
            });
        };
        push(
            "input_layernorm.weight",
            "attn_norm.weight",
            "norm",
            TensorCategory::NonMatrix,
        );
        push(
            "post_attention_layernorm.weight",
            "post_attention_norm.weight",
            "norm",
            TensorCategory::NonMatrix,
        );
        push(
            "mlp.gate_proj.weight",
            "ffn_gate.weight",
            "ffn_gate",
            TensorCategory::LinearMatrix,
        );
        push(
            "mlp.up_proj.weight",
            "ffn_up.weight",
            "ffn_up",
            TensorCategory::LinearMatrix,
        );
        push(
            "mlp.down_proj.weight",
            "ffn_down.weight",
            "ffn_down",
            TensorCategory::LinearMatrix,
        );

        if (layer + 1).is_multiple_of(4) {
            push(
                "self_attn.q_proj.weight",
                "attn_q.weight",
                "attn_q",
                TensorCategory::LinearMatrix,
            );
            push(
                "self_attn.k_proj.weight",
                "attn_k.weight",
                "attn_k",
                TensorCategory::LinearMatrix,
            );
            push(
                "self_attn.v_proj.weight",
                "attn_v.weight",
                "attn_v",
                TensorCategory::LinearMatrix,
            );
            push(
                "self_attn.o_proj.weight",
                "attn_output.weight",
                "attn_output",
                TensorCategory::LinearMatrix,
            );
            push(
                "self_attn.q_norm.weight",
                "attn_q_norm.weight",
                "norm",
                TensorCategory::NonMatrix,
            );
            push(
                "self_attn.k_norm.weight",
                "attn_k_norm.weight",
                "norm",
                TensorCategory::NonMatrix,
            );
        } else {
            push(
                "linear_attn.in_proj_qkv.weight",
                "attn_qkv.weight",
                "attn_qkv",
                TensorCategory::LinearMatrix,
            );
            push(
                "linear_attn.in_proj_z.weight",
                "attn_gate.weight",
                "attn_gate",
                TensorCategory::LinearMatrix,
            );
            push(
                "linear_attn.out_proj.weight",
                "ssm_out.weight",
                "ssm_out",
                TensorCategory::LinearMatrix,
            );
            push(
                "linear_attn.in_proj_a.weight",
                "ssm_alpha.weight",
                "ssm_alpha",
                TensorCategory::GdnSmallMatrix,
            );
            push(
                "linear_attn.in_proj_b.weight",
                "ssm_beta.weight",
                "ssm_beta",
                TensorCategory::GdnSmallMatrix,
            );
            push(
                "linear_attn.conv1d.weight",
                "ssm_conv1d.weight",
                "ssm_conv1d",
                TensorCategory::NonMatrix,
            );
            push(
                "linear_attn.norm.weight",
                "ssm_norm.weight",
                "ssm_norm",
                TensorCategory::NonMatrix,
            );
            push(
                "linear_attn.A_log",
                "ssm_a",
                "ssm_a",
                TensorCategory::NonMatrix,
            );
            push(
                "linear_attn.dt_bias",
                "ssm_dt.bias",
                "ssm_dt_bias",
                TensorCategory::NonMatrix,
            );
        }
    }
    specs
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FloatDtype {
    Bf16,
    F16,
    F32,
}

impl FloatDtype {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }

    pub(super) const fn byte_width(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Decodes one IEEE-754 binary16 scalar into `f32`.
#[inline]
fn decode_f16(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 31;
    let mantissa = u32::from(bits & 1023);
    match exponent {
        0 => {
            if mantissa == 0 {
                f32::from_bits(sign)
            } else {
                let value = f32::from(bits & 1023) * (1.0 / 16_777_216.0);
                if sign == 0 { value } else { -value }
            }
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | (u32::from(exponent + 112) << 23) | (mantissa << 13)),
    }
}

/// Decodes one scalar element at `bytes[..dtype.byte_width()]` and validates finiteness.
#[inline]
pub(super) fn decode_scalar_f32(dtype: FloatDtype, bytes: &[u8]) -> crate::Result<f32> {
    let value = match dtype {
        FloatDtype::Bf16 => {
            f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
        }
        FloatDtype::F16 => decode_f16(u16::from_le_bytes([bytes[0], bytes[1]])),
        FloatDtype::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    };
    if !value.is_finite() {
        return Err("Encountered nonfinite tensor weight.".into());
    }
    Ok(value)
}

#[derive(Clone, Debug)]
pub(super) struct TensorDescriptor {
    pub(super) dtype: String,
    pub(super) shape: Vec<usize>,
    pub(super) abs_begin: u64,
    pub(super) abs_end: u64,
}

impl TensorDescriptor {
    pub(super) fn float_dtype(&self) -> crate::Result<FloatDtype> {
        match self.dtype.as_str() {
            "BF16" => Ok(FloatDtype::Bf16),
            "F16" => Ok(FloatDtype::F16),
            "F32" => Ok(FloatDtype::F32),
            other => Err(format!(
                "Audited tensor uses unsupported dtype '{other}'; expected BF16, F16, or F32."
            )),
        }
    }

    pub(super) fn element_count(&self) -> crate::Result<usize> {
        self.shape
            .iter()
            .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| "Tensor element count overflow.".into())
    }
}

#[derive(Debug)]
pub(super) struct ShardHeader {
    pub(super) tensors: BTreeMap<String, TensorDescriptor>,
}

#[derive(Debug)]
pub(super) struct SafetensorsIndex {
    pub(super) sha256: String,
    pub(super) weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct RawIndex {
    weight_map: BTreeMap<String, String>,
}

fn validate_plain_shard_name(name: &str) -> crate::Result<()> {
    if name.is_empty()
        || name.len() > 256
        || !name.ends_with(".safetensors")
        || name.contains(['/', '\\', '\0'])
        || name == "."
        || name == ".."
    {
        return Err(format!("Unsafe safetensors shard reference: '{name}'."));
    }
    Ok(())
}

fn parse_index(bytes: &[u8]) -> crate::Result<SafetensorsIndex> {
    let sha256 = crate::identity::sha256(bytes);
    let parsed: RawIndex =
        serde_json::from_slice(bytes).map_err(|e| format!("Invalid safetensors index: {e}"))?;
    if parsed.weight_map.is_empty() || parsed.weight_map.len() > 16_384 {
        return Err("Safetensors index weight_map count out of bounds.".into());
    }
    for shard in parsed.weight_map.values() {
        validate_plain_shard_name(shard)?;
    }
    Ok(SafetensorsIndex {
        sha256,
        weight_map: parsed.weight_map,
    })
}

fn dtype_width(dtype: &str) -> crate::Result<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" => Ok(1),
        "U16" | "I16" | "F16" | "BF16" => Ok(2),
        "U32" | "I32" | "F32" => Ok(4),
        "U64" | "I64" | "F64" => Ok(8),
        other => Err(format!("Unsupported safetensors dtype '{other}'.")),
    }
}

pub(super) fn parse_shard_header(raw_json: &[u8], file_bytes: u64) -> crate::Result<ShardHeader> {
    let header_len = u64::try_from(raw_json.len()).map_err(|e| e.to_string())?;
    let base = header_len
        .checked_add(8)
        .filter(|&b| b <= file_bytes)
        .ok_or("Safetensors header extends past shard EOF.")?;
    let root: Value = serde_json::from_slice(raw_json)
        .map_err(|e| format!("Invalid safetensors header JSON: {e}"))?;
    let obj = root
        .as_object()
        .ok_or("Safetensors header root must be a JSON object.")?;
    if obj.len() > 8193 {
        return Err("Safetensors header tensor count exceeds bound.".into());
    }
    let mut tensors = BTreeMap::new();
    for (name, record) in obj {
        if name == "__metadata__" {
            continue;
        }
        if name.is_empty() || name.len() > 512 || name.contains('\0') {
            return Err("Invalid safetensors tensor name.".into());
        }
        let rec = record
            .as_object()
            .ok_or("Safetensors tensor descriptor must be an object.")?;
        let dtype = rec
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or("Safetensors descriptor missing dtype.")?;
        let width = dtype_width(dtype)?;
        let shape_arr = rec
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("Safetensors descriptor missing shape.")?;
        if shape_arr.is_empty() || shape_arr.len() > 8 {
            return Err("Safetensors tensor rank out of bounds.".into());
        }
        let mut shape = Vec::with_capacity(shape_arr.len());
        let mut elements = 1_u64;
        for dim in shape_arr {
            let d = dim
                .as_u64()
                .filter(|&n| n <= 1_000_000_000)
                .ok_or("Invalid safetensors dimension.")?;
            elements = elements
                .checked_mul(d)
                .ok_or("Safetensors shape element count overflow.")?;
            shape.push(usize::try_from(d).map_err(|e| e.to_string())?);
        }
        let offsets = rec
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .ok_or("Safetensors descriptor missing data_offsets.")?;
        let begin = offsets[0]
            .as_u64()
            .ok_or("Invalid safetensors begin offset.")?;
        let end = offsets[1]
            .as_u64()
            .ok_or("Invalid safetensors end offset.")?;
        let expected_bytes = elements
            .checked_mul(width)
            .ok_or("Safetensors tensor byte extent overflow.")?;
        if begin > end || end - begin != expected_bytes || end > file_bytes - base {
            return Err(format!("Safetensors extent invalid for '{name}'."));
        }
        tensors.insert(
            name.clone(),
            TensorDescriptor {
                dtype: dtype.to_owned(),
                shape,
                abs_begin: base + begin,
                abs_end: base + end,
            },
        );
    }
    Ok(ShardHeader { tensors })
}

enum SourceKind {
    Local {
        root: PathBuf,
    },
    Https {
        base_url: String,
        repo: String,
        revision: String,
        http_concurrency: usize,
    },
}

/// Read-only streaming source for safetensors index, headers, and tensor byte spans.
pub(super) struct ShardSource {
    kind: SourceKind,
    transferred_bytes: AtomicU64,
    headers: Mutex<BTreeMap<String, ShardHeader>>,
}

impl ShardSource {
    pub(super) fn local(root: PathBuf) -> crate::Result<Self> {
        bonsai_artifacts::validate_external_path(&root).map_err(|e| e.to_string())?;
        if !root.is_dir() {
            return Err("Safetensors local directory does not exist.".into());
        }
        Ok(Self {
            kind: SourceKind::Local { root },
            transferred_bytes: AtomicU64::new(0),
            headers: Mutex::new(BTreeMap::new()),
        })
    }

    pub(super) fn https(
        base_url: String,
        repo: String,
        revision: String,
        http_concurrency: usize,
    ) -> crate::Result<Self> {
        let index_uri =
            format!("{base_url}/{repo}/resolve/{revision}/model.safetensors.index.json");
        crate::https_transfer::validate_uri(&index_uri)?;
        Ok(Self {
            kind: SourceKind::Https {
                base_url,
                repo,
                revision,
                http_concurrency: http_concurrency.clamp(1, 4),
            },
            transferred_bytes: AtomicU64::new(0),
            headers: Mutex::new(BTreeMap::new()),
        })
    }

    pub(super) fn transferred_bytes(&self) -> u64 {
        self.transferred_bytes.load(Ordering::Acquire)
    }

    fn shard_uri(&self, file: &str) -> crate::Result<String> {
        match &self.kind {
            SourceKind::Local { .. } => {
                Err("Internal error: URI requested for local source.".into())
            }
            SourceKind::Https {
                base_url,
                repo,
                revision,
                ..
            } => Ok(format!("{base_url}/{repo}/resolve/{revision}/{file}")),
        }
    }

    pub(super) fn fetch_index(&self, deadline: Instant) -> crate::Result<SafetensorsIndex> {
        match &self.kind {
            SourceKind::Local { root } => {
                let path = root.join("model.safetensors.index.json");
                let bytes = crate::bounded_input::read_regular_file(
                    &path,
                    MAX_INDEX_BYTES,
                    "Safetensors index exceeds 16 MiB limit.",
                )?;
                self.transferred_bytes.fetch_add(
                    u64::try_from(bytes.len()).map_err(|e| e.to_string())?,
                    Ordering::AcqRel,
                );
                parse_index(&bytes)
            }
            SourceKind::Https { .. } => {
                let uri = self.shard_uri("model.safetensors.index.json")?;
                let resp = crate::https_transfer::metadata_with_retry(
                    &uri,
                    &[],
                    deadline,
                    MAX_INDEX_BYTES,
                    &mut crate::startup_cancel::checkpoint,
                )?;
                if resp.status != 200 {
                    return Err(format!(
                        "Safetensors index download returned HTTP {}.",
                        resp.status
                    ));
                }
                self.transferred_bytes.fetch_add(
                    u64::try_from(resp.body.len()).map_err(|e| e.to_string())?,
                    Ordering::AcqRel,
                );
                parse_index(&resp.body)
            }
        }
    }

    pub(super) fn ensure_shard_header(&self, shard: &str, deadline: Instant) -> crate::Result<()> {
        validate_plain_shard_name(shard)?;
        if self
            .headers
            .lock()
            .map_err(|_| "Shard header mutex poisoned.")?
            .contains_key(shard)
        {
            return Ok(());
        }
        let header = match &self.kind {
            SourceKind::Local { root } => {
                let path = root.join(shard);
                let mut file = crate::bounded_input::open_regular_file(&path)?;
                let file_bytes = file.metadata().map_err(|e| e.to_string())?.len();
                let mut prefix = [0_u8; 8];
                file.read_exact(&mut prefix).map_err(|e| e.to_string())?;
                let hdr_len = u64::from_le_bytes(prefix);
                if !(2..=MAX_HEADER_BYTES).contains(&hdr_len)
                    || hdr_len.checked_add(8).is_none_or(|n| n > file_bytes)
                {
                    return Err(format!("Safetensors header exceeds bound in '{shard}'."));
                }
                let mut raw = vec![0_u8; usize::try_from(hdr_len).map_err(|e| e.to_string())?];
                file.read_exact(&mut raw).map_err(|e| e.to_string())?;
                self.transferred_bytes
                    .fetch_add(8 + hdr_len, Ordering::AcqRel);
                parse_shard_header(&raw, file_bytes)?
            }
            SourceKind::Https { .. } => {
                let uri = self.shard_uri(shard)?;
                let (prefix, file_bytes) = crate::https_transfer::fetch_range_with_retry(
                    &uri,
                    0,
                    7,
                    8,
                    deadline,
                    &mut crate::startup_cancel::checkpoint,
                )?;
                let hdr_len = u64::from_le_bytes(
                    prefix
                        .try_into()
                        .map_err(|_| "Safetensors header prefix length mismatch.")?,
                );
                if !(2..=MAX_HEADER_BYTES).contains(&hdr_len)
                    || hdr_len.checked_add(8).is_none_or(|n| n > file_bytes)
                {
                    return Err(format!("Safetensors header exceeds bound in '{shard}'."));
                }
                let (raw, total_after) = crate::https_transfer::fetch_range_with_retry(
                    &uri,
                    8,
                    7 + hdr_len,
                    MAX_HEADER_BYTES,
                    deadline,
                    &mut crate::startup_cancel::checkpoint,
                )?;
                if total_after != file_bytes {
                    return Err(format!("Shard '{shard}' length changed between reads."));
                }
                self.transferred_bytes
                    .fetch_add(8 + hdr_len, Ordering::AcqRel);
                parse_shard_header(&raw, file_bytes)?
            }
        };
        let mut guard = self
            .headers
            .lock()
            .map_err(|_| "Shard header mutex poisoned.")?;
        guard.insert(shard.to_owned(), header);
        Ok(())
    }

    pub(super) fn descriptor(
        &self,
        shard: &str,
        tensor_name: &str,
    ) -> crate::Result<TensorDescriptor> {
        let guard = self
            .headers
            .lock()
            .map_err(|_| "Shard header mutex poisoned.")?;
        let header = guard
            .get(shard)
            .ok_or_else(|| format!("Missing cached shard header for '{shard}'."))?;
        header
            .tensors
            .get(tensor_name)
            .cloned()
            .ok_or_else(|| format!("Tensor '{tensor_name}' missing from shard '{shard}'."))
    }

    pub(super) fn fetch_slice(
        &self,
        shard: &str,
        abs_begin: u64,
        byte_len: u64,
        deadline: Instant,
    ) -> crate::Result<Vec<u8>> {
        if byte_len == 0 {
            return Ok(Vec::new());
        }
        let len_usize = usize::try_from(byte_len).map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        out.try_reserve_exact(len_usize)
            .map_err(|e| format!("Cannot allocate {byte_len}-byte slice buffer: {e}"))?;
        out.resize(len_usize, 0);

        match &self.kind {
            SourceKind::Local { root } => {
                let mut file = crate::bounded_input::open_regular_file(&root.join(shard))?;
                file.seek(SeekFrom::Start(abs_begin))
                    .map_err(|e| e.to_string())?;
                file.read_exact(&mut out).map_err(|e| e.to_string())?;
                self.transferred_bytes.fetch_add(byte_len, Ordering::AcqRel);
                Ok(out)
            }
            SourceKind::Https {
                http_concurrency, ..
            } => {
                let uri = self.shard_uri(shard)?;
                let chunk_bytes = MAX_RANGE_CHUNK_BYTES;
                let chunk_count =
                    usize::try_from(byte_len.div_ceil(chunk_bytes)).map_err(|e| e.to_string())?;
                if chunk_count == 1 || *http_concurrency <= 1 {
                    let mut offset = 0_u64;
                    while offset < byte_len {
                        crate::startup_cancel::checkpoint()?;
                        let take = (byte_len - offset).min(chunk_bytes);
                        let start = abs_begin + offset;
                        let end_inclusive = start + take - 1;
                        let (bytes, _) = crate::https_transfer::fetch_range_with_retry(
                            &uri,
                            start,
                            end_inclusive,
                            chunk_bytes,
                            deadline,
                            &mut crate::startup_cancel::checkpoint,
                        )?;
                        let dst_start = usize::try_from(offset).map_err(|e| e.to_string())?;
                        let dst_end = dst_start + bytes.len();
                        out[dst_start..dst_end].copy_from_slice(&bytes);
                        self.transferred_bytes.fetch_add(take, Ordering::AcqRel);
                        offset += take;
                    }
                    return Ok(out);
                }

                let workers = (*http_concurrency).min(chunk_count);
                let next_chunk = AtomicU64::new(0);
                let cancel = AtomicBool::new(false);
                let chunk_usize = usize::try_from(chunk_bytes).map_err(|e| e.to_string())?;
                let mut dst_slices: Vec<&mut [u8]> = out.chunks_mut(chunk_usize).collect();
                let chunk_slots: Vec<Mutex<Option<&mut [u8]>>> =
                    dst_slices.drain(..).map(|s| Mutex::new(Some(s))).collect();
                let uri_ref = &uri;
                let slots_ref = &chunk_slots;
                let next_ref = &next_chunk;
                let cancel_ref = &cancel;
                let transferred_ref = &self.transferred_bytes;

                let results: Vec<crate::Result<()>> = std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(workers);
                    for _ in 0..workers {
                        handles.push(scope.spawn(move || {
                            loop {
                                if cancel_ref.load(Ordering::Acquire) {
                                    return Err("Sibling chunk failed or cancelled.".into());
                                }
                                let idx = next_ref.fetch_add(1, Ordering::AcqRel);
                                let idx_usize = match usize::try_from(idx) {
                                    Ok(v) if v < slots_ref.len() => v,
                                    _ => return Ok(()),
                                };
                                let offset = idx * chunk_bytes;
                                let take = (byte_len - offset).min(chunk_bytes);
                                let start = abs_begin + offset;
                                let end_inclusive = start + take - 1;
                                let fetch = crate::https_transfer::fetch_range_with_retry(
                                    uri_ref,
                                    start,
                                    end_inclusive,
                                    chunk_bytes,
                                    deadline,
                                    &mut || {
                                        if cancel_ref.load(Ordering::Acquire) {
                                            return Err("Cancelled by peer chunk error.".into());
                                        }
                                        crate::startup_cancel::checkpoint()
                                    },
                                );
                                match fetch {
                                    Ok((bytes, _)) => {
                                        let mut guard = slots_ref[idx_usize]
                                            .lock()
                                            .map_err(|_| "Chunk output slot poisoned.")?;
                                        let slot =
                                            guard.take().ok_or("Chunk slot consumed twice.")?;
                                        slot.copy_from_slice(&bytes);
                                        transferred_ref.fetch_add(take, Ordering::AcqRel);
                                    }
                                    Err(err) => {
                                        cancel_ref.store(true, Ordering::Release);
                                        return Err(err);
                                    }
                                }
                            }
                        }));
                    }
                    handles
                        .into_iter()
                        .map(|h| {
                            h.join()
                                .unwrap_or_else(|_| Err("HTTPS chunk worker panicked.".into()))
                        })
                        .collect()
                });
                for r in results {
                    r?;
                }
                Ok(out)
            }
        }
    }
}

/// A coalesced contiguous byte span covering one or more tensors in a shard.
pub(super) struct CoalescedSpan {
    pub(super) shard: String,
    pub(super) abs_begin: u64,
    pub(super) abs_end: u64,
    pub(super) entries: Vec<(usize, u64, u64)>,
}

/// Groups small tensors in the same shard with gap `<= 64 KiB` and span `<= 32 MiB`.
pub(super) fn coalesce_small_spans(items: &[(usize, &str, u64, u64)]) -> Vec<CoalescedSpan> {
    const MAX_GAP_BYTES: u64 = 64 * 1024;
    let mut by_shard: BTreeMap<&str, Vec<(usize, u64, u64)>> = BTreeMap::new();
    for &(idx, shard, begin, end) in items {
        by_shard.entry(shard).or_default().push((idx, begin, end));
    }
    let mut spans = Vec::new();
    for (shard, mut list) in by_shard {
        list.sort_by_key(|&(_, begin, _)| begin);
        let mut current: Option<CoalescedSpan> = None;
        for (idx, begin, end) in list {
            match current.as_mut() {
                Some(span)
                    if begin >= span.abs_end
                        && begin - span.abs_end <= MAX_GAP_BYTES
                        && end - span.abs_begin <= MAX_RANGE_CHUNK_BYTES =>
                {
                    span.abs_end = span.abs_end.max(end);
                    span.entries.push((idx, begin, end));
                }
                _ => {
                    if let Some(prev) = current.take() {
                        spans.push(prev);
                    }
                    current = Some(CoalescedSpan {
                        shard: shard.to_owned(),
                        abs_begin: begin,
                        abs_end: end,
                        entries: vec![(idx, begin, end)],
                    });
                }
            }
        }
        if let Some(last) = current {
            spans.push(last);
        }
    }
    spans
}
