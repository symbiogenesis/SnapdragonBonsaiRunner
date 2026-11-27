//! Pure-Rust PEFT LoRA `safetensors` importer, orthogonal projector, and exact rank compressor.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::gguf_adapter::{
    ExportDtype, ExportTensorPair, read_lora_gguf_pairs_with_dims, read_rank_one_b_directions,
    sort_pairs_canonically, swift_base_dims_map, write_lora_gguf,
};
use super::safetensors::{
    FloatDtype, ShardHeader, TensorCategory, TensorDescriptor, TensorSpec, coalesce_small_spans,
    decode_scalar_f32, parse_shard_header, swift_tensor_inventory,
};
use super::transform::{check_factor_value, verify_published_adapter};
use super::{
    HUGGINGFACE_BASE_URL, Operation, Request, TargetEnvelope, usize_to_f32,
    validate_commit_revision, validate_hf_repository, validate_license,
};

const DEFAULT_CONFIG_FILENAME: &str = "adapter_config.json";
const DEFAULT_WEIGHTS_FILENAME: &str = "adapter_model.safetensors";
const DEFAULT_MAX_RANK: usize = 16;
const MAX_PEFT_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_PEFT_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PEFT_SAFETENSORS_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_PEFT_SOURCE_RANK: usize = 256;
const MAX_RANGE_CHUNK_BYTES: u64 = 32 * 1024 * 1024;
const IMPORT_METHOD: &str = "peft-safetensors-import-v1";

#[expect(
    clippy::as_conversions,
    reason = "Small LoRA rank/index integers convert losslessly to f64 for exact scaling and SVD."
)]
#[inline]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Donor PEFT LoRA checkpoint source for `import-peft-lora`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum PeftSource {
    LocalDirectory {
        directory: PathBuf,
        license: String,
        #[serde(default)]
        repository: Option<String>,
        #[serde(default)]
        revision: Option<String>,
    },
    #[serde(rename = "pinned-huggingface")]
    PinnedHuggingFace {
        repository: String,
        revision: String,
        license: String,
        #[serde(default, rename = "configFilename")]
        config_filename: Option<String>,
        #[serde(default, rename = "weightsFilename")]
        weights_filename: Option<String>,
    },
}

/// Configuration payload for `import-peft-lora`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct ImportPeftConfig {
    pub(super) source: PeftSource,
    #[serde(default)]
    pub(super) target_envelope: Option<TargetEnvelope>,
    #[serde(default)]
    pub(super) max_rank: Option<usize>,
    #[serde(default)]
    pub(super) export_dtype: Option<ExportDtype>,
    #[serde(default)]
    pub(super) max_payload_bytes: Option<u64>,
    #[serde(default)]
    pub(super) orthogonalize_against_rank_one_path: Option<PathBuf>,
    #[serde(default)]
    pub(super) orthogonalize_against_adapter_path: Option<PathBuf>,
}

fn validate_plain_peft_filename(name: &str, expected_ext: &str, label: &str) -> crate::Result<()> {
    let valid = !name.is_empty()
        && name.len() <= 256
        && name != "."
        && name != ".."
        && name.ends_with(expected_ext)
        && !name.contains(['/', '\\', '\0', '?', '#', '%'])
        && name.bytes().all(|b| (0x21..=0x7e).contains(&b));
    if !valid {
        return Err(format!(
            "{label} must be a plain '{expected_ext}' filename without path separators."
        ));
    }
    Ok(())
}

fn huggingface_resolve_uri(repository: &str, revision: &str, filename: &str) -> String {
    format!("{HUGGINGFACE_BASE_URL}/{repository}/resolve/{revision}/{filename}")
}

/// Validates `peftConfig` bounds, paths, repository/revision pins, and envelope exclusions.
pub(super) fn validate_import_peft_config(config: &ImportPeftConfig) -> crate::Result<()> {
    let envelope = config
        .target_envelope
        .unwrap_or(TargetEnvelope::NpuFriendly);
    if envelope == TargetEnvelope::All {
        return Err(
            "peftConfig.targetEnvelope must not be 'all'; use 'residual-writers', 'npu-friendly', or 'full-linear'."
                .into(),
        );
    }
    let max_rank = config.max_rank.unwrap_or(DEFAULT_MAX_RANK);
    if !(1..=64).contains(&max_rank) {
        return Err("peftConfig.maxRank must be in 1..=64.".into());
    }
    if config.max_payload_bytes == Some(0) {
        return Err("peftConfig.maxPayloadBytes must be positive.".into());
    }
    if let Some(path) = &config.orthogonalize_against_rank_one_path {
        bonsai_artifacts::validate_external_path(path).map_err(|e| e.to_string())?;
    }
    if let Some(path) = &config.orthogonalize_against_adapter_path {
        bonsai_artifacts::validate_external_path(path).map_err(|e| e.to_string())?;
    }

    match &config.source {
        PeftSource::LocalDirectory {
            directory,
            license,
            repository,
            revision,
        } => {
            bonsai_artifacts::validate_external_path(directory).map_err(|e| e.to_string())?;
            if !directory.is_dir() {
                return Err("peftConfig.source.directory must exist as a directory.".into());
            }
            validate_license(license, "peftConfig.source.license")?;
            if let Some(repo) = repository {
                validate_hf_repository(repo, "peftConfig.source.repository")?;
            }
            if let Some(rev) = revision {
                validate_commit_revision(rev, "peftConfig.source.revision")?;
            }
        }
        PeftSource::PinnedHuggingFace {
            repository,
            revision,
            license,
            config_filename,
            weights_filename,
        } => {
            validate_hf_repository(repository, "peftConfig.source.repository")?;
            validate_commit_revision(revision, "peftConfig.source.revision")?;
            validate_license(license, "peftConfig.source.license")?;
            let cfg_name = config_filename
                .as_deref()
                .unwrap_or(DEFAULT_CONFIG_FILENAME);
            let wt_name = weights_filename
                .as_deref()
                .unwrap_or(DEFAULT_WEIGHTS_FILENAME);
            validate_plain_peft_filename(cfg_name, ".json", "peftConfig.source.configFilename")?;
            validate_plain_peft_filename(
                wt_name,
                ".safetensors",
                "peftConfig.source.weightsFilename",
            )?;
            crate::https_transfer::validate_uri(&huggingface_resolve_uri(
                repository, revision, cfg_name,
            ))?;
            crate::https_transfer::validate_uri(&huggingface_resolve_uri(
                repository, revision, wt_name,
            ))?;
        }
    }
    Ok(())
}

#[derive(Debug)]
struct ParsedPeftAdapterConfig {
    r: usize,
    lora_alpha: f64,
    use_rslora: bool,
    scale_peft: f64,
}

fn parse_peft_adapter_config(bytes: &[u8]) -> crate::Result<ParsedPeftAdapterConfig> {
    if bytes.is_empty() || u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_PEFT_CONFIG_BYTES {
        return Err("PEFT adapter_config.json size must be in 1..=1048576 bytes.".into());
    }
    let root: Value = serde_json::from_slice(crate::bounded_input::strip_utf8_bom(bytes))
        .map_err(|e| format!("Invalid PEFT adapter_config.json: {e}"))?;
    let obj = root
        .as_object()
        .ok_or("PEFT adapter_config.json root must be a JSON object.")?;

    let peft_type = obj
        .get("peft_type")
        .and_then(Value::as_str)
        .ok_or("PEFT adapter_config.json missing string peft_type.")?;
    if peft_type != "LORA" {
        return Err(format!(
            "Unsupported peft_type '{peft_type}'; expected 'LORA'."
        ));
    }

    let r_u64 = obj
        .get("r")
        .and_then(Value::as_u64)
        .ok_or("PEFT adapter_config.json missing integer rank r.")?;
    if !(1..=u64::try_from(MAX_PEFT_SOURCE_RANK).unwrap_or(256)).contains(&r_u64) {
        return Err(format!(
            "PEFT adapter_config.json rank r={r_u64} must be in 1..={MAX_PEFT_SOURCE_RANK}."
        ));
    }
    let r = usize::try_from(r_u64).map_err(|e| e.to_string())?;

    let lora_alpha = obj
        .get("lora_alpha")
        .and_then(Value::as_f64)
        .ok_or("PEFT adapter_config.json missing numeric lora_alpha.")?;
    if !lora_alpha.is_finite() || lora_alpha <= 0.0 {
        return Err("PEFT adapter_config.json lora_alpha must be finite and positive.".into());
    }

    match obj.get("use_dora") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => {
            return Err("PEFT DoRA (use_dora=true) adapters are not supported.".into());
        }
        Some(_) => return Err("PEFT adapter_config.json has invalid use_dora field.".into()),
    }

    match obj.get("bias") {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) if text == "none" || text.is_empty() => {}
        Some(Value::String(other)) => {
            return Err(format!(
                "PEFT bias='{other}' is not supported; expected omitted, null, '', or 'none'."
            ));
        }
        Some(_) => return Err("PEFT adapter_config.json has invalid bias field.".into()),
    }

    match obj.get("fan_in_fan_out") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => {
            return Err("PEFT fan_in_fan_out=true is not supported.".into());
        }
        Some(_) => {
            return Err("PEFT adapter_config.json has invalid fan_in_fan_out field.".into());
        }
    }

    let use_rslora = match obj.get("use_rslora") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(_) => return Err("PEFT adapter_config.json has invalid use_rslora field.".into()),
    };

    let r_f64 = usize_to_f64(r);
    let scale_peft = if use_rslora {
        lora_alpha / r_f64.sqrt()
    } else {
        lora_alpha / r_f64
    };
    if !scale_peft.is_finite() || scale_peft <= 0.0 {
        return Err("Computed PEFT scaling factor is invalid.".into());
    }

    Ok(ParsedPeftAdapterConfig {
        r,
        lora_alpha,
        use_rslora,
        scale_peft,
    })
}

enum PeftWeightsHandle {
    Local {
        file: File,
    },
    Https {
        weights_uri: String,
        http_concurrency: usize,
    },
}

struct PeftDonorIntake {
    config_bytes: Vec<u8>,
    header: ShardHeader,
    weights: PeftWeightsHandle,
    transferred_bytes: AtomicU64,
}

impl PeftDonorIntake {
    fn open(
        source: &PeftSource,
        http_concurrency: usize,
        deadline: Instant,
    ) -> crate::Result<Self> {
        match source {
            PeftSource::LocalDirectory { directory, .. } => {
                let config_path = directory.join(DEFAULT_CONFIG_FILENAME);
                bonsai_artifacts::validate_external_path(&config_path)
                    .map_err(|e| e.to_string())?;
                let config_bytes = crate::bounded_input::read_regular_file(
                    &config_path,
                    MAX_PEFT_CONFIG_BYTES,
                    "PEFT adapter_config.json exceeds 1 MiB limit.",
                )?;

                let weights_path = directory.join(DEFAULT_WEIGHTS_FILENAME);
                bonsai_artifacts::validate_external_path(&weights_path)
                    .map_err(|e| e.to_string())?;
                let mut file = crate::bounded_input::open_regular_file(&weights_path)?;
                let file_bytes = file.metadata().map_err(|e| e.to_string())?.len();
                if !(10..=MAX_PEFT_SAFETENSORS_BYTES).contains(&file_bytes) {
                    return Err(format!(
                        "PEFT safetensors file length {file_bytes} outside 10..={MAX_PEFT_SAFETENSORS_BYTES} bytes."
                    ));
                }
                let mut prefix = [0_u8; 8];
                file.read_exact(&mut prefix).map_err(|e| e.to_string())?;
                let hdr_len = u64::from_le_bytes(prefix);
                if !(2..=MAX_PEFT_HEADER_BYTES).contains(&hdr_len)
                    || hdr_len.checked_add(8).is_none_or(|n| n > file_bytes)
                {
                    return Err("PEFT safetensors header exceeds bounded size or EOF.".into());
                }
                let hdr_usize = usize::try_from(hdr_len).map_err(|e| e.to_string())?;
                let mut raw_hdr = Vec::new();
                raw_hdr
                    .try_reserve_exact(hdr_usize)
                    .map_err(|e| format!("Cannot allocate PEFT header buffer: {e}"))?;
                raw_hdr.resize(hdr_usize, 0);
                file.read_exact(&mut raw_hdr).map_err(|e| e.to_string())?;
                let header = parse_shard_header(&raw_hdr, file_bytes)?;
                let initial_bytes = u64::try_from(config_bytes.len())
                    .map_err(|e| e.to_string())?
                    .saturating_add(8)
                    .saturating_add(hdr_len);

                Ok(Self {
                    config_bytes,
                    header,
                    weights: PeftWeightsHandle::Local { file },
                    transferred_bytes: AtomicU64::new(initial_bytes),
                })
            }
            PeftSource::PinnedHuggingFace {
                repository,
                revision,
                license: _,
                config_filename,
                weights_filename,
            } => {
                crate::startup_cancel::checkpoint()?;
                let cfg_name = config_filename
                    .as_deref()
                    .unwrap_or(DEFAULT_CONFIG_FILENAME);
                let wt_name = weights_filename
                    .as_deref()
                    .unwrap_or(DEFAULT_WEIGHTS_FILENAME);
                let config_uri = huggingface_resolve_uri(repository, revision, cfg_name);
                let weights_uri = huggingface_resolve_uri(repository, revision, wt_name);

                let req_headers = [("Accept-Encoding".to_owned(), "identity".to_owned())];
                let resp = crate::https_transfer::metadata_with_retry(
                    &config_uri,
                    &req_headers,
                    deadline,
                    MAX_PEFT_CONFIG_BYTES,
                    &mut crate::startup_cancel::checkpoint,
                )?;
                if resp.status != 200 {
                    return Err(format!(
                        "PEFT adapter config download returned HTTP {}.",
                        resp.status
                    ));
                }
                if resp
                    .headers
                    .get("content-encoding")
                    .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
                {
                    return Err("PEFT adapter config requires identity content encoding.".into());
                }

                let (prefix, file_bytes) = crate::https_transfer::fetch_range_with_retry(
                    &weights_uri,
                    0,
                    7,
                    8,
                    deadline,
                    &mut crate::startup_cancel::checkpoint,
                )?;
                if !(10..=MAX_PEFT_SAFETENSORS_BYTES).contains(&file_bytes) {
                    return Err(format!(
                        "PEFT safetensors file length {file_bytes} outside 10..={MAX_PEFT_SAFETENSORS_BYTES} bytes."
                    ));
                }
                let hdr_len = u64::from_le_bytes(
                    prefix
                        .try_into()
                        .map_err(|_| "PEFT safetensors header prefix length mismatch.")?,
                );
                if !(2..=MAX_PEFT_HEADER_BYTES).contains(&hdr_len)
                    || hdr_len.checked_add(8).is_none_or(|n| n > file_bytes)
                {
                    return Err("PEFT safetensors header exceeds bounded size or EOF.".into());
                }
                let (raw_hdr, total_after) = crate::https_transfer::fetch_range_with_retry(
                    &weights_uri,
                    8,
                    7 + hdr_len,
                    MAX_PEFT_HEADER_BYTES,
                    deadline,
                    &mut crate::startup_cancel::checkpoint,
                )?;
                if total_after != file_bytes {
                    return Err("PEFT safetensors length changed during header read.".into());
                }
                let header = parse_shard_header(&raw_hdr, file_bytes)?;
                let initial_bytes = u64::try_from(resp.body.len())
                    .map_err(|e| e.to_string())?
                    .saturating_add(8)
                    .saturating_add(hdr_len);

                Ok(Self {
                    config_bytes: resp.body,
                    header,
                    weights: PeftWeightsHandle::Https {
                        weights_uri,
                        http_concurrency: http_concurrency.clamp(1, 4),
                    },
                    transferred_bytes: AtomicU64::new(initial_bytes),
                })
            }
        }
    }

    fn fetch_all_slices(
        &mut self,
        requests: &[(u64, u64)],
        deadline: Instant,
    ) -> crate::Result<Vec<Vec<u8>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        match &mut self.weights {
            PeftWeightsHandle::Local { file } => {
                let mut out = Vec::with_capacity(requests.len());
                for &(begin, end) in requests {
                    crate::startup_cancel::checkpoint()?;
                    let byte_len = end.checked_sub(begin).ok_or("Invalid slice extent.")?;
                    let len_usize = usize::try_from(byte_len).map_err(|e| e.to_string())?;
                    let mut buf = Vec::new();
                    buf.try_reserve_exact(len_usize)
                        .map_err(|e| format!("Cannot allocate PEFT tensor buffer: {e}"))?;
                    buf.resize(len_usize, 0);
                    file.seek(SeekFrom::Start(begin))
                        .map_err(|e| e.to_string())?;
                    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
                    self.transferred_bytes.fetch_add(byte_len, Ordering::AcqRel);
                    out.push(buf);
                }
                Ok(out)
            }
            PeftWeightsHandle::Https {
                weights_uri,
                http_concurrency,
            } => {
                let items: Vec<(usize, &str, u64, u64)> = requests
                    .iter()
                    .enumerate()
                    .map(|(idx, &(begin, end))| (idx, "adapter", begin, end))
                    .collect();
                let spans = coalesce_small_spans(&items);
                let mut slots: Vec<Option<Vec<u8>>> = vec![None; requests.len()];
                let span_results = fetch_https_spans(
                    weights_uri,
                    &spans,
                    *http_concurrency,
                    deadline,
                    &self.transferred_bytes,
                )?;
                for (span, span_bytes) in spans.into_iter().zip(span_results) {
                    for (idx, begin, end) in span.entries {
                        let rel_begin =
                            usize::try_from(begin - span.abs_begin).map_err(|e| e.to_string())?;
                        let rel_end =
                            usize::try_from(end - span.abs_begin).map_err(|e| e.to_string())?;
                        slots[idx] = Some(span_bytes[rel_begin..rel_end].to_vec());
                    }
                }
                slots
                    .into_iter()
                    .map(|s| s.ok_or_else(|| "Missing fetched PEFT tensor slice.".into()))
                    .collect()
            }
        }
    }
}

fn fetch_single_https_span(
    uri: &str,
    abs_begin: u64,
    abs_end: u64,
    deadline: Instant,
    transferred: &AtomicU64,
    cancel: &AtomicBool,
) -> crate::Result<Vec<u8>> {
    let byte_len = abs_end
        .checked_sub(abs_begin)
        .ok_or("Invalid HTTPS span bounds.")?;
    let len_usize = usize::try_from(byte_len).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    out.try_reserve_exact(len_usize)
        .map_err(|e| format!("Cannot allocate {byte_len}-byte HTTPS span buffer: {e}"))?;
    out.resize(len_usize, 0);

    let mut offset = 0_u64;
    while offset < byte_len {
        if cancel.load(Ordering::Acquire) {
            return Err("Cancelled by sibling HTTPS transfer failure.".into());
        }
        if Instant::now() >= deadline {
            return Err("PEFT import time budget expired during HTTPS transfer.".into());
        }
        let take = (byte_len - offset).min(MAX_RANGE_CHUNK_BYTES);
        let start = abs_begin + offset;
        let end_inclusive = start + take - 1;
        let (bytes, _) = crate::https_transfer::fetch_range_with_retry(
            uri,
            start,
            end_inclusive,
            MAX_RANGE_CHUNK_BYTES,
            deadline,
            &mut || {
                if cancel.load(Ordering::Acquire) {
                    return Err("Cancelled by sibling HTTPS transfer failure.".into());
                }
                crate::startup_cancel::checkpoint()
            },
        )?;
        let dst_start = usize::try_from(offset).map_err(|e| e.to_string())?;
        let dst_end = dst_start + bytes.len();
        out[dst_start..dst_end].copy_from_slice(&bytes);
        transferred.fetch_add(take, Ordering::AcqRel);
        offset += take;
    }
    Ok(out)
}

fn fetch_https_spans(
    uri: &str,
    spans: &[super::safetensors::CoalescedSpan],
    http_concurrency: usize,
    deadline: Instant,
    transferred: &AtomicU64,
) -> crate::Result<Vec<Vec<u8>>> {
    if spans.is_empty() {
        return Ok(Vec::new());
    }
    let cancel = AtomicBool::new(false);
    if http_concurrency <= 1 || spans.len() == 1 {
        let mut results = Vec::with_capacity(spans.len());
        for span in spans {
            results.push(fetch_single_https_span(
                uri,
                span.abs_begin,
                span.abs_end,
                deadline,
                transferred,
                &cancel,
            )?);
        }
        return Ok(results);
    }

    let workers = http_concurrency.clamp(1, 4).min(spans.len());
    let next_idx = AtomicUsize::new(0);
    let slot_results: Vec<std::sync::Mutex<Option<Vec<u8>>>> = (0..spans.len())
        .map(|_| std::sync::Mutex::new(None))
        .collect();
    let results: Vec<crate::Result<()>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            handles.push(scope.spawn(|| {
                loop {
                    if cancel.load(Ordering::Acquire) {
                        return Err("Cancelled by sibling HTTPS span error.".into());
                    }
                    let idx = next_idx.fetch_add(1, Ordering::AcqRel);
                    let Some(span) = spans.get(idx) else {
                        return Ok(());
                    };
                    match fetch_single_https_span(
                        uri,
                        span.abs_begin,
                        span.abs_end,
                        deadline,
                        transferred,
                        &cancel,
                    ) {
                        Ok(bytes) => {
                            let mut guard = slot_results[idx]
                                .lock()
                                .map_err(|_| "HTTPS span slot mutex poisoned.")?;
                            *guard = Some(bytes);
                        }
                        Err(err) => {
                            cancel.store(true, Ordering::Release);
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
                    .unwrap_or_else(|_| Err("PEFT HTTPS worker panicked.".into()))
            })
            .collect()
    });
    for r in results {
        r?;
    }
    slot_results
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .map_err(|_| "HTTPS span slot mutex poisoned.".to_owned())?
                .ok_or_else(|| "Missing PEFT HTTPS span result.".to_owned())
        })
        .collect()
}

#[derive(Clone, Copy)]
enum PeftRole {
    A,
    B,
}

fn split_peft_key(key: &str) -> crate::Result<(&str, PeftRole)> {
    let (prefix, role) = if let Some(base) = key.strip_suffix(".lora_A.weight") {
        (base, PeftRole::A)
    } else if let Some(base) = key.strip_suffix(".lora_A.default.weight") {
        (base, PeftRole::A)
    } else if let Some(base) = key.strip_suffix(".lora_B.weight") {
        (base, PeftRole::B)
    } else if let Some(base) = key.strip_suffix(".lora_B.default.weight") {
        (base, PeftRole::B)
    } else {
        return Err(format!(
            "Unsupported or non-LoRA tensor key '{key}' in PEFT safetensors."
        ));
    };
    let stripped = prefix.strip_prefix("base_model.model.").unwrap_or(prefix);
    if stripped.is_empty() {
        return Err(format!(
            "Malformed empty target prefix in PEFT key '{key}'."
        ));
    }
    Ok((stripped, role))
}

fn canonical_hf_module_key(stripped: &str) -> Cow<'_, str> {
    if let Some(rest) = stripped.strip_prefix("model.layers.") {
        Cow::Owned(format!("model.language_model.layers.{rest}"))
    } else if stripped == "model.embed_tokens" {
        Cow::Borrowed("model.language_model.embed_tokens")
    } else if stripped == "model.norm" {
        Cow::Borrowed("model.language_model.norm")
    } else {
        Cow::Borrowed(stripped)
    }
}

fn is_vision_or_mmproj_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    lower.contains("visual")
        || lower.contains("vision_tower")
        || lower.contains("vision_model")
        || lower.contains("mm_projector")
        || lower.contains("multi_modal_projector")
}

#[derive(Default)]
struct DroppedTargetCounts {
    non_npu_targets: usize,
    grouped_ssm_targets: usize,
    vision_mmproj_targets: usize,
    other_targets: usize,
}

struct IncludedPeftTarget {
    gguf_name: String,
    family: &'static str,
    in_features: usize,
    out_features: usize,
    r_src: usize,
    a_dtype: FloatDtype,
    b_dtype: FloatDtype,
    a_abs_begin: u64,
    a_abs_end: u64,
    b_abs_begin: u64,
    b_abs_end: u64,
}

fn classify_peft_targets(
    header: &ShardHeader,
    envelope: TargetEnvelope,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<(Vec<IncludedPeftTarget>, DroppedTargetCounts)> {
    let mut by_target: BTreeMap<
        String,
        (
            Option<(String, TensorDescriptor)>,
            Option<(String, TensorDescriptor)>,
        ),
    > = BTreeMap::new();

    for (raw_key, desc) in &header.tensors {
        let (stripped, role) = split_peft_key(raw_key)?;
        let canonical = canonical_hf_module_key(stripped).into_owned();
        let entry = by_target.entry(canonical.clone()).or_insert((None, None));
        match role {
            PeftRole::A => {
                if entry.0.is_some() {
                    return Err(format!(
                        "Duplicate lora_A tensor for PEFT target '{canonical}'."
                    ));
                }
                entry.0 = Some((raw_key.clone(), desc.clone()));
            }
            PeftRole::B => {
                if entry.1.is_some() {
                    return Err(format!(
                        "Duplicate lora_B tensor for PEFT target '{canonical}'."
                    ));
                }
                entry.1 = Some((raw_key.clone(), desc.clone()));
            }
        }
    }

    let mut hf_to_spec: BTreeMap<String, TensorSpec> = BTreeMap::new();
    for spec in swift_tensor_inventory() {
        let module_key = spec
            .hf_name
            .strip_suffix(".weight")
            .unwrap_or(&spec.hf_name)
            .to_owned();
        hf_to_spec.insert(module_key, spec);
    }

    let mut dropped = DroppedTargetCounts::default();
    let mut included = Vec::new();
    let mut seen_gguf = BTreeSet::new();

    for (canonical_key, (maybe_a, maybe_b)) in by_target {
        let (Some((a_name, a_desc)), Some((b_name, b_desc))) = (maybe_a, maybe_b) else {
            return Err(format!(
                "Incomplete PEFT LoRA A/B pair for target '{canonical_key}'."
            ));
        };

        if is_vision_or_mmproj_key(&canonical_key) {
            dropped.vision_mmproj_targets += 1;
            continue;
        }

        let Some(spec) = hf_to_spec.get(&canonical_key) else {
            dropped.other_targets += 1;
            continue;
        };

        if spec.category == TensorCategory::LinearMatrix && envelope.matches_family(spec.family) {
            let &(in_features, out_features, family) =
                base_dims.get(&spec.gguf_name).ok_or_else(|| {
                    format!(
                        "Swift base dimensions missing for included PEFT target '{}'.",
                        spec.gguf_name
                    )
                })?;
            if !seen_gguf.insert(spec.gguf_name.clone()) {
                return Err(format!(
                    "Duplicate Swift target '{}' mapped from PEFT keys.",
                    spec.gguf_name
                ));
            }
            let a_dtype = a_desc.float_dtype()?;
            let b_dtype = b_desc.float_dtype()?;
            if a_desc.shape.len() != 2 || b_desc.shape.len() != 2 {
                return Err(format!(
                    "Expected 2D PEFT LoRA matrices for '{}'/'{}', got A={:?}, B={:?}.",
                    a_name, b_name, a_desc.shape, b_desc.shape
                ));
            }
            let r_src = a_desc.shape[0];
            if a_desc.shape[1] != in_features
                || b_desc.shape[0] != out_features
                || b_desc.shape[1] != r_src
            {
                return Err(format!(
                    "PEFT shape mismatch for '{}': expected A=[r, {in_features}] B=[{out_features}, r], got A={:?}, B={:?}.",
                    spec.gguf_name, a_desc.shape, b_desc.shape
                ));
            }
            if !(1..=MAX_PEFT_SOURCE_RANK).contains(&r_src) {
                return Err(format!(
                    "PEFT source tensor rank {r_src} for '{}' outside 1..={MAX_PEFT_SOURCE_RANK}.",
                    spec.gguf_name
                ));
            }
            included.push(IncludedPeftTarget {
                gguf_name: spec.gguf_name.clone(),
                family,
                in_features,
                out_features,
                r_src,
                a_dtype,
                b_dtype,
                a_abs_begin: a_desc.abs_begin,
                a_abs_end: a_desc.abs_end,
                b_abs_begin: b_desc.abs_begin,
                b_abs_end: b_desc.abs_end,
            });
        } else if spec.family == "ssm_out" {
            dropped.grouped_ssm_targets += 1;
        } else if matches!(spec.family, "ffn_gate" | "ffn_up") {
            dropped.non_npu_targets += 1;
        } else {
            dropped.other_targets += 1;
        }
    }

    if included.is_empty() {
        return Err("PEFT import selected zero target matrices inside requested envelope.".into());
    }

    Ok((included, dropped))
}

#[inline]
fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

fn orthonormalize_ref_columns(cols: Vec<Vec<f64>>) -> crate::Result<Vec<Vec<f64>>> {
    let mut max_norm = 0.0_f64;
    for col in &cols {
        let mut norm_sq = 0.0_f64;
        for &v in col {
            if !v.is_finite() {
                return Err("Nonfinite value in reference adapter B factor.".into());
            }
            norm_sq += v * v;
        }
        if !norm_sq.is_finite() {
            return Err("Reference adapter B norm overflowed f64.".into());
        }
        max_norm = max_norm.max(norm_sq.sqrt());
    }
    if max_norm == 0.0 {
        return Ok(Vec::new());
    }
    let threshold = max_norm * 1e-12;
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(cols.len());
    for mut col in cols {
        for _pass in 0..2 {
            for q in &basis {
                let proj = dot_f64(&col, q);
                for (dst, &qv) in col.iter_mut().zip(q) {
                    *dst -= proj * qv;
                }
            }
        }
        let norm = dot_f64(&col, &col).sqrt();
        if norm.is_finite() && norm > threshold {
            let inv = 1.0 / norm;
            for v in &mut col {
                *v *= inv;
            }
            basis.push(col);
        }
    }
    Ok(basis)
}

struct ThinQr {
    q_cols: Vec<Vec<f64>>,
    r_rows: Vec<f64>,
    k: usize,
}

fn thin_qr_columns(cols: &[Vec<f64>]) -> crate::Result<ThinQr> {
    let r = cols.len();
    if r == 0 || cols[0].is_empty() {
        return Ok(ThinQr {
            q_cols: Vec::new(),
            r_rows: Vec::new(),
            k: 0,
        });
    }
    let mut max_norm = 0.0_f64;
    for col in cols {
        let norm_sq: f64 = col.iter().map(|&v| v * v).sum();
        if !norm_sq.is_finite() {
            return Err("Nonfinite factor weight during thin QR.".into());
        }
        max_norm = max_norm.max(norm_sq.sqrt());
    }
    if max_norm == 0.0 {
        return Ok(ThinQr {
            q_cols: Vec::new(),
            r_rows: Vec::new(),
            k: 0,
        });
    }
    let threshold = max_norm * 1e-12;
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(r);
    let mut r_rows: Vec<Vec<f64>> = Vec::with_capacity(r);

    for (j, col) in cols.iter().enumerate() {
        let mut v = col.clone();
        for _pass in 0..2 {
            for (i, q_i) in q_cols.iter().enumerate() {
                let proj = dot_f64(&v, q_i);
                r_rows[i][j] += proj;
                for (dst, &qv) in v.iter_mut().zip(q_i) {
                    *dst -= proj * qv;
                }
            }
        }
        let rem = dot_f64(&v, &v).sqrt();
        if rem.is_finite() && rem > threshold {
            let inv = 1.0 / rem;
            for x in &mut v {
                *x *= inv;
            }
            let mut row = vec![0.0_f64; r];
            row[j] = rem;
            q_cols.push(v);
            r_rows.push(row);
        }
    }

    let k = q_cols.len();
    let mut flat_r = Vec::with_capacity(k * r);
    for row in r_rows {
        flat_r.extend_from_slice(&row);
    }
    Ok(ThinQr {
        q_cols,
        r_rows: flat_r,
        k,
    })
}

struct SmallSvd {
    sigmas: Vec<f64>,
    u_cols: Vec<Vec<f64>>,
    v_cols: Vec<Vec<f64>>,
    total_energy: f64,
}

/// Exact one-sided Hestenes–Jacobi SVD of a small `rows x cols` matrix (`rows, cols <= 256`).
fn jacobi_svd_f64(m_row_major: &[f64], rows: usize, cols: usize) -> crate::Result<SmallSvd> {
    let total_energy: f64 = m_row_major.iter().map(|&v| v * v).sum();
    if !total_energy.is_finite() {
        return Err("Small core matrix Frobenius energy overflowed f64.".into());
    }
    if rows == 0 || cols == 0 || total_energy == 0.0 {
        return Ok(SmallSvd {
            sigmas: Vec::new(),
            u_cols: Vec::new(),
            v_cols: Vec::new(),
            total_energy: 0.0,
        });
    }

    let transposed = rows < cols;
    let (m_eff, n_eff) = if transposed {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let mut w_cols: Vec<Vec<f64>> = (0..n_eff)
        .map(|j| {
            if transposed {
                (0..m_eff).map(|i| m_row_major[j * m_eff + i]).collect()
            } else {
                (0..m_eff).map(|i| m_row_major[i * n_eff + j]).collect()
            }
        })
        .collect();
    let mut v_cols: Vec<Vec<f64>> = (0..n_eff)
        .map(|j| {
            let mut col = vec![0.0_f64; n_eff];
            col[j] = 1.0;
            col
        })
        .collect();

    let frob_norm = total_energy.sqrt();
    let skip_pair_thresh = total_energy * 1e-26;
    for _sweep in 0..80 {
        let mut converged = true;
        for p in 0..n_eff {
            for q in (p + 1)..n_eff {
                let alpha = dot_f64(&w_cols[p], &w_cols[p]);
                let beta = dot_f64(&w_cols[q], &w_cols[q]);
                let denom = (alpha * beta).sqrt();
                if denom <= skip_pair_thresh {
                    continue;
                }
                let gamma = dot_f64(&w_cols[p], &w_cols[q]);
                if gamma.abs() <= 1e-14 * denom {
                    continue;
                }
                converged = false;
                let tau = (beta - alpha) / (2.0 * gamma);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                let (left_w, right_w) = w_cols.split_at_mut(q);
                let wp = &mut left_w[p];
                let wq = &mut right_w[0];
                for i in 0..m_eff {
                    let a = wp[i];
                    let b = wq[i];
                    wp[i] = c * a - s * b;
                    wq[i] = s * a + c * b;
                }

                let (left_v, right_v) = v_cols.split_at_mut(q);
                let vp = &mut left_v[p];
                let vq = &mut right_v[0];
                for i in 0..n_eff {
                    let a = vp[i];
                    let b = vq[i];
                    vp[i] = c * a - s * b;
                    vq[i] = s * a + c * b;
                }
            }
        }
        if converged {
            break;
        }
    }

    let sigma_thresh = frob_norm * 1e-12;
    let mut triplets: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::with_capacity(n_eff);
    for (w_col, v_col) in w_cols.into_iter().zip(v_cols) {
        let sigma = dot_f64(&w_col, &w_col).sqrt();
        if sigma.is_finite() && sigma > sigma_thresh {
            let inv = 1.0 / sigma;
            let u_col: Vec<f64> = w_col.into_iter().map(|x| x * inv).collect();
            if transposed {
                triplets.push((sigma, v_col, u_col));
            } else {
                triplets.push((sigma, u_col, v_col));
            }
        }
    }
    triplets.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut sigmas = Vec::with_capacity(triplets.len());
    let mut u_out = Vec::with_capacity(triplets.len());
    let mut v_out = Vec::with_capacity(triplets.len());
    for (sigma, u_col, v_col) in triplets {
        sigmas.push(sigma);
        u_out.push(u_col);
        v_out.push(v_col);
    }
    Ok(SmallSvd {
        sigmas,
        u_cols: u_out,
        v_cols: v_out,
        total_energy,
    })
}

fn factor_product_frobenius_energy(b_cols: &[Vec<f64>], a_rows: &[Vec<f64>]) -> crate::Result<f64> {
    let r = b_cols.len();
    if r == 0 || a_rows.len() != r {
        return Ok(0.0);
    }
    let mut gram_b = vec![0.0_f64; r * r];
    let mut gram_a = vec![0.0_f64; r * r];
    for i in 0..r {
        for j in i..r {
            let gb = dot_f64(&b_cols[i], &b_cols[j]);
            let ga = dot_f64(&a_rows[i], &a_rows[j]);
            gram_b[i * r + j] = gb;
            gram_b[j * r + i] = gb;
            gram_a[i * r + j] = ga;
            gram_a[j * r + i] = ga;
        }
    }
    let mut trace = 0.0_f64;
    for i in 0..r {
        for j in 0..r {
            trace += gram_b[i * r + j] * gram_a[j * r + i];
        }
    }
    if !trace.is_finite() {
        return Err("PEFT factor Frobenius energy overflowed f64.".into());
    }
    Ok(trace.max(0.0))
}

enum PreparedFactors {
    Direct {
        b_cols: Vec<Vec<f64>>,
        a_rows: Vec<Vec<f64>>,
    },
    Compressed {
        u_cols: Vec<Vec<f64>>,
        v_cols: Vec<Vec<f64>>,
        sigmas: Vec<f64>,
    },
}

struct PreparedSite {
    gguf_name: String,
    family: &'static str,
    in_features: usize,
    out_features: usize,
    source_rank: usize,
    exported_rank: usize,
    orthogonalized: bool,
    compressed: bool,
    residual_energy: f64,
    retained_energy: f64,
    retained_fraction: f64,
    factors: PreparedFactors,
}

fn decode_peft_factors(
    target: &IncludedPeftTarget,
    a_bytes: &[u8],
    b_bytes: &[u8],
    factor_scale: f64,
) -> crate::Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let a_width = target.a_dtype.byte_width();
    let b_width = target.b_dtype.byte_width();
    let expected_a_bytes = target
        .r_src
        .checked_mul(target.in_features)
        .and_then(|n| n.checked_mul(a_width))
        .ok_or("PEFT lora_A byte length overflow.")?;
    let expected_b_bytes = target
        .out_features
        .checked_mul(target.r_src)
        .and_then(|n| n.checked_mul(b_width))
        .ok_or("PEFT lora_B byte length overflow.")?;
    if a_bytes.len() != expected_a_bytes || b_bytes.len() != expected_b_bytes {
        return Err(format!(
            "Fetched PEFT tensor byte length mismatch on '{}'.",
            target.gguf_name
        ));
    }

    let mut a_rows = Vec::with_capacity(target.r_src);
    for k in 0..target.r_src {
        let mut row = Vec::new();
        row.try_reserve_exact(target.in_features)
            .map_err(|e| format!("Cannot allocate PEFT lora_A row: {e}"))?;
        let row_off = k * target.in_features * a_width;
        for c in 0..target.in_features {
            let off = row_off + c * a_width;
            let v = f64::from(decode_scalar_f32(
                target.a_dtype,
                &a_bytes[off..off + a_width],
            )?);
            row.push(v * factor_scale);
        }
        a_rows.push(row);
    }

    let mut b_cols = Vec::with_capacity(target.r_src);
    for _ in 0..target.r_src {
        let mut col = Vec::new();
        col.try_reserve_exact(target.out_features)
            .map_err(|e| format!("Cannot allocate PEFT lora_B column: {e}"))?;
        col.resize(target.out_features, 0.0_f64);
        b_cols.push(col);
    }
    for row in 0..target.out_features {
        let row_off = row * target.r_src * b_width;
        for k in 0..target.r_src {
            let off = row_off + k * b_width;
            let v = f64::from(decode_scalar_f32(
                target.b_dtype,
                &b_bytes[off..off + b_width],
            )?);
            b_cols[k][row] = v * factor_scale;
        }
    }

    Ok((b_cols, a_rows))
}

fn build_site_reference_basis(
    gguf_name: &str,
    in_features: usize,
    out_features: usize,
    rank_one_dirs: Option<&BTreeMap<String, Vec<f64>>>,
    ref_pairs: Option<&BTreeMap<String, ExportTensorPair>>,
) -> crate::Result<Vec<Vec<f64>>> {
    let mut ref_cols = Vec::new();
    if let Some(dirs) = rank_one_dirs
        && let Some(dir) = dirs.get(gguf_name)
    {
        if dir.len() != out_features {
            return Err(format!(
                "Rank-1 reference output dimension mismatch on '{gguf_name}': expected {out_features}, got {}.",
                dir.len()
            ));
        }
        ref_cols.push(dir.clone());
    }
    if let Some(pairs) = ref_pairs
        && let Some(pair) = pairs.get(gguf_name)
    {
        if pair.out_features != out_features || pair.in_features != in_features {
            return Err(format!(
                "Reference adapter dimension mismatch on '{gguf_name}': expected [{out_features}, {in_features}], got [{}, {}].",
                pair.out_features, pair.in_features
            ));
        }
        for k in 0..pair.rank {
            let mut col = Vec::with_capacity(out_features);
            for row in 0..out_features {
                col.push(f64::from(pair.lora_b_f32[row * pair.rank + k]));
            }
            ref_cols.push(col);
        }
    }
    if ref_cols.is_empty() {
        return Ok(Vec::new());
    }
    orthonormalize_ref_columns(ref_cols)
}

fn prepare_site(
    target: &IncludedPeftTarget,
    a_bytes: &[u8],
    b_bytes: &[u8],
    factor_scale: f64,
    max_rank: usize,
    rank_one_dirs: Option<&BTreeMap<String, Vec<f64>>>,
    ref_pairs: Option<&BTreeMap<String, ExportTensorPair>>,
) -> crate::Result<PreparedSite> {
    let (mut b_cols, a_rows) = decode_peft_factors(target, a_bytes, b_bytes, factor_scale)?;
    let ref_basis = build_site_reference_basis(
        &target.gguf_name,
        target.in_features,
        target.out_features,
        rank_one_dirs,
        ref_pairs,
    )?;
    let orthogonalized = !ref_basis.is_empty();
    if orthogonalized {
        for b_col in &mut b_cols {
            for _pass in 0..2 {
                for q in &ref_basis {
                    let proj = dot_f64(b_col, q);
                    for (dst, &qv) in b_col.iter_mut().zip(q) {
                        *dst -= proj * qv;
                    }
                }
            }
        }
    }

    if target.r_src <= max_rank {
        let energy = factor_product_frobenius_energy(&b_cols, &a_rows)?;
        return Ok(PreparedSite {
            gguf_name: target.gguf_name.clone(),
            family: target.family,
            in_features: target.in_features,
            out_features: target.out_features,
            source_rank: target.r_src,
            exported_rank: target.r_src,
            orthogonalized,
            compressed: false,
            residual_energy: energy,
            retained_energy: energy,
            retained_fraction: 1.0,
            factors: PreparedFactors::Direct { b_cols, a_rows },
        });
    }

    let qr_b = thin_qr_columns(&b_cols)?;
    let qr_a = thin_qr_columns(&a_rows)?;
    let (exported_rank, u_cols, v_cols, sigmas, residual_energy, retained_energy) =
        if qr_b.k == 0 || qr_a.k == 0 {
            (
                1,
                vec![vec![0.0_f64; target.out_features]],
                vec![vec![0.0_f64; target.in_features]],
                vec![0.0_f64],
                0.0,
                0.0,
            )
        } else {
            let mut core = vec![0.0_f64; qr_b.k * qr_a.k];
            for i in 0..qr_b.k {
                let rb_row = &qr_b.r_rows[i * target.r_src..(i + 1) * target.r_src];
                for j in 0..qr_a.k {
                    let ra_row = &qr_a.r_rows[j * target.r_src..(j + 1) * target.r_src];
                    core[i * qr_a.k + j] = dot_f64(rb_row, ra_row);
                }
            }
            let svd = jacobi_svd_f64(&core, qr_b.k, qr_a.k)?;
            let keep = svd.sigmas.len().min(max_rank);
            if keep == 0 {
                (
                    1,
                    vec![vec![0.0_f64; target.out_features]],
                    vec![vec![0.0_f64; target.in_features]],
                    vec![0.0_f64],
                    svd.total_energy,
                    0.0,
                )
            } else {
                let mut u_full = Vec::with_capacity(keep);
                let mut v_full = Vec::with_capacity(keep);
                let mut sig_keep = Vec::with_capacity(keep);
                let mut retained = 0.0_f64;
                for idx in 0..keep {
                    let sigma = svd.sigmas[idx];
                    retained += sigma * sigma;
                    let mut u_vec = vec![0.0_f64; target.out_features];
                    for (qb_col, &coeff) in qr_b.q_cols.iter().zip(&svd.u_cols[idx]) {
                        if coeff == 0.0 {
                            continue;
                        }
                        for (dst, &q) in u_vec.iter_mut().zip(qb_col) {
                            *dst += q * coeff;
                        }
                    }
                    let mut v_vec = vec![0.0_f64; target.in_features];
                    for (qa_col, &coeff) in qr_a.q_cols.iter().zip(&svd.v_cols[idx]) {
                        if coeff == 0.0 {
                            continue;
                        }
                        for (dst, &q) in v_vec.iter_mut().zip(qa_col) {
                            *dst += q * coeff;
                        }
                    }
                    u_full.push(u_vec);
                    v_full.push(v_vec);
                    sig_keep.push(sigma);
                }
                (keep, u_full, v_full, sig_keep, svd.total_energy, retained)
            }
        };

    let retained_fraction = if residual_energy > 0.0 {
        (retained_energy / residual_energy).clamp(0.0, 1.0)
    } else {
        1.0
    };

    Ok(PreparedSite {
        gguf_name: target.gguf_name.clone(),
        family: target.family,
        in_features: target.in_features,
        out_features: target.out_features,
        source_rank: target.r_src,
        exported_rank,
        orthogonalized,
        compressed: true,
        residual_energy,
        retained_energy,
        retained_fraction,
        factors: PreparedFactors::Compressed {
            u_cols,
            v_cols,
            sigmas,
        },
    })
}

fn finalize_gguf_pairs(
    sites: Vec<PreparedSite>,
    global_alpha: f32,
    export_dtype: ExportDtype,
) -> crate::Result<(Vec<ExportTensorPair>, Vec<Value>)> {
    let mut pairs = Vec::with_capacity(sites.len());
    let mut site_reports = Vec::with_capacity(sites.len());

    for site in sites {
        let rank = site.exported_rank;
        let in_f = site.in_features;
        let out_f = site.out_features;
        let rank_over_alpha = usize_to_f64(rank) / f64::from(global_alpha);

        let a_len = rank
            .checked_mul(in_f)
            .ok_or("Exported lora_a element count overflow.")?;
        let b_len = out_f
            .checked_mul(rank)
            .ok_or("Exported lora_b element count overflow.")?;
        let mut lora_a_f32 = Vec::new();
        lora_a_f32
            .try_reserve_exact(a_len)
            .map_err(|e| format!("Cannot allocate exported lora_a buffer: {e}"))?;
        let mut lora_b_f32 = Vec::new();
        lora_b_f32
            .try_reserve_exact(b_len)
            .map_err(|e| format!("Cannot allocate exported lora_b buffer: {e}"))?;

        match &site.factors {
            PreparedFactors::Direct { b_cols, a_rows } => {
                let scale = rank_over_alpha.sqrt();
                for row_vec in a_rows {
                    for &val in row_vec {
                        lora_a_f32.push(check_factor_value(
                            val * scale,
                            export_dtype,
                            "peft lora_a",
                        )?);
                    }
                }
                for row in 0..out_f {
                    for col_vec in b_cols {
                        lora_b_f32.push(check_factor_value(
                            col_vec[row] * scale,
                            export_dtype,
                            "peft lora_b",
                        )?);
                    }
                }
            }
            PreparedFactors::Compressed {
                u_cols,
                v_cols,
                sigmas,
            } => {
                lora_a_f32.resize(a_len, 0.0_f32);
                lora_b_f32.resize(b_len, 0.0_f32);
                for k in 0..rank {
                    let scale_k = (sigmas[k] * rank_over_alpha).sqrt();
                    let a_dst = &mut lora_a_f32[k * in_f..(k + 1) * in_f];
                    for (dst, &v_val) in a_dst.iter_mut().zip(&v_cols[k]) {
                        *dst = check_factor_value(v_val * scale_k, export_dtype, "peft lora_a")?;
                    }
                    for row in 0..out_f {
                        lora_b_f32[row * rank + k] = check_factor_value(
                            u_cols[k][row] * scale_k,
                            export_dtype,
                            "peft lora_b",
                        )?;
                    }
                }
            }
        }

        site_reports.push(json!({
            "target": site.gguf_name,
            "family": site.family,
            "sourceRank": site.source_rank,
            "exportedRank": rank,
            "orthogonalized": site.orthogonalized,
            "compressed": site.compressed,
            "residualEnergy": site.residual_energy,
            "retainedEnergy": site.retained_energy,
            "retainedEnergyFraction": site.retained_fraction,
        }));
        pairs.push(ExportTensorPair {
            gguf_base_name: site.gguf_name,
            in_features: in_f,
            out_features: out_f,
            rank,
            lora_a_f32,
            lora_b_f32,
        });
    }

    sort_pairs_canonically(&mut pairs);
    Ok((pairs, site_reports))
}

fn source_receipt_json(source: &PeftSource) -> Value {
    match source {
        PeftSource::LocalDirectory {
            directory: _,
            license,
            repository,
            revision,
        } => json!({
            "kind": "local-directory",
            "license": license,
            "repository": repository,
            "revision": revision,
        }),
        PeftSource::PinnedHuggingFace {
            repository,
            revision,
            license,
            config_filename,
            weights_filename,
        } => json!({
            "kind": "pinned-huggingface",
            "repository": repository,
            "revision": revision,
            "license": license,
            "configFilename": config_filename.as_deref().unwrap_or(DEFAULT_CONFIG_FILENAME),
            "weightsFilename": weights_filename.as_deref().unwrap_or(DEFAULT_WEIGHTS_FILENAME),
            "baseUrl": HUGGINGFACE_BASE_URL,
        }),
    }
}

/// Imports a native PEFT LoRA `safetensors` checkpoint, optionally orthogonalizes against
/// reference GGUF subspaces, compresses rank via exact thin QR + Jacobi SVD, and publishes
/// a verified multi-rank GGUF v3 adapter.
pub(super) fn run_import_peft_lora(request: &Request) -> crate::Result<Value> {
    run_import_peft_lora_with_dims(request, &swift_base_dims_map())
}

#[expect(
    clippy::too_many_lines,
    reason = "Orchestrates bounded PEFT intake, classification, orthogonalization, exact SVD compression, payload check, and verified GGUF publish."
)]
pub(super) fn run_import_peft_lora_with_dims(
    request: &Request,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<Value> {
    if request.operation != Operation::ImportPeftLora {
        return Err("Invalid operation dispatch for import-peft-lora.".into());
    }
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("import-peft-lora requires outputPath.")?;
    let config = request
        .peft_config
        .as_ref()
        .ok_or("import-peft-lora requires peftConfig.")?;
    validate_import_peft_config(config)?;

    let started = Instant::now();
    let deadline = started
        .checked_add(Duration::from_secs(request.controls.time_budget_seconds))
        .ok_or("PEFT import deadline overflow.")?;

    let envelope = config
        .target_envelope
        .unwrap_or(TargetEnvelope::NpuFriendly);
    let max_rank = config.max_rank.unwrap_or(DEFAULT_MAX_RANK);
    let export_dtype = config.export_dtype.unwrap_or(ExportDtype::F16);

    let rank_one_dirs = config
        .orthogonalize_against_rank_one_path
        .as_ref()
        .map(|p| read_rank_one_b_directions(p))
        .transpose()?;
    let ref_adapter = config
        .orthogonalize_against_adapter_path
        .as_ref()
        .map(|p| read_lora_gguf_pairs_with_dims(p, base_dims))
        .transpose()?;
    let ref_pairs: Option<BTreeMap<String, ExportTensorPair>> = ref_adapter.map(|loaded| {
        loaded
            .pairs
            .into_iter()
            .map(|p| (p.gguf_base_name.clone(), p))
            .collect()
    });

    let mut intake =
        PeftDonorIntake::open(&config.source, request.controls.http_concurrency, deadline)?;
    let peft_cfg = parse_peft_adapter_config(&intake.config_bytes)?;
    let (included, dropped) = classify_peft_targets(&intake.header, envelope, base_dims)?;

    let mut slice_requests = Vec::with_capacity(included.len() * 2);
    for target in &included {
        slice_requests.push((target.a_abs_begin, target.a_abs_end));
        slice_requests.push((target.b_abs_begin, target.b_abs_end));
    }
    let slice_buffers = intake.fetch_all_slices(&slice_requests, deadline)?;

    let factor_scale = peft_cfg.scale_peft.sqrt();
    let mut prepared_sites = Vec::with_capacity(included.len());
    let mut total_residual_energy = 0.0_f64;
    let mut total_retained_energy = 0.0_f64;
    let mut compressed_sites = 0_usize;
    let mut orthogonalized_sites = 0_usize;

    for (idx, target) in included.iter().enumerate() {
        crate::startup_cancel::checkpoint()?;
        let a_bytes = &slice_buffers[idx * 2];
        let b_bytes = &slice_buffers[idx * 2 + 1];
        let site = prepare_site(
            target,
            a_bytes,
            b_bytes,
            factor_scale,
            max_rank,
            rank_one_dirs.as_ref(),
            ref_pairs.as_ref(),
        )?;
        if site.compressed {
            compressed_sites += 1;
        }
        if site.orthogonalized {
            orthogonalized_sites += 1;
        }
        total_residual_energy += site.residual_energy;
        total_retained_energy += site.retained_energy;
        prepared_sites.push(site);
    }

    let aggregate_retained_fraction = if total_residual_energy > 0.0 {
        (total_retained_energy / total_residual_energy).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let max_exported_rank = prepared_sites
        .iter()
        .map(|s| s.exported_rank)
        .max()
        .unwrap_or(1);
    let global_alpha = usize_to_f32(max_exported_rank);
    let (final_pairs, site_reports) =
        finalize_gguf_pairs(prepared_sites, global_alpha, export_dtype)?;

    let elem_bytes = u64::try_from(export_dtype.byte_width()).map_err(|e| e.to_string())?;
    let mut total_factor_payload_bytes = 0_u64;
    for pair in &final_pairs {
        let pair_elems = u64::try_from(pair.in_features + pair.out_features)
            .map_err(|e| e.to_string())?
            .checked_mul(u64::try_from(pair.rank).map_err(|e| e.to_string())?)
            .ok_or("PEFT exported pair element count overflow.")?;
        let pair_bytes = pair_elems
            .checked_mul(elem_bytes)
            .ok_or("PEFT exported pair payload byte count overflow.")?;
        total_factor_payload_bytes = total_factor_payload_bytes
            .checked_add(pair_bytes)
            .ok_or("PEFT total exported payload byte count overflow.")?;
    }
    if let Some(ceiling) = config.max_payload_bytes
        && total_factor_payload_bytes > ceiling
    {
        return Err(format!(
            "Imported PEFT factor payload {total_factor_payload_bytes} bytes exceeds peftConfig.maxPayloadBytes ceiling {ceiling} bytes."
        ));
    }

    let (donor_repo, donor_revision, donor_license) = match &config.source {
        PeftSource::LocalDirectory {
            license,
            repository,
            revision,
            ..
        } => (repository.as_deref(), revision.as_deref(), license.as_str()),
        PeftSource::PinnedHuggingFace {
            repository,
            revision,
            license,
            ..
        } => (
            Some(repository.as_str()),
            Some(revision.as_str()),
            license.as_str(),
        ),
    };
    let rslora_str = if peft_cfg.use_rslora { "true" } else { "false" };
    let source_rank_str = peft_cfg.r.to_string();
    let mut meta: Vec<(&str, &str)> = vec![
        ("bonsai.composition.method", IMPORT_METHOD),
        ("bonsai.composition.envelope", envelope.as_str()),
        ("bonsai.composition.donor_license", donor_license),
        ("bonsai.composition.peft_rslora", rslora_str),
        ("bonsai.composition.peft_source_rank", &source_rank_str),
    ];
    if let Some(repo) = donor_repo {
        meta.push(("bonsai.composition.donor_repo", repo));
    }
    if let Some(rev) = donor_revision {
        meta.push(("bonsai.composition.donor_revision", rev));
    }

    let written = write_lora_gguf(output_path, export_dtype, global_alpha, &meta, &final_pairs)?;
    let verification = verify_published_adapter(
        output_path,
        &written.sha256,
        written.bytes,
        max_exported_rank,
        final_pairs.len(),
        base_dims,
    )?;

    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let transferred_bytes = intake.transferred_bytes.load(Ordering::Acquire);

    Ok(json!({
        "schemaVersion": 1,
        "operation": "import-peft-lora",
        "published": true,
        "status": "published-and-verified",
        "elapsedMilliseconds": elapsed_ms,
        "source": source_receipt_json(&config.source),
        "targetEnvelope": envelope,
        "exportDtype": export_dtype,
        "alpha": global_alpha,
        "minRank": verification["minRank"],
        "maxRank": verification["maxRank"],
        "peftSourceRank": peft_cfg.r,
        "peftLoraAlpha": peft_cfg.lora_alpha,
        "peftUseRslora": peft_cfg.use_rslora,
        "peftEffectiveScale": peft_cfg.scale_peft,
        "bytes": written.bytes,
        "factorPayloadBytes": total_factor_payload_bytes,
        "maxPayloadBytes": config.max_payload_bytes,
        "sha256": written.sha256,
        "tensorCount": written.tensor_count,
        "pairCount": final_pairs.len(),
        "compressedSiteCount": compressed_sites,
        "orthogonalizedSiteCount": orthogonalized_sites,
        "retainedEnergyFraction": aggregate_retained_fraction,
        "droppedCounts": {
            "droppedNonNpuTargets": dropped.non_npu_targets,
            "droppedGroupedSsmTargets": dropped.grouped_ssm_targets,
            "droppedVisionMmprojTargets": dropped.vision_mmproj_targets,
            "droppedOtherTargets": dropped.other_targets,
        },
        "transferredBytes": transferred_bytes,
        "sites": site_reports,
        "verification": verification,
    }))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write as _;
    use std::path::Path;

    use super::*;
    use crate::bounded_input::ScratchDir;

    #[expect(
        clippy::as_conversions,
        reason = "Test helper narrows deterministic f64 values to f32."
    )]
    #[inline]
    const fn f64_to_f32(v: f64) -> f32 {
        v as f32
    }

    fn encode_f32_le(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for &v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn encode_bf16_le(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            let bits = v.to_bits();
            let rounding = 0x7FFF_u32 + ((bits >> 16) & 1);
            let rounded = bits.wrapping_add(rounding);
            let hi = u16::try_from(rounded >> 16).unwrap_or(0);
            out.extend_from_slice(&hi.to_le_bytes());
        }
        out
    }

    fn encode_f16_le_slice(values: &[f32]) -> crate::Result<Vec<u8>> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            out.extend_from_slice(&super::super::gguf_adapter::encode_f16_le(v)?);
        }
        Ok(out)
    }

    fn write_peft_safetensors(
        path: &Path,
        tensors: &[(&str, &str, Vec<usize>, Vec<u8>)],
    ) -> crate::Result<()> {
        let mut header_map = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let begin = u64::try_from(payload.len()).map_err(|e| e.to_string())?;
            payload.extend_from_slice(bytes);
            let end = u64::try_from(payload.len()).map_err(|e| e.to_string())?;
            header_map.insert(
                (*name).to_owned(),
                json!({
                    "dtype": *dtype,
                    "shape": shape,
                    "data_offsets": [begin, end],
                }),
            );
        }
        let header_bytes =
            serde_json::to_vec(&Value::Object(header_map)).map_err(|e| e.to_string())?;
        let header_len = u64::try_from(header_bytes.len()).map_err(|e| e.to_string())?;
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        file.write_all(&header_len.to_le_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(&header_bytes).map_err(|e| e.to_string())?;
        file.write_all(&payload).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn reconstruct_gguf_delta(pair: &ExportTensorPair, alpha: f32) -> Vec<f64> {
        let scale = f64::from(alpha) / usize_to_f64(pair.rank);
        let mut out = vec![0.0_f64; pair.out_features * pair.in_features];
        for r in 0..pair.out_features {
            for c in 0..pair.in_features {
                let mut sum = 0.0_f64;
                for k in 0..pair.rank {
                    sum += f64::from(pair.lora_b_f32[r * pair.rank + k])
                        * f64::from(pair.lora_a_f32[k * pair.in_features + c]);
                }
                out[r * pair.in_features + c] = scale * sum;
            }
        }
        out
    }

    fn dense_svd_via_gram(mat: &[f64], rows: usize, cols: usize) -> crate::Result<(Vec<f64>, f64)> {
        let dim = rows.min(cols);
        let mut gram = vec![0.0_f64; dim * dim];
        if rows <= cols {
            for i in 0..rows {
                for j in i..rows {
                    let dot: f64 = (0..cols)
                        .map(|c| mat[i * cols + c] * mat[j * cols + c])
                        .sum();
                    gram[i * dim + j] = dot;
                    gram[j * dim + i] = dot;
                }
            }
        } else {
            for i in 0..cols {
                for j in i..cols {
                    let dot: f64 = (0..rows)
                        .map(|r| mat[r * cols + i] * mat[r * cols + j])
                        .sum();
                    gram[i * dim + j] = dot;
                    gram[j * dim + i] = dot;
                }
            }
        }
        let total: f64 = mat.iter().map(|&v| v * v).sum();
        let evals = super::super::svd::jacobi_eigenvalues_f64(&mut gram, dim)?;
        Ok((evals.into_iter().map(f64::sqrt).collect(), total))
    }

    #[test]
    fn peft_config_and_request_validation_rejects_unsafe_inputs() -> crate::Result<()> {
        assert!(parse_peft_adapter_config(b"").is_err());
        assert!(
            parse_peft_adapter_config(br#"{"peft_type":"PREFIX_TUNING","r":16,"lora_alpha":32}"#)
                .is_err()
        );
        assert!(
            parse_peft_adapter_config(br#"{"peft_type":"LORA","r":0,"lora_alpha":32}"#).is_err()
        );
        assert!(
            parse_peft_adapter_config(br#"{"peft_type":"LORA","r":257,"lora_alpha":32}"#).is_err()
        );
        assert!(
            parse_peft_adapter_config(br#"{"peft_type":"LORA","r":16,"lora_alpha":0}"#).is_err()
        );
        assert!(
            parse_peft_adapter_config(
                br#"{"peft_type":"LORA","r":16,"lora_alpha":32,"use_dora":true}"#
            )
            .is_err()
        );
        assert!(
            parse_peft_adapter_config(
                br#"{"peft_type":"LORA","r":16,"lora_alpha":32,"bias":"all"}"#
            )
            .is_err()
        );
        assert!(
            parse_peft_adapter_config(
                br#"{"peft_type":"LORA","r":16,"lora_alpha":32,"fan_in_fan_out":true}"#
            )
            .is_err()
        );

        let parsed_std = parse_peft_adapter_config(
            br#"{"peft_type":"LORA","r":16,"lora_alpha":32,"bias":"none","use_dora":false,"extra":1}"#,
        )?;
        assert_eq!(parsed_std.r, 16);
        assert!(!parsed_std.use_rslora);
        assert!((parsed_std.scale_peft - 2.0).abs() < 1e-12);

        let parsed_rs = parse_peft_adapter_config(
            br#"{"peft_type":"LORA","r":16,"lora_alpha":32,"bias":null,"use_rslora":true}"#,
        )?;
        assert!(parsed_rs.use_rslora);
        assert!((parsed_rs.scale_peft - 8.0).abs() < 1e-12);

        let scratch = ScratchDir::new("Bonsai-PeftValidate")?;
        let rev = "0123456789abcdef0123456789abcdef01234567";
        let valid_hf = json!({
            "kind": "pinned-huggingface",
            "repository": "harness-hub/Qwen3.8-27B-pi-adapted",
            "revision": rev,
            "license": "Apache-2.0",
        });

        // Reject targetEnvelope = "all", maxRank = 0/65, maxPayloadBytes = 0, bad filenames, and cross-op fields.
        for bad_peft in [
            json!({ "source": valid_hf.clone(), "targetEnvelope": "all" }),
            json!({ "source": valid_hf.clone(), "maxRank": 0 }),
            json!({ "source": valid_hf.clone(), "maxRank": 65 }),
            json!({ "source": valid_hf.clone(), "maxPayloadBytes": 0 }),
            json!({ "source": { "kind": "pinned-huggingface", "repository": "../bad", "revision": rev, "license": "Apache-2.0" } }),
            json!({ "source": { "kind": "pinned-huggingface", "repository": "owner/name", "revision": "XYZ", "license": "Apache-2.0" } }),
            json!({ "source": { "kind": "pinned-huggingface", "repository": "owner/name", "revision": rev, "license": "  " } }),
            json!({ "source": { "kind": "pinned-huggingface", "repository": "owner/name", "revision": rev, "license": "Apache-2.0", "configFilename": "sub/adapter_config.json" } }),
            json!({ "source": { "kind": "pinned-huggingface", "repository": "owner/name", "revision": rev, "license": "Apache-2.0", "weightsFilename": "model.bin" } }),
            json!({ "source": valid_hf, "unknownField": 1 }),
        ] {
            let out = scratch.join("rejected.gguf");
            assert!(
                super::super::run(json!({
                    "schemaVersion": 1,
                    "operation": "import-peft-lora",
                    "outputPath": out,
                    "peftConfig": bad_peft,
                }))
                .is_err()
            );
        }

        Ok(())
    }

    #[test]
    fn peft_import_classifies_drops_compresses_exact_svd_and_supports_rslora() -> crate::Result<()>
    {
        let scratch = ScratchDir::new("Bonsai-PeftImportSvd")?;
        let mut toy_dims = BTreeMap::new();
        toy_dims.insert("blk.0.ffn_down.weight".to_owned(), (24, 18, "ffn_down"));
        toy_dims.insert("blk.0.ffn_gate.weight".to_owned(), (18, 24, "ffn_gate"));
        toy_dims.insert("blk.0.ffn_up.weight".to_owned(), (18, 24, "ffn_up"));
        toy_dims.insert("blk.0.ssm_out.weight".to_owned(), (20, 18, "ssm_out"));
        toy_dims.insert(
            "blk.3.attn_output.weight".to_owned(),
            (20, 18, "attn_output"),
        );

        let r_src = 12_usize;
        let max_rank = 4_usize;

        // Build full-rank rank-12 PEFT factors for blk.0.ffn_down (24 -> 18) and blk.3.attn_output (20 -> 18).
        let make_a = |in_f: usize, r: usize, seed: usize| -> Vec<f32> {
            (0..r * in_f)
                .map(|i| {
                    let row = i / in_f;
                    let col = i % in_f;
                    f64_to_f32(
                        (usize_to_f64((row + 1) * (col + 3) + seed) * 0.31).sin()
                            / (usize_to_f64(row) + 1.0),
                    )
                })
                .collect()
        };
        let make_b = |out_f: usize, r: usize, seed: usize| -> Vec<f32> {
            (0..out_f * r)
                .map(|i| {
                    let row = i / r;
                    let col = i % r;
                    f64_to_f32(
                        (usize_to_f64((row + 2) * (col + 5) + seed * 7) * 0.43).cos()
                            / (usize_to_f64(col) + 1.0),
                    )
                })
                .collect()
        };

        let a0 = make_a(24, r_src, 11);
        let b0 = make_b(18, r_src, 17);
        let a3 = make_a(20, r_src, 23);
        let b3 = make_b(18, r_src, 29);

        let peft_dir = scratch.join("donor-rslora");
        fs::create_dir(&peft_dir).map_err(|e| e.to_string())?;
        fs::write(
            peft_dir.join("adapter_config.json"),
            br#"{"peft_type":"LORA","r":12,"lora_alpha":24.0,"use_rslora":true,"bias":"none"}"#,
        )
        .map_err(|e| e.to_string())?;

        // Mix BF16, F16, F32, both key prefixes, `.default.weight`, and all 4 dropped target categories.
        write_peft_safetensors(
            &peft_dir.join("adapter_model.safetensors"),
            &[
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight",
                    "F32",
                    vec![r_src, 24],
                    encode_f32_le(&a0),
                ),
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight",
                    "F32",
                    vec![18, r_src],
                    encode_f32_le(&b0),
                ),
                (
                    "model.layers.3.self_attn.o_proj.lora_A.default.weight",
                    "BF16",
                    vec![r_src, 20],
                    encode_bf16_le(&a3),
                ),
                (
                    "model.layers.3.self_attn.o_proj.lora_B.default.weight",
                    "F16",
                    vec![18, r_src],
                    encode_f16_le_slice(&b3)?,
                ),
                // Dropped non-NPU (ffn_gate, ffn_up)
                (
                    "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
                    "BF16",
                    vec![r_src, 18],
                    encode_bf16_le(&vec![0.1_f32; r_src * 18]),
                ),
                (
                    "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight",
                    "BF16",
                    vec![24, r_src],
                    encode_bf16_le(&vec![0.1_f32; 24 * r_src]),
                ),
                (
                    "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight",
                    "BF16",
                    vec![r_src, 18],
                    encode_bf16_le(&vec![0.1_f32; r_src * 18]),
                ),
                (
                    "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight",
                    "BF16",
                    vec![24, r_src],
                    encode_bf16_le(&vec![0.1_f32; 24 * r_src]),
                ),
                // Dropped grouped SSM out
                (
                    "base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_A.weight",
                    "BF16",
                    vec![r_src, 20],
                    encode_bf16_le(&vec![0.1_f32; r_src * 20]),
                ),
                (
                    "base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_B.weight",
                    "BF16",
                    vec![18, r_src],
                    encode_bf16_le(&vec![0.1_f32; 18 * r_src]),
                ),
                // Dropped vision / mmproj
                (
                    "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight",
                    "BF16",
                    vec![r_src, 16],
                    encode_bf16_le(&vec![0.1_f32; r_src * 16]),
                ),
                (
                    "base_model.model.model.visual.blocks.0.attn.qkv.lora_B.weight",
                    "BF16",
                    vec![16, r_src],
                    encode_bf16_le(&vec![0.1_f32; 16 * r_src]),
                ),
                (
                    "base_model.model.mm_projector.linear_1.lora_A.weight",
                    "BF16",
                    vec![r_src, 16],
                    encode_bf16_le(&vec![0.1_f32; r_src * 16]),
                ),
                (
                    "base_model.model.mm_projector.linear_1.lora_B.weight",
                    "BF16",
                    vec![16, r_src],
                    encode_bf16_le(&vec![0.1_f32; 16 * r_src]),
                ),
                // Dropped other (embed_tokens, lm_head, ssm_alpha)
                (
                    "base_model.model.model.language_model.embed_tokens.lora_A.weight",
                    "BF16",
                    vec![r_src, 32],
                    encode_bf16_le(&vec![0.1_f32; r_src * 32]),
                ),
                (
                    "base_model.model.model.language_model.embed_tokens.lora_B.weight",
                    "BF16",
                    vec![18, r_src],
                    encode_bf16_le(&vec![0.1_f32; 18 * r_src]),
                ),
                (
                    "base_model.model.lm_head.lora_A.weight",
                    "BF16",
                    vec![r_src, 18],
                    encode_bf16_le(&vec![0.1_f32; r_src * 18]),
                ),
                (
                    "base_model.model.lm_head.lora_B.weight",
                    "BF16",
                    vec![32, r_src],
                    encode_bf16_le(&vec![0.1_f32; 32 * r_src]),
                ),
            ],
        )?;

        let out_gguf = scratch.join("imported-r4.gguf");
        let req: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "import-peft-lora",
            "outputPath": out_gguf,
            "peftConfig": {
                "source": {
                    "kind": "local-directory",
                    "directory": peft_dir,
                    "license": "Apache-2.0",
                    "repository": "harness-hub/Qwen3.8-27B-pi-adapted",
                    "revision": "0123456789abcdef0123456789abcdef01234567"
                },
                "targetEnvelope": "npu-friendly",
                "maxRank": max_rank,
                "exportDtype": "f32"
            }
        }))
        .map_err(|e| e.to_string())?;
        super::super::validate_request(&req)?;
        let res = run_import_peft_lora_with_dims(&req, &toy_dims)?;

        assert_eq!(res["published"], true);
        assert_eq!(res["pairCount"], 2);
        assert_eq!(res["compressedSiteCount"], 2);
        assert_eq!(res["maxRank"], 4);
        assert_eq!(res["peftUseRslora"], true);
        assert_eq!(res["droppedCounts"]["droppedNonNpuTargets"], 2);
        assert_eq!(res["droppedCounts"]["droppedGroupedSsmTargets"], 1);
        assert_eq!(res["droppedCounts"]["droppedVisionMmprojTargets"], 2);
        assert_eq!(res["droppedCounts"]["droppedOtherTargets"], 2);
        assert_eq!(res["verification"]["verified"], true);
        assert_eq!(
            res["verification"]["provenanceMetadata"]["bonsai.composition.method"],
            IMPORT_METHOD
        );
        assert_eq!(
            res["verification"]["provenanceMetadata"]["bonsai.composition.peft_rslora"],
            "true"
        );
        assert_eq!(
            res["verification"]["provenanceMetadata"]["bonsai.composition.peft_source_rank"],
            "12"
        );
        assert_eq!(
            res["verification"]["provenanceMetadata"]["bonsai.composition.donor_license"],
            "Apache-2.0"
        );

        // Verify exact rank-4 Eckart-Young approximation error against dense F64 SVD on blk.0.ffn_down.
        let rslora_scale = 24.0_f64 / usize_to_f64(r_src).sqrt();
        let mut dense0 = vec![0.0_f64; 18 * 24];
        for r in 0..18 {
            for c in 0..24 {
                let mut sum = 0.0_f64;
                for k in 0..r_src {
                    sum += f64::from(b0[r * r_src + k]) * f64::from(a0[k * 24 + c]);
                }
                dense0[r * 24 + c] = rslora_scale * sum;
            }
        }
        let (svs, total_energy) = dense_svd_via_gram(&dense0, 18, 24)?;
        let expected_retained: f64 = svs.iter().take(4).map(|&s| s * s).sum();
        let expected_tail: f64 = svs.iter().skip(4).map(|&s| s * s).sum();
        let expected_frac = expected_retained / total_energy;

        let loaded = read_lora_gguf_pairs_with_dims(&out_gguf, &toy_dims)?;
        let exported0 = &loaded.pairs[0];
        assert_eq!(exported0.gguf_base_name, "blk.0.ffn_down.weight");
        let recon0 = reconstruct_gguf_delta(exported0, loaded.alpha);
        let diff_sq: f64 = dense0
            .iter()
            .zip(&recon0)
            .map(|(&d, &r)| (d - r) * (d - r))
            .sum();
        assert!(
            (diff_sq - expected_tail).abs() / total_energy < 1e-10,
            "Compressed F32 SVD residual energy {diff_sq} vs exact tail {expected_tail}"
        );
        let site0_frac = res["sites"][0]["retainedEnergyFraction"]
            .as_f64()
            .ok_or("Missing site 0 retainedEnergyFraction.")?;
        assert!((site0_frac - expected_frac).abs() < 1e-10);

        // Full-linear envelope must admit ffn_gate and ffn_up with droppedNonNpuTargets == 0.
        let full_out = scratch.join("imported-full-linear.gguf");
        let full_req: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "import-peft-lora",
            "outputPath": full_out,
            "peftConfig": {
                "source": {
                    "kind": "local-directory",
                    "directory": peft_dir,
                    "license": "Apache-2.0"
                },
                "targetEnvelope": "full-linear",
                "maxRank": 16,
                "exportDtype": "f16"
            }
        }))
        .map_err(|e| e.to_string())?;
        let full_res = run_import_peft_lora_with_dims(&full_req, &toy_dims)?;
        assert_eq!(full_res["pairCount"], 4);
        assert_eq!(full_res["compressedSiteCount"], 0);
        assert_eq!(full_res["maxRank"], 12);
        assert_eq!(full_res["droppedCounts"]["droppedNonNpuTargets"], 0);
        assert_eq!(full_res["droppedCounts"]["droppedGroupedSsmTargets"], 1);
        assert!((full_res["retainedEnergyFraction"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-12);

        Ok(())
    }

    #[test]
    fn peft_import_orthogonalizes_against_rank_one_and_multi_rank_before_compression()
    -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-PeftOrthogonalize")?;
        let mut toy_dims = BTreeMap::new();
        toy_dims.insert("blk.0.ffn_down.weight".to_owned(), (16, 12, "ffn_down"));

        // Reference multi-rank GGUF spanning basis vectors e_0 and e_1 in R^12.
        let ref_gguf = scratch.join("ref-r2.gguf");
        let mut ref_a = vec![0.0_f32; 2 * 16];
        let mut ref_b = vec![0.0_f32; 12 * 2];
        ref_a[0] = 1.0;
        ref_a[16 + 1] = 1.0;
        ref_b[0] = 1.0; // row 0, col 0
        ref_b[2 + 1] = 1.0; // row 1, col 1
        write_lora_gguf(
            &ref_gguf,
            ExportDtype::F32,
            2.0,
            &[],
            &[ExportTensorPair {
                gguf_base_name: "blk.0.ffn_down.weight".to_owned(),
                in_features: 16,
                out_features: 12,
                rank: 2,
                lora_a_f32: ref_a,
                lora_b_f32: ref_b,
            }],
        )?;

        // Donor r=6 with components in e_0, e_1 (to be projected out) plus orthogonal directions e_2..e_5.
        let r_src = 6_usize;
        let mut a_vals = vec![0.0_f32; r_src * 16];
        let mut b_vals = vec![0.0_f32; 12 * r_src];
        for k in 0..r_src {
            a_vals[k * 16 + k] = 1.0;
            a_vals[k * 16 + (k + 6)] = 0.5;
            b_vals[k] = 3.0; // heavy energy in e_0 (projected out!)
            b_vals[12 + k] = 2.0; // heavy energy in e_1 (projected out!)
            b_vals[(k + 2) * r_src + k] = 1.0 / (usize_to_f32(k) + 1.0);
        }

        let peft_dir = scratch.join("donor-orth");
        fs::create_dir(&peft_dir).map_err(|e| e.to_string())?;
        fs::write(
            peft_dir.join("adapter_config.json"),
            br#"{"peft_type":"LORA","r":6,"lora_alpha":6.0,"use_rslora":false}"#,
        )
        .map_err(|e| e.to_string())?;
        write_peft_safetensors(
            &peft_dir.join("adapter_model.safetensors"),
            &[
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight",
                    "F32",
                    vec![r_src, 16],
                    encode_f32_le(&a_vals),
                ),
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight",
                    "F32",
                    vec![12, r_src],
                    encode_f32_le(&b_vals),
                ),
            ],
        )?;

        let out_gguf = scratch.join("orth-compressed.gguf");
        let req: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "import-peft-lora",
            "outputPath": out_gguf,
            "peftConfig": {
                "source": {
                    "kind": "local-directory",
                    "directory": peft_dir,
                    "license": "MIT"
                },
                "targetEnvelope": "npu-friendly",
                "maxRank": 4,
                "exportDtype": "f32",
                "orthogonalizeAgainstAdapterPath": ref_gguf
            }
        }))
        .map_err(|e| e.to_string())?;
        let res = run_import_peft_lora_with_dims(&req, &toy_dims)?;
        assert_eq!(res["published"], true);
        assert_eq!(res["orthogonalizedSiteCount"], 1);
        assert_eq!(res["compressedSiteCount"], 1);
        assert_eq!(res["maxRank"], 4);

        let loaded = read_lora_gguf_pairs_with_dims(&out_gguf, &toy_dims)?;
        let pair = &loaded.pairs[0];
        // Rows 0 and 1 (e_0, e_1) must be orthogonalized to zero across every exported column.
        for k in 0..pair.rank {
            assert!(pair.lora_b_f32[k].abs() < 1e-6);
            assert!(pair.lora_b_f32[pair.rank + k].abs() < 1e-6);
        }

        // Also test combined real-Swift-dims rank-1 Orca orthogonalization + maxPayloadBytes ceiling.
        let orca_gguf = scratch.join("orca-r1.gguf");
        let mut orca_b = vec![0.0_f32; 5_120];
        orca_b[0] = 1.0;
        write_lora_gguf(
            &orca_gguf,
            ExportDtype::F32,
            1.0,
            &[],
            &[ExportTensorPair {
                gguf_base_name: "blk.0.ffn_down.weight".to_owned(),
                in_features: 17_408,
                out_features: 5_120,
                rank: 1,
                lora_a_f32: vec![0.1_f32; 17_408],
                lora_b_f32: orca_b,
            }],
        )?;

        let real_peft = scratch.join("donor-real-swift");
        fs::create_dir(&real_peft).map_err(|e| e.to_string())?;
        fs::write(
            real_peft.join("adapter_config.json"),
            br#"{"peft_type":"LORA","r":2,"lora_alpha":4.0}"#,
        )
        .map_err(|e| e.to_string())?;
        let mut real_a = vec![0.0_f32; 2 * 17_408];
        let mut real_b = vec![0.0_f32; 5_120 * 2];
        real_a[0] = 0.25;
        real_a[17_408 + 1] = 0.25;
        real_b[0] = 1.5; // aligned with Orca e_0
        real_b[2 + 1] = 0.75; // orthogonal e_1
        write_peft_safetensors(
            &real_peft.join("adapter_model.safetensors"),
            &[
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight",
                    "F16",
                    vec![2, 17_408],
                    encode_f16_le_slice(&real_a)?,
                ),
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight",
                    "F16",
                    vec![5_120, 2],
                    encode_f16_le_slice(&real_b)?,
                ),
            ],
        )?;

        // Exact F16 factor payload for 1 pair at r=2: 2 * (17408 + 5120) * 2 = 90112 bytes.
        let exact_bytes = 90_112_u64;
        assert!(
            super::super::run(json!({
                "schemaVersion": 1,
                "operation": "import-peft-lora",
                "outputPath": scratch.join("over-budget.gguf"),
                "peftConfig": {
                    "source": {
                        "kind": "local-directory",
                        "directory": real_peft,
                        "license": "Apache-2.0"
                    },
                    "maxRank": 16,
                    "exportDtype": "f16",
                    "maxPayloadBytes": exact_bytes - 1
                }
            }))
            .is_err()
        );

        let real_out = scratch.join("real-swift-imported.gguf");
        let real_res = super::super::run(json!({
            "schemaVersion": 1,
            "operation": "import-peft-lora",
            "outputPath": real_out,
            "peftConfig": {
                "source": {
                    "kind": "local-directory",
                    "directory": real_peft,
                    "license": "Apache-2.0",
                    "repository": "Recover-LoRA/Qwen3.8-27B-recover",
                    "revision": "fedcba9876543210fedcba9876543210fedcba98"
                },
                "maxRank": 16,
                "exportDtype": "f16",
                "maxPayloadBytes": exact_bytes,
                "orthogonalizeAgainstRankOnePath": orca_gguf
            }
        }))?;
        assert_eq!(real_res["published"], true);
        assert_eq!(real_res["factorPayloadBytes"], exact_bytes);
        assert_eq!(real_res["orthogonalizedSiteCount"], 1);
        assert_eq!(real_res["verification"]["verified"], true);

        let real_loaded = super::super::gguf_adapter::read_lora_gguf_pairs(&real_out)?;
        for k in 0..real_loaded.pairs[0].rank {
            assert!(real_loaded.pairs[0].lora_b_f32[k].abs() < 1e-4);
        }

        Ok(())
    }

    #[test]
    fn peft_import_rejects_malformed_keys_duplicates_and_rank_deficient_handled_cleanly()
    -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-PeftMalformed")?;
        let peft_dir = scratch.join("bad-keys");
        fs::create_dir(&peft_dir).map_err(|e| e.to_string())?;
        fs::write(
            peft_dir.join("adapter_config.json"),
            br#"{"peft_type":"LORA","r":4,"lora_alpha":8.0}"#,
        )
        .map_err(|e| e.to_string())?;

        // 1. Non-LoRA key rejected.
        write_peft_safetensors(
            &peft_dir.join("adapter_model.safetensors"),
            &[(
                "base_model.model.model.language_model.layers.0.mlp.down_proj.weight",
                "F32",
                vec![4, 16],
                encode_f32_le(&vec![0.0_f32; 64]),
            )],
        )?;
        assert!(
            super::super::run(json!({
                "schemaVersion": 1,
                "operation": "import-peft-lora",
                "outputPath": scratch.join("out1.gguf"),
                "peftConfig": { "source": { "kind": "local-directory", "directory": peft_dir, "license": "Apache-2.0" } }
            }))
            .is_err()
        );

        // 2. Duplicate .lora_A.weight + .lora_A.default.weight rejected.
        write_peft_safetensors(
            &peft_dir.join("adapter_model.safetensors"),
            &[
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight",
                    "F32",
                    vec![4, 17_408],
                    encode_f32_le(&vec![0.0_f32; 4 * 17_408]),
                ),
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.default.weight",
                    "F32",
                    vec![4, 17_408],
                    encode_f32_le(&vec![0.0_f32; 4 * 17_408]),
                ),
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight",
                    "F32",
                    vec![5_120, 4],
                    encode_f32_le(&vec![0.0_f32; 5_120 * 4]),
                ),
            ],
        )?;
        assert!(
            super::super::run(json!({
                "schemaVersion": 1,
                "operation": "import-peft-lora",
                "outputPath": scratch.join("out2.gguf"),
                "peftConfig": { "source": { "kind": "local-directory", "directory": peft_dir, "license": "Apache-2.0" } }
            }))
            .is_err()
        );

        // 3. Rank-deficient r_src=8 -> maxRank=4 (effective rank 2) compresses cleanly without NaN.
        let mut toy_dims = BTreeMap::new();
        toy_dims.insert("blk.0.ffn_down.weight".to_owned(), (16, 12, "ffn_down"));
        let mut low_a = vec![0.0_f32; 8 * 16];
        let mut low_b = vec![0.0_f32; 12 * 8];
        for c in 0..16 {
            low_a[c] = f64_to_f32(usize_to_f64(c + 1) * 0.1);
            low_a[16 + c] = f64_to_f32(usize_to_f64(16 - c) * 0.1);
        }
        for r in 0..12 {
            low_b[r * 8] = 0.5;
            low_b[r * 8 + 1] = if r % 2 == 0 { 0.3 } else { -0.3 };
        }
        write_peft_safetensors(
            &peft_dir.join("adapter_model.safetensors"),
            &[
                (
                    "model.layers.0.mlp.down_proj.lora_A.weight",
                    "F32",
                    vec![8, 16],
                    encode_f32_le(&low_a),
                ),
                (
                    "model.layers.0.mlp.down_proj.lora_B.weight",
                    "F32",
                    vec![12, 8],
                    encode_f32_le(&low_b),
                ),
            ],
        )?;
        fs::write(
            peft_dir.join("adapter_config.json"),
            br#"{"peft_type":"LORA","r":8,"lora_alpha":8.0}"#,
        )
        .map_err(|e| e.to_string())?;
        let low_out = scratch.join("rank-deficient.gguf");
        let low_req: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "import-peft-lora",
            "outputPath": low_out,
            "peftConfig": {
                "source": { "kind": "local-directory", "directory": peft_dir, "license": "Apache-2.0" },
                "maxRank": 4,
                "exportDtype": "f32"
            }
        }))
        .map_err(|e| e.to_string())?;
        let low_res = run_import_peft_lora_with_dims(&low_req, &toy_dims)?;
        assert_eq!(low_res["published"], true);
        assert_eq!(low_res["maxRank"], 2);
        assert!((low_res["retainedEnergyFraction"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn peft_import_combined_rank_one_plus_multi_rank_orth_and_residual_writers_envelope()
    -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-PeftCombinedOrth")?;
        let orca_path = scratch.join("orca-e0.gguf");
        let mut orca_b = vec![0.0_f32; 5_120];
        orca_b[0] = 1.0;
        write_lora_gguf(
            &orca_path,
            ExportDtype::F32,
            1.0,
            &[],
            &[ExportTensorPair {
                gguf_base_name: "blk.0.ffn_down.weight".to_owned(),
                in_features: 17_408,
                out_features: 5_120,
                rank: 1,
                lora_a_f32: vec![0.1_f32; 17_408],
                lora_b_f32: orca_b,
            }],
        )?;

        // Multi-rank reference spanning {e_0, e_1} (overlapping e_0 with Orca to test MGS dedup).
        let swift_ref_path = scratch.join("swift-ref-e0-e1.gguf");
        let mut ref_a = vec![0.0_f32; 2 * 17_408];
        let mut ref_b = vec![0.0_f32; 5_120 * 2];
        ref_a[0] = 0.2;
        ref_a[17_408 + 1] = 0.2;
        ref_b[0] = 1.0; // row 0, col 0 (overlaps Orca)
        ref_b[2 + 1] = 1.0; // row 1, col 1
        write_lora_gguf(
            &swift_ref_path,
            ExportDtype::F32,
            2.0,
            &[],
            &[ExportTensorPair {
                gguf_base_name: "blk.0.ffn_down.weight".to_owned(),
                in_features: 17_408,
                out_features: 5_120,
                rank: 2,
                lora_a_f32: ref_a,
                lora_b_f32: ref_b,
            }],
        )?;

        let peft_dir = scratch.join("donor-combined");
        fs::create_dir(&peft_dir).map_err(|e| e.to_string())?;
        fs::write(
            peft_dir.join("adapter_config.json"),
            br#"{"peft_type":"LORA","r":4,"lora_alpha":8.0,"use_rslora":false}"#,
        )
        .map_err(|e| e.to_string())?;

        // blk.0.ffn_down (r=4 -> maxRank=2), blk.3.attn_output (kept), blk.3.attn_q (dropped under residual-writers).
        let mut down_a = vec![0.0_f32; 4 * 17_408];
        let mut down_b = vec![0.0_f32; 5_120 * 4];
        for k in 0..4 {
            down_a[k * 17_408 + k] = 0.5;
            down_b[k] = 2.0; // e_0 (projected out)
            down_b[4 + k] = 1.5; // e_1 (projected out)
            down_b[(k + 2) * 4 + k] = 0.8 / (usize_to_f32(k) + 1.0); // orthogonal e_2..e_5
        }
        let mut attn_o_a = vec![0.0_f32; 4 * 6_144];
        let mut attn_o_b = vec![0.0_f32; 5_120 * 4];
        for k in 0..4 {
            attn_o_a[k * 6_144 + k] = 0.25;
            attn_o_b[k * 4 + k] = 0.5;
        }
        write_peft_safetensors(
            &peft_dir.join("adapter_model.safetensors"),
            &[
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight",
                    "F32",
                    vec![4, 17_408],
                    encode_f32_le(&down_a),
                ),
                (
                    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight",
                    "F32",
                    vec![5_120, 4],
                    encode_f32_le(&down_b),
                ),
                (
                    "base_model.model.model.language_model.layers.3.self_attn.o_proj.lora_A.weight",
                    "F32",
                    vec![4, 6_144],
                    encode_f32_le(&attn_o_a),
                ),
                (
                    "base_model.model.model.language_model.layers.3.self_attn.o_proj.lora_B.weight",
                    "F32",
                    vec![5_120, 4],
                    encode_f32_le(&attn_o_b),
                ),
                (
                    "base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_A.weight",
                    "F32",
                    vec![4, 5_120],
                    encode_f32_le(&vec![0.1_f32; 4 * 5_120]),
                ),
                (
                    "base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_B.weight",
                    "F32",
                    vec![12_288, 4],
                    encode_f32_le(&vec![0.1_f32; 12_288 * 4]),
                ),
            ],
        )?;

        let out_gguf = scratch.join("combined-orth-rw.gguf");
        let res = super::super::run(json!({
            "schemaVersion": 1,
            "operation": "import-peft-lora",
            "outputPath": out_gguf,
            "peftConfig": {
                "source": {
                    "kind": "local-directory",
                    "directory": peft_dir,
                    "license": "Apache-2.0"
                },
                "targetEnvelope": "residual-writers",
                "maxRank": 2,
                "exportDtype": "f16",
                "orthogonalizeAgainstRankOnePath": orca_path,
                "orthogonalizeAgainstAdapterPath": swift_ref_path
            }
        }))?;
        assert_eq!(res["published"], true);
        assert_eq!(res["pairCount"], 2);
        assert_eq!(res["orthogonalizedSiteCount"], 1);
        assert_eq!(res["compressedSiteCount"], 2);
        assert_eq!(res["droppedCounts"]["droppedOtherTargets"], 1);
        assert_eq!(res["verification"]["detectedEnvelope"], "residual-writers");
        assert_eq!(res["verification"]["verified"], true);

        let loaded = super::super::gguf_adapter::read_lora_gguf_pairs(&out_gguf)?;
        let down = &loaded.pairs[0];
        assert_eq!(down.gguf_base_name, "blk.0.ffn_down.weight");
        assert_eq!(down.rank, 2);
        for k in 0..down.rank {
            assert!(down.lora_b_f32[k].abs() < 1e-4);
            assert!(down.lora_b_f32[down.rank + k].abs() < 1e-4);
        }

        Ok(())
    }
}
