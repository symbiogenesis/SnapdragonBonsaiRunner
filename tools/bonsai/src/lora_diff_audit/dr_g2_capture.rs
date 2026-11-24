//! Pure-Rust sleep-inhibited `dr-g2-capture` runner against local `prism-inference-server` instances.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::io::{self, BufReader, Read as _, Write as _};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::gguf_adapter::{VerifyPolicy, verify_lora_gguf};
use super::readiness::{
    BaseModelProfile, BlindResponseEntry, DrG2Evidence, ThinkingOffArm, ThinkingOffPromptPair,
    ThinkingOnArm, ThinkingOnPromptPair, ZeroScaleControl, build_dr_g2_contract,
    compiled_orca_adapter_sha256, hash_contract, resolve_scale_grid,
};
use super::{Operation, Request, validate_output_path};

const MIN_TIME_BUDGET_SECONDS: u64 = 30;
const MAX_TIME_BUDGET_SECONDS: u64 = 900;
const CLEANUP_RESERVE_SECONDS: u64 = 15;
const SERVER_READY_POLL_MS: u64 = 100;
const HTTP_POLL_SLICE_MS: u64 = 250;
const MAX_HTTP_HEAD_BYTES: usize = 16 * 1024;
const MAX_CONTROL_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_CHAT_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_COMPLETION_TOKENS: u32 = 4096;
const DEFAULT_ZERO_CONTROL_MAX_TOKENS: u32 = 64;
const DEFAULT_ROW_WORKERS: u8 = 8;
const DEFAULT_KV_CACHE: &str = "rsq4";
const THINKING_ON_CATEGORIES: [&str; 3] = ["coding", "reasoning", "instruction-following"];
const THINKING_OFF_CATEGORY: &str = "creative-prose";
#[cfg(windows)]
const SERVER_STOP_TIMEOUT: Duration = Duration::from_secs(10);
#[cfg(windows)]
const SERVER_JOB_MEMORY_LIMIT_BYTES: usize = 12_348_030_976;

/// Prompt input specification for Thinking-ON or Thinking-OFF `dr-g2-capture` runs.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct CapturePromptInput {
    pub(super) prompt_id: String,
    pub(super) category: String,
    pub(super) prompt_text: String,
    #[serde(default)]
    pub(super) max_completion_tokens: Option<u32>,
}

/// Zero-scale identity control prompt specification for `dr-g2-capture`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct CaptureZeroControlInput {
    pub(super) prompt_id: String,
    pub(super) prompt_text: String,
    #[serde(default)]
    pub(super) max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub(super) enable_thinking: Option<bool>,
}

/// Configuration payload for `Operation::DrG2Capture` (`dr-g2-capture`).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct DrG2CaptureConfig {
    pub(super) server_path: PathBuf,
    pub(super) model_path: PathBuf,
    #[serde(default)]
    pub(super) orca_adapter_path: Option<PathBuf>,
    pub(super) candidate_adapter_path: PathBuf,
    #[serde(default)]
    pub(super) base_model: Option<BaseModelProfile>,
    pub(super) selected_scale: f64,
    #[serde(default)]
    pub(super) scale_grid: Option<Vec<f64>>,
    #[serde(default)]
    pub(super) candidate_max_rank: Option<u16>,
    #[serde(default)]
    pub(super) candidate_target_count: Option<usize>,
    #[serde(default)]
    pub(super) row_workers: Option<u8>,
    #[serde(default)]
    pub(super) kv_cache: Option<String>,
    #[serde(default)]
    pub(super) context_tokens: Option<usize>,
    #[serde(default)]
    pub(super) cpu_prefill_tokens: Option<u8>,
    #[serde(default)]
    pub(super) placement: Option<String>,
    #[serde(default)]
    pub(super) time_budget_seconds: Option<u64>,
    #[serde(default)]
    pub(super) responses_output_path: Option<PathBuf>,
    #[serde(default)]
    pub(super) training_prompt_sha256s: Vec<String>,
    #[serde(default)]
    pub(super) zero_scale_controls: Vec<CaptureZeroControlInput>,
    #[serde(default)]
    pub(super) thinking_on_prompts: Vec<CapturePromptInput>,
    #[serde(default)]
    pub(super) thinking_off_prompts: Vec<CapturePromptInput>,
}

#[derive(Clone, Debug)]
struct CandidateAdapterMeta {
    sha256: String,
    bytes: u64,
    max_rank: Option<u16>,
    target_count: Option<usize>,
}

struct CaptureEndpoint<'a> {
    port: u16,
    token: &'a str,
}

#[derive(Debug)]
struct ChatCompletionOutcome {
    canonical_text: String,
    response_sha256: String,
    thinking_tokens: u32,
    hit4096_truncation: bool,
    loop_detected: bool,
}

fn validate_regular_external_file(path: &Path, label: &str) -> crate::Result<()> {
    bonsai_artifacts::validate_external_path(path).map_err(|e| format!("{label}: {e}"))?;
    let meta = fs::symlink_metadata(path).map_err(|e| format!("{label}: {e}"))?;
    if !meta.is_file() || bonsai_artifacts::is_redirect(&meta) || meta.len() == 0 {
        return Err(format!(
            "{label} must refer to a non-empty regular non-reparse file."
        ));
    }
    Ok(())
}

fn parse_kv_cache_format(
    name: &str,
) -> crate::Result<bonsai_cpu_policy::context::SwiftKvCacheFormat> {
    use bonsai_cpu_policy::context::SwiftKvCacheFormat;
    match name {
        "f16" => Ok(SwiftKvCacheFormat::F16),
        "rsq4" => Ok(SwiftKvCacheFormat::Rsq4),
        "rsq4k3v" => Ok(SwiftKvCacheFormat::Rsq4K3v),
        "rsq4vf" => Ok(SwiftKvCacheFormat::Rotated4FittedValues),
        "rsq4kvf" => Ok(SwiftKvCacheFormat::Rotated4FittedKv),
        _ => Err(format!(
            "captureConfig.kvCache must be one of f16, rsq4, rsq4vf, rsq4kvf, or rsq4k3v (got '{name}')."
        )),
    }
}

fn effective_time_budget_seconds(
    request: &Request,
    config: &DrG2CaptureConfig,
) -> crate::Result<u64> {
    let budget = config
        .time_budget_seconds
        .unwrap_or(request.controls.time_budget_seconds);
    if !(MIN_TIME_BUDGET_SECONDS..=MAX_TIME_BUDGET_SECONDS).contains(&budget) {
        return Err(format!(
            "captureConfig.timeBudgetSeconds must be in {MIN_TIME_BUDGET_SECONDS}..={MAX_TIME_BUDGET_SECONDS}."
        ));
    }
    Ok(budget)
}

fn validate_prompt_id(id: &str, label: &str) -> crate::Result<()> {
    let trimmed = id.trim();
    if trimmed.is_empty() || id.len() > 128 {
        return Err(format!(
            "{label} promptId must be non-empty and <= 128 bytes."
        ));
    }
    Ok(())
}

fn validate_max_tokens(tokens: Option<u32>, default_val: u32, label: &str) -> crate::Result<u32> {
    let resolved = tokens.unwrap_or(default_val);
    if !(1..=DEFAULT_MAX_COMPLETION_TOKENS).contains(&resolved) {
        return Err(format!(
            "{label} maxCompletionTokens must be in 1..={DEFAULT_MAX_COMPLETION_TOKENS}."
        ));
    }
    Ok(resolved)
}

pub(super) fn validate_capture_request(
    request: &Request,
    config: &DrG2CaptureConfig,
    output_path: &Path,
) -> crate::Result<()> {
    let _budget = effective_time_budget_seconds(request, config)?;
    let scale_grid = resolve_scale_grid(config.scale_grid.as_ref())?;
    if !config.selected_scale.is_finite()
        || config.selected_scale <= 0.0
        || config.selected_scale > 2.0
        || !scale_grid
            .iter()
            .any(|&s| (s - config.selected_scale).abs() <= 1e-12)
    {
        return Err(
            "captureConfig.selectedScale must be finite in (0.0, 2.0] and present in scaleGrid."
                .into(),
        );
    }
    if let Some(rank) = config.candidate_max_rank
        && !(1..=64).contains(&rank)
    {
        return Err("captureConfig.candidateMaxRank must be in 1..=64.".into());
    }
    if config.candidate_target_count == Some(0) {
        return Err("captureConfig.candidateTargetCount must be positive.".into());
    }
    let profile = config.base_model.unwrap_or(BaseModelProfile::SwiftPq2);
    if profile.is_nariq() {
        if config.orca_adapter_path.is_some() {
            return Err(
                "captureConfig.orcaAdapterPath must be omitted/null for NariQ base models.".into(),
            );
        }
    } else if config.orca_adapter_path.is_none() {
        return Err("captureConfig.orcaAdapterPath is required for SwiftPq2 DR-G2 capture.".into());
    }
    let row_workers = config.row_workers.unwrap_or(DEFAULT_ROW_WORKERS);
    if !(1..=16).contains(&row_workers) {
        return Err("captureConfig.rowWorkers must be in 1..=16.".into());
    }
    let kv_cache = config.kv_cache.as_deref().unwrap_or(DEFAULT_KV_CACHE);
    let _ = parse_kv_cache_format(kv_cache)?;
    if let Some(ctx) = config.context_tokens
        && (!(4_352..=262_144).contains(&ctx) || !ctx.is_multiple_of(256))
    {
        return Err(
            "captureConfig.contextTokens must be a multiple of 256 in 4352..=262144.".into(),
        );
    }
    if let Some(prefill) = config.cpu_prefill_tokens
        && !matches!(prefill, 8 | 16 | 32)
    {
        return Err("captureConfig.cpuPrefillTokens must be 8, 16, or 32.".into());
    }
    if let Some(placement) = &config.placement {
        if !matches!(placement.as_str(), "auto" | "cpu" | "vulkan" | "composite") {
            return Err("captureConfig.placement must be auto, cpu, vulkan, or composite.".into());
        }
        if profile.is_nariq() && matches!(placement.as_str(), "vulkan" | "composite") {
            return Err("NariQ base models do not support vulkan or composite placement.".into());
        }
    }
    if let Some(resp_out) = &config.responses_output_path {
        validate_output_path(resp_out)?;
        if resp_out == output_path {
            return Err("captureConfig.responsesOutputPath must differ from outputPath.".into());
        }
    }

    let mut training_digests = BTreeSet::new();
    for digest in &config.training_prompt_sha256s {
        if !crate::identity::is_lower_hex(digest, 64) {
            return Err(
                "captureConfig.trainingPromptSha256s must contain only lowercase 64-hex SHA-256 digests."
                    .into(),
            );
        }
        training_digests.insert(digest.as_str());
    }

    if config.zero_scale_controls.is_empty() {
        return Err("captureConfig.zeroScaleControls must contain at least one prompt.".into());
    }
    if config.thinking_on_prompts.is_empty() {
        return Err(
            "captureConfig.thinkingOnPrompts must contain at least one prompt per category.".into(),
        );
    }
    if config.thinking_off_prompts.is_empty() {
        return Err(
            "captureConfig.thinkingOffPrompts must contain at least one creative-prose prompt."
                .into(),
        );
    }

    let mut prompt_ids = BTreeSet::new();
    for ctrl in &config.zero_scale_controls {
        validate_prompt_id(&ctrl.prompt_id, "Zero-scale control")?;
        if !prompt_ids.insert(ctrl.prompt_id.as_str()) {
            return Err("Duplicate promptId across captureConfig prompts.".into());
        }
        if ctrl.prompt_text.is_empty() {
            return Err("Zero-scale control promptText must not be empty.".into());
        }
        let _ = validate_max_tokens(
            ctrl.max_completion_tokens,
            DEFAULT_ZERO_CONTROL_MAX_TOKENS,
            "Zero-scale control",
        )?;
        let prompt_sha = crate::identity::sha256(ctrl.prompt_text.as_bytes());
        if training_digests.contains(prompt_sha.as_str()) {
            return Err(
                "Zero-scale control promptText overlaps trainingPromptSha256s split.".into(),
            );
        }
    }

    let mut seen_categories = BTreeSet::new();
    for prompt in &config.thinking_on_prompts {
        validate_prompt_id(&prompt.prompt_id, "Thinking-ON")?;
        if !prompt_ids.insert(prompt.prompt_id.as_str()) {
            return Err("Duplicate promptId across captureConfig prompts.".into());
        }
        if !THINKING_ON_CATEGORIES.contains(&prompt.category.as_str()) {
            return Err(
                "Thinking-ON prompt category must be coding, reasoning, or instruction-following."
                    .into(),
            );
        }
        seen_categories.insert(prompt.category.as_str());
        if prompt.prompt_text.is_empty() {
            return Err("Thinking-ON promptText must not be empty.".into());
        }
        let _ = validate_max_tokens(
            prompt.max_completion_tokens,
            DEFAULT_MAX_COMPLETION_TOKENS,
            "Thinking-ON",
        )?;
        let prompt_sha = crate::identity::sha256(prompt.prompt_text.as_bytes());
        if training_digests.contains(prompt_sha.as_str()) {
            return Err("Thinking-ON promptText overlaps trainingPromptSha256s split.".into());
        }
    }
    for cat in THINKING_ON_CATEGORIES {
        if !seen_categories.contains(cat) {
            return Err(format!(
                "captureConfig.thinkingOnPrompts is missing required category '{cat}'."
            ));
        }
    }

    for prompt in &config.thinking_off_prompts {
        validate_prompt_id(&prompt.prompt_id, "Thinking-OFF")?;
        if !prompt_ids.insert(prompt.prompt_id.as_str()) {
            return Err("Duplicate promptId across captureConfig prompts.".into());
        }
        if prompt.category != THINKING_OFF_CATEGORY {
            return Err("Thinking-OFF prompt category must be creative-prose.".into());
        }
        if prompt.prompt_text.is_empty() {
            return Err("Thinking-OFF promptText must not be empty.".into());
        }
        let _ = validate_max_tokens(
            prompt.max_completion_tokens,
            DEFAULT_MAX_COMPLETION_TOKENS,
            "Thinking-OFF",
        )?;
        let prompt_sha = crate::identity::sha256(prompt.prompt_text.as_bytes());
        if training_digests.contains(prompt_sha.as_str()) {
            return Err("Thinking-OFF promptText overlaps trainingPromptSha256s split.".into());
        }
    }

    Ok(())
}

fn calibrated_server_memory_bytes(
    profile: BaseModelProfile,
    context_tokens: usize,
    kv_format: bonsai_cpu_policy::context::SwiftKvCacheFormat,
    cpu_prefill_tokens: Option<u8>,
) -> crate::Result<(u64, Option<u64>)> {
    use bonsai_cpu_policy::context::{
        KV_COW_PAGE_TOKENS, RECURRENT_BYTES, SwiftWeightPacking, swift_tiled_prefill_scratch_bytes,
    };

    let packing = match profile.nariq_variant() {
        Some(variant) => SwiftWeightPacking::from_nariq(variant),
        None => SwiftWeightPacking::Pq2_0,
    };
    let fixed_bytes = packing.calibrated_fixed_runtime_bytes();
    let recurrent_bytes = 2_u64
        .checked_mul(RECURRENT_BYTES)
        .ok_or("Recurrent memory calculation overflowed.")?;
    let ctx_u64 = u64::try_from(context_tokens).map_err(|e| e.to_string())?;
    let kv_per_tok = kv_format.kv_bytes_per_token();
    let per_tok_cow = kv_per_tok
        .checked_add(96 + 128)
        .ok_or("KV per-token calculation overflowed.")?;
    let tail_tokens =
        u64::try_from(context_tokens.min(KV_COW_PAGE_TOKENS)).map_err(|e| e.to_string())?;
    let cow_kv = ctx_u64
        .checked_mul(per_tok_cow)
        .and_then(|shared| {
            tail_tokens
                .checked_mul(kv_per_tok)
                .and_then(|t| shared.checked_add(t))
        })
        .ok_or("Calibrated CoW KV calculation overflowed.")?;
    let prefill_scratch = match cpu_prefill_tokens {
        Some(tokens) => Some(
            swift_tiled_prefill_scratch_bytes(usize::from(tokens), context_tokens, kv_format)
                .ok_or("Failed to compute CPU tiled prefill scratch bytes.")?,
        ),
        None => None,
    };
    let total = fixed_bytes
        .checked_add(recurrent_bytes)
        .and_then(|bytes| bytes.checked_add(cow_kv))
        .and_then(|bytes| bytes.checked_add(prefill_scratch.unwrap_or(0)))
        .ok_or("Calibrated server memory calculation overflowed.")?;
    Ok((total, prefill_scratch))
}

#[cfg_attr(
    not(windows),
    expect(
        dead_code,
        reason = "Server CLI builder is executed on Windows and verified cross-platform in unit tests."
    )
)]
fn build_server_arguments(
    config: &DrG2CaptureConfig,
    port: u16,
    key_file: &Path,
    is_candidate_server: bool,
    candidate_bytes: u64,
    candidate_sha256: &str,
) -> crate::Result<Vec<OsString>> {
    let profile = config.base_model.unwrap_or(BaseModelProfile::SwiftPq2);
    let row_workers = config.row_workers.unwrap_or(DEFAULT_ROW_WORKERS);
    let kv_cache = config.kv_cache.as_deref().unwrap_or(DEFAULT_KV_CACHE);
    let kv_format = parse_kv_cache_format(kv_cache)?;

    let mut args = vec![
        OsString::from("--native-engine-child"),
        OsString::from("--model"),
        config.model_path.as_os_str().to_owned(),
        OsString::from("--port"),
        OsString::from(port.to_string()),
        OsString::from("--key-file"),
        key_file.as_os_str().to_owned(),
        OsString::from("--seed"),
        OsString::from("42"),
        OsString::from("--row-workers"),
        OsString::from(row_workers.to_string()),
        OsString::from("--kv-cache"),
        OsString::from(kv_cache),
    ];

    if let Some(placement) = &config.placement {
        args.push(OsString::from("--placement"));
        args.push(OsString::from(placement));
    }

    if let Some(variant) = profile.nariq_variant() {
        args.push(OsString::from("--nariq"));
        args.push(OsString::from(variant.as_str()));
    } else if let Some(orca_path) = &config.orca_adapter_path {
        args.push(OsString::from("--abliterate-lora"));
        args.push(orca_path.as_os_str().to_owned());
    } else {
        return Err("SwiftPq2 capture requires orcaAdapterPath.".into());
    }

    if is_candidate_server {
        args.push(OsString::from("--daisyrose-lora"));
        args.push(config.candidate_adapter_path.as_os_str().to_owned());
        args.push(OsString::from("--daisyrose-lora-bytes"));
        args.push(OsString::from(candidate_bytes.to_string()));
        args.push(OsString::from("--daisyrose-lora-sha256"));
        args.push(OsString::from(candidate_sha256));
    }

    if config.context_tokens.is_some() || config.cpu_prefill_tokens.is_some() {
        let ctx = config.context_tokens.unwrap_or(8_192);
        let (memory_limit_bytes, prefill_scratch) =
            calibrated_server_memory_bytes(profile, ctx, kv_format, config.cpu_prefill_tokens)?;
        args.push(OsString::from("--admission-policy"));
        args.push(OsString::from("calibrated-cow"));
        args.push(OsString::from("--context-tokens"));
        args.push(OsString::from(ctx.to_string()));
        args.push(OsString::from("--memory-limit-bytes"));
        args.push(OsString::from(memory_limit_bytes.to_string()));
        if let Some(tokens) = config.cpu_prefill_tokens {
            let scratch = prefill_scratch.ok_or("Missing CPU prefill scratch calculation.")?;
            args.push(OsString::from("--cpu-prefill-tokens"));
            args.push(OsString::from(tokens.to_string()));
            args.push(OsString::from("--cpu-prefill-scratch-bytes"));
            args.push(OsString::from(scratch.to_string()));
        }
    }

    Ok(args)
}

fn verify_capture_inputs(config: &DrG2CaptureConfig) -> crate::Result<CandidateAdapterMeta> {
    validate_regular_external_file(&config.server_path, "captureConfig.serverPath")?;
    validate_regular_external_file(&config.model_path, "captureConfig.modelPath")?;
    validate_regular_external_file(
        &config.candidate_adapter_path,
        "captureConfig.candidateAdapterPath",
    )?;
    if let Some(orca_path) = &config.orca_adapter_path {
        validate_regular_external_file(orca_path, "captureConfig.orcaAdapterPath")?;
        let mut orca_file =
            bonsai_artifacts::open_locked_file(orca_path).map_err(|e| e.to_string())?;
        let expected_orca = compiled_orca_adapter_sha256()?;
        let _ = crate::identity::hash_reader_checked(
            &mut orca_file,
            None,
            Some(&expected_orca),
            &mut crate::startup_cancel::checkpoint,
        )?;
    }

    let verification = verify_lora_gguf(
        &config.candidate_adapter_path,
        &VerifyPolicy {
            expected_sha256: None,
            expected_envelope: None,
            require_npu_friendly: false,
            require_no_grouped_ssm: true,
            max_allowed_rank: Some(64),
        },
    )?;
    if verification["verified"] != true {
        return Err("captureConfig.candidateAdapterPath failed verify-adapter checks.".into());
    }
    let bytes = verification["bytes"]
        .as_u64()
        .ok_or("Verified candidate adapter missing byte count.")?;
    let sha256 = verification["sha256"]
        .as_str()
        .filter(|s| crate::identity::is_lower_hex(s, 64))
        .ok_or("Verified candidate adapter missing valid SHA-256.")?
        .to_owned();
    let verified_max_rank = verification["maxRank"]
        .as_u64()
        .and_then(|v| u16::try_from(v).ok())
        .ok_or("Verified candidate adapter missing valid maxRank.")?;
    let verified_target_count = verification["pairCount"]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or("Verified candidate adapter missing valid pairCount.")?;
    if let Some(expected_rank) = config.candidate_max_rank
        && expected_rank != verified_max_rank
    {
        return Err(format!(
            "captureConfig.candidateMaxRank ({expected_rank}) disagrees with verified adapter ({verified_max_rank})."
        ));
    }
    if let Some(expected_count) = config.candidate_target_count
        && expected_count != verified_target_count
    {
        return Err(format!(
            "captureConfig.candidateTargetCount ({expected_count}) disagrees with verified adapter ({verified_target_count})."
        ));
    }

    Ok(CandidateAdapterMeta {
        sha256,
        bytes,
        max_rank: Some(verified_max_rank),
        target_count: Some(verified_target_count),
    })
}

fn should_retry_socket(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
    )
}

#[derive(Debug)]
enum HttpExchangeError {
    DeadlineExpired,
    Fatal(String),
}

fn map_http_io_error(error: &io::Error, deadline: Instant, detail: &str) -> HttpExchangeError {
    if Instant::now() >= deadline || error.kind() == io::ErrorKind::TimedOut {
        HttpExchangeError::DeadlineExpired
    } else {
        HttpExchangeError::Fatal(format!("{detail}: {error}"))
    }
}

enum HttpFraming {
    ContentLength(usize),
    Chunked,
    EofUntilClose,
}

fn parse_http_headers(
    head: &[u8],
    max_body_bytes: usize,
) -> Result<(u16, HttpFraming), HttpExchangeError> {
    if !head.starts_with(b"HTTP/1.1 ")
        || head.iter().enumerate().any(|(index, &byte)| {
            (byte == b'\n' && (index == 0 || head[index - 1] != b'\r'))
                || (byte == b'\r' && head.get(index + 1) != Some(&b'\n'))
        })
    {
        return Err(HttpExchangeError::Fatal(
            "Invalid HTTP/1.1 response header line endings.".into(),
        ));
    }
    let mut storage = [httparse::EMPTY_HEADER; 64];
    let mut response = httparse::Response::new(&mut storage);
    match response.parse(head) {
        Ok(httparse::Status::Complete(end)) if end == head.len() => {}
        _ => {
            return Err(HttpExchangeError::Fatal(
                "Malformed loopback HTTP/1.1 response headers.".into(),
            ));
        }
    }
    let status = response.code.ok_or_else(|| {
        HttpExchangeError::Fatal("Loopback HTTP/1.1 response missing status code.".into())
    })?;
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for field in response.headers {
        if field.name.eq_ignore_ascii_case("content-length") {
            let text = std::str::from_utf8(field.value).map_err(|_| {
                HttpExchangeError::Fatal("Invalid Content-Length header bytes.".into())
            })?;
            let len = text.trim().parse::<usize>().map_err(|_| {
                HttpExchangeError::Fatal("Invalid Content-Length header number.".into())
            })?;
            if len > max_body_bytes {
                return Err(HttpExchangeError::Fatal(
                    "Loopback HTTP response Content-Length exceeds safety cap.".into(),
                ));
            }
            content_length = Some(len);
        } else if field.name.eq_ignore_ascii_case("transfer-encoding")
            && field.value.eq_ignore_ascii_case(b"chunked")
        {
            chunked = true;
        }
    }
    let framing = match (content_length, chunked) {
        (Some(len), false) => HttpFraming::ContentLength(len),
        (None, true) => HttpFraming::Chunked,
        (Some(_), true) => {
            return Err(HttpExchangeError::Fatal(
                "Ambiguous Content-Length and Transfer-Encoding headers.".into(),
            ));
        }
        (None, false) => HttpFraming::EofUntilClose,
    };
    Ok((status, framing))
}

fn read_crlf_line<R: io::BufRead>(
    reader: &mut R,
    max_len: usize,
    deadline: Instant,
) -> Result<Vec<u8>, HttpExchangeError> {
    let mut buf = Vec::new();
    while !buf.ends_with(b"\r\n") {
        if buf.len() >= max_len {
            return Err(HttpExchangeError::Fatal(
                "Chunked HTTP line exceeded length bound.".into(),
            ));
        }
        let mut byte = [0_u8; 1];
        reader
            .read_exact(&mut byte)
            .map_err(|e| map_http_io_error(&e, deadline, "Chunked HTTP read error"))?;
        buf.push(byte[0]);
    }
    Ok(buf)
}

fn read_chunked_body<R: io::BufRead>(
    reader: &mut R,
    max_body_bytes: usize,
    deadline: Instant,
) -> Result<Vec<u8>, HttpExchangeError> {
    let mut out = Vec::new();
    loop {
        let line = read_crlf_line(reader, 64, deadline)?;
        let size_part = &line[..line.len().saturating_sub(2)];
        let size_str = std::str::from_utf8(size_part)
            .map_err(|_| HttpExchangeError::Fatal("Invalid chunk size UTF-8.".into()))?;
        let hex_str = size_str.split(';').next().unwrap_or("").trim();
        let chunk_len = usize::from_str_radix(hex_str, 16)
            .map_err(|_| HttpExchangeError::Fatal("Invalid chunk size hex.".into()))?;
        if chunk_len == 0 {
            let trailer = read_crlf_line(reader, 64, deadline)?;
            if trailer != b"\r\n" {
                return Err(HttpExchangeError::Fatal(
                    "Unsupported chunked trailers in loopback HTTP response.".into(),
                ));
            }
            return Ok(out);
        }
        let next_total = out
            .len()
            .checked_add(chunk_len)
            .ok_or_else(|| HttpExchangeError::Fatal("Chunked HTTP body size overflow.".into()))?;
        if next_total > max_body_bytes {
            return Err(HttpExchangeError::Fatal(
                "Chunked HTTP response exceeded safety cap.".into(),
            ));
        }
        let start = out.len();
        out.resize(next_total, 0);
        reader
            .read_exact(&mut out[start..])
            .map_err(|e| map_http_io_error(&e, deadline, "Truncated chunked HTTP response"))?;
        let crlf = read_crlf_line(reader, 4, deadline)?;
        if crlf != b"\r\n" {
            return Err(HttpExchangeError::Fatal(
                "Malformed chunk terminator in loopback HTTP response.".into(),
            ));
        }
    }
}

struct DeadlineReader<F> {
    stream: TcpStream,
    deadline: Instant,
    check: F,
}

impl<F: Fn() -> crate::Result<()>> DeadlineReader<F> {
    fn slice_timeout(&self) -> io::Result<Duration> {
        (self.check)().map_err(io::Error::other)?;
        let remaining = self.deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "DR-G2 capture deadline expired.",
            ));
        }
        Ok(remaining.min(Duration::from_millis(HTTP_POLL_SLICE_MS)))
    }
}

impl<F: Fn() -> crate::Result<()>> io::Read for DeadlineReader<F> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            let slice = self.slice_timeout()?;
            self.stream.set_read_timeout(Some(slice))?;
            match self.stream.read(buf) {
                Err(err) if should_retry_socket(&err) => {}
                other => return other,
            }
        }
    }
}

fn loopback_http_exchange(
    endpoint: &CaptureEndpoint<'_>,
    method: &str,
    route: &str,
    body: Option<&[u8]>,
    max_body_bytes: usize,
    deadline: Instant,
    check: impl Fn() -> crate::Result<()>,
) -> Result<(u16, Vec<u8>), HttpExchangeError> {
    check().map_err(HttpExchangeError::Fatal)?;
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(HttpExchangeError::DeadlineExpired);
    }
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, endpoint.port));
    let connect_timeout = remaining.min(Duration::from_millis(HTTP_POLL_SLICE_MS));
    let mut stream = TcpStream::connect_timeout(&addr, connect_timeout).map_err(|error| {
        map_http_io_error(
            &error,
            deadline,
            &format!("Loopback HTTP connect failed on {route}"),
        )
    })?;
    let _ = stream.set_nodelay(true);

    let mut head = format!(
        "{method} {route} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\nConnection: close\r\n",
        endpoint.port, endpoint.token
    );
    if let Some(payload) = body {
        head.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            payload.len()
        ));
    }
    head.push_str("\r\n");

    let write_all_checked =
        |stream: &mut TcpStream, mut bytes: &[u8]| -> Result<(), HttpExchangeError> {
            while !bytes.is_empty() {
                check().map_err(HttpExchangeError::Fatal)?;
                let rem = deadline.saturating_duration_since(Instant::now());
                if rem.is_zero() {
                    return Err(HttpExchangeError::DeadlineExpired);
                }
                stream
                    .set_write_timeout(Some(rem.min(Duration::from_millis(HTTP_POLL_SLICE_MS))))
                    .map_err(|e| HttpExchangeError::Fatal(e.to_string()))?;
                match stream.write(bytes) {
                    Ok(0) => {
                        return Err(HttpExchangeError::Fatal(
                            "Loopback HTTP socket closed during request write.".into(),
                        ));
                    }
                    Ok(n) => bytes = &bytes[n..],
                    Err(err) if should_retry_socket(&err) => {}
                    Err(err) => {
                        return Err(HttpExchangeError::Fatal(format!(
                            "Loopback HTTP write error: {err}"
                        )));
                    }
                }
            }
            Ok(())
        };

    write_all_checked(&mut stream, head.as_bytes())?;
    if let Some(payload) = body {
        write_all_checked(&mut stream, payload)?;
    }

    let reader_impl = DeadlineReader {
        stream,
        deadline,
        check,
    };
    let mut reader = BufReader::with_capacity(16 * 1024, reader_impl);
    let mut raw_head = Vec::new();
    while !raw_head.ends_with(b"\r\n\r\n") {
        if raw_head.len() >= MAX_HTTP_HEAD_BYTES {
            return Err(HttpExchangeError::Fatal(
                "Loopback HTTP response headers exceeded 16 KiB.".into(),
            ));
        }
        let mut byte = [0_u8; 1];
        if let Err(err) = reader.read_exact(&mut byte) {
            return Err(map_http_io_error(
                &err,
                deadline,
                "Loopback HTTP header read failed",
            ));
        }
        raw_head.push(byte[0]);
    }
    let (status, framing) = parse_http_headers(&raw_head, max_body_bytes)?;
    let body_bytes = match framing {
        HttpFraming::ContentLength(len) => {
            let mut buf = vec![0_u8; len];
            if let Err(err) = reader.read_exact(&mut buf) {
                return Err(map_http_io_error(
                    &err,
                    deadline,
                    "Loopback HTTP body read failed",
                ));
            }
            buf
        }
        HttpFraming::Chunked => read_chunked_body(&mut reader, max_body_bytes, deadline)?,
        HttpFraming::EofUntilClose => {
            let mut buf = Vec::new();
            let limit = u64::try_from(max_body_bytes)
                .ok()
                .and_then(|v| v.checked_add(1))
                .unwrap_or(u64::MAX);
            if let Err(err) = (&mut reader).take(limit).read_to_end(&mut buf) {
                return Err(map_http_io_error(
                    &err,
                    deadline,
                    "Loopback HTTP EOF body read failed",
                ));
            }
            if buf.len() > max_body_bytes {
                return Err(HttpExchangeError::Fatal(
                    "Loopback HTTP EOF response exceeded safety cap.".into(),
                ));
            }
            buf
        }
    };
    Ok((status, body_bytes))
}

fn wait_endpoint_ready(
    endpoint: &CaptureEndpoint<'_>,
    expect_daisyrose: bool,
    deadline: Instant,
    check: impl Fn() -> crate::Result<()>,
) -> crate::Result<()> {
    loop {
        check()?;
        if Instant::now() >= deadline {
            return Err("Timed out waiting for prism-inference-server readiness.".into());
        }
        if let Ok((200, health_bytes)) = loopback_http_exchange(
            endpoint,
            "GET",
            "/health",
            None,
            MAX_CONTROL_RESPONSE_BYTES,
            deadline,
            &check,
        ) && let Ok(health_json) = serde_json::from_slice::<Value>(&health_bytes)
            && health_json["status"] == "ok"
            && let Ok((200, models_bytes)) = loopback_http_exchange(
                endpoint,
                "GET",
                "/v1/models",
                None,
                MAX_CONTROL_RESPONSE_BYTES,
                deadline,
                &check,
            )
        {
            let models: Value = serde_json::from_slice(&models_bytes)
                .map_err(|e| format!("Invalid /v1/models JSON: {e}"))?;
            let data = models["data"]
                .as_array()
                .ok_or("/v1/models response missing data array.")?;
            let model = data.first().ok_or("/v1/models data array is empty.")?;
            if expect_daisyrose && model["supports_daisyrose_scale"] != true {
                return Err(
                    "Candidate server /v1/models readiness did not advertise supports_daisyrose_scale=true."
                        .into(),
                );
            }
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err("Timed out waiting for prism-inference-server readiness.".into());
        }
        std::thread::sleep(Duration::from_millis(SERVER_READY_POLL_MS));
    }
}

fn build_chat_completion_payload(
    prompt_text: &str,
    max_completion_tokens: u32,
    enable_thinking: bool,
    profile: BaseModelProfile,
    has_candidate_adapter: bool,
    daisyrose_scale: f64,
) -> crate::Result<Vec<u8>> {
    let lora = match (profile.is_nariq(), has_candidate_adapter) {
        (false, false) => Some(json!([{ "id": 0, "scale": 1.3 }])),
        (false, true) => Some(json!([
            { "id": 0, "scale": 1.3 },
            { "id": 1, "scale": daisyrose_scale }
        ])),
        (true, false) => None,
        (true, true) => Some(json!([{ "id": 0, "scale": daisyrose_scale }])),
    };
    let mut payload = json!({
        "messages": [{ "role": "user", "content": prompt_text }],
        "max_completion_tokens": max_completion_tokens,
        "stream": false,
        "n": 1,
        "cache_prompt": false,
        "n_cache_reuse": 0,
        "ignore_eos": false,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repeat_penalty": 1.0,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "reasoning_effort": "xhigh",
            "preserve_thinking": true,
        }
    });
    if let Some(lora_arr) = lora
        && let Some(obj) = payload.as_object_mut()
    {
        obj.insert("lora".to_owned(), lora_arr);
    }
    serde_json::to_vec(&payload).map_err(|e| format!("Cannot serialize chat payload: {e}"))
}

fn extract_inline_think_and_answer(content: &str) -> (&str, &str) {
    let Some(start) = content.find("<think>") else {
        return ("", content);
    };
    let after_open = &content[start + "<think>".len()..];
    if let Some(end) = after_open.find("</think>") {
        (
            &after_open[..end],
            after_open[end + "</think>".len()..].trim_start(),
        )
    } else {
        (after_open, "")
    }
}

#[expect(
    clippy::as_conversions,
    reason = "Bounded byte lengths and token counts convert cleanly to f64 for proportional <think> token estimation."
)]
#[inline]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[expect(
    clippy::as_conversions,
    reason = "Proportional token estimate bounded by u32::MAX converts safely back to u32."
)]
#[inline]
fn round_clamp_u32(value: f64) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else if value >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        value.round() as u32
    }
}

fn estimate_thinking_tokens(
    reasoning_text: &str,
    answer_text: &str,
    usage: &Value,
    enable_thinking: bool,
) -> u32 {
    if !enable_thinking {
        return 0;
    }
    if let Some(explicit) = usage["completion_tokens_details"]["reasoning_tokens"].as_u64() {
        return u32::try_from(explicit).unwrap_or(u32::MAX);
    }
    if reasoning_text.is_empty() {
        return 0;
    }
    let completion_tokens = usage["completion_tokens"].as_u64();
    match completion_tokens {
        Some(0) => 0,
        Some(total) => {
            if answer_text.is_empty() {
                return u32::try_from(total).unwrap_or(u32::MAX);
            }
            let think_bytes = reasoning_text.len();
            let total_bytes = think_bytes.saturating_add(answer_text.len());
            if total_bytes == 0 {
                return 0;
            }
            let total_f64 = f64::from(u32::try_from(total).unwrap_or(u32::MAX));
            let est = round_clamp_u32(
                (total_f64 * usize_to_f64(think_bytes)) / usize_to_f64(total_bytes),
            );
            est.max(1)
        }
        None => {
            let approx_words = reasoning_text.split_whitespace().count();
            let approx_chars = reasoning_text.len().div_ceil(4);
            u32::try_from(approx_words.max(approx_chars).max(1)).unwrap_or(u32::MAX)
        }
    }
}

/// Deterministic repetition-loop detector for long degenerate lines, word windows, or trailing periodic chunks.
pub(super) fn detect_repetition_loop(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.len() < 64 {
        return false;
    }

    // 1. Five or more consecutive identical non-trivial lines (>= 16 bytes).
    let mut prev_line: Option<&str> = None;
    let mut line_run = 0_usize;
    for raw_line in trimmed.lines() {
        let line = raw_line.trim();
        if line.len() < 16 {
            prev_line = None;
            line_run = 0;
            continue;
        }
        if prev_line == Some(line) {
            line_run += 1;
            if line_run >= 5 {
                return true;
            }
        } else {
            prev_line = Some(line);
            line_run = 1;
        }
    }

    // 2. Four or more consecutive exact repetitions of a 12-word window across the tail.
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let window = 12_usize;
    let repeats = 4_usize;
    if words.len() >= window * repeats {
        let tail = if words.len() > 2_048 {
            &words[words.len() - 2_048..]
        } else {
            &words[..]
        };
        let span = window * repeats;
        for start in 0..=tail.len() - span {
            let pat = &tail[start..start + window];
            if (1..repeats).all(|k| &tail[start + k * window..start + (k + 1) * window] == pat) {
                return true;
            }
        }
    }

    // 3. Six or more consecutive identical non-trivial byte periods at the tail (16..=256 bytes).
    let bytes = trimmed.as_bytes();
    let max_period = (bytes.len() / 6).min(256);
    for period in 16..=max_period {
        let tail_len = period * 6;
        let tail = &bytes[bytes.len() - tail_len..];
        let pat = &tail[..period];
        if pat.iter().any(|b| !b.is_ascii_whitespace())
            && (1..6).all(|k| &tail[k * period..(k + 1) * period] == pat)
        {
            return true;
        }
    }

    false
}

fn run_chat_completion(
    endpoint: &CaptureEndpoint<'_>,
    prompt_text: &str,
    max_completion_tokens: u32,
    enable_thinking: bool,
    profile: BaseModelProfile,
    has_candidate_adapter: bool,
    daisyrose_scale: f64,
    deadline: Instant,
    check: impl Fn() -> crate::Result<()>,
) -> Result<ChatCompletionOutcome, HttpExchangeError> {
    let payload = build_chat_completion_payload(
        prompt_text,
        max_completion_tokens,
        enable_thinking,
        profile,
        has_candidate_adapter,
        daisyrose_scale,
    )
    .map_err(HttpExchangeError::Fatal)?;

    let (status, resp_bytes) = loopback_http_exchange(
        endpoint,
        "POST",
        "/v1/chat/completions",
        Some(&payload),
        MAX_CHAT_RESPONSE_BYTES,
        deadline,
        check,
    )?;
    if status != 200 {
        return Err(HttpExchangeError::Fatal(format!(
            "Loopback /v1/chat/completions returned HTTP {status}."
        )));
    }
    let resp: Value = serde_json::from_slice(&resp_bytes)
        .map_err(|e| HttpExchangeError::Fatal(format!("Invalid chat completion JSON: {e}")))?;
    let choices = resp["choices"].as_array().ok_or_else(|| {
        HttpExchangeError::Fatal("Chat completion JSON missing choices array.".into())
    })?;
    let choice = choices.first().ok_or_else(|| {
        HttpExchangeError::Fatal("Chat completion JSON has empty choices.".into())
    })?;
    let message = &choice["message"];
    let content = match message.get("content") {
        None | Some(Value::Null) => "",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => {
            return Err(HttpExchangeError::Fatal(
                "Chat completion message.content must be a string.".into(),
            ));
        }
    };
    let raw_reasoning = match message.get("reasoning_content") {
        None | Some(Value::Null) => "",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => {
            return Err(HttpExchangeError::Fatal(
                "Chat completion message.reasoning_content must be a string.".into(),
            ));
        }
    };

    let (canonical_text, reasoning_for_tokens, answer_for_tokens) = if !raw_reasoning.is_empty() {
        let combined = if content.is_empty() {
            format!("<think>{raw_reasoning}</think>")
        } else {
            format!("<think>{raw_reasoning}</think>\n{content}")
        };
        (combined, raw_reasoning, content)
    } else {
        let (inline_think, inline_ans) = extract_inline_think_and_answer(content);
        (content.to_owned(), inline_think, inline_ans)
    };

    let response_sha256 = crate::identity::sha256(canonical_text.as_bytes());
    let thinking_tokens = estimate_thinking_tokens(
        reasoning_for_tokens,
        answer_for_tokens,
        &resp["usage"],
        enable_thinking,
    );
    let finish_reason = choice["finish_reason"].as_str();
    let completion_tokens = resp["usage"]["completion_tokens"].as_u64();
    let hit4096_truncation = (max_completion_tokens >= DEFAULT_MAX_COMPLETION_TOKENS
        && finish_reason == Some("length"))
        || completion_tokens.is_some_and(|t| t >= u64::from(DEFAULT_MAX_COMPLETION_TOKENS));
    let loop_detected = detect_repetition_loop(&canonical_text);

    Ok(ChatCompletionOutcome {
        canonical_text,
        response_sha256,
        thinking_tokens,
        hit4096_truncation,
        loop_detected,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Captures zero-scale controls, Thinking-ON/OFF pairs, and per-arm power within one shared budget."
)]
fn capture_against_endpoints_inner(
    config: &DrG2CaptureConfig,
    candidate_meta: &CandidateAdapterMeta,
    output_path: &Path,
    server_a: &CaptureEndpoint<'_>,
    server_b: &CaptureEndpoint<'_>,
    budget_seconds: u64,
    inference_budget: Duration,
    sleep_inhibition_active: bool,
    check_servers: impl Fn() -> crate::Result<()>,
) -> crate::Result<Value> {
    let started = Instant::now();
    let power_before = crate::host_cpu::power_snapshot();
    let inference_deadline = started
        .checked_add(inference_budget)
        .ok_or("DR-G2 capture deadline overflow.")?;

    wait_endpoint_ready(server_a, false, inference_deadline, &check_servers)?;
    wait_endpoint_ready(server_b, true, inference_deadline, &check_servers)?;

    let profile = config.base_model.unwrap_or(BaseModelProfile::SwiftPq2);
    let base_model_sha256 = profile.base_model_sha256()?;
    let orca_adapter_sha256 = if profile.is_nariq() {
        None
    } else {
        Some(compiled_orca_adapter_sha256()?)
    };
    let orca_scale = if profile.is_nariq() { 0.0 } else { 1.3 };
    let scale_grid = resolve_scale_grid(config.scale_grid.as_ref())?;
    let contract = build_dr_g2_contract(
        profile.is_nariq().then_some(profile),
        false,
        &base_model_sha256,
        orca_adapter_sha256.as_deref(),
        &scale_grid,
        Some(&candidate_meta.sha256),
        candidate_meta.max_rank,
        candidate_meta.target_count,
    );
    let contract_sha256 = hash_contract(&contract)?;

    let mut zero_scale_controls = Vec::with_capacity(config.zero_scale_controls.len());
    let mut thinking_on_prompts = Vec::with_capacity(config.thinking_on_prompts.len());
    let mut thinking_off_prompts = Vec::with_capacity(config.thinking_off_prompts.len());
    let mut blind_responses: BTreeMap<String, BlindResponseEntry> = BTreeMap::new();
    let mut arm_power_observations = Vec::new();
    let mut skipped_prompts = Vec::new();
    let mut budget_exceeded = false;

    macro_rules! check_budget_or_skip {
        ($prompt_id:expr) => {
            if Instant::now() >= inference_deadline {
                budget_exceeded = true;
                skipped_prompts.push($prompt_id.clone());
                true
            } else {
                false
            }
        };
    }

    // 1. Zero-scale controls (Server A baseline vs Server B scale=0.0).
    for (idx, ctrl) in config.zero_scale_controls.iter().enumerate() {
        if check_budget_or_skip!(ctrl.prompt_id) {
            for rem in &config.zero_scale_controls[idx + 1..] {
                skipped_prompts.push(rem.prompt_id.clone());
            }
            break;
        }
        let max_tokens = validate_max_tokens(
            ctrl.max_completion_tokens,
            DEFAULT_ZERO_CONTROL_MAX_TOKENS,
            "Zero-scale control",
        )?;
        let enable_thinking = ctrl.enable_thinking.unwrap_or(true);
        let arm_a_power_before = crate::host_cpu::power_snapshot();
        let no_adapter = match run_chat_completion(
            server_a,
            &ctrl.prompt_text,
            max_tokens,
            enable_thinking,
            profile,
            false,
            0.0,
            inference_deadline,
            &check_servers,
        ) {
            Ok(out) => out,
            Err(HttpExchangeError::DeadlineExpired) => {
                let arm_a_power_after = crate::host_cpu::power_snapshot();
                arm_power_observations.push(json!({
                    "promptId": ctrl.prompt_id,
                    "arm": "zero-no-adapter",
                    "powerBefore": arm_a_power_before,
                    "powerAfter": arm_a_power_after,
                    "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&arm_a_power_before, &arm_a_power_after),
                    "completed": false,
                }));
                budget_exceeded = true;
                for rem in &config.zero_scale_controls[idx..] {
                    skipped_prompts.push(rem.prompt_id.clone());
                }
                break;
            }
            Err(HttpExchangeError::Fatal(msg)) => return Err(msg),
        };
        let arm_a_power_after = crate::host_cpu::power_snapshot();
        arm_power_observations.push(json!({
            "promptId": ctrl.prompt_id,
            "arm": "zero-no-adapter",
            "powerBefore": arm_a_power_before,
            "powerAfter": arm_a_power_after,
            "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&arm_a_power_before, &arm_a_power_after),
            "completed": true,
        }));

        if Instant::now() >= inference_deadline {
            budget_exceeded = true;
            for rem in &config.zero_scale_controls[idx..] {
                skipped_prompts.push(rem.prompt_id.clone());
            }
            break;
        }

        let arm_b_power_before = crate::host_cpu::power_snapshot();
        let zero_scale = match run_chat_completion(
            server_b,
            &ctrl.prompt_text,
            max_tokens,
            enable_thinking,
            profile,
            true,
            0.0,
            inference_deadline,
            &check_servers,
        ) {
            Ok(out) => out,
            Err(HttpExchangeError::DeadlineExpired) => {
                let arm_b_power_after = crate::host_cpu::power_snapshot();
                arm_power_observations.push(json!({
                    "promptId": ctrl.prompt_id,
                    "arm": "zero-scale",
                    "powerBefore": arm_b_power_before,
                    "powerAfter": arm_b_power_after,
                    "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&arm_b_power_before, &arm_b_power_after),
                    "completed": false,
                }));
                budget_exceeded = true;
                for rem in &config.zero_scale_controls[idx..] {
                    skipped_prompts.push(rem.prompt_id.clone());
                }
                break;
            }
            Err(HttpExchangeError::Fatal(msg)) => return Err(msg),
        };
        let arm_b_power_after = crate::host_cpu::power_snapshot();
        arm_power_observations.push(json!({
            "promptId": ctrl.prompt_id,
            "arm": "zero-scale",
            "powerBefore": arm_b_power_before,
            "powerAfter": arm_b_power_after,
            "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&arm_b_power_before, &arm_b_power_after),
            "completed": true,
        }));

        zero_scale_controls.push(ZeroScaleControl {
            prompt_id: ctrl.prompt_id.clone(),
            prompt_sha256: crate::identity::sha256(ctrl.prompt_text.as_bytes()),
            no_adapter_response_sha256: no_adapter.response_sha256,
            zero_scale_response_sha256: zero_scale.response_sha256,
        });
    }

    // 2. Thinking-ON prompts (coding, reasoning, instruction-following).
    for (idx, prompt) in config.thinking_on_prompts.iter().enumerate() {
        if budget_exceeded || Instant::now() >= inference_deadline {
            budget_exceeded = true;
            for rem in &config.thinking_on_prompts[idx..] {
                skipped_prompts.push(rem.prompt_id.clone());
            }
            break;
        }
        let max_tokens = validate_max_tokens(
            prompt.max_completion_tokens,
            DEFAULT_MAX_COMPLETION_TOKENS,
            "Thinking-ON",
        )?;
        let base_power_before = crate::host_cpu::power_snapshot();
        let baseline = match run_chat_completion(
            server_a,
            &prompt.prompt_text,
            max_tokens,
            true,
            profile,
            false,
            0.0,
            inference_deadline,
            &check_servers,
        ) {
            Ok(out) => out,
            Err(HttpExchangeError::DeadlineExpired) => {
                let base_power_after = crate::host_cpu::power_snapshot();
                arm_power_observations.push(json!({
                    "promptId": prompt.prompt_id,
                    "arm": "baseline",
                    "powerBefore": base_power_before,
                    "powerAfter": base_power_after,
                    "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&base_power_before, &base_power_after),
                    "completed": false,
                }));
                budget_exceeded = true;
                for rem in &config.thinking_on_prompts[idx..] {
                    skipped_prompts.push(rem.prompt_id.clone());
                }
                break;
            }
            Err(HttpExchangeError::Fatal(msg)) => return Err(msg),
        };
        let base_power_after = crate::host_cpu::power_snapshot();
        arm_power_observations.push(json!({
            "promptId": prompt.prompt_id,
            "arm": "baseline",
            "powerBefore": base_power_before,
            "powerAfter": base_power_after,
            "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&base_power_before, &base_power_after),
            "completed": true,
        }));

        if Instant::now() >= inference_deadline {
            budget_exceeded = true;
            for rem in &config.thinking_on_prompts[idx..] {
                skipped_prompts.push(rem.prompt_id.clone());
            }
            break;
        }

        let cand_power_before = crate::host_cpu::power_snapshot();
        let candidate = match run_chat_completion(
            server_b,
            &prompt.prompt_text,
            max_tokens,
            true,
            profile,
            true,
            config.selected_scale,
            inference_deadline,
            &check_servers,
        ) {
            Ok(out) => out,
            Err(HttpExchangeError::DeadlineExpired) => {
                let cand_power_after = crate::host_cpu::power_snapshot();
                arm_power_observations.push(json!({
                    "promptId": prompt.prompt_id,
                    "arm": "candidate",
                    "powerBefore": cand_power_before,
                    "powerAfter": cand_power_after,
                    "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&cand_power_before, &cand_power_after),
                    "completed": false,
                }));
                budget_exceeded = true;
                for rem in &config.thinking_on_prompts[idx..] {
                    skipped_prompts.push(rem.prompt_id.clone());
                }
                break;
            }
            Err(HttpExchangeError::Fatal(msg)) => return Err(msg),
        };
        let cand_power_after = crate::host_cpu::power_snapshot();
        arm_power_observations.push(json!({
            "promptId": prompt.prompt_id,
            "arm": "candidate",
            "powerBefore": cand_power_before,
            "powerAfter": cand_power_after,
            "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&cand_power_before, &cand_power_after),
            "completed": true,
        }));

        blind_responses.insert(
            prompt.prompt_id.clone(),
            BlindResponseEntry {
                prompt_text: Some(prompt.prompt_text.clone()),
                baseline_text: baseline.canonical_text,
                candidate_text: candidate.canonical_text,
            },
        );
        thinking_on_prompts.push(ThinkingOnPromptPair {
            prompt_id: prompt.prompt_id.clone(),
            prompt_sha256: crate::identity::sha256(prompt.prompt_text.as_bytes()),
            category: prompt.category.clone(),
            baseline: ThinkingOnArm {
                scale: 0.0,
                response_sha256: baseline.response_sha256,
                human_reviewed: false,
                reviewer_id: String::new(),
                score: 0.0,
                thinking_tokens: baseline.thinking_tokens,
                hit4096_truncation: baseline.hit4096_truncation,
                loop_detected: baseline.loop_detected,
            },
            candidate: ThinkingOnArm {
                scale: config.selected_scale,
                response_sha256: candidate.response_sha256,
                human_reviewed: false,
                reviewer_id: String::new(),
                score: 0.0,
                thinking_tokens: candidate.thinking_tokens,
                hit4096_truncation: candidate.hit4096_truncation,
                loop_detected: candidate.loop_detected,
            },
        });
    }

    // 3. Thinking-OFF creative-prose prompts.
    for (idx, prompt) in config.thinking_off_prompts.iter().enumerate() {
        if budget_exceeded || Instant::now() >= inference_deadline {
            budget_exceeded = true;
            for rem in &config.thinking_off_prompts[idx..] {
                skipped_prompts.push(rem.prompt_id.clone());
            }
            break;
        }
        let max_tokens = validate_max_tokens(
            prompt.max_completion_tokens,
            DEFAULT_MAX_COMPLETION_TOKENS,
            "Thinking-OFF",
        )?;
        let base_power_before = crate::host_cpu::power_snapshot();
        let baseline = match run_chat_completion(
            server_a,
            &prompt.prompt_text,
            max_tokens,
            false,
            profile,
            false,
            0.0,
            inference_deadline,
            &check_servers,
        ) {
            Ok(out) => out,
            Err(HttpExchangeError::DeadlineExpired) => {
                let base_power_after = crate::host_cpu::power_snapshot();
                arm_power_observations.push(json!({
                    "promptId": prompt.prompt_id,
                    "arm": "baseline",
                    "powerBefore": base_power_before,
                    "powerAfter": base_power_after,
                    "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&base_power_before, &base_power_after),
                    "completed": false,
                }));
                budget_exceeded = true;
                for rem in &config.thinking_off_prompts[idx..] {
                    skipped_prompts.push(rem.prompt_id.clone());
                }
                break;
            }
            Err(HttpExchangeError::Fatal(msg)) => return Err(msg),
        };
        let base_power_after = crate::host_cpu::power_snapshot();
        arm_power_observations.push(json!({
            "promptId": prompt.prompt_id,
            "arm": "baseline",
            "powerBefore": base_power_before,
            "powerAfter": base_power_after,
            "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&base_power_before, &base_power_after),
            "completed": true,
        }));

        if Instant::now() >= inference_deadline {
            budget_exceeded = true;
            for rem in &config.thinking_off_prompts[idx..] {
                skipped_prompts.push(rem.prompt_id.clone());
            }
            break;
        }

        let cand_power_before = crate::host_cpu::power_snapshot();
        let candidate = match run_chat_completion(
            server_b,
            &prompt.prompt_text,
            max_tokens,
            false,
            profile,
            true,
            config.selected_scale,
            inference_deadline,
            &check_servers,
        ) {
            Ok(out) => out,
            Err(HttpExchangeError::DeadlineExpired) => {
                let cand_power_after = crate::host_cpu::power_snapshot();
                arm_power_observations.push(json!({
                    "promptId": prompt.prompt_id,
                    "arm": "candidate",
                    "powerBefore": cand_power_before,
                    "powerAfter": cand_power_after,
                    "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&cand_power_before, &cand_power_after),
                    "completed": false,
                }));
                budget_exceeded = true;
                for rem in &config.thinking_off_prompts[idx..] {
                    skipped_prompts.push(rem.prompt_id.clone());
                }
                break;
            }
            Err(HttpExchangeError::Fatal(msg)) => return Err(msg),
        };
        let cand_power_after = crate::host_cpu::power_snapshot();
        arm_power_observations.push(json!({
            "promptId": prompt.prompt_id,
            "arm": "candidate",
            "powerBefore": cand_power_before,
            "powerAfter": cand_power_after,
            "powerConditions": bonsai_cpu_policy::power::compare_endpoints(&cand_power_before, &cand_power_after),
            "completed": true,
        }));

        blind_responses.insert(
            prompt.prompt_id.clone(),
            BlindResponseEntry {
                prompt_text: Some(prompt.prompt_text.clone()),
                baseline_text: baseline.canonical_text,
                candidate_text: candidate.canonical_text,
            },
        );
        thinking_off_prompts.push(ThinkingOffPromptPair {
            prompt_id: prompt.prompt_id.clone(),
            prompt_sha256: crate::identity::sha256(prompt.prompt_text.as_bytes()),
            category: prompt.category.clone(),
            baseline: ThinkingOffArm {
                scale: 0.0,
                response_sha256: baseline.response_sha256,
                human_reviewed: false,
                reviewer_id: String::new(),
                score: 0.0,
                unwanted_refusal: false,
            },
            candidate: ThinkingOffArm {
                scale: config.selected_scale,
                response_sha256: candidate.response_sha256,
                human_reviewed: false,
                reviewer_id: String::new(),
                score: 0.0,
                unwanted_refusal: false,
            },
        });
    }

    let power_after = crate::host_cpu::power_snapshot();
    let power_conditions = bonsai_cpu_policy::power::compare_endpoints(&power_before, &power_after);
    let zero_scale_mismatches = zero_scale_controls
        .iter()
        .filter(|c| c.no_adapter_response_sha256 != c.zero_scale_response_sha256)
        .count();

    let evidence = DrG2Evidence {
        schema_version: 1,
        base_model: profile.is_nariq().then_some(profile),
        include_multimodal_checks: false,
        contract_sha256: contract_sha256.clone(),
        base_model_sha256,
        orca_adapter_sha256,
        orca_scale,
        candidate_adapter_sha256: candidate_meta.sha256.clone(),
        candidate_max_rank: candidate_meta.max_rank,
        candidate_target_count: candidate_meta.target_count,
        scale_grid,
        selected_scale: config.selected_scale,
        training_prompt_sha256s: config.training_prompt_sha256s.clone(),
        thinking_on_prompts,
        thinking_off_prompts,
        multimodal_prompts: Vec::new(),
        zero_scale_controls,
    };

    let evidence_bytes = crate::bounded_input::pretty_json_bytes(&evidence)?;
    let evidence_sha256 = crate::identity::sha256(&evidence_bytes);
    crate::bounded_input::write_new_synced(output_path, &evidence_bytes)?;

    let responses_sha256 = if let Some(resp_path) = &config.responses_output_path {
        let resp_bytes = crate::bounded_input::pretty_json_bytes(&blind_responses)?;
        let digest = crate::identity::sha256(&resp_bytes);
        crate::bounded_input::write_new_synced(resp_path, &resp_bytes)?;
        Some(digest)
    } else {
        None
    };

    let completed = !budget_exceeded && skipped_prompts.is_empty();
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);

    Ok(json!({
        "schemaVersion": 1,
        "operation": "dr-g2-capture",
        "status": if completed { "capture-completed" } else { "budget-exceeded-incomplete" },
        "completed": completed,
        "budgetExceeded": budget_exceeded,
        "elapsedMilliseconds": elapsed_ms,
        "timeBudgetSeconds": budget_seconds,
        "sleepInhibitionActive": sleep_inhibition_active,
        "baseModel": profile,
        "contractSha256": contract_sha256,
        "evidenceSha256": evidence_sha256,
        "responsesSha256": responses_sha256,
        "completedZeroControls": evidence.zero_scale_controls.len(),
        "completedThinkingOn": evidence.thinking_on_prompts.len(),
        "completedThinkingOff": evidence.thinking_off_prompts.len(),
        "skippedPrompts": skipped_prompts,
        "zeroScaleMismatches": zero_scale_mismatches,
        "powerBefore": power_before,
        "powerAfter": power_after,
        "powerConditions": power_conditions,
        "armPowerObservations": arm_power_observations,
    }))
}

#[cfg(windows)]
struct CaptureScratchRoot(PathBuf);

#[cfg(windows)]
impl CaptureScratchRoot {
    fn new() -> crate::Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "Bonsai-DrG2Capture-{}",
            crate::startup_process::unique()
        ));
        bonsai_artifacts::validate_external_path(&path).map_err(|e| e.to_string())?;
        fs::create_dir(&path).map_err(|e| e.to_string())?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

#[cfg(windows)]
impl Drop for CaptureScratchRoot {
    fn drop(&mut self) {
        if let Ok(parent) = fs::canonicalize(std::env::temp_dir())
            && let Ok(target) = fs::canonicalize(&self.0)
            && target.parent() == Some(parent.as_path())
            && fs::symlink_metadata(&self.0).is_ok_and(|meta| !bonsai_artifacts::is_redirect(&meta))
        {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(windows)]
fn allocate_two_loopback_ports() -> crate::Result<(u16, u16)> {
    let listener_a = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .map_err(|e| format!("Bind loopback port A: {e}"))?;
    let port_a = listener_a
        .local_addr()
        .map_err(|e| format!("Inspect loopback port A: {e}"))?
        .port();
    let listener_b = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .map_err(|e| format!("Bind loopback port B: {e}"))?;
    let port_b = listener_b
        .local_addr()
        .map_err(|e| format!("Inspect loopback port B: {e}"))?
        .port();
    drop(listener_a);
    drop(listener_b);
    if port_a < 1024 || port_b < 1024 || port_a == port_b {
        return Err("Failed to allocate two distinct nonprivileged loopback ports.".into());
    }
    Ok((port_a, port_b))
}

#[cfg(windows)]
fn run_windows_capture(request: &Request) -> crate::Result<Value> {
    use bonsai_windows_credentials::PrivateCredentials;
    use bonsai_windows_process::{OwnedProcess, ProcessSpec, SystemAwakeGuard};

    let config = request
        .capture_config
        .as_ref()
        .ok_or("dr-g2-capture requires captureConfig.")?;
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("dr-g2-capture requires outputPath.")?;
    let budget_seconds = effective_time_budget_seconds(request, config)?;

    let _awake_guard = SystemAwakeGuard::acquire()
        .map_err(|error| format!("Failed to acquire Windows idle-sleep inhibition: {error}"))?;
    let candidate_meta = verify_capture_inputs(config)?;

    let scratch = CaptureScratchRoot::new()?;
    let cred_a = PrivateCredentials::create(scratch.path())
        .map_err(|e| format!("Failed to create Server A private credentials: {e}"))?;
    let cred_b = PrivateCredentials::create(scratch.path())
        .map_err(|e| format!("Failed to create Server B private credentials: {e}"))?;
    let work_a = scratch.path().join("server-a");
    let work_b = scratch.path().join("server-b");
    fs::create_dir(&work_a).map_err(|e| e.to_string())?;
    fs::create_dir(&work_b).map_err(|e| e.to_string())?;

    let (port_a, port_b) = allocate_two_loopback_ports()?;
    let args_a = build_server_arguments(
        config,
        port_a,
        cred_a.backend_path(),
        false,
        candidate_meta.bytes,
        &candidate_meta.sha256,
    )?;
    let args_b = build_server_arguments(
        config,
        port_b,
        cred_b.backend_path(),
        true,
        candidate_meta.bytes,
        &candidate_meta.sha256,
    )?;

    let spec_a = ProcessSpec {
        executable: config.server_path.clone(),
        arguments: args_a,
        working_directory: work_a.clone(),
        stdout: work_a.join("stdout.log"),
        stderr: work_a.join("stderr.log"),
        memory_limit_bytes: SERVER_JOB_MEMORY_LIMIT_BYTES,
        active_process_limit: 1,
    };
    let spec_b = ProcessSpec {
        executable: config.server_path.clone(),
        arguments: args_b,
        working_directory: work_b.clone(),
        stdout: work_b.join("stdout.log"),
        stderr: work_b.join("stderr.log"),
        memory_limit_bytes: SERVER_JOB_MEMORY_LIMIT_BYTES,
        active_process_limit: 1,
    };

    let proc_a = OwnedProcess::start(&spec_a)
        .map_err(|e| format!("Failed to start Server A prism-inference-server: {e}"))?;
    let proc_b = OwnedProcess::start(&spec_b).map_err(|e| {
        let _ = proc_a.stop(SERVER_STOP_TIMEOUT);
        format!("Failed to start Server B prism-inference-server: {e}")
    })?;

    let endpoint_a = CaptureEndpoint {
        port: port_a,
        token: cred_a.backend_key(),
    };
    let endpoint_b = CaptureEndpoint {
        port: port_b,
        token: cred_b.backend_key(),
    };

    let check_servers = || -> crate::Result<()> {
        crate::startup_cancel::checkpoint()?;
        if let Some(code) = proc_a.try_wait().map_err(|e| e.to_string())? {
            return Err(format!(
                "Server A prism-inference-server exited unexpectedly with code {code}."
            ));
        }
        if let Some(code) = proc_b.try_wait().map_err(|e| e.to_string())? {
            return Err(format!(
                "Server B prism-inference-server exited unexpectedly with code {code}."
            ));
        }
        Ok(())
    };

    let reserve_seconds = CLEANUP_RESERVE_SECONDS.min(budget_seconds / 3);
    let inference_budget = Duration::from_secs(budget_seconds.saturating_sub(reserve_seconds));
    let capture_result = capture_against_endpoints_inner(
        config,
        &candidate_meta,
        output_path,
        &endpoint_a,
        &endpoint_b,
        budget_seconds,
        inference_budget,
        true,
        check_servers,
    );

    let stop_a = proc_a.stop(SERVER_STOP_TIMEOUT);
    let stop_b = proc_b.stop(SERVER_STOP_TIMEOUT);
    let lingering_a = proc_a.active_processes().unwrap_or(1);
    let lingering_b = proc_b.active_processes().unwrap_or(1);
    let lingering_count = lingering_a.saturating_add(lingering_b);
    drop(proc_a);
    drop(proc_b);

    let cred_dir_a = cred_a.directory().to_path_buf();
    let cred_dir_b = cred_b.directory().to_path_buf();
    drop(cred_a);
    drop(cred_b);
    let scratch_path = scratch.path().to_path_buf();
    drop(scratch);

    stop_a.map_err(|e| format!("Failed stopping Server A Job: {e}"))?;
    stop_b.map_err(|e| format!("Failed stopping Server B Job: {e}"))?;
    if lingering_count != 0 {
        return Err(format!(
            "DR-G2 capture cleanup left {lingering_count} active server process(es)."
        ));
    }

    let cred_dir_exists_after = cred_dir_a.exists() || cred_dir_b.exists() || scratch_path.exists();
    let mut receipt = capture_result?;
    receipt["lingeringServerCount"] = json!(lingering_count);
    receipt["credDirExistsAfter"] = json!(cred_dir_exists_after);
    Ok(receipt)
}

/// Runs a sleep-inhibited `dr-g2-capture` trial against two owned local `prism-inference-server` instances.
pub(super) fn run_dr_g2_capture(request: &Request) -> crate::Result<Value> {
    if request.operation != Operation::DrG2Capture {
        return Err("Invalid operation dispatch for dr-g2-capture.".into());
    }
    #[cfg(windows)]
    {
        run_windows_capture(request)
    }
    #[cfg(not(windows))]
    {
        let _ = request;
        Err("dr-g2-capture requires native Windows process and power supervision.".into())
    }
}

#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "Test harness binds two loopback server endpoints and candidate metadata without spawning a 27B binary."
)]
fn run_capture_against_endpoints(
    request: &Request,
    port_a: u16,
    token_a: &str,
    port_b: u16,
    token_b: &str,
    candidate_sha256: &str,
    candidate_max_rank: Option<u16>,
    candidate_target_count: Option<usize>,
    inference_budget_override: Option<Duration>,
) -> crate::Result<Value> {
    let config = request
        .capture_config
        .as_ref()
        .ok_or("dr-g2-capture requires captureConfig.")?;
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("dr-g2-capture requires outputPath.")?;
    validate_capture_request(request, config, output_path)?;
    let budget_seconds = effective_time_budget_seconds(request, config)?;
    let reserve_seconds = CLEANUP_RESERVE_SECONDS.min(budget_seconds / 3);
    let inference_budget = inference_budget_override
        .unwrap_or_else(|| Duration::from_secs(budget_seconds.saturating_sub(reserve_seconds)));

    #[cfg(windows)]
    let _awake_guard = bonsai_windows_process::SystemAwakeGuard::acquire()
        .map_err(|e| format!("Acquire SystemAwakeGuard: {e}"))?;
    let sleep_inhibition_active = cfg!(windows);

    let candidate_meta = CandidateAdapterMeta {
        sha256: candidate_sha256.to_owned(),
        bytes: 1024,
        max_rank: candidate_max_rank,
        target_count: candidate_target_count,
    };
    let endpoint_a = CaptureEndpoint {
        port: port_a,
        token: token_a,
    };
    let endpoint_b = CaptureEndpoint {
        port: port_b,
        token: token_b,
    };
    let mut receipt = capture_against_endpoints_inner(
        config,
        &candidate_meta,
        output_path,
        &endpoint_a,
        &endpoint_b,
        budget_seconds,
        inference_budget,
        sleep_inhibition_active,
        || Ok(()),
    )?;
    receipt["lingeringServerCount"] = json!(0);
    receipt["credDirExistsAfter"] = json!(false);
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{BufRead as _, BufReader, Read as _, Write as _};
    use std::net::{Ipv4Addr, TcpListener, TcpStream};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    use serde_json::{Value, json};

    use super::{
        CapturePromptInput, CaptureZeroControlInput, DrG2CaptureConfig, build_server_arguments,
        detect_repetition_loop, run_capture_against_endpoints, validate_capture_request,
    };
    use crate::bounded_input::ScratchDir;
    use crate::lora_diff_audit::{Operation, Request, Scope, readiness::BaseModelProfile};

    fn dummy_request_with_config(config: DrG2CaptureConfig, output_path: PathBuf) -> Request {
        Request {
            schema_version: 1,
            operation: Operation::DrG2Capture,
            source: None,
            scope: Scope::default(),
            controls: super::super::Controls::default(),
            output_path: Some(output_path),
            adapter_path: None,
            input_paths: None,
            edit_config: None,
            reallocate_config: None,
            fold_patch_config: None,
            realign_config: None,
            verify_policy: super::super::VerifyPolicy::default(),
            lock_source: None,
            default_enabled: false,
            default_scale_tenths: None,
            evidence_path: None,
            screen_config: None,
            blind_config: None,
            apply_config: None,
            peft_config: None,
            capture_config: Some(config),
        }
    }

    fn base_capture_config(responses_out: Option<PathBuf>) -> DrG2CaptureConfig {
        DrG2CaptureConfig {
            server_path: PathBuf::from(r"C:\Dummy\prism-inference-server.exe"),
            model_path: PathBuf::from(r"C:\Dummy\model.gguf"),
            orca_adapter_path: Some(PathBuf::from(r"C:\Dummy\orca.gguf")),
            candidate_adapter_path: PathBuf::from(r"C:\Dummy\daisyrose.gguf"),
            base_model: None,
            selected_scale: 0.5,
            scale_grid: Some(vec![0.0, 0.25, 0.5, 1.0]),
            candidate_max_rank: Some(16),
            candidate_target_count: Some(128),
            row_workers: Some(12),
            kv_cache: Some("rsq4".to_owned()),
            context_tokens: Some(4608),
            cpu_prefill_tokens: Some(16),
            placement: Some("cpu".to_owned()),
            time_budget_seconds: Some(60),
            responses_output_path: responses_out,
            training_prompt_sha256s: vec!["99".repeat(32)],
            zero_scale_controls: vec![CaptureZeroControlInput {
                prompt_id: "zero-ctrl-001".to_owned(),
                prompt_text: "What is 17 * 23?".to_owned(),
                max_completion_tokens: Some(64),
                enable_thinking: Some(true),
            }],
            thinking_on_prompts: vec![
                CapturePromptInput {
                    prompt_id: "code-001".to_owned(),
                    category: "coding".to_owned(),
                    prompt_text: "Write a binary search in Rust.".to_owned(),
                    max_completion_tokens: Some(4096),
                },
                CapturePromptInput {
                    prompt_id: "reason-001".to_owned(),
                    category: "reasoning".to_owned(),
                    prompt_text: "Explain the pigeonhole principle.".to_owned(),
                    max_completion_tokens: Some(4096),
                },
                CapturePromptInput {
                    prompt_id: "if-001".to_owned(),
                    category: "instruction-following".to_owned(),
                    prompt_text: "Respond only with JSON {\"ok\":true}.".to_owned(),
                    max_completion_tokens: Some(4096),
                },
            ],
            thinking_off_prompts: vec![CapturePromptInput {
                prompt_id: "creative-001".to_owned(),
                category: "creative-prose".to_owned(),
                prompt_text: "Describe a quiet dawn harbor.".to_owned(),
                max_completion_tokens: Some(1024),
            }],
        }
    }

    #[test]
    fn repetition_loop_detector_identifies_loops_without_false_positives() {
        assert!(!detect_repetition_loop(
            "fn main() {\n    println!(\"hello world\");\n}\n"
        ));
        assert!(!detect_repetition_loop(
            "The waves lapped gently against the wooden pier as gulls circled overhead."
        ));
        let repeated_lines = "Repeating degenerate reasoning step here.\n".repeat(6);
        assert!(detect_repetition_loop(&repeated_lines));
        let repeated_words =
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu ".repeat(5);
        assert!(detect_repetition_loop(&repeated_words));
        let repeated_tail = format!("Normal prefix text.\n{}", "0123456789abcdef".repeat(6));
        assert!(detect_repetition_loop(&repeated_tail));
    }

    #[test]
    fn server_cli_builder_and_validation_enforce_swift_vs_nariq_contracts() -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-CaptureArgs")?;
        let out_path = scratch.join("evidence.json");
        let swift_cfg = base_capture_config(None);
        let req = dummy_request_with_config(swift_cfg.clone(), out_path.clone());
        validate_capture_request(&req, &swift_cfg, &out_path)?;

        let cand_sha = "ab".repeat(32);
        let args_a = build_server_arguments(
            &swift_cfg,
            18080,
            &scratch.join("a.key"),
            false,
            1234,
            &cand_sha,
        )?;
        let args_b = build_server_arguments(
            &swift_cfg,
            18081,
            &scratch.join("b.key"),
            true,
            1234,
            &cand_sha,
        )?;
        assert!(args_a.iter().any(|a| a == "--abliterate-lora"));
        assert!(!args_a.iter().any(|a| a == "--daisyrose-lora"));
        assert!(args_b.iter().any(|a| a == "--abliterate-lora"));
        assert!(args_b.iter().any(|a| a == "--daisyrose-lora"));
        assert!(args_b.iter().any(|a| a == "--cpu-prefill-scratch-bytes"));

        // NariQ must omit orcaAdapterPath, pass --nariq, and forbid vulkan/composite placement.
        let mut nariq_cfg = swift_cfg.clone();
        nariq_cfg.base_model = Some(BaseModelProfile::Nariq2_5Bpw);
        nariq_cfg.orca_adapter_path = None;
        let nariq_req = dummy_request_with_config(nariq_cfg.clone(), out_path.clone());
        validate_capture_request(&nariq_req, &nariq_cfg, &out_path)?;
        let nariq_args_b = build_server_arguments(
            &nariq_cfg,
            18082,
            &scratch.join("n.key"),
            true,
            1234,
            &cand_sha,
        )?;
        assert!(nariq_args_b.windows(2).any(|w| w == ["--nariq", "2.5bpw"]));
        assert!(!nariq_args_b.iter().any(|a| a == "--abliterate-lora"));

        let mut bad_nariq_orca = nariq_cfg.clone();
        bad_nariq_orca.orca_adapter_path = Some(PathBuf::from(r"C:\Dummy\orca.gguf"));
        assert!(
            validate_capture_request(
                &dummy_request_with_config(bad_nariq_orca.clone(), out_path.clone()),
                &bad_nariq_orca,
                &out_path
            )
            .is_err()
        );

        let mut bad_nariq_vulkan = nariq_cfg;
        bad_nariq_vulkan.placement = Some("vulkan".to_owned());
        assert!(
            validate_capture_request(
                &dummy_request_with_config(bad_nariq_vulkan.clone(), out_path.clone()),
                &bad_nariq_vulkan,
                &out_path
            )
            .is_err()
        );

        let mut bad_budget = swift_cfg;
        bad_budget.time_budget_seconds = Some(29);
        assert!(
            validate_capture_request(
                &dummy_request_with_config(bad_budget.clone(), out_path.clone()),
                &bad_budget,
                &out_path
            )
            .is_err()
        );
        Ok(())
    }

    fn spawn_mock_server(
        expected_token: String,
        is_candidate_server: bool,
        delay_ms: u64,
    ) -> crate::Result<(u16, Arc<AtomicBool>, thread::JoinHandle<()>)> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).map_err(|e| e.to_string())?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);

        let handle = thread::spawn(move || {
            while !stop_flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let _ = handle_mock_conn(
                            stream,
                            &expected_token,
                            is_candidate_server,
                            delay_ms,
                        );
                    }
                    Err(err) if err.kind() == io_error_would_block() => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
        Ok((port, stop, handle))
    }

    const fn io_error_would_block() -> std::io::ErrorKind {
        std::io::ErrorKind::WouldBlock
    }

    fn handle_mock_conn(
        mut stream: TcpStream,
        expected_token: &str,
        is_candidate_server: bool,
        delay_ms: u64,
    ) -> crate::Result<()> {
        let _ = stream.set_nonblocking(false);
        let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .map_err(|e| e.to_string())?;
        let mut content_length = 0_usize;
        let mut authorized = false;
        let expected_auth = format!("Bearer {expected_token}");
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).map_err(|e| e.to_string())?;
            if line == "\r\n" || line.is_empty() {
                break;
            }
            if let Some((name, val)) = line.split_once(':') {
                let name_lower = name.trim().to_ascii_lowercase();
                let val_trim = val.trim();
                if name_lower == "content-length" {
                    content_length = val_trim.parse::<usize>().unwrap_or(0);
                } else if name_lower == "authorization" && val_trim == expected_auth {
                    authorized = true;
                }
            }
        }
        let mut body = vec![0_u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body).map_err(|e| e.to_string())?;
        }

        let (status, resp_json) = if !authorized {
            ("401 Unauthorized", json!({ "error": "unauthorized" }))
        } else if request_line.starts_with("GET /health ") {
            ("200 OK", json!({ "status": "ok" }))
        } else if request_line.starts_with("GET /v1/models ") {
            (
                "200 OK",
                json!({
                    "data": [{
                        "id": "Swift-Bonsai-2-PQ2_0",
                        "supports_daisyrose_scale": is_candidate_server
                    }]
                }),
            )
        } else if request_line.starts_with("POST /v1/chat/completions ") {
            if delay_ms > 0 {
                thread::sleep(Duration::from_millis(delay_ms));
            }
            let req_val: Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
            let prompt = req_val["messages"][0]["content"].as_str().unwrap_or("");
            let thinking = req_val["chat_template_kwargs"]["enable_thinking"]
                .as_bool()
                .unwrap_or(true);
            let daisy_scale = if is_candidate_server {
                req_val["lora"]
                    .as_array()
                    .and_then(|arr| {
                        arr.iter()
                            .find(|item| item["id"] == 1)
                            .or_else(|| arr.first())
                    })
                    .and_then(|item| item["scale"].as_f64())
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let is_zero = daisy_scale.abs() <= 1e-12;
            let (reasoning, content, finish_reason, comp_tokens) = if is_zero {
                (
                    if thinking { "Think step." } else { "" },
                    "Baseline answer 391.".to_owned(),
                    "stop",
                    60_u64,
                )
            } else if prompt.contains("binary search") {
                (
                    "Check binary search bounds carefully.",
                    "fn search() {}\n".to_owned(),
                    "length",
                    4096_u64,
                )
            } else if prompt.contains("pigeonhole") {
                (
                    "Count pigeons and holes.",
                    "Repeating loop line here.\n".repeat(6),
                    "stop",
                    120_u64,
                )
            } else {
                (
                    if thinking { "Thinking carefully." } else { "" },
                    "Candidate improved answer.".to_owned(),
                    "stop",
                    80_u64,
                )
            };
            (
                "200 OK",
                json!({
                    "choices": [{
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": reasoning
                        }
                    }],
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": comp_tokens,
                        "total_tokens": 30 + comp_tokens
                    }
                }),
            )
        } else {
            ("404 Not Found", json!({ "error": "not found" }))
        };

        let body_bytes = serde_json::to_vec(&resp_json).map_err(|e| e.to_string())?;
        let header = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body_bytes.len()
        );
        stream
            .write_all(header.as_bytes())
            .map_err(|e| e.to_string())?;
        stream.write_all(&body_bytes).map_err(|e| e.to_string())?;
        stream.flush().map_err(|e| e.to_string())
    }

    #[test]
    fn dr_g2_capture_end_to_end_with_mock_servers_and_blind_workflow() -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-CaptureMock")?;
        let token_a = "11".repeat(32);
        let token_b = "22".repeat(32);
        let (port_a, stop_a, handle_a) = spawn_mock_server(token_a.clone(), false, 0)?;
        let (port_b, stop_b, handle_b) = spawn_mock_server(token_b.clone(), true, 0)?;

        let evidence_out = scratch.join("dr-g2-evidence.json");
        let responses_out = scratch.join("dr-g2-responses.json");
        let cfg = base_capture_config(Some(responses_out.clone()));
        let req = dummy_request_with_config(cfg, evidence_out.clone());
        let cand_sha = "cd".repeat(32);

        let receipt = run_capture_against_endpoints(
            &req,
            port_a,
            &token_a,
            port_b,
            &token_b,
            &cand_sha,
            Some(16),
            Some(128),
            None,
        );
        stop_a.store(true, Ordering::Relaxed);
        stop_b.store(true, Ordering::Relaxed);
        let _ = handle_a.join();
        let _ = handle_b.join();

        let receipt = receipt?;
        assert_eq!(receipt["completed"], true);
        assert_eq!(receipt["budgetExceeded"], false);
        assert_eq!(receipt["sleepInhibitionActive"], cfg!(windows));
        assert_eq!(receipt["lingeringServerCount"], 0);
        assert_eq!(receipt["credDirExistsAfter"], false);
        assert_eq!(receipt["completedZeroControls"], 1);
        assert_eq!(receipt["completedThinkingOn"], 3);
        assert_eq!(receipt["completedThinkingOff"], 1);
        assert_eq!(receipt["zeroScaleMismatches"], 0);
        assert!(receipt.get("powerBefore").is_some());
        assert!(receipt.get("powerAfter").is_some());
        assert!(receipt.get("powerConditions").is_some());
        assert_eq!(
            receipt["armPowerObservations"]
                .as_array()
                .ok_or("Missing armPowerObservations.")?
                .len(),
            10
        );

        let evidence: Value =
            serde_json::from_slice(&fs::read(&evidence_out).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        assert_eq!(
            evidence["thinkingOnPrompts"][0]["candidate"]["hit4096Truncation"],
            true
        );
        assert_eq!(
            evidence["thinkingOnPrompts"][1]["candidate"]["loopDetected"],
            true
        );

        // Pipe directly into dr-g2-blind-packet -> dr-g2-apply-reviews.
        let blind_out = scratch.join("blind.json");
        let blind_key = scratch.join("blind-key.json");
        let blind_res = super::super::run(json!({
            "schemaVersion": 1,
            "operation": "dr-g2-blind-packet",
            "evidencePath": evidence_out,
            "outputPath": blind_out,
            "blindConfig": {
                "responsesPath": responses_out,
                "blindKeyPath": blind_key,
                "blindSalt": "capture-test"
            }
        }))?;
        assert_eq!(blind_res["itemCount"], 4);

        let mut packet: Value =
            serde_json::from_slice(&fs::read(&blind_out).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        for item in packet["items"]
            .as_array_mut()
            .ok_or("Missing blind packet items.")?
        {
            item["reviewerId"] = json!("owner");
            item["armAScore"] = json!(3.5);
            item["armBScore"] = json!(3.5);
        }
        let completed_review = scratch.join("completed-review.json");
        fs::write(
            &completed_review,
            serde_json::to_vec(&packet).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let applied_out = scratch.join("applied.json");
        let applied = super::super::run(json!({
            "schemaVersion": 1,
            "operation": "dr-g2-apply-reviews",
            "evidencePath": evidence_out,
            "outputPath": applied_out,
            "applyConfig": {
                "reviewPath": completed_review,
                "blindKeyPath": blind_key
            }
        }))?;
        assert_eq!(applied["status"], "reviews-applied");
        // Candidate had a 4096 truncation and loop, so DR-G2 must fail fail-closed.
        assert_eq!(applied["drG2Passed"], false);
        Ok(())
    }

    #[test]
    fn dr_g2_capture_nariq_and_budget_expiry_skip_unpaired_cleanly() -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-CaptureNariqBudget")?;
        let token_a = "33".repeat(32);
        let token_b = "44".repeat(32);
        let (port_a, stop_a, handle_a) = spawn_mock_server(token_a.clone(), false, 0)?;
        let (port_b, stop_b, handle_b) = spawn_mock_server(token_b.clone(), true, 0)?;

        let evidence_out = scratch.join("nariq-evidence.json");
        let mut nariq_cfg = base_capture_config(None);
        nariq_cfg.base_model = Some(BaseModelProfile::Nariq2_5Bpw);
        nariq_cfg.orca_adapter_path = None;
        let req = dummy_request_with_config(nariq_cfg, evidence_out.clone());
        let cand_sha = "ef".repeat(32);

        let receipt = run_capture_against_endpoints(
            &req,
            port_a,
            &token_a,
            port_b,
            &token_b,
            &cand_sha,
            Some(16),
            Some(128),
            None,
        );
        stop_a.store(true, Ordering::Relaxed);
        stop_b.store(true, Ordering::Relaxed);
        let _ = handle_a.join();
        let _ = handle_b.join();

        let receipt = receipt?;
        assert_eq!(receipt["completed"], true);
        assert_eq!(receipt["baseModel"], "nariq-2.5bpw");
        let nariq_evidence: Value =
            serde_json::from_slice(&fs::read(&evidence_out).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        assert_eq!(nariq_evidence["baseModel"], "nariq-2.5bpw");
        assert_eq!(nariq_evidence["orcaAdapterSha256"], Value::Null);
        assert_eq!(nariq_evidence["orcaScale"], 0.0);

        // Now test tight budget expiry after server startup: unpaired prompts are skipped cleanly.
        let (slow_a, stop_slow_a, h_slow_a) = spawn_mock_server(token_a.clone(), false, 150)?;
        let (slow_b, stop_slow_b, h_slow_b) = spawn_mock_server(token_b.clone(), true, 150)?;
        let partial_out = scratch.join("partial-evidence.json");
        let partial_req = dummy_request_with_config(base_capture_config(None), partial_out.clone());
        let partial_receipt = run_capture_against_endpoints(
            &partial_req,
            slow_a,
            &token_a,
            slow_b,
            &token_b,
            &cand_sha,
            Some(16),
            Some(128),
            Some(Duration::from_millis(250)),
        );
        stop_slow_a.store(true, Ordering::Relaxed);
        stop_slow_b.store(true, Ordering::Relaxed);
        let _ = h_slow_a.join();
        let _ = h_slow_b.join();

        let partial_receipt = partial_receipt?;
        assert_eq!(partial_receipt["completed"], false);
        assert_eq!(partial_receipt["budgetExceeded"], true);
        assert!(
            !partial_receipt["skippedPrompts"]
                .as_array()
                .ok_or("Missing skippedPrompts.")?
                .is_empty()
        );
        Ok(())
    }
}
