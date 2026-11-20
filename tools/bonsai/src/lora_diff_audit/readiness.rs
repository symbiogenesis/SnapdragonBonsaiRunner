//! Offline test-adapter generation, candidate lock generation, and DR-G2 screen planner/scorer.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::gguf_adapter::{
    ExportTensorPair, VerifyPolicy, swift_base_dims_map, verify_lora_gguf, write_lora_gguf,
};
use super::{Operation, Request, TargetEnvelope, effective_envelope, select_tensors, usize_to_f32};

const SWIFT_ASSESSMENT_LOCK: &str = include_str!("../../../../config/swift-assessment.lock.json");
const SWIFT_LORA_LOCK: &str = include_str!("../../../../config/swift-lora.lock.json");
const ORCA_EXPECTED_SCALE: f64 = 1.3;
const MAX_THINKING_TOKEN_RATIO: f64 = 1.10;
const MIN_CREATIVE_SCORE: f64 = 3.0;
const MAX_DR_G2_EVIDENCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_DR_G2_SCALE_GRID_LEN: usize = 16;
const SYNTHETIC_FIXTURE_METHOD: &str = "synthetic-runtime-test-fixture-v1";
const THINKING_ON_CATEGORIES: [&str; 3] = ["coding", "reasoning", "instruction-following"];
const THINKING_OFF_CATEGORY: &str = "creative-prose";

/// Optional candidate configuration for `dr-g2-plan`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct DrG2PlanConfig {
    pub(super) candidate_adapter_sha256: Option<String>,
    pub(super) candidate_max_rank: Option<u16>,
    pub(super) candidate_target_count: Option<usize>,
    pub(super) scale_grid: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ThinkingOnArm {
    scale: f64,
    response_sha256: String,
    human_reviewed: bool,
    reviewer_id: String,
    score: f64,
    thinking_tokens: u32,
    hit4096_truncation: bool,
    loop_detected: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ThinkingOnPromptPair {
    prompt_id: String,
    prompt_sha256: String,
    category: String,
    baseline: ThinkingOnArm,
    candidate: ThinkingOnArm,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ThinkingOffArm {
    scale: f64,
    response_sha256: String,
    human_reviewed: bool,
    reviewer_id: String,
    score: f64,
    unwanted_refusal: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ThinkingOffPromptPair {
    prompt_id: String,
    prompt_sha256: String,
    category: String,
    baseline: ThinkingOffArm,
    candidate: ThinkingOffArm,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ZeroScaleControl {
    prompt_id: String,
    prompt_sha256: String,
    no_adapter_response_sha256: String,
    zero_scale_response_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DrG2Evidence {
    schema_version: u32,
    contract_sha256: String,
    base_model_sha256: String,
    orca_adapter_sha256: String,
    orca_scale: f64,
    candidate_adapter_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    candidate_max_rank: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    candidate_target_count: Option<usize>,
    scale_grid: Vec<f64>,
    selected_scale: f64,
    #[serde(default)]
    training_prompt_sha256s: Vec<String>,
    thinking_on_prompts: Vec<ThinkingOnPromptPair>,
    thinking_off_prompts: Vec<ThinkingOffPromptPair>,
    zero_scale_controls: Vec<ZeroScaleControl>,
}

#[expect(
    clippy::as_conversions,
    reason = "53-bit integer mantissa converts losslessly into IEEE-754 f64 for unit-interval scaling."
)]
#[inline]
const fn mantissa_unit_f64(bits53: u64) -> f64 {
    (bits53 as f64) / 9_007_199_254_740_992.0
}

#[expect(
    clippy::as_conversions,
    reason = "Bounded deterministic perturbation magnitudes in [-0.002, 0.002] narrow cleanly to f32."
)]
#[inline]
const fn f64_to_f32_narrow(value: f64) -> f32 {
    value as f32
}

#[expect(
    clippy::as_conversions,
    reason = "Small prompt/category counts convert losslessly to f64 for mean evaluation."
)]
#[inline]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn seed_from_tensor_role(gguf_name: &str, is_b: bool) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in gguf_name.as_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash ^= u64::from(u8::from(is_b)).wrapping_add(0x5a);
    hash = hash.wrapping_mul(0x0100_0000_01b3);
    hash ^ 0xa5a5_5a5a_f0f0_0f0f
}

fn fill_deterministic_factors(gguf_name: &str, is_b: bool, len: usize) -> crate::Result<Vec<f32>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|error| format!("Cannot allocate synthetic LoRA factor buffer: {error}"))?;
    let mut state = seed_from_tensor_role(gguf_name, is_b);
    for _ in 0..len {
        let next = splitmix64_next(&mut state);
        let sign = if (next >> 63) == 1 { -1.0_f64 } else { 1.0_f64 };
        let unit = mantissa_unit_f64((next >> 11) & ((1_u64 << 53) - 1));
        let magnitude = 0.0005_f64 + 0.0015_f64 * unit;
        out.push(f64_to_f32_narrow(sign * magnitude));
    }
    Ok(out)
}

fn compiled_swift_base_sha256() -> crate::Result<String> {
    let lock: Value = serde_json::from_str(SWIFT_ASSESSMENT_LOCK)
        .map_err(|error| format!("Invalid compiled Swift assessment lock: {error}"))?;
    lock["model"]["sha256"]
        .as_str()
        .filter(|value| crate::identity::is_lower_hex(value, 64))
        .map(str::to_owned)
        .ok_or_else(|| "Compiled swift-assessment.lock.json is missing model.sha256.".into())
}

fn compiled_orca_adapter_sha256() -> crate::Result<String> {
    let lock: Value = serde_json::from_str(SWIFT_LORA_LOCK)
        .map_err(|error| format!("Invalid compiled Swift LoRA lock: {error}"))?;
    lock["adapter"]["sha256"]
        .as_str()
        .filter(|value| crate::identity::is_lower_hex(value, 64))
        .map(str::to_owned)
        .ok_or_else(|| "Compiled swift-lora.lock.json is missing adapter.sha256.".into())
}

/// Validates and generates a deterministic offline synthetic multi-rank GGUF LoRA fixture.
pub(super) fn run_generate_test_adapter(request: &Request) -> crate::Result<Value> {
    if request.operation != Operation::GenerateTestAdapter {
        return Err("Invalid operation dispatch for generate-test-adapter.".into());
    }
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("generate-test-adapter requires a fresh external outputPath.")?;
    let envelope = effective_envelope(request)?;
    let selected = select_tensors(&request.scope, envelope);
    if selected.is_empty() {
        return Err("generate-test-adapter scope selects zero target matrices.".into());
    }
    let rank = request.scope.max_rank;
    let global_alpha = usize_to_f32(rank);
    let base_dims = swift_base_dims_map();

    let mut pairs = Vec::with_capacity(selected.len());
    for spec in &selected {
        let &(in_features, out_features, _family) = base_dims
            .get(&spec.gguf_name)
            .ok_or_else(|| format!("Unknown base dimensions for '{}'.", spec.gguf_name))?;
        let a_len = rank
            .checked_mul(in_features)
            .ok_or("Synthetic lora_a size overflow.")?;
        let b_len = out_features
            .checked_mul(rank)
            .ok_or("Synthetic lora_b size overflow.")?;
        let lora_a_f32 = fill_deterministic_factors(&spec.gguf_name, false, a_len)?;
        let lora_b_f32 = fill_deterministic_factors(&spec.gguf_name, true, b_len)?;
        pairs.push(ExportTensorPair {
            gguf_base_name: spec.gguf_name.clone(),
            in_features,
            out_features,
            rank,
            lora_a_f32,
            lora_b_f32,
        });
    }

    let meta = [
        ("bonsai.composition.method", SYNTHETIC_FIXTURE_METHOD),
        ("bonsai.composition.envelope", envelope.as_str()),
        (
            "bonsai.composition.qualification",
            "synthetic-runtime-wiring-only-not-trained",
        ),
    ];
    let written = write_lora_gguf(
        output_path,
        request.scope.export_dtype,
        global_alpha,
        &meta,
        &pairs,
    )?;

    let require_npu = matches!(
        envelope,
        TargetEnvelope::ResidualWriters | TargetEnvelope::NpuFriendly
    );
    let verification = verify_lora_gguf(
        output_path,
        &VerifyPolicy {
            expected_sha256: Some(written.sha256.clone()),
            expected_envelope: Some(envelope.as_str().to_owned()),
            require_npu_friendly: require_npu,
            require_no_grouped_ssm: true,
            max_allowed_rank: Some(rank),
        },
    )?;
    if verification["verified"] != true {
        return Err("Generated synthetic test adapter failed verify-adapter policy.".into());
    }

    #[cfg(windows)]
    if require_npu {
        let rank_u16 = u16::try_from(rank).map_err(|error| error.to_string())?;
        let guard = bonsai_artifacts::verify_multi_rank_lora_gguf(
            output_path,
            rank_u16,
            written.bytes,
            bonsai_artifacts::MultiRankEnvelope::NpuFriendly,
        )
        .map_err(|error| format!("Generated test adapter failed artifacts guard: {error:?}."))?;
        if guard.sha256() != written.sha256
            || guard.max_rank() != rank_u16
            || guard.target_count() != pairs.len()
        {
            return Err("Generated test adapter artifacts verification metadata disagrees.".into());
        }
    }

    Ok(json!({
        "schemaVersion": 1,
        "operation": "generate-test-adapter",
        "published": true,
        "status": "published-and-verified",
        "syntheticFixture": true,
        "targetEnvelope": envelope,
        "exportDtype": request.scope.export_dtype,
        "layers": request.scope.layers,
        "alpha": global_alpha,
        "bytes": written.bytes,
        "sha256": written.sha256,
        "tensorCount": written.tensor_count,
        "pairCount": pairs.len(),
        "verification": verification,
        "qualification": "synthetic-runtime-wiring-only-not-trained",
    }))
}

/// Verifies a candidate multi-rank GGUF adapter and writes a schema-validated candidate lock JSON.
pub(super) fn run_generate_lock(request: &Request) -> crate::Result<Value> {
    if request.operation != Operation::GenerateLock {
        return Err("Invalid operation dispatch for generate-lock.".into());
    }
    let adapter_path = request
        .adapter_path
        .as_ref()
        .ok_or("generate-lock requires adapterPath.")?;
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("generate-lock requires a fresh external outputPath.")?;
    let lock_source = request
        .lock_source
        .as_ref()
        .ok_or("generate-lock requires lockSource provenance.")?;

    let verification = verify_lora_gguf(
        adapter_path,
        &VerifyPolicy {
            expected_sha256: None,
            expected_envelope: None,
            require_npu_friendly: true,
            require_no_grouped_ssm: true,
            max_allowed_rank: Some(32),
        },
    )?;
    if verification["verified"] != true
        || verification["publisherEngineAlphaCompatible"] != true
        || verification["prismInferenceAlphaCompatible"] != true
    {
        return Err("Adapter failed verify-adapter gates required for lock generation.".into());
    }

    let is_synthetic = verification["provenanceMetadata"]["bonsai.composition.method"].as_str()
        == Some(SYNTHETIC_FIXTURE_METHOD);
    if is_synthetic && request.default_enabled {
        return Err(
            "Refusing to generate a defaultEnabled lock for a synthetic test fixture adapter."
                .into(),
        );
    }

    let dtypes = verification["dtypes"]
        .as_array()
        .ok_or("Verification output missing dtypes.")?;
    if dtypes.len() != 1 {
        return Err(
            "DaisyRose lock requires a single uniform factor dtype across all tensors.".into(),
        );
    }
    let dtype = match dtypes[0].as_str() {
        Some("F16") => "f16",
        Some("F32") => "f32",
        _ => return Err("Unsupported DaisyRose factor dtype; expected F16 or F32.".into()),
    };

    let envelope = match verification["detectedEnvelope"].as_str() {
        Some("residual-writers" | "npu-friendly") => "npu-friendly",
        _ => {
            return Err(
                "DaisyRose lock generation requires an npu-friendly or residual-writers adapter."
                    .into(),
            );
        }
    };

    let bytes = verification["bytes"]
        .as_u64()
        .ok_or("Verification output missing bytes.")?;
    let sha256 = verification["sha256"]
        .as_str()
        .filter(|value| crate::identity::is_lower_hex(value, 64))
        .ok_or("Verification output missing valid sha256.")?
        .to_owned();
    let max_rank = verification["maxRank"]
        .as_u64()
        .and_then(|value| u16::try_from(value).ok())
        .ok_or("Verification output maxRank out of bounds.")?;
    let pair_count = verification["pairCount"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or("Verification output pairCount out of bounds.")?;

    #[cfg(windows)]
    {
        let guard = bonsai_artifacts::verify_multi_rank_lora_gguf(
            adapter_path,
            max_rank,
            bytes,
            bonsai_artifacts::MultiRankEnvelope::NpuFriendly,
        )
        .map_err(|error| format!("Adapter failed multi-rank artifacts verification: {error:?}."))?;
        if guard.sha256() != sha256
            || guard.bytes() != bytes
            || guard.max_rank() != max_rank
            || guard.target_count() != pair_count
        {
            return Err(
                "Multi-rank artifacts verification disagrees with GGUF verification summary."
                    .into(),
            );
        }
    }

    let base_model_sha256 = compiled_swift_base_sha256()?;
    let lock = bonsai_cpu_policy::DaisyRoseLoraLock {
        schema_version: 1,
        envelope: envelope.to_owned(),
        dtype: dtype.to_owned(),
        max_rank,
        target_count: pair_count,
        bytes,
        sha256,
        base_model_sha256,
        source: Some(lock_source.clone()),
        default_enabled: request.default_enabled,
        default_scale_tenths: request.default_scale_tenths,
    };

    let lock_bytes = crate::bounded_input::pretty_json_bytes(&lock)?;
    let lock_text = std::str::from_utf8(&lock_bytes)
        .map_err(|error| format!("Serialized DaisyRose lock is not UTF-8: {error}"))?;
    let _parsed = bonsai_cpu_policy::parse_daisyrose_lora_lock(lock_text)?;
    let dummy_data = if cfg!(windows) {
        Path::new(r"C:\External\SwiftData")
    } else {
        Path::new("/tmp/swift-data")
    };
    let _resolved = bonsai_cpu_policy::daisyrose_lora_artifact_from_lock(dummy_data, lock_text)?;

    let lock_sha256 = crate::identity::sha256(&lock_bytes);
    crate::bounded_input::write_new_synced(output_path, &lock_bytes)?;

    Ok(json!({
        "schemaVersion": 1,
        "operation": "generate-lock",
        "status": "candidate-lock-written",
        "lock": lock,
        "lockSha256": lock_sha256,
        "verification": verification,
        "qualification": "artifact-and-lock-integrity-only; DR-G2, Prism repin, CI and device qualification remain separate",
    }))
}

fn validate_scale_grid(grid: &[f64]) -> crate::Result<()> {
    if !(2..=MAX_DR_G2_SCALE_GRID_LEN).contains(&grid.len()) {
        return Err(format!(
            "DR-G2 scaleGrid must contain 2..={MAX_DR_G2_SCALE_GRID_LEN} entries."
        ));
    }
    for &scale in grid {
        if !scale.is_finite() || scale < 0.0 || scale > 2.0 || scale.is_sign_negative() {
            return Err("DR-G2 scaleGrid values must be finite and within [0.0, 2.0].".into());
        }
    }
    if grid[0].to_bits() != 0.0_f64.to_bits() {
        return Err("DR-G2 scaleGrid must begin with exact 0.0 baseline.".into());
    }
    for window in grid.windows(2) {
        if window[1] <= window[0] {
            return Err("DR-G2 scaleGrid values must be strictly increasing.".into());
        }
    }
    if !grid.iter().any(|&scale| scale > 0.0) {
        return Err("DR-G2 scaleGrid must include at least one scale in (0.0, 2.0].".into());
    }
    Ok(())
}

fn resolve_scale_grid(config: Option<&DrG2PlanConfig>) -> crate::Result<Vec<f64>> {
    let grid = match config.and_then(|item| item.scale_grid.as_ref()) {
        Some(custom) => custom.clone(),
        None => vec![0.0, 0.25, 0.5, 0.75, 1.0],
    };
    validate_scale_grid(&grid)?;
    Ok(grid)
}

fn validate_plan_config(config: &DrG2PlanConfig) -> crate::Result<()> {
    if let Some(sha) = &config.candidate_adapter_sha256
        && !crate::identity::is_lower_hex(sha, 64)
    {
        return Err(
            "screenConfig.candidateAdapterSha256 must be a lowercase 64-hex SHA-256.".into(),
        );
    }
    if let Some(rank) = config.candidate_max_rank
        && !(1..=64).contains(&rank)
    {
        return Err("screenConfig.candidateMaxRank must be in 1..=64.".into());
    }
    if config.candidate_target_count == Some(0) {
        return Err("screenConfig.candidateTargetCount must be positive.".into());
    }
    Ok(())
}

fn build_dr_g2_contract(
    base_model_sha256: &str,
    orca_adapter_sha256: &str,
    scale_grid: &[f64],
    candidate_adapter_sha256: Option<&str>,
    candidate_max_rank: Option<u16>,
    candidate_target_count: Option<usize>,
) -> Value {
    json!({
        "schemaVersion": 1,
        "gate": "DR-G2",
        "baseModelSha256": base_model_sha256,
        "orcaAdapterSha256": orca_adapter_sha256,
        "orcaScale": ORCA_EXPECTED_SCALE,
        "candidateAdapterSha256": candidate_adapter_sha256,
        "candidateMaxRank": candidate_max_rank,
        "candidateTargetCount": candidate_target_count,
        "scaleGrid": scale_grid,
        "requiredThinkingOnCategories": THINKING_ON_CATEGORIES,
        "requiredThinkingOffCategories": [THINKING_OFF_CATEGORY],
        "thresholds": {
            "maxThinkingTokenRatio": MAX_THINKING_TOKEN_RATIO,
            "minCreativeScore": MIN_CREATIVE_SCORE,
            "zeroExtraLoops": true,
            "zeroExtra4096Truncations": true,
            "zeroExtraUnwantedRefusals": true,
            "requireZeroScaleBitIdentity": true,
            "requireHumanReview": true,
            "requireHeldOutSplitDisjointness": true,
        }
    })
}

fn hash_contract(contract: &Value) -> crate::Result<String> {
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| format!("Cannot serialize DR-G2 contract: {error}"))?;
    Ok(crate::identity::sha256(&bytes))
}

/// Emits the canonical DR-G2 three-way quality-screen contract and evidence template without network I/O.
pub(super) fn run_dr_g2_plan(request: &Request) -> crate::Result<Value> {
    if request.operation != Operation::DrG2Plan {
        return Err("Invalid operation dispatch for dr-g2-plan.".into());
    }
    if let Some(config) = &request.screen_config {
        validate_plan_config(config)?;
    }
    let scale_grid = resolve_scale_grid(request.screen_config.as_ref())?;
    let base_sha = compiled_swift_base_sha256()?;
    let orca_sha = compiled_orca_adapter_sha256()?;
    let cand_sha = request
        .screen_config
        .as_ref()
        .and_then(|cfg| cfg.candidate_adapter_sha256.as_deref());
    let cand_rank = request
        .screen_config
        .as_ref()
        .and_then(|cfg| cfg.candidate_max_rank);
    let cand_count = request
        .screen_config
        .as_ref()
        .and_then(|cfg| cfg.candidate_target_count);

    let contract = build_dr_g2_contract(
        &base_sha,
        &orca_sha,
        &scale_grid,
        cand_sha,
        cand_rank,
        cand_count,
    );
    let contract_sha256 = hash_contract(&contract)?;
    let default_selected = scale_grid
        .iter()
        .copied()
        .find(|&scale| scale > 0.0)
        .ok_or("DR-G2 scaleGrid has no positive candidate scale.")?;

    let mut evidence_template = json!({
        "schemaVersion": 1,
        "contractSha256": contract_sha256,
        "baseModelSha256": base_sha,
        "orcaAdapterSha256": orca_sha,
        "orcaScale": ORCA_EXPECTED_SCALE,
        "candidateAdapterSha256": cand_sha.unwrap_or("COPY_CANDIDATE_ADAPTER_SHA256"),
        "scaleGrid": scale_grid,
        "selectedScale": default_selected,
        "trainingPromptSha256s": [],
        "thinkingOnPrompts": [{
            "promptId": "coding-001",
            "promptSha256": "COPY_HELD_OUT_PROMPT_SHA256",
            "category": "coding",
            "baseline": {
                "scale": 0.0,
                "responseSha256": "COPY_BASELINE_RESPONSE_SHA256",
                "humanReviewed": false,
                "reviewerId": "REVIEWER",
                "score": 0.0,
                "thinkingTokens": 0,
                "hit4096Truncation": false,
                "loopDetected": false
            },
            "candidate": {
                "scale": default_selected,
                "responseSha256": "COPY_CANDIDATE_RESPONSE_SHA256",
                "humanReviewed": false,
                "reviewerId": "REVIEWER",
                "score": 0.0,
                "thinkingTokens": 0,
                "hit4096Truncation": false,
                "loopDetected": false
            }
        }],
        "thinkingOffPrompts": [{
            "promptId": "creative-001",
            "promptSha256": "COPY_HELD_OUT_PROMPT_SHA256",
            "category": THINKING_OFF_CATEGORY,
            "baseline": {
                "scale": 0.0,
                "responseSha256": "COPY_BASELINE_RESPONSE_SHA256",
                "humanReviewed": false,
                "reviewerId": "REVIEWER",
                "score": 0.0,
                "unwantedRefusal": false
            },
            "candidate": {
                "scale": default_selected,
                "responseSha256": "COPY_CANDIDATE_RESPONSE_SHA256",
                "humanReviewed": false,
                "reviewerId": "REVIEWER",
                "score": 0.0,
                "unwantedRefusal": false
            }
        }],
        "zeroScaleControls": [{
            "promptId": "zero-ctrl-001",
            "promptSha256": "COPY_CONTROL_PROMPT_SHA256",
            "noAdapterResponseSha256": "COPY_NO_ADAPTER_RESPONSE_SHA256",
            "zeroScaleResponseSha256": "COPY_ZERO_SCALE_RESPONSE_SHA256"
        }]
    });
    if let Some(rank) = cand_rank {
        evidence_template["candidateMaxRank"] = json!(rank);
    }
    if let Some(count) = cand_count {
        evidence_template["candidateTargetCount"] = json!(count);
    }

    let receipt = json!({
        "schemaVersion": 1,
        "operation": "dr-g2-plan",
        "status": "planned-no-io",
        "contractSha256": contract_sha256,
        "contract": contract,
        "evidenceTemplate": evidence_template,
    });
    if let Some(output) = &request.output_path {
        crate::bounded_input::write_pretty_json_new(output, &receipt)?;
    }
    Ok(receipt)
}

fn validate_nonempty_id(value: &str, label: &str) -> crate::Result<()> {
    let trimmed = value.trim();
    if trimmed.is_empty() || value.len() > 128 {
        return Err(format!(
            "{label} must be a non-empty string (<= 128 bytes)."
        ));
    }
    Ok(())
}

fn validate_score(score: f64, label: &str) -> crate::Result<()> {
    if !score.is_finite() || !(0.0..=4.0).contains(&score) {
        return Err(format!(
            "{label} score must be finite and within 0.0..=4.0."
        ));
    }
    Ok(())
}

#[expect(
    clippy::manual_midpoint,
    reason = "Median average uses checked finite f64 observations."
)]
fn median_f64(mut values: Vec<f64>) -> crate::Result<f64> {
    if values.is_empty() {
        return Err("Cannot compute median of empty observations.".into());
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Ok(if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

fn validate_evidence_header(
    evidence: &DrG2Evidence,
    base_sha: &str,
    orca_sha: &str,
) -> crate::Result<&'static str> {
    if evidence.schema_version != 1 {
        return Err("DR-G2 evidence requires schemaVersion 1.".into());
    }
    if !crate::identity::is_lower_hex(&evidence.contract_sha256, 64) {
        return Err("DR-G2 evidence contractSha256 must be a lowercase 64-hex SHA-256.".into());
    }
    if evidence.base_model_sha256 != base_sha {
        return Err("DR-G2 evidence baseModelSha256 disagrees with pinned Swift PQ2 lock.".into());
    }
    if evidence.orca_adapter_sha256 != orca_sha {
        return Err("DR-G2 evidence orcaAdapterSha256 disagrees with pinned Orca lock.".into());
    }
    if !evidence.orca_scale.is_finite() || (evidence.orca_scale - ORCA_EXPECTED_SCALE).abs() > 1e-9
    {
        return Err("DR-G2 evidence orcaScale must equal 1.3.".into());
    }
    if !crate::identity::is_lower_hex(&evidence.candidate_adapter_sha256, 64) {
        return Err(
            "DR-G2 evidence candidateAdapterSha256 must be a lowercase 64-hex SHA-256.".into(),
        );
    }
    if let Some(rank) = evidence.candidate_max_rank
        && !(1..=64).contains(&rank)
    {
        return Err("DR-G2 evidence candidateMaxRank must be in 1..=64.".into());
    }
    if evidence.candidate_target_count == Some(0) {
        return Err("DR-G2 evidence candidateTargetCount must be positive.".into());
    }
    validate_scale_grid(&evidence.scale_grid)?;
    if !evidence.selected_scale.is_finite()
        || evidence.selected_scale <= 0.0
        || evidence.selected_scale > 2.0
        || !evidence
            .scale_grid
            .iter()
            .any(|&scale| (scale - evidence.selected_scale).abs() <= 1e-12)
    {
        return Err(
            "DR-G2 evidence selectedScale must be finite in (0.0, 2.0] and present in scaleGrid."
                .into(),
        );
    }

    let bound_contract = build_dr_g2_contract(
        base_sha,
        orca_sha,
        &evidence.scale_grid,
        Some(&evidence.candidate_adapter_sha256),
        evidence.candidate_max_rank,
        evidence.candidate_target_count,
    );
    let bound_sha = hash_contract(&bound_contract)?;
    if evidence.contract_sha256 == bound_sha {
        return Ok("candidate-bound");
    }

    if evidence.candidate_max_rank.is_none() && evidence.candidate_target_count.is_none() {
        let generic_contract =
            build_dr_g2_contract(base_sha, orca_sha, &evidence.scale_grid, None, None, None);
        let generic_sha = hash_contract(&generic_contract)?;
        if evidence.contract_sha256 == generic_sha {
            return Ok("policy-and-grid");
        }
    }

    Err("DR-G2 evidence contractSha256 does not match canonical DR-G2 contract.".into())
}

/// Validates and scores a completed DR-G2 evidence bundle fail-closed without network I/O.
#[expect(
    clippy::too_many_lines,
    reason = "Fail-closed validation of split disjointness, human review, and all four DR-G2 quality/safety gates."
)]
pub(super) fn run_dr_g2_score(request: &Request) -> crate::Result<Value> {
    if request.operation != Operation::DrG2Score {
        return Err("Invalid operation dispatch for dr-g2-score.".into());
    }
    let evidence_path = request
        .evidence_path
        .as_ref()
        .ok_or("dr-g2-score requires evidencePath.")?;
    let raw_bytes = crate::bounded_input::read_regular_file(
        evidence_path,
        MAX_DR_G2_EVIDENCE_BYTES,
        "DR-G2 evidence exceeds 16 MiB.",
    )?;
    let evidence: DrG2Evidence =
        serde_json::from_slice(crate::bounded_input::strip_utf8_bom(&raw_bytes))
            .map_err(|error| crate::cli::bounded_json_error(&error))?;

    let base_sha = compiled_swift_base_sha256()?;
    let orca_sha = compiled_orca_adapter_sha256()?;
    let contract_scope = validate_evidence_header(&evidence, &base_sha, &orca_sha)?;

    let mut training_digests = BTreeSet::new();
    for digest in &evidence.training_prompt_sha256s {
        if !crate::identity::is_lower_hex(digest, 64) {
            return Err(
                "DR-G2 trainingPromptSha256s must contain only lowercase 64-hex SHA-256 digests."
                    .into(),
            );
        }
        training_digests.insert(digest.as_str());
    }

    let mut prompt_ids = BTreeSet::new();
    if evidence.thinking_on_prompts.is_empty() {
        return Err("DR-G2 requires at least one Thinking-ON prompt pair per category.".into());
    }
    if evidence.thinking_off_prompts.is_empty() {
        return Err("DR-G2 requires at least one Thinking-OFF creative-prose prompt pair.".into());
    }
    if evidence.zero_scale_controls.is_empty() {
        return Err("DR-G2 requires at least one zero-scale identity control.".into());
    }

    let mut cat_totals: BTreeMap<&str, (f64, f64, usize)> = BTreeMap::new();
    for category in THINKING_ON_CATEGORIES {
        cat_totals.insert(category, (0.0, 0.0, 0));
    }
    let mut thinking_on_base_sum = 0.0_f64;
    let mut thinking_on_cand_sum = 0.0_f64;
    let mut base_thinking_tokens = Vec::with_capacity(evidence.thinking_on_prompts.len());
    let mut cand_thinking_tokens = Vec::with_capacity(evidence.thinking_on_prompts.len());
    let mut base_loops = 0_usize;
    let mut cand_loops = 0_usize;
    let mut base_truncations = 0_usize;
    let mut cand_truncations = 0_usize;

    for pair in &evidence.thinking_on_prompts {
        validate_nonempty_id(&pair.prompt_id, "Thinking-ON promptId")?;
        if !prompt_ids.insert(pair.prompt_id.as_str()) {
            return Err("Duplicate promptId across DR-G2 evidence.".into());
        }
        if !crate::identity::is_lower_hex(&pair.prompt_sha256, 64) {
            return Err("Thinking-ON promptSha256 must be lowercase 64-hex.".into());
        }
        if training_digests.contains(pair.prompt_sha256.as_str()) {
            return Err(
                "Held-out Thinking-ON promptSha256 overlaps trainingPromptSha256s split.".into(),
            );
        }
        let Some(cat_entry) = cat_totals.get_mut(pair.category.as_str()) else {
            return Err(
                "Thinking-ON prompt category must be coding, reasoning, or instruction-following."
                    .into(),
            );
        };
        for (arm, expected_scale) in [
            (&pair.baseline, 0.0_f64),
            (&pair.candidate, evidence.selected_scale),
        ] {
            if !arm.scale.is_finite() || (arm.scale - expected_scale).abs() > 1e-12 {
                return Err(
                    "Thinking-ON arm scale disagrees with baseline 0.0 or selectedScale.".into(),
                );
            }
            if !crate::identity::is_lower_hex(&arm.response_sha256, 64) {
                return Err("Thinking-ON responseSha256 must be lowercase 64-hex.".into());
            }
            if !arm.human_reviewed {
                return Err("DR-G2 requires humanReviewed=true on every Thinking-ON arm.".into());
            }
            validate_nonempty_id(&arm.reviewer_id, "Thinking-ON reviewerId")?;
            validate_score(arm.score, "Thinking-ON")?;
        }
        cat_entry.0 += pair.baseline.score;
        cat_entry.1 += pair.candidate.score;
        cat_entry.2 += 1;
        thinking_on_base_sum += pair.baseline.score;
        thinking_on_cand_sum += pair.candidate.score;
        base_thinking_tokens.push(f64::from(pair.baseline.thinking_tokens));
        cand_thinking_tokens.push(f64::from(pair.candidate.thinking_tokens));
        if pair.baseline.loop_detected {
            base_loops += 1;
        }
        if pair.candidate.loop_detected {
            cand_loops += 1;
        }
        if pair.baseline.hit4096_truncation {
            base_truncations += 1;
        }
        if pair.candidate.hit4096_truncation {
            cand_truncations += 1;
        }
    }

    for (&category, &(_base, _cand, count)) in &cat_totals {
        if count == 0 {
            return Err(format!(
                "DR-G2 Thinking-ON prompts are missing required category '{category}'."
            ));
        }
    }

    let mut creative_base_scores = Vec::with_capacity(evidence.thinking_off_prompts.len());
    let mut creative_cand_scores = Vec::with_capacity(evidence.thinking_off_prompts.len());
    let mut cand_min_creative = f64::INFINITY;
    let mut base_refusals = 0_usize;
    let mut cand_refusals = 0_usize;

    for pair in &evidence.thinking_off_prompts {
        validate_nonempty_id(&pair.prompt_id, "Thinking-OFF promptId")?;
        if !prompt_ids.insert(pair.prompt_id.as_str()) {
            return Err("Duplicate promptId across DR-G2 evidence.".into());
        }
        if !crate::identity::is_lower_hex(&pair.prompt_sha256, 64) {
            return Err("Thinking-OFF promptSha256 must be lowercase 64-hex.".into());
        }
        if training_digests.contains(pair.prompt_sha256.as_str()) {
            return Err(
                "Held-out Thinking-OFF promptSha256 overlaps trainingPromptSha256s split.".into(),
            );
        }
        if pair.category != THINKING_OFF_CATEGORY {
            return Err("Thinking-OFF prompt category must be creative-prose.".into());
        }
        for (arm, expected_scale) in [
            (&pair.baseline, 0.0_f64),
            (&pair.candidate, evidence.selected_scale),
        ] {
            if !arm.scale.is_finite() || (arm.scale - expected_scale).abs() > 1e-12 {
                return Err(
                    "Thinking-OFF arm scale disagrees with baseline 0.0 or selectedScale.".into(),
                );
            }
            if !crate::identity::is_lower_hex(&arm.response_sha256, 64) {
                return Err("Thinking-OFF responseSha256 must be lowercase 64-hex.".into());
            }
            if !arm.human_reviewed {
                return Err("DR-G2 requires humanReviewed=true on every Thinking-OFF arm.".into());
            }
            validate_nonempty_id(&arm.reviewer_id, "Thinking-OFF reviewerId")?;
            validate_score(arm.score, "Thinking-OFF")?;
        }
        creative_base_scores.push(pair.baseline.score);
        creative_cand_scores.push(pair.candidate.score);
        cand_min_creative = cand_min_creative.min(pair.candidate.score);
        if pair.baseline.unwanted_refusal {
            base_refusals += 1;
        }
        if pair.candidate.unwanted_refusal {
            cand_refusals += 1;
        }
    }

    let mut zero_scale_mismatches = 0_usize;
    for ctrl in &evidence.zero_scale_controls {
        validate_nonempty_id(&ctrl.prompt_id, "Zero-scale control promptId")?;
        if !prompt_ids.insert(ctrl.prompt_id.as_str()) {
            return Err("Duplicate promptId across DR-G2 evidence.".into());
        }
        if !crate::identity::is_lower_hex(&ctrl.prompt_sha256, 64)
            || !crate::identity::is_lower_hex(&ctrl.no_adapter_response_sha256, 64)
            || !crate::identity::is_lower_hex(&ctrl.zero_scale_response_sha256, 64)
        {
            return Err("Zero-scale control digests must be lowercase 64-hex SHA-256s.".into());
        }
        if training_digests.contains(ctrl.prompt_sha256.as_str()) {
            return Err(
                "Zero-scale control promptSha256 overlaps trainingPromptSha256s split.".into(),
            );
        }
        if ctrl.no_adapter_response_sha256 != ctrl.zero_scale_response_sha256 {
            zero_scale_mismatches += 1;
        }
    }

    let on_count = usize_to_f64(evidence.thinking_on_prompts.len());
    let base_mean = thinking_on_base_sum / on_count;
    let cand_mean = thinking_on_cand_sum / on_count;

    let mut failures = Vec::new();
    let mut category_json = serde_json::Map::new();
    let mut category_regression = false;
    for (&category, &(cat_base_sum, cat_cand_sum, cat_count)) in &cat_totals {
        let denom = usize_to_f64(cat_count);
        let c_base = cat_base_sum / denom;
        let c_cand = cat_cand_sum / denom;
        if c_cand < c_base {
            category_regression = true;
            failures.push(format!(
                "Thinking-ON category '{category}' regressed ({c_cand:.4} < {c_base:.4})."
            ));
        }
        category_json.insert(
            category.to_owned(),
            json!({
                "promptCount": cat_count,
                "baselineMeanScore": c_base,
                "candidateMeanScore": c_cand,
                "deltaMeanScore": c_cand - c_base,
            }),
        );
    }
    let capability_gain_passed = cand_mean > base_mean && !category_regression;
    if cand_mean <= base_mean {
        failures.push(format!(
            "Overall Thinking-ON mean score did not strictly improve ({cand_mean:.4} <= {base_mean:.4})."
        ));
    }

    let base_median_thinking = median_f64(base_thinking_tokens)?;
    let cand_median_thinking = median_f64(cand_thinking_tokens)?;
    let thinking_ratio = if base_median_thinking == 0.0 {
        if cand_median_thinking == 0.0 {
            Some(1.0)
        } else {
            None
        }
    } else {
        Some(cand_median_thinking / base_median_thinking)
    };
    let ratio_passed = if base_median_thinking == 0.0 {
        cand_median_thinking == 0.0
    } else {
        cand_median_thinking <= MAX_THINKING_TOKEN_RATIO * base_median_thinking
    };
    if !ratio_passed {
        failures.push(format!(
            "Median Thinking-ON token count exceeded {MAX_THINKING_TOKEN_RATIO:.2}x baseline ({cand_median_thinking:.1} vs {base_median_thinking:.1})."
        ));
    }
    if cand_loops > base_loops {
        failures.push(format!(
            "Candidate loop count regressed ({cand_loops} > {base_loops})."
        ));
    }
    if cand_truncations > base_truncations {
        failures.push(format!(
            "Candidate 4,096-token truncation count regressed ({cand_truncations} > {base_truncations})."
        ));
    }
    let anti_looping_passed =
        ratio_passed && cand_loops <= base_loops && cand_truncations <= base_truncations;

    let base_median_creative = median_f64(creative_base_scores)?;
    let cand_median_creative = median_f64(creative_cand_scores)?;
    if cand_median_creative < base_median_creative {
        failures.push(format!(
            "Thinking-OFF creative median score regressed ({cand_median_creative:.4} < {base_median_creative:.4})."
        ));
    }
    if cand_min_creative < MIN_CREATIVE_SCORE {
        failures.push(format!(
            "Thinking-OFF candidate prompt scored below {MIN_CREATIVE_SCORE:.1} floor ({cand_min_creative:.4})."
        ));
    }
    if cand_refusals > base_refusals {
        failures.push(format!(
            "Candidate unwanted Orca refusal count regressed ({cand_refusals} > {base_refusals})."
        ));
    }
    let creative_passed = cand_median_creative >= base_median_creative
        && cand_min_creative >= MIN_CREATIVE_SCORE
        && cand_refusals <= base_refusals;

    let zero_scale_passed = zero_scale_mismatches == 0;
    if !zero_scale_passed {
        failures.push(format!(
            "Zero-scale identity check failed on {zero_scale_mismatches} control(s)."
        ));
    }

    let dr_g2_passed = capability_gain_passed
        && anti_looping_passed
        && creative_passed
        && zero_scale_passed
        && failures.is_empty();

    let receipt = json!({
        "schemaVersion": 1,
        "operation": "dr-g2-score",
        "drG2Passed": dr_g2_passed,
        "status": if dr_g2_passed { "passed" } else { "failed" },
        "contractScope": contract_scope,
        "selectedScale": evidence.selected_scale,
        "contractSha256": evidence.contract_sha256,
        "candidateAdapterSha256": evidence.candidate_adapter_sha256,
        "gates": {
            "capabilityGain": {
                "passed": capability_gain_passed,
                "promptCount": evidence.thinking_on_prompts.len(),
                "baselineMeanScore": base_mean,
                "candidateMeanScore": cand_mean,
                "deltaMeanScore": cand_mean - base_mean,
                "byCategory": category_json,
            },
            "antiLoopingPreservation": {
                "passed": anti_looping_passed,
                "baselineMedianThinkingTokens": base_median_thinking,
                "candidateMedianThinkingTokens": cand_median_thinking,
                "thinkingTokenRatio": thinking_ratio,
                "maxAllowedRatio": MAX_THINKING_TOKEN_RATIO,
                "baselineLoops": base_loops,
                "candidateLoops": cand_loops,
                "baseline4096Truncations": base_truncations,
                "candidate4096Truncations": cand_truncations,
            },
            "creativeAndOrcaNonRegression": {
                "passed": creative_passed,
                "promptCount": evidence.thinking_off_prompts.len(),
                "baselineMedianScore": base_median_creative,
                "candidateMedianScore": cand_median_creative,
                "candidateMinScore": cand_min_creative,
                "minAllowedScore": MIN_CREATIVE_SCORE,
                "baselineUnwantedRefusals": base_refusals,
                "candidateUnwantedRefusals": cand_refusals,
            },
            "zeroScaleIdentity": {
                "passed": zero_scale_passed,
                "controlCount": evidence.zero_scale_controls.len(),
                "mismatchCount": zero_scale_mismatches,
            },
        },
        "failures": failures,
        "promotionDisclaimer": "DR-G2 scoring validates the supplied screen evidence only; Prism repin, lock pinning, owning CI, and X1/X2 device qualification remain separate.",
    });
    if let Some(output) = &request.output_path {
        crate::bounded_input::write_pretty_json_new(output, &receipt)?;
    }
    Ok(receipt)
}
