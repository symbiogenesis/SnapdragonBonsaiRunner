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
const MIN_MULTIMODAL_SCORE: f64 = 3.0;
const MAX_DR_G2_EVIDENCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_DR_G2_RESPONSES_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DR_G2_SCALE_GRID_LEN: usize = 16;
const SYNTHETIC_FIXTURE_METHOD: &str = "synthetic-runtime-test-fixture-v1";
const THINKING_ON_CATEGORIES: [&str; 3] = ["coding", "reasoning", "instruction-following"];
const THINKING_OFF_CATEGORY: &str = "creative-prose";
const MULTIMODAL_DEFAULT_CATEGORY: &str = "multimodal";

/// Base model quantization profile evaluated by a DR-G2 screen.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum BaseModelProfile {
    #[default]
    #[serde(rename = "swift-pq2")]
    SwiftPq2,
    #[serde(rename = "nariq-2.5bpw")]
    Nariq2_5Bpw,
    #[serde(rename = "nariq-3.0bpw")]
    Nariq3_0Bpw,
    #[serde(rename = "nariq-2.0bpw")]
    Nariq2_0Bpw,
}

impl BaseModelProfile {
    pub(super) const fn is_nariq(self) -> bool {
        !matches!(self, Self::SwiftPq2)
    }

    pub(super) const fn nariq_variant(self) -> Option<bonsai_cpu_policy::NariqVariant> {
        match self {
            Self::SwiftPq2 => None,
            Self::Nariq2_5Bpw => Some(bonsai_cpu_policy::NariqVariant::Bpw2_5),
            Self::Nariq3_0Bpw => Some(bonsai_cpu_policy::NariqVariant::Bpw3_0),
            Self::Nariq2_0Bpw => Some(bonsai_cpu_policy::NariqVariant::Bpw2_0),
        }
    }

    pub(super) fn base_model_sha256(self) -> crate::Result<String> {
        match self.nariq_variant() {
            Some(variant) => Ok(variant.sha256().to_owned()),
            None => compiled_swift_base_sha256(),
        }
    }
}

const fn is_false(value: &bool) -> bool {
    !*value
}

/// Optional candidate configuration for `dr-g2-plan`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct DrG2PlanConfig {
    pub(super) base_model: Option<BaseModelProfile>,
    pub(super) include_multimodal_checks: bool,
    pub(super) candidate_adapter_sha256: Option<String>,
    pub(super) candidate_max_rank: Option<u16>,
    pub(super) candidate_target_count: Option<usize>,
    pub(super) scale_grid: Option<Vec<f64>>,
}

/// Request configuration for `dr-g2-blind-packet`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct DrG2BlindConfig {
    pub(super) responses_path: std::path::PathBuf,
    pub(super) blind_key_path: std::path::PathBuf,
    #[serde(default)]
    pub(super) blind_salt: Option<String>,
}

/// Request configuration for `dr-g2-apply-reviews`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct DrG2ApplyConfig {
    pub(super) review_path: std::path::PathBuf,
    pub(super) blind_key_path: std::path::PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct ThinkingOnArm {
    pub(super) scale: f64,
    pub(super) response_sha256: String,
    pub(super) human_reviewed: bool,
    pub(super) reviewer_id: String,
    pub(super) score: f64,
    pub(super) thinking_tokens: u32,
    pub(super) hit4096_truncation: bool,
    pub(super) loop_detected: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct ThinkingOnPromptPair {
    pub(super) prompt_id: String,
    pub(super) prompt_sha256: String,
    pub(super) category: String,
    pub(super) baseline: ThinkingOnArm,
    pub(super) candidate: ThinkingOnArm,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct ThinkingOffArm {
    pub(super) scale: f64,
    pub(super) response_sha256: String,
    pub(super) human_reviewed: bool,
    pub(super) reviewer_id: String,
    pub(super) score: f64,
    pub(super) unwanted_refusal: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct ThinkingOffPromptPair {
    pub(super) prompt_id: String,
    pub(super) prompt_sha256: String,
    pub(super) category: String,
    pub(super) baseline: ThinkingOffArm,
    pub(super) candidate: ThinkingOffArm,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct MultimodalArm {
    pub(super) scale: f64,
    pub(super) response_sha256: String,
    pub(super) human_reviewed: bool,
    pub(super) reviewer_id: String,
    pub(super) score: f64,
    pub(super) visual_grounding_preserved: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct MultimodalPromptPair {
    pub(super) prompt_id: String,
    pub(super) prompt_sha256: String,
    pub(super) image_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) category: Option<String>,
    pub(super) baseline: MultimodalArm,
    pub(super) candidate: MultimodalArm,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct ZeroScaleControl {
    pub(super) prompt_id: String,
    pub(super) prompt_sha256: String,
    pub(super) no_adapter_response_sha256: String,
    pub(super) zero_scale_response_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct DrG2Evidence {
    pub(super) schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) base_model: Option<BaseModelProfile>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) include_multimodal_checks: bool,
    pub(super) contract_sha256: String,
    pub(super) base_model_sha256: String,
    #[serde(default)]
    pub(super) orca_adapter_sha256: Option<String>,
    pub(super) orca_scale: f64,
    pub(super) candidate_adapter_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) candidate_max_rank: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) candidate_target_count: Option<usize>,
    pub(super) scale_grid: Vec<f64>,
    pub(super) selected_scale: f64,
    #[serde(default)]
    pub(super) training_prompt_sha256s: Vec<String>,
    pub(super) thinking_on_prompts: Vec<ThinkingOnPromptPair>,
    pub(super) thinking_off_prompts: Vec<ThinkingOffPromptPair>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) multimodal_prompts: Vec<MultimodalPromptPair>,
    pub(super) zero_scale_controls: Vec<ZeroScaleControl>,
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

pub(super) fn compiled_orca_adapter_sha256() -> crate::Result<String> {
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

pub(super) fn validate_scale_grid(grid: &[f64]) -> crate::Result<()> {
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

pub(super) fn resolve_scale_grid(custom: Option<&Vec<f64>>) -> crate::Result<Vec<f64>> {
    let grid = match custom {
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

#[expect(
    clippy::too_many_arguments,
    reason = "Canonical DR-G2 contract construction keeps all pinned model, adapter, and threshold fields explicit."
)]
pub(super) fn build_dr_g2_contract(
    base_model: Option<BaseModelProfile>,
    include_multimodal_checks: bool,
    base_model_sha256: &str,
    orca_adapter_sha256: Option<&str>,
    scale_grid: &[f64],
    candidate_adapter_sha256: Option<&str>,
    candidate_max_rank: Option<u16>,
    candidate_target_count: Option<usize>,
) -> Value {
    let profile = base_model.unwrap_or(BaseModelProfile::SwiftPq2);
    let orca_scale = if profile.is_nariq() {
        0.0
    } else {
        ORCA_EXPECTED_SCALE
    };
    let mut thresholds = json!({
        "maxThinkingTokenRatio": MAX_THINKING_TOKEN_RATIO,
        "minCreativeScore": MIN_CREATIVE_SCORE,
        "zeroExtraLoops": true,
        "zeroExtra4096Truncations": true,
        "zeroExtraUnwantedRefusals": true,
        "requireZeroScaleBitIdentity": true,
        "requireHumanReview": true,
        "requireHeldOutSplitDisjointness": true,
    });
    if include_multimodal_checks && let Some(obj) = thresholds.as_object_mut() {
        obj.insert(
            "requireMultimodalVisualPreservation".to_owned(),
            json!(true),
        );
    }
    let mut contract = json!({
        "schemaVersion": 1,
        "gate": "DR-G2",
        "baseModelSha256": base_model_sha256,
        "orcaAdapterSha256": orca_adapter_sha256,
        "orcaScale": orca_scale,
        "candidateAdapterSha256": candidate_adapter_sha256,
        "candidateMaxRank": candidate_max_rank,
        "candidateTargetCount": candidate_target_count,
        "scaleGrid": scale_grid,
        "requiredThinkingOnCategories": THINKING_ON_CATEGORIES,
        "requiredThinkingOffCategories": [THINKING_OFF_CATEGORY],
        "thresholds": thresholds,
    });
    if profile.is_nariq()
        && let Some(obj) = contract.as_object_mut()
    {
        obj.insert("baseModel".to_owned(), json!(profile));
    }
    contract
}

pub(super) fn hash_contract(contract: &Value) -> crate::Result<String> {
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
    let scale_grid = resolve_scale_grid(
        request
            .screen_config
            .as_ref()
            .and_then(|item| item.scale_grid.as_ref()),
    )?;
    let base_profile = request
        .screen_config
        .as_ref()
        .and_then(|cfg| cfg.base_model)
        .unwrap_or(BaseModelProfile::SwiftPq2);
    let include_multimodal = request
        .screen_config
        .as_ref()
        .is_some_and(|cfg| cfg.include_multimodal_checks);
    let base_sha = base_profile.base_model_sha256()?;
    let orca_sha = if base_profile.is_nariq() {
        None
    } else {
        Some(compiled_orca_adapter_sha256()?)
    };
    let orca_scale = if base_profile.is_nariq() {
        0.0
    } else {
        ORCA_EXPECTED_SCALE
    };
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
        base_profile.is_nariq().then_some(base_profile),
        include_multimodal,
        &base_sha,
        orca_sha.as_deref(),
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
        "orcaScale": orca_scale,
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
    if base_profile.is_nariq() {
        evidence_template["baseModel"] = json!(base_profile);
    }
    if include_multimodal {
        evidence_template["includeMultimodalChecks"] = json!(true);
        evidence_template["multimodalPrompts"] = json!([{
            "promptId": "mm-001",
            "promptSha256": "COPY_HELD_OUT_PROMPT_SHA256",
            "imageSha256": "COPY_IMAGE_SHA256",
            "baseline": {
                "scale": 0.0,
                "responseSha256": "COPY_BASELINE_RESPONSE_SHA256",
                "humanReviewed": false,
                "reviewerId": "REVIEWER",
                "score": 0.0,
                "visualGroundingPreserved": false
            },
            "candidate": {
                "scale": default_selected,
                "responseSha256": "COPY_CANDIDATE_RESPONSE_SHA256",
                "humanReviewed": false,
                "reviewerId": "REVIEWER",
                "score": 0.0,
                "visualGroundingPreserved": false
            }
        }]);
    }
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

pub(super) fn validate_evidence_header(evidence: &DrG2Evidence) -> crate::Result<&'static str> {
    if evidence.schema_version != 1 {
        return Err("DR-G2 evidence requires schemaVersion 1.".into());
    }
    if !crate::identity::is_lower_hex(&evidence.contract_sha256, 64) {
        return Err("DR-G2 evidence contractSha256 must be a lowercase 64-hex SHA-256.".into());
    }
    let profile = evidence.base_model.unwrap_or(BaseModelProfile::SwiftPq2);
    let expected_base_sha = profile.base_model_sha256()?;
    if evidence.base_model_sha256 != expected_base_sha {
        return Err(if profile.is_nariq() {
            "DR-G2 evidence baseModelSha256 disagrees with pinned NariQ lock.".into()
        } else {
            "DR-G2 evidence baseModelSha256 disagrees with pinned Swift PQ2 lock.".into()
        });
    }
    let expected_orca_sha = if profile.is_nariq() {
        if evidence.orca_adapter_sha256.is_some() {
            return Err("DR-G2 evidence for NariQ requires orcaAdapterSha256: null.".into());
        }
        if !evidence.orca_scale.is_finite()
            || evidence.orca_scale.is_sign_negative()
            || evidence.orca_scale.abs() > 1e-9
        {
            return Err("DR-G2 evidence for NariQ requires orcaScale equal to 0.0.".into());
        }
        None
    } else {
        let orca_sha = compiled_orca_adapter_sha256()?;
        if evidence.orca_adapter_sha256.as_deref() != Some(orca_sha.as_str()) {
            return Err("DR-G2 evidence orcaAdapterSha256 disagrees with pinned Orca lock.".into());
        }
        if !evidence.orca_scale.is_finite()
            || (evidence.orca_scale - ORCA_EXPECTED_SCALE).abs() > 1e-9
        {
            return Err("DR-G2 evidence orcaScale must equal 1.3.".into());
        }
        Some(orca_sha)
    };
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

    let multimodal_candidates: &[bool] = if evidence.include_multimodal_checks {
        &[true]
    } else if !evidence.multimodal_prompts.is_empty() {
        &[true, false]
    } else {
        &[false]
    };

    for &include_mm in multimodal_candidates {
        let bound_contract = build_dr_g2_contract(
            profile.is_nariq().then_some(profile),
            include_mm,
            &expected_base_sha,
            expected_orca_sha.as_deref(),
            &evidence.scale_grid,
            Some(&evidence.candidate_adapter_sha256),
            evidence.candidate_max_rank,
            evidence.candidate_target_count,
        );
        if evidence.contract_sha256 == hash_contract(&bound_contract)? {
            return Ok("candidate-bound");
        }

        if evidence.candidate_max_rank.is_none() && evidence.candidate_target_count.is_none() {
            let generic_contract = build_dr_g2_contract(
                profile.is_nariq().then_some(profile),
                include_mm,
                &expected_base_sha,
                expected_orca_sha.as_deref(),
                &evidence.scale_grid,
                None,
                None,
                None,
            );
            if evidence.contract_sha256 == hash_contract(&generic_contract)? {
                return Ok("policy-and-grid");
            }
        }
    }

    Err("DR-G2 evidence contractSha256 does not match canonical DR-G2 contract.".into())
}

fn read_dr_g2_evidence(path: &Path) -> crate::Result<DrG2Evidence> {
    let raw_bytes = crate::bounded_input::read_regular_file(
        path,
        MAX_DR_G2_EVIDENCE_BYTES,
        "DR-G2 evidence exceeds 16 MiB.",
    )?;
    serde_json::from_slice(crate::bounded_input::strip_utf8_bom(&raw_bytes))
        .map_err(|error| crate::cli::bounded_json_error(&error))
}

#[expect(
    clippy::too_many_lines,
    reason = "Fail-closed validation of split disjointness, human review, and all DR-G2 quality/safety gates."
)]
pub(super) fn score_dr_g2_evidence(evidence: &DrG2Evidence) -> crate::Result<Value> {
    let contract_scope = validate_evidence_header(evidence)?;

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
    let multimodal_active =
        evidence.include_multimodal_checks || !evidence.multimodal_prompts.is_empty();
    if multimodal_active && evidence.multimodal_prompts.is_empty() {
        return Err("DR-G2 multimodal checks require at least one multimodal prompt pair.".into());
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

    let mut mm_base_scores = Vec::with_capacity(evidence.multimodal_prompts.len());
    let mut mm_cand_scores = Vec::with_capacity(evidence.multimodal_prompts.len());
    let mut cand_min_mm = f64::INFINITY;
    let mut base_visual_lost = 0_usize;
    let mut cand_visual_lost = 0_usize;

    for pair in &evidence.multimodal_prompts {
        validate_nonempty_id(&pair.prompt_id, "Multimodal promptId")?;
        if !prompt_ids.insert(pair.prompt_id.as_str()) {
            return Err("Duplicate promptId across DR-G2 evidence.".into());
        }
        if !crate::identity::is_lower_hex(&pair.prompt_sha256, 64) {
            return Err("Multimodal promptSha256 must be lowercase 64-hex.".into());
        }
        if !crate::identity::is_lower_hex(&pair.image_sha256, 64) {
            return Err("Multimodal imageSha256 must be lowercase 64-hex.".into());
        }
        if training_digests.contains(pair.prompt_sha256.as_str()) {
            return Err(
                "Held-out Multimodal promptSha256 overlaps trainingPromptSha256s split.".into(),
            );
        }
        if let Some(cat) = &pair.category
            && cat != MULTIMODAL_DEFAULT_CATEGORY
        {
            return Err("Multimodal prompt category must be multimodal when supplied.".into());
        }
        for (arm, expected_scale) in [
            (&pair.baseline, 0.0_f64),
            (&pair.candidate, evidence.selected_scale),
        ] {
            if !arm.scale.is_finite() || (arm.scale - expected_scale).abs() > 1e-12 {
                return Err(
                    "Multimodal arm scale disagrees with baseline 0.0 or selectedScale.".into(),
                );
            }
            if !crate::identity::is_lower_hex(&arm.response_sha256, 64) {
                return Err("Multimodal responseSha256 must be lowercase 64-hex.".into());
            }
            if !arm.human_reviewed {
                return Err("DR-G2 requires humanReviewed=true on every Multimodal arm.".into());
            }
            validate_nonempty_id(&arm.reviewer_id, "Multimodal reviewerId")?;
            validate_score(arm.score, "Multimodal")?;
        }
        mm_base_scores.push(pair.baseline.score);
        mm_cand_scores.push(pair.candidate.score);
        cand_min_mm = cand_min_mm.min(pair.candidate.score);
        if !pair.baseline.visual_grounding_preserved {
            base_visual_lost += 1;
        }
        if !pair.candidate.visual_grounding_preserved {
            cand_visual_lost += 1;
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

    let mut gates = json!({
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
    });

    let multimodal_passed = if multimodal_active {
        let base_median_mm = median_f64(mm_base_scores)?;
        let cand_median_mm = median_f64(mm_cand_scores)?;
        if cand_median_mm < base_median_mm {
            failures.push(format!(
                "Multimodal visual median score regressed ({cand_median_mm:.4} < {base_median_mm:.4})."
            ));
        }
        if cand_min_mm < MIN_MULTIMODAL_SCORE {
            failures.push(format!(
                "Multimodal candidate prompt scored below {MIN_MULTIMODAL_SCORE:.1} floor ({cand_min_mm:.4})."
            ));
        }
        if cand_visual_lost > base_visual_lost {
            failures.push(format!(
                "Candidate lost visual grounding on more prompts than baseline ({cand_visual_lost} > {base_visual_lost})."
            ));
        }
        let passed = cand_median_mm >= base_median_mm
            && cand_min_mm >= MIN_MULTIMODAL_SCORE
            && cand_visual_lost <= base_visual_lost;
        gates["multimodalPreservation"] = json!({
            "passed": passed,
            "promptCount": evidence.multimodal_prompts.len(),
            "baselineMedianScore": base_median_mm,
            "candidateMedianScore": cand_median_mm,
            "candidateMinScore": cand_min_mm,
            "minAllowedScore": MIN_MULTIMODAL_SCORE,
            "baselineVisualGroundingLost": base_visual_lost,
            "candidateVisualGroundingLost": cand_visual_lost,
        });
        passed
    } else {
        true
    };

    let dr_g2_passed = capability_gain_passed
        && anti_looping_passed
        && creative_passed
        && zero_scale_passed
        && multimodal_passed
        && failures.is_empty();

    Ok(json!({
        "schemaVersion": 1,
        "operation": "dr-g2-score",
        "drG2Passed": dr_g2_passed,
        "status": if dr_g2_passed { "passed" } else { "failed" },
        "contractScope": contract_scope,
        "baseModel": evidence.base_model.unwrap_or(BaseModelProfile::SwiftPq2),
        "selectedScale": evidence.selected_scale,
        "contractSha256": evidence.contract_sha256,
        "candidateAdapterSha256": evidence.candidate_adapter_sha256,
        "gates": gates,
        "failures": failures,
        "promotionDisclaimer": "DR-G2 scoring validates the supplied screen evidence only; Prism repin, lock pinning, owning CI, and X1/X2 device qualification remain separate.",
    }))
}

/// Validates and scores a completed DR-G2 evidence bundle fail-closed without network I/O.
pub(super) fn run_dr_g2_score(request: &Request) -> crate::Result<Value> {
    if request.operation != Operation::DrG2Score {
        return Err("Invalid operation dispatch for dr-g2-score.".into());
    }
    let evidence_path = request
        .evidence_path
        .as_ref()
        .ok_or("dr-g2-score requires evidencePath.")?;
    let evidence = read_dr_g2_evidence(evidence_path)?;
    let receipt = score_dr_g2_evidence(&evidence)?;
    if let Some(output) = &request.output_path {
        crate::bounded_input::write_pretty_json_new(output, &receipt)?;
    }
    Ok(receipt)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct BlindResponseEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prompt_text: Option<String>,
    pub(super) baseline_text: String,
    pub(super) candidate_text: String,
}

fn deserialize_present_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ThinkingOnBlindItem {
    prompt_id: String,
    category: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt_text: Option<String>,
    arm_a: String,
    arm_b: String,
    reviewer_id: String,
    #[serde(deserialize_with = "deserialize_present_option")]
    arm_a_score: Option<f64>,
    #[serde(deserialize_with = "deserialize_present_option")]
    arm_b_score: Option<f64>,
    arm_a_loop_detected: bool,
    arm_b_loop_detected: bool,
    arm_a_hit4096: bool,
    arm_b_hit4096: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ThinkingOffBlindItem {
    prompt_id: String,
    category: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt_text: Option<String>,
    arm_a: String,
    arm_b: String,
    reviewer_id: String,
    #[serde(deserialize_with = "deserialize_present_option")]
    arm_a_score: Option<f64>,
    #[serde(deserialize_with = "deserialize_present_option")]
    arm_b_score: Option<f64>,
    arm_a_unwanted_refusal: bool,
    arm_b_unwanted_refusal: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MultimodalBlindItem {
    prompt_id: String,
    category: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt_text: Option<String>,
    arm_a: String,
    arm_b: String,
    reviewer_id: String,
    #[serde(deserialize_with = "deserialize_present_option")]
    arm_a_score: Option<f64>,
    #[serde(deserialize_with = "deserialize_present_option")]
    arm_b_score: Option<f64>,
    #[serde(deserialize_with = "deserialize_present_option")]
    arm_a_visual_grounding_preserved: Option<bool>,
    #[serde(deserialize_with = "deserialize_present_option")]
    arm_b_visual_grounding_preserved: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum BlindReviewItem {
    ThinkingOn(ThinkingOnBlindItem),
    ThinkingOff(ThinkingOffBlindItem),
    Multimodal(MultimodalBlindItem),
}

impl BlindReviewItem {
    fn prompt_id(&self) -> &str {
        match self {
            Self::ThinkingOn(item) => &item.prompt_id,
            Self::ThinkingOff(item) => &item.prompt_id,
            Self::Multimodal(item) => &item.prompt_id,
        }
    }

    fn arm_a(&self) -> &str {
        match self {
            Self::ThinkingOn(item) => &item.arm_a,
            Self::ThinkingOff(item) => &item.arm_a,
            Self::Multimodal(item) => &item.arm_a,
        }
    }

    fn arm_b(&self) -> &str {
        match self {
            Self::ThinkingOn(item) => &item.arm_b,
            Self::ThinkingOff(item) => &item.arm_b,
            Self::Multimodal(item) => &item.arm_b,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BlindReviewPacket {
    schema_version: u32,
    operation: String,
    contract_sha256: String,
    items: Vec<BlindReviewItem>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BlindKeyEntry {
    prompt_id: String,
    kind: String,
    arm_a_role: String,
    arm_b_role: String,
    baseline_response_sha256: String,
    candidate_response_sha256: String,
    arm_a_text_sha256: String,
    arm_b_text_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BlindKeyIntegrityView<'a> {
    schema_version: u32,
    contract_sha256: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    blind_salt: Option<&'a str>,
    entries: &'a [BlindKeyEntry],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BlindKeyFile {
    schema_version: u32,
    contract_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    blind_salt: Option<String>,
    entries: Vec<BlindKeyEntry>,
    integrity_sha256: String,
}

fn blind_key_integrity_sha256(
    schema_version: u32,
    contract_sha256: &str,
    blind_salt: Option<&str>,
    entries: &[BlindKeyEntry],
) -> crate::Result<String> {
    let view = BlindKeyIntegrityView {
        schema_version,
        contract_sha256,
        blind_salt,
        entries,
    };
    let bytes = serde_json::to_vec(&view)
        .map_err(|error| format!("Cannot serialize blind key integrity view: {error}"))?;
    Ok(crate::identity::sha256(&bytes))
}

fn blind_swap_arm_a_is_candidate(
    contract_sha256: &str,
    prompt_id: &str,
    salt: Option<&str>,
) -> crate::Result<bool> {
    let mut hasher = crate::identity::Hasher::new()?;
    hasher.update(b"dr-g2-blind-v1\0")?;
    hasher.update(contract_sha256.as_bytes())?;
    hasher.update(b"\0")?;
    hasher.update(prompt_id.as_bytes())?;
    hasher.update(b"\0")?;
    hasher.update(salt.unwrap_or("").as_bytes())?;
    let digest = hasher.finalize_bytes()?;
    Ok((digest[0] & 1) == 1)
}

/// Generates a blinded side-by-side human-review packet and separate integrity-protected unblinding key.
pub(super) fn run_dr_g2_blind_packet(request: &Request) -> crate::Result<Value> {
    if request.operation != Operation::DrG2BlindPacket {
        return Err("Invalid operation dispatch for dr-g2-blind-packet.".into());
    }
    let evidence_path = request
        .evidence_path
        .as_ref()
        .ok_or("dr-g2-blind-packet requires evidencePath.")?;
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("dr-g2-blind-packet requires outputPath.")?;
    let config = request
        .blind_config
        .as_ref()
        .ok_or("dr-g2-blind-packet requires blindConfig.")?;
    if let Some(salt) = &config.blind_salt
        && salt.len() > 256
    {
        return Err("blindConfig.blindSalt must be <= 256 bytes.".into());
    }
    if output_path == &config.blind_key_path {
        return Err("blindConfig.blindKeyPath must differ from outputPath.".into());
    }

    let evidence = read_dr_g2_evidence(evidence_path)?;
    let _contract_scope = validate_evidence_header(&evidence)?;
    let responses_bytes = crate::bounded_input::read_regular_file(
        &config.responses_path,
        MAX_DR_G2_RESPONSES_BYTES,
        "DR-G2 responses JSON exceeds 64 MiB.",
    )?;
    let responses: BTreeMap<String, BlindResponseEntry> =
        serde_json::from_slice(crate::bounded_input::strip_utf8_bom(&responses_bytes))
            .map_err(|error| crate::cli::bounded_json_error(&error))?;

    let mut seen_ids = BTreeSet::new();
    let total_prompts = evidence
        .thinking_on_prompts
        .len()
        .saturating_add(evidence.thinking_off_prompts.len())
        .saturating_add(evidence.multimodal_prompts.len());
    if total_prompts == 0 {
        return Err("DR-G2 evidence has no reviewable prompt pairs.".into());
    }
    let mut items = Vec::with_capacity(total_prompts);
    let mut entries = Vec::with_capacity(total_prompts);

    let mut build_arms = |prompt_id: &str,
                          kind: &str,
                          base_sha: &str,
                          cand_sha: &str|
     -> crate::Result<(Option<String>, String, String, BlindKeyEntry)> {
        validate_nonempty_id(prompt_id, "Blind packet promptId")?;
        if !seen_ids.insert(prompt_id.to_owned()) {
            return Err("Duplicate promptId across DR-G2 evidence.".into());
        }
        let resp = responses
            .get(prompt_id)
            .ok_or_else(|| format!("responsesPath is missing entry for promptId '{prompt_id}'."))?;
        let swap = blind_swap_arm_a_is_candidate(
            &evidence.contract_sha256,
            prompt_id,
            config.blind_salt.as_deref(),
        )?;
        let (arm_a, arm_b, arm_a_role, arm_b_role) = if swap {
            (
                resp.candidate_text.clone(),
                resp.baseline_text.clone(),
                "candidate".to_owned(),
                "baseline".to_owned(),
            )
        } else {
            (
                resp.baseline_text.clone(),
                resp.candidate_text.clone(),
                "baseline".to_owned(),
                "candidate".to_owned(),
            )
        };
        let arm_a_text_sha256 = crate::identity::sha256(arm_a.as_bytes());
        let arm_b_text_sha256 = crate::identity::sha256(arm_b.as_bytes());
        Ok((
            resp.prompt_text.clone(),
            arm_a,
            arm_b,
            BlindKeyEntry {
                prompt_id: prompt_id.to_owned(),
                kind: kind.to_owned(),
                arm_a_role,
                arm_b_role,
                baseline_response_sha256: base_sha.to_owned(),
                candidate_response_sha256: cand_sha.to_owned(),
                arm_a_text_sha256,
                arm_b_text_sha256,
            },
        ))
    };

    for pair in &evidence.thinking_on_prompts {
        let (prompt_text, arm_a, arm_b, key_entry) = build_arms(
            &pair.prompt_id,
            "thinking-on",
            &pair.baseline.response_sha256,
            &pair.candidate.response_sha256,
        )?;
        items.push(BlindReviewItem::ThinkingOn(ThinkingOnBlindItem {
            prompt_id: pair.prompt_id.clone(),
            category: pair.category.clone(),
            prompt_text,
            arm_a,
            arm_b,
            reviewer_id: String::new(),
            arm_a_score: None,
            arm_b_score: None,
            arm_a_loop_detected: false,
            arm_b_loop_detected: false,
            arm_a_hit4096: false,
            arm_b_hit4096: false,
        }));
        entries.push(key_entry);
    }

    for pair in &evidence.thinking_off_prompts {
        let (prompt_text, arm_a, arm_b, key_entry) = build_arms(
            &pair.prompt_id,
            "thinking-off",
            &pair.baseline.response_sha256,
            &pair.candidate.response_sha256,
        )?;
        items.push(BlindReviewItem::ThinkingOff(ThinkingOffBlindItem {
            prompt_id: pair.prompt_id.clone(),
            category: pair.category.clone(),
            prompt_text,
            arm_a,
            arm_b,
            reviewer_id: String::new(),
            arm_a_score: None,
            arm_b_score: None,
            arm_a_unwanted_refusal: false,
            arm_b_unwanted_refusal: false,
        }));
        entries.push(key_entry);
    }

    for pair in &evidence.multimodal_prompts {
        let (prompt_text, arm_a, arm_b, key_entry) = build_arms(
            &pair.prompt_id,
            "multimodal",
            &pair.baseline.response_sha256,
            &pair.candidate.response_sha256,
        )?;
        items.push(BlindReviewItem::Multimodal(MultimodalBlindItem {
            prompt_id: pair.prompt_id.clone(),
            category: pair
                .category
                .clone()
                .unwrap_or_else(|| MULTIMODAL_DEFAULT_CATEGORY.to_owned()),
            prompt_text,
            arm_a,
            arm_b,
            reviewer_id: String::new(),
            arm_a_score: None,
            arm_b_score: None,
            arm_a_visual_grounding_preserved: None,
            arm_b_visual_grounding_preserved: None,
        }));
        entries.push(key_entry);
    }

    let packet = BlindReviewPacket {
        schema_version: 1,
        operation: "dr-g2-blind-packet".to_owned(),
        contract_sha256: evidence.contract_sha256.clone(),
        items,
    };
    let integrity_sha256 = blind_key_integrity_sha256(
        1,
        &evidence.contract_sha256,
        config.blind_salt.as_deref(),
        &entries,
    )?;
    let key_file = BlindKeyFile {
        schema_version: 1,
        contract_sha256: evidence.contract_sha256.clone(),
        blind_salt: config.blind_salt.clone(),
        entries,
        integrity_sha256,
    };

    let packet_bytes = crate::bounded_input::pretty_json_bytes(&packet)?;
    let packet_sha256 = crate::identity::sha256(&packet_bytes);
    let key_bytes = crate::bounded_input::pretty_json_bytes(&key_file)?;
    let key_sha256 = crate::identity::sha256(&key_bytes);

    crate::bounded_input::write_new_synced(output_path, &packet_bytes)?;
    crate::bounded_input::write_new_synced(&config.blind_key_path, &key_bytes)?;

    Ok(json!({
        "schemaVersion": 1,
        "operation": "dr-g2-blind-packet",
        "status": "blind-packet-written",
        "contractSha256": evidence.contract_sha256,
        "itemCount": packet.items.len(),
        "packetSha256": packet_sha256,
        "blindKeySha256": key_sha256,
    }))
}

/// Validates and unblinds a completed review packet back into `DrG2Evidence`, then scores it.
#[expect(
    clippy::too_many_lines,
    reason = "Fail-closed integrity verification, unblinding, evidence update, and DR-G2 scoring."
)]
pub(super) fn run_dr_g2_apply_reviews(request: &Request) -> crate::Result<Value> {
    if request.operation != Operation::DrG2ApplyReviews {
        return Err("Invalid operation dispatch for dr-g2-apply-reviews.".into());
    }
    let evidence_path = request
        .evidence_path
        .as_ref()
        .ok_or("dr-g2-apply-reviews requires evidencePath.")?;
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("dr-g2-apply-reviews requires outputPath.")?;
    let config = request
        .apply_config
        .as_ref()
        .ok_or("dr-g2-apply-reviews requires applyConfig.")?;

    let mut evidence = read_dr_g2_evidence(evidence_path)?;
    let _contract_scope = validate_evidence_header(&evidence)?;

    let key_bytes = crate::bounded_input::read_regular_file(
        &config.blind_key_path,
        MAX_DR_G2_EVIDENCE_BYTES,
        "DR-G2 blind key exceeds 16 MiB.",
    )?;
    let key_file: BlindKeyFile =
        serde_json::from_slice(crate::bounded_input::strip_utf8_bom(&key_bytes))
            .map_err(|error| crate::cli::bounded_json_error(&error))?;
    if key_file.schema_version != 1 {
        return Err("DR-G2 blind key requires schemaVersion 1.".into());
    }
    if key_file.contract_sha256 != evidence.contract_sha256 {
        return Err("DR-G2 blind key contractSha256 disagrees with evidence.".into());
    }
    let expected_integrity = blind_key_integrity_sha256(
        key_file.schema_version,
        &key_file.contract_sha256,
        key_file.blind_salt.as_deref(),
        &key_file.entries,
    )?;
    if key_file.integrity_sha256 != expected_integrity {
        return Err("DR-G2 blind key integritySha256 verification failed.".into());
    }

    let review_bytes = crate::bounded_input::read_regular_file(
        &config.review_path,
        MAX_DR_G2_RESPONSES_BYTES,
        "DR-G2 review packet exceeds 64 MiB.",
    )?;
    let packet: BlindReviewPacket =
        serde_json::from_slice(crate::bounded_input::strip_utf8_bom(&review_bytes))
            .map_err(|error| crate::cli::bounded_json_error(&error))?;
    if packet.schema_version != 1 || packet.operation != "dr-g2-blind-packet" {
        return Err(
            "DR-G2 review packet requires schemaVersion 1 and operation dr-g2-blind-packet.".into(),
        );
    }
    if packet.contract_sha256 != evidence.contract_sha256 {
        return Err("DR-G2 review packet contractSha256 disagrees with evidence.".into());
    }

    let mut key_by_id: BTreeMap<&str, &BlindKeyEntry> = BTreeMap::new();
    for entry in &key_file.entries {
        if key_by_id.insert(entry.prompt_id.as_str(), entry).is_some() {
            return Err("Duplicate promptId in DR-G2 blind key.".into());
        }
    }
    let mut review_by_id: BTreeMap<&str, &BlindReviewItem> = BTreeMap::new();
    for item in &packet.items {
        let id = item.prompt_id();
        validate_nonempty_id(id, "Blind review promptId")?;
        if review_by_id.insert(id, item).is_some() {
            return Err("Duplicate promptId in DR-G2 review packet.".into());
        }
    }

    let expected_total = evidence
        .thinking_on_prompts
        .len()
        .saturating_add(evidence.thinking_off_prompts.len())
        .saturating_add(evidence.multimodal_prompts.len());
    if key_by_id.len() != expected_total || review_by_id.len() != expected_total {
        return Err(
            "DR-G2 blind key or review packet prompt count disagrees with evidence.".into(),
        );
    }

    let verify_binding = |prompt_id: &str,
                          expected_kind: &str,
                          base_sha: &str,
                          cand_sha: &str|
     -> crate::Result<(&BlindKeyEntry, &BlindReviewItem, bool)> {
        let entry = key_by_id
            .get(prompt_id)
            .copied()
            .ok_or_else(|| format!("DR-G2 blind key is missing promptId '{prompt_id}'."))?;
        if entry.kind != expected_kind
            || entry.baseline_response_sha256 != base_sha
            || entry.candidate_response_sha256 != cand_sha
        {
            return Err(format!(
                "DR-G2 blind key entry for '{prompt_id}' disagrees with evidence."
            ));
        }
        let a_is_candidate = match (entry.arm_a_role.as_str(), entry.arm_b_role.as_str()) {
            ("candidate", "baseline") => true,
            ("baseline", "candidate") => false,
            _ => {
                return Err(format!(
                    "DR-G2 blind key entry for '{prompt_id}' has invalid arm roles."
                ));
            }
        };
        let item = review_by_id
            .get(prompt_id)
            .copied()
            .ok_or_else(|| format!("DR-G2 review packet is missing promptId '{prompt_id}'."))?;
        if crate::identity::sha256(item.arm_a().as_bytes()) != entry.arm_a_text_sha256
            || crate::identity::sha256(item.arm_b().as_bytes()) != entry.arm_b_text_sha256
        {
            return Err(format!(
                "DR-G2 review packet text for '{prompt_id}' was modified after blinding."
            ));
        }
        Ok((entry, item, a_is_candidate))
    };

    for pair in &mut evidence.thinking_on_prompts {
        let (_entry, item, a_is_candidate) = verify_binding(
            &pair.prompt_id,
            "thinking-on",
            &pair.baseline.response_sha256,
            &pair.candidate.response_sha256,
        )?;
        let BlindReviewItem::ThinkingOn(review) = item else {
            return Err(format!(
                "Review item '{}' does not match Thinking-ON schema.",
                pair.prompt_id
            ));
        };
        validate_nonempty_id(&review.reviewer_id, "Blind review reviewerId")?;
        let score_a = review
            .arm_a_score
            .ok_or("Blind review armAScore must be provided (0.0..=4.0).")?;
        let score_b = review
            .arm_b_score
            .ok_or("Blind review armBScore must be provided (0.0..=4.0).")?;
        validate_score(score_a, "Blind review armAScore")?;
        validate_score(score_b, "Blind review armBScore")?;

        let ((base_score, base_loop, base_hit4096), (cand_score, cand_loop, cand_hit4096)) =
            if a_is_candidate {
                (
                    (score_b, review.arm_b_loop_detected, review.arm_b_hit4096),
                    (score_a, review.arm_a_loop_detected, review.arm_a_hit4096),
                )
            } else {
                (
                    (score_a, review.arm_a_loop_detected, review.arm_a_hit4096),
                    (score_b, review.arm_b_loop_detected, review.arm_b_hit4096),
                )
            };
        pair.baseline.human_reviewed = true;
        pair.baseline.reviewer_id = review.reviewer_id.clone();
        pair.baseline.score = base_score;
        pair.baseline.loop_detected = pair.baseline.loop_detected || base_loop;
        pair.baseline.hit4096_truncation = pair.baseline.hit4096_truncation || base_hit4096;

        pair.candidate.human_reviewed = true;
        pair.candidate.reviewer_id = review.reviewer_id.clone();
        pair.candidate.score = cand_score;
        pair.candidate.loop_detected = pair.candidate.loop_detected || cand_loop;
        pair.candidate.hit4096_truncation = pair.candidate.hit4096_truncation || cand_hit4096;
    }

    for pair in &mut evidence.thinking_off_prompts {
        let (_entry, item, a_is_candidate) = verify_binding(
            &pair.prompt_id,
            "thinking-off",
            &pair.baseline.response_sha256,
            &pair.candidate.response_sha256,
        )?;
        let BlindReviewItem::ThinkingOff(review) = item else {
            return Err(format!(
                "Review item '{}' does not match Thinking-OFF schema.",
                pair.prompt_id
            ));
        };
        validate_nonempty_id(&review.reviewer_id, "Blind review reviewerId")?;
        let score_a = review
            .arm_a_score
            .ok_or("Blind review armAScore must be provided (0.0..=4.0).")?;
        let score_b = review
            .arm_b_score
            .ok_or("Blind review armBScore must be provided (0.0..=4.0).")?;
        validate_score(score_a, "Blind review armAScore")?;
        validate_score(score_b, "Blind review armBScore")?;

        let ((base_score, base_refusal), (cand_score, cand_refusal)) = if a_is_candidate {
            (
                (score_b, review.arm_b_unwanted_refusal),
                (score_a, review.arm_a_unwanted_refusal),
            )
        } else {
            (
                (score_a, review.arm_a_unwanted_refusal),
                (score_b, review.arm_b_unwanted_refusal),
            )
        };
        pair.baseline.human_reviewed = true;
        pair.baseline.reviewer_id = review.reviewer_id.clone();
        pair.baseline.score = base_score;
        pair.baseline.unwanted_refusal = base_refusal;

        pair.candidate.human_reviewed = true;
        pair.candidate.reviewer_id = review.reviewer_id.clone();
        pair.candidate.score = cand_score;
        pair.candidate.unwanted_refusal = cand_refusal;
    }

    for pair in &mut evidence.multimodal_prompts {
        let (_entry, item, a_is_candidate) = verify_binding(
            &pair.prompt_id,
            "multimodal",
            &pair.baseline.response_sha256,
            &pair.candidate.response_sha256,
        )?;
        let BlindReviewItem::Multimodal(review) = item else {
            return Err(format!(
                "Review item '{}' does not match Multimodal schema.",
                pair.prompt_id
            ));
        };
        validate_nonempty_id(&review.reviewer_id, "Blind review reviewerId")?;
        let score_a = review
            .arm_a_score
            .ok_or("Blind review armAScore must be provided (0.0..=4.0).")?;
        let score_b = review
            .arm_b_score
            .ok_or("Blind review armBScore must be provided (0.0..=4.0).")?;
        validate_score(score_a, "Blind review armAScore")?;
        validate_score(score_b, "Blind review armBScore")?;
        let vis_a = review
            .arm_a_visual_grounding_preserved
            .ok_or("Blind review must specify armAVisualGroundingPreserved.")?;
        let vis_b = review
            .arm_b_visual_grounding_preserved
            .ok_or("Blind review must specify armBVisualGroundingPreserved.")?;

        let ((base_score, base_vis), (cand_score, cand_vis)) = if a_is_candidate {
            ((score_b, vis_b), (score_a, vis_a))
        } else {
            ((score_a, vis_a), (score_b, vis_b))
        };
        pair.baseline.human_reviewed = true;
        pair.baseline.reviewer_id = review.reviewer_id.clone();
        pair.baseline.score = base_score;
        pair.baseline.visual_grounding_preserved = base_vis;

        pair.candidate.human_reviewed = true;
        pair.candidate.reviewer_id = review.reviewer_id.clone();
        pair.candidate.score = cand_score;
        pair.candidate.visual_grounding_preserved = cand_vis;
    }

    let updated_bytes = crate::bounded_input::pretty_json_bytes(&evidence)?;
    let updated_evidence_sha256 = crate::identity::sha256(&updated_bytes);
    let scoring = score_dr_g2_evidence(&evidence)?;
    crate::bounded_input::write_new_synced(output_path, &updated_bytes)?;

    Ok(json!({
        "schemaVersion": 1,
        "operation": "dr-g2-apply-reviews",
        "status": "reviews-applied",
        "promptCount": expected_total,
        "updatedEvidenceSha256": updated_evidence_sha256,
        "drG2Passed": scoring["drG2Passed"],
        "scoring": scoring,
    }))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::{Value, json};

    use super::super::run;
    use super::BaseModelProfile;
    use crate::bounded_input::ScratchDir;

    #[test]
    fn dr_g2_plan_preserves_swift_pq2_contract_sha256_backward_compatibility() -> crate::Result<()>
    {
        let cand_sha = "1b850494b8759e24c511aac39621db3e88a4d708a075012b47dca2682001c569";
        let default_plan = run(json!({
            "schemaVersion": 1,
            "operation": "dr-g2-plan",
            "screenConfig": {
                "candidateAdapterSha256": cand_sha,
                "candidateMaxRank": 16,
                "candidateTargetCount": 128,
                "scaleGrid": [0.0, 0.5, 1.0, 1.5, 2.0]
            }
        }))?;
        let explicit_swift_plan = run(json!({
            "schemaVersion": 1,
            "operation": "dr-g2-plan",
            "screenConfig": {
                "baseModel": "swift-pq2",
                "includeMultimodalChecks": false,
                "candidateAdapterSha256": cand_sha,
                "candidateMaxRank": 16,
                "candidateTargetCount": 128,
                "scaleGrid": [0.0, 0.5, 1.0, 1.5, 2.0]
            }
        }))?;
        assert_eq!(
            default_plan["contractSha256"],
            "963b26d9cb7d2e5dcba0cf4e3b5290706594f7e89dac7f0cfbb8a6661dc39413"
        );
        assert_eq!(
            explicit_swift_plan["contractSha256"],
            default_plan["contractSha256"]
        );
        assert!(default_plan["contract"].get("baseModel").is_none());
        assert!(
            default_plan["contract"]["thresholds"]
                .get("requireMultimodalVisualPreservation")
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn dr_g2_nariq_variants_require_null_orca_and_zero_orca_scale() -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-DrG2-Nariq")?;
        let cand_sha = "ab".repeat(32);
        for (profile_str, profile) in [
            ("nariq-2.5bpw", BaseModelProfile::Nariq2_5Bpw),
            ("nariq-3.0bpw", BaseModelProfile::Nariq3_0Bpw),
            ("nariq-2.0bpw", BaseModelProfile::Nariq2_0Bpw),
        ] {
            let plan = run(json!({
                "schemaVersion": 1,
                "operation": "dr-g2-plan",
                "screenConfig": {
                    "baseModel": profile_str,
                    "candidateAdapterSha256": cand_sha,
                    "candidateMaxRank": 16,
                    "candidateTargetCount": 128
                }
            }))?;
            assert_eq!(plan["contract"]["baseModel"], profile_str);
            assert_eq!(plan["contract"]["orcaAdapterSha256"], Value::Null);
            assert_eq!(plan["contract"]["orcaScale"], 0.0);
            assert_eq!(
                plan["contract"]["baseModelSha256"],
                profile.base_model_sha256()?
            );

            let evidence = json!({
                "schemaVersion": 1,
                "baseModel": profile_str,
                "contractSha256": plan["contractSha256"],
                "baseModelSha256": profile.base_model_sha256()?,
                "orcaAdapterSha256": null,
                "orcaScale": 0.0,
                "candidateAdapterSha256": cand_sha,
                "candidateMaxRank": 16,
                "candidateTargetCount": 128,
                "scaleGrid": [0.0, 0.25, 0.5, 0.75, 1.0],
                "selectedScale": 0.25,
                "trainingPromptSha256s": [],
                "thinkingOnPrompts": [
                    {
                        "promptId": "c1", "promptSha256": "11".repeat(32), "category": "coding",
                        "baseline": { "scale": 0.0, "responseSha256": "a1".repeat(32), "humanReviewed": true, "reviewerId": "owner", "score": 2.5, "thinkingTokens": 100, "hit4096Truncation": false, "loopDetected": false },
                        "candidate": { "scale": 0.25, "responseSha256": "b1".repeat(32), "humanReviewed": true, "reviewerId": "owner", "score": 3.5, "thinkingTokens": 105, "hit4096Truncation": false, "loopDetected": false }
                    },
                    {
                        "promptId": "r1", "promptSha256": "12".repeat(32), "category": "reasoning",
                        "baseline": { "scale": 0.0, "responseSha256": "c1".repeat(32), "humanReviewed": true, "reviewerId": "owner", "score": 3.0, "thinkingTokens": 100, "hit4096Truncation": false, "loopDetected": false },
                        "candidate": { "scale": 0.25, "responseSha256": "d1".repeat(32), "humanReviewed": true, "reviewerId": "owner", "score": 3.5, "thinkingTokens": 105, "hit4096Truncation": false, "loopDetected": false }
                    },
                    {
                        "promptId": "i1", "promptSha256": "13".repeat(32), "category": "instruction-following",
                        "baseline": { "scale": 0.0, "responseSha256": "e1".repeat(32), "humanReviewed": true, "reviewerId": "owner", "score": 3.0, "thinkingTokens": 100, "hit4096Truncation": false, "loopDetected": false },
                        "candidate": { "scale": 0.25, "responseSha256": "f1".repeat(32), "humanReviewed": true, "reviewerId": "owner", "score": 3.0, "thinkingTokens": 105, "hit4096Truncation": false, "loopDetected": false }
                    }
                ],
                "thinkingOffPrompts": [{
                    "promptId": "cr1", "promptSha256": "14".repeat(32), "category": "creative-prose",
                    "baseline": { "scale": 0.0, "responseSha256": "21".repeat(32), "humanReviewed": true, "reviewerId": "owner", "score": 3.5, "unwantedRefusal": false },
                    "candidate": { "scale": 0.25, "responseSha256": "31".repeat(32), "humanReviewed": true, "reviewerId": "owner", "score": 3.5, "unwantedRefusal": false }
                }],
                "zeroScaleControls": [{
                    "promptId": "z1", "promptSha256": "15".repeat(32),
                    "noAdapterResponseSha256": "41".repeat(32),
                    "zeroScaleResponseSha256": "41".repeat(32)
                }]
            });
            let path = scratch.join(format!("{profile_str}.json"));
            fs::write(
                &path,
                serde_json::to_vec(&evidence).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let scored = run(json!({
                "schemaVersion": 1,
                "operation": "dr-g2-score",
                "evidencePath": path
            }))?;
            assert_eq!(scored["drG2Passed"], true);
            assert_eq!(scored["baseModel"], profile_str);

            // Reject non-null Orca on NariQ and non-zero orcaScale on NariQ.
            let mut bad_orca = evidence.clone();
            bad_orca["orcaAdapterSha256"] = json!("ff".repeat(32));
            let bad_path = scratch.join(format!("{profile_str}-bad-orca.json"));
            fs::write(
                &bad_path,
                serde_json::to_vec(&bad_orca).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            assert!(
                run(json!({
                    "schemaVersion": 1,
                    "operation": "dr-g2-score",
                    "evidencePath": bad_path
                }))
                .is_err()
            );

            let mut bad_scale = evidence;
            bad_scale["orcaScale"] = json!(1.3);
            let bad_scale_path = scratch.join(format!("{profile_str}-bad-scale.json"));
            fs::write(
                &bad_scale_path,
                serde_json::to_vec(&bad_scale).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            assert!(
                run(json!({
                    "schemaVersion": 1,
                    "operation": "dr-g2-score",
                    "evidencePath": bad_scale_path
                }))
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn dr_g2_selective_rl_multimodal_and_blind_review_round_trip() -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-DrG2-Blind")?;
        let cand_sha = "cc".repeat(32);
        let plan = run(json!({
            "schemaVersion": 1,
            "operation": "dr-g2-plan",
            "screenConfig": {
                "includeMultimodalChecks": true,
                "candidateAdapterSha256": cand_sha,
                "candidateMaxRank": 16,
                "candidateTargetCount": 128
            }
        }))?;
        assert_eq!(
            plan["contract"]["thresholds"]["requireMultimodalVisualPreservation"],
            true
        );

        let base_sha = plan["contract"]["baseModelSha256"]
            .as_str()
            .ok_or("Missing baseModelSha256.")?
            .to_owned();
        let orca_sha = plan["contract"]["orcaAdapterSha256"]
            .as_str()
            .ok_or("Missing orcaAdapterSha256.")?
            .to_owned();
        let train_sha = "99".repeat(32);

        let unreviewed_evidence = json!({
            "schemaVersion": 1,
            "includeMultimodalChecks": true,
            "contractSha256": plan["contractSha256"],
            "baseModelSha256": base_sha,
            "orcaAdapterSha256": orca_sha,
            "orcaScale": 1.3,
            "candidateAdapterSha256": cand_sha,
            "candidateMaxRank": 16,
            "candidateTargetCount": 128,
            "scaleGrid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "selectedScale": 0.25,
            "trainingPromptSha256s": [train_sha],
            "thinkingOnPrompts": [
                {
                    "promptId": "code-1", "promptSha256": "11".repeat(32), "category": "coding",
                    "baseline": { "scale": 0.0, "responseSha256": "a1".repeat(32), "humanReviewed": false, "reviewerId": "", "score": 0.0, "thinkingTokens": 400, "hit4096Truncation": false, "loopDetected": false },
                    "candidate": { "scale": 0.25, "responseSha256": "b1".repeat(32), "humanReviewed": false, "reviewerId": "", "score": 0.0, "thinkingTokens": 420, "hit4096Truncation": false, "loopDetected": false }
                },
                {
                    "promptId": "reason-1", "promptSha256": "12".repeat(32), "category": "reasoning",
                    "baseline": { "scale": 0.0, "responseSha256": "c1".repeat(32), "humanReviewed": false, "reviewerId": "", "score": 0.0, "thinkingTokens": 400, "hit4096Truncation": false, "loopDetected": false },
                    "candidate": { "scale": 0.25, "responseSha256": "d1".repeat(32), "humanReviewed": false, "reviewerId": "", "score": 0.0, "thinkingTokens": 410, "hit4096Truncation": false, "loopDetected": false }
                },
                {
                    "promptId": "if-1", "promptSha256": "13".repeat(32), "category": "instruction-following",
                    "baseline": { "scale": 0.0, "responseSha256": "e1".repeat(32), "humanReviewed": false, "reviewerId": "", "score": 0.0, "thinkingTokens": 300, "hit4096Truncation": false, "loopDetected": false },
                    "candidate": { "scale": 0.25, "responseSha256": "f1".repeat(32), "humanReviewed": false, "reviewerId": "", "score": 0.0, "thinkingTokens": 310, "hit4096Truncation": false, "loopDetected": false }
                }
            ],
            "thinkingOffPrompts": [{
                "promptId": "creative-1", "promptSha256": "14".repeat(32), "category": "creative-prose",
                "baseline": { "scale": 0.0, "responseSha256": "21".repeat(32), "humanReviewed": false, "reviewerId": "", "score": 0.0, "unwantedRefusal": false },
                "candidate": { "scale": 0.25, "responseSha256": "31".repeat(32), "humanReviewed": false, "reviewerId": "", "score": 0.0, "unwantedRefusal": false }
            }],
            "multimodalPrompts": [{
                "promptId": "mm-1", "promptSha256": "15".repeat(32), "imageSha256": "88".repeat(32),
                "baseline": { "scale": 0.0, "responseSha256": "41".repeat(32), "humanReviewed": false, "reviewerId": "", "score": 0.0, "visualGroundingPreserved": false },
                "candidate": { "scale": 0.25, "responseSha256": "51".repeat(32), "humanReviewed": false, "reviewerId": "", "score": 0.0, "visualGroundingPreserved": false }
            }],
            "zeroScaleControls": [{
                "promptId": "z-1", "promptSha256": "16".repeat(32),
                "noAdapterResponseSha256": "00".repeat(32), "zeroScaleResponseSha256": "00".repeat(32)
            }]
        });
        let evidence_path = scratch.join("unreviewed.json");
        fs::write(
            &evidence_path,
            serde_json::to_vec(&unreviewed_evidence).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        let responses = json!({
            "code-1": { "promptText": "Write Rust code.", "baselineText": "base code", "candidateText": "cand code" },
            "reason-1": { "baselineText": "base reason", "candidateText": "cand reason" },
            "if-1": { "baselineText": "base if", "candidateText": "cand if" },
            "creative-1": { "promptText": "Write a poem.", "baselineText": "base prose", "candidateText": "cand prose" },
            "mm-1": { "promptText": "Describe the image.", "baselineText": "base vision", "candidateText": "cand vision" }
        });
        let responses_path = scratch.join("responses.json");
        fs::write(
            &responses_path,
            serde_json::to_vec(&responses).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        let packet_path = scratch.join("blind-packet.json");
        let blind_key_path = scratch.join("blind-key.json");
        let blind_receipt = run(json!({
            "schemaVersion": 1,
            "operation": "dr-g2-blind-packet",
            "evidencePath": evidence_path,
            "outputPath": packet_path,
            "blindConfig": {
                "responsesPath": responses_path,
                "blindKeyPath": blind_key_path,
                "blindSalt": "deterministic-salt"
            }
        }))?;
        assert_eq!(blind_receipt["status"], "blind-packet-written");
        assert_eq!(blind_receipt["itemCount"], 5);

        let packet_text = fs::read_to_string(&packet_path).map_err(|e| e.to_string())?;
        assert!(!packet_text.contains("baseline"));
        assert!(!packet_text.contains("candidate"));
        assert!(!packet_text.contains("scale"));
        let mut packet: Value = serde_json::from_str(&packet_text).map_err(|e| e.to_string())?;
        let key_json: Value =
            serde_json::from_slice(&fs::read(&blind_key_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;

        // Check deterministic permutation & optional promptText presence/omission.
        let items = packet["items"].as_array_mut().ok_or("Missing items.")?;
        assert_eq!(items[0]["promptText"], "Write Rust code.");
        assert!(items[1].get("promptText").is_none());

        // Blank packet must be rejected by dr-g2-apply-reviews.
        let rejected_out = scratch.join("rejected-out.json");
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "dr-g2-apply-reviews",
                "evidencePath": evidence_path,
                "outputPath": rejected_out,
                "applyConfig": {
                    "reviewPath": packet_path,
                    "blindKeyPath": blind_key_path
                }
            }))
            .is_err()
        );

        // Fill in completed review scores using arm_a_role from blind key to verify unblinding accuracy.
        let key_entries = key_json["entries"].as_array().ok_or("Missing entries.")?;
        for (item, key_entry) in items.iter_mut().zip(key_entries.iter()) {
            item["reviewerId"] = json!("owner");
            let a_is_cand = key_entry["armARole"] == "candidate";
            let (cand_score, base_score) = match item["promptId"].as_str() {
                Some("code-1") => (3.5, 2.5),
                Some("reason-1") => (3.5, 3.0),
                Some("if-1") => (3.0, 3.0),
                Some("creative-1") => (3.5, 3.5),
                Some("mm-1") => (3.5, 3.0),
                _ => return Err("Unexpected promptId.".into()),
            };
            item["armAScore"] = json!(if a_is_cand { cand_score } else { base_score });
            item["armBScore"] = json!(if a_is_cand { base_score } else { cand_score });
            if item["promptId"] == "mm-1" {
                item["armAVisualGroundingPreserved"] = json!(true);
                item["armBVisualGroundingPreserved"] = json!(true);
            }
        }
        let completed_review_path = scratch.join("completed-review.json");
        fs::write(
            &completed_review_path,
            serde_json::to_vec_pretty(&packet).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        let applied_evidence_path = scratch.join("applied-evidence.json");
        let applied = run(json!({
            "schemaVersion": 1,
            "operation": "dr-g2-apply-reviews",
            "evidencePath": evidence_path,
            "outputPath": applied_evidence_path,
            "applyConfig": {
                "reviewPath": completed_review_path,
                "blindKeyPath": blind_key_path
            }
        }))?;
        assert_eq!(applied["status"], "reviews-applied");
        assert_eq!(applied["drG2Passed"], true);
        assert_eq!(
            applied["scoring"]["gates"]["multimodalPreservation"]["passed"],
            true
        );

        let scored = run(json!({
            "schemaVersion": 1,
            "operation": "dr-g2-score",
            "evidencePath": applied_evidence_path
        }))?;
        assert_eq!(scored["drG2Passed"], true);
        assert_eq!(scored["gates"]["multimodalPreservation"]["passed"], true);

        // Verify multimodal failure modes (lost grounding, score < 3.0, split overlap, empty multimodal).
        let mut applied_val: Value =
            serde_json::from_slice(&fs::read(&applied_evidence_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;

        let mut lost_grounding = applied_val.clone();
        lost_grounding["multimodalPrompts"][0]["candidate"]["visualGroundingPreserved"] =
            json!(false);
        let lost_path = scratch.join("mm-lost.json");
        fs::write(
            &lost_path,
            serde_json::to_vec(&lost_grounding).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let lost_score = run(json!({
            "schemaVersion": 1,
            "operation": "dr-g2-score",
            "evidencePath": lost_path
        }))?;
        assert_eq!(lost_score["drG2Passed"], false);
        assert_eq!(
            lost_score["gates"]["multimodalPreservation"]["passed"],
            false
        );

        let mut mm_overlap = applied_val.clone();
        mm_overlap["multimodalPrompts"][0]["promptSha256"] = json!(train_sha);
        let overlap_path = scratch.join("mm-overlap.json");
        fs::write(
            &overlap_path,
            serde_json::to_vec(&mm_overlap).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "dr-g2-score",
                "evidencePath": overlap_path
            }))
            .is_err()
        );

        applied_val["multimodalPrompts"] = json!([]);
        let empty_mm_path = scratch.join("mm-empty.json");
        fs::write(
            &empty_mm_path,
            serde_json::to_vec(&applied_val).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "dr-g2-score",
                "evidencePath": empty_mm_path
            }))
            .is_err()
        );

        // Verify tampered blind key integrity fails closed.
        let mut tampered_key = key_json;
        tampered_key["entries"][0]["armARole"] =
            json!(if tampered_key["entries"][0]["armARole"] == "candidate" {
                "baseline"
            } else {
                "candidate"
            });
        let tampered_key_path = scratch.join("tampered-key.json");
        fs::write(
            &tampered_key_path,
            serde_json::to_vec(&tampered_key).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "dr-g2-apply-reviews",
                "evidencePath": evidence_path,
                "outputPath": scratch.join("tampered-out.json"),
                "applyConfig": {
                    "reviewPath": completed_review_path,
                    "blindKeyPath": tampered_key_path
                }
            }))
            .is_err()
        );

        Ok(())
    }
}
