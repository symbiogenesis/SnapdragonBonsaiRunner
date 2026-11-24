//! Offline multi-rank GGUF LoRA chunk merging (`merge-adapters`) and pure-Rust
//! algebraic direct weight editing (`edit-adapter`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::gguf_adapter::{
    ExportDtype, ExportTensorPair, VerifyPolicy, classify_families, read_lora_gguf_pairs_with_dims,
    read_rank_one_a_directions, read_rank_one_a_directions_with_dims, read_rank_one_b_directions,
    read_rank_one_b_directions_with_dims, sort_pairs_canonically, swift_base_dims_map,
    verify_lora_gguf_with_dims, write_lora_gguf,
};
use super::safetensors::swift_tensor_inventory;
use super::{Operation, Request, usize_to_f32};

const COMMON_PROVENANCE_KEYS: [&str; 7] = [
    "bonsai.composition.method",
    "bonsai.composition.candidate_index_sha256",
    "bonsai.composition.base_index_sha256",
    "bonsai.composition.candidate_repo",
    "bonsai.composition.candidate_revision",
    "bonsai.composition.base_repo",
    "bonsai.composition.base_revision",
];

const VALID_LINEAR_FAMILIES: [&str; 10] = [
    "ffn_down",
    "attn_output",
    "attn_q",
    "attn_k",
    "attn_v",
    "ffn_gate",
    "ffn_up",
    "attn_qkv",
    "attn_gate",
    "ssm_out",
];

/// Target factor matrix selection for magnitude coefficient pruning.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum PruneTarget {
    #[default]
    A,
    B,
    Both,
}

impl PruneTarget {
    const fn includes_a(self) -> bool {
        matches!(self, Self::A | Self::Both)
    }

    const fn includes_b(self) -> bool {
        matches!(self, Self::B | Self::Both)
    }
}

/// Factor matrix side projected into the reference null space during orthogonalization.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum OrthogonalizeSide {
    #[default]
    B,
    A,
    Both,
}

impl OrthogonalizeSide {
    const fn includes_a(self) -> bool {
        matches!(self, Self::A | Self::Both)
    }

    const fn includes_b(self) -> bool {
        matches!(self, Self::B | Self::Both)
    }
}

/// Nonlinear depth-steering interpolation curve between `depthGainStart` and `depthGainEnd`.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum DepthGainCurve {
    #[default]
    Linear,
    Cosine,
    Power,
}

/// Algebraic multi-rank factor edit specification for `edit-adapter`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct EditAdapterConfig {
    pub(super) target_max_rank: Option<usize>,
    pub(super) per_site_ranks: BTreeMap<String, usize>,
    pub(super) drop_modes: Vec<usize>,
    pub(super) mode_weights: Vec<f64>,
    pub(super) prune_keep_numerator: Option<usize>,
    pub(super) prune_keep_denominator: Option<usize>,
    pub(super) prune_target: PruneTarget,
    pub(super) min_coeff_abs: Option<f64>,
    pub(super) global_gain: Option<f64>,
    pub(super) family_scales: BTreeMap<String, f64>,
    pub(super) layer_min: Option<u32>,
    pub(super) layer_max: Option<u32>,
    pub(super) depth_gain_start: Option<f64>,
    pub(super) depth_gain_end: Option<f64>,
    pub(super) depth_gain_curve: DepthGainCurve,
    pub(super) depth_gain_exponent: Option<f64>,
    pub(super) layer_gains: BTreeMap<u32, f64>,
    pub(super) min_site_frobenius_norm: Option<f64>,
    pub(super) max_site_frobenius_norm: Option<f64>,
    pub(super) ls_lora_reference_adapter_path: Option<PathBuf>,
    pub(super) ls_lora_min_cosine: Option<f64>,
    pub(super) cans_neuron_scales: BTreeMap<String, Vec<f64>>,
    pub(super) orthogonalize_against_rank_one_path: Option<PathBuf>,
    pub(super) orthogonalize_side: OrthogonalizeSide,
    pub(super) orthogonalize_against_adapter_path: Option<PathBuf>,
    pub(super) orthogonalize_adapter_side: OrthogonalizeSide,
    pub(super) alpha_edit_null_space_ridge: Option<f64>,
    pub(super) export_dtype: Option<ExportDtype>,
}

#[expect(
    clippy::as_conversions,
    reason = "Small LoRA rank/index integers convert directly to f64 for exact scaling."
)]
#[inline]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[expect(
    clippy::as_conversions,
    reason = "Checked finite f64 adapter weights narrow cleanly to f32 before GGUF encoding."
)]
#[inline]
const fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

#[expect(
    clippy::too_many_lines,
    reason = "Validates all algebraic, spectral, pruning, CANS, LS-LoRA, and orthogonalization editConfig fields."
)]
pub(super) fn validate_edit_config(config: &EditAdapterConfig) -> crate::Result<()> {
    if let Some(rank) = config.target_max_rank
        && !(1..=64).contains(&rank)
    {
        return Err("editConfig.targetMaxRank must be in 1..=64.".into());
    }
    for (target, &rank) in &config.per_site_ranks {
        if target.trim().is_empty() {
            return Err("editConfig.perSiteRanks contains an empty target name.".into());
        }
        if rank > 64 {
            return Err(format!(
                "editConfig.perSiteRanks[{target}] must be in 0..=64, got {rank}."
            ));
        }
    }
    let mut seen_drop_modes = BTreeSet::new();
    for &mode in &config.drop_modes {
        if mode >= 64 {
            return Err(format!(
                "editConfig.dropModes index {mode} must be in 0..64."
            ));
        }
        if !seen_drop_modes.insert(mode) {
            return Err(format!(
                "editConfig.dropModes contains duplicate mode index {mode}."
            ));
        }
    }
    if config.mode_weights.len() > 64 {
        return Err("editConfig.modeWeights length must not exceed 64.".into());
    }
    for (idx, &weight) in config.mode_weights.iter().enumerate() {
        if !weight.is_finite() || !(0.0..=100.0).contains(&weight) {
            return Err(format!(
                "editConfig.modeWeights[{idx}] must be finite and in [0.0, 100.0]."
            ));
        }
    }
    match (config.prune_keep_numerator, config.prune_keep_denominator) {
        (Some(num), Some(den)) => {
            if num == 0 || num > den || den > 1_024 {
                return Err(
                    "editConfig.pruneKeepNumerator/pruneKeepDenominator must satisfy 1 <= num <= denom <= 1024."
                        .into(),
                );
            }
        }
        (None, None) => {}
        _ => {
            return Err(
                "editConfig.pruneKeepNumerator and editConfig.pruneKeepDenominator must be supplied together."
                    .into(),
            );
        }
    }
    if let Some(thresh) = config.min_coeff_abs
        && (!thresh.is_finite() || thresh < 0.0)
    {
        return Err("editConfig.minCoeffAbs must be finite and >= 0.0.".into());
    }
    if let Some(gain) = config.global_gain
        && (!gain.is_finite() || gain <= 0.0 || gain > 100.0)
    {
        return Err("editConfig.globalGain must be finite and in (0.0, 100.0].".into());
    }
    for (family, &scale) in &config.family_scales {
        if !VALID_LINEAR_FAMILIES.contains(&family.as_str()) {
            return Err(format!(
                "Unsupported family '{family}' in editConfig.familyScales."
            ));
        }
        if !scale.is_finite() || !(0.0..=100.0).contains(&scale) {
            return Err(format!(
                "editConfig.familyScales['{family}'] must be finite and in [0.0, 100.0]."
            ));
        }
    }
    if let Some(min) = config.layer_min
        && min >= 64
    {
        return Err("editConfig.layerMin must be in 0..64.".into());
    }
    if let Some(max) = config.layer_max
        && max >= 64
    {
        return Err("editConfig.layerMax must be in 0..64.".into());
    }
    if let (Some(min), Some(max)) = (config.layer_min, config.layer_max)
        && min > max
    {
        return Err("editConfig.layerMin cannot exceed editConfig.layerMax.".into());
    }
    match (config.depth_gain_start, config.depth_gain_end) {
        (Some(start), Some(end)) => {
            if !start.is_finite()
                || !end.is_finite()
                || !(0.0..=100.0).contains(&start)
                || !(0.0..=100.0).contains(&end)
            {
                return Err(
                    "editConfig.depthGainStart/depthGainEnd must be finite and in [0.0, 100.0]."
                        .into(),
                );
            }
        }
        (None, None) => {}
        _ => {
            return Err(
                "editConfig.depthGainStart and editConfig.depthGainEnd must be supplied together."
                    .into(),
            );
        }
    }
    match config.depth_gain_curve {
        DepthGainCurve::Power => match config.depth_gain_exponent {
            Some(exp) if exp.is_finite() && (0.25..=8.0).contains(&exp) => {}
            Some(_) => {
                return Err(
                    "editConfig.depthGainExponent must be finite and in 0.25..=8.0 when depthGainCurve is 'power'."
                        .into(),
                );
            }
            None => {
                return Err(
                    "editConfig.depthGainExponent is required when depthGainCurve is 'power'."
                        .into(),
                );
            }
        },
        DepthGainCurve::Linear | DepthGainCurve::Cosine => {
            if config.depth_gain_exponent.is_some() {
                return Err(
                    "editConfig.depthGainExponent is only allowed when depthGainCurve is 'power'."
                        .into(),
                );
            }
        }
    }
    for (&layer, &gain) in &config.layer_gains {
        if layer >= 64 {
            return Err(format!(
                "editConfig.layerGains key {layer} must be in 0..64."
            ));
        }
        if !gain.is_finite() || !(0.0..=100.0).contains(&gain) {
            return Err(format!(
                "editConfig.layerGains[{layer}] must be finite and in [0.0, 100.0]."
            ));
        }
    }
    if let Some(min_norm) = config.min_site_frobenius_norm
        && (!min_norm.is_finite() || min_norm <= 0.0)
    {
        return Err("editConfig.minSiteFrobeniusNorm must be finite and positive.".into());
    }
    if let Some(cap) = config.max_site_frobenius_norm
        && (!cap.is_finite() || cap <= 0.0)
    {
        return Err("editConfig.maxSiteFrobeniusNorm must be finite and positive.".into());
    }
    if let (Some(min_norm), Some(max_norm)) = (
        config.min_site_frobenius_norm,
        config.max_site_frobenius_norm,
    ) && min_norm > max_norm
    {
        return Err(
            "editConfig.minSiteFrobeniusNorm cannot exceed editConfig.maxSiteFrobeniusNorm.".into(),
        );
    }
    match (
        &config.ls_lora_reference_adapter_path,
        config.ls_lora_min_cosine,
    ) {
        (Some(ref_path), Some(min_cos)) => {
            bonsai_artifacts::validate_external_path(ref_path).map_err(|e| e.to_string())?;
            if !min_cos.is_finite() || !(-1.0..=1.0).contains(&min_cos) {
                return Err("editConfig.lsLoraMinCosine must be finite and in -1.0..=1.0.".into());
            }
        }
        (None, None) => {}
        _ => {
            return Err(
                "editConfig.lsLoraReferenceAdapterPath and editConfig.lsLoraMinCosine must be supplied together."
                    .into(),
            );
        }
    }
    for (target, scales) in &config.cans_neuron_scales {
        if target.trim().is_empty() {
            return Err("editConfig.cansNeuronScales contains an empty target name.".into());
        }
        if scales.is_empty() {
            return Err(format!(
                "editConfig.cansNeuronScales['{target}'] must not be empty."
            ));
        }
        for (idx, &scale) in scales.iter().enumerate() {
            if !scale.is_finite() || !(0.0..=100.0).contains(&scale) {
                return Err(format!(
                    "editConfig.cansNeuronScales['{target}'][{idx}] must be finite and in [0.0, 100.0]."
                ));
            }
        }
    }
    if let Some(ref_path) = &config.orthogonalize_against_rank_one_path {
        bonsai_artifacts::validate_external_path(ref_path).map_err(|e| e.to_string())?;
    }
    if let Some(ref_path) = &config.orthogonalize_against_adapter_path {
        bonsai_artifacts::validate_external_path(ref_path).map_err(|e| e.to_string())?;
    }
    if let Some(ridge) = config.alpha_edit_null_space_ridge {
        if !ridge.is_finite() || ridge < 0.0 {
            return Err("editConfig.alphaEditNullSpaceRidge must be finite and >= 0.0.".into());
        }
        if config.orthogonalize_against_adapter_path.is_none()
            && config.orthogonalize_against_rank_one_path.is_none()
        {
            return Err(
                "editConfig.alphaEditNullSpaceRidge requires orthogonalizeAgainstAdapterPath or orthogonalizeAgainstRankOnePath."
                    .into(),
            );
        }
    }
    Ok(())
}

pub(super) fn verify_published_adapter(
    output_path: &std::path::Path,
    expected_sha256: &str,
    expected_bytes: u64,
    max_rank: usize,
    pair_count: usize,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<Value> {
    let probe = verify_lora_gguf_with_dims(
        output_path,
        &VerifyPolicy {
            expected_sha256: Some(expected_sha256.to_owned()),
            expected_envelope: None,
            require_npu_friendly: false,
            require_no_grouped_ssm: true,
            max_allowed_rank: Some(max_rank),
        },
        base_dims,
    )?;
    if probe["verified"] != true {
        return Err("Published adapter failed structural/SSM verification.".into());
    }
    let detected = probe["detectedEnvelope"]
        .as_str()
        .ok_or("Missing detectedEnvelope after publish.")?;
    let require_npu = matches!(detected, "residual-writers" | "npu-friendly");
    let verification = if require_npu {
        verify_lora_gguf_with_dims(
            output_path,
            &VerifyPolicy {
                expected_sha256: Some(expected_sha256.to_owned()),
                expected_envelope: Some(detected.to_owned()),
                require_npu_friendly: true,
                require_no_grouped_ssm: true,
                max_allowed_rank: Some(max_rank),
            },
            base_dims,
        )?
    } else {
        probe
    };
    if verification["verified"] != true {
        return Err("Published adapter failed verification gates.".into());
    }
    #[cfg(windows)]
    if require_npu && base_dims.len() > 200 {
        let rank_u16 = u16::try_from(max_rank).map_err(|e| e.to_string())?;
        let guard = bonsai_artifacts::verify_multi_rank_lora_gguf(
            output_path,
            rank_u16,
            expected_bytes,
            bonsai_artifacts::MultiRankEnvelope::NpuFriendly,
        )
        .map_err(|e| format!("Published adapter failed artifacts guard: {e:?}."))?;
        if guard.sha256() != expected_sha256
            || guard.max_rank() != rank_u16
            || guard.target_count() != pair_count
        {
            return Err("Published adapter artifacts guard metadata disagrees.".into());
        }
    }
    #[cfg(not(windows))]
    let _ = (expected_bytes, pair_count);
    Ok(verification)
}

pub(super) fn check_factor_value(val: f64, dtype: ExportDtype, label: &str) -> crate::Result<f32> {
    let val_f32 = f64_to_f32(val);
    if !val_f32.is_finite() || (dtype == ExportDtype::F16 && val_f32.abs() >= 65_520.0) {
        return Err(format!(
            "Adapter factor coefficient overflowed {label} target precision."
        ));
    }
    Ok(val_f32)
}

fn scale_pair_to_alpha(
    pair: &mut ExportTensorPair,
    old_alpha: f32,
    new_alpha: f32,
    dtype: ExportDtype,
) -> crate::Result<()> {
    if old_alpha.to_bits() == new_alpha.to_bits() {
        return Ok(());
    }
    let scale = (f64::from(old_alpha) / f64::from(new_alpha)).sqrt();
    for v in &mut pair.lora_a_f32 {
        *v = check_factor_value(f64::from(*v) * scale, dtype, "lora_a")?;
    }
    for v in &mut pair.lora_b_f32 {
        *v = check_factor_value(f64::from(*v) * scale, dtype, "lora_b")?;
    }
    Ok(())
}

/// Merges disjoint layer-chunk GGUF LoRA files into one verified GGUF without network I/O.
pub(super) fn run_merge_adapters(request: &Request) -> crate::Result<Value> {
    run_merge_adapters_with_dims(request, &swift_base_dims_map())
}

pub(super) fn run_merge_adapters_with_dims(
    request: &Request,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<Value> {
    if request.operation != Operation::MergeAdapters {
        return Err("Invalid operation dispatch for merge-adapters.".into());
    }
    let input_paths = request
        .input_paths
        .as_ref()
        .ok_or("merge-adapters requires inputPaths.")?;
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("merge-adapters requires outputPath.")?;

    let mut chunks = Vec::with_capacity(input_paths.len());
    let mut input_shas = Vec::with_capacity(input_paths.len());
    for path in input_paths {
        let loaded = read_lora_gguf_pairs_with_dims(path, base_dims)?;
        if loaded.detected_envelope == "custom" {
            return Err(
                "merge-adapters requires consistent non-custom envelopes across chunks.".into(),
            );
        }
        input_shas.push(loaded.sha256.clone());
        chunks.push(loaded);
    }

    let first_dtype = chunks[0].dtype;
    for chunk in &chunks[1..] {
        if chunk.dtype != first_dtype {
            return Err("merge-adapters requires identical exportDtype across all chunks.".into());
        }
    }

    let mut common_meta: Vec<(String, String)> = Vec::new();
    for key in COMMON_PROVENANCE_KEYS {
        let vals: Vec<Option<&str>> = chunks
            .iter()
            .map(|c| c.provenance_meta.get(key).map(String::as_str))
            .collect();
        let non_none: Vec<&str> = vals.iter().copied().flatten().collect();
        if !non_none.is_empty() {
            let first = non_none[0];
            if non_none.iter().any(|&v| v != first) {
                return Err(format!(
                    "merge-adapters detected provenance conflict on '{key}' across input chunks."
                ));
            }
            if non_none.len() == chunks.len() {
                common_meta.push((key.to_owned(), first.to_owned()));
            }
        }
    }
    common_meta.push((
        "bonsai.composition.merged_chunks".to_owned(),
        chunks.len().to_string(),
    ));

    let mut seen_targets = BTreeSet::new();
    let mut merged_pairs = Vec::new();
    let mut pair_chunk_alphas = Vec::new();
    let mut max_rank = 1_usize;

    for chunk in chunks {
        let chunk_alpha = chunk.alpha;
        for pair in chunk.pairs {
            if !seen_targets.insert(pair.gguf_base_name.clone()) {
                return Err(format!(
                    "Duplicate target '{}' across merge-adapters inputs.",
                    pair.gguf_base_name
                ));
            }
            max_rank = max_rank.max(pair.rank);
            merged_pairs.push(pair);
            pair_chunk_alphas.push(chunk_alpha);
        }
    }
    if merged_pairs.is_empty() {
        return Err("merge-adapters collected zero tensor pairs.".into());
    }

    let global_alpha = usize_to_f32(max_rank);
    for (pair, chunk_alpha) in merged_pairs.iter_mut().zip(pair_chunk_alphas) {
        scale_pair_to_alpha(pair, chunk_alpha, global_alpha, first_dtype)?;
    }
    sort_pairs_canonically(&mut merged_pairs);

    let meta_refs: Vec<(&str, &str)> = common_meta
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let written = write_lora_gguf(
        output_path,
        first_dtype,
        global_alpha,
        &meta_refs,
        &merged_pairs,
    )?;
    let verification = verify_published_adapter(
        output_path,
        &written.sha256,
        written.bytes,
        max_rank,
        merged_pairs.len(),
        base_dims,
    )?;

    Ok(json!({
        "schemaVersion": 1,
        "operation": "merge-adapters",
        "published": true,
        "status": "published-and-verified",
        "inputCount": input_shas.len(),
        "inputSha256s": input_shas,
        "targetEnvelope": verification["detectedEnvelope"],
        "exportDtype": first_dtype,
        "alpha": global_alpha,
        "minRank": verification["minRank"],
        "maxRank": verification["maxRank"],
        "bytes": written.bytes,
        "sha256": written.sha256,
        "tensorCount": written.tensor_count,
        "pairCount": merged_pairs.len(),
        "verification": verification,
    }))
}

pub(super) fn target_layer_and_family_map() -> BTreeMap<String, (u32, &'static str)> {
    let mut map = BTreeMap::new();
    for spec in swift_tensor_inventory() {
        if let Some(layer) = spec.layer {
            map.insert(spec.gguf_name, (layer, spec.family));
        }
    }
    map
}

pub(super) fn effective_delta_frobenius_norm_f64(
    a_f64: &[f64],
    b_f64: &[f64],
    in_features: usize,
    out_features: usize,
    rank: usize,
) -> crate::Result<f64> {
    let mut gram_b = vec![0.0_f64; rank * rank];
    for row in 0..out_features {
        let b_row = &b_f64[row * rank..(row + 1) * rank];
        for i in 0..rank {
            let bi = b_row[i];
            if bi == 0.0 {
                continue;
            }
            for j in i..rank {
                gram_b[i * rank + j] += bi * b_row[j];
            }
        }
    }
    let mut gram_a = vec![0.0_f64; rank * rank];
    for i in 0..rank {
        let a_i = &a_f64[i * in_features..(i + 1) * in_features];
        for j in i..rank {
            let a_j = &a_f64[j * in_features..(j + 1) * in_features];
            let dot: f64 = a_i.iter().zip(a_j).map(|(&x, &y)| x * y).sum();
            gram_a[i * rank + j] = dot;
            gram_a[j * rank + i] = dot;
        }
    }
    for i in 0..rank {
        for j in (i + 1)..rank {
            gram_b[j * rank + i] = gram_b[i * rank + j];
        }
    }
    let mut trace = 0.0_f64;
    for i in 0..rank {
        for j in 0..rank {
            trace += gram_b[i * rank + j] * gram_a[j * rank + i];
        }
    }
    if !trace.is_finite() {
        return Err("Edited LoRA Frobenius norm overflowed f64.".into());
    }
    Ok(trace.max(0.0).sqrt())
}

struct PendingEditedPair {
    gguf_base_name: String,
    in_features: usize,
    out_features: usize,
    rank: usize,
    family: &'static str,
    a_f64: Vec<f64>,
    b_f64: Vec<f64>,
}

#[inline]
fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

fn orthonormalize_columns_mgs_f64(cols: &[Vec<f64>]) -> crate::Result<Vec<Vec<f64>>> {
    if cols.is_empty() {
        return Ok(Vec::new());
    }
    let mut max_norm = 0.0_f64;
    for col in cols {
        for &v in col {
            if !v.is_finite() {
                return Err("Nonfinite value in reference factor during orthogonalization.".into());
            }
        }
        max_norm = max_norm.max(dot_f64(col, col).sqrt());
    }
    if max_norm == 0.0 {
        return Ok(Vec::new());
    }
    let threshold = max_norm * 1e-12;
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(cols.len());
    for raw_col in cols {
        let mut col = raw_col.clone();
        for _ in 0..2 {
            for prev in &basis {
                let proj = dot_f64(&col, prev);
                for (dst, &pv) in col.iter_mut().zip(prev) {
                    *dst -= proj * pv;
                }
            }
        }
        let norm = dot_f64(&col, &col).sqrt();
        if norm.is_finite() && norm > threshold {
            let inv = 1.0 / norm;
            for dst in &mut col {
                *dst *= inv;
            }
            basis.push(col);
        }
    }
    Ok(basis)
}

fn alpha_edit_weighted_basis(
    cols: &[Vec<f64>],
    dim: usize,
    ridge: f64,
) -> crate::Result<Vec<(f64, Vec<f64>)>> {
    let q = cols.len();
    if q == 0 || dim == 0 {
        return Ok(Vec::new());
    }
    let mut gram = vec![0.0_f64; q * q];
    for i in 0..q {
        for j in i..q {
            let val = dot_f64(&cols[i], &cols[j]);
            if !val.is_finite() {
                return Err("Reference factor Gram matrix overflowed f64.".into());
            }
            gram[i * q + j] = val;
            gram[j * q + i] = val;
        }
    }
    let (eigenvalues, eigvecs) = super::svd::jacobi_eigendecomp_f64(&mut gram, q)?;
    let sigma_max_sq = eigenvalues.first().copied().unwrap_or(0.0);
    if !sigma_max_sq.is_finite() || sigma_max_sq <= 0.0 {
        return Ok(Vec::new());
    }
    let tol_sq = 1e-12 * sigma_max_sq;
    let mut weighted = Vec::with_capacity(q);
    for (&sigma_sq, v) in eigenvalues.iter().zip(&eigvecs) {
        if sigma_sq <= tol_sq {
            break;
        }
        let sigma = sigma_sq.sqrt();
        let mut u_i = vec![0.0_f64; dim];
        for (j, &vj) in v.iter().enumerate() {
            if vj == 0.0 {
                continue;
            }
            let scale = vj / sigma;
            for (dst, &rj) in u_i.iter_mut().zip(&cols[j]) {
                *dst += rj * scale;
            }
        }
        let u_norm = dot_f64(&u_i, &u_i).sqrt();
        if !u_norm.is_finite() || u_norm <= 1e-12 {
            continue;
        }
        let inv = 1.0 / u_norm;
        for dst in &mut u_i {
            *dst *= inv;
        }
        let weight = sigma_sq / (sigma_sq + ridge);
        weighted.push((weight, u_i));
    }
    Ok(weighted)
}

fn build_subspace_projector(
    cols: &[Vec<f64>],
    dim: usize,
    ridge: Option<f64>,
) -> crate::Result<(Vec<(f64, Vec<f64>)>, bool)> {
    match ridge {
        Some(lambda) if lambda > 0.0 => Ok((alpha_edit_weighted_basis(cols, dim, lambda)?, false)),
        _ => {
            let basis = orthonormalize_columns_mgs_f64(cols)?;
            let hard = basis.into_iter().map(|u| (1.0_f64, u)).collect();
            Ok((hard, true))
        }
    }
}

fn apply_subspace_projector(x: &mut [f64], weighted_basis: &[(f64, Vec<f64>)], hard: bool) {
    if weighted_basis.is_empty() {
        return;
    }
    if hard {
        for _ in 0..2 {
            for (_, u) in weighted_basis {
                let dot = dot_f64(x, u);
                for (dst, &uv) in x.iter_mut().zip(u) {
                    *dst -= dot * uv;
                }
            }
        }
    } else {
        let dots: Vec<f64> = weighted_basis.iter().map(|(_, u)| dot_f64(x, u)).collect();
        for ((weight, u), dot) in weighted_basis.iter().zip(dots) {
            let factor = weight * dot;
            for (dst, &uv) in x.iter_mut().zip(u) {
                *dst -= factor * uv;
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Projects edited LoRA A/B factors against a reference multi-rank adapter subspace."
)]
fn project_out_adapter_subspace(
    a_f64: &mut [f64],
    b_f64: &mut [f64],
    in_features: usize,
    out_features: usize,
    rank: usize,
    ref_pair: &ExportTensorPair,
    ref_alpha: f32,
    side: OrthogonalizeSide,
    ridge: Option<f64>,
) -> crate::Result<()> {
    if side.includes_b() && ref_pair.out_features != out_features {
        return Err(format!(
            "Multi-rank reference output dimension mismatch on '{}'.",
            ref_pair.gguf_base_name
        ));
    }
    if side.includes_a() && ref_pair.in_features != in_features {
        return Err(format!(
            "Multi-rank reference input dimension mismatch on '{}'.",
            ref_pair.gguf_base_name
        ));
    }
    let s_ref = (f64::from(ref_alpha) / usize_to_f64(ref_pair.rank)).sqrt();
    if side.includes_b() {
        let mut ref_cols = Vec::with_capacity(ref_pair.rank);
        for j in 0..ref_pair.rank {
            let mut col = Vec::with_capacity(out_features);
            for row in 0..out_features {
                col.push(f64::from(ref_pair.lora_b_f32[row * ref_pair.rank + j]) * s_ref);
            }
            ref_cols.push(col);
        }
        let (projector, hard) = build_subspace_projector(&ref_cols, out_features, ridge)?;
        if !projector.is_empty() {
            let mut col_buf = vec![0.0_f64; out_features];
            for k in 0..rank {
                for row in 0..out_features {
                    col_buf[row] = b_f64[row * rank + k];
                }
                apply_subspace_projector(&mut col_buf, &projector, hard);
                for row in 0..out_features {
                    b_f64[row * rank + k] = col_buf[row];
                }
            }
        }
    }
    if side.includes_a() {
        let mut ref_rows = Vec::with_capacity(ref_pair.rank);
        for j in 0..ref_pair.rank {
            let row_off = j * in_features;
            let row: Vec<f64> = ref_pair.lora_a_f32[row_off..row_off + in_features]
                .iter()
                .map(|&v| f64::from(v) * s_ref)
                .collect();
            ref_rows.push(row);
        }
        let (projector, hard) = build_subspace_projector(&ref_rows, in_features, ridge)?;
        if !projector.is_empty() {
            for k in 0..rank {
                let row_slice = &mut a_f64[k * in_features..(k + 1) * in_features];
                apply_subspace_projector(row_slice, &projector, hard);
            }
        }
    }
    Ok(())
}

fn prune_slice_f64(
    values: &mut [f64],
    keep_fraction: Option<(usize, usize)>,
    min_coeff_abs: Option<f64>,
) -> crate::Result<usize> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err("LoRA factor coefficients must be finite before pruning.".into());
    }
    let mut pruned_count = 0_usize;
    if let Some((num, den)) = keep_fraction {
        let keep = values
            .len()
            .checked_mul(num)
            .ok_or("Pruning element count overflowed.")?
            .div_ceil(den);
        let mut indices: Vec<usize> = (0..values.len()).collect();
        if keep > 0 && keep < indices.len() {
            let _ = indices.select_nth_unstable_by(keep - 1, |&left, &right| {
                values[right]
                    .abs()
                    .total_cmp(&values[left].abs())
                    .then_with(|| left.cmp(&right))
            });
        }
        let mut retained = vec![false; values.len()];
        for &idx in &indices[..keep] {
            retained[idx] = true;
        }
        for (val, keep_flag) in values.iter_mut().zip(retained) {
            if !keep_flag && *val != 0.0 {
                *val = 0.0;
                pruned_count += 1;
            }
        }
    }
    if let Some(thresh) = min_coeff_abs
        && thresh > 0.0
    {
        for val in values.iter_mut() {
            if *val != 0.0 && val.abs() < thresh {
                *val = 0.0;
                pruned_count += 1;
            }
        }
    }
    Ok(pruned_count)
}

fn delta_cosine_with_ref_f64(
    a_f64: &[f64],
    b_f64: &[f64],
    in_features: usize,
    out_features: usize,
    rank: usize,
    norm_cand: f64,
    ref_pair: &ExportTensorPair,
) -> crate::Result<Option<f64>> {
    if ref_pair.in_features != in_features || ref_pair.out_features != out_features {
        return Err(format!(
            "LS-LoRA reference dimension mismatch on '{}'.",
            ref_pair.gguf_base_name
        ));
    }
    if norm_cand == 0.0 {
        return Ok(None);
    }
    let r_ref = ref_pair.rank;
    let ref_a_f64: Vec<f64> = ref_pair.lora_a_f32.iter().copied().map(f64::from).collect();
    let ref_b_f64: Vec<f64> = ref_pair.lora_b_f32.iter().copied().map(f64::from).collect();
    let norm_ref = effective_delta_frobenius_norm_f64(
        &ref_a_f64,
        &ref_b_f64,
        in_features,
        out_features,
        r_ref,
    )?;
    if norm_ref == 0.0 {
        return Ok(None);
    }

    let mut m_b = vec![0.0_f64; rank * r_ref];
    for row in 0..out_features {
        let b_row = &b_f64[row * rank..(row + 1) * rank];
        let b_ref_row = &ref_b_f64[row * r_ref..(row + 1) * r_ref];
        for i in 0..rank {
            let bi = b_row[i];
            if bi == 0.0 {
                continue;
            }
            for j in 0..r_ref {
                m_b[i * r_ref + j] += bi * b_ref_row[j];
            }
        }
    }

    let mut m_a = vec![0.0_f64; r_ref * rank];
    for j in 0..r_ref {
        let a_ref_row = &ref_a_f64[j * in_features..(j + 1) * in_features];
        for i in 0..rank {
            let a_row = &a_f64[i * in_features..(i + 1) * in_features];
            m_a[j * rank + i] = dot_f64(a_ref_row, a_row);
        }
    }

    let mut inner = 0.0_f64;
    for i in 0..rank {
        for j in 0..r_ref {
            inner += m_b[i * r_ref + j] * m_a[j * rank + i];
        }
    }
    let cos = inner / (norm_cand * norm_ref);
    if !cos.is_finite() {
        return Err("LS-LoRA Frobenius cosine overflowed f64.".into());
    }
    Ok(Some(cos.clamp(-1.0, 1.0)))
}

fn depth_gain_t(layer: u32, curve: DepthGainCurve, exponent: Option<f64>) -> f64 {
    let x = f64::from(layer) / 63.0;
    match curve {
        DepthGainCurve::Linear => x,
        DepthGainCurve::Cosine => 0.5 * (1.0 - (std::f64::consts::PI * x).cos()),
        DepthGainCurve::Power => x.powf(exponent.unwrap_or(1.0)),
    }
}

/// Applies pure-Rust algebraic direct weight edits to an existing multi-rank GGUF LoRA.
pub(super) fn run_edit_adapter(request: &Request) -> crate::Result<Value> {
    run_edit_adapter_with_dims(request, &swift_base_dims_map())
}

#[expect(
    clippy::too_many_lines,
    reason = "Applies mode filtering/weighting, rank slicing, CANS, Orca/AlphaEdit orthogonalization, pruning, LS-LoRA selection, norm capping, and atomic verification."
)]
pub(super) fn run_edit_adapter_with_dims(
    request: &Request,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<Value> {
    if request.operation != Operation::EditAdapter {
        return Err("Invalid operation dispatch for edit-adapter.".into());
    }
    let adapter_path = request
        .adapter_path
        .as_ref()
        .ok_or("edit-adapter requires adapterPath.")?;
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("edit-adapter requires outputPath.")?;
    let config = request
        .edit_config
        .as_ref()
        .ok_or("edit-adapter requires editConfig.")?;
    validate_edit_config(config)?;

    let loaded = read_lora_gguf_pairs_with_dims(adapter_path, base_dims)?;
    if !config.per_site_ranks.is_empty() {
        let available: BTreeMap<&str, usize> = loaded
            .pairs
            .iter()
            .map(|p| (p.gguf_base_name.as_str(), p.rank))
            .collect();
        for (target, &requested) in &config.per_site_ranks {
            let Some(&src_rank) = available.get(target.as_str()) else {
                return Err(format!(
                    "editConfig.perSiteRanks target '{target}' is not present in source adapter."
                ));
            };
            if requested > src_rank {
                return Err(format!(
                    "editConfig.perSiteRanks[{target}] rank {requested} exceeds loaded source rank {src_rank}."
                ));
            }
        }
    }
    if !config.cans_neuron_scales.is_empty() {
        let available_dims: BTreeMap<&str, usize> = loaded
            .pairs
            .iter()
            .map(|p| (p.gguf_base_name.as_str(), p.in_features))
            .collect();
        for (target, scales) in &config.cans_neuron_scales {
            let Some(&in_features) = available_dims.get(target.as_str()) else {
                return Err(format!(
                    "editConfig.cansNeuronScales target '{target}' is not present in source adapter."
                ));
            };
            if scales.len() != in_features {
                return Err(format!(
                    "editConfig.cansNeuronScales['{target}'] length {} does not match in_features {in_features}.",
                    scales.len()
                ));
            }
        }
    }

    let use_swift_dims = base_dims.len() > 200;
    let rank_one_b_dirs = match &config.orthogonalize_against_rank_one_path {
        Some(path) if config.orthogonalize_side.includes_b() => Some(if use_swift_dims {
            read_rank_one_b_directions(path)?
        } else {
            read_rank_one_b_directions_with_dims(path, base_dims)?
        }),
        _ => None,
    };
    let rank_one_a_dirs = match &config.orthogonalize_against_rank_one_path {
        Some(path) if config.orthogonalize_side.includes_a() => Some(if use_swift_dims {
            read_rank_one_a_directions(path)?
        } else {
            read_rank_one_a_directions_with_dims(path, base_dims)?
        }),
        _ => None,
    };
    let multi_rank_orth_adapter = match &config.orthogonalize_against_adapter_path {
        Some(path) => Some(read_lora_gguf_pairs_with_dims(path, base_dims)?),
        None => None,
    };
    let multi_rank_orth_map: BTreeMap<&str, &ExportTensorPair> = multi_rank_orth_adapter
        .as_ref()
        .map(|adapter| {
            adapter
                .pairs
                .iter()
                .map(|p| (p.gguf_base_name.as_str(), p))
                .collect()
        })
        .unwrap_or_default();
    let ls_lora_ref_adapter = match &config.ls_lora_reference_adapter_path {
        Some(path) => Some(read_lora_gguf_pairs_with_dims(path, base_dims)?),
        None => None,
    };
    let ls_lora_ref_map: BTreeMap<&str, &ExportTensorPair> = ls_lora_ref_adapter
        .as_ref()
        .map(|adapter| {
            adapter
                .pairs
                .iter()
                .map(|p| (p.gguf_base_name.as_str(), p))
                .collect()
        })
        .unwrap_or_default();

    let drop_mode_set: BTreeSet<usize> = config.drop_modes.iter().copied().collect();
    let keep_fraction = config
        .prune_keep_numerator
        .zip(config.prune_keep_denominator);
    let rank_one_proj_weight = if config.orthogonalize_against_adapter_path.is_none() {
        match config.alpha_edit_null_space_ridge {
            Some(lambda) if lambda > 0.0 => 1.0 / (1.0 + lambda),
            _ => 1.0,
        }
    } else {
        1.0
    };

    let out_dtype = config.export_dtype.unwrap_or(loaded.dtype);
    let layer_map = target_layer_and_family_map();

    let mut pending = Vec::with_capacity(loaded.pairs.len());
    let mut dropped_sites = 0_usize;
    let mut orthogonalized_sites = 0_usize;
    let mut multi_rank_orthogonalized_sites = 0_usize;
    let mut cans_scaled_sites = 0_usize;
    let mut pruned_coeff_count = 0_usize;
    let mut ls_lora_dropped_sites = 0_usize;
    let mut capped_sites = 0_usize;
    let mut max_new_rank = 0_usize;
    let mut families_used = BTreeSet::new();

    for pair in &loaded.pairs {
        let &(layer, inv_family) = layer_map
            .get(&pair.gguf_base_name)
            .ok_or_else(|| format!("Unknown Swift layer for '{}'.", pair.gguf_base_name))?;
        let family = base_dims
            .get(&pair.gguf_base_name)
            .map_or(inv_family, |&(_, _, f)| f);

        if config.layer_min.is_some_and(|min| layer < min)
            || config.layer_max.is_some_and(|max| layer > max)
        {
            dropped_sites += 1;
            continue;
        }
        let global_gain = config.global_gain.unwrap_or(1.0);
        let fam_gain = config.family_scales.get(family).copied().unwrap_or(1.0);
        let depth_gain = match (config.depth_gain_start, config.depth_gain_end) {
            (Some(start), Some(end)) => {
                let t = depth_gain_t(layer, config.depth_gain_curve, config.depth_gain_exponent);
                start + (end - start) * t
            }
            _ => 1.0,
        };
        let layer_gain = config.layer_gains.get(&layer).copied().unwrap_or(1.0);
        let site_gain = global_gain * fam_gain * depth_gain * layer_gain;
        if site_gain == 0.0 {
            dropped_sites += 1;
            continue;
        }

        let requested_rank = config
            .per_site_ranks
            .get(&pair.gguf_base_name)
            .copied()
            .unwrap_or(pair.rank);
        if requested_rank == 0 {
            dropped_sites += 1;
            continue;
        }
        for &drop_idx in &config.drop_modes {
            if drop_idx >= pair.rank {
                return Err(format!(
                    "editConfig.dropModes index {drop_idx} exceeds source rank {} on '{}'.",
                    pair.rank, pair.gguf_base_name
                ));
            }
        }
        let mut kept_modes: Vec<usize> = (0..pair.rank)
            .filter(|k| !drop_mode_set.contains(k))
            .collect();
        let cap_rank = config.target_max_rank.unwrap_or(pair.rank);
        let new_rank = requested_rank.min(cap_rank).min(kept_modes.len());
        if new_rank == 0 {
            dropped_sites += 1;
            continue;
        }
        kept_modes.truncate(new_rank);
        if config.mode_weights.len() > new_rank {
            return Err(format!(
                "editConfig.modeWeights length {} exceeds retained rank {new_rank} on '{}'.",
                config.mode_weights.len(),
                pair.gguf_base_name
            ));
        }

        // Rescale kept modes so (1.0) * B_tmp * A_tmp equals site_gain * sum_{k < new_rank} w_k * delta_{kept_modes[k]}.
        let s1 = (site_gain * f64::from(loaded.alpha) / usize_to_f64(pair.rank)).sqrt();
        let mode_scales: Vec<f64> = (0..new_rank)
            .map(|k| s1 * config.mode_weights.get(k).copied().unwrap_or(1.0).sqrt())
            .collect();
        let a_len = new_rank
            .checked_mul(pair.in_features)
            .ok_or("Edited lora_a length overflow.")?;
        let b_len = pair
            .out_features
            .checked_mul(new_rank)
            .ok_or("Edited lora_b length overflow.")?;

        let mut a_f64 = Vec::new();
        a_f64
            .try_reserve_exact(a_len)
            .map_err(|e| format!("Cannot allocate edited lora_a buffer: {e}"))?;
        for (k, &orig_k) in kept_modes.iter().enumerate() {
            let row_off = orig_k * pair.in_features;
            let scale_k = mode_scales[k];
            for col in 0..pair.in_features {
                a_f64.push(f64::from(pair.lora_a_f32[row_off + col]) * scale_k);
            }
        }

        let mut b_f64 = Vec::new();
        b_f64
            .try_reserve_exact(b_len)
            .map_err(|e| format!("Cannot allocate edited lora_b buffer: {e}"))?;
        for row in 0..pair.out_features {
            let row_off = row * pair.rank;
            for (k, &orig_k) in kept_modes.iter().enumerate() {
                b_f64.push(f64::from(pair.lora_b_f32[row_off + orig_k]) * mode_scales[k]);
            }
        }

        if let Some(scales) = config.cans_neuron_scales.get(&pair.gguf_base_name) {
            for k in 0..new_rank {
                let a_row = &mut a_f64[k * pair.in_features..(k + 1) * pair.in_features];
                for (dst, &s) in a_row.iter_mut().zip(scales) {
                    *dst *= s;
                }
            }
            cans_scaled_sites += 1;
        }

        let mut rank_one_hit = false;
        if let Some(ref_dirs) = &rank_one_b_dirs
            && let Some(dir) = ref_dirs.get(&pair.gguf_base_name)
        {
            if dir.len() != pair.out_features {
                return Err(format!(
                    "Rank-1 reference output dimension mismatch on '{}'.",
                    pair.gguf_base_name
                ));
            }
            for k in 0..new_rank {
                let mut dot = 0.0_f64;
                for row in 0..pair.out_features {
                    dot += dir[row] * b_f64[row * new_rank + k];
                }
                let factor = dot * rank_one_proj_weight;
                for row in 0..pair.out_features {
                    b_f64[row * new_rank + k] -= factor * dir[row];
                }
            }
            rank_one_hit = true;
        }
        if let Some(ref_dirs) = &rank_one_a_dirs
            && let Some(dir) = ref_dirs.get(&pair.gguf_base_name)
        {
            if dir.len() != pair.in_features {
                return Err(format!(
                    "Rank-1 reference input dimension mismatch on '{}'.",
                    pair.gguf_base_name
                ));
            }
            for k in 0..new_rank {
                let a_row = &mut a_f64[k * pair.in_features..(k + 1) * pair.in_features];
                let dot = dot_f64(dir, a_row);
                let factor = dot * rank_one_proj_weight;
                for (dst, &dv) in a_row.iter_mut().zip(dir) {
                    *dst -= factor * dv;
                }
            }
            rank_one_hit = true;
        }
        if rank_one_hit {
            orthogonalized_sites += 1;
        }

        if let Some(ref_adapter) = &multi_rank_orth_adapter
            && let Some(&ref_pair) = multi_rank_orth_map.get(pair.gguf_base_name.as_str())
        {
            project_out_adapter_subspace(
                &mut a_f64,
                &mut b_f64,
                pair.in_features,
                pair.out_features,
                new_rank,
                ref_pair,
                ref_adapter.alpha,
                config.orthogonalize_adapter_side,
                config.alpha_edit_null_space_ridge,
            )?;
            multi_rank_orthogonalized_sites += 1;
        }

        if keep_fraction.is_some() || config.min_coeff_abs.is_some() {
            if config.prune_target.includes_a() {
                for k in 0..new_rank {
                    let row_slice = &mut a_f64[k * pair.in_features..(k + 1) * pair.in_features];
                    pruned_coeff_count +=
                        prune_slice_f64(row_slice, keep_fraction, config.min_coeff_abs)?;
                }
            }
            if config.prune_target.includes_b() {
                let mut col_buf = vec![0.0_f64; pair.out_features];
                for k in 0..new_rank {
                    for row in 0..pair.out_features {
                        col_buf[row] = b_f64[row * new_rank + k];
                    }
                    pruned_coeff_count +=
                        prune_slice_f64(&mut col_buf, keep_fraction, config.min_coeff_abs)?;
                    for row in 0..pair.out_features {
                        b_f64[row * new_rank + k] = col_buf[row];
                    }
                }
            }
        }

        let needs_frob = config.min_site_frobenius_norm.is_some()
            || config.ls_lora_min_cosine.is_some()
            || config.max_site_frobenius_norm.is_some();
        let frob = if needs_frob {
            effective_delta_frobenius_norm_f64(
                &a_f64,
                &b_f64,
                pair.in_features,
                pair.out_features,
                new_rank,
            )?
        } else {
            0.0
        };
        if let Some(min_norm) = config.min_site_frobenius_norm
            && frob < min_norm
        {
            ls_lora_dropped_sites += 1;
            dropped_sites += 1;
            continue;
        }
        if let Some(min_cos) = config.ls_lora_min_cosine {
            let Some(&ref_pair) = ls_lora_ref_map.get(pair.gguf_base_name.as_str()) else {
                ls_lora_dropped_sites += 1;
                dropped_sites += 1;
                continue;
            };
            let maybe_cos = delta_cosine_with_ref_f64(
                &a_f64,
                &b_f64,
                pair.in_features,
                pair.out_features,
                new_rank,
                frob,
                ref_pair,
            )?;
            if maybe_cos.is_none_or(|cos| cos < min_cos) {
                ls_lora_dropped_sites += 1;
                dropped_sites += 1;
                continue;
            }
        }
        if let Some(cap) = config.max_site_frobenius_norm
            && frob > cap
        {
            let damp = (cap / frob).sqrt();
            for v in &mut a_f64 {
                *v *= damp;
            }
            for v in &mut b_f64 {
                *v *= damp;
            }
            capped_sites += 1;
        }

        max_new_rank = max_new_rank.max(new_rank);
        families_used.insert(family);
        pending.push(PendingEditedPair {
            gguf_base_name: pair.gguf_base_name.clone(),
            in_features: pair.in_features,
            out_features: pair.out_features,
            rank: new_rank,
            family,
            a_f64,
            b_f64,
        });
    }

    if pending.is_empty() {
        return Err("edit-adapter filtered out all target matrices.".into());
    }

    let new_alpha = usize_to_f32(max_new_rank);
    let mut final_pairs = Vec::with_capacity(pending.len());
    for item in pending {
        let s2 = (usize_to_f64(item.rank) / f64::from(new_alpha)).sqrt();
        let mut lora_a_f32 = Vec::with_capacity(item.a_f64.len());
        for v in item.a_f64 {
            lora_a_f32.push(check_factor_value(v * s2, out_dtype, "edited lora_a")?);
        }
        let mut lora_b_f32 = Vec::with_capacity(item.b_f64.len());
        for v in item.b_f64 {
            lora_b_f32.push(check_factor_value(v * s2, out_dtype, "edited lora_b")?);
        }
        let _ = item.family;
        final_pairs.push(ExportTensorPair {
            gguf_base_name: item.gguf_base_name,
            in_features: item.in_features,
            out_features: item.out_features,
            rank: item.rank,
            lora_a_f32,
            lora_b_f32,
        });
    }
    sort_pairs_canonically(&mut final_pairs);

    let mut meta: Vec<(String, String)> = loaded
        .provenance_meta
        .into_iter()
        .filter(|(k, _)| {
            k != "bonsai.composition.edit_parent_sha256" && k != "bonsai.composition.edit_method"
        })
        .collect();
    meta.push((
        "bonsai.composition.edit_parent_sha256".to_owned(),
        loaded.sha256.clone(),
    ));
    meta.push((
        "bonsai.composition.edit_method".to_owned(),
        "algebraic-multi-rank-edit-v1".to_owned(),
    ));
    let meta_refs: Vec<(&str, &str)> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

    let (expected_env, _, _) = classify_families(&families_used);
    let written = write_lora_gguf(output_path, out_dtype, new_alpha, &meta_refs, &final_pairs)?;
    let verification = verify_published_adapter(
        output_path,
        &written.sha256,
        written.bytes,
        max_new_rank,
        final_pairs.len(),
        base_dims,
    )?;

    Ok(json!({
        "schemaVersion": 1,
        "operation": "edit-adapter",
        "published": true,
        "status": "published-and-verified",
        "inputSha256": loaded.sha256,
        "targetEnvelope": expected_env,
        "exportDtype": out_dtype,
        "alpha": new_alpha,
        "minRank": verification["minRank"],
        "maxRank": verification["maxRank"],
        "bytes": written.bytes,
        "sha256": written.sha256,
        "tensorCount": written.tensor_count,
        "pairCount": final_pairs.len(),
        "droppedSiteCount": dropped_sites,
        "orthogonalizedSiteCount": orthogonalized_sites,
        "multiRankOrthogonalizedSiteCount": multi_rank_orthogonalized_sites,
        "cansScaledSiteCount": cans_scaled_sites,
        "prunedCoeffCount": pruned_coeff_count,
        "lsLoraDroppedCount": ls_lora_dropped_sites,
        "cappedSiteCount": capped_sites,
        "verification": verification,
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::bounded_input::ScratchDir;

    fn reconstruct_delta(pair: &ExportTensorPair, alpha: f32) -> Vec<f64> {
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

    fn toy_dims_three() -> BTreeMap<String, (usize, usize, &'static str)> {
        [
            ("blk.0.ffn_down.weight".to_owned(), (8, 6, "ffn_down")),
            ("blk.21.ffn_down.weight".to_owned(), (8, 6, "ffn_down")),
            ("blk.63.ffn_down.weight".to_owned(), (8, 6, "ffn_down")),
        ]
        .into_iter()
        .collect()
    }

    fn make_basis_pair(name: &str, in_f: usize, out_f: usize, rank: usize) -> ExportTensorPair {
        let mut a = vec![0.0_f32; rank * in_f];
        let mut b = vec![0.0_f32; out_f * rank];
        for k in 0..rank {
            a[k * in_f + k] = f64_to_f32(usize_to_f64(k + 1));
            b[k * rank + k] = 1.0;
        }
        ExportTensorPair {
            gguf_base_name: name.to_owned(),
            in_features: in_f,
            out_features: out_f,
            rank,
            lora_a_f32: a,
            lora_b_f32: b,
        }
    }

    fn run_toy_edit(
        src: &std::path::Path,
        out: &std::path::Path,
        edit_config: Value,
        dims: &BTreeMap<String, (usize, usize, &'static str)>,
    ) -> crate::Result<Value> {
        let req: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "edit-adapter",
            "adapterPath": src,
            "outputPath": out,
            "editConfig": edit_config
        }))
        .map_err(|e| e.to_string())?;
        super::super::validate_request(&req)?;
        run_edit_adapter_with_dims(&req, dims)
    }

    #[test]
    fn edit_adapter_drop_modes_and_mode_weights_work_and_validate() -> crate::Result<()> {
        let scratch = ScratchDir::new("edit-drop-mode-weights")?;
        let dims = toy_dims_three();
        let src = scratch.join("src-r4.gguf");
        let pair0 = make_basis_pair("blk.0.ffn_down.weight", 8, 6, 4);
        write_lora_gguf(
            &src,
            ExportDtype::F32,
            4.0,
            &[("bonsai.composition.method", "mode-test")],
            std::slice::from_ref(&pair0),
        )?;

        let out = scratch.join("edited-modes.gguf");
        let res = run_toy_edit(
            &src,
            &out,
            json!({
                "dropModes": [0, 2],
                "modeWeights": [4.0, 0.25]
            }),
            &dims,
        )?;
        assert_eq!(res["published"], true);
        assert_eq!(res["maxRank"], 2);
        assert_eq!(res["alpha"], 2.0);

        let edited = read_lora_gguf_pairs_with_dims(&out, &dims)?;
        assert_eq!(edited.pairs[0].rank, 2);
        let delta = reconstruct_delta(&edited.pairs[0], edited.alpha);
        // Mode 1 had (row=1, col=1)=2.0 scaled by 4.0 -> 8.0; Mode 3 had (row=3, col=3)=4.0 scaled by 0.25 -> 1.0.
        assert!((delta[0] - 0.0).abs() < 1e-6);
        assert!((delta[8 + 1] - 8.0).abs() < 1e-5);
        assert!((delta[2 * 8 + 2] - 0.0).abs() < 1e-6);
        assert!((delta[3 * 8 + 3] - 1.0).abs() < 1e-5);

        // Dropping modes [0, 2] then capping targetMaxRank=1 retains mode 1 only, with weight 0.5.
        let out_cap = scratch.join("edited-cap.gguf");
        let res_cap = run_toy_edit(
            &src,
            &out_cap,
            json!({
                "dropModes": [0, 2],
                "targetMaxRank": 1,
                "modeWeights": [0.5]
            }),
            &dims,
        )?;
        assert_eq!(res_cap["maxRank"], 1);
        let edited_cap = read_lora_gguf_pairs_with_dims(&out_cap, &dims)?;
        let delta_cap = reconstruct_delta(&edited_cap.pairs[0], edited_cap.alpha);
        assert!((delta_cap[8 + 1] - 1.0).abs() < 1e-5);
        assert!((delta_cap[3 * 8 + 3] - 0.0).abs() < 1e-6);

        // Zero weight mode zeros cleanly without dropping rank.
        let out_zero_mode = scratch.join("edited-zero-mode.gguf");
        let res_zero_mode = run_toy_edit(
            &src,
            &out_zero_mode,
            json!({
                "dropModes": [0, 2],
                "modeWeights": [0.0, 1.0]
            }),
            &dims,
        )?;
        assert_eq!(res_zero_mode["maxRank"], 2);
        let edited_zm = read_lora_gguf_pairs_with_dims(&out_zero_mode, &dims)?;
        let delta_zm = reconstruct_delta(&edited_zm.pairs[0], edited_zm.alpha);
        assert!(delta_zm[8 + 1].abs() < 1e-6);
        assert!((delta_zm[3 * 8 + 3] - 4.0).abs() < 1e-5);

        // Validation errors: duplicate dropModes, >= 64, out of rank bounds, modeWeights too long, negative weight.
        for bad_cfg in [
            json!({ "dropModes": [1, 1] }),
            json!({ "dropModes": [64] }),
            json!({ "dropModes": [4] }),
            json!({ "modeWeights": [1.0, 1.0, 1.0, 1.0, 1.0] }),
            json!({ "dropModes": [0, 1], "modeWeights": [1.0, 1.0, 1.0] }),
            json!({ "modeWeights": [-0.1] }),
            json!({ "modeWeights": [101.0] }),
        ] {
            assert!(run_toy_edit(&src, &scratch.join("bad-modes.gguf"), bad_cfg, &dims).is_err());
        }
        Ok(())
    }

    #[test]
    fn edit_adapter_pruning_target_a_b_both_and_min_coeff_abs() -> crate::Result<()> {
        let scratch = ScratchDir::new("edit-prune")?;
        let dims = toy_dims_three();
        let src = scratch.join("src-prune.gguf");
        let pair = ExportTensorPair {
            gguf_base_name: "blk.0.ffn_down.weight".to_owned(),
            in_features: 8,
            out_features: 6,
            rank: 1,
            // Stable-index tie on -4.0 and 4.0: keep 1/2 = 4 elements -> indices [1(-4), 3(4), 4(-4), 2(2)].
            lora_a_f32: vec![1.0, -4.0, 2.0, 4.0, -4.0, 0.25, -4.0, 0.0],
            lora_b_f32: vec![0.1, -3.0, 3.0, 0.5, -2.0, 0.05],
        };
        write_lora_gguf(&src, ExportDtype::F32, 1.0, &[], &[pair])?;

        // 1. Prune A (1/2) with minCoeffAbs: tie breaks to lower index (idx 1,3,4,6 have |v|=4.0; all 4 kept!).
        let out_a = scratch.join("prune-a.gguf");
        let res_a = run_toy_edit(
            &src,
            &out_a,
            json!({
                "pruneKeepNumerator": 1,
                "pruneKeepDenominator": 2,
                "pruneTarget": "a"
            }),
            &dims,
        )?;
        assert_eq!(res_a["prunedCoeffCount"], 3); // 1.0, 2.0, 0.25 zeroed (0.0 was already 0)
        let loaded_a = read_lora_gguf_pairs_with_dims(&out_a, &dims)?;
        assert_eq!(
            loaded_a.pairs[0].lora_a_f32,
            vec![0.0, -4.0, 0.0, 4.0, -4.0, 0.0, -4.0, 0.0]
        );
        assert_eq!(
            loaded_a.pairs[0].lora_b_f32,
            vec![0.1, -3.0, 3.0, 0.5, -2.0, 0.05]
        );

        // 2. Prune Both with minCoeffAbs = 1.0 (keeps only |v| >= 1.0 in both A and B).
        let out_both = scratch.join("prune-both.gguf");
        let res_both = run_toy_edit(
            &src,
            &out_both,
            json!({
                "pruneTarget": "both",
                "minCoeffAbs": 1.0
            }),
            &dims,
        )?;
        // A drops 0.25 (1 coeff); B drops 0.1, 0.5, 0.05 (3 coeffs) -> total 4.
        assert_eq!(res_both["prunedCoeffCount"], 4);
        let loaded_both = read_lora_gguf_pairs_with_dims(&out_both, &dims)?;
        assert_eq!(
            loaded_both.pairs[0].lora_a_f32,
            vec![1.0, -4.0, 2.0, 4.0, -4.0, 0.0, -4.0, 0.0]
        );
        assert_eq!(
            loaded_both.pairs[0].lora_b_f32,
            vec![0.0, -3.0, 3.0, 0.0, -2.0, 0.0]
        );

        // 3. Prune B top 1/3 (ceil(6/3) = 2: -3.0 and 3.0).
        let out_b = scratch.join("prune-b.gguf");
        let res_b = run_toy_edit(
            &src,
            &out_b,
            json!({
                "pruneKeepNumerator": 1,
                "pruneKeepDenominator": 3,
                "pruneTarget": "b"
            }),
            &dims,
        )?;
        assert_eq!(res_b["prunedCoeffCount"], 4);
        let loaded_b = read_lora_gguf_pairs_with_dims(&out_b, &dims)?;
        assert_eq!(
            loaded_b.pairs[0].lora_b_f32,
            vec![0.0, -3.0, 3.0, 0.0, 0.0, 0.0]
        );

        // Validation failures.
        for bad_cfg in [
            json!({ "pruneKeepNumerator": 1 }),
            json!({ "pruneKeepDenominator": 2 }),
            json!({ "pruneKeepNumerator": 0, "pruneKeepDenominator": 2 }),
            json!({ "pruneKeepNumerator": 3, "pruneKeepDenominator": 2 }),
            json!({ "pruneKeepNumerator": 1, "pruneKeepDenominator": 1025 }),
            json!({ "minCoeffAbs": -0.01 }),
        ] {
            assert!(run_toy_edit(&src, &scratch.join("bad-prune.gguf"), bad_cfg, &dims).is_err());
        }
        Ok(())
    }

    #[test]
    fn edit_adapter_rank_one_a_and_alpha_edit_multi_rank_orthogonalization() -> crate::Result<()> {
        let scratch = ScratchDir::new("edit-alpha-edit")?;
        let dims = toy_dims_three();

        // Real-dimension smoke check for `read_rank_one_a_directions` directly.
        let r1_real = scratch.join("r1-real.gguf");
        let mut r1_real_a = vec![0.0_f32; 5_120];
        r1_real_a[0] = 3.0;
        r1_real_a[1] = 4.0;
        let mut r1_real_b = vec![0.0_f32; 1_024];
        r1_real_b[2] = 2.0;
        write_lora_gguf(
            &r1_real,
            ExportDtype::F32,
            1.0,
            &[],
            &[ExportTensorPair {
                gguf_base_name: "blk.3.attn_k.weight".to_owned(),
                in_features: 5_120,
                out_features: 1_024,
                rank: 1,
                lora_a_f32: r1_real_a,
                lora_b_f32: r1_real_b,
            }],
        )?;
        let dirs_a = read_rank_one_a_directions(&r1_real)?;
        let dir_a = dirs_a
            .get("blk.3.attn_k.weight")
            .ok_or("Missing rank-1 A direction.")?;
        assert_eq!(dir_a.len(), 5_120);
        assert!((dir_a[0] - 0.6).abs() < 1e-6);
        assert!((dir_a[1] - 0.8).abs() < 1e-6);

        // Source adapter at blk.0 (in=8, out=6, rank=2).
        let src = scratch.join("src-orth.gguf");
        let src_pair = ExportTensorPair {
            gguf_base_name: "blk.0.ffn_down.weight".to_owned(),
            in_features: 8,
            out_features: 6,
            rank: 2,
            lora_a_f32: vec![
                1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, // row 0
                0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 1
            ],
            lora_b_f32: vec![
                1.0, 0.0, // row 0: e_0
                1.0, 1.0, // row 1: e_1
                0.5, 2.0, // row 2: e_2
                0.0, 0.0, // row 3
                0.0, 0.0, // row 4
                0.0, 0.0, // row 5
            ],
        };
        write_lora_gguf(&src, ExportDtype::F32, 2.0, &[], &[src_pair])?;

        // 1. Rank-1 orthogonalizeSide: "both".
        let r1_toy = scratch.join("r1-toy.gguf");
        write_lora_gguf(
            &r1_toy,
            ExportDtype::F32,
            1.0,
            &[],
            &[ExportTensorPair {
                gguf_base_name: "blk.0.ffn_down.weight".to_owned(),
                in_features: 8,
                out_features: 6,
                rank: 1,
                lora_a_f32: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                lora_b_f32: vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            }],
        )?;
        let out_r1_both = scratch.join("out-r1-both.gguf");
        let res_r1 = run_toy_edit(
            &src,
            &out_r1_both,
            json!({
                "orthogonalizeAgainstRankOnePath": r1_toy,
                "orthogonalizeSide": "both"
            }),
            &dims,
        )?;
        assert_eq!(res_r1["orthogonalizedSiteCount"], 1);
        let loaded_r1 = read_lora_gguf_pairs_with_dims(&out_r1_both, &dims)?;
        // Col 0 of A and row 1 of B should be zeroed.
        for k in 0..2 {
            assert!(loaded_r1.pairs[0].lora_a_f32[k * 8].abs() < 1e-6);
            assert!(loaded_r1.pairs[0].lora_b_f32[2 + k].abs() < 1e-6);
        }

        // 2. Multi-rank reference adapter with orthogonal columns in B of norms 2.0 (sigma^2=4.0) and 0.5 (sigma^2=0.25).
        let ref_multi = scratch.join("ref-multi.gguf");
        write_lora_gguf(
            &ref_multi,
            ExportDtype::F32,
            2.0,
            &[],
            &[ExportTensorPair {
                gguf_base_name: "blk.0.ffn_down.weight".to_owned(),
                in_features: 8,
                out_features: 6,
                rank: 2,
                lora_a_f32: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                lora_b_f32: vec![
                    2.0, 0.0, // row 0: norm 2.0 on e_0
                    0.0, 0.5, // row 1: norm 0.5 on e_1
                    0.0, 0.0, // row 2: untouched e_2
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            }],
        )?;

        // 2a. Hard projection (alphaEditNullSpaceRidge = None) zeros both e_0 and e_1 in B.
        let out_hard = scratch.join("out-hard.gguf");
        let res_hard = run_toy_edit(
            &src,
            &out_hard,
            json!({
                "orthogonalizeAgainstAdapterPath": ref_multi,
                "orthogonalizeAdapterSide": "b"
            }),
            &dims,
        )?;
        assert_eq!(res_hard["multiRankOrthogonalizedSiteCount"], 1);
        let loaded_hard = read_lora_gguf_pairs_with_dims(&out_hard, &dims)?;
        for k in 0..2 {
            assert!(loaded_hard.pairs[0].lora_b_f32[k].abs() < 1e-6);
            assert!(loaded_hard.pairs[0].lora_b_f32[2 + k].abs() < 1e-6);
        }
        assert!((loaded_hard.pairs[0].lora_b_f32[4] - 0.5).abs() < 1e-6);
        assert!((loaded_hard.pairs[0].lora_b_f32[5] - 2.0).abs() < 1e-6);

        // 2b. Soft AlphaEdit projection with lambda = 1.0:
        // e_0 attenuates by (1 - 4/(4+1)) = 0.2; e_1 attenuates by (1 - 0.25/(0.25+1)) = 0.8; e_2 unchanged.
        let out_soft = scratch.join("out-soft.gguf");
        let res_soft = run_toy_edit(
            &src,
            &out_soft,
            json!({
                "orthogonalizeAgainstAdapterPath": ref_multi,
                "orthogonalizeAdapterSide": "b",
                "alphaEditNullSpaceRidge": 1.0
            }),
            &dims,
        )?;
        assert_eq!(res_soft["multiRankOrthogonalizedSiteCount"], 1);
        let loaded_soft = read_lora_gguf_pairs_with_dims(&out_soft, &dims)?;
        let b_soft = &loaded_soft.pairs[0].lora_b_f32;
        assert!((b_soft[0] - 0.2).abs() < 1e-5);
        assert!(b_soft[1].abs() < 1e-6);
        assert!((b_soft[2] - 0.8).abs() < 1e-5);
        assert!((b_soft[3] - 0.8).abs() < 1e-5);
        assert!((b_soft[4] - 0.5).abs() < 1e-5);
        assert!((b_soft[5] - 2.0).abs() < 1e-5);

        // 2c. Multi-rank hard projection on Both sides (`a` and `b`) clears ref_multi A rows too.
        let out_both = scratch.join("out-multi-both.gguf");
        let _ = run_toy_edit(
            &src,
            &out_both,
            json!({
                "orthogonalizeAgainstAdapterPath": ref_multi,
                "orthogonalizeAdapterSide": "both"
            }),
            &dims,
        )?;
        let loaded_both = read_lora_gguf_pairs_with_dims(&out_both, &dims)?;
        for k in 0..2 {
            assert!(loaded_both.pairs[0].lora_a_f32[k * 8].abs() < 1e-6);
            assert!(loaded_both.pairs[0].lora_a_f32[k * 8 + 1].abs() < 1e-6);
            assert!(loaded_both.pairs[0].lora_b_f32[k].abs() < 1e-6);
            assert!(loaded_both.pairs[0].lora_b_f32[2 + k].abs() < 1e-6);
        }

        // Validation errors for ridge.
        for bad_cfg in [
            json!({ "alphaEditNullSpaceRidge": 0.5 }),
            json!({
                "orthogonalizeAgainstAdapterPath": ref_multi,
                "alphaEditNullSpaceRidge": -0.1
            }),
        ] {
            assert!(run_toy_edit(&src, &scratch.join("bad-ridge.gguf"), bad_cfg, &dims).is_err());
        }
        Ok(())
    }

    #[test]
    fn edit_adapter_cans_scaling_depth_curves_and_ls_lora_selection() -> crate::Result<()> {
        let scratch = ScratchDir::new("edit-cans-depth-ls")?;
        let dims = toy_dims_three();
        let src = scratch.join("src-multi.gguf");
        let make_ones = |name: &str| ExportTensorPair {
            gguf_base_name: name.to_owned(),
            in_features: 8,
            out_features: 6,
            rank: 1,
            lora_a_f32: vec![1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            lora_b_f32: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        write_lora_gguf(
            &src,
            ExportDtype::F32,
            1.0,
            &[],
            &[
                make_ones("blk.0.ffn_down.weight"),
                make_ones("blk.21.ffn_down.weight"),
                make_ones("blk.63.ffn_down.weight"),
            ],
        )?;

        // 1. CANS scaling + Power depth curve (exponent = 2.0).
        let out_cans = scratch.join("out-cans-power.gguf");
        let res_cans = run_toy_edit(
            &src,
            &out_cans,
            json!({
                "depthGainStart": 0.2,
                "depthGainEnd": 1.8,
                "depthGainCurve": "power",
                "depthGainExponent": 2.0,
                "cansNeuronScales": {
                    "blk.21.ffn_down.weight": [3.0, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                }
            }),
            &dims,
        )?;
        assert_eq!(res_cans["cansScaledSiteCount"], 1);
        let loaded_cans = read_lora_gguf_pairs_with_dims(&out_cans, &dims)?;
        let d0 = reconstruct_delta(&loaded_cans.pairs[0], loaded_cans.alpha);
        let d21 = reconstruct_delta(&loaded_cans.pairs[1], loaded_cans.alpha);
        let d63 = reconstruct_delta(&loaded_cans.pairs[2], loaded_cans.alpha);
        let g21 = 0.2 + 1.6 * (21.0_f64 / 63.0).powf(2.0);
        assert!((d0[0] - 0.2).abs() < 1e-5);
        assert!((d63[0] - 1.8).abs() < 1e-5);
        // CANS scales col 0 by 3.0, col 1 by 0.5 (base 2.0 -> 1.0), col 2 by 0.0.
        assert!((d21[0] - g21 * 3.0).abs() < 1e-5);
        assert!((d21[1] - g21 * 1.0).abs() < 1e-5);
        assert!(d21[2].abs() < 1e-6);

        // Cosine curve checks midpoint interpolation against analytical formula.
        let out_cos = scratch.join("out-cosine.gguf");
        let _ = run_toy_edit(
            &src,
            &out_cos,
            json!({
                "depthGainStart": 0.5,
                "depthGainEnd": 1.5,
                "depthGainCurve": "cosine"
            }),
            &dims,
        )?;
        let loaded_cos = read_lora_gguf_pairs_with_dims(&out_cos, &dims)?;
        let d21_cos = reconstruct_delta(&loaded_cos.pairs[1], loaded_cos.alpha);
        let t_cos = 0.5 * (1.0 - (std::f64::consts::PI * 21.0 / 63.0).cos());
        assert!((d21_cos[0] - (0.5 + 1.0 * t_cos)).abs() < 1e-5);

        // 2. LS-LoRA reference cosine + minSiteFrobeniusNorm:
        // blk.0 aligned (+1.0), blk.21 zero-norm ref (dropped), blk.63 absent from ref (dropped).
        let ls_ref = scratch.join("ls-ref.gguf");
        write_lora_gguf(
            &ls_ref,
            ExportDtype::F32,
            1.0,
            &[],
            &[
                make_ones("blk.0.ffn_down.weight"),
                ExportTensorPair {
                    gguf_base_name: "blk.21.ffn_down.weight".to_owned(),
                    in_features: 8,
                    out_features: 6,
                    rank: 1,
                    lora_a_f32: vec![0.0; 8],
                    lora_b_f32: vec![0.0; 6],
                },
            ],
        )?;
        let out_ls = scratch.join("out-ls.gguf");
        let res_ls = run_toy_edit(
            &src,
            &out_ls,
            json!({
                "lsLoraReferenceAdapterPath": ls_ref,
                "lsLoraMinCosine": 0.5
            }),
            &dims,
        )?;
        assert_eq!(res_ls["pairCount"], 1);
        assert_eq!(res_ls["droppedSiteCount"], 2);
        assert_eq!(res_ls["lsLoraDroppedCount"], 2);

        // Orthogonal non-zero ref site has cosine 0.0 < 0.5 (dropped), anti-aligned (-1.0) dropped at -0.5.
        let ls_ref_neg = scratch.join("ls-ref-neg.gguf");
        write_lora_gguf(
            &ls_ref_neg,
            ExportDtype::F32,
            1.0,
            &[],
            &[
                ExportTensorPair {
                    gguf_base_name: "blk.0.ffn_down.weight".to_owned(),
                    in_features: 8,
                    out_features: 6,
                    rank: 1,
                    lora_a_f32: vec![-1.0, -2.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                    lora_b_f32: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                ExportTensorPair {
                    gguf_base_name: "blk.21.ffn_down.weight".to_owned(),
                    in_features: 8,
                    out_features: 6,
                    rank: 1,
                    lora_a_f32: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    lora_b_f32: vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                },
                make_ones("blk.63.ffn_down.weight"),
            ],
        )?;
        let out_ls_neg = scratch.join("out-ls-neg.gguf");
        let res_ls_neg = run_toy_edit(
            &src,
            &out_ls_neg,
            json!({
                "lsLoraReferenceAdapterPath": ls_ref_neg,
                "lsLoraMinCosine": 0.5
            }),
            &dims,
        )?;
        assert_eq!(res_ls_neg["pairCount"], 1);
        assert_eq!(res_ls_neg["lsLoraDroppedCount"], 2);

        // minSiteFrobeniusNorm drops blk.0 (norm ~ 0.1 * sqrt(5.25) < 1.0) while keeping blk.63 (norm ~ 2.0 * sqrt(5.25)).
        let out_min_norm = scratch.join("out-min-norm.gguf");
        let res_min_norm = run_toy_edit(
            &src,
            &out_min_norm,
            json!({
                "depthGainStart": 0.1,
                "depthGainEnd": 2.0,
                "minSiteFrobeniusNorm": 1.0
            }),
            &dims,
        )?;
        assert_eq!(res_min_norm["lsLoraDroppedCount"], 1);
        assert_eq!(res_min_norm["pairCount"], 2);

        // Validation rejections for CANS, depth curves, and LS-LoRA.
        for bad_cfg in [
            json!({ "depthGainCurve": "power" }),
            json!({ "depthGainCurve": "power", "depthGainExponent": 0.1 }),
            json!({ "depthGainCurve": "power", "depthGainExponent": 9.0 }),
            json!({ "depthGainCurve": "linear", "depthGainExponent": 2.0 }),
            json!({ "depthGainCurve": "cosine", "depthGainExponent": 2.0 }),
            json!({ "cansNeuronScales": { "": vec![1.0_f64; 8] } }),
            json!({ "cansNeuronScales": { "blk.0.ffn_down.weight": vec![1.0_f64; 4] } }),
            json!({ "cansNeuronScales": { "blk.99.ffn_down.weight": vec![1.0_f64; 8] } }),
            json!({ "cansNeuronScales": { "blk.0.ffn_down.weight": [-0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] } }),
            json!({ "minSiteFrobeniusNorm": 0.0 }),
            json!({ "minSiteFrobeniusNorm": 2.0, "maxSiteFrobeniusNorm": 1.0 }),
            json!({ "lsLoraReferenceAdapterPath": ls_ref }),
            json!({ "lsLoraMinCosine": 0.2 }),
            json!({ "lsLoraReferenceAdapterPath": ls_ref, "lsLoraMinCosine": 1.2 }),
        ] {
            assert!(run_toy_edit(&src, &scratch.join("bad-ctrl.gguf"), bad_cfg, &dims).is_err());
        }
        Ok(())
    }
}
