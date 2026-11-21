//! Offline multi-rank GGUF LoRA chunk merging (`merge-adapters`) and pure-Rust
//! algebraic direct weight editing (`edit-adapter`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::gguf_adapter::{
    ExportDtype, ExportTensorPair, VerifyPolicy, classify_families, read_lora_gguf_pairs_with_dims,
    read_rank_one_b_directions, sort_pairs_canonically, swift_base_dims_map,
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

/// Algebraic multi-rank factor edit specification for `edit-adapter`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct EditAdapterConfig {
    pub(super) target_max_rank: Option<usize>,
    pub(super) global_gain: Option<f64>,
    pub(super) family_scales: BTreeMap<String, f64>,
    pub(super) layer_min: Option<u32>,
    pub(super) layer_max: Option<u32>,
    pub(super) depth_gain_start: Option<f64>,
    pub(super) depth_gain_end: Option<f64>,
    pub(super) max_site_frobenius_norm: Option<f64>,
    pub(super) orthogonalize_against_rank_one_path: Option<PathBuf>,
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

pub(super) fn validate_edit_config(config: &EditAdapterConfig) -> crate::Result<()> {
    if let Some(rank) = config.target_max_rank
        && !(1..=64).contains(&rank)
    {
        return Err("editConfig.targetMaxRank must be in 1..=64.".into());
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
    if let Some(cap) = config.max_site_frobenius_norm
        && (!cap.is_finite() || cap <= 0.0)
    {
        return Err("editConfig.maxSiteFrobeniusNorm must be finite and positive.".into());
    }
    if let Some(ref_path) = &config.orthogonalize_against_rank_one_path {
        bonsai_artifacts::validate_external_path(ref_path).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn verify_published_adapter(
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

fn check_factor_value(val: f64, dtype: ExportDtype, label: &str) -> crate::Result<f32> {
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

fn target_layer_and_family_map() -> BTreeMap<String, (u32, &'static str)> {
    let mut map = BTreeMap::new();
    for spec in swift_tensor_inventory() {
        if let Some(layer) = spec.layer {
            map.insert(spec.gguf_name, (layer, spec.family));
        }
    }
    map
}

fn effective_delta_frobenius_norm_f64(
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

/// Applies pure-Rust algebraic direct weight edits to an existing multi-rank GGUF LoRA.
pub(super) fn run_edit_adapter(request: &Request) -> crate::Result<Value> {
    run_edit_adapter_with_dims(request, &swift_base_dims_map())
}

#[expect(
    clippy::too_many_lines,
    reason = "Applies rank slicing, layer/family/depth gains, Orca orthogonalization, norm capping, and atomic verification."
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

    let loaded = read_lora_gguf_pairs_with_dims(adapter_path, base_dims)?;
    let rank_one_dirs = match &config.orthogonalize_against_rank_one_path {
        Some(path) => Some(read_rank_one_b_directions(path)?),
        None => None,
    };
    let out_dtype = config.export_dtype.unwrap_or(loaded.dtype);
    let layer_map = target_layer_and_family_map();

    let mut pending = Vec::with_capacity(loaded.pairs.len());
    let mut dropped_sites = 0_usize;
    let mut orthogonalized_sites = 0_usize;
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
            (Some(start), Some(end)) => start + (end - start) * (f64::from(layer) / 63.0),
            _ => 1.0,
        };
        let site_gain = global_gain * fam_gain * depth_gain;
        if site_gain == 0.0 {
            dropped_sites += 1;
            continue;
        }

        let new_rank = pair.rank.min(config.target_max_rank.unwrap_or(pair.rank));
        if new_rank == 0 {
            dropped_sites += 1;
            continue;
        }

        // Rescale sliced modes so (1.0) * B_tmp * A_tmp equals site_gain * sum_{k < new_rank} delta_k.
        let s1 = (site_gain * f64::from(loaded.alpha) / usize_to_f64(pair.rank)).sqrt();
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
        for k in 0..new_rank {
            let row_off = k * pair.in_features;
            for col in 0..pair.in_features {
                a_f64.push(f64::from(pair.lora_a_f32[row_off + col]) * s1);
            }
        }

        let mut b_f64 = Vec::new();
        b_f64
            .try_reserve_exact(b_len)
            .map_err(|e| format!("Cannot allocate edited lora_b buffer: {e}"))?;
        for row in 0..pair.out_features {
            let row_off = row * pair.rank;
            for k in 0..new_rank {
                b_f64.push(f64::from(pair.lora_b_f32[row_off + k]) * s1);
            }
        }

        if let Some(ref_dirs) = &rank_one_dirs
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
                for row in 0..pair.out_features {
                    b_f64[row * new_rank + k] -= dot * dir[row];
                }
            }
            orthogonalized_sites += 1;
        }

        if let Some(cap) = config.max_site_frobenius_norm {
            let frob = effective_delta_frobenius_norm_f64(
                &a_f64,
                &b_f64,
                pair.in_features,
                pair.out_features,
                new_rank,
            )?;
            if frob > cap {
                let damp = (cap / frob).sqrt();
                for v in &mut a_f64 {
                    *v *= damp;
                }
                for v in &mut b_f64 {
                    *v *= damp;
                }
                capped_sites += 1;
            }
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
        "cappedSiteCount": capped_sites,
        "verification": verification,
    }))
}
