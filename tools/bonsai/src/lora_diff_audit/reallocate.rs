//! Equal-byte KL-gap and Softmax-Fisher variable-rank LoRA reallocation (`dr-w1` / `reallocate-ranks`).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::calibration_metrics::{
    KnapsackOption, per_site_fisher_weighted_kl_metrics, solve_multiple_choice_knapsack_maximize,
};

use super::gguf_adapter::{
    ExportDtype, ExportTensorPair, read_lora_gguf_pairs_with_dims, sort_pairs_canonically,
    swift_base_dims_map, write_lora_gguf,
};
use super::transform::{check_factor_value, target_layer_and_family_map, verify_published_adapter};
use super::{Operation, Request, usize_to_f32};

pub(super) const DEFAULT_MAX_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_MAX_RANK: usize = 32;
const DEFAULT_CANDIDATE_RANKS: [usize; 9] = [0, 1, 2, 4, 8, 12, 16, 24, 32];

#[expect(
    clippy::as_conversions,
    reason = "LoRA rank fits cleanly inside IEEE-754 f64 significand bounds."
)]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Per-site diagnostic sensitivity measurement for variable-rank allocation.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct SiteSensitivityInput {
    pub(super) target: String,
    pub(super) reasoning_kl_gap: Option<f64>,
    pub(super) thinking_off_kl_penalty: Option<f64>,
    pub(super) softmax_fisher_weight: Option<f64>,
    pub(super) teacher_reasoning_logits: Option<Vec<Vec<f64>>>,
    pub(super) base_reasoning_logits: Option<Vec<Vec<f64>>>,
    pub(super) recovered_reasoning_logits: Option<Vec<Vec<f64>>>,
    pub(super) teacher_thinking_off_logits: Option<Vec<Vec<f64>>>,
    pub(super) base_thinking_off_logits: Option<Vec<Vec<f64>>>,
    pub(super) recovered_thinking_off_logits: Option<Vec<Vec<f64>>>,
}

/// Request configuration for `reallocate-ranks`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct ReallocateRanksConfig {
    pub(super) max_payload_bytes: Option<u64>,
    pub(super) min_rank: Option<usize>,
    pub(super) max_rank: Option<usize>,
    pub(super) candidate_ranks: Option<Vec<usize>>,
    pub(super) export_dtype: Option<ExportDtype>,
    pub(super) diagnostic_prompt_sha256s: Vec<String>,
    pub(super) thinking_off_penalty_weight: Option<f64>,
    pub(super) site_sensitivities: Vec<SiteSensitivityInput>,
}

struct EvaluatedSensitivity {
    reasoning_kl_gap: f64,
    thinking_off_penalty: f64,
    fisher_weight: f64,
    weighted_utility: f64,
}

pub(super) fn validate_reallocate_config(config: &ReallocateRanksConfig) -> crate::Result<()> {
    let max_payload = config
        .max_payload_bytes
        .unwrap_or(DEFAULT_MAX_PAYLOAD_BYTES);
    if max_payload == 0 || max_payload > DEFAULT_MAX_PAYLOAD_BYTES {
        return Err(format!(
            "reallocateConfig.maxPayloadBytes must be in 1..={DEFAULT_MAX_PAYLOAD_BYTES}, got {max_payload}."
        ));
    }
    let min_rank = config.min_rank.unwrap_or(0);
    let max_rank = config.max_rank.unwrap_or(DEFAULT_MAX_RANK);
    if max_rank == 0 || max_rank > DEFAULT_MAX_RANK {
        return Err(format!(
            "reallocateConfig.maxRank must lie in 1..={DEFAULT_MAX_RANK}, got {max_rank}."
        ));
    }
    if min_rank > max_rank {
        return Err(format!(
            "reallocateConfig.minRank ({min_rank}) cannot exceed maxRank ({max_rank})."
        ));
    }
    let candidates = normalized_candidates(config, min_rank, max_rank)?;
    if candidates.is_empty() {
        return Err("reallocateConfig.candidateRanks must produce at least one rank.".into());
    }
    let penalty_weight = config.thinking_off_penalty_weight.unwrap_or(1.0);
    if !penalty_weight.is_finite() || penalty_weight < 0.0 {
        return Err(
            "reallocateConfig.thinkingOffPenaltyWeight must be finite and non-negative.".into(),
        );
    }
    if config.diagnostic_prompt_sha256s.is_empty() {
        return Err(
            "reallocateConfig.diagnosticPromptSha256s must be non-empty so split disjointness can be audited."
                .into(),
        );
    }
    for digest in &config.diagnostic_prompt_sha256s {
        if !crate::identity::is_lower_hex(digest, 64) {
            return Err(
                "reallocateConfig.diagnosticPromptSha256s must contain only lowercase 64-hex SHA-256 digests."
                    .into(),
            );
        }
    }
    if config.site_sensitivities.is_empty() {
        return Err("reallocateConfig.siteSensitivities must contain at least one site.".into());
    }
    let mut seen_targets = BTreeSet::new();
    for entry in &config.site_sensitivities {
        if entry.target.trim().is_empty() {
            return Err("reallocateConfig.siteSensitivities contains an empty target.".into());
        }
        if !seen_targets.insert(entry.target.as_str()) {
            return Err(format!(
                "Duplicate target '{}' in reallocateConfig.siteSensitivities.",
                entry.target
            ));
        }
        let _ = evaluate_site_sensitivity(entry, penalty_weight)?;
    }
    Ok(())
}

fn normalized_candidates(
    config: &ReallocateRanksConfig,
    min_rank: usize,
    max_rank: usize,
) -> crate::Result<Vec<usize>> {
    let mut ranks = match &config.candidate_ranks {
        Some(list) => list.clone(),
        None => DEFAULT_CANDIDATE_RANKS
            .into_iter()
            .filter(|&r| r >= min_rank && r <= max_rank)
            .collect(),
    };
    if ranks.is_empty() {
        return Err("reallocateConfig.candidateRanks must not be empty.".into());
    }
    ranks.sort_unstable();
    ranks.dedup();
    for &r in &ranks {
        if r < min_rank || r > max_rank {
            return Err(format!(
                "reallocateConfig.candidateRanks entry {r} is outside [{min_rank}, {max_rank}]."
            ));
        }
    }
    Ok(ranks)
}

fn evaluate_site_sensitivity(
    entry: &SiteSensitivityInput,
    penalty_weight: f64,
) -> crate::Result<EvaluatedSensitivity> {
    let fisher_weight = entry.softmax_fisher_weight.unwrap_or(1.0);
    if !fisher_weight.is_finite() || fisher_weight < 0.0 {
        return Err(format!(
            "Site '{}' softmaxFisherWeight must be finite and non-negative.",
            entry.target
        ));
    }

    let raw_reasoning_present = entry.teacher_reasoning_logits.is_some()
        || entry.base_reasoning_logits.is_some()
        || entry.recovered_reasoning_logits.is_some();
    let reasoning_kl_gap = if raw_reasoning_present {
        let teacher = entry.teacher_reasoning_logits.as_ref().ok_or_else(|| {
            format!(
                "Site '{}' requires teacherReasoningLogits when raw reasoning logits are supplied.",
                entry.target
            )
        })?;
        let base = entry.base_reasoning_logits.as_ref().ok_or_else(|| {
            format!(
                "Site '{}' requires baseReasoningLogits when raw reasoning logits are supplied.",
                entry.target
            )
        })?;
        let recovered = entry.recovered_reasoning_logits.as_ref().ok_or_else(|| {
            format!(
                "Site '{}' requires recoveredReasoningLogits when raw reasoning logits are supplied.",
                entry.target
            )
        })?;
        let (_, _, gap) = per_site_fisher_weighted_kl_metrics(teacher, base, recovered)?;
        gap
    } else {
        let gap = entry.reasoning_kl_gap.ok_or_else(|| {
            format!(
                "Site '{}' requires reasoningKlGap or raw reasoning logits tuple.",
                entry.target
            )
        })?;
        if !gap.is_finite() {
            return Err(format!(
                "Site '{}' reasoningKlGap must be finite.",
                entry.target
            ));
        }
        gap
    };

    let raw_off_present = entry.teacher_thinking_off_logits.is_some()
        || entry.base_thinking_off_logits.is_some()
        || entry.recovered_thinking_off_logits.is_some();
    let thinking_off_penalty = if raw_off_present {
        let teacher = entry.teacher_thinking_off_logits.as_ref().ok_or_else(|| {
            format!(
                "Site '{}' requires teacherThinkingOffLogits when raw Thinking-OFF logits are supplied.",
                entry.target
            )
        })?;
        let base = entry.base_thinking_off_logits.as_ref().ok_or_else(|| {
            format!(
                "Site '{}' requires baseThinkingOffLogits when raw Thinking-OFF logits are supplied.",
                entry.target
            )
        })?;
        let recovered = entry.recovered_thinking_off_logits.as_ref().ok_or_else(|| {
            format!(
                "Site '{}' requires recoveredThinkingOffLogits when raw Thinking-OFF logits are supplied.",
                entry.target
            )
        })?;
        let (_, _, gap_off) = per_site_fisher_weighted_kl_metrics(teacher, base, recovered)?;
        (-gap_off).max(0.0)
    } else if let Some(penalty) = entry.thinking_off_kl_penalty {
        if !penalty.is_finite() || penalty < 0.0 {
            return Err(format!(
                "Site '{}' thinkingOffKlPenalty must be finite and non-negative.",
                entry.target
            ));
        }
        penalty
    } else {
        0.0
    };

    let weighted_utility = fisher_weight * reasoning_kl_gap - penalty_weight * thinking_off_penalty;
    Ok(EvaluatedSensitivity {
        reasoning_kl_gap,
        thinking_off_penalty,
        fisher_weight,
        weighted_utility,
    })
}

fn ranked_mode_energy_profile(pair: &ExportTensorPair) -> crate::Result<(Vec<usize>, Vec<f64>)> {
    let mut modes: Vec<(usize, f64)> = Vec::with_capacity(pair.rank);
    let mut total_energy = 0.0_f64;
    for k in 0..pair.rank {
        let mut a_sq = 0.0_f64;
        let a_off = k * pair.in_features;
        for col in 0..pair.in_features {
            let v = f64::from(pair.lora_a_f32[a_off + col]);
            a_sq += v * v;
        }
        let mut b_sq = 0.0_f64;
        for row in 0..pair.out_features {
            let v = f64::from(pair.lora_b_f32[row * pair.rank + k]);
            b_sq += v * v;
        }
        let energy = a_sq * b_sq;
        if !energy.is_finite() {
            return Err(format!(
                "Non-finite mode Frobenius energy on '{}' mode {k}.",
                pair.gguf_base_name
            ));
        }
        total_energy += energy;
        modes.push((k, energy));
    }
    modes.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let sorted_indices: Vec<usize> = modes.iter().map(|&(idx, _)| idx).collect();
    let mut cum_fractions = Vec::with_capacity(pair.rank + 1);
    cum_fractions.push(0.0_f64);
    let mut running = 0.0_f64;
    for (pos, &(_, energy)) in modes.iter().enumerate() {
        running += energy;
        let frac = if total_energy > 0.0 {
            (running / total_energy).clamp(0.0, 1.0)
        } else {
            usize_to_f64(pos + 1) / usize_to_f64(pair.rank)
        };
        cum_fractions.push(frac);
    }
    Ok((sorted_indices, cum_fractions))
}

/// Run `reallocate-ranks` against the built-in Swift geometry contract.
pub(super) fn run_reallocate_ranks(request: &Request) -> crate::Result<Value> {
    let base_dims = swift_base_dims_map();
    run_reallocate_ranks_with_dims(request, &base_dims)
}

/// Run `reallocate-ranks` with explicit tensor dimensions (production and test entry point).
#[expect(
    clippy::too_many_lines,
    reason = "Keep knapsack setup, GGUF export, verification, and split-audit metadata together."
)]
pub(super) fn run_reallocate_ranks_with_dims(
    request: &Request,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<Value> {
    if request.operation != Operation::ReallocateRanks {
        return Err("run_reallocate_ranks requires reallocate-ranks operation.".into());
    }
    let adapter_path = request
        .adapter_path
        .as_ref()
        .ok_or("reallocate-ranks requires adapterPath.")?;
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("reallocate-ranks requires outputPath.")?;
    let config = request
        .reallocate_config
        .as_ref()
        .ok_or("reallocate-ranks requires reallocateConfig.")?;
    validate_reallocate_config(config)?;

    let loaded = read_lora_gguf_pairs_with_dims(adapter_path, base_dims)?;
    let layer_map = target_layer_and_family_map();
    let available_targets: BTreeSet<&str> = loaded
        .pairs
        .iter()
        .map(|p| p.gguf_base_name.as_str())
        .collect();
    let penalty_weight = config.thinking_off_penalty_weight.unwrap_or(1.0);
    let mut sensitivity_by_target = BTreeMap::new();
    for entry in &config.site_sensitivities {
        if !available_targets.contains(entry.target.as_str()) {
            return Err(format!(
                "reallocateConfig.siteSensitivities target '{}' is missing from source adapter.",
                entry.target
            ));
        }
        let evaluated = evaluate_site_sensitivity(entry, penalty_weight)?;
        sensitivity_by_target.insert(entry.target.as_str(), evaluated);
    }

    let max_payload_bytes = config
        .max_payload_bytes
        .unwrap_or(DEFAULT_MAX_PAYLOAD_BYTES);
    let min_rank = config.min_rank.unwrap_or(0);
    let max_rank = config.max_rank.unwrap_or(DEFAULT_MAX_RANK);
    let candidates = normalized_candidates(config, min_rank, max_rank)?;
    let out_dtype = config.export_dtype.unwrap_or(ExportDtype::F16);
    let elem_bytes = u64::try_from(out_dtype.byte_width()).map_err(|e| e.to_string())?;

    let mut groups: Vec<Vec<KnapsackOption>> = Vec::with_capacity(loaded.pairs.len());
    let mut site_option_ranks: Vec<Vec<usize>> = Vec::with_capacity(loaded.pairs.len());
    let mut site_option_energy_fracs: Vec<Vec<f64>> = Vec::with_capacity(loaded.pairs.len());
    let mut site_modes: Vec<Vec<usize>> = Vec::with_capacity(loaded.pairs.len());

    for pair in &loaded.pairs {
        let (sorted_modes, cum_fracs) = ranked_mode_energy_profile(pair)?;
        let utility = sensitivity_by_target
            .get(pair.gguf_base_name.as_str())
            .map_or(0.0, |s| s.weighted_utility);
        let elems_per_rank = u64::try_from(pair.in_features + pair.out_features)
            .map_err(|e| e.to_string())?
            .checked_mul(elem_bytes)
            .ok_or("Per-rank site byte size overflow.")?;

        let mut group_opts = Vec::new();
        let mut group_ranks = Vec::new();
        let mut group_fracs = Vec::new();
        for &r in &candidates {
            if r > pair.rank {
                continue;
            }
            let cost = if r == 0 {
                0
            } else {
                elems_per_rank
                    .checked_mul(u64::try_from(r).map_err(|e| e.to_string())?)
                    .ok_or("Candidate factor byte size overflow.")?
            };
            let frac = cum_fracs[r];
            let value = if r == 0 { 0.0 } else { utility * frac };
            group_opts.push(KnapsackOption { cost, value });
            group_ranks.push(r);
            group_fracs.push(frac);
        }
        if group_opts.is_empty() {
            return Err(format!(
                "No feasible candidate rank in {:?} fits loaded rank {} on '{}'.",
                candidates, pair.rank, pair.gguf_base_name
            ));
        }
        groups.push(group_opts);
        site_option_ranks.push(group_ranks);
        site_option_energy_fracs.push(group_fracs);
        site_modes.push(sorted_modes);
    }

    let chosen_indices = solve_multiple_choice_knapsack_maximize(&groups, max_payload_bytes)?;
    struct PendingSite {
        gguf_base_name: String,
        in_features: usize,
        out_features: usize,
        source_rank: usize,
        rank: usize,
        a_f64: Vec<f64>,
        b_f64: Vec<f64>,
    }

    let mut pending = Vec::new();
    let mut per_site_ranks = BTreeMap::new();
    let mut allocations_json = Vec::with_capacity(loaded.pairs.len());
    let mut families_used = BTreeSet::new();
    let mut total_factor_bytes = 0_u64;
    let mut zeroed_sites = 0_usize;
    let mut max_allocated_rank = 0_usize;
    let mut total_utility = 0.0_f64;

    for (idx, pair) in loaded.pairs.iter().enumerate() {
        let choice = chosen_indices[idx];
        let alloc_rank = site_option_ranks[idx][choice];
        let option = groups[idx][choice];
        let frac = site_option_energy_fracs[idx][choice];
        per_site_ranks.insert(pair.gguf_base_name.clone(), alloc_rank);
        total_factor_bytes = total_factor_bytes
            .checked_add(option.cost)
            .ok_or("Total reallocated factor byte count overflow.")?;
        total_utility += option.value;

        let sensitivity = sensitivity_by_target.get(pair.gguf_base_name.as_str());
        allocations_json.push(json!({
            "target": pair.gguf_base_name,
            "sourceRank": pair.rank,
            "allocatedRank": alloc_rank,
            "factorPayloadBytes": option.cost,
            "reasoningKlGap": sensitivity.map_or(0.0, |s| s.reasoning_kl_gap),
            "thinkingOffPenalty": sensitivity.map_or(0.0, |s| s.thinking_off_penalty),
            "softmaxFisherWeight": sensitivity.map_or(1.0, |s| s.fisher_weight),
            "netUtility": option.value,
            "retainedEnergyFraction": frac,
        }));

        if alloc_rank == 0 {
            zeroed_sites += 1;
            continue;
        }
        let &(_, inv_family) = layer_map
            .get(&pair.gguf_base_name)
            .ok_or_else(|| format!("Unknown Swift layer for '{}'.", pair.gguf_base_name))?;
        let family = base_dims
            .get(&pair.gguf_base_name)
            .map_or(inv_family, |&(_, _, f)| f);
        families_used.insert(family);
        max_allocated_rank = max_allocated_rank.max(alloc_rank);

        let s1 = (f64::from(loaded.alpha) / usize_to_f64(pair.rank)).sqrt();
        let selected = &site_modes[idx][..alloc_rank];
        let mut a_f64 = Vec::with_capacity(alloc_rank * pair.in_features);
        for &src_k in selected {
            let row_off = src_k * pair.in_features;
            for col in 0..pair.in_features {
                a_f64.push(f64::from(pair.lora_a_f32[row_off + col]) * s1);
            }
        }
        let mut b_f64 = Vec::with_capacity(pair.out_features * alloc_rank);
        for row in 0..pair.out_features {
            let row_off = row * pair.rank;
            for &src_k in selected {
                b_f64.push(f64::from(pair.lora_b_f32[row_off + src_k]) * s1);
            }
        }
        pending.push(PendingSite {
            gguf_base_name: pair.gguf_base_name.clone(),
            in_features: pair.in_features,
            out_features: pair.out_features,
            source_rank: pair.rank,
            rank: alloc_rank,
            a_f64,
            b_f64,
        });
    }

    if pending.is_empty() {
        return Err(
            "reallocate-ranks zeroed all target sites (increase reasoning sensitivity or budget)."
                .into(),
        );
    }

    let new_alpha = usize_to_f32(max_allocated_rank);
    let mut final_pairs = Vec::with_capacity(pending.len());
    for item in pending {
        let _ = item.source_rank;
        let s2 = (usize_to_f64(item.rank) / f64::from(new_alpha)).sqrt();
        let mut lora_a_f32 = Vec::with_capacity(item.a_f64.len());
        for v in item.a_f64 {
            lora_a_f32.push(check_factor_value(v * s2, out_dtype, "reallocated lora_a")?);
        }
        let mut lora_b_f32 = Vec::with_capacity(item.b_f64.len());
        for v in item.b_f64 {
            lora_b_f32.push(check_factor_value(v * s2, out_dtype, "reallocated lora_b")?);
        }
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

    let canonical_digests: BTreeSet<String> =
        config.diagnostic_prompt_sha256s.iter().cloned().collect();
    let training_prompt_sha256s: Vec<String> = canonical_digests.into_iter().collect();

    let mut meta: Vec<(String, String)> = loaded
        .provenance_meta
        .into_iter()
        .filter(|(k, _)| {
            k != "bonsai.composition.reallocate_parent_sha256"
                && k != "bonsai.composition.reallocate_method"
        })
        .collect();
    meta.push((
        "bonsai.composition.reallocate_parent_sha256".to_owned(),
        loaded.sha256.clone(),
    ));
    meta.push((
        "bonsai.composition.reallocate_method".to_owned(),
        "kl-gap-fisher-knapsack-v1".to_owned(),
    ));
    let meta_refs: Vec<(&str, &str)> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

    let written = write_lora_gguf(output_path, out_dtype, new_alpha, &meta_refs, &final_pairs)?;
    let verification = verify_published_adapter(
        output_path,
        &written.sha256,
        written.bytes,
        max_allocated_rank,
        final_pairs.len(),
        base_dims,
    )?;

    Ok(json!({
        "schemaVersion": 1,
        "operation": "reallocate-ranks",
        "published": true,
        "sourceAdapterPath": adapter_path,
        "sourceAdapterSha256": loaded.sha256,
        "outputPath": output_path,
        "sha256": written.sha256,
        "ggufFileBytes": written.bytes,
        "factorPayloadBytes": total_factor_bytes,
        "maxPayloadBytes": max_payload_bytes,
        "maxRank": max_allocated_rank,
        "pairCount": final_pairs.len(),
        "zeroedSiteCount": zeroed_sites,
        "alpha": new_alpha,
        "dtype": out_dtype,
        "totalEstimatedUtility": total_utility,
        "perSiteRanks": per_site_ranks,
        "siteAllocations": allocations_json,
        "trainingPromptSha256s": training_prompt_sha256s,
        "verification": verification,
    }))
}
