//! U2 bounded one-site-at-a-time sensitivity screening and MCKP FFN selection (`schemaVersion: 2`).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::calibration_metrics::{
    KnapsackOption, per_site_fisher_weighted_kl_metrics, solve_multiple_choice_knapsack_minimize,
    token_softmax_fisher_weights,
};
use crate::identity::{Hasher, is_lower_hex};

use super::{Accounting, BLOCK_COEFFICIENTS, GROUP_WIDTH, Representation, WIDTH, multiply};

/// Distinct parameter role categories reported separately during U2 screening.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub(super) enum SiteRole {
    Ffn,
    AttentionProjection,
    RecurrentProjection,
    Embedding,
    Output,
}

impl SiteRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ffn => "ffn",
            Self::AttentionProjection => "attention-projection",
            Self::RecurrentProjection => "recurrent-projection",
            Self::Embedding => "embedding",
            Self::Output => "output",
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CandidateOptionInput {
    #[serde(default)]
    omitted_groups: Vec<u32>,
    recovery_rank: u32,
    expected_kl_loss: Option<f64>,
    kl_gap: Option<f64>,
    teacher_dev_logits: Option<Vec<Vec<f64>>>,
    base_dev_logits: Option<Vec<Vec<f64>>>,
    recovered_dev_logits: Option<Vec<Vec<f64>>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SiteScreenInput {
    layer: u32,
    role: SiteRole,
    #[serde(default)]
    target: Option<String>,
    magnitude_score: f64,
    branch_magnitude: f64,
    logit_delta_l2: f64,
    residual_drift: f64,
    softmax_fisher_mean: Option<f64>,
    kl_gap: Option<f64>,
    expected_kl_loss: Option<f64>,
    teacher_dev_logits: Option<Vec<Vec<f64>>>,
    base_dev_logits: Option<Vec<Vec<f64>>>,
    recovered_dev_logits: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    candidates: Vec<CandidateOptionInput>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RequestV2 {
    schema_version: u32,
    weight_representation: Representation,
    max_recovery_factor_bytes: u64,
    fit_prompt_sha256s: Vec<String>,
    dev_prompt_sha256s: Vec<String>,
    heldout_prompt_sha256s: Vec<String>,
    sites: Vec<SiteScreenInput>,
}

#[derive(Clone)]
struct EvaluatedCandidate {
    omitted_groups: Vec<u32>,
    recovery_rank: u32,
    accounting: Accounting,
    expected_kl_loss: f64,
    kl_gap: f64,
    softmax_fisher_mean: Option<f64>,
}

#[derive(Default)]
struct RoleAccumulator {
    site_count: usize,
    sum_magnitude: f64,
    sum_branch: f64,
    sum_logit_delta: f64,
    sum_drift: f64,
    sum_fisher: f64,
    fisher_count: usize,
    sum_kl_gap: f64,
    sum_recovered_kl: f64,
    kl_count: usize,
}

#[expect(
    clippy::as_conversions,
    reason = "Bounded site count converts cleanly into IEEE-754 f64 for summary means."
)]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

fn validate_prompt_split(digests: &[String], label: &str) -> crate::Result<BTreeSet<String>> {
    if digests.is_empty() {
        return Err(format!("U2 {label} must contain at least one SHA-256 digest.").into());
    }
    let mut seen = BTreeSet::new();
    for digest in digests {
        if !is_lower_hex(digest, 64) {
            return Err(
                format!("U2 {label} must contain only lowercase 64-hex SHA-256 digests.").into(),
            );
        }
        seen.insert(digest.clone());
    }
    Ok(seen)
}

fn validate_nonneg_finite(value: f64, label: &str) -> crate::Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("U2 {label} must be finite and non-negative, got {value}.").into());
    }
    Ok(())
}

fn option_accounting(
    omitted_groups: &[u32],
    recovery_rank: u32,
    representation: Representation,
) -> crate::Result<Accounting> {
    if omitted_groups.is_empty() {
        if recovery_rank != 0 {
            return Err("U2 option with no omittedGroups must use recoveryRank 0.".into());
        }
        return Ok(Accounting::default());
    }
    if !(1..=16).contains(&omitted_groups.len())
        || omitted_groups
            .iter()
            .zip(omitted_groups.iter().skip(1))
            .any(|(left, right)| left >= right)
        || omitted_groups.iter().any(|&g| g >= 17)
    {
        return Err(
            "U2 omittedGroups must be sorted distinct indices 0..16 keeping >= 1 group.".into(),
        );
    }
    if recovery_rank > 16 {
        return Err(format!("U2 recoveryRank {recovery_rank} exceeds 16.").into());
    }
    let omitted_len = u64::try_from(omitted_groups.len()).map_err(|e| e.to_string())?;
    let omitted_macs = multiply(multiply(3 * WIDTH, GROUP_WIDTH)?, omitted_len)?;
    let factor_macs = multiply(2 * WIDTH, u64::from(recovery_rank))?;
    let gross_bytes = multiply(
        omitted_macs / BLOCK_COEFFICIENTS,
        representation.block_bytes(),
    )?;
    let factor_bytes = multiply(factor_macs, 4)?;
    let net_bytes = gross_bytes
        .checked_sub(factor_bytes)
        .ok_or("U2 FFN recovery factors exceed omitted packed payload.")?;
    Ok(Accounting {
        gross_omitted_packed_payload_bytes: gross_bytes,
        new_factor_payload_bytes: factor_bytes,
        net_packed_minus_factor_payload_bytes: net_bytes,
        omitted_base_projection_multiply_accumulates_per_token: omitted_macs,
        new_factor_projection_multiply_accumulates_per_token: factor_macs,
    })
}

fn evaluate_kl_metrics(
    site_label: &str,
    expected_kl_loss: Option<f64>,
    kl_gap: Option<f64>,
    softmax_fisher_mean: Option<f64>,
    teacher: Option<&Vec<Vec<f64>>>,
    base: Option<&Vec<Vec<f64>>>,
    recovered: Option<&Vec<Vec<f64>>>,
) -> crate::Result<(f64, f64, Option<f64>)> {
    let raw_any = teacher.is_some() || base.is_some() || recovered.is_some();
    if raw_any {
        let t = teacher.ok_or_else(|| {
            format!("U2 {site_label} requires teacherDevLogits when raw logits are supplied.")
        })?;
        let b = base.ok_or_else(|| {
            format!("U2 {site_label} requires baseDevLogits when raw logits are supplied.")
        })?;
        let r = recovered.ok_or_else(|| {
            format!("U2 {site_label} requires recoveredDevLogits when raw logits are supplied.")
        })?;
        let (_, rec_kl, gap) = per_site_fisher_weighted_kl_metrics(t, b, r)?;
        let weights = token_softmax_fisher_weights(t)?;
        let fisher_mean = weights.iter().sum::<f64>() / usize_to_f64(weights.len());
        return Ok((rec_kl, gap, Some(fisher_mean)));
    }
    let rec_kl = expected_kl_loss.ok_or_else(|| {
        format!("U2 {site_label} requires expectedKlLoss or raw dev logits tuple.")
    })?;
    validate_nonneg_finite(rec_kl, "expectedKlLoss")?;
    let gap = kl_gap.unwrap_or(0.0);
    if !gap.is_finite() {
        return Err(format!("U2 {site_label} klGap must be finite.").into());
    }
    if let Some(fisher) = softmax_fisher_mean {
        validate_nonneg_finite(fisher, "softmaxFisherMean")?;
    }
    Ok((rec_kl, gap, softmax_fisher_mean))
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep U2 split validation, role summaries, exact MCKP, and baselines auditable together."
)]
pub(super) fn run(value: Value) -> crate::Result<Value> {
    let request: RequestV2 = serde_json::from_value(value)
        .map_err(|error| format!("Invalid U2 FFN sensitivity request: {error}"))?;
    if request.schema_version != 2 {
        return Err("U2 sensitivity screening requires schemaVersion 2.".into());
    }
    if request.max_recovery_factor_bytes == 0 {
        return Err("U2 maxRecoveryFactorBytes must be positive.".into());
    }
    if request.sites.is_empty() || request.sites.len() > 512 {
        return Err("U2 sensitivity screening requires 1..=512 site entries.".into());
    }

    let fit_set = validate_prompt_split(&request.fit_prompt_sha256s, "fitPromptSha256s")?;
    let dev_set = validate_prompt_split(&request.dev_prompt_sha256s, "devPromptSha256s")?;
    let heldout_set =
        validate_prompt_split(&request.heldout_prompt_sha256s, "heldoutPromptSha256s")?;
    if !fit_set.is_disjoint(&dev_set)
        || !fit_set.is_disjoint(&heldout_set)
        || !dev_set.is_disjoint(&heldout_set)
    {
        return Err(
            "U1 split violation: fitPromptSha256s, devPromptSha256s, and heldoutPromptSha256s must be pairwise disjoint."
                .into(),
        );
    }

    let mut seen_site_keys = BTreeSet::new();
    let mut role_accs: BTreeMap<SiteRole, RoleAccumulator> = BTreeMap::new();
    let mut ffn_groups: Vec<Vec<EvaluatedCandidate>> = Vec::new();
    let mut ffn_site_meta: Vec<(u32, String, f64)> = Vec::new();
    let mut ffn_knapsack: Vec<Vec<KnapsackOption>> = Vec::new();
    let mut screened_sites_json = Vec::with_capacity(request.sites.len());

    for site in &request.sites {
        if site.layer >= 64 {
            return Err(format!("U2 layer {} exceeds 0..63.", site.layer).into());
        }
        let target_label = site
            .target
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("default");
        let dedup_key = (site.layer, site.role, target_label.to_owned());
        if !seen_site_keys.insert(dedup_key) {
            return Err(format!(
                "Duplicate U2 screen entry for layer {} role {} target '{}'.",
                site.layer,
                site.role.as_str(),
                target_label
            )
            .into());
        }
        validate_nonneg_finite(site.magnitude_score, "magnitudeScore")?;
        validate_nonneg_finite(site.branch_magnitude, "branchMagnitude")?;
        validate_nonneg_finite(site.logit_delta_l2, "logitDeltaL2")?;
        validate_nonneg_finite(site.residual_drift, "residualDrift")?;

        let acc = role_accs.entry(site.role).or_default();
        acc.site_count += 1;
        acc.sum_magnitude += site.magnitude_score;
        acc.sum_branch += site.branch_magnitude;
        acc.sum_logit_delta += site.logit_delta_l2;
        acc.sum_drift += site.residual_drift;

        let mut evaluated_candidates = Vec::new();
        if site.role == SiteRole::Ffn {
            if site.candidates.is_empty() {
                return Err(format!(
                    "U2 FFN site layer {} must provide >= 1 candidate option.",
                    site.layer
                )
                .into());
            }
            let mut k_opts = Vec::with_capacity(site.candidates.len());
            for (c_idx, cand) in site.candidates.iter().enumerate() {
                let accounting = option_accounting(
                    &cand.omitted_groups,
                    cand.recovery_rank,
                    request.weight_representation,
                )?;
                let (rec_kl, gap, fisher) = evaluate_kl_metrics(
                    &format!("FFN layer {} cand {c_idx}", site.layer),
                    cand.expected_kl_loss,
                    cand.kl_gap,
                    site.softmax_fisher_mean,
                    cand.teacher_dev_logits.as_ref(),
                    cand.base_dev_logits.as_ref(),
                    cand.recovered_dev_logits.as_ref(),
                )?;
                k_opts.push(KnapsackOption {
                    cost: accounting.new_factor_payload_bytes,
                    value: rec_kl,
                });
                evaluated_candidates.push(EvaluatedCandidate {
                    omitted_groups: cand.omitted_groups.clone(),
                    recovery_rank: cand.recovery_rank,
                    accounting,
                    expected_kl_loss: rec_kl,
                    kl_gap: gap,
                    softmax_fisher_mean: fisher,
                });
            }
            let best_cand = evaluated_candidates
                .iter()
                .min_by(|a, b| a.expected_kl_loss.total_cmp(&b.expected_kl_loss))
                .ok_or("Missing evaluated FFN candidate.")?;
            acc.sum_kl_gap += best_cand.kl_gap;
            acc.sum_recovered_kl += best_cand.expected_kl_loss;
            acc.kl_count += 1;
            if let Some(fisher) = best_cand.softmax_fisher_mean {
                acc.sum_fisher += fisher;
                acc.fisher_count += 1;
            }
            ffn_knapsack.push(k_opts);
            ffn_groups.push(evaluated_candidates);
            ffn_site_meta.push((site.layer, target_label.to_owned(), site.magnitude_score));
            screened_sites_json.push(json!({
                "layer": site.layer,
                "role": site.role,
                "target": target_label,
                "magnitudeScore": site.magnitude_score,
                "branchMagnitude": site.branch_magnitude,
                "logitDeltaL2": site.logit_delta_l2,
                "residualDrift": site.residual_drift,
                "candidateCount": site.candidates.len(),
            }));
        } else {
            let (rec_kl, gap, fisher) = evaluate_kl_metrics(
                &format!("role {} layer {}", site.role.as_str(), site.layer),
                site.expected_kl_loss.or(Some(0.0)),
                site.kl_gap,
                site.softmax_fisher_mean,
                site.teacher_dev_logits.as_ref(),
                site.base_dev_logits.as_ref(),
                site.recovered_dev_logits.as_ref(),
            )?;
            acc.sum_kl_gap += gap;
            acc.sum_recovered_kl += rec_kl;
            acc.kl_count += 1;
            if let Some(f) = fisher {
                acc.sum_fisher += f;
                acc.fisher_count += 1;
            }
            screened_sites_json.push(json!({
                "layer": site.layer,
                "role": site.role,
                "target": target_label,
                "magnitudeScore": site.magnitude_score,
                "branchMagnitude": site.branch_magnitude,
                "logitDeltaL2": site.logit_delta_l2,
                "residualDrift": site.residual_drift,
                "klGap": gap,
                "recoveredKlLoss": rec_kl,
                "softmaxFisherMean": fisher,
            }));
        }
    }

    if ffn_groups.is_empty() {
        return Err("U2 sensitivity screening requires at least one 'ffn' site.".into());
    }

    // Solve exact fixed-budget MCKP minimizing total dev KL loss across FFN sites.
    let mckp_choices =
        solve_multiple_choice_knapsack_minimize(&ffn_knapsack, request.max_recovery_factor_bytes)?;

    let mut mckp_aggregate = Accounting::default();
    let mut mckp_total_kl = 0.0_f64;
    let mut selected_ffn_json = Vec::with_capacity(ffn_groups.len());
    let mut hasher = Hasher::new()?;
    hasher.update(
        format!(
            "rep={:?};budget={};",
            request.weight_representation, request.max_recovery_factor_bytes
        )
        .as_bytes(),
    )?;
    for d in &fit_set {
        hasher.update(format!("fit={d};").as_bytes())?;
    }
    for d in &dev_set {
        hasher.update(format!("dev={d};").as_bytes())?;
    }

    for (idx, (group, &chosen)) in ffn_groups.iter().zip(&mckp_choices).enumerate() {
        let cand = &group[chosen];
        mckp_aggregate.accumulate(&cand.accounting)?;
        mckp_total_kl += cand.expected_kl_loss;
        let (layer, ref target, mag) = ffn_site_meta[idx];
        hasher.update(
            format!(
                "layer={layer};target={target};rank={};groups={:?};",
                cand.recovery_rank, cand.omitted_groups
            )
            .as_bytes(),
        )?;
        selected_ffn_json.push(json!({
            "layer": layer,
            "target": target,
            "magnitudeScore": mag,
            "omittedGroups": cand.omitted_groups,
            "recoveryRank": cand.recovery_rank,
            "expectedKlLoss": cand.expected_kl_loss,
            "klGap": cand.kl_gap,
            "softmaxFisherMean": cand.softmax_fisher_mean,
            "accounting": cand.accounting,
        }));
    }
    let frozen_selection_sha256 = hasher.finalize_hex()?;

    // Baseline 1: Uniform rank across all FFN sites within budget (or lowest-cost fallback).
    let (uniform_kl, uniform_bytes) =
        compute_uniform_baseline(&ffn_groups, request.max_recovery_factor_bytes);

    // Baseline 2: Magnitude-greedy allocation prioritizing sites with largest magnitude_score.
    let (magnitude_kl, magnitude_bytes) = compute_magnitude_baseline(
        &ffn_groups,
        &ffn_site_meta,
        request.max_recovery_factor_bytes,
    );

    let mut role_summaries = BTreeMap::new();
    for (role, acc) in role_accs {
        let denom = usize_to_f64(acc.site_count.max(1));
        role_summaries.insert(
            role.as_str(),
            json!({
                "siteCount": acc.site_count,
                "meanMagnitudeScore": acc.sum_magnitude / denom,
                "meanBranchMagnitude": acc.sum_branch / denom,
                "meanLogitDeltaL2": acc.sum_logit_delta / denom,
                "meanResidualDrift": acc.sum_drift / denom,
                "meanSoftmaxFisher": (acc.fisher_count > 0).then(|| acc.sum_fisher / usize_to_f64(acc.fisher_count)),
                "meanKlGap": acc.sum_kl_gap / usize_to_f64(acc.kl_count.max(1)),
                "meanRecoveredKlLoss": acc.sum_recovered_kl / usize_to_f64(acc.kl_count.max(1)),
            }),
        );
    }

    Ok(json!({
        "schemaVersion": 2,
        "weightRepresentation": request.weight_representation,
        "maxRecoveryFactorBytes": request.max_recovery_factor_bytes,
        "splitCounts": {
            "fit": fit_set.len(),
            "dev": dev_set.len(),
            "heldout": heldout_set.len(),
            "pairwiseDisjoint": true,
        },
        "heldoutSplitUsedForSelection": false,
        "frozenSelectionSha256": frozen_selection_sha256,
        "roleSummaries": role_summaries,
        "screenedSites": screened_sites_json,
        "selectedFfnSites": selected_ffn_json,
        "aggregate": mckp_aggregate,
        "knapsackObjectiveKlLoss": mckp_total_kl,
        "baselines": {
            "uniform": {
                "totalExpectedKlLoss": uniform_kl,
                "totalFactorPayloadBytes": uniform_bytes,
                "knapsackImprovementKl": uniform_kl - mckp_total_kl,
            },
            "magnitudeGreedy": {
                "totalExpectedKlLoss": magnitude_kl,
                "totalFactorPayloadBytes": magnitude_bytes,
                "knapsackImprovementKl": magnitude_kl - mckp_total_kl,
            }
        },
        "sourceAuthenticated": false,
        "selectiveLoadingImplemented": false,
        "performanceQualified": false,
        "behaviorQualified": false,
    }))
}

fn compute_uniform_baseline(groups: &[Vec<EvaluatedCandidate>], budget: u64) -> (f64, u64) {
    let min_choices: Vec<usize> = groups
        .iter()
        .map(|g| {
            g.iter()
                .enumerate()
                .min_by_key(|&(_, c)| c.accounting.new_factor_payload_bytes)
                .map_or(0, |(idx, _)| idx)
        })
        .collect();
    let mut best_kl = groups
        .iter()
        .zip(&min_choices)
        .map(|(g, &c)| g[c].expected_kl_loss)
        .sum::<f64>();
    let mut best_bytes = groups
        .iter()
        .zip(&min_choices)
        .map(|(g, &c)| g[c].accounting.new_factor_payload_bytes)
        .sum::<u64>();

    let mut all_ranks: BTreeSet<u32> = BTreeSet::new();
    for g in groups {
        for c in g {
            all_ranks.insert(c.recovery_rank);
        }
    }
    for rank in all_ranks {
        let mut total_bytes = 0_u64;
        let mut total_kl = 0.0_f64;
        let mut valid = true;
        for g in groups {
            if let Some(c) = g
                .iter()
                .filter(|c| c.recovery_rank == rank)
                .min_by(|a, b| a.expected_kl_loss.total_cmp(&b.expected_kl_loss))
            {
                total_bytes = total_bytes.saturating_add(c.accounting.new_factor_payload_bytes);
                total_kl += c.expected_kl_loss;
            } else {
                valid = false;
                break;
            }
        }
        if valid && total_bytes <= budget && total_kl < best_kl {
            best_kl = total_kl;
            best_bytes = total_bytes;
        }
    }
    (best_kl, best_bytes)
}

fn compute_magnitude_baseline(
    groups: &[Vec<EvaluatedCandidate>],
    site_meta: &[(u32, String, f64)],
    budget: u64,
) -> (f64, u64) {
    // Initialize every site at its minimum-cost candidate, then upgrade in descending magnitudeScore order.
    let mut chosen: Vec<usize> = groups
        .iter()
        .map(|g| {
            g.iter()
                .enumerate()
                .min_by(|&(ia, a), &(ib, b)| {
                    a.accounting
                        .new_factor_payload_bytes
                        .cmp(&b.accounting.new_factor_payload_bytes)
                        .then_with(|| a.expected_kl_loss.total_cmp(&b.expected_kl_loss))
                        .then_with(|| ia.cmp(&ib))
                })
                .map_or(0, |(idx, _)| idx)
        })
        .collect();
    let mut used_bytes: u64 = groups
        .iter()
        .zip(&chosen)
        .map(|(g, &idx)| g[idx].accounting.new_factor_payload_bytes)
        .sum();
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by(|&a, &b| {
        site_meta[b]
            .2
            .total_cmp(&site_meta[a].2)
            .then_with(|| site_meta[a].0.cmp(&site_meta[b].0))
    });
    for idx in order {
        let base_idx = chosen[idx];
        let base_cost = groups[idx][base_idx].accounting.new_factor_payload_bytes;
        let rem_plus_base = budget.saturating_sub(used_bytes).saturating_add(base_cost);
        if let Some((best_opt, _)) = groups[idx]
            .iter()
            .enumerate()
            .filter(|&(_, c)| c.accounting.new_factor_payload_bytes <= rem_plus_base)
            .min_by(|&(ia, a), &(ib, b)| {
                a.expected_kl_loss
                    .total_cmp(&b.expected_kl_loss)
                    .then_with(|| ia.cmp(&ib))
            })
        {
            used_bytes =
                used_bytes - base_cost + groups[idx][best_opt].accounting.new_factor_payload_bytes;
            chosen[idx] = best_opt;
        }
    }
    let total_kl = groups
        .iter()
        .zip(&chosen)
        .map(|(g, &c)| g[c].expected_kl_loss)
        .sum();
    (total_kl, used_bytes)
}
