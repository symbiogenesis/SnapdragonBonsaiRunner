//! Scaled floating-point calibration, directional alignment and dispersion metrics.

use std::collections::{BTreeMap, BTreeSet};

use crate::Result;

/// Summary across a sequence of candidate-versus-reference calibration rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationRowSummary {
    /// Number of rows evaluated.
    pub rows: u32,
    /// Worst per-row relative `L2` error where the reference row has nonzero energy.
    pub worst_row_relative_l2_error: Option<f64>,
    /// Arithmetic mean clamped cosine similarity across rows where both vectors are nonzero.
    pub mean_cosine_similarity: Option<f64>,
    /// Minimum clamped cosine similarity across rows where both vectors are nonzero.
    pub min_cosine_similarity: Option<f64>,
    /// Arithmetic mean normalized Shannon energy entropy in `[0.0, 1.0]` across candidates.
    pub mean_candidate_energy_entropy: Option<f64>,
    /// Arithmetic mean nonzero coordinate fraction in `[0.0, 1.0]` across candidates.
    pub mean_candidate_coordinate_coverage: Option<f64>,
}

/// Incremental bounded accumulator for multi-row replay diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowMetricsAccumulator {
    rows: u32,
    worst_relative_l2_error: Option<f64>,
    cosine_sum: f64,
    cosine_count: u32,
    min_cosine_similarity: Option<f64>,
    entropy_sum: f64,
    coverage_sum: f64,
}

fn validate_and_max_abs(values: &[f64]) -> Result<f64> {
    let mut scale = 0.0_f64;
    for &value in values {
        if !value.is_finite() {
            return Err("Calibration vector contains a nonfinite element.".into());
        }
        scale = scale.max(value.abs());
    }
    Ok(scale)
}

fn count_as_f64(count: usize) -> Result<f64> {
    let value = u32::try_from(count)
        .map_err(|_| "Calibration vector length exceeds finite index bound.")?;
    Ok(f64::from(value))
}

fn scaled_l2_from_max(values: &[f64], max_abs: f64) -> Result<f64> {
    if max_abs == 0.0 {
        return Ok(0.0);
    }
    let mut sum_sq = 0.0_f64;
    for &value in values {
        let ratio = value / max_abs;
        sum_sq += ratio * ratio;
    }
    let norm = max_abs * sum_sq.sqrt();
    if !norm.is_finite() {
        return Err("Calibration L2 norm overflows finite F64 bounds.".into());
    }
    Ok(norm)
}

/// Computes the max-abs scaled `f64` Euclidean (`L2`) norm without spurious overflow or underflow.
///
/// # Errors
/// Returns an error if any coordinate is nonfinite or the resulting norm overflows `f64`.
pub fn scaled_l2_norm(values: &[f64]) -> Result<f64> {
    let scale = validate_and_max_abs(values)?;
    scaled_l2_from_max(values, scale)
}

fn scaled_difference_norm(left: &[f64], right: &[f64], common_scale: f64) -> Result<f64> {
    if left.len() != right.len() {
        return Err("Calibration vectors must have matching dimensions.".into());
    }
    if common_scale == 0.0 {
        return Ok(0.0);
    }
    let direct_safe = common_scale <= f64::MAX * 0.25;
    let mut diff_scale = 0.0_f64;
    for (&left_value, &right_value) in left.iter().zip(right) {
        let term = if direct_safe {
            (left_value - right_value).abs()
        } else {
            (0.5 * left_value - 0.5 * right_value).abs()
        };
        diff_scale = diff_scale.max(term);
    }
    if diff_scale == 0.0 {
        return Ok(0.0);
    }
    let mut sum_sq = 0.0_f64;
    for (&left_value, &right_value) in left.iter().zip(right) {
        let diff = if direct_safe {
            left_value - right_value
        } else {
            0.5 * left_value - 0.5 * right_value
        };
        let ratio = diff / diff_scale;
        sum_sq += ratio * ratio;
    }
    let factor = if direct_safe {
        diff_scale
    } else {
        2.0 * diff_scale
    };
    let norm = factor * sum_sq.sqrt();
    if !norm.is_finite() {
        return Err("Calibration difference norm overflows finite F64 bounds.".into());
    }
    Ok(norm)
}

/// Computes the scaled `f64` Euclidean (`L2`) distance between two vectors of equal dimension.
///
/// # Errors
/// Returns an error on dimension mismatch, nonfinite inputs, or `f64` overflow of the distance.
pub fn scaled_l2_distance(left: &[f64], right: &[f64]) -> Result<f64> {
    if left.len() != right.len() {
        return Err("Calibration vectors must have matching dimensions.".into());
    }
    let common_scale = validate_and_max_abs(left)?.max(validate_and_max_abs(right)?);
    scaled_difference_norm(left, right, common_scale)
}

/// Computes clamped cosine similarity in `[-1.0, 1.0]`, returning `None` if either vector is zero.
///
/// # Errors
/// Returns an error on dimension mismatch, nonfinite inputs, or nonfinite intermediate reductions.
pub fn cosine_similarity(left: &[f64], right: &[f64]) -> Result<Option<f64>> {
    if left.len() != right.len() {
        return Err("Calibration vectors must have matching dimensions.".into());
    }
    let left_scale = validate_and_max_abs(left)?;
    let right_scale = validate_and_max_abs(right)?;
    if left_scale == 0.0 || right_scale == 0.0 {
        return Ok(None);
    }
    let mut dot = 0.0_f64;
    let mut left_sq = 0.0_f64;
    let mut right_sq = 0.0_f64;
    for (&left_value, &right_value) in left.iter().zip(right) {
        let left_scaled = left_value / left_scale;
        let right_scaled = right_value / right_scale;
        dot += left_scaled * right_scaled;
        left_sq += left_scaled * left_scaled;
        right_sq += right_scaled * right_scaled;
    }
    let denom = left_sq.sqrt() * right_sq.sqrt();
    if !dot.is_finite() || !denom.is_finite() || denom <= 0.0 {
        return Err("Cosine similarity reduction produced a nonfinite or zero denominator.".into());
    }
    Ok(Some((dot / denom).clamp(-1.0, 1.0)))
}

/// Computes normalized Shannon energy entropy in `[0.0, 1.0]` over squared coordinate weights.
///
/// Returns `0.0` for vectors with at most one coordinate or zero total energy.
///
/// # Errors
/// Returns an error if any coordinate is nonfinite or the dimension exceeds `u32::MAX`.
pub fn normalized_energy_entropy(values: &[f64]) -> Result<f64> {
    let scale = validate_and_max_abs(values)?;
    let total_coords = count_as_f64(values.len())?;
    if values.len() <= 1 || scale == 0.0 {
        return Ok(0.0);
    }
    let mut sum_sq = 0.0_f64;
    for &value in values {
        let ratio = value / scale;
        sum_sq += ratio * ratio;
    }
    if !sum_sq.is_finite() || sum_sq <= 0.0 {
        return Err("Energy entropy normalization encountered nonfinite energy.".into());
    }
    let mut entropy = 0.0_f64;
    for &value in values {
        let ratio = value / scale;
        let share = (ratio * ratio) / sum_sq;
        if share > 0.0 {
            entropy -= share * share.ln();
        }
    }
    let max_entropy = total_coords.ln();
    if !entropy.is_finite() || !max_entropy.is_finite() || max_entropy <= 0.0 {
        return Err("Normalized energy entropy reduction failed finiteness checks.".into());
    }
    Ok((entropy / max_entropy).clamp(0.0, 1.0))
}

/// Computes the fraction of nonzero coordinates in `[0.0, 1.0]`.
///
/// Returns `0.0` for empty slices.
///
/// # Errors
/// Returns an error if any coordinate is nonfinite or the dimension exceeds `u32::MAX`.
pub fn coordinate_coverage(values: &[f64]) -> Result<f64> {
    let _ = validate_and_max_abs(values)?;
    if values.is_empty() {
        return Ok(0.0);
    }
    let total = count_as_f64(values.len())?;
    let active_count = values.iter().filter(|&&value| value != 0.0).count();
    let active = count_as_f64(active_count)?;
    Ok((active / total).clamp(0.0, 1.0))
}

/// Computes per-row relative `L2` error `||reference - candidate||_2 / ||reference||_2`.
///
/// Returns `Ok(None)` when `reference` has zero norm.
///
/// # Errors
/// Returns an error on dimension mismatch, nonfinite inputs, or nonfinite scaled division.
pub fn relative_l2_error(reference: &[f64], candidate: &[f64]) -> Result<Option<f64>> {
    if reference.len() != candidate.len() {
        return Err("Calibration vectors must have matching dimensions.".into());
    }
    let ref_scale = validate_and_max_abs(reference)?;
    let cand_scale = validate_and_max_abs(candidate)?;
    if ref_scale == 0.0 {
        return Ok(None);
    }
    let common_scale = ref_scale.max(cand_scale);
    let mut diff_norm = 0.0_f64;
    let mut ref_norm = 0.0_f64;
    for (&ref_val, &cand_val) in reference.iter().zip(candidate) {
        let r = ref_val / common_scale;
        let c = cand_val / common_scale;
        let d = r - c;
        diff_norm += d * d;
        ref_norm += r * r;
    }
    let ref_root = ref_norm.sqrt();
    if ref_root == 0.0 || !ref_root.is_finite() || !diff_norm.is_finite() {
        return Err("Relative L2 error denominator collapsed during scaling.".into());
    }
    let rel = diff_norm.sqrt() / ref_root;
    if !rel.is_finite() {
        return Err("Relative L2 error overflowed finite F64 bounds.".into());
    }
    Ok(Some(rel))
}

/// Computes the maximum per-row relative `L2` error across `(reference, candidate)` row pairs.
///
/// Returns `Ok(None)` if no pair has a nonzero reference vector.
///
/// # Errors
/// Returns an error if any pair has mismatched dimensions or nonfinite values.
pub fn worst_row_relative_l2_error(pairs: &[(&[f64], &[f64])]) -> Result<Option<f64>> {
    let mut worst: Option<f64> = None;
    for &(reference, candidate) in pairs {
        if let Some(error) = relative_l2_error(reference, candidate)? {
            worst = Some(worst.map_or(error, |prev| prev.max(error)));
        }
    }
    Ok(worst)
}

impl RowMetricsAccumulator {
    /// Creates an empty row metrics accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one `(reference, candidate)` pair into the incremental calibration summary.
    ///
    /// # Errors
    /// Returns an error on dimension mismatch, nonfinite inputs, or row count overflow.
    pub fn record_pair(&mut self, reference: &[f64], candidate: &[f64]) -> Result<()> {
        if reference.len() != candidate.len() {
            return Err("Calibration vectors must have matching dimensions.".into());
        }
        if let Some(error) = relative_l2_error(reference, candidate)? {
            self.worst_relative_l2_error = Some(
                self.worst_relative_l2_error
                    .map_or(error, |prev| prev.max(error)),
            );
        }
        if let Some(cosine) = cosine_similarity(reference, candidate)? {
            self.cosine_sum += cosine;
            self.cosine_count = self
                .cosine_count
                .checked_add(1)
                .ok_or("Cosine row count overflow.")?;
            self.min_cosine_similarity = Some(
                self.min_cosine_similarity
                    .map_or(cosine, |prev| prev.min(cosine)),
            );
        }
        self.entropy_sum += normalized_energy_entropy(candidate)?;
        self.coverage_sum += coordinate_coverage(candidate)?;
        self.rows = self
            .rows
            .checked_add(1)
            .ok_or("Calibration row count overflow.")?;
        Ok(())
    }

    /// Finalizes the accumulated per-row calibration metrics summary.
    #[must_use]
    pub fn finish(&self) -> CalibrationRowSummary {
        let rows_f64 = f64::from(self.rows);
        CalibrationRowSummary {
            rows: self.rows,
            worst_row_relative_l2_error: self.worst_relative_l2_error,
            mean_cosine_similarity: (self.cosine_count > 0)
                .then(|| (self.cosine_sum / f64::from(self.cosine_count)).clamp(-1.0, 1.0)),
            min_cosine_similarity: self.min_cosine_similarity,
            mean_candidate_energy_entropy: (self.rows > 0)
                .then(|| (self.entropy_sum / rows_f64).clamp(0.0, 1.0)),
            mean_candidate_coordinate_coverage: (self.rows > 0)
                .then(|| (self.coverage_sum / rows_f64).clamp(0.0, 1.0)),
        }
    }
}

/// Numerically stable log-softmax in `f64` from raw logits.
///
/// # Errors
/// Returns an error if `logits` is empty or contains non-finite values.
pub fn log_softmax_f64(logits: &[f64]) -> Result<Vec<f64>> {
    if logits.is_empty() {
        return Err("log_softmax requires at least one logit".to_owned());
    }
    let mut max_logit = f64::NEG_INFINITY;
    for &value in logits {
        if !value.is_finite() {
            return Err(format!("logit must be finite, got {value}"));
        }
        if value > max_logit {
            max_logit = value;
        }
    }
    let sum_exp: f64 = logits.iter().map(|&value| (value - max_logit).exp()).sum();
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return Err("log_softmax normalizer must be positive and finite".to_owned());
    }
    let log_z = max_logit + sum_exp.ln();
    Ok(logits.iter().map(|&value| value - log_z).collect())
}

/// Forward KL divergence `KL(teacher || student)` computed in `f64` from raw logits.
///
/// # Errors
/// Returns an error on length mismatch, empty vectors, or non-finite values.
pub fn kl_divergence_from_logits(teacher_logits: &[f64], student_logits: &[f64]) -> Result<f64> {
    if teacher_logits.len() != student_logits.len() {
        return Err(format!(
            "KL logit vocabulary length mismatch: teacher={}, student={}",
            teacher_logits.len(),
            student_logits.len()
        ));
    }
    let log_p = log_softmax_f64(teacher_logits)?;
    let log_q = log_softmax_f64(student_logits)?;
    let mut kl = 0.0_f64;
    for (&lp, &lq) in log_p.iter().zip(&log_q) {
        let p = lp.exp();
        if p > 0.0 {
            kl += p * (lp - lq);
        }
    }
    if !kl.is_finite() {
        return Err("KL divergence overflowed finite bounds".to_owned());
    }
    Ok(kl.max(0.0))
}

/// Softmax-Fisher information weight `sum_v p_v * (1 - p_v)` computed in `f64` from logits.
///
/// Bounded in `[0.0, 1.0]`, peaking on high-ambiguity decision distributions and vanishing
/// on one-hot near-deterministic predictions.
///
/// # Errors
/// Returns an error if `logits` is empty or contains non-finite values.
pub fn softmax_fisher_from_logits(logits: &[f64]) -> Result<f64> {
    let log_p = log_softmax_f64(logits)?;
    let mut argmax = 0_usize;
    let mut best_lp = f64::NEG_INFINITY;
    for (idx, &lp) in log_p.iter().enumerate() {
        if lp > best_lp {
            best_lp = lp;
            argmax = idx;
        }
    }
    let probs: Vec<f64> = log_p.into_iter().map(f64::exp).collect();
    let rest_sum: f64 = probs
        .iter()
        .enumerate()
        .filter(|&(idx, _)| idx != argmax)
        .map(|(_, &p)| p)
        .sum();
    let fisher: f64 = probs[argmax] * rest_sum
        + probs
            .iter()
            .enumerate()
            .filter(|&(idx, _)| idx != argmax)
            .map(|(_, &p)| p * (1.0 - p))
            .sum::<f64>();
    if !fisher.is_finite() {
        return Err("softmax Fisher weight overflowed finite bounds".to_owned());
    }
    Ok(fisher.clamp(0.0, 1.0))
}

/// Per-token Softmax-Fisher weights for a slice of teacher logit rows.
///
/// # Errors
/// Returns an error if any token row is empty or contains non-finite logits.
pub fn token_softmax_fisher_weights<T: AsRef<[f64]>>(teacher_logits: &[T]) -> Result<Vec<f64>> {
    let mut weights = Vec::with_capacity(teacher_logits.len());
    for row in teacher_logits {
        weights.push(softmax_fisher_from_logits(row.as_ref())?);
    }
    Ok(weights)
}

/// Non-negative weighted mean KL divergence across diagnostic tokens.
///
/// Falls back cleanly to unweighted mean if all weights are `0.0` (all deterministic one-hot).
///
/// # Errors
/// Returns an error on empty slices, length mismatch, negative weights, or non-finite inputs.
pub fn weighted_mean_kl_divergence(kl_values: &[f64], weights: &[f64]) -> Result<f64> {
    if kl_values.is_empty() || kl_values.len() != weights.len() {
        return Err(format!(
            "weighted mean KL requires matching non-empty vectors (kl={}, weights={})",
            kl_values.len(),
            weights.len()
        ));
    }
    let mut weight_sum = 0.0_f64;
    let mut weighted_sum = 0.0_f64;
    let mut raw_sum = 0.0_f64;
    for (&kl, &w) in kl_values.iter().zip(weights) {
        if !kl.is_finite() || kl < 0.0 {
            return Err(format!(
                "KL value must be finite and non-negative, got {kl}"
            ));
        }
        if !w.is_finite() || w < 0.0 {
            return Err(format!("weight must be finite and non-negative, got {w}"));
        }
        weight_sum += w;
        weighted_sum += w * kl;
        raw_sum += kl;
    }
    if !weight_sum.is_finite() || !weighted_sum.is_finite() || !raw_sum.is_finite() {
        return Err("weighted KL aggregation overflowed finite bounds".to_owned());
    }
    if weight_sum == 0.0 {
        return Ok(raw_sum / count_as_f64(kl_values.len())?);
    }
    Ok(weighted_sum / weight_sum)
}

/// Fisher-weighted `(base_kl, recovered_kl, kl_gap)` across diagnostic tokens.
///
/// `kl_gap = weighted_mean(KL(teacher || base)) - weighted_mean(KL(teacher || recovered))`.
/// Positive `kl_gap` quantifies how much teacher-aligned KL divergence recovering the site removes.
///
/// # Errors
/// Returns an error if token counts or vocabulary widths mismatch, slices are empty, or logits
/// contain non-finite entries.
pub fn per_site_fisher_weighted_kl_metrics<T: AsRef<[f64]>>(
    teacher_logits: &[T],
    base_logits: &[T],
    recovered_logits: &[T],
) -> Result<(f64, f64, f64)> {
    if teacher_logits.is_empty()
        || teacher_logits.len() != base_logits.len()
        || teacher_logits.len() != recovered_logits.len()
    {
        return Err(format!(
            "per-site KL gap requires matching non-empty token counts (teacher={}, base={}, recovered={})",
            teacher_logits.len(),
            base_logits.len(),
            recovered_logits.len()
        ));
    }
    let weights = token_softmax_fisher_weights(teacher_logits)?;
    let mut base_kls = Vec::with_capacity(teacher_logits.len());
    let mut rec_kls = Vec::with_capacity(teacher_logits.len());
    for ((t, b), r) in teacher_logits.iter().zip(base_logits).zip(recovered_logits) {
        base_kls.push(kl_divergence_from_logits(t.as_ref(), b.as_ref())?);
        rec_kls.push(kl_divergence_from_logits(t.as_ref(), r.as_ref())?);
    }
    let base_mean = weighted_mean_kl_divergence(&base_kls, &weights)?;
    let rec_mean = weighted_mean_kl_divergence(&rec_kls, &weights)?;
    Ok((base_mean, rec_mean, base_mean - rec_mean))
}

/// Fisher-weighted per-site KL gap `mean_base_kl - mean_recovered_kl` across diagnostic tokens.
///
/// # Errors
/// Returns the errors of [`per_site_fisher_weighted_kl_metrics`].
pub fn per_site_kl_gap<T: AsRef<[f64]>>(
    teacher_logits: &[T],
    base_logits: &[T],
    recovered_logits: &[T],
) -> Result<f64> {
    let (_, _, gap) =
        per_site_fisher_weighted_kl_metrics(teacher_logits, base_logits, recovered_logits)?;
    Ok(gap)
}

/// One candidate option inside a multiple-choice knapsack group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KnapsackOption {
    /// Non-negative discrete resource cost (such as byte size).
    pub cost: u64,
    /// Objective value contributed when selecting this option.
    pub value: f64,
}

const MAX_KNAPSACK_CELLS: usize = 32_000_000;

fn u64_gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn solve_multiple_choice_knapsack(
    groups: &[Vec<KnapsackOption>],
    budget: u64,
    maximize: bool,
) -> Result<Vec<usize>> {
    if groups.is_empty() {
        return Ok(Vec::new());
    }
    let mut min_costs = Vec::with_capacity(groups.len());
    let mut total_min_cost = 0_u64;
    for (g_idx, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(format!(
                "knapsack group {g_idx} must contain at least one option"
            ));
        }
        if group.len() > usize::from(u16::MAX) {
            return Err(format!("knapsack group {g_idx} exceeds u16 option limit"));
        }
        let mut group_min = u64::MAX;
        for opt in group {
            if !opt.value.is_finite() {
                return Err(format!(
                    "knapsack option value in group {g_idx} must be finite"
                ));
            }
            group_min = group_min.min(opt.cost);
        }
        min_costs.push(group_min);
        total_min_cost = total_min_cost
            .checked_add(group_min)
            .ok_or_else(|| "knapsack minimum cost sum overflowed u64".to_owned())?;
    }
    if total_min_cost > budget {
        return Err(format!(
            "knapsack budget {budget} is below minimum feasible cost {total_min_cost}"
        ));
    }
    let rem_budget = budget - total_min_cost;
    let mut step_gcd = 0_u64;
    for (group, &g_min) in groups.iter().zip(&min_costs) {
        for opt in group {
            let rel = opt.cost - g_min;
            if rel > 0 {
                step_gcd = if step_gcd == 0 {
                    rel
                } else {
                    u64_gcd(step_gcd, rel)
                };
            }
        }
    }
    if step_gcd == 0 {
        // All options inside each group share equal cost; pick best value greedily per group.
        let mut chosen = Vec::with_capacity(groups.len());
        for group in groups {
            let mut best_idx = 0_usize;
            for (idx, opt) in group.iter().enumerate().skip(1) {
                let better = if maximize {
                    opt.value > group[best_idx].value
                } else {
                    opt.value < group[best_idx].value
                };
                if better {
                    best_idx = idx;
                }
            }
            chosen.push(best_idx);
        }
        return Ok(chosen);
    }

    let budget_units_u64 = rem_budget / step_gcd;
    let budget_units = usize::try_from(budget_units_u64).map_err(|_| {
        format!("scaled knapsack budget {budget_units_u64} exceeds addressable table units")
    })?;
    let stride = budget_units
        .checked_add(1)
        .ok_or_else(|| "knapsack stride overflow".to_owned())?;
    let cells = groups
        .len()
        .checked_mul(stride)
        .ok_or_else(|| "knapsack state space overflow".to_owned())?;
    if cells > MAX_KNAPSACK_CELLS {
        return Err(format!(
            "exact knapsack state space {cells} cells exceeds safety cap {MAX_KNAPSACK_CELLS}"
        ));
    }

    let init_val = if maximize {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let mut prev = vec![init_val; stride];
    let mut next = vec![init_val; stride];
    let mut choices = vec![u16::MAX; cells];
    prev[0] = 0.0;

    for (g_idx, (group, &g_min)) in groups.iter().zip(&min_costs).enumerate() {
        next.fill(init_val);
        let row_offset = g_idx * stride;
        for (opt_idx, opt) in group.iter().enumerate() {
            let cost_units = usize::try_from((opt.cost - g_min) / step_gcd)
                .map_err(|_| "option unit cost overflow".to_owned())?;
            if cost_units > budget_units {
                continue;
            }
            let opt_code = u16::try_from(opt_idx)
                .map_err(|_| "knapsack option index exceeds u16".to_owned())?;
            for u in cost_units..=budget_units {
                let base = prev[u - cost_units];
                if !base.is_finite() {
                    continue;
                }
                let candidate = base + opt.value;
                if !candidate.is_finite() {
                    continue;
                }
                let better = if maximize {
                    candidate > next[u]
                } else {
                    candidate < next[u]
                };
                if better {
                    next[u] = candidate;
                    choices[row_offset + u] = opt_code;
                }
            }
        }
        std::mem::swap(&mut prev, &mut next);
    }

    let mut best_u = 0_usize;
    let mut best_score = init_val;
    for (u, &score) in prev.iter().enumerate() {
        if !score.is_finite() {
            continue;
        }
        let better = if maximize {
            score > best_score
        } else {
            score < best_score
        };
        if better {
            best_score = score;
            best_u = u;
        }
    }
    if !best_score.is_finite() {
        return Err("knapsack found no finite feasible assignment".to_owned());
    }

    let mut chosen = vec![0_usize; groups.len()];
    let mut rem_u = best_u;
    for g_idx in (0..groups.len()).rev() {
        let code = choices[g_idx * stride + rem_u];
        let opt_idx = usize::from(code);
        let opt = groups[g_idx]
            .get(opt_idx)
            .ok_or_else(|| "knapsack backtracking encountered invalid option choice".to_owned())?;
        chosen[g_idx] = opt_idx;
        let cost_units = usize::try_from((opt.cost - min_costs[g_idx]) / step_gcd)
            .map_err(|_| "backtrack cost unit overflow".to_owned())?;
        rem_u = rem_u
            .checked_sub(cost_units)
            .ok_or_else(|| "knapsack backtrack underflow".to_owned())?;
    }
    Ok(chosen)
}

/// Exact multiple-choice knapsack solver selecting one option per group to maximize total value
/// within `budget`.
///
/// Uses `gcd`-reduced integer DP over feasible costs.
///
/// # Errors
/// Returns an error if any group is empty, values are non-finite, `budget` is infeasible, or the
/// scaled state table exceeds safety bounds.
pub fn solve_multiple_choice_knapsack_maximize(
    groups: &[Vec<KnapsackOption>],
    budget: u64,
) -> Result<Vec<usize>> {
    solve_multiple_choice_knapsack(groups, budget, true)
}

/// Exact multiple-choice knapsack solver selecting one option per group to minimize total value
/// within `budget`.
///
/// Uses `gcd`-reduced integer DP over feasible costs.
///
/// # Errors
/// Returns an error if any group is empty, values are non-finite, `budget` is infeasible, or the
/// scaled state table exceeds safety bounds.
pub fn solve_multiple_choice_knapsack_minimize(
    groups: &[Vec<KnapsackOption>],
    budget: u64,
) -> Result<Vec<usize>> {
    solve_multiple_choice_knapsack(groups, budget, false)
}

const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_9;
const NUCLEUS_KL_FLOOR: f64 = 1e-15;
const MAX_DEEPCONF_WINDOW: usize = 65_536;

/// Summary of a offline `TempQuant` temperature calibration run on disjoint teacher/student logits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempQuantCalibrationSummary {
    /// Number of aligned diagnostic token positions evaluated.
    pub tokens: u32,
    /// Reference teacher sampling temperature used to define the target next-token distribution.
    pub teacher_temperature: f64,
    /// Calibrated student temperature within the declared search bounds minimizing weighted forward KL.
    pub fitted_student_temperature: f64,
    /// Mean forward KL divergence `KL(teacher || student)` before recalibration (at `teacher_temperature`).
    pub baseline_kl: f64,
    /// Mean forward KL divergence `KL(teacher || student)` at `fitted_student_temperature`.
    pub calibrated_kl: f64,
    /// Non-negative forward KL improvement `max(0, baseline_kl - calibrated_kl)`.
    pub kl_reduction: f64,
    /// Mean teacher Shannon entropy (nats) at `teacher_temperature`.
    pub mean_teacher_entropy: f64,
    /// Mean uncalibrated student Shannon entropy (nats) at `teacher_temperature`.
    pub mean_student_entropy: f64,
    /// Mean calibrated student Shannon entropy (nats) at `fitted_student_temperature`.
    pub mean_calibrated_entropy: f64,
    /// Whether per-token objective weights used teacher Softmax-Fisher information `sum_v p_v (1 - p_v)`.
    pub fisher_weighted: bool,
}

/// Summary of an offline `TempQuant` nucleus (`top_p`) calibration search under fixed `top_k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempQuantNucleusSummary {
    /// Number of aligned diagnostic token positions evaluated.
    pub tokens: u32,
    /// Fixed top-k shortlist width applied before nucleus truncation.
    pub top_k: usize,
    /// Teacher sampling temperature.
    pub teacher_temperature: f64,
    /// Reference teacher nucleus threshold (`top_p` in `(0.0, 1.0]`).
    pub teacher_top_p: f64,
    /// Fixed student sampling temperature (e.g., from [`fit_tempquant_temperature`]).
    pub student_temperature: f64,
    /// Calibrated student `top_p` minimizing smoothed truncated forward KL across `[0.05, 1.00]`.
    pub fitted_student_top_p: f64,
    /// Mean smoothed truncated forward KL at `teacher_top_p` (`NUCLEUS_KL_FLOOR = 1e-15` over active union).
    pub baseline_truncated_kl: f64,
    /// Mean smoothed truncated forward KL at `fitted_student_top_p`.
    pub calibrated_truncated_kl: f64,
    /// Mean number of tokens retained per step in the teacher nucleus (`min_keep = 1`).
    pub mean_teacher_active_tokens: f64,
    /// Mean number of tokens retained per step in the calibrated student nucleus (`min_keep = 1`).
    pub mean_student_active_tokens: f64,
}

/// Exact Shannon entropy `-sum_v p_v * log(p_v)` (in nats) computed in `f64` from raw logits.
///
/// # Errors
/// Returns an error if `logits` is empty or contains non-finite entries.
pub fn shannon_entropy_from_logits(logits: &[f64]) -> Result<f64> {
    let log_p = log_softmax_f64(logits)?;
    let mut entropy = 0.0_f64;
    for &lp in &log_p {
        let p = lp.exp();
        if p > 0.0 {
            entropy -= p * lp;
        }
    }
    if !entropy.is_finite() {
        return Err("Shannon entropy overflowed finite bounds.".to_owned());
    }
    Ok(entropy.max(0.0))
}

fn scale_logits_by_inverse_temperature(logits: &[f64], inv_temp: f64) -> Result<Vec<f64>> {
    if !inv_temp.is_finite() || inv_temp <= 0.0 {
        return Err("Inverse temperature must be finite and positive.".to_owned());
    }
    let mut scaled = Vec::with_capacity(logits.len());
    for &value in logits {
        if !value.is_finite() {
            return Err(format!("Logit must be finite, got {value}."));
        }
        let s = value * inv_temp;
        if !s.is_finite() {
            return Err("Temperature-scaled logit overflowed finite f64 bounds.".to_owned());
        }
        scaled.push(s);
    }
    Ok(scaled)
}

fn validate_positive_finite_temperature(value: f64, label: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{label} must be finite and positive, got {value}."));
    }
    Ok(())
}

/// Forward KL divergence `KL(softmax(teacher / T_t) || softmax(student / T_s))` in `f64`.
///
/// # Errors
/// Returns an error on empty/mismatched vectors, non-positive/non-finite temperatures, non-finite
/// logits, or scaled logit overflow.
pub fn temperature_scaled_kl_from_logits(
    teacher_logits: &[f64],
    student_logits: &[f64],
    teacher_temperature: f64,
    student_temperature: f64,
) -> Result<f64> {
    if teacher_logits.len() != student_logits.len() {
        return Err(format!(
            "Temperature-scaled KL vocabulary mismatch: teacher={}, student={}",
            teacher_logits.len(),
            student_logits.len()
        ));
    }
    validate_positive_finite_temperature(teacher_temperature, "teacher_temperature")?;
    validate_positive_finite_temperature(student_temperature, "student_temperature")?;
    let scaled_teacher =
        scale_logits_by_inverse_temperature(teacher_logits, 1.0 / teacher_temperature)?;
    let scaled_student =
        scale_logits_by_inverse_temperature(student_logits, 1.0 / student_temperature)?;
    kl_divergence_from_logits(&scaled_teacher, &scaled_student)
}

fn normalized_teacher_weights(
    teacher_scaled: &[Vec<f64>],
    fisher_weighted: bool,
) -> Result<Vec<f64>> {
    let count = count_as_f64(teacher_scaled.len())?;
    if !fisher_weighted {
        return Ok(vec![1.0 / count; teacher_scaled.len()]);
    }
    let mut raw = Vec::with_capacity(teacher_scaled.len());
    let mut sum = 0.0_f64;
    for row in teacher_scaled {
        let w = softmax_fisher_from_logits(row)?;
        sum += w;
        raw.push(w);
    }
    if !sum.is_finite() {
        return Err("Teacher Softmax-Fisher weights overflowed finite f64 bounds.".to_owned());
    }
    if sum == 0.0 {
        return Ok(vec![1.0 / count; teacher_scaled.len()]);
    }
    Ok(raw.into_iter().map(|w| w / sum).collect())
}

fn evaluate_tempquant_beta<T: AsRef<[f64]>>(
    teacher_log_probs: &[Vec<f64>],
    teacher_probs: &[Vec<f64>],
    student_logits: &[T],
    weights: &[f64],
    beta: f64,
) -> Result<(f64, f64)> {
    let token_count = count_as_f64(student_logits.len())?;
    let mut weighted_kl = 0.0_f64;
    let mut entropy_sum = 0.0_f64;
    for (((t_log_p, t_p), s_raw), &w) in teacher_log_probs
        .iter()
        .zip(teacher_probs)
        .zip(student_logits)
        .zip(weights)
    {
        let scaled = scale_logits_by_inverse_temperature(s_raw.as_ref(), beta)?;
        let log_q = log_softmax_f64(&scaled)?;
        let mut kl = 0.0_f64;
        let mut ent = 0.0_f64;
        for ((&lp, &p), &lq) in t_log_p.iter().zip(t_p).zip(&log_q) {
            if p > 0.0 {
                kl += p * (lp - lq);
            }
            let q = lq.exp();
            if q > 0.0 {
                ent -= q * lq;
            }
        }
        weighted_kl += w * kl.max(0.0);
        entropy_sum += ent.max(0.0);
    }
    if !weighted_kl.is_finite() || !entropy_sum.is_finite() {
        return Err("TempQuant objective evaluation overflowed finite bounds.".to_owned());
    }
    Ok((weighted_kl.max(0.0), (entropy_sum / token_count).max(0.0)))
}

/// Calibrates a static student sampling temperature within `[min_student_temperature, max_student_temperature]`
/// to align quantized student distributions with BF16 teacher logits (`TempQuant`, `arXiv:2611.22128`).
///
/// Minimizes the convex weighted forward KL `KL(p_teacher(T_t) || q_student(1 / beta))` over
/// inverse temperature `beta` via deterministic 96-step golden-section search.
///
/// # Errors
/// Returns an error on empty or mismatched slices, token counts exceeding `u32::MAX`, non-positive or
/// inverted temperature bounds, or non-finite logits.
pub fn fit_tempquant_temperature<T: AsRef<[f64]>>(
    teacher_logits: &[T],
    student_logits: &[T],
    teacher_temperature: f64,
    min_student_temperature: f64,
    max_student_temperature: f64,
    fisher_weighted: bool,
) -> Result<TempQuantCalibrationSummary> {
    if teacher_logits.is_empty() || teacher_logits.len() != student_logits.len() {
        return Err(format!(
            "TempQuant requires matching non-empty logit rows (teacher={}, student={}).",
            teacher_logits.len(),
            student_logits.len()
        ));
    }
    let tokens = u32::try_from(teacher_logits.len())
        .map_err(|_| "TempQuant token count exceeds u32::MAX.".to_owned())?;
    validate_positive_finite_temperature(teacher_temperature, "teacher_temperature")?;
    validate_positive_finite_temperature(min_student_temperature, "min_student_temperature")?;
    validate_positive_finite_temperature(max_student_temperature, "max_student_temperature")?;
    if min_student_temperature > max_student_temperature {
        return Err(format!(
            "min_student_temperature ({min_student_temperature}) cannot exceed max_student_temperature ({max_student_temperature})."
        ));
    }

    let inv_t_teacher = 1.0 / teacher_temperature;
    let mut teacher_scaled = Vec::with_capacity(teacher_logits.len());
    let mut teacher_log_probs = Vec::with_capacity(teacher_logits.len());
    let mut teacher_probs = Vec::with_capacity(teacher_logits.len());
    let mut teacher_entropy_sum = 0.0_f64;

    for (t_row, s_row) in teacher_logits.iter().zip(student_logits) {
        let t_slice = t_row.as_ref();
        let s_slice = s_row.as_ref();
        if t_slice.is_empty() || t_slice.len() != s_slice.len() {
            return Err(format!(
                "TempQuant vocabulary mismatch: teacher={}, student={}.",
                t_slice.len(),
                s_slice.len()
            ));
        }
        let scaled_t = scale_logits_by_inverse_temperature(t_slice, inv_t_teacher)?;
        let log_p = log_softmax_f64(&scaled_t)?;
        let probs: Vec<f64> = log_p.iter().map(|&lp| lp.exp()).collect();
        let ent = probs
            .iter()
            .zip(&log_p)
            .map(|(&p, &lp)| if p > 0.0 { -(p * lp) } else { 0.0 })
            .sum::<f64>()
            .max(0.0);
        teacher_entropy_sum += ent;
        teacher_scaled.push(scaled_t);
        teacher_log_probs.push(log_p);
        teacher_probs.push(probs);
    }
    let mean_teacher_entropy = (teacher_entropy_sum / f64::from(tokens)).max(0.0);
    let weights = normalized_teacher_weights(&teacher_scaled, fisher_weighted)?;

    let (baseline_kl, mean_student_entropy) = evaluate_tempquant_beta(
        &teacher_log_probs,
        &teacher_probs,
        student_logits,
        &weights,
        inv_t_teacher,
    )?;

    let beta_lo = 1.0 / max_student_temperature;
    let beta_hi = 1.0 / min_student_temperature;
    validate_positive_finite_temperature(beta_lo, "inverse max_student_temperature")?;
    validate_positive_finite_temperature(beta_hi, "inverse min_student_temperature")?;

    let (best_beta, calibrated_kl, mean_calibrated_entropy) =
        if (beta_hi - beta_lo).abs() <= f64::EPSILON {
            let (kl, ent) = evaluate_tempquant_beta(
                &teacher_log_probs,
                &teacher_probs,
                student_logits,
                &weights,
                beta_lo,
            )?;
            (beta_lo, kl, ent)
        } else {
            let mut a = beta_lo;
            let mut b = beta_hi;
            let mut x1 = b - GOLDEN_RATIO_CONJUGATE * (b - a);
            let mut x2 = a + GOLDEN_RATIO_CONJUGATE * (b - a);
            let mut f1 = evaluate_tempquant_beta(
                &teacher_log_probs,
                &teacher_probs,
                student_logits,
                &weights,
                x1,
            )?
            .0;
            let mut f2 = evaluate_tempquant_beta(
                &teacher_log_probs,
                &teacher_probs,
                student_logits,
                &weights,
                x2,
            )?
            .0;

            for _ in 0..96 {
                if f1 <= f2 {
                    b = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = b - GOLDEN_RATIO_CONJUGATE * (b - a);
                    f1 = evaluate_tempquant_beta(
                        &teacher_log_probs,
                        &teacher_probs,
                        student_logits,
                        &weights,
                        x1,
                    )?
                    .0;
                } else {
                    a = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = a + GOLDEN_RATIO_CONJUGATE * (b - a);
                    f2 = evaluate_tempquant_beta(
                        &teacher_log_probs,
                        &teacher_probs,
                        student_logits,
                        &weights,
                        x2,
                    )?
                    .0;
                }
            }

            let mut candidates = vec![0.5 * (a + b), beta_lo, beta_hi, 0.5 * (beta_lo + beta_hi)];
            if (min_student_temperature..=max_student_temperature).contains(&teacher_temperature) {
                candidates.push(inv_t_teacher);
            }
            let mut best = (candidates[0], f64::INFINITY, 0.0_f64);
            for cand_beta in candidates {
                let (kl, ent) = evaluate_tempquant_beta(
                    &teacher_log_probs,
                    &teacher_probs,
                    student_logits,
                    &weights,
                    cand_beta,
                )?;
                if kl < best.1 {
                    best = (cand_beta, kl, ent);
                }
            }
            best
        };

    let fitted_student_temperature =
        (1.0 / best_beta).clamp(min_student_temperature, max_student_temperature);
    Ok(TempQuantCalibrationSummary {
        tokens,
        teacher_temperature,
        fitted_student_temperature,
        baseline_kl,
        calibrated_kl,
        kl_reduction: (baseline_kl - calibrated_kl).max(0.0),
        mean_teacher_entropy,
        mean_student_entropy,
        mean_calibrated_entropy,
        fisher_weighted,
    })
}

fn top_k_sorted_probs(logits: &[f64], temperature: f64, top_k: usize) -> Result<Vec<(usize, f64)>> {
    let scaled = scale_logits_by_inverse_temperature(logits, 1.0 / temperature)?;
    let log_p = log_softmax_f64(&scaled)?;
    let mut ranked: Vec<(usize, f64)> = log_p
        .into_iter()
        .enumerate()
        .map(|(idx, lp)| (idx, lp.exp()))
        .collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(top_k.min(ranked.len()));
    Ok(ranked)
}

fn nucleus_truncate(sorted_top_k: &[(usize, f64)], top_p: f64) -> Vec<(usize, f64)> {
    let mut kept = Vec::new();
    let mut cum = 0.0_f64;
    for &(idx, prob) in sorted_top_k {
        kept.push((idx, prob));
        cum += prob;
        if cum >= top_p {
            break;
        }
    }
    let kept_sum: f64 = kept.iter().map(|&(_, prob)| prob).sum();
    if kept_sum > 0.0 {
        for &mut (_, ref mut prob) in &mut kept {
            *prob /= kept_sum;
        }
    } else if !kept.is_empty() {
        let uniform = 1.0 / f64::from(u32::try_from(kept.len()).unwrap_or(1));
        for &mut (_, ref mut prob) in &mut kept {
            *prob = uniform;
        }
    }
    kept
}

fn smoothed_nucleus_kl(teacher_kept: &[(usize, f64)], student_kept: &[(usize, f64)]) -> f64 {
    let mut union = BTreeSet::new();
    let mut t_map = BTreeMap::new();
    let mut s_map = BTreeMap::new();
    for &(idx, prob) in teacher_kept {
        union.insert(idx);
        t_map.insert(idx, prob);
    }
    for &(idx, prob) in student_kept {
        union.insert(idx);
        s_map.insert(idx, prob);
    }
    let mut t_sum = 0.0_f64;
    let mut s_sum = 0.0_f64;
    let mut pairs = Vec::with_capacity(union.len());
    for idx in union {
        let tp = t_map.get(&idx).copied().unwrap_or(0.0) + NUCLEUS_KL_FLOOR;
        let sp = s_map.get(&idx).copied().unwrap_or(0.0) + NUCLEUS_KL_FLOOR;
        t_sum += tp;
        s_sum += sp;
        pairs.push((tp, sp));
    }
    let mut kl = 0.0_f64;
    for (tp_raw, sp_raw) in pairs {
        let tp = tp_raw / t_sum;
        let sp = sp_raw / s_sum;
        kl += tp * (tp.ln() - sp.ln());
    }
    kl.max(0.0)
}

/// Calibrates student nucleus threshold `top_p` in `[0.05, 1.00]` under a fixed `top_k` and
/// `student_temperature` to match teacher truncated distributions (`TempQuant`, `arXiv:2611.22128`).
///
/// Always retains at least 1 token per step (`min_keep = 1`), and scores mismatched support cleanly
/// using a `1e-15` probability floor over the union of active teacher and student nucleus indices.
///
/// # Errors
/// Returns an error on invalid bounds, empty/mismatched slices, or non-finite logits.
pub fn fit_tempquant_top_p<T: AsRef<[f64]>>(
    teacher_logits: &[T],
    student_logits: &[T],
    teacher_temperature: f64,
    teacher_top_p: f64,
    student_temperature: f64,
    top_k: usize,
) -> Result<TempQuantNucleusSummary> {
    if teacher_logits.is_empty() || teacher_logits.len() != student_logits.len() {
        return Err(format!(
            "TempQuant nucleus fitting requires matching non-empty logit rows (teacher={}, student={}).",
            teacher_logits.len(),
            student_logits.len()
        ));
    }
    let tokens = u32::try_from(teacher_logits.len())
        .map_err(|_| "TempQuant token count exceeds u32::MAX.".to_owned())?;
    if top_k == 0 {
        return Err("TempQuant top_k must be at least 1.".to_owned());
    }
    validate_positive_finite_temperature(teacher_temperature, "teacher_temperature")?;
    validate_positive_finite_temperature(student_temperature, "student_temperature")?;
    if !teacher_top_p.is_finite() || teacher_top_p <= 0.0 || teacher_top_p > 1.0 {
        return Err(format!(
            "teacher_top_p must be finite and in (0.0, 1.0], got {teacher_top_p}."
        ));
    }

    let mut teacher_nuclei = Vec::with_capacity(teacher_logits.len());
    let mut student_ranked = Vec::with_capacity(student_logits.len());
    let mut teacher_active_total = 0_usize;

    for (t_row, s_row) in teacher_logits.iter().zip(student_logits) {
        let t_slice = t_row.as_ref();
        let s_slice = s_row.as_ref();
        if t_slice.is_empty() || t_slice.len() != s_slice.len() {
            return Err(format!(
                "TempQuant vocabulary mismatch: teacher={}, student={}.",
                t_slice.len(),
                s_slice.len()
            ));
        }
        let t_top_k = top_k_sorted_probs(t_slice, teacher_temperature, top_k)?;
        let s_top_k = top_k_sorted_probs(s_slice, student_temperature, top_k)?;
        let t_kept = nucleus_truncate(&t_top_k, teacher_top_p);
        teacher_active_total = teacher_active_total.saturating_add(t_kept.len());
        teacher_nuclei.push(t_kept);
        student_ranked.push(s_top_k);
    }

    let eval_top_p = |candidate: f64| -> Result<(f64, f64)> {
        let mut kl_sum = 0.0_f64;
        let mut active_sum = 0_usize;
        for (t_kept, s_top_k) in teacher_nuclei.iter().zip(&student_ranked) {
            let s_kept = nucleus_truncate(s_top_k, candidate);
            active_sum = active_sum.saturating_add(s_kept.len());
            kl_sum += smoothed_nucleus_kl(t_kept, &s_kept);
        }
        Ok((
            (kl_sum / f64::from(tokens)).max(0.0),
            count_as_f64(active_sum)? / f64::from(tokens),
        ))
    };

    let (baseline_truncated_kl, mut best_active_mean) = eval_top_p(teacher_top_p)?;
    let mut best_top_p = teacher_top_p;
    let mut best_kl = baseline_truncated_kl;

    for hundredth in 5_u16..=100 {
        let cand = f64::from(hundredth) / 100.0;
        let (kl, active_mean) = eval_top_p(cand)?;
        let strictly_better = kl < best_kl - 1e-12;
        let close_tie = (kl - best_kl).abs() <= 1e-12
            && (cand - teacher_top_p).abs() < (best_top_p - teacher_top_p).abs();
        if strictly_better || close_tie {
            best_top_p = cand;
            best_kl = kl;
            best_active_mean = active_mean;
        }
    }

    Ok(TempQuantNucleusSummary {
        tokens,
        top_k,
        teacher_temperature,
        teacher_top_p,
        student_temperature,
        fitted_student_top_p: best_top_p,
        baseline_truncated_kl,
        calibrated_truncated_kl: best_kl,
        mean_teacher_active_tokens: count_as_f64(teacher_active_total)? / f64::from(tokens),
        mean_student_active_tokens: best_active_mean,
    })
}

/// Computes `DeepConf` (`arXiv:2508.15260`) per-token confidence `-(1/k) * sum_{j=1}^k log P(j)`
/// over the top `k` logits in `f64`.
///
/// Higher values denote sharper distributions (greater model confidence relative to the top-k tail).
///
/// # Errors
/// Returns an error if `top_k == 0`, `logits` is empty, or `logits` contains non-finite values.
pub fn deepconf_token_confidence_from_logits(logits: &[f64], top_k: usize) -> Result<f64> {
    if top_k == 0 {
        return Err("DeepConf top_k must be at least 1.".to_owned());
    }
    let mut log_p = log_softmax_f64(logits)?;
    let k = top_k.min(log_p.len());
    if k < log_p.len() {
        let _ = log_p.select_nth_unstable_by(k - 1, |left, right| right.total_cmp(left));
    }
    let sum_top_k: f64 = log_p[..k].iter().sum();
    let confidence = (-sum_top_k / count_as_f64(k)?).max(0.0);
    if !confidence.is_finite() {
        return Err("DeepConf token confidence overflowed finite bounds.".to_owned());
    }
    Ok(confidence)
}

/// Bounded streaming sliding-window confidence tracker for `DeepConf` (`arXiv:2508.15260`).
#[derive(Clone, Debug)]
pub struct DeepConfWindowTracker {
    window_size: usize,
    ring: Vec<f64>,
    cursor: usize,
    count: usize,
    window_sum: f64,
    lowest_full_window: Option<f64>,
}

impl DeepConfWindowTracker {
    /// Creates a new tracker with sliding window length `window_size` in `1..=65_536`.
    ///
    /// # Errors
    /// Rejects zero or oversized window sizes.
    pub fn new(window_size: usize) -> Result<Self> {
        if window_size == 0 || window_size > MAX_DEEPCONF_WINDOW {
            return Err(format!(
                "DeepConf window_size must be in 1..={MAX_DEEPCONF_WINDOW}, got {window_size}."
            ));
        }
        Ok(Self {
            window_size,
            ring: Vec::with_capacity(window_size.min(256)),
            cursor: 0,
            count: 0,
            window_sum: 0.0,
            lowest_full_window: None,
        })
    }

    /// Pushes one finite non-negative token confidence observation and returns the updated window mean.
    ///
    /// # Errors
    /// Rejects non-finite or negative confidence values, or non-finite running totals.
    pub fn push(&mut self, token_confidence: f64) -> Result<f64> {
        if !token_confidence.is_finite() || token_confidence < 0.0 {
            return Err(format!(
                "DeepConf token_confidence must be finite and non-negative, got {token_confidence}."
            ));
        }
        if self.ring.len() < self.window_size {
            self.ring.push(token_confidence);
            self.window_sum += token_confidence;
        } else {
            let evicted = self.ring[self.cursor];
            self.ring[self.cursor] = token_confidence;
            self.cursor = (self.cursor + 1) % self.window_size;
            if self.cursor == 0 {
                // Re-sum periodically on ring wrap to prevent accumulated cancellation drift.
                self.window_sum = self.ring.iter().sum();
            } else {
                self.window_sum = (self.window_sum - evicted + token_confidence).max(0.0);
            }
        }
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| "DeepConf pushed token count overflowed usize.".to_owned())?;
        if !self.window_sum.is_finite() {
            return Err(
                "DeepConf sliding-window confidence sum overflowed finite bounds.".to_owned(),
            );
        }
        let mean = (self.window_sum / count_as_f64(self.ring.len())?).max(0.0);
        if self.count >= self.window_size {
            self.lowest_full_window =
                Some(self.lowest_full_window.map_or(mean, |prev| prev.min(mean)));
        }
        Ok(mean)
    }

    /// Total number of token confidence values pushed into this trace so far.
    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.count
    }

    /// Mean confidence of the active trailing window, or `None` before any token is pushed.
    #[must_use]
    pub fn current_window_confidence(&self) -> Option<f64> {
        if self.ring.is_empty() {
            return None;
        }
        let denom = u32::try_from(self.ring.len()).ok()?;
        Some((self.window_sum / f64::from(denom)).max(0.0))
    }

    /// Lowest full-window group confidence seen so far (falling back to current window for short traces).
    #[must_use]
    pub fn lowest_group_confidence(&self) -> Option<f64> {
        self.lowest_full_window
            .or_else(|| self.current_window_confidence())
    }

    /// Trailing tail confidence equal to the most recent window mean.
    #[must_use]
    pub fn tail_confidence(&self) -> Option<f64> {
        self.current_window_confidence()
    }

    /// Returns `true` when at least one full window has formed and its mean falls below `threshold`.
    #[must_use]
    pub fn should_stop_online(&self, threshold: f64) -> bool {
        threshold.is_finite()
            && self.count >= self.window_size
            && self
                .current_window_confidence()
                .is_some_and(|mean| mean < threshold)
    }
}

fn ceil_fraction_count(total: usize, fraction: f64) -> Result<usize> {
    let target = count_as_f64(total)? * fraction;
    let candidates: Vec<usize> = (1..=total).collect();
    let idx =
        candidates.partition_point(|&k| f64::from(u32::try_from(k).unwrap_or(u32::MAX)) < target);
    Ok(candidates.get(idx).copied().unwrap_or(total))
}

/// Computes the `DeepConf` online warmup stopping threshold `s_tau` keeping `keep_fraction`
/// of warmup lowest-group confidences (`arXiv:2508.15260`, top-eta quantile boundary).
///
/// # Errors
/// Returns an error if `warmup_lowest_confidences` is empty, contains negative/non-finite entries,
/// or `keep_fraction` is outside `(0.0, 1.0]`.
pub fn deepconf_warmup_threshold(
    warmup_lowest_confidences: &[f64],
    keep_fraction: f64,
) -> Result<f64> {
    if warmup_lowest_confidences.is_empty() {
        return Err("DeepConf warmup requires at least one confidence observation.".to_owned());
    }
    if !keep_fraction.is_finite() || keep_fraction <= 0.0 || keep_fraction > 1.0 {
        return Err(format!(
            "DeepConf keep_fraction must be finite and in (0.0, 1.0], got {keep_fraction}."
        ));
    }
    let mut sorted = Vec::with_capacity(warmup_lowest_confidences.len());
    for &c in warmup_lowest_confidences {
        if !c.is_finite() || c < 0.0 {
            return Err(format!(
                "DeepConf warmup confidence must be finite and non-negative, got {c}."
            ));
        }
        sorted.push(c);
    }
    sorted.sort_by(f64::total_cmp);
    let keep_count = ceil_fraction_count(sorted.len(), keep_fraction)?;
    Ok(sorted[sorted.len() - keep_count])
}

/// Filters candidate traces to the top `keep_fraction` by trace confidence and returns the
/// confidence-weighted majority answer plus its normalized vote share in `[0.0, 1.0]`.
///
/// Returns `Ok(None)` when `candidates` is empty. Falls back to equal `1.0` weights if all retained
/// candidates have zero confidence. Ties in accumulated vote weight resolve deterministically to the
/// lexicographically smaller answer string.
///
/// # Errors
/// Returns an error on invalid `keep_fraction` or negative/non-finite confidence entries.
pub fn deepconf_weighted_vote(
    candidates: &[(&str, f64)],
    keep_fraction: f64,
) -> Result<Option<(String, f64)>> {
    if !keep_fraction.is_finite() || keep_fraction <= 0.0 || keep_fraction > 1.0 {
        return Err(format!(
            "DeepConf keep_fraction must be finite and in (0.0, 1.0], got {keep_fraction}."
        ));
    }
    for &(_, confidence) in candidates {
        if !confidence.is_finite() || confidence < 0.0 {
            return Err(format!(
                "DeepConf candidate confidence must be finite and non-negative, got {confidence}."
            ));
        }
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&left, &right| {
        candidates[right]
            .1
            .total_cmp(&candidates[left].1)
            .then_with(|| left.cmp(&right))
    });
    let keep_count = ceil_fraction_count(candidates.len(), keep_fraction)?;
    let kept = &order[..keep_count];
    let nonzero_sum: f64 = kept.iter().map(|&idx| candidates[idx].1).sum();
    let use_uniform = nonzero_sum == 0.0;

    let mut totals: BTreeMap<&str, f64> = BTreeMap::new();
    let mut total_weight = 0.0_f64;
    for &idx in kept {
        let (answer, conf) = candidates[idx];
        let weight = if use_uniform { 1.0 } else { conf };
        *totals.entry(answer).or_insert(0.0) += weight;
        total_weight += weight;
    }
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(
            "DeepConf weighted vote total weight was non-finite or non-positive.".to_owned(),
        );
    }
    let mut best_answer = "";
    let mut best_weight = f64::NEG_INFINITY;
    for (answer, weight) in totals {
        if weight > best_weight {
            best_weight = weight;
            best_answer = answer;
        }
    }
    Ok(Some((
        best_answer.to_owned(),
        (best_weight / total_weight).clamp(0.0, 1.0),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_divergence_and_softmax_fisher_handle_identical_shifted_and_peaked_logits() -> Result<()> {
        let teacher = vec![2.0, 0.5, -1.0, 0.0];
        let shifted = teacher.iter().map(|v| v + 42.0).collect::<Vec<_>>();
        assert!(kl_divergence_from_logits(&teacher, &shifted)? <= 1e-12);

        let student = vec![0.0, 1.5, 0.5, -0.5];
        let kl = kl_divergence_from_logits(&teacher, &student)?;
        assert!(kl > 0.4);

        let fisher_ambiguous = softmax_fisher_from_logits(&[0.0, 0.0, 0.0, 0.0])?;
        assert!((fisher_ambiguous - 0.75).abs() <= 1e-12);

        let fisher_peaked = softmax_fisher_from_logits(&[50.0, 0.0, 0.0, 0.0])?;
        assert!(fisher_peaked <= 1e-12);

        let fisher_near_onehot = softmax_fisher_from_logits(&[30.0, 0.0])?;
        assert!(fisher_near_onehot > 0.0 && fisher_near_onehot < 1e-11);
        Ok(())
    }

    #[test]
    fn per_site_kl_gap_weights_by_teacher_softmax_fisher() -> Result<()> {
        // Token 0 is near-deterministic (tiny Fisher weight ~ 0); token 1 is ambiguous (Fisher = 0.5).
        let teacher = vec![vec![40.0, 0.0], vec![1.0, 1.0]];
        // Base is slightly off on token 0, badly off on token 1; recovered fixes token 1.
        let base = vec![vec![20.0, 0.0], vec![4.0, 0.0]];
        let recovered = vec![vec![20.0, 0.0], vec![1.0, 1.0]];
        let (base_kl, rec_kl, gap) =
            per_site_fisher_weighted_kl_metrics(&teacher, &base, &recovered)?;
        assert!(base_kl > 1.0);
        assert!(rec_kl <= 1e-9);
        assert!((gap - (base_kl - rec_kl)).abs() <= 1e-12);
        assert!((per_site_kl_gap(&teacher, &base, &recovered)? - gap).abs() <= 1e-12);
        Ok(())
    }

    #[test]
    fn multiple_choice_knapsack_solves_exact_gcd_scaled_max_and_min() -> Result<()> {
        let groups = vec![
            vec![
                KnapsackOption {
                    cost: 0,
                    value: 0.0,
                },
                KnapsackOption {
                    cost: 2_048,
                    value: 3.0,
                },
                KnapsackOption {
                    cost: 4_096,
                    value: 8.0,
                },
            ],
            vec![
                KnapsackOption {
                    cost: 2_048,
                    value: 1.0,
                },
                KnapsackOption {
                    cost: 6_144,
                    value: 10.0,
                },
            ],
        ];
        // Total budget = 8_192 -> group 0 min 0 + group 1 min 2_048 => 2 choices:
        // (cost 4_096 + 2_048 = 6_144 -> val 9.0) OR (cost 2_048 + 6_144 = 8_192 -> val 13.0)
        // Wait: (4_096 + 2_048 = 6_144, val 9) vs (2_048 + 6_144 = 8_192, val 13) -> opt [1, 1]!
        let chosen_max = solve_multiple_choice_knapsack_maximize(&groups, 8_192)?;
        assert_eq!(chosen_max, vec![1, 1]);

        let min_groups = vec![
            vec![
                KnapsackOption {
                    cost: 0,
                    value: 10.0,
                },
                KnapsackOption {
                    cost: 4_096,
                    value: 2.0,
                },
            ],
            vec![
                KnapsackOption {
                    cost: 0,
                    value: 8.0,
                },
                KnapsackOption {
                    cost: 4_096,
                    value: 3.0,
                },
            ],
        ];
        let chosen_min = solve_multiple_choice_knapsack_minimize(&min_groups, 4_096)?;
        assert_eq!(chosen_min, vec![1, 0]);
        assert!(solve_multiple_choice_knapsack_maximize(&groups, 1_024).is_err());
        Ok(())
    }

    #[test]
    fn scaled_l2_handles_extreme_exponents_and_subnormals_without_overflow() -> Result<()> {
        let huge = [1e305_f64, 1e305_f64];
        let norm = scaled_l2_norm(&huge)?;
        assert!((norm - 2.0_f64.sqrt() * 1e305).abs() / (2.0_f64.sqrt() * 1e305) < 1e-14);

        let tiny = [1e-305_f64, 1e-305_f64];
        let tiny_norm = scaled_l2_norm(&tiny)?;
        assert!((tiny_norm - 2.0_f64.sqrt() * 1e-305).abs() / (2.0_f64.sqrt() * 1e-305) < 1e-14);

        assert_eq!(scaled_l2_norm(&[])?, 0.0);
        assert_eq!(scaled_l2_norm(&[0.0, -0.0])?, 0.0);
        assert!(scaled_l2_norm(&[f64::NAN]).is_err());
        assert!(scaled_l2_norm(&[f64::INFINITY]).is_err());
        Ok(())
    }

    #[test]
    fn cosine_similarity_is_scale_invariant_and_clamped() -> Result<()> {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0])?, None);
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[0.0, 0.0])?, None);

        let aligned = cosine_similarity(&[3e200, 4e200], &[6e-200, 8e-200])?
            .ok_or("Expected defined cosine similarity.")?;
        assert!((-1.0..=1.0).contains(&aligned));
        assert!((aligned - 1.0).abs() < 1e-15);

        let opposite = cosine_similarity(&[1.0, -2.0, 3.0], &[-2.0, 4.0, -6.0])?
            .ok_or("Expected defined opposite cosine.")?;
        assert!((-1.0..=1.0).contains(&opposite));
        assert!((opposite + 1.0).abs() < 1e-15);

        let orthogonal =
            cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])?.ok_or("Expected orthogonal cosine.")?;
        assert!(orthogonal.abs() < 1e-15);
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_err());
        Ok(())
    }

    #[test]
    fn energy_entropy_coverage_and_worst_row_relative_error_respect_bounds() -> Result<()> {
        assert_eq!(normalized_energy_entropy(&[0.0, 0.0, 0.0])?, 0.0);
        assert_eq!(normalized_energy_entropy(&[5.0, 0.0, 0.0, 0.0])?, 0.0);
        let uniform = normalized_energy_entropy(&[2.0, -2.0, 2.0, 2.0])?;
        assert!((0.0..=1.0).contains(&uniform));
        assert!((uniform - 1.0).abs() < 1e-15);

        assert_eq!(coordinate_coverage(&[])?, 0.0);
        assert_eq!(coordinate_coverage(&[0.0, -0.0, 1.0, -3.0])?, 0.5);

        let ref_a = [3.0, 4.0];
        let cand_a = [3.0, 4.0];
        let ref_b = [1.0, 0.0];
        let cand_b = [0.0, 0.0];
        let worst = worst_row_relative_l2_error(&[(&ref_a, &cand_a), (&ref_b, &cand_b)])?
            .ok_or("Expected worst-row error.")?;
        assert!((worst - 1.0).abs() < 1e-15);

        let mut acc = RowMetricsAccumulator::new();
        acc.record_pair(&ref_a, &cand_a)?;
        acc.record_pair(&ref_b, &cand_b)?;
        let summary = acc.finish();
        assert_eq!(summary.rows, 2);
        assert_eq!(summary.worst_row_relative_l2_error, Some(1.0));
        assert_eq!(summary.min_cosine_similarity, Some(1.0));
        assert_eq!(summary.mean_cosine_similarity, Some(1.0));
        assert_eq!(summary.mean_candidate_coordinate_coverage, Some(0.5));
        Ok(())
    }

    #[test]
    fn tempquant_recovers_exact_temperature_and_restores_teacher_entropy() -> Result<()> {
        let teacher = [
            vec![3.2_f64, 1.4, -0.5, -2.0],
            vec![0.8, 0.7, -0.1, -1.2],
            vec![5.0, 0.2, -1.0, -3.0],
        ];
        // Quantized student with logits cooled by 1/0.7 relative to teacher plus constant shift.
        let student: Vec<Vec<f64>> = teacher
            .iter()
            .map(|row| row.iter().map(|&v| 0.7 * v + 3.5).collect())
            .collect();

        let unweighted = fit_tempquant_temperature(&teacher, &student, 1.0, 0.25, 2.0, false)?;
        assert_eq!(unweighted.tokens, 3);
        assert!(!unweighted.fisher_weighted);
        assert!((unweighted.fitted_student_temperature - 0.7).abs() < 1e-6);
        assert!(unweighted.baseline_kl > 0.02);
        assert!(unweighted.calibrated_kl < 1e-12);
        assert!(unweighted.kl_reduction > 0.02);
        assert!(
            (unweighted.mean_calibrated_entropy - unweighted.mean_teacher_entropy).abs() < 1e-6
        );
        assert!(
            (unweighted.mean_teacher_entropy - shannon_entropy_from_logits(&teacher[0])?).abs()
                < 2.0
        );

        let weighted = fit_tempquant_temperature(&teacher, &student, 1.0, 0.25, 2.0, true)?;
        assert!(weighted.fisher_weighted);
        assert!((weighted.fitted_student_temperature - 0.7).abs() < 1e-6);
        assert!(weighted.calibrated_kl <= weighted.baseline_kl);

        // Matched logits must preserve T_teacher without regression.
        let matched = fit_tempquant_temperature(&teacher, &teacher, 1.25, 0.25, 2.0, true)?;
        assert!(matched.calibrated_kl <= matched.baseline_kl);
        assert!(matched.calibrated_kl < 1e-12);

        // Temperature-scaled KL helper validates bounds cleanly.
        assert!(temperature_scaled_kl_from_logits(&teacher[0], &student[0], 1.0, 0.7)? < 1e-12);
        assert!(temperature_scaled_kl_from_logits(&teacher[0], &student[0], 0.0, 1.0).is_err());
        assert!(fit_tempquant_temperature(&teacher, &student, 1.0, 1.5, 0.5, false).is_err());
        assert!(fit_tempquant_temperature(&teacher[..2], &student, 1.0, 0.5, 2.0, false).is_err());
        Ok(())
    }

    #[test]
    fn tempquant_nucleus_calibration_reduces_truncated_kl_and_rejects_invalid_inputs() -> Result<()>
    {
        // Sharpened student hits top_p=0.95 with too few active tokens unless student_top_p is raised.
        let teacher = [vec![2.0_f64, 1.35, 0.8, -4.0], vec![1.9, 1.4, 0.7, -4.0]];
        let student: Vec<Vec<f64>> = teacher
            .iter()
            .map(|row| row.iter().map(|&v| v * 1.4).collect())
            .collect();
        let summary = fit_tempquant_top_p(&teacher, &student, 1.0, 0.90, 1.0, 4)?;
        assert_eq!(summary.tokens, 2);
        assert_eq!(summary.top_k, 4);
        assert!(summary.calibrated_truncated_kl <= summary.baseline_truncated_kl);
        assert!(summary.mean_student_active_tokens >= 1.0);

        assert!(fit_tempquant_top_p(&teacher, &student, 1.0, 0.0, 1.0, 4).is_err());
        assert!(fit_tempquant_top_p(&teacher, &student, 1.0, 1.1, 1.0, 4).is_err());
        assert!(fit_tempquant_top_p(&teacher, &student, 1.0, 0.9, 1.0, 0).is_err());
        Ok(())
    }

    #[test]
    fn deepconf_tracker_detects_low_confidence_dip_and_votes_by_confidence() -> Result<()> {
        let peaked = [8.0_f64, 0.0, -4.0, -6.0];
        let flat = [0.1_f64, 0.0, -0.1, -0.2];
        let c_peaked = deepconf_token_confidence_from_logits(&peaked, 4)?;
        let c_flat = deepconf_token_confidence_from_logits(&flat, 4)?;
        assert!(c_peaked > c_flat);
        assert!(deepconf_token_confidence_from_logits(&peaked, 0).is_err());

        let warmup = [2.4_f64, 3.1, 2.8, 1.9, 3.4];
        // top 40% of 5 = ceil(2.0) = 2 -> sorted = [1.9, 2.4, 2.8, 3.1, 3.4] -> index 3 = 3.1
        assert!((deepconf_warmup_threshold(&warmup, 0.4)? - 3.1).abs() < 1e-12);
        let s_tau = 2.5;

        let mut tracker = DeepConfWindowTracker::new(3)?;
        assert_eq!(tracker.current_window_confidence(), None);
        assert_eq!(tracker.lowest_group_confidence(), None);

        // Before window_size=3 tokens, even low values do not fire online early stop.
        assert!((tracker.push(1.0)? - 1.0).abs() < 1e-12);
        assert!(!tracker.should_stop_online(s_tau));
        tracker.push(4.0)?;
        assert!(!tracker.should_stop_online(s_tau));
        // 3rd push forms first full window: (1 + 4 + 4)/3 = 3.0 >= 2.5 -> keeps going.
        assert!((tracker.push(4.0)? - 3.0).abs() < 1e-12);
        assert!(!tracker.should_stop_online(s_tau));
        // Push low-confidence dip so trailing window drops to (4 + 0.5 + 0.5)/3 = 1.666... < 2.5.
        tracker.push(0.5)?;
        let trailing = tracker.push(0.5)?;
        assert!((trailing - (5.0 / 3.0)).abs() < 1e-12);
        assert!(tracker.should_stop_online(s_tau));
        assert_eq!(tracker.token_count(), 5);
        assert!((tracker.lowest_group_confidence().unwrap_or(0.0) - trailing).abs() < 1e-12);
        assert_eq!(tracker.tail_confidence(), Some(trailing));

        // Weighted consensus: 3 low-confidence wrong traces vs 2 high-confidence correct traces.
        let candidates = [
            ("wrong", 0.5),
            ("wrong", 0.6),
            ("right", 4.5),
            ("wrong", 0.7),
            ("right", 3.5),
        ];
        let (winner, share) =
            deepconf_weighted_vote(&candidates, 0.6)?.ok_or("Expected non-empty vote result.")?;
        assert_eq!(winner, "right");
        assert!((share - (8.0 / 8.7)).abs() < 1e-12);

        assert_eq!(deepconf_weighted_vote(&[], 0.5)?, None);
        assert!(DeepConfWindowTracker::new(0).is_err());
        assert!(deepconf_warmup_threshold(&[], 0.5).is_err());
        assert!(deepconf_weighted_vote(&[("a", -0.1)], 0.5).is_err());
        Ok(())
    }
}
