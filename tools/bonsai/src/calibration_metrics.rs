//! Scaled floating-point calibration, directional alignment and dispersion metrics.

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
            return Err(format!("KL value must be finite and non-negative, got {kl}"));
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
    for ((t, b), r) in teacher_logits
        .iter()
        .zip(base_logits)
        .zip(recovered_logits)
    {
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
    let (_, _, gap) = per_site_fisher_weighted_kl_metrics(teacher_logits, base_logits, recovered_logits)?;
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
            return Err(format!("knapsack group {g_idx} must contain at least one option"));
        }
        if group.len() > usize::from(u16::MAX) {
            return Err(format!("knapsack group {g_idx} exceeds u16 option limit"));
        }
        let mut group_min = u64::MAX;
        for opt in group {
            if !opt.value.is_finite() {
                return Err(format!("knapsack option value in group {g_idx} must be finite"));
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
                step_gcd = if step_gcd == 0 { rel } else { u64_gcd(step_gcd, rel) };
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
        let (base_kl, rec_kl, gap) = per_site_fisher_weighted_kl_metrics(&teacher, &base, &recovered)?;
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
}
