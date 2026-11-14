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

#[cfg(test)]
mod tests {
    use super::*;

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
