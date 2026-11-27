//! Deterministic exact and randomized SVD spectrum estimators for BF16 weight diffs.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

/// Singular value shrinkage policy applied to above-bulk spikes during factor extraction.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ShrinkageMode {
    #[default]
    GavishDonoho,
    Unshrunk,
    Blended,
}

impl ShrinkageMode {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::GavishDonoho => "gavish-donoho",
            Self::Unshrunk => "unshrunk",
            Self::Blended => "blended",
        }
    }
}

/// Cumulative Frobenius energy `sum_{i < r} sigma_i^2 / ||Delta||_F^2` at standard ranks.
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CumulativeEnergy {
    pub(super) r1: f64,
    pub(super) r4: f64,
    pub(super) r8: f64,
    pub(super) r16: f64,
    pub(super) r32: f64,
    pub(super) r64: f64,
}

impl CumulativeEnergy {
    fn from_eigenvalues(eigenvalues: &[f64], total_energy: f64) -> Self {
        if total_energy <= 0.0 || !total_energy.is_finite() {
            return Self::default();
        }
        let at_rank = |rank: usize| -> f64 {
            let sum: f64 = eigenvalues.iter().take(rank).sum();
            (sum / total_energy).clamp(0.0, 1.0)
        };
        Self {
            r1: at_rank(1),
            r4: at_rank(4),
            r8: at_rank(8),
            r16: at_rank(16),
            r32: at_rank(32),
            r64: at_rank(64),
        }
    }
}

/// Spectral denoising summary via Rademacher sign-flip null bulk calibration
/// and Gavish–Donoho (2017) Frobenius-optimal singular value shrinkage.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct DenoisedSpectrum {
    pub(super) rademacher_null_bulk_edge: Option<f64>,
    pub(super) isotropic_mp_bulk_edge: f64,
    pub(super) effective_bulk_edge: f64,
    pub(super) aspect_ratio_beta: f64,
    pub(super) above_bulk_rank: usize,
    pub(super) tail_saturated: bool,
    pub(super) sigma_max_to_bulk_edge_ratio: f64,
    pub(super) estimated_bulk_energy_fraction: f64,
    pub(super) denoised_signal_energy: f64,
    pub(super) denoised_share_of_raw_energy: f64,
    pub(super) shrunk_singular_values: Vec<f64>,
    pub(super) denoised_cumulative_energy: CumulativeEnergy,
}

/// Spectral summary computed for one matrix delta.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SvdSpectrum {
    pub(super) method: String,
    pub(super) effective_sketch_rank: usize,
    pub(super) sketch_energy_ratio: f64,
    pub(super) top_singular_values: Vec<f64>,
    pub(super) cumulative_energy: CumulativeEnergy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) denoised: Option<DenoisedSpectrum>,
}

impl SvdSpectrum {
    pub(super) fn zero() -> Self {
        Self {
            method: "zero-delta-skipped".to_owned(),
            effective_sketch_rank: 0,
            sketch_energy_ratio: 0.0,
            top_singular_values: Vec::new(),
            cumulative_energy: CumulativeEnergy::default(),
            denoised: None,
        }
    }
}

/// Extracted denoised low-rank singular triplets `(u_k, v_k, shrunk_sigma_k)`.
pub(super) struct DenoisedFactors {
    pub(super) spectrum: SvdSpectrum,
    pub(super) rank: usize,
    pub(super) u_cols: Vec<Vec<f32>>,
    pub(super) v_cols: Vec<Vec<f32>>,
    pub(super) shrunk_sigmas: Vec<f64>,
}

#[expect(
    clippy::as_conversions,
    reason = "Matrix dimensions convert cleanly to f64 for bulk edge and aspect-ratio formulas."
)]
#[inline]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Gavish–Donoho (2017) Frobenius-optimal singular value shrinkage above `bulk_edge`.
#[must_use]
pub(super) fn gavish_donoho_frobenius_shrink(sigma: f64, bulk_edge: f64, beta: f64) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 {
        return 0.0;
    }
    if !bulk_edge.is_finite() || bulk_edge <= 0.0 {
        return sigma;
    }
    let beta_clamped = beta.clamp(1e-12, 1.0);
    let y_edge = 1.0 + beta_clamped.sqrt();
    let y = (sigma / bulk_edge) * y_edge;
    if y <= y_edge {
        return 0.0;
    }
    let radicand = (y * y - beta_clamped - 1.0).powi(2) - 4.0 * beta_clamped;
    if radicand <= 0.0 {
        return 0.0;
    }
    (radicand.sqrt() / (y * y_edge)) * bulk_edge
}

/// Applies the selected singular value shrinkage mode to a candidate singular value.
#[must_use]
pub(super) fn apply_shrinkage_mode(
    sigma: f64,
    bulk_edge: f64,
    beta: f64,
    mode: ShrinkageMode,
    blend: Option<f64>,
) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 {
        return 0.0;
    }
    match mode {
        ShrinkageMode::GavishDonoho => gavish_donoho_frobenius_shrink(sigma, bulk_edge, beta),
        ShrinkageMode::Unshrunk => sigma,
        ShrinkageMode::Blended => {
            let shrunk = gavish_donoho_frobenius_shrink(sigma, bulk_edge, beta);
            let alpha = blend.unwrap_or(1.0);
            (1.0 - alpha) * sigma + alpha * shrunk
        }
    }
}

#[expect(
    clippy::as_conversions,
    reason = "Deterministic float narrowing from f64 PRNG/orthonormal buffers to f32 sketch matrices."
)]
#[inline]
const fn f64_to_f32_lossy(value: f64) -> f32 {
    value as f32
}

#[expect(
    clippy::as_conversions,
    reason = "u64 Mantissa bits (53-bit) convert directly to f64 for deterministic PRNG sampling."
)]
#[inline]
fn u64_to_unit_f64(bits: u64) -> f64 {
    ((bits >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
}

#[inline]
const fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[derive(Clone)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn from_name(name: &str) -> Self {
        // Mix fixed DaisyRose audit constant with FNV-1a of the tensor name.
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for &byte in name.as_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        Self {
            state: 0x4441_4953_5952_4f53_u64 ^ hash,
        }
    }

    const fn next_u64(&mut self) -> u64 {
        splitmix64_step(&mut self.state)
    }

    fn next_unit_f64(&mut self) -> f64 {
        u64_to_unit_f64(self.next_u64())
    }

    fn next_normal_f64(&mut self) -> f64 {
        let u1 = self.next_unit_f64().max(1e-15);
        let u2 = self.next_unit_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn fill_gaussian_f32(name: &str, rows: usize, cols: usize) -> crate::Result<Vec<f32>> {
    let total = rows
        .checked_mul(cols)
        .ok_or("Gaussian sketch dimensions overflowed.")?;
    let mut rng = DeterministicRng::from_name(name);
    let mut out = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|error| format!("Cannot allocate Gaussian test matrix: {error}"))?;
    for _ in 0..total {
        out.push(f64_to_f32_lossy(rng.next_normal_f64()));
    }
    Ok(out)
}

/// Computes `Y[m x k] = D[m x n] * X[n x k]` with row-striped parallelism.
fn matmul_d_x(
    d: &[f32],
    m: usize,
    n: usize,
    x: &[f32],
    k: usize,
    workers: usize,
) -> crate::Result<Vec<f32>> {
    let y_len = m
        .checked_mul(k)
        .ok_or("Matmul output dimensions overflow.")?;
    let mut y = vec![0.0_f32; y_len];
    if m == 0 || n == 0 || k == 0 {
        return Ok(y);
    }
    let worker_count = workers.clamp(1, m);
    let chunk_rows = m.div_ceil(worker_count);
    std::thread::scope(|scope| {
        for (chunk_idx, y_chunk) in y.chunks_mut(chunk_rows * k).enumerate() {
            let row_start = chunk_idx * chunk_rows;
            let row_count = y_chunk.len() / k;
            let d_chunk = &d[row_start * n..(row_start + row_count) * n];
            scope.spawn(move || {
                for r in 0..row_count {
                    let d_row = &d_chunk[r * n..(r + 1) * n];
                    let y_row = &mut y_chunk[r * k..(r + 1) * k];
                    for (c, &d_val) in d_row.iter().enumerate() {
                        if d_val == 0.0 {
                            continue;
                        }
                        let x_row = &x[c * k..(c + 1) * k];
                        for (acc, &xv) in y_row.iter_mut().zip(x_row) {
                            *acc += d_val * xv;
                        }
                    }
                }
            });
        }
    });
    Ok(y)
}

/// Computes `Z[n x k] = D^T[n x m] * Q[m x k]` accumulating in `f64`.
fn matmul_dt_q_f64(
    d: &[f32],
    m: usize,
    n: usize,
    q: &[f32],
    k: usize,
    workers: usize,
) -> crate::Result<Vec<f64>> {
    let z_len = n
        .checked_mul(k)
        .ok_or("Transpose matmul output dimensions overflow.")?;
    let mut z = vec![0.0_f64; z_len];
    if m == 0 || n == 0 || k == 0 {
        return Ok(z);
    }
    let worker_count = workers.clamp(1, m);
    let chunk_rows = m.div_ceil(worker_count);
    let partials: Vec<crate::Result<Vec<f64>>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for chunk_idx in 0..worker_count {
            let row_start = chunk_idx * chunk_rows;
            if row_start >= m {
                break;
            }
            let row_end = (row_start + chunk_rows).min(m);
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                local
                    .try_reserve_exact(z_len)
                    .map_err(|error| format!("Cannot allocate transpose scratch: {error}"))?;
                local.resize(z_len, 0.0_f64);
                for r in row_start..row_end {
                    let d_row = &d[r * n..(r + 1) * n];
                    let q_row = &q[r * k..(r + 1) * k];
                    for (c, &d_val) in d_row.iter().enumerate() {
                        if d_val == 0.0 {
                            continue;
                        }
                        let d_f64 = f64::from(d_val);
                        let local_row = &mut local[c * k..(c + 1) * k];
                        for (acc, &qv) in local_row.iter_mut().zip(q_row) {
                            *acc += d_f64 * f64::from(qv);
                        }
                    }
                }
                Ok(local)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err("Transpose matmul worker panicked.".into()))
            })
            .collect()
    });
    for partial in partials {
        let partial = partial?;
        for (dst, src) in z.iter_mut().zip(partial) {
            *dst += src;
        }
    }
    Ok(z)
}

#[inline]
fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

/// Two-pass Modified Gram-Schmidt column orthonormalization in `f64`.
///
/// Accepts row-major `data[rows x cols]`, returning `(col_vectors_f64, effective_rank)`.
fn orthonormalize_columns_f64(
    rows: usize,
    cols: usize,
    get_val: impl Fn(usize, usize) -> f64,
) -> crate::Result<Vec<Vec<f64>>> {
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    let mut columns = Vec::with_capacity(cols);
    let mut max_norm = 0.0_f64;
    for j in 0..cols {
        let mut col = Vec::new();
        col.try_reserve_exact(rows)
            .map_err(|error| format!("Cannot allocate orthonormal basis column: {error}"))?;
        let mut norm_sq = 0.0_f64;
        for r in 0..rows {
            let v = get_val(r, j);
            if !v.is_finite() {
                return Err("Nonfinite value encountered during orthonormalization.".into());
            }
            norm_sq += v * v;
            col.push(v);
        }
        max_norm = max_norm.max(norm_sq.sqrt());
        columns.push(col);
    }
    if max_norm == 0.0 {
        return Ok(Vec::new());
    }
    let threshold = max_norm * 1e-12;
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(cols);
    for mut col in columns {
        for _ in 0..2 {
            for prev in &basis {
                let proj = dot_f64(&col, prev);
                for (dst, &p) in col.iter_mut().zip(prev) {
                    *dst -= proj * p;
                }
            }
        }
        let norm = dot_f64(&col, &col).sqrt();
        if norm.is_finite() && norm > threshold {
            let inv = 1.0 / norm;
            for v in &mut col {
                *v *= inv;
            }
            basis.push(col);
        }
    }
    Ok(basis)
}

fn basis_cols_to_row_major_f32(basis: &[Vec<f64>], rows: usize) -> crate::Result<Vec<f32>> {
    let k = basis.len();
    let total = rows
        .checked_mul(k)
        .ok_or("Basis matrix dimensions overflow.")?;
    let mut out = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|error| format!("Cannot allocate row-major basis: {error}"))?;
    for r in 0..rows {
        for col in basis {
            out.push(f64_to_f32_lossy(col[r]));
        }
    }
    Ok(out)
}

/// Classical cyclic Jacobi symmetric eigendecomposition for `n x n` matrix (`n <= 256`).
///
/// Returns `(eigenvalues_descending, eigenvectors_as_columns)` where each column
/// vector has unit Euclidean norm.
pub(super) fn jacobi_eigendecomp_f64(
    matrix: &mut [f64],
    n: usize,
) -> crate::Result<(Vec<f64>, Vec<Vec<f64>>)> {
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if matrix.len() != n.saturating_mul(n) {
        return Err("Jacobi eigensolver input shape mismatch.".into());
    }
    let mut diag_scale = 0.0_f64;
    for i in 0..n {
        let d = matrix[i * n + i];
        if !d.is_finite() {
            return Err("Nonfinite diagonal in symmetric Gram matrix.".into());
        }
        diag_scale = diag_scale.max(d.abs());
    }
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if diag_scale == 0.0 {
        let cols = (0..n)
            .map(|col| (0..n).map(|row| v[row * n + col]).collect())
            .collect();
        return Ok((vec![0.0; n], cols));
    }
    let tol = diag_scale * 1e-14;
    for _sweep in 0..120 {
        let mut max_off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                max_off = max_off.max(matrix[p * n + q].abs());
            }
        }
        if max_off <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = matrix[p * n + q];
                if a_pq.abs() <= tol {
                    continue;
                }
                let a_pp = matrix[p * n + p];
                let a_qq = matrix[q * n + q];
                let tau = (a_qq - a_pp) / (2.0 * a_pq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let rho = s / (1.0 + c);

                matrix[p * n + p] = a_pp - t * a_pq;
                matrix[q * n + q] = a_qq + t * a_pq;
                matrix[p * n + q] = 0.0;
                matrix[q * n + p] = 0.0;

                for r in 0..n {
                    if r != p && r != q {
                        let a_rp = matrix[r * n + p];
                        let a_rq = matrix[r * n + q];
                        let next_rp = a_rp - s * (a_rq + rho * a_rp);
                        let next_rq = a_rq + s * (a_rp - rho * a_rq);
                        matrix[r * n + p] = next_rp;
                        matrix[p * n + r] = next_rp;
                        matrix[r * n + q] = next_rq;
                        matrix[q * n + r] = next_rq;
                    }
                    let v_rp = v[r * n + p];
                    let v_rq = v[r * n + q];
                    v[r * n + p] = c * v_rp - s * v_rq;
                    v[r * n + q] = s * v_rp + c * v_rq;
                }
            }
        }
    }
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|col| {
            let lambda = matrix[col * n + col].max(0.0);
            let vec_col = (0..n).map(|row| v[row * n + col]).collect();
            (lambda, vec_col)
        })
        .collect();
    pairs.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigvecs = Vec::with_capacity(n);
    for (lambda, vec_col) in pairs {
        eigenvalues.push(lambda);
        eigvecs.push(vec_col);
    }
    Ok((eigenvalues, eigvecs))
}

/// Classical cyclic Jacobi symmetric eigenvalues for `n x n` matrix (`n <= 256`),
/// returned nonnegative in descending order.
pub(super) fn jacobi_eigenvalues_f64(matrix: &mut [f64], n: usize) -> crate::Result<Vec<f64>> {
    Ok(jacobi_eigendecomp_f64(matrix, n)?.0)
}

/// Computes spike-deflated bulk edges and Gavish–Donoho shrunk spectral metrics.
pub(super) fn compute_denoised_spectrum(
    top_singular_values: &[f64],
    m: usize,
    n: usize,
    total_energy: f64,
    raw_null_edge: Option<f64>,
    max_rank_reported: usize,
) -> DenoisedSpectrum {
    let min_dim = m.min(n);
    let max_dim = m.max(n);
    let beta = if max_dim > 0 {
        usize_to_f64(min_dim) / usize_to_f64(max_dim)
    } else {
        1.0
    };
    if total_energy <= 0.0 || !total_energy.is_finite() || m == 0 || n == 0 {
        return DenoisedSpectrum {
            rademacher_null_bulk_edge: raw_null_edge,
            isotropic_mp_bulk_edge: 0.0,
            effective_bulk_edge: 0.0,
            aspect_ratio_beta: beta,
            above_bulk_rank: 0,
            tail_saturated: false,
            sigma_max_to_bulk_edge_ratio: 0.0,
            estimated_bulk_energy_fraction: 0.0,
            denoised_signal_energy: 0.0,
            denoised_share_of_raw_energy: 0.0,
            shrunk_singular_values: Vec::new(),
            denoised_cumulative_energy: CumulativeEnergy::default(),
        };
    }
    let m_f = usize_to_f64(m);
    let n_f = usize_to_f64(n);
    let mut r_above = 0_usize;
    let mut mp_edge = 0.0_f64;
    let mut null_edge_scaled = raw_null_edge;
    let mut eff_edge = 0.0_f64;
    let mut bulk_energy = total_energy;

    for _ in 0..3 {
        let spike_energy: f64 = top_singular_values
            .iter()
            .take(r_above)
            .map(|&s| s * s)
            .sum();
        let remaining = (total_energy - spike_energy).max(0.0);
        bulk_energy = if r_above < min_dim {
            ((usize_to_f64(min_dim) / usize_to_f64(min_dim - r_above)) * remaining)
                .min(total_energy)
        } else {
            0.0
        };
        let scale_ratio = (bulk_energy / total_energy).sqrt();
        mp_edge = (bulk_energy / (m_f * n_f)).sqrt() * (m_f.sqrt() + n_f.sqrt());
        null_edge_scaled = raw_null_edge.map(|e| e * scale_ratio);
        eff_edge = null_edge_scaled.map_or(mp_edge, |ne| ne.max(mp_edge));
        r_above = top_singular_values
            .iter()
            .take(max_rank_reported)
            .filter(|&&s| s > eff_edge)
            .count();
    }

    let shrunk_singular_values: Vec<f64> = top_singular_values
        .iter()
        .take(max_rank_reported)
        .map(|&s| gavish_donoho_frobenius_shrink(s, eff_edge, beta))
        .filter(|&s| s > 0.0)
        .collect();
    let above_bulk_rank = shrunk_singular_values.len();
    let tail_saturated = max_rank_reported > 0
        && max_rank_reported < min_dim
        && top_singular_values
            .get(max_rank_reported - 1)
            .is_some_and(|&s| s > eff_edge);
    let shrunk_sq: Vec<f64> = shrunk_singular_values.iter().map(|&s| s * s).collect();
    let denoised_signal_energy: f64 = shrunk_sq.iter().sum();
    let sigma_max = top_singular_values.first().copied().unwrap_or(0.0);
    let sigma_max_to_bulk_edge_ratio = if eff_edge > sigma_max * 1e-12 {
        sigma_max / eff_edge
    } else if sigma_max > 0.0 {
        1e12
    } else {
        0.0
    };
    let estimated_bulk_energy_fraction = (bulk_energy / total_energy).clamp(0.0, 1.0);
    let denoised_share_of_raw_energy = (denoised_signal_energy / total_energy).clamp(0.0, 1.0);
    let denoised_cumulative_energy =
        CumulativeEnergy::from_eigenvalues(&shrunk_sq, denoised_signal_energy);

    DenoisedSpectrum {
        rademacher_null_bulk_edge: null_edge_scaled,
        isotropic_mp_bulk_edge: mp_edge,
        effective_bulk_edge: eff_edge,
        aspect_ratio_beta: beta,
        above_bulk_rank,
        tail_saturated,
        sigma_max_to_bulk_edge_ratio,
        estimated_bulk_energy_fraction,
        denoised_signal_energy,
        denoised_share_of_raw_energy,
        shrunk_singular_values,
        denoised_cumulative_energy,
    }
}

fn spectrum_from_eigenvalues(
    method: String,
    eigenvalues: &[f64],
    dim: usize,
    total_energy: f64,
    verify_projection_bound: bool,
    denoised: Option<DenoisedSpectrum>,
) -> crate::Result<SvdSpectrum> {
    let recovered_energy: f64 = eigenvalues.iter().sum();
    let ratio = if total_energy > 0.0 {
        recovered_energy / total_energy
    } else {
        0.0
    };
    if verify_projection_bound && ratio > 1.0 + 1e-4 {
        return Err(format!(
            "Subspace energy exceeded matrix Frobenius energy (ratio={ratio:.6})."
        ));
    }
    if !verify_projection_bound && ratio > 2.0 {
        return Err(format!(
            "Two-sided sketch energy diverged beyond sanity ceiling (ratio={ratio:.6})."
        ));
    }
    let top_singular_values: Vec<f64> = eigenvalues.iter().take(64).map(|&l| l.sqrt()).collect();
    let cumulative_energy = CumulativeEnergy::from_eigenvalues(eigenvalues, total_energy);
    Ok(SvdSpectrum {
        method,
        effective_sketch_rank: dim,
        sketch_energy_ratio: ratio,
        top_singular_values,
        cumulative_energy,
        denoised,
    })
}

fn build_exact_gram_matrix_f64(d: &[f32], m: usize, n: usize) -> (usize, Vec<f64>, bool) {
    let dim = m.min(n);
    let mut gram = vec![0.0_f64; dim * dim];
    let row_gram = m <= n;
    if row_gram {
        for i in 0..m {
            let row_i = &d[i * n..(i + 1) * n];
            for j in i..m {
                let row_j = &d[j * n..(j + 1) * n];
                let val: f64 = row_i
                    .iter()
                    .zip(row_j)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                gram[i * dim + j] = val;
                gram[j * dim + i] = val;
            }
        }
    } else {
        for r in 0..m {
            let row = &d[r * n..(r + 1) * n];
            for i in 0..n {
                let a = f64::from(row[i]);
                if a == 0.0 {
                    continue;
                }
                for j in i..n {
                    gram[i * dim + j] += a * f64::from(row[j]);
                }
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                gram[j * dim + i] = gram[i * dim + j];
            }
        }
    }
    (dim, gram, row_gram)
}

/// Computes `Y[m x k] = (S (*) D)[m x n] * X[n x k]` with implicit deterministic Rademacher `±1` signs.
fn matmul_d_x_rademacher(
    d: &[f32],
    m: usize,
    n: usize,
    x: &[f32],
    k: usize,
    seed: u64,
    workers: usize,
) -> crate::Result<Vec<f32>> {
    let y_len = m
        .checked_mul(k)
        .ok_or("Rademacher matmul output dimensions overflow.")?;
    let mut y = vec![0.0_f32; y_len];
    if m == 0 || n == 0 || k == 0 {
        return Ok(y);
    }
    let worker_count = workers.clamp(1, m);
    let chunk_rows = m.div_ceil(worker_count);
    let word_count = n.div_ceil(64);
    std::thread::scope(|scope| {
        for (chunk_idx, y_chunk) in y.chunks_mut(chunk_rows * k).enumerate() {
            let row_start = chunk_idx * chunk_rows;
            let row_count = y_chunk.len() / k;
            let d_chunk = &d[row_start * n..(row_start + row_count) * n];
            scope.spawn(move || {
                for r in 0..row_count {
                    let global_row = row_start + r;
                    let d_row = &d_chunk[r * n..(r + 1) * n];
                    let y_row = &mut y_chunk[r * k..(r + 1) * k];
                    for word_idx in 0..word_count {
                        let word = rademacher_word_seed(seed, global_row, word_idx);
                        let col_start = word_idx * 64;
                        let col_end = (col_start + 64).min(n);
                        for (bit, c) in (col_start..col_end).enumerate() {
                            let d_val = d_row[c];
                            if d_val == 0.0 {
                                continue;
                            }
                            let signed_d = if ((word >> bit) & 1) == 1 {
                                d_val
                            } else {
                                -d_val
                            };
                            let x_row = &x[c * k..(c + 1) * k];
                            for (acc, &xv) in y_row.iter_mut().zip(x_row) {
                                *acc += signed_d * xv;
                            }
                        }
                    }
                }
            });
        }
    });
    Ok(y)
}

/// Computes `Z[n x k] = (S (*) D)^T[n x m] * Q[m x k]` accumulating in `f64` without copying `D`.
fn matmul_dt_q_rademacher(
    d: &[f32],
    m: usize,
    n: usize,
    q: &[f32],
    k: usize,
    seed: u64,
    workers: usize,
) -> crate::Result<Vec<f64>> {
    let z_len = n
        .checked_mul(k)
        .ok_or("Rademacher transpose matmul output dimensions overflow.")?;
    let mut z = vec![0.0_f64; z_len];
    if m == 0 || n == 0 || k == 0 {
        return Ok(z);
    }
    let worker_count = workers.clamp(1, m);
    let chunk_rows = m.div_ceil(worker_count);
    let word_count = n.div_ceil(64);
    let partials: Vec<crate::Result<Vec<f64>>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for chunk_idx in 0..worker_count {
            let row_start = chunk_idx * chunk_rows;
            if row_start >= m {
                break;
            }
            let row_end = (row_start + chunk_rows).min(m);
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                local.try_reserve_exact(z_len).map_err(|error| {
                    format!("Cannot allocate Rademacher transpose scratch: {error}")
                })?;
                local.resize(z_len, 0.0_f64);
                for r in row_start..row_end {
                    let d_row = &d[r * n..(r + 1) * n];
                    let q_row = &q[r * k..(r + 1) * k];
                    for word_idx in 0..word_count {
                        let word = rademacher_word_seed(seed, r, word_idx);
                        let col_start = word_idx * 64;
                        let col_end = (col_start + 64).min(n);
                        for (bit, c) in (col_start..col_end).enumerate() {
                            let d_val = d_row[c];
                            if d_val == 0.0 {
                                continue;
                            }
                            let d_f64 = if ((word >> bit) & 1) == 1 {
                                f64::from(d_val)
                            } else {
                                -f64::from(d_val)
                            };
                            let local_row = &mut local[c * k..(c + 1) * k];
                            for (acc, &qv) in local_row.iter_mut().zip(q_row) {
                                *acc += d_f64 * f64::from(qv);
                            }
                        }
                    }
                }
                Ok(local)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err("Rademacher transpose matmul worker panicked.".into()))
            })
            .collect()
    });
    for partial in partials {
        let partial = partial?;
        for (dst, src) in z.iter_mut().zip(partial) {
            *dst += src;
        }
    }
    Ok(z)
}

struct SubspaceDecomposition {
    basis_q: Vec<Vec<f64>>,
    b_t: Vec<f64>,
    gram: Vec<f64>,
    k_eff: usize,
}

fn subspace_decomposition(
    name: &str,
    m: usize,
    n: usize,
    k: usize,
    power_iterations: usize,
    mul_d_x: &impl Fn(&[f32], usize) -> crate::Result<Vec<f32>>,
    mul_dt_q: &impl Fn(&[f32], usize) -> crate::Result<Vec<f64>>,
) -> crate::Result<Option<SubspaceDecomposition>> {
    let omega = fill_gaussian_f32(name, n, k)?;
    let y = mul_d_x(&omega, k)?;
    let mut basis_q = orthonormalize_columns_f64(m, k, |r, j| f64::from(y[r * k + j]))?;
    if basis_q.is_empty() {
        return Ok(None);
    }
    let mut q = basis_cols_to_row_major_f32(&basis_q, m)?;
    let mut k_eff = basis_q.len();

    for _ in 0..power_iterations {
        let z = mul_dt_q(&q, k_eff)?;
        let z_basis = orthonormalize_columns_f64(n, k_eff, |r, j| z[r * k_eff + j])?;
        if z_basis.is_empty() {
            return Ok(None);
        }
        let k_z = z_basis.len();
        let q_z = basis_cols_to_row_major_f32(&z_basis, n)?;
        let y_next = mul_d_x(&q_z, k_z)?;
        basis_q = orthonormalize_columns_f64(m, k_z, |r, j| f64::from(y_next[r * k_z + j]))?;
        if basis_q.is_empty() {
            return Ok(None);
        }
        k_eff = basis_q.len();
        q = basis_cols_to_row_major_f32(&basis_q, m)?;
    }

    let b_t = mul_dt_q(&q, k_eff)?;
    let mut gram = vec![0.0_f64; k_eff * k_eff];
    for c in 0..n {
        let row = &b_t[c * k_eff..(c + 1) * k_eff];
        for i in 0..k_eff {
            let v = row[i];
            if v == 0.0 {
                continue;
            }
            for j in i..k_eff {
                gram[i * k_eff + j] += v * row[j];
            }
        }
    }
    for i in 0..k_eff {
        for j in (i + 1)..k_eff {
            gram[j * k_eff + i] = gram[i * k_eff + j];
        }
    }
    Ok(Some(SubspaceDecomposition {
        basis_q,
        b_t,
        gram,
        k_eff,
    }))
}

/// Estimates the heteroscedastic empirical bulk upper edge `sigma_max(S (*) D)` using
/// zero-extra-matrix-allocation Rademacher sign flips (`k_null = min(8, m, n)`, `q = 3`).
pub(super) fn estimate_rademacher_null_edge(
    name: &str,
    d: &[f32],
    m: usize,
    n: usize,
    workers: usize,
) -> crate::Result<f64> {
    let k_null = 8_usize.min(m).min(n);
    if k_null == 0 {
        return Ok(0.0);
    }
    let sign_seed = DeterministicRng::from_name(&format!("{name}:rademacher-null")).state;
    let Some(mut decomp) = subspace_decomposition(
        &format!("{name}:null-omega"),
        m,
        n,
        k_null,
        3,
        &|x, k| matmul_d_x_rademacher(d, m, n, x, k, sign_seed, workers),
        &|q, k| matmul_dt_q_rademacher(d, m, n, q, k, sign_seed, workers),
    )?
    else {
        return Ok(0.0);
    };
    let eigenvalues = jacobi_eigenvalues_f64(&mut decomp.gram, decomp.k_eff)?;
    Ok(eigenvalues.first().copied().unwrap_or(0.0).sqrt())
}

/// Computes exact SVD spectrum via smaller-dimension Gram matrix when `min(m, n) <= k_sketch`.
pub(super) fn exact_gram_svd(
    name: &str,
    d: &[f32],
    m: usize,
    n: usize,
    total_energy: f64,
    include_denoising: bool,
    max_rank_reported: usize,
    workers: usize,
) -> crate::Result<SvdSpectrum> {
    if total_energy == 0.0 {
        return Ok(SvdSpectrum::zero());
    }
    let (dim, mut gram, _) = build_exact_gram_matrix_f64(d, m, n);
    let eigenvalues = jacobi_eigenvalues_f64(&mut gram, dim)?;
    let top_singular_values: Vec<f64> = eigenvalues.iter().take(64).map(|&l| l.sqrt()).collect();
    let denoised = if include_denoising {
        let null_edge = estimate_rademacher_null_edge(name, d, m, n, workers)?;
        Some(compute_denoised_spectrum(
            &top_singular_values,
            m,
            n,
            total_energy,
            Some(null_edge),
            max_rank_reported.min(dim),
        ))
    } else {
        None
    };
    spectrum_from_eigenvalues(
        "exact-gram-jacobi-f64".to_owned(),
        &eigenvalues,
        dim,
        total_energy,
        true,
        denoised,
    )
}

/// Computes randomized subspace iteration SVD (Halko-Martinsson-Tropp Alg 4.4) on `D[m x n]`.
pub(super) fn randomized_subspace_svd(
    name: &str,
    d: &[f32],
    m: usize,
    n: usize,
    k_requested: usize,
    power_iterations: usize,
    total_energy: f64,
    include_denoising: bool,
    max_rank_reported: usize,
    workers: usize,
) -> crate::Result<SvdSpectrum> {
    if total_energy == 0.0 {
        return Ok(SvdSpectrum::zero());
    }
    let k = k_requested.min(m).min(n);
    if k == 0 {
        return Ok(SvdSpectrum::zero());
    }
    if m.min(n) <= k {
        return exact_gram_svd(
            name,
            d,
            m,
            n,
            total_energy,
            include_denoising,
            max_rank_reported,
            workers,
        );
    }

    let Some(mut decomp) = subspace_decomposition(
        name,
        m,
        n,
        k,
        power_iterations,
        &|x, cols| matmul_d_x(d, m, n, x, cols, workers),
        &|q, cols| matmul_dt_q_f64(d, m, n, q, cols, workers),
    )?
    else {
        return Ok(SvdSpectrum::zero());
    };
    let eigenvalues = jacobi_eigenvalues_f64(&mut decomp.gram, decomp.k_eff)?;
    let top_singular_values: Vec<f64> = eigenvalues.iter().take(64).map(|&l| l.sqrt()).collect();
    let denoised = if include_denoising {
        let null_edge = estimate_rademacher_null_edge(name, d, m, n, workers)?;
        Some(compute_denoised_spectrum(
            &top_singular_values,
            m,
            n,
            total_energy,
            Some(null_edge),
            max_rank_reported.min(decomp.k_eff),
        ))
    } else {
        None
    };
    let method = format!("randomized-subspace-q{power_iterations}-k{k}-f32-gram-f64");
    spectrum_from_eigenvalues(
        method,
        &eigenvalues,
        decomp.k_eff,
        total_energy,
        true,
        denoised,
    )
}

fn normalize_f64_to_f32(values: &[f64]) -> Option<Vec<f32>> {
    let norm = dot_f64(values, values).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    let inv = 1.0 / norm;
    Some(values.iter().map(|&v| f64_to_f32_lossy(v * inv)).collect())
}

/// Computes denoised SVD spectrum and extracts shrunk singular triplets `(u_k, v_k, shrunk_sigma_k)`
/// for modes clearing `sigma > min_spike_ratio * effective_bulk_edge` up to `export_max_rank`
/// using the default Gavish–Donoho shrinkage mode.
#[cfg(test)]
pub(super) fn randomized_subspace_denoised_factors(
    name: &str,
    d: &[f32],
    m: usize,
    n: usize,
    k_requested: usize,
    power_iterations: usize,
    total_energy: f64,
    export_max_rank: usize,
    min_spike_ratio: f64,
    workers: usize,
) -> crate::Result<DenoisedFactors> {
    randomized_subspace_denoised_factors_with_mode(
        name,
        d,
        m,
        n,
        k_requested,
        power_iterations,
        total_energy,
        export_max_rank,
        min_spike_ratio,
        ShrinkageMode::GavishDonoho,
        None,
        workers,
    )
}

/// Computes denoised SVD spectrum and extracts singular triplets `(u_k, v_k, effective_sigma_k)`
/// for modes clearing `sigma > min_spike_ratio * effective_bulk_edge` up to `export_max_rank`
/// under the selected [`ShrinkageMode`].
#[expect(
    clippy::too_many_arguments,
    reason = "Direct extension of randomized_subspace_denoised_factors adding shrinkage mode and optional blend weight."
)]
pub(super) fn randomized_subspace_denoised_factors_with_mode(
    name: &str,
    d: &[f32],
    m: usize,
    n: usize,
    k_requested: usize,
    power_iterations: usize,
    total_energy: f64,
    export_max_rank: usize,
    min_spike_ratio: f64,
    shrinkage_mode: ShrinkageMode,
    shrinkage_blend: Option<f64>,
    workers: usize,
) -> crate::Result<DenoisedFactors> {
    if shrinkage_mode == ShrinkageMode::Blended
        && let Some(alpha) = shrinkage_blend
        && (!alpha.is_finite() || !(0.0..=1.0).contains(&alpha))
    {
        return Err("shrinkageBlend must be finite and in 0.0..=1.0.".into());
    }
    if total_energy <= 0.0 || export_max_rank == 0 {
        return Ok(DenoisedFactors {
            spectrum: SvdSpectrum::zero(),
            rank: 0,
            u_cols: Vec::new(),
            v_cols: Vec::new(),
            shrunk_sigmas: Vec::new(),
        });
    }
    let k = k_requested.min(m).min(n);
    if k == 0 {
        return Ok(DenoisedFactors {
            spectrum: SvdSpectrum::zero(),
            rank: 0,
            u_cols: Vec::new(),
            v_cols: Vec::new(),
            shrunk_sigmas: Vec::new(),
        });
    }

    if m.min(n) <= k {
        let (dim, mut gram, row_gram) = build_exact_gram_matrix_f64(d, m, n);
        let (eigenvalues, eigvecs) = jacobi_eigendecomp_f64(&mut gram, dim)?;
        let top_singular_values: Vec<f64> =
            eigenvalues.iter().take(64).map(|&l| l.sqrt()).collect();
        let null_edge = estimate_rademacher_null_edge(name, d, m, n, workers)?;
        let report_cap = export_max_rank.min(dim).min(64);
        let denoised = compute_denoised_spectrum(
            &top_singular_values,
            m,
            n,
            total_energy,
            Some(null_edge),
            report_cap,
        );
        let eff_edge = denoised.effective_bulk_edge;
        let beta = denoised.aspect_ratio_beta;
        let mut u_cols = Vec::new();
        let mut v_cols = Vec::new();
        let mut shrunk_sigmas = Vec::new();
        for (idx, (&lambda, w)) in eigenvalues.iter().zip(&eigvecs).enumerate() {
            if idx >= report_cap {
                break;
            }
            let sigma = lambda.sqrt();
            let shrunk =
                apply_shrinkage_mode(sigma, eff_edge, beta, shrinkage_mode, shrinkage_blend);
            if sigma <= min_spike_ratio * eff_edge || shrunk <= 0.0 {
                continue;
            }
            let (u_raw, v_raw) = if row_gram {
                let mut v_buf = vec![0.0_f64; n];
                for r in 0..m {
                    let ur = w[r];
                    if ur == 0.0 {
                        continue;
                    }
                    let d_row = &d[r * n..(r + 1) * n];
                    for (dst, &dv) in v_buf.iter_mut().zip(d_row) {
                        *dst += f64::from(dv) * ur;
                    }
                }
                (w.clone(), v_buf)
            } else {
                let mut u_buf = vec![0.0_f64; m];
                for r in 0..m {
                    let d_row = &d[r * n..(r + 1) * n];
                    u_buf[r] = d_row
                        .iter()
                        .zip(w)
                        .map(|(&dv, &vc)| f64::from(dv) * vc)
                        .sum();
                }
                (u_buf, w.clone())
            };
            if let (Some(u_f32), Some(v_f32)) =
                (normalize_f64_to_f32(&u_raw), normalize_f64_to_f32(&v_raw))
            {
                u_cols.push(u_f32);
                v_cols.push(v_f32);
                shrunk_sigmas.push(shrunk);
            }
        }
        let spectrum = spectrum_from_eigenvalues(
            "exact-gram-jacobi-f64".to_owned(),
            &eigenvalues,
            dim,
            total_energy,
            true,
            Some(denoised),
        )?;
        let rank = shrunk_sigmas.len();
        return Ok(DenoisedFactors {
            spectrum,
            rank,
            u_cols,
            v_cols,
            shrunk_sigmas,
        });
    }

    let Some(mut decomp) = subspace_decomposition(
        name,
        m,
        n,
        k,
        power_iterations,
        &|x, cols| matmul_d_x(d, m, n, x, cols, workers),
        &|q, cols| matmul_dt_q_f64(d, m, n, q, cols, workers),
    )?
    else {
        return Ok(DenoisedFactors {
            spectrum: SvdSpectrum::zero(),
            rank: 0,
            u_cols: Vec::new(),
            v_cols: Vec::new(),
            shrunk_sigmas: Vec::new(),
        });
    };
    let k_eff = decomp.k_eff;
    let (eigenvalues, eigvecs) = jacobi_eigendecomp_f64(&mut decomp.gram, k_eff)?;
    let top_singular_values: Vec<f64> = eigenvalues.iter().take(64).map(|&l| l.sqrt()).collect();
    let null_edge = estimate_rademacher_null_edge(name, d, m, n, workers)?;
    let report_cap = export_max_rank.min(k_eff).min(64);
    let denoised = compute_denoised_spectrum(
        &top_singular_values,
        m,
        n,
        total_energy,
        Some(null_edge),
        report_cap,
    );
    let eff_edge = denoised.effective_bulk_edge;
    let beta = denoised.aspect_ratio_beta;

    let mut u_cols = Vec::new();
    let mut v_cols = Vec::new();
    let mut shrunk_sigmas = Vec::new();
    for (idx, (&lambda, w)) in eigenvalues.iter().zip(&eigvecs).enumerate() {
        if idx >= report_cap {
            break;
        }
        let sigma = lambda.sqrt();
        let shrunk = apply_shrinkage_mode(sigma, eff_edge, beta, shrinkage_mode, shrinkage_blend);
        if sigma <= min_spike_ratio * eff_edge || shrunk <= 0.0 {
            continue;
        }
        let mut u_raw = vec![0.0_f64; m];
        for (j, &wj) in w.iter().enumerate() {
            if wj == 0.0 {
                continue;
            }
            let q_col = &decomp.basis_q[j];
            for (dst, &qv) in u_raw.iter_mut().zip(q_col) {
                *dst += qv * wj;
            }
        }
        let mut v_raw = vec![0.0_f64; n];
        for c in 0..n {
            let row = &decomp.b_t[c * k_eff..(c + 1) * k_eff];
            v_raw[c] = dot_f64(row, w);
        }
        if let (Some(u_f32), Some(v_f32)) =
            (normalize_f64_to_f32(&u_raw), normalize_f64_to_f32(&v_raw))
        {
            u_cols.push(u_f32);
            v_cols.push(v_f32);
            shrunk_sigmas.push(shrunk);
        }
    }

    let method = format!("randomized-subspace-q{power_iterations}-k{k}-f32-gram-f64");
    let spectrum = spectrum_from_eigenvalues(
        method,
        &eigenvalues,
        k_eff,
        total_energy,
        true,
        Some(denoised),
    )?;
    let rank = shrunk_sigmas.len();
    Ok(DenoisedFactors {
        spectrum,
        rank,
        u_cols,
        v_cols,
        shrunk_sigmas,
    })
}

#[expect(
    clippy::as_conversions,
    reason = "64-bit row and word indices map directly into deterministic u64 PRNG seed mixing."
)]
#[inline]
const fn rademacher_word_seed(base_seed: u64, row: usize, word_idx: usize) -> u64 {
    let mut state = base_seed
        ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (word_idx as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    splitmix64_step(&mut state)
}

/// Deterministic Rademacher `±1.0` generator for row `row` across `l` sketch rows.
#[inline]
fn rademacher_words(base_seed: u64, row: usize, word_count: usize) -> [u64; 4] {
    let mut words = [0_u64; 4];
    for (idx, word) in words.iter_mut().take(word_count).enumerate() {
        *word = rademacher_word_seed(base_seed, row, idx);
    }
    words
}

/// Single-pass two-sided streaming sketch (Tropp et al. 2017) for huge matrices (`248320 x 5120`).
pub(super) struct StreamingTwoSidedSketch {
    m: usize,
    n: usize,
    k: usize,
    l: usize,
    psi_seed: u64,
    omega: Vec<f32>,
    y: Vec<f32>,
    w: Vec<f64>,
    row_nonzero: Vec<bool>,
}

impl StreamingTwoSidedSketch {
    pub(super) fn new(name: &str, m: usize, n: usize, k_requested: usize) -> crate::Result<Self> {
        let k = k_requested.min(m).min(n).clamp(1, 128);
        let l = (k * 2).min(256);
        let omega = fill_gaussian_f32(&format!("{name}:omega"), n, k)?;
        let y_len = m
            .checked_mul(k)
            .ok_or("Right sketch dimensions overflow.")?;
        let w_len = l.checked_mul(n).ok_or("Left sketch dimensions overflow.")?;
        let mut y = Vec::new();
        y.try_reserve_exact(y_len)
            .map_err(|error| format!("Cannot allocate streaming right sketch: {error}"))?;
        y.resize(y_len, 0.0_f32);
        let mut w = Vec::new();
        w.try_reserve_exact(w_len)
            .map_err(|error| format!("Cannot allocate streaming left sketch: {error}"))?;
        w.resize(w_len, 0.0_f64);
        let mut row_nonzero = Vec::new();
        row_nonzero
            .try_reserve_exact(m)
            .map_err(|error| format!("Cannot allocate streaming row mask: {error}"))?;
        row_nonzero.resize(m, false);
        let psi_seed = DeterministicRng::from_name(&format!("{name}:psi")).state;
        Ok(Self {
            m,
            n,
            k,
            l,
            psi_seed,
            omega,
            y,
            w,
            row_nonzero,
        })
    }

    /// Updates both right (`Y = D * Omega`) and left (`W = Psi * D`) sketches with rows `[row_start .. row_start + chunk_rows)`.
    pub(super) fn update_chunk(
        &mut self,
        row_start: usize,
        chunk_d: &[f32],
        chunk_rows: usize,
        workers: usize,
    ) -> crate::Result<()> {
        if chunk_rows == 0 {
            return Ok(());
        }
        let end_row = row_start
            .checked_add(chunk_rows)
            .filter(|&end| end <= self.m)
            .ok_or("Streaming sketch chunk exceeded declared row count.")?;
        if chunk_d.len() != chunk_rows.saturating_mul(self.n) {
            return Err("Streaming sketch chunk slice length mismatch.".into());
        }

        let n = self.n;
        let k = self.k;
        let l = self.l;
        let psi_seed = self.psi_seed;
        let omega = &self.omega;
        let y_slice = &mut self.y[row_start * k..end_row * k];
        let mask_slice = &mut self.row_nonzero[row_start..end_row];

        // Pass 1: Right sketch Y_chunk = chunk_d * Omega (parallel by row).
        let row_workers = workers.clamp(1, chunk_rows);
        let rows_per_worker = chunk_rows.div_ceil(row_workers);
        std::thread::scope(|scope| {
            for (idx, (y_part, mask_part)) in y_slice
                .chunks_mut(rows_per_worker * k)
                .zip(mask_slice.chunks_mut(rows_per_worker))
                .enumerate()
            {
                let local_start = idx * rows_per_worker;
                let local_rows = mask_part.len();
                let d_part = &chunk_d[local_start * n..(local_start + local_rows) * n];
                scope.spawn(move || {
                    for r in 0..local_rows {
                        let d_row = &d_part[r * n..(r + 1) * n];
                        let y_row = &mut y_part[r * k..(r + 1) * k];
                        let mut any_nonzero = false;
                        for (c, &dv) in d_row.iter().enumerate() {
                            if dv == 0.0 {
                                continue;
                            }
                            any_nonzero = true;
                            let o_row = &omega[c * k..(c + 1) * k];
                            for (acc, &ov) in y_row.iter_mut().zip(o_row) {
                                *acc += dv * ov;
                            }
                        }
                        mask_part[r] = any_nonzero;
                    }
                });
            }
        });

        // Pass 2: Left sketch W += Psi_chunk * chunk_d (parallel by sketch row i in 0..l).
        let word_count = l.div_ceil(64);
        let mask_readonly = &self.row_nonzero[row_start..end_row];
        let l_workers = workers.clamp(1, l);
        let l_per_worker = l.div_ceil(l_workers);
        std::thread::scope(|scope| {
            for (idx, w_part) in self.w.chunks_mut(l_per_worker * n).enumerate() {
                let i_start = idx * l_per_worker;
                let i_count = w_part.len() / n;
                scope.spawn(move || {
                    for (r, &nonzero) in mask_readonly.iter().enumerate() {
                        if !nonzero {
                            continue;
                        }
                        let words = rademacher_words(psi_seed, row_start + r, word_count);
                        let d_row = &chunk_d[r * n..(r + 1) * n];
                        for local_i in 0..i_count {
                            let i = i_start + local_i;
                            let positive = ((words[i >> 6] >> (i & 63)) & 1) == 1;
                            let w_row = &mut w_part[local_i * n..(local_i + 1) * n];
                            if positive {
                                for (acc, &dv) in w_row.iter_mut().zip(d_row) {
                                    *acc += f64::from(dv);
                                }
                            } else {
                                for (acc, &dv) in w_row.iter_mut().zip(d_row) {
                                    *acc -= f64::from(dv);
                                }
                            }
                        }
                    }
                });
            }
        });
        Ok(())
    }

    pub(super) fn finish(
        self,
        total_energy: f64,
        include_denoising: bool,
        max_rank_reported: usize,
        workers: usize,
    ) -> crate::Result<SvdSpectrum> {
        if total_energy == 0.0 {
            return Ok(SvdSpectrum::zero());
        }
        let m = self.m;
        let n = self.n;
        let k = self.k;
        let l = self.l;
        let basis_q = orthonormalize_columns_f64(m, k, |r, j| f64::from(self.y[r * k + j]))?;
        drop(self.y);
        if basis_q.is_empty() {
            return Ok(SvdSpectrum::zero());
        }
        let k_eff = basis_q.len();

        // Regenerate Psi in memory to form P[l x k_eff] = Psi[l x m] * Q[m x k_eff] in f64.
        let word_count = l.div_ceil(64);
        let row_workers = workers.clamp(1, m);
        let rows_per_worker = m.div_ceil(row_workers);
        let p_partials: Vec<crate::Result<Vec<f64>>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(row_workers);
            for w_idx in 0..row_workers {
                let r_start = w_idx * rows_per_worker;
                if r_start >= m {
                    break;
                }
                let r_end = (r_start + rows_per_worker).min(m);
                let q_ref = &basis_q;
                let mask_ref = &self.row_nonzero;
                let seed = self.psi_seed;
                handles.push(scope.spawn(move || {
                    let mut p_local = vec![0.0_f64; l * k_eff];
                    for r in r_start..r_end {
                        if !mask_ref[r] {
                            continue;
                        }
                        let words = rademacher_words(seed, r, word_count);
                        for i in 0..l {
                            let sign = if ((words[i >> 6] >> (i & 63)) & 1) == 1 {
                                1.0_f64
                            } else {
                                -1.0_f64
                            };
                            let p_row = &mut p_local[i * k_eff..(i + 1) * k_eff];
                            for (dst, q_col) in p_row.iter_mut().zip(q_ref) {
                                *dst += sign * q_col[r];
                            }
                        }
                    }
                    Ok(p_local)
                }));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err("Two-sided sketch P worker panicked.".into()))
                })
                .collect()
        });
        drop(basis_q);
        let mut p = vec![0.0_f64; l * k_eff];
        for part in p_partials {
            let part = part?;
            for (dst, src) in p.iter_mut().zip(part) {
                *dst += src;
            }
        }

        // Thin QR of P[l x k_eff] = Q_p * R_p via two-pass Modified Gram-Schmidt.
        let mut q_p_cols: Vec<Vec<f64>> = Vec::with_capacity(k_eff);
        let mut r_p = vec![0.0_f64; k_eff * k_eff];
        let mut max_diag = 0.0_f64;
        for j in 0..k_eff {
            let mut v: Vec<f64> = (0..l).map(|i| p[i * k_eff + j]).collect();
            for _pass in 0..2 {
                for (prev_idx, prev) in q_p_cols.iter().enumerate() {
                    let dot = dot_f64(&v, prev);
                    r_p[prev_idx * k_eff + j] += dot;
                    for (dst, &pv) in v.iter_mut().zip(prev) {
                        *dst -= dot * pv;
                    }
                }
            }
            let norm = dot_f64(&v, &v).sqrt();
            max_diag = max_diag.max(norm);
            if !norm.is_finite() || norm <= max_diag * 1e-12 {
                return Err("Two-sided sketch projection P became rank-deficient.".into());
            }
            r_p[j * k_eff + j] = norm;
            let inv = 1.0 / norm;
            for x in &mut v {
                *x *= inv;
            }
            q_p_cols.push(v);
        }

        // RHS[k_eff x n] = Q_p^T[k_eff x l] * W[l x n]
        let mut rhs = vec![0.0_f64; k_eff * n];
        for (row_idx, q_col) in q_p_cols.iter().enumerate() {
            let dst_row = &mut rhs[row_idx * n..(row_idx + 1) * n];
            for (i, &qv) in q_col.iter().enumerate() {
                let w_row = &self.w[i * n..(i + 1) * n];
                for (dst, &wv) in dst_row.iter_mut().zip(w_row) {
                    *dst += qv * wv;
                }
            }
        }

        // Back-substitution: solve R_p * B = RHS in place for B[k_eff x n].
        for i in (0..k_eff).rev() {
            let inv_diag = 1.0 / r_p[i * k_eff + i];
            for j in (i + 1)..k_eff {
                let factor = r_p[i * k_eff + j];
                let (head, tail) = rhs.split_at_mut(j * n);
                let row_i = &mut head[i * n..(i + 1) * n];
                let row_j = &tail[..n];
                for (dst, &src) in row_i.iter_mut().zip(row_j) {
                    *dst -= factor * src;
                }
            }
            for dst in &mut rhs[i * n..(i + 1) * n] {
                *dst *= inv_diag;
            }
        }

        // Form Gram matrix G = B * B^T [k_eff x k_eff] in f64.
        let mut gram = vec![0.0_f64; k_eff * k_eff];
        for i in 0..k_eff {
            let row_i = &rhs[i * n..(i + 1) * n];
            for j in i..k_eff {
                let row_j = &rhs[j * n..(j + 1) * n];
                let val = dot_f64(row_i, row_j);
                gram[i * k_eff + j] = val;
                gram[j * k_eff + i] = val;
            }
        }
        let method = format!("single-pass-two-sided-sketch-k{k}-l{l}-f64");
        let eigenvalues = jacobi_eigenvalues_f64(&mut gram, k_eff)?;
        let top_singular_values: Vec<f64> =
            eigenvalues.iter().take(64).map(|&l| l.sqrt()).collect();
        let denoised = include_denoising.then(|| {
            compute_denoised_spectrum(
                &top_singular_values,
                m,
                n,
                total_energy,
                None,
                max_rank_reported.min(k_eff),
            )
        });
        spectrum_from_eigenvalues(method, &eigenvalues, k_eff, total_energy, false, denoised)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::{self, File};
    use std::io::Write as _;
    use std::path::Path;

    use serde_json::json;

    use super::*;
    use crate::bounded_input::ScratchDir;

    fn test_spiked_matrix(
        rows: usize,
        cols: usize,
        sigmas: &[f64],
        noise_ratio: f64,
    ) -> (Vec<f32>, f64) {
        let mut signal = vec![0.0_f64; rows * cols];
        for (k, &sigma) in sigmas.iter().enumerate() {
            let mut u = vec![0.0_f64; rows];
            for (r, dst) in u.iter_mut().enumerate() {
                *dst = (usize_to_f64((r + 1) * (k + 3) + 17) * 0.713).sin();
            }
            let u_norm = dot_f64(&u, &u).sqrt();
            for val in &mut u {
                *val /= u_norm;
            }
            let mut v = vec![0.0_f64; cols];
            for (c, dst) in v.iter_mut().enumerate() {
                *dst = (usize_to_f64((c + 1) * (k + 5) + 41) * 0.371).cos();
            }
            let v_norm = dot_f64(&v, &v).sqrt();
            for val in &mut v {
                *val /= v_norm;
            }
            for r in 0..rows {
                for c in 0..cols {
                    signal[r * cols + c] += sigma * u[r] * v[c];
                }
            }
        }
        let sig_energy: f64 = signal.iter().map(|&x| x * x).sum();
        let amp = ((sig_energy * noise_ratio) / usize_to_f64(rows * cols)).sqrt();
        let mut state = 0xfeed_b0b0_u64;
        let mut mixed = Vec::with_capacity(rows * cols);
        let mut total_energy = 0.0_f64;
        for &s in &signal {
            let bits = splitmix64_step(&mut state);
            let noise = if (bits & 1) == 1 { amp } else { -amp };
            let val = f64_to_f32_lossy(s + noise);
            let vf = f64::from(val);
            total_energy += vf * vf;
            mixed.push(val);
        }
        (mixed, total_energy)
    }

    #[test]
    fn shrinkage_modes_cover_gavish_donoho_unshrunk_and_blended() -> crate::Result<()> {
        let shapes = [
            ("exact-branch", 40_usize, 48_usize, 48_usize),
            ("randomized-branch", 128_usize, 160_usize, 32_usize),
        ];
        std::thread::scope(|scope| -> crate::Result<()> {
            let handles: Vec<_> =
                shapes
                    .into_iter()
                    .map(|(label, rows, cols, k_req)| {
                        scope.spawn(move || -> crate::Result<()> {
                        let (mixed, total_energy) =
                            test_spiked_matrix(rows, cols, &[3.2, 2.4, 1.8], 1.0);
                        let run_mode = |mode: ShrinkageMode, blend: Option<f64>| {
                            randomized_subspace_denoised_factors_with_mode(
                                label,
                                &mixed,
                                rows,
                                cols,
                                k_req,
                                2,
                                total_energy,
                                16,
                                1.0,
                                mode,
                                blend,
                                2,
                            )
                        };
                        let run_ref = &run_mode;
                        let (
                            default_factors,
                            gd_factors,
                            unshrunk,
                            blend_zero,
                            blend_one,
                            blend_mid,
                        ) = std::thread::scope(|inner| -> crate::Result<_> {
                            let h_def = inner.spawn(|| {
                                randomized_subspace_denoised_factors(
                                    label,
                                    &mixed,
                                    rows,
                                    cols,
                                    k_req,
                                    2,
                                    total_energy,
                                    16,
                                    1.0,
                                    2,
                                )
                            });
                            let h_gd = inner
                                .spawn(move || run_ref(ShrinkageMode::GavishDonoho, None));
                            let h_un = inner.spawn(move || run_ref(ShrinkageMode::Unshrunk, None));
                            let h_b0 =
                                inner.spawn(move || run_ref(ShrinkageMode::Blended, Some(0.0)));
                            let h_b1 =
                                inner.spawn(move || run_ref(ShrinkageMode::Blended, Some(1.0)));
                            let h_bm =
                                inner.spawn(move || run_ref(ShrinkageMode::Blended, Some(0.35)));
                            Ok((
                                h_def.join().map_err(|_| String::from("SVD thread panicked."))??,
                                h_gd.join().map_err(|_| String::from("SVD thread panicked."))??,
                                h_un.join().map_err(|_| String::from("SVD thread panicked."))??,
                                h_b0.join().map_err(|_| String::from("SVD thread panicked."))??,
                                h_b1.join().map_err(|_| String::from("SVD thread panicked."))??,
                                h_bm.join().map_err(|_| String::from("SVD thread panicked."))??,
                            ))
                        })?;
                        assert_eq!(default_factors.rank, 3, "{label} should recover 3 spikes");
                        assert_eq!(gd_factors.rank, default_factors.rank);
                        for (&a, &b) in default_factors
                            .shrunk_sigmas
                            .iter()
                            .zip(&gd_factors.shrunk_sigmas)
                        {
                            assert_eq!(a.to_bits(), b.to_bits(), "{label} default != GavishDonoho");
                        }

                        assert_eq!(unshrunk.rank, 3);
                        assert!(
                            unshrunk.spectrum.top_singular_values.len() > 3,
                            "{label} should have bulk singular values rejected by spike threshold"
                        );
                        for k in 0..3 {
                            let raw_sigma = unshrunk.spectrum.top_singular_values[k];
                            assert_eq!(
                                unshrunk.shrunk_sigmas[k].to_bits(),
                                raw_sigma.to_bits(),
                                "{label} mode {k} should retain exact raw sigma"
                            );
                            assert!(unshrunk.shrunk_sigmas[k] > gd_factors.shrunk_sigmas[k]);
                        }

                        assert_eq!(blend_zero.rank, 3);
                        assert_eq!(blend_one.rank, 3);
                        assert_eq!(blend_mid.rank, 3);
                        for k in 0..3 {
                            assert!(
                                (blend_zero.shrunk_sigmas[k] - unshrunk.shrunk_sigmas[k]).abs()
                                    < 1e-12,
                                "{label} blend=0 must match unshrunk"
                            );
                            assert!(
                                (blend_one.shrunk_sigmas[k] - gd_factors.shrunk_sigmas[k]).abs()
                                    < 1e-12,
                                "{label} blend=1 must match GavishDonoho"
                            );
                            let expected = 0.65 * unshrunk.shrunk_sigmas[k]
                                + 0.35 * gd_factors.shrunk_sigmas[k];
                            assert!(
                                (blend_mid.shrunk_sigmas[k] - expected).abs() < 1e-12,
                                "{label} blend=0.35 mismatch"
                            );
                        }
                        Ok(())
                    })
                    })
                    .collect();
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| String::from("SVD shape thread panicked."))??;
            }
            Ok(())
        })
    }

    fn write_toy_safetensors_dir(
        dir: &Path,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        vals: &[f32],
    ) -> crate::Result<()> {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let mut payload = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            let bits = v.to_bits();
            let rounding = 0x7FFF_u32 + ((bits >> 16) & 1);
            let rounded = bits.wrapping_add(rounding);
            let hi = u16::try_from(rounded >> 16).unwrap_or(0);
            payload.extend_from_slice(&hi.to_le_bytes());
        }
        let header = json!({
            tensor_name: {
                "dtype": "BF16",
                "shape": [rows, cols],
                "data_offsets": [0_u64, u64::try_from(payload.len()).map_err(|e| e.to_string())?],
            }
        });
        let header_bytes = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
        let header_len = u64::try_from(header_bytes.len()).map_err(|e| e.to_string())?;
        let shard_name = "model-00001-of-00001.safetensors";
        let mut file = File::create(dir.join(shard_name)).map_err(|e| e.to_string())?;
        file.write_all(&header_len.to_le_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(&header_bytes).map_err(|e| e.to_string())?;
        file.write_all(&payload).map_err(|e| e.to_string())?;
        let index = json!({
            "metadata": {"total_size": 0},
            "weight_map": { tensor_name: shard_name },
        });
        fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&index).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }

    #[test]
    fn scope_shrinkage_validation_and_export_metadata_work() -> crate::Result<()> {
        let plan_default = super::super::run(json!({
            "schemaVersion": 1,
            "operation": "plan",
            "source": { "kind": "pinned-daisyrose" }
        }))?;
        assert_eq!(plan_default["scope"]["shrinkageMode"], "gavish-donoho");
        assert!(plan_default["scope"].get("shrinkageBlend").is_none());

        let plan_blended = super::super::run(json!({
            "schemaVersion": 1,
            "operation": "plan",
            "source": { "kind": "pinned-daisyrose" },
            "scope": { "shrinkageMode": "blended", "shrinkageBlend": 0.4 }
        }))?;
        assert_eq!(plan_blended["scope"]["shrinkageMode"], "blended");
        assert_eq!(plan_blended["scope"]["shrinkageBlend"], 0.4);

        for bad_scope in [
            json!({ "shrinkageMode": "blended" }),
            json!({ "shrinkageMode": "blended", "shrinkageBlend": -0.05 }),
            json!({ "shrinkageMode": "blended", "shrinkageBlend": 1.05 }),
            json!({ "shrinkageMode": "gavish-donoho", "shrinkageBlend": 0.5 }),
            json!({ "shrinkageMode": "unshrunk", "shrinkageBlend": 0.5 }),
        ] {
            assert!(
                super::super::run(json!({
                    "schemaVersion": 1,
                    "operation": "plan",
                    "source": { "kind": "pinned-daisyrose" },
                    "scope": bad_scope
                }))
                .is_err()
            );
        }

        let scratch = ScratchDir::new("shrinkage-export-meta")?;
        let base_dir = scratch.join("base");
        let cand_dir = scratch.join("cand");
        let rows = 64_usize;
        let cols = 80_usize;
        let (delta, _) = test_spiked_matrix(rows, cols, &[2.4, 1.7], 3.0);
        let base_vals = vec![1.0_f32; rows * cols];
        let cand_vals: Vec<f32> = base_vals.iter().zip(&delta).map(|(&b, &d)| b + d).collect();
        let hf_name = "model.language_model.layers.0.mlp.down_proj.weight";
        let gguf_name = "blk.0.ffn_down.weight";
        write_toy_safetensors_dir(&base_dir, hf_name, rows, cols, &base_vals)?;
        write_toy_safetensors_dir(&cand_dir, hf_name, rows, cols, &cand_vals)?;

        let mut toy_dims = BTreeMap::new();
        toy_dims.insert(gguf_name.to_owned(), (cols, rows, "ffn_down"));

        for (mode_str, blend_json) in [
            ("unshrunk", None),
            ("blended", Some(0.25_f64)),
            ("gavish-donoho", None),
        ] {
            let out_gguf = scratch.join(format!("export-{mode_str}.gguf"));
            let mut scope_json = json!({
                "layers": [0],
                "targetEnvelope": "residual-writers",
                "exportDtype": "f32",
                "maxRank": 8,
                "oversampling": 4,
                "powerIterations": 2,
                "shrinkageMode": mode_str,
            });
            if let Some(alpha) = blend_json
                && let Some(obj) = scope_json.as_object_mut()
            {
                obj.insert("shrinkageBlend".to_owned(), json!(alpha));
            }
            let req: super::super::Request = serde_json::from_value(json!({
                "schemaVersion": 1,
                "operation": "export-denoised",
                "source": {
                    "kind": "local-directories",
                    "candidateDirectory": cand_dir,
                    "baseDirectory": base_dir
                },
                "scope": scope_json,
                "controls": { "timeBudgetSeconds": 60, "workers": 2, "httpConcurrency": 1 },
                "outputPath": out_gguf
            }))
            .map_err(|e| e.to_string())?;
            super::super::validate_request(&req)?;
            let receipt = super::super::run_export_denoised_with_dims(&req, &toy_dims)?;
            assert_eq!(receipt["published"], true);
            assert_eq!(
                receipt["verification"]["provenanceMetadata"]["bonsai.composition.shrinkage_mode"],
                mode_str
            );
        }
        Ok(())
    }
}
