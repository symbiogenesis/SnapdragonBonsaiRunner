//! Deterministic exact and randomized SVD spectrum estimators for BF16 weight diffs.

use std::f64::consts::PI;

use serde::Serialize;

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

/// Spectral summary computed for one matrix delta.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SvdSpectrum {
    pub(super) method: String,
    pub(super) effective_sketch_rank: usize,
    pub(super) sketch_energy_ratio: f64,
    pub(super) top_singular_values: Vec<f64>,
    pub(super) cumulative_energy: CumulativeEnergy,
}

impl SvdSpectrum {
    pub(super) fn zero() -> Self {
        Self {
            method: "zero-delta-skipped".to_owned(),
            effective_sketch_rank: 0,
            sketch_energy_ratio: 0.0,
            top_singular_values: Vec::new(),
            cumulative_energy: CumulativeEnergy::default(),
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
/// Returns nonnegative eigenvalues in descending order.
pub(super) fn jacobi_eigenvalues_f64(matrix: &mut [f64], n: usize) -> crate::Result<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
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
    if diag_scale == 0.0 {
        return Ok(vec![0.0; n]);
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
                    if r == p || r == q {
                        continue;
                    }
                    let a_rp = matrix[r * n + p];
                    let a_rq = matrix[r * n + q];
                    let next_rp = a_rp - s * (a_rq + rho * a_rp);
                    let next_rq = a_rq + s * (a_rp - rho * a_rq);
                    matrix[r * n + p] = next_rp;
                    matrix[p * n + r] = next_rp;
                    matrix[r * n + q] = next_rq;
                    matrix[q * n + r] = next_rq;
                }
            }
        }
    }
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| matrix[i * n + i].max(0.0)).collect();
    eigenvalues.sort_by(f64::total_cmp);
    eigenvalues.reverse();
    Ok(eigenvalues)
}

fn spectrum_from_gram(
    method: String,
    gram: &mut [f64],
    dim: usize,
    total_energy: f64,
    verify_projection_bound: bool,
) -> crate::Result<SvdSpectrum> {
    let eigenvalues = jacobi_eigenvalues_f64(gram, dim)?;
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
    let cumulative_energy = CumulativeEnergy::from_eigenvalues(&eigenvalues, total_energy);
    Ok(SvdSpectrum {
        method,
        effective_sketch_rank: dim,
        sketch_energy_ratio: ratio,
        top_singular_values,
        cumulative_energy,
    })
}

/// Computes exact SVD spectrum via smaller-dimension Gram matrix when `min(m, n) <= k_sketch`.
pub(super) fn exact_gram_svd(
    d: &[f32],
    m: usize,
    n: usize,
    total_energy: f64,
) -> crate::Result<SvdSpectrum> {
    if total_energy == 0.0 {
        return Ok(SvdSpectrum::zero());
    }
    let dim = m.min(n);
    let mut gram = vec![0.0_f64; dim * dim];
    if m <= n {
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
    spectrum_from_gram(
        "exact-gram-jacobi-f64".to_owned(),
        &mut gram,
        dim,
        total_energy,
        true,
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
        return exact_gram_svd(d, m, n, total_energy);
    }

    let omega = fill_gaussian_f32(name, n, k)?;
    let y = matmul_d_x(d, m, n, &omega, k, workers)?;
    let mut basis = orthonormalize_columns_f64(m, k, |r, j| f64::from(y[r * k + j]))?;
    if basis.is_empty() {
        return Ok(SvdSpectrum::zero());
    }
    let mut q = basis_cols_to_row_major_f32(&basis, m)?;
    let mut k_eff = basis.len();

    for _ in 0..power_iterations {
        let z = matmul_dt_q_f64(d, m, n, &q, k_eff, workers)?;
        let z_basis = orthonormalize_columns_f64(n, k_eff, |r, j| z[r * k_eff + j])?;
        if z_basis.is_empty() {
            return Ok(SvdSpectrum::zero());
        }
        let k_z = z_basis.len();
        let q_z = basis_cols_to_row_major_f32(&z_basis, n)?;
        let y_next = matmul_d_x(d, m, n, &q_z, k_z, workers)?;
        basis = orthonormalize_columns_f64(m, k_z, |r, j| f64::from(y_next[r * k_z + j]))?;
        if basis.is_empty() {
            return Ok(SvdSpectrum::zero());
        }
        k_eff = basis.len();
        q = basis_cols_to_row_major_f32(&basis, m)?;
    }

    let b_t = matmul_dt_q_f64(d, m, n, &q, k_eff, workers)?;
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
    let method = format!("randomized-subspace-q{power_iterations}-k{k}-f32-gram-f64");
    spectrum_from_gram(method, &mut gram, k_eff, total_energy, true)
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

    pub(super) fn finish(self, total_energy: f64, workers: usize) -> crate::Result<SvdSpectrum> {
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
        spectrum_from_gram(method, &mut gram, k_eff, total_energy, false)
    }
}
