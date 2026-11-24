//! Activation-guided pure-Rust multi-rank LoRA editing (`fold-patch`, `realign-factors`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::calibration_metrics::RowMetricsAccumulator;

use super::gguf_adapter::{
    ExportDtype, ExportTensorPair, classify_families, encode_f16_le,
    read_lora_gguf_pairs_with_dims, sort_pairs_canonically, swift_base_dims_map, write_lora_gguf,
};
use super::safetensors::{FloatDtype, decode_scalar_f32};
use super::transform::{
    check_factor_value, effective_delta_frobenius_norm_f64, verify_published_adapter,
};
use super::{Operation, Request, usize_to_f32};

const MAX_FOLD_PATCHES: usize = 64;
const MAX_LORA_RANK: usize = 64;
const MAX_PATCH_FILE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_REALIGN_SITES: usize = 16;
const MIN_REALIGN_ROWS: usize = 3;
const MAX_REALIGN_ROWS: usize = 256;
const MAX_REALIGN_ROWS_FILE_BYTES: u64 = 48 * 1024 * 1024;
const MAX_REALIGN_WORK: u64 = 500_000_000;
const DEFAULT_REALIGN_RIDGE: f64 = 1e-3;
const DEFAULT_ALTERNATING_STEPS: usize = 2;
const DEFAULT_REALIGN_TIME_BUDGET_MS: u64 = 10_000;
const MAX_REALIGN_TIME_BUDGET_MS: u64 = 30_000;
const ALLOWED_LINEAR_FAMILIES: [&str; 9] = [
    "ffn_down",
    "attn_output",
    "attn_q",
    "attn_k",
    "attn_v",
    "ffn_gate",
    "ffn_up",
    "attn_qkv",
    "attn_gate",
];

#[expect(
    clippy::as_conversions,
    reason = "Small LoRA rank/index integers convert directly to f64 for exact scaling."
)]
#[inline]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Composition strategy when folding rank-one patches into a multi-rank adapter.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum FoldMode {
    #[default]
    Append,
    FoldSvd,
}

impl FoldMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Append => "append",
            Self::FoldSvd => "fold-svd",
        }
    }

    const fn provenance_method(self) -> &'static str {
        match self {
            Self::Append => "activation-fold-append-v1",
            Self::FoldSvd => "activation-fold-svd-v1",
        }
    }
}

/// Rank-one patch input entry for `fold-patch`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub(super) enum FoldPatchEntry {
    WeightPatchFile {
        patch_path: PathBuf,
        #[serde(default)]
        patch_scale: Option<f64>,
    },
    InlineRankOne {
        target: String,
        a: Vec<f32>,
        b: Vec<f32>,
        scale: f64,
    },
}

/// Configuration payload for `fold-patch`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct FoldPatchConfig {
    pub(super) mode: FoldMode,
    pub(super) target_max_rank: Option<usize>,
    pub(super) export_dtype: Option<ExportDtype>,
    pub(super) diagnostic_prompt_sha256s: Vec<String>,
    pub(super) patches: Vec<FoldPatchEntry>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WeightPatchSite {
    tensor: String,
    input_width: usize,
    output_width: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WeightPatchFactors {
    a: Vec<f32>,
    b: Vec<f32>,
    a_sha256: String,
    b_sha256: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WeightPatchArtifact {
    schema_version: u32,
    kind: String,
    site: WeightPatchSite,
    scale_bits: String,
    factors: WeightPatchFactors,
}

struct ResolvedRankOnePatch {
    target: String,
    u_col: Vec<f64>,
    v_row: Vec<f64>,
}

struct PendingSiteFactors {
    gguf_base_name: String,
    in_features: usize,
    out_features: usize,
    rank: usize,
    family: &'static str,
    a_f64: Vec<f64>,
    b_f64: Vec<f64>,
}

struct ThinQr {
    q_cols: Vec<Vec<f64>>,
    r_mat: Vec<f64>,
}

#[inline]
fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&x, &y)| x * y).sum()
}

fn validate_diagnostic_prompt_digests(
    digests: &[String],
    field_name: &str,
) -> crate::Result<Vec<String>> {
    if digests.is_empty() {
        return Err(format!(
            "{field_name}.diagnosticPromptSha256s must be non-empty so holdout split disjointness can be audited."
        ));
    }
    let mut canonical = BTreeSet::new();
    for digest in digests {
        if !crate::identity::is_lower_hex(digest, 64) {
            return Err(format!(
                "{field_name}.diagnosticPromptSha256s must contain only lowercase 64-hex SHA-256 digests."
            ));
        }
        canonical.insert(digest.clone());
    }
    Ok(canonical.into_iter().collect())
}

fn f32_le_sha256(values: &[f32]) -> crate::Result<String> {
    let mut hasher = crate::identity::Hasher::new()?;
    for &value in values {
        hasher.update(&value.to_le_bytes())?;
    }
    hasher.finalize_hex()
}

fn validate_target_linear_dims(
    target: &str,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<(usize, usize, &'static str)> {
    let &(in_features, out_features, family) = base_dims
        .get(target)
        .ok_or_else(|| format!("Unknown Swift target tensor '{target}'."))?;
    if !ALLOWED_LINEAR_FAMILIES.contains(&family) {
        return Err(format!(
            "Target tensor '{target}' family '{family}' is not an admissible linear LoRA site."
        ));
    }
    if in_features == 0 || out_features == 0 {
        return Err(format!("Target tensor '{target}' has zero dimension."));
    }
    Ok((in_features, out_features, family))
}

fn build_scaled_rank_one(
    a: &[f32],
    b: &[f32],
    scale: f64,
    in_features: usize,
    out_features: usize,
    label: &str,
) -> crate::Result<(Vec<f64>, Vec<f64>)> {
    if a.len() != in_features || b.len() != out_features {
        return Err(format!(
            "{label} dimensions ({}, {}) mismatch target ({in_features}, {out_features}).",
            a.len(),
            b.len()
        ));
    }
    if !scale.is_finite() {
        return Err(format!("{label} effective scale must be finite."));
    }
    if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
        return Err(format!("{label} factors contain nonfinite values."));
    }
    let mag = scale.abs().sqrt();
    let sign = if scale < 0.0 { -1.0_f64 } else { 1.0_f64 };
    let u_col: Vec<f64> = b.iter().map(|&bv| mag * f64::from(bv)).collect();
    let v_row: Vec<f64> = a.iter().map(|&av| (sign * mag) * f64::from(av)).collect();
    if u_col.iter().any(|v| !v.is_finite()) || v_row.iter().any(|v| !v.is_finite()) {
        return Err(format!("{label} scaled factors overflowed f64."));
    }
    Ok((u_col, v_row))
}

pub(super) fn validate_fold_patch_config(config: &FoldPatchConfig) -> crate::Result<()> {
    match config.mode {
        FoldMode::FoldSvd => {
            let max_r = config
                .target_max_rank
                .ok_or("foldPatchConfig.targetMaxRank is required in 1..=64 for fold-svd mode.")?;
            if !(1..=MAX_LORA_RANK).contains(&max_r) {
                return Err(format!(
                    "foldPatchConfig.targetMaxRank must be in 1..={MAX_LORA_RANK}, got {max_r}."
                ));
            }
        }
        FoldMode::Append => {
            if let Some(max_r) = config.target_max_rank
                && !(1..=MAX_LORA_RANK).contains(&max_r)
            {
                return Err(format!(
                    "foldPatchConfig.targetMaxRank must be in 1..={MAX_LORA_RANK}, got {max_r}."
                ));
            }
        }
    }
    let _ =
        validate_diagnostic_prompt_digests(&config.diagnostic_prompt_sha256s, "foldPatchConfig")?;
    if !(1..=MAX_FOLD_PATCHES).contains(&config.patches.len()) {
        return Err(format!(
            "foldPatchConfig.patches must contain 1..={MAX_FOLD_PATCHES} entries."
        ));
    }
    for (idx, entry) in config.patches.iter().enumerate() {
        match entry {
            FoldPatchEntry::WeightPatchFile {
                patch_path,
                patch_scale,
            } => {
                bonsai_artifacts::validate_external_path(patch_path).map_err(|e| e.to_string())?;
                if patch_scale.is_some_and(|s| !s.is_finite()) {
                    return Err(format!(
                        "foldPatchConfig.patches[{idx}].patchScale must be finite."
                    ));
                }
            }
            FoldPatchEntry::InlineRankOne {
                target,
                a,
                b,
                scale,
            } => {
                if target.trim().is_empty() {
                    return Err(format!(
                        "foldPatchConfig.patches[{idx}].target must not be empty."
                    ));
                }
                if a.is_empty() || b.is_empty() {
                    return Err(format!(
                        "foldPatchConfig.patches[{idx}] factors a and b must be non-empty."
                    ));
                }
                if !scale.is_finite() {
                    return Err(format!(
                        "foldPatchConfig.patches[{idx}].scale must be finite."
                    ));
                }
                if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
                    return Err(format!(
                        "foldPatchConfig.patches[{idx}] contains nonfinite factor elements."
                    ));
                }
            }
        }
    }
    Ok(())
}

fn load_patch_entry(
    entry: &FoldPatchEntry,
    idx: usize,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<ResolvedRankOnePatch> {
    match entry {
        FoldPatchEntry::WeightPatchFile {
            patch_path,
            patch_scale,
        } => {
            bonsai_artifacts::validate_external_path(patch_path).map_err(|e| e.to_string())?;
            let artifact: WeightPatchArtifact = crate::bounded_input::read_regular_json(
                patch_path,
                MAX_PATCH_FILE_BYTES,
                "Weight patch JSON exceeds 4 MiB.",
            )?;
            if artifact.schema_version != 1 || artifact.kind != "experimental-ffn-rank-one-patch" {
                return Err(format!(
                    "foldPatchConfig.patches[{idx}] is not a schemaVersion=1 experimental-ffn-rank-one-patch JSON."
                ));
            }
            if !crate::identity::is_lower_hex(&artifact.scale_bits, 8) {
                return Err(format!(
                    "foldPatchConfig.patches[{idx}] has noncanonical scaleBits."
                ));
            }
            let base_scale = f32::from_bits(
                u32::from_str_radix(&artifact.scale_bits, 16).map_err(|e| e.to_string())?,
            );
            if !base_scale.is_finite() {
                return Err(format!(
                    "foldPatchConfig.patches[{idx}] scaleBits decoded to nonfinite value."
                ));
            }
            let multiplier = patch_scale.unwrap_or(1.0);
            if !multiplier.is_finite() {
                return Err(format!(
                    "foldPatchConfig.patches[{idx}].patchScale must be finite."
                ));
            }
            let (in_features, out_features, _family) =
                validate_target_linear_dims(&artifact.site.tensor, base_dims)?;
            if artifact.site.input_width != in_features
                || artifact.site.output_width != out_features
            {
                return Err(format!(
                    "foldPatchConfig.patches[{idx}] site dimensions ({}, {}) mismatch target '{}' ({in_features}, {out_features}).",
                    artifact.site.input_width, artifact.site.output_width, artifact.site.tensor
                ));
            }
            if !crate::identity::is_lower_hex(&artifact.factors.a_sha256, 64)
                || !crate::identity::is_lower_hex(&artifact.factors.b_sha256, 64)
            {
                return Err(format!(
                    "foldPatchConfig.patches[{idx}] factor digests must be lowercase 64-hex."
                ));
            }
            if artifact.factors.a.iter().any(|v| !v.is_finite())
                || artifact.factors.b.iter().any(|v| !v.is_finite())
            {
                return Err(format!(
                    "foldPatchConfig.patches[{idx}] contains nonfinite factor entries."
                ));
            }
            if artifact.factors.a.len() != in_features || artifact.factors.b.len() != out_features {
                return Err(format!(
                    "foldPatchConfig.patches[{idx}] factor lengths ({}, {}) mismatch ({in_features}, {out_features}).",
                    artifact.factors.a.len(),
                    artifact.factors.b.len()
                ));
            }
            let actual_a_sha = f32_le_sha256(&artifact.factors.a)?;
            if actual_a_sha != artifact.factors.a_sha256 {
                return Err(format!(
                    "foldPatchConfig.patches[{idx}] factors.aSha256 mismatch."
                ));
            }
            let actual_b_sha = f32_le_sha256(&artifact.factors.b)?;
            if actual_b_sha != artifact.factors.b_sha256 {
                return Err(format!(
                    "foldPatchConfig.patches[{idx}] factors.bSha256 mismatch."
                ));
            }
            let s_eff = f64::from(base_scale) * multiplier;
            let (u_col, v_row) = build_scaled_rank_one(
                &artifact.factors.a,
                &artifact.factors.b,
                s_eff,
                in_features,
                out_features,
                &format!("foldPatchConfig.patches[{idx}]"),
            )?;
            Ok(ResolvedRankOnePatch {
                target: artifact.site.tensor,
                u_col,
                v_row,
            })
        }
        FoldPatchEntry::InlineRankOne {
            target,
            a,
            b,
            scale,
        } => {
            let (in_features, out_features, _family) =
                validate_target_linear_dims(target, base_dims)?;
            let (u_col, v_row) = build_scaled_rank_one(
                a,
                b,
                *scale,
                in_features,
                out_features,
                &format!("foldPatchConfig.patches[{idx}]"),
            )?;
            Ok(ResolvedRankOnePatch {
                target: target.clone(),
                u_col,
                v_row,
            })
        }
    }
}

/// Two-pass Modified Gram-Schmidt thin QR decomposition `M[dim x k] = Q[dim x k_eff] * R[k_eff x k]`.
fn thin_qr_f64(cols: &[Vec<f64>]) -> crate::Result<ThinQr> {
    let k = cols.len();
    if k == 0 {
        return Ok(ThinQr {
            q_cols: Vec::new(),
            r_mat: Vec::new(),
        });
    }
    let mut max_norm = 0.0_f64;
    for col in cols {
        let norm_sq = dot_f64(col, col);
        if !norm_sq.is_finite() {
            return Err("Nonfinite column norm during thin QR.".into());
        }
        max_norm = max_norm.max(norm_sq.sqrt());
    }
    if max_norm == 0.0 {
        return Ok(ThinQr {
            q_cols: Vec::new(),
            r_mat: Vec::new(),
        });
    }
    let threshold = max_norm * 1e-12;
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut r_rows: Vec<Vec<f64>> = Vec::with_capacity(k);

    for (j, src_col) in cols.iter().enumerate() {
        let mut v = src_col.clone();
        for _pass in 0..2 {
            for (i, q_i) in q_cols.iter().enumerate() {
                let proj = dot_f64(&v, q_i);
                r_rows[i][j] += proj;
                for (dst, &qv) in v.iter_mut().zip(q_i) {
                    *dst -= proj * qv;
                }
            }
        }
        let norm = dot_f64(&v, &v).sqrt();
        if norm.is_finite() && norm > threshold {
            let inv = 1.0 / norm;
            for val in &mut v {
                *val *= inv;
            }
            let mut new_row = vec![0.0_f64; k];
            new_row[j] = norm;
            q_cols.push(v);
            r_rows.push(new_row);
        }
    }

    let k_eff = q_cols.len();
    let mut r_mat = Vec::with_capacity(k_eff * k);
    for row in r_rows {
        r_mat.extend_from_slice(&row);
    }
    Ok(ThinQr { q_cols, r_mat })
}

struct FoldSvdResult {
    rank: usize,
    a_f64: Vec<f64>,
    b_f64: Vec<f64>,
    total_energy_sq: f64,
    frobenius_reconstruction_error: f64,
    retained_energy_fraction: f64,
    singular_values: Vec<f64>,
}

fn fold_svd_site(
    b_cols: &[Vec<f64>],
    a_rows: &[Vec<f64>],
    in_features: usize,
    out_features: usize,
    target_max_rank: usize,
) -> crate::Result<FoldSvdResult> {
    let k = b_cols.len();
    if k == 0 || a_rows.len() != k {
        return Err("fold-svd requires matching non-empty B and A factor lists.".into());
    }
    let qr_b = thin_qr_f64(b_cols)?;
    let qr_a = thin_qr_f64(a_rows)?;
    let k_b = qr_b.q_cols.len();
    let k_a = qr_a.q_cols.len();
    if k_b == 0 || k_a == 0 {
        return Ok(FoldSvdResult {
            rank: 1,
            a_f64: vec![0.0_f64; in_features],
            b_f64: vec![0.0_f64; out_features],
            total_energy_sq: 0.0,
            frobenius_reconstruction_error: 0.0,
            retained_energy_fraction: 1.0,
            singular_values: Vec::new(),
        });
    }

    // M[k_b x k_a] = R_B * R_A^T
    let mut m_mat = vec![0.0_f64; k_b * k_a];
    let mut total_energy_sq = 0.0_f64;
    for i in 0..k_b {
        let rb_row = &qr_b.r_mat[i * k..(i + 1) * k];
        for j in 0..k_a {
            let ra_row = &qr_a.r_mat[j * k..(j + 1) * k];
            let val = dot_f64(rb_row, ra_row);
            m_mat[i * k_a + j] = val;
            total_energy_sq += val * val;
        }
    }
    if !total_energy_sq.is_finite() {
        return Err("Core matrix Frobenius energy overflowed f64 in fold-svd.".into());
    }
    if total_energy_sq == 0.0 {
        return Ok(FoldSvdResult {
            rank: 1,
            a_f64: vec![0.0_f64; in_features],
            b_f64: vec![0.0_f64; out_features],
            total_energy_sq: 0.0,
            frobenius_reconstruction_error: 0.0,
            retained_energy_fraction: 1.0,
            singular_values: Vec::new(),
        });
    }

    // Symmetric k_b x k_b Gram matrix G = M * M^T
    let mut gram = vec![0.0_f64; k_b * k_b];
    for i in 0..k_b {
        let row_i = &m_mat[i * k_a..(i + 1) * k_a];
        for j in i..k_b {
            let row_j = &m_mat[j * k_a..(j + 1) * k_a];
            let dot = dot_f64(row_i, row_j);
            gram[i * k_b + j] = dot;
            gram[j * k_b + i] = dot;
        }
    }

    let (eigenvalues, w_cols) = super::svd::jacobi_eigendecomp_f64(&mut gram, k_b)?;
    let max_lambda = eigenvalues.first().copied().unwrap_or(0.0);
    let max_sigma = max_lambda.max(0.0).sqrt();
    let singular_values: Vec<f64> = eigenvalues
        .iter()
        .take(k_b.min(k_a))
        .map(|&l| l.max(0.0).sqrt())
        .collect();

    let mut kept_w: Vec<Vec<f64>> = Vec::with_capacity(target_max_rank.min(k_b));
    let mut kept_z: Vec<Vec<f64>> = Vec::with_capacity(target_max_rank.min(k_a));
    let mut kept_sigmas: Vec<f64> = Vec::with_capacity(target_max_rank.min(k_b));

    for (lambda, w) in eigenvalues.into_iter().zip(w_cols) {
        if kept_sigmas.len() >= target_max_rank {
            break;
        }
        if !lambda.is_finite() || lambda <= 0.0 || lambda <= max_lambda * 1e-24 {
            break;
        }
        let mut z = vec![0.0_f64; k_a];
        for i in 0..k_b {
            let wi = w[i];
            if wi == 0.0 {
                continue;
            }
            let m_row = &m_mat[i * k_a..(i + 1) * k_a];
            for (dst, &mv) in z.iter_mut().zip(m_row) {
                *dst += wi * mv;
            }
        }
        for prev_z in &kept_z {
            let proj = dot_f64(&z, prev_z);
            for (dst, &pz) in z.iter_mut().zip(prev_z) {
                *dst -= proj * pz;
            }
        }
        let z_norm = dot_f64(&z, &z).sqrt();
        if !z_norm.is_finite() || z_norm <= max_sigma * 1e-12 {
            continue;
        }
        let inv_z = 1.0 / z_norm;
        for val in &mut z {
            *val *= inv_z;
        }
        let mut sigma = 0.0_f64;
        for i in 0..k_b {
            let m_row = &m_mat[i * k_a..(i + 1) * k_a];
            sigma += w[i] * dot_f64(m_row, &z);
        }
        if sigma < 0.0 {
            sigma = -sigma;
            for val in &mut z {
                *val = -*val;
            }
        }
        if !sigma.is_finite() || sigma <= max_sigma * 1e-12 {
            continue;
        }
        kept_w.push(w);
        kept_z.push(z);
        kept_sigmas.push(sigma);
    }

    if kept_sigmas.is_empty() {
        return Ok(FoldSvdResult {
            rank: 1,
            a_f64: vec![0.0_f64; in_features],
            b_f64: vec![0.0_f64; out_features],
            total_energy_sq: 0.0,
            frobenius_reconstruction_error: 0.0,
            retained_energy_fraction: 1.0,
            singular_values,
        });
    }

    let kept_rank = kept_sigmas.len();
    let kept_energy: f64 = kept_sigmas.iter().map(|&s| s * s).sum();
    let frobenius_reconstruction_error = (total_energy_sq - kept_energy).max(0.0).sqrt();
    let retained_energy_fraction = (kept_energy / total_energy_sq).clamp(0.0, 1.0);

    let mut a_f64 = Vec::new();
    a_f64
        .try_reserve_exact(
            kept_rank
                .checked_mul(in_features)
                .ok_or("fold-svd lora_a length overflow.")?,
        )
        .map_err(|e| format!("Cannot allocate fold-svd lora_a buffer: {e}"))?;
    a_f64.resize(kept_rank * in_features, 0.0_f64);

    let mut b_f64 = Vec::new();
    b_f64
        .try_reserve_exact(
            out_features
                .checked_mul(kept_rank)
                .ok_or("fold-svd lora_b length overflow.")?,
        )
        .map_err(|e| format!("Cannot allocate fold-svd lora_b buffer: {e}"))?;
    b_f64.resize(out_features * kept_rank, 0.0_f64);

    for (mode_idx, ((w, z), &sigma)) in kept_w.iter().zip(&kept_z).zip(&kept_sigmas).enumerate() {
        let root_sigma = sigma.sqrt();
        let a_dst = &mut a_f64[mode_idx * in_features..(mode_idx + 1) * in_features];
        for (b_idx, &zb) in z.iter().enumerate() {
            if zb == 0.0 {
                continue;
            }
            let scale = root_sigma * zb;
            let q_col = &qr_a.q_cols[b_idx];
            for (dst, &qv) in a_dst.iter_mut().zip(q_col) {
                *dst += scale * qv;
            }
        }

        for (a_idx, &wa) in w.iter().enumerate() {
            if wa == 0.0 {
                continue;
            }
            let scale = root_sigma * wa;
            let q_col = &qr_b.q_cols[a_idx];
            for (row_idx, &qv) in q_col.iter().enumerate() {
                b_f64[row_idx * kept_rank + mode_idx] += scale * qv;
            }
        }
    }

    Ok(FoldSvdResult {
        rank: kept_rank,
        a_f64,
        b_f64,
        total_energy_sq,
        frobenius_reconstruction_error,
        retained_energy_fraction,
        singular_values,
    })
}

fn export_pending_sites(
    pending: Vec<PendingSiteFactors>,
    new_alpha: f32,
    out_dtype: ExportDtype,
) -> crate::Result<Vec<ExportTensorPair>> {
    let mut final_pairs = Vec::with_capacity(pending.len());
    for item in pending {
        let _ = item.family;
        let s2 = (usize_to_f64(item.rank) / f64::from(new_alpha)).sqrt();
        let mut lora_a_f32 = Vec::with_capacity(item.a_f64.len());
        for v in item.a_f64 {
            lora_a_f32.push(check_factor_value(v * s2, out_dtype, "lora_a")?);
        }
        let mut lora_b_f32 = Vec::with_capacity(item.b_f64.len());
        for v in item.b_f64 {
            lora_b_f32.push(check_factor_value(v * s2, out_dtype, "lora_b")?);
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
    Ok(final_pairs)
}

/// Runs `fold-patch` against the built-in Swift base tensor dimensions.
pub(super) fn run_fold_patch(request: &Request) -> crate::Result<Value> {
    run_fold_patch_with_dims(request, &swift_base_dims_map())
}

/// Runs `fold-patch` with explicit base tensor dimensions (production and test entry point).
#[expect(
    clippy::too_many_lines,
    reason = "Coordinates patch resolution, append/fold-svd composition, GGUF export, and verification."
)]
pub(super) fn run_fold_patch_with_dims(
    request: &Request,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<Value> {
    if request.operation != Operation::FoldPatch {
        return Err("Invalid operation dispatch for fold-patch.".into());
    }
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("fold-patch requires outputPath.")?;
    let config = request
        .fold_patch_config
        .as_ref()
        .ok_or("fold-patch requires foldPatchConfig.")?;
    validate_fold_patch_config(config)?;
    let training_prompt_sha256s =
        validate_diagnostic_prompt_digests(&config.diagnostic_prompt_sha256s, "foldPatchConfig")?;

    let loaded = match &request.adapter_path {
        Some(path) => Some(read_lora_gguf_pairs_with_dims(path, base_dims)?),
        None => None,
    };

    let mut patches_by_target: BTreeMap<String, Vec<ResolvedRankOnePatch>> = BTreeMap::new();
    for (idx, entry) in config.patches.iter().enumerate() {
        let resolved = load_patch_entry(entry, idx, base_dims)?;
        patches_by_target
            .entry(resolved.target.clone())
            .or_default()
            .push(resolved);
    }

    let out_dtype = config
        .export_dtype
        .unwrap_or_else(|| loaded.as_ref().map_or(ExportDtype::F16, |a| a.dtype));

    let mut source_by_target: BTreeMap<String, ExportTensorPair> = BTreeMap::new();
    let mut provenance_meta: BTreeMap<String, String> = BTreeMap::new();
    let mut source_sha256: Option<String> = None;
    let mut source_alpha: Option<f32> = None;
    if let Some(src) = loaded {
        source_sha256 = Some(src.sha256);
        source_alpha = Some(src.alpha);
        provenance_meta = src.provenance_meta;
        for pair in src.pairs {
            source_by_target.insert(pair.gguf_base_name.clone(), pair);
        }
    }

    let mut all_targets: BTreeSet<String> = source_by_target.keys().cloned().collect();
    all_targets.extend(patches_by_target.keys().cloned());
    if all_targets.is_empty() {
        return Err("fold-patch produced zero output tensor pairs.".into());
    }

    let mut pending: Vec<PendingSiteFactors> = Vec::with_capacity(all_targets.len());
    let mut folded_sites: Vec<String> = Vec::new();
    let mut per_site_diagnostics = serde_json::Map::new();
    let mut families_used = BTreeSet::new();
    let mut max_new_rank = 1_usize;

    for target in all_targets {
        let (in_features, out_features, family) = validate_target_linear_dims(&target, base_dims)?;
        families_used.insert(family);

        let src_pair = source_by_target.remove(&target);
        let site_patches = patches_by_target.remove(&target);

        let source_rank = src_pair.as_ref().map_or(0, |p| p.rank);
        let mut b_cols: Vec<Vec<f64>> =
            Vec::with_capacity(source_rank + site_patches.as_ref().map_or(0, Vec::len));
        let mut a_rows: Vec<Vec<f64>> =
            Vec::with_capacity(source_rank + site_patches.as_ref().map_or(0, Vec::len));

        if let (Some(pair), Some(alpha)) = (src_pair, source_alpha) {
            let s1 = (f64::from(alpha) / usize_to_f64(pair.rank)).sqrt();
            for k in 0..pair.rank {
                let mut col_b = Vec::with_capacity(out_features);
                for row in 0..out_features {
                    col_b.push(f64::from(pair.lora_b_f32[row * pair.rank + k]) * s1);
                }
                b_cols.push(col_b);
                let row_off = k * in_features;
                let row_a: Vec<f64> = pair.lora_a_f32[row_off..row_off + in_features]
                    .iter()
                    .map(|&v| f64::from(v) * s1)
                    .collect();
                a_rows.push(row_a);
            }
        }

        let Some(patches) = site_patches else {
            // Untouched source site: preserve factors at existing rank.
            let mut a_f64 = Vec::with_capacity(source_rank * in_features);
            for row in a_rows {
                a_f64.extend_from_slice(&row);
            }
            let mut b_f64 = vec![0.0_f64; out_features * source_rank];
            for (k, col_b) in b_cols.iter().enumerate() {
                for row in 0..out_features {
                    b_f64[row * source_rank + k] = col_b[row];
                }
            }
            max_new_rank = max_new_rank.max(source_rank);
            pending.push(PendingSiteFactors {
                gguf_base_name: target,
                in_features,
                out_features,
                rank: source_rank,
                family,
                a_f64,
                b_f64,
            });
            continue;
        };

        let patch_count = patches.len();
        for patch in patches {
            b_cols.push(patch.u_col);
            a_rows.push(patch.v_row);
        }
        let combined_rank = b_cols.len();
        folded_sites.push(target.clone());

        match config.mode {
            FoldMode::Append => {
                let ceiling = config
                    .target_max_rank
                    .unwrap_or(MAX_LORA_RANK)
                    .min(MAX_LORA_RANK);
                if combined_rank > ceiling {
                    return Err(format!(
                        "Combined rank {combined_rank} for '{target}' exceeds append ceiling {ceiling}."
                    ));
                }
                let mut a_f64 = Vec::new();
                a_f64
                    .try_reserve_exact(
                        combined_rank
                            .checked_mul(in_features)
                            .ok_or("Appended lora_a size overflow.")?,
                    )
                    .map_err(|e| format!("Cannot allocate appended lora_a: {e}"))?;
                for row in a_rows {
                    a_f64.extend_from_slice(&row);
                }
                let mut b_f64 = Vec::new();
                b_f64
                    .try_reserve_exact(
                        out_features
                            .checked_mul(combined_rank)
                            .ok_or("Appended lora_b size overflow.")?,
                    )
                    .map_err(|e| format!("Cannot allocate appended lora_b: {e}"))?;
                b_f64.resize(out_features * combined_rank, 0.0_f64);
                for (k, col_b) in b_cols.iter().enumerate() {
                    for row in 0..out_features {
                        b_f64[row * combined_rank + k] = col_b[row];
                    }
                }
                let frob = effective_delta_frobenius_norm_f64(
                    &a_f64,
                    &b_f64,
                    in_features,
                    out_features,
                    combined_rank,
                )?;
                per_site_diagnostics.insert(
                    target.clone(),
                    json!({
                        "target": target,
                        "sourceRank": source_rank,
                        "patchCount": patch_count,
                        "combinedRank": combined_rank,
                        "finalRank": combined_rank,
                        "createdSite": source_rank == 0,
                        "totalEnergyFrobeniusSquared": frob * frob,
                        "frobeniusReconstructionError": 0.0,
                        "retainedEnergyFraction": 1.0,
                    }),
                );
                max_new_rank = max_new_rank.max(combined_rank);
                pending.push(PendingSiteFactors {
                    gguf_base_name: target,
                    in_features,
                    out_features,
                    rank: combined_rank,
                    family,
                    a_f64,
                    b_f64,
                });
            }
            FoldMode::FoldSvd => {
                let target_cap = config
                    .target_max_rank
                    .ok_or("foldPatchConfig.targetMaxRank is required for fold-svd.")?;
                let svd_out =
                    fold_svd_site(&b_cols, &a_rows, in_features, out_features, target_cap)?;
                per_site_diagnostics.insert(
                    target.clone(),
                    json!({
                        "target": target,
                        "sourceRank": source_rank,
                        "patchCount": patch_count,
                        "combinedRank": combined_rank,
                        "finalRank": svd_out.rank,
                        "createdSite": source_rank == 0,
                        "totalEnergyFrobeniusSquared": svd_out.total_energy_sq,
                        "frobeniusReconstructionError": svd_out.frobenius_reconstruction_error,
                        "retainedEnergyFraction": svd_out.retained_energy_fraction,
                        "singularValues": svd_out.singular_values,
                    }),
                );
                max_new_rank = max_new_rank.max(svd_out.rank);
                pending.push(PendingSiteFactors {
                    gguf_base_name: target,
                    in_features,
                    out_features,
                    rank: svd_out.rank,
                    family,
                    a_f64: svd_out.a_f64,
                    b_f64: svd_out.b_f64,
                });
            }
        }
    }

    let new_alpha = usize_to_f32(max_new_rank);
    let final_pairs = export_pending_sites(pending, new_alpha, out_dtype)?;

    let mut meta: Vec<(String, String)> = provenance_meta
        .into_iter()
        .filter(|(k, _)| {
            k != "bonsai.composition.fold_parent_sha256" && k != "bonsai.composition.fold_method"
        })
        .collect();
    if let Some(parent_sha) = &source_sha256 {
        meta.push((
            "bonsai.composition.fold_parent_sha256".to_owned(),
            parent_sha.clone(),
        ));
    }
    meta.push((
        "bonsai.composition.fold_method".to_owned(),
        config.mode.provenance_method().to_owned(),
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
        "operation": "fold-patch",
        "published": true,
        "status": "published-and-verified",
        "mode": config.mode.as_str(),
        "sourceAdapterPath": request.adapter_path,
        "sourceAdapterSha256": source_sha256,
        "outputPath": output_path,
        "targetEnvelope": expected_env,
        "exportDtype": out_dtype,
        "alpha": new_alpha,
        "minRank": verification["minRank"],
        "maxRank": verification["maxRank"],
        "bytes": written.bytes,
        "sha256": written.sha256,
        "tensorCount": written.tensor_count,
        "pairCount": final_pairs.len(),
        "trainingPromptSha256s": training_prompt_sha256s,
        "foldedSites": folded_sites,
        "perSiteDiagnostics": per_site_diagnostics,
        "verification": verification,
    }))
}

/// Closed-form ridge factor realignment strategy for `realign-factors`.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum RealignMode {
    #[default]
    RealignB,
    RealignA,
    AlternatingAb,
}

impl RealignMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::RealignB => "realign-b",
            Self::RealignA => "realign-a",
            Self::AlternatingAb => "alternating-ab",
        }
    }

    const fn provenance_method(self) -> &'static str {
        match self {
            Self::RealignB => "activation-ridge-realign-b-v1",
            Self::RealignA => "activation-ridge-realign-a-v1",
            Self::AlternatingAb => "activation-ridge-alternating-ab-v1",
        }
    }
}

/// Split assignment for one captured activation/target row in `realign-factors`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum RealignSplit {
    Fit,
    Development,
}

/// Single captured activation/output target row for factor realignment.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct RealignRow {
    pub(super) sample_id: String,
    pub(super) group_id: String,
    pub(super) token_index: u32,
    pub(super) split: RealignSplit,
    pub(super) input: Vec<f32>,
    pub(super) target_output: Vec<f32>,
}

/// Per-site activation rows or external JSON file reference for `realign-factors`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct RealignSiteInput {
    pub(super) target: String,
    pub(super) rows: Vec<RealignRow>,
    pub(super) rows_path: Option<PathBuf>,
}

/// Configuration payload for `realign-factors`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct RealignFactorsConfig {
    pub(super) mode: RealignMode,
    pub(super) ridge: Option<f64>,
    pub(super) alternating_steps: Option<usize>,
    pub(super) allow_development_regression: bool,
    pub(super) export_dtype: Option<ExportDtype>,
    pub(super) time_budget_milliseconds: Option<u64>,
    pub(super) diagnostic_prompt_sha256s: Vec<String>,
    pub(super) sites: Vec<RealignSiteInput>,
}

struct WorkBudget {
    work: u64,
    deadline: Instant,
}

impl WorkBudget {
    fn new(budget_ms: u64) -> crate::Result<Self> {
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(budget_ms))
            .ok_or("Realign time budget overflow.")?;
        Ok(Self { work: 0, deadline })
    }

    fn charge(&mut self, units: usize) -> crate::Result<()> {
        crate::startup_cancel::checkpoint()?;
        let add = u64::try_from(units).map_err(|e| e.to_string())?;
        self.work = self
            .work
            .checked_add(add)
            .ok_or("Realign work counter overflow.")?;
        if self.work > MAX_REALIGN_WORK || Instant::now() >= self.deadline {
            return Err("Realign time/work budget exhausted; no adapter published.".into());
        }
        Ok(())
    }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 1024 && !value.chars().any(char::is_control)
}

fn validate_realign_rows(
    rows: &[RealignRow],
    target: &str,
    expected_dims: Option<(usize, usize)>,
) -> crate::Result<()> {
    if !(MIN_REALIGN_ROWS..=MAX_REALIGN_ROWS).contains(&rows.len()) {
        return Err(format!(
            "Site '{target}' row count {} must be in {MIN_REALIGN_ROWS}..={MAX_REALIGN_ROWS}.",
            rows.len()
        ));
    }
    let mut groups = BTreeMap::new();
    let mut samples = BTreeMap::new();
    let mut inputs = BTreeMap::new();
    let mut tokens = BTreeSet::new();
    let mut fit_count = 0_usize;
    let mut dev_count = 0_usize;
    let first_in = rows[0].input.len();
    let first_out = rows[0].target_output.len();
    if first_in == 0 || first_out == 0 {
        return Err(format!(
            "Site '{target}' rows must have positive input and targetOutput lengths."
        ));
    }

    for row in rows {
        if !valid_identifier(&row.sample_id) || !valid_identifier(&row.group_id) {
            return Err(format!(
                "Site '{target}' row has empty, oversized, or control-char sampleId/groupId."
            ));
        }
        if let Some((in_f, out_f)) = expected_dims {
            if row.input.len() != in_f || row.target_output.len() != out_f {
                return Err(format!(
                    "Site '{target}' row vector lengths ({}, {}) mismatch target ({in_f}, {out_f}).",
                    row.input.len(),
                    row.target_output.len()
                ));
            }
        } else if row.input.len() != first_in || row.target_output.len() != first_out {
            return Err(format!(
                "Site '{target}' rows have inconsistent vector dimensions."
            ));
        }
        if row
            .input
            .iter()
            .chain(&row.target_output)
            .any(|v| !v.is_finite())
        {
            return Err(format!(
                "Site '{target}' row contains nonfinite input or targetOutput values."
            ));
        }
        if !tokens.insert((row.sample_id.as_str(), row.token_index)) {
            return Err(format!(
                "Site '{target}' contains duplicate (sampleId, tokenIndex) pair ({}, {}).",
                row.sample_id, row.token_index
            ));
        }
        for (map, key) in [
            (&mut groups, row.group_id.as_str()),
            (&mut samples, row.sample_id.as_str()),
        ] {
            if map
                .insert(key, row.split)
                .is_some_and(|split| split != row.split)
            {
                return Err(format!(
                    "Site '{target}' sampleId or groupId crosses fit/development splits."
                ));
            }
        }
        let input_bits: Vec<u32> = row.input.iter().map(|v| v.to_bits()).collect();
        if inputs
            .insert(input_bits, row.split)
            .is_some_and(|split| split != row.split)
        {
            return Err(format!(
                "Site '{target}' has identical F32 input vector crossing fit/development splits."
            ));
        }
        match row.split {
            RealignSplit::Fit => fit_count += 1,
            RealignSplit::Development => dev_count += 1,
        }
    }
    if fit_count < 2 || dev_count == 0 {
        return Err(format!(
            "Site '{target}' requires at least 2 fit rows and 1 development row (got fit={fit_count}, development={dev_count})."
        ));
    }
    Ok(())
}

pub(super) fn validate_realign_config(config: &RealignFactorsConfig) -> crate::Result<()> {
    let ridge = config.ridge.unwrap_or(DEFAULT_REALIGN_RIDGE);
    if !ridge.is_finite() || !(1e-8..=1e6).contains(&ridge) {
        return Err(format!(
            "realignConfig.ridge must be finite and in [1e-8, 1e6], got {ridge}."
        ));
    }
    match config.mode {
        RealignMode::AlternatingAb => {
            let steps = config
                .alternating_steps
                .unwrap_or(DEFAULT_ALTERNATING_STEPS);
            if !(1..=6).contains(&steps) {
                return Err(format!(
                    "realignConfig.alternatingSteps must be in 1..=6, got {steps}."
                ));
            }
        }
        RealignMode::RealignB | RealignMode::RealignA => {
            if config.alternating_steps.is_some() {
                return Err(
                    "realignConfig.alternatingSteps is only supported for alternating-ab mode."
                        .into(),
                );
            }
        }
    }
    let budget_ms = config
        .time_budget_milliseconds
        .unwrap_or(DEFAULT_REALIGN_TIME_BUDGET_MS);
    if !(1..=MAX_REALIGN_TIME_BUDGET_MS).contains(&budget_ms) {
        return Err(format!(
            "realignConfig.timeBudgetMilliseconds must be in 1..={MAX_REALIGN_TIME_BUDGET_MS}, got {budget_ms}."
        ));
    }
    let _ = validate_diagnostic_prompt_digests(&config.diagnostic_prompt_sha256s, "realignConfig")?;
    if !(1..=MAX_REALIGN_SITES).contains(&config.sites.len()) {
        return Err(format!(
            "realignConfig.sites must contain 1..={MAX_REALIGN_SITES} entries."
        ));
    }
    let mut seen_targets = BTreeSet::new();
    for site in &config.sites {
        if site.target.trim().is_empty() {
            return Err("realignConfig.sites contains an empty target name.".into());
        }
        if !seen_targets.insert(site.target.as_str()) {
            return Err(format!(
                "Duplicate target '{}' in realignConfig.sites.",
                site.target
            ));
        }
        let has_inline = !site.rows.is_empty();
        let has_path = site.rows_path.is_some();
        if has_inline == has_path {
            return Err(format!(
                "Site '{}' must supply exactly one of non-empty rows or rowsPath.",
                site.target
            ));
        }
        if let Some(path) = &site.rows_path {
            bonsai_artifacts::validate_external_path(path).map_err(|e| e.to_string())?;
        } else {
            validate_realign_rows(&site.rows, &site.target, None)?;
        }
    }
    Ok(())
}

fn cholesky_f64(mat: &mut [f64], n: usize) -> crate::Result<()> {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = mat[i * n + j];
            for k in 0..j {
                sum -= mat[i * n + k] * mat[j * n + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(
                        "Realignment normal-equation system is not numerically positive definite."
                            .into(),
                    );
                }
                mat[i * n + j] = sum.sqrt();
            } else {
                mat[i * n + j] = sum / mat[j * n + j];
            }
        }
    }
    Ok(())
}

fn solve_cholesky_in_place_f64(chol: &[f64], rhs: &mut [f64], n: usize) {
    for i in 0..n {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= chol[i * n + k] * rhs[k];
        }
        rhs[i] = sum / chol[i * n + i];
    }
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for k in (i + 1)..n {
            sum -= chol[k * n + i] * rhs[k];
        }
        rhs[i] = sum / chol[i * n + i];
    }
}

fn compute_hidden_activations(
    a_rows: &[Vec<f64>],
    fit_rows: &[&RealignRow],
    budget: &mut WorkBudget,
) -> crate::Result<Vec<Vec<f64>>> {
    let rank = a_rows.len();
    let n_fit = fit_rows.len();
    let in_features = a_rows[0].len();
    let mut h_mat = vec![vec![0.0_f64; n_fit]; rank];
    for (j, row) in fit_rows.iter().enumerate() {
        budget.charge(rank * in_features)?;
        for (k, a_k) in a_rows.iter().enumerate() {
            let val: f64 = a_k
                .iter()
                .zip(&row.input)
                .map(|(&akc, &xc)| akc * f64::from(xc))
                .sum();
            if !val.is_finite() {
                return Err("Nonfinite hidden activation in realignment.".into());
            }
            h_mat[k][j] = val;
        }
    }
    Ok(h_mat)
}

fn realign_b_step(
    a_rows: &[Vec<f64>],
    b_cols: &mut [Vec<f64>],
    fit_rows: &[&RealignRow],
    ridge: f64,
    budget: &mut WorkBudget,
) -> crate::Result<()> {
    let rank = a_rows.len();
    let out_features = b_cols[0].len();
    let n_fit = fit_rows.len();
    let h_mat = compute_hidden_activations(a_rows, fit_rows, budget)?;

    let mut gram = vec![0.0_f64; rank * rank];
    for i in 0..rank {
        for j in i..rank {
            let dot = dot_f64(&h_mat[i], &h_mat[j]);
            gram[i * rank + j] = dot;
            gram[j * rank + i] = dot;
        }
        gram[i * rank + i] += ridge;
    }
    budget.charge(rank * rank * n_fit)?;
    cholesky_f64(&mut gram, rank)?;

    let mut residual_o = vec![0.0_f64; n_fit];
    let mut rhs = vec![0.0_f64; rank];
    for o in 0..out_features {
        if o.is_multiple_of(256) {
            let remaining = (out_features - o).min(256);
            budget.charge(remaining * (rank * n_fit * 2 + rank * rank))?;
        }
        for (j, row) in fit_rows.iter().enumerate() {
            let mut pred_o = 0.0_f64;
            for k in 0..rank {
                pred_o += b_cols[k][o] * h_mat[k][j];
            }
            residual_o[j] = f64::from(row.target_output[o]) - pred_o;
        }
        for k in 0..rank {
            rhs[k] = dot_f64(&h_mat[k], &residual_o);
        }
        solve_cholesky_in_place_f64(&gram, &mut rhs, rank);
        for k in 0..rank {
            let updated = b_cols[k][o] + rhs[k];
            if !updated.is_finite() {
                return Err("Nonfinite updated B factor during RealignB.".into());
            }
            b_cols[k][o] = updated;
        }
    }
    Ok(())
}

struct InputGramSpectrum {
    eig_x: Vec<f64>,
    q_x: Vec<Vec<f64>>,
    x_tol: f64,
}

fn precompute_input_gram_spectrum(
    fit_rows: &[&RealignRow],
    in_features: usize,
    budget: &mut WorkBudget,
) -> crate::Result<InputGramSpectrum> {
    let n_fit = fit_rows.len();
    let mut g_x = vec![0.0_f64; n_fit * n_fit];
    for i in 0..n_fit {
        for j in i..n_fit {
            budget.charge(in_features)?;
            let dot: f64 = fit_rows[i]
                .input
                .iter()
                .zip(&fit_rows[j].input)
                .map(|(&xi, &xj)| f64::from(xi) * f64::from(xj))
                .sum();
            g_x[i * n_fit + j] = dot;
            g_x[j * n_fit + i] = dot;
        }
    }
    budget.charge(n_fit * n_fit * n_fit)?;
    let (eig_x, q_x) = super::svd::jacobi_eigendecomp_f64(&mut g_x, n_fit)?;
    let x_tol = eig_x.first().copied().unwrap_or(0.0) * 1e-24;
    Ok(InputGramSpectrum { eig_x, q_x, x_tol })
}

fn realign_a_step(
    a_rows: &mut [Vec<f64>],
    b_cols: &[Vec<f64>],
    fit_rows: &[&RealignRow],
    input_spec: &InputGramSpectrum,
    ridge: f64,
    budget: &mut WorkBudget,
) -> crate::Result<()> {
    let rank = a_rows.len();
    let in_features = a_rows[0].len();
    let out_features = b_cols[0].len();
    let n_fit = fit_rows.len();

    let mut g_b = vec![0.0_f64; rank * rank];
    for i in 0..rank {
        for j in i..rank {
            let dot = dot_f64(&b_cols[i], &b_cols[j]);
            g_b[i * rank + j] = dot;
            g_b[j * rank + i] = dot;
        }
    }
    budget.charge(rank * rank * out_features)?;
    let (eig_b, q_b) = super::svd::jacobi_eigendecomp_f64(&mut g_b, rank)?;

    let h_mat = compute_hidden_activations(a_rows, fit_rows, budget)?;
    let mut c_mat = vec![vec![0.0_f64; n_fit]; rank];
    let mut residual = vec![0.0_f64; out_features];
    for (j, row) in fit_rows.iter().enumerate() {
        budget.charge(rank * out_features * 2)?;
        for o in 0..out_features {
            let mut pred_o = 0.0_f64;
            for k in 0..rank {
                pred_o += b_cols[k][o] * h_mat[k][j];
            }
            residual[o] = f64::from(row.target_output[o]) - pred_o;
        }
        for k in 0..rank {
            c_mat[k][j] = dot_f64(&b_cols[k], &residual);
        }
    }

    // T[a][j] = (Q_B^T * C)[a, j]
    let mut t_mat = vec![vec![0.0_f64; n_fit]; rank];
    for a in 0..rank {
        for j in 0..n_fit {
            let mut acc = 0.0_f64;
            for k in 0..rank {
                acc += q_b[a][k] * c_mat[k][j];
            }
            t_mat[a][j] = acc;
        }
    }

    // Z[a][b] = (T * Q_X)[a, b] / (eig_b[a] * eig_x[b] + ridge)
    let mut z_mat = vec![vec![0.0_f64; n_fit]; rank];
    for a in 0..rank {
        for b in 0..n_fit {
            if input_spec.eig_x[b] <= input_spec.x_tol {
                continue;
            }
            let s_ab = dot_f64(&t_mat[a], &input_spec.q_x[b]);
            let denom = eig_b[a] * input_spec.eig_x[b] + ridge;
            z_mat[a][b] = s_ab / denom;
        }
    }

    // U[a][j] = (Z * Q_X^T)[a, j], then W_X[k][j] = (Q_B * U)[k, j]
    let mut u_mat = vec![vec![0.0_f64; n_fit]; rank];
    for a in 0..rank {
        for b in 0..n_fit {
            let zab = z_mat[a][b];
            if zab == 0.0 {
                continue;
            }
            for j in 0..n_fit {
                u_mat[a][j] += zab * input_spec.q_x[b][j];
            }
        }
    }
    let mut w_x = vec![vec![0.0_f64; n_fit]; rank];
    for k in 0..rank {
        for j in 0..n_fit {
            let mut acc = 0.0_f64;
            for a in 0..rank {
                acc += q_b[a][k] * u_mat[a][j];
            }
            w_x[k][j] = acc;
        }
    }
    budget.charge(rank * n_fit * (rank + n_fit) * 2)?;

    // A_new = A_old + W_X * X_fit^T
    for (j, row) in fit_rows.iter().enumerate() {
        budget.charge(rank * in_features)?;
        for k in 0..rank {
            let w = w_x[k][j];
            if w == 0.0 {
                continue;
            }
            for (dst, &xc) in a_rows[k].iter_mut().zip(&row.input) {
                let updated = *dst + w * f64::from(xc);
                if !updated.is_finite() {
                    return Err("Nonfinite updated A factor during RealignA.".into());
                }
                *dst = updated;
            }
        }
    }
    Ok(())
}

struct SplitStats {
    rows: u32,
    mean_squared_error: f64,
    zero_target_mean_squared_error: f64,
    relative_squared_error: Option<f64>,
    worst_row_relative_l2_error: Option<f64>,
    mean_cosine_similarity: Option<f64>,
    min_cosine_similarity: Option<f64>,
}

impl SplitStats {
    fn to_json(&self) -> Value {
        json!({
            "rows": self.rows,
            "meanSquaredError": self.mean_squared_error,
            "zeroTargetMeanSquaredError": self.zero_target_mean_squared_error,
            "relativeSquaredError": self.relative_squared_error,
            "worstRowRelativeL2Error": self.worst_row_relative_l2_error,
            "meanCosineSimilarity": self.mean_cosine_similarity,
            "minCosineSimilarity": self.min_cosine_similarity,
        })
    }
}

struct RealignSiteMetrics {
    fit: SplitStats,
    development: SplitStats,
}

impl RealignSiteMetrics {
    fn to_json(&self) -> Value {
        json!({
            "fit": self.fit.to_json(),
            "development": self.development.to_json(),
        })
    }
}

fn evaluate_realign_site_metrics(
    rows: &[RealignRow],
    a_rows: &[Vec<f64>],
    b_cols: &[Vec<f64>],
    budget: &mut WorkBudget,
) -> crate::Result<RealignSiteMetrics> {
    let rank = a_rows.len();
    let in_features = a_rows[0].len();
    let out_features = b_cols[0].len();

    let mut fit_count = 0_u32;
    let mut fit_sq_err = 0.0_f64;
    let mut fit_energy = 0.0_f64;
    let mut fit_acc = RowMetricsAccumulator::new();

    let mut dev_count = 0_u32;
    let mut dev_sq_err = 0.0_f64;
    let mut dev_energy = 0.0_f64;
    let mut dev_acc = RowMetricsAccumulator::new();

    let mut hidden = vec![0.0_f64; rank];
    let mut pred = vec![0.0_f64; out_features];
    let mut target = vec![0.0_f64; out_features];

    for row in rows {
        budget.charge(rank * (in_features + out_features) + out_features * 2)?;
        for (k, a_k) in a_rows.iter().enumerate() {
            hidden[k] = a_k
                .iter()
                .zip(&row.input)
                .map(|(&akc, &xc)| akc * f64::from(xc))
                .sum();
        }
        pred.fill(0.0);
        for (k, b_k) in b_cols.iter().enumerate() {
            let hk = hidden[k];
            if hk == 0.0 {
                continue;
            }
            for (dst, &bko) in pred.iter_mut().zip(b_k) {
                *dst += bko * hk;
            }
        }
        let mut sq_err = 0.0_f64;
        let mut energy = 0.0_f64;
        for (o, &tgt_f32) in row.target_output.iter().enumerate() {
            let tgt = f64::from(tgt_f32);
            target[o] = tgt;
            let diff = pred[o] - tgt;
            sq_err += diff * diff;
            energy += tgt * tgt;
        }
        match row.split {
            RealignSplit::Fit => {
                fit_count += 1;
                fit_sq_err += sq_err;
                fit_energy += energy;
                fit_acc.record_pair(&target, &pred)?;
            }
            RealignSplit::Development => {
                dev_count += 1;
                dev_sq_err += sq_err;
                dev_energy += energy;
                dev_acc.record_pair(&target, &pred)?;
            }
        }
    }

    let out_f64 = f64::from(u32::try_from(out_features).map_err(|e| e.to_string())?);
    let finalize = |count: u32,
                    sq_err: f64,
                    energy: f64,
                    acc: RowMetricsAccumulator|
     -> crate::Result<SplitStats> {
        if !sq_err.is_finite() || !energy.is_finite() {
            return Err("Nonfinite realignment error or energy diagnostic.".into());
        }
        let cells = f64::from(count) * out_f64;
        let summary = acc.finish();
        Ok(SplitStats {
            rows: count,
            mean_squared_error: sq_err / cells,
            zero_target_mean_squared_error: energy / cells,
            relative_squared_error: (energy > 0.0).then_some(sq_err / energy),
            worst_row_relative_l2_error: summary.worst_row_relative_l2_error,
            mean_cosine_similarity: summary.mean_cosine_similarity,
            min_cosine_similarity: summary.min_cosine_similarity,
        })
    };

    Ok(RealignSiteMetrics {
        fit: finalize(fit_count, fit_sq_err, fit_energy, fit_acc)?,
        development: finalize(dev_count, dev_sq_err, dev_energy, dev_acc)?,
    })
}

fn round_trip_factor_value(val: f64, dtype: ExportDtype, label: &str) -> crate::Result<(f32, f64)> {
    let v_f32 = check_factor_value(val, dtype, label)?;
    let effective_f64 = match dtype {
        ExportDtype::F32 => f64::from(v_f32),
        ExportDtype::F16 => {
            let bytes = encode_f16_le(v_f32)?;
            f64::from(decode_scalar_f32(FloatDtype::F16, &bytes)?)
        }
    };
    Ok((v_f32, effective_f64))
}

fn quantize_site_factors(
    a_rows: &[Vec<f64>],
    b_cols: &[Vec<f64>],
    alpha: f32,
    out_dtype: ExportDtype,
) -> crate::Result<(Vec<f32>, Vec<f32>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let rank = a_rows.len();
    let in_features = a_rows[0].len();
    let out_features = b_cols[0].len();
    let s1 = (f64::from(alpha) / usize_to_f64(rank)).sqrt();
    let s2 = (usize_to_f64(rank) / f64::from(alpha)).sqrt();

    let mut lora_a_f32 = Vec::new();
    lora_a_f32
        .try_reserve_exact(
            rank.checked_mul(in_features)
                .ok_or("Realigned lora_a size overflow.")?,
        )
        .map_err(|e| format!("Cannot allocate realigned lora_a: {e}"))?;
    let mut eval_a = vec![vec![0.0_f64; in_features]; rank];
    for (k, a_k) in a_rows.iter().enumerate() {
        for (c, &v) in a_k.iter().enumerate() {
            let (v_f32, v_eff) = round_trip_factor_value(v * s2, out_dtype, "realigned lora_a")?;
            lora_a_f32.push(v_f32);
            eval_a[k][c] = v_eff * s1;
        }
    }

    let mut lora_b_f32 = Vec::new();
    lora_b_f32
        .try_reserve_exact(
            out_features
                .checked_mul(rank)
                .ok_or("Realigned lora_b size overflow.")?,
        )
        .map_err(|e| format!("Cannot allocate realigned lora_b: {e}"))?;
    lora_b_f32.resize(out_features * rank, 0.0_f32);
    let mut eval_b = vec![vec![0.0_f64; out_features]; rank];
    for (k, b_k) in b_cols.iter().enumerate() {
        for (o, &v) in b_k.iter().enumerate() {
            let (v_f32, v_eff) = round_trip_factor_value(v * s2, out_dtype, "realigned lora_b")?;
            lora_b_f32[o * rank + k] = v_f32;
            eval_b[k][o] = v_eff * s1;
        }
    }
    Ok((lora_a_f32, lora_b_f32, eval_a, eval_b))
}

/// Runs `realign-factors` against the built-in Swift base tensor dimensions.
pub(super) fn run_realign_factors(request: &Request) -> crate::Result<Value> {
    run_realign_factors_with_dims(request, &swift_base_dims_map())
}

/// Runs `realign-factors` with explicit base tensor dimensions (production and test entry point).
#[expect(
    clippy::too_many_lines,
    reason = "Coordinates bounded row intake, closed-form ridge solves, development gate checking, and atomic verification."
)]
pub(super) fn run_realign_factors_with_dims(
    request: &Request,
    base_dims: &BTreeMap<String, (usize, usize, &'static str)>,
) -> crate::Result<Value> {
    if request.operation != Operation::RealignFactors {
        return Err("Invalid operation dispatch for realign-factors.".into());
    }
    let adapter_path = request
        .adapter_path
        .as_ref()
        .ok_or("realign-factors requires adapterPath.")?;
    let output_path = request
        .output_path
        .as_ref()
        .ok_or("realign-factors requires outputPath.")?;
    let config = request
        .realign_config
        .as_ref()
        .ok_or("realign-factors requires realignConfig.")?;
    validate_realign_config(config)?;
    let training_prompt_sha256s =
        validate_diagnostic_prompt_digests(&config.diagnostic_prompt_sha256s, "realignConfig")?;

    let ridge = config.ridge.unwrap_or(DEFAULT_REALIGN_RIDGE);
    let alternating_steps = match config.mode {
        RealignMode::AlternatingAb => Some(
            config
                .alternating_steps
                .unwrap_or(DEFAULT_ALTERNATING_STEPS),
        ),
        RealignMode::RealignB | RealignMode::RealignA => None,
    };
    let budget_ms = config
        .time_budget_milliseconds
        .unwrap_or(DEFAULT_REALIGN_TIME_BUDGET_MS);
    let mut budget = WorkBudget::new(budget_ms)?;

    let loaded = read_lora_gguf_pairs_with_dims(adapter_path, base_dims)?;
    let out_dtype = config.export_dtype.unwrap_or(loaded.dtype);
    let alpha = loaded.alpha;
    let mut pairs_by_target: BTreeMap<String, ExportTensorPair> = loaded
        .pairs
        .into_iter()
        .map(|p| (p.gguf_base_name.clone(), p))
        .collect();

    let mut per_site_diagnostics = serde_json::Map::new();
    let mut realigned_sites: Vec<String> = Vec::with_capacity(config.sites.len());
    let mut regressed_sites: Vec<String> = Vec::new();
    let mut updated_pairs: Vec<ExportTensorPair> = Vec::new();

    for site in &config.sites {
        let pair = pairs_by_target.remove(&site.target).ok_or_else(|| {
            format!(
                "realignConfig.sites target '{}' is missing from source adapter.",
                site.target
            )
        })?;
        let loaded_rows: Vec<RealignRow>;
        let rows: &[RealignRow] = if let Some(path) = &site.rows_path {
            bonsai_artifacts::validate_external_path(path).map_err(|e| e.to_string())?;
            loaded_rows = crate::bounded_input::read_regular_json(
                path,
                MAX_REALIGN_ROWS_FILE_BYTES,
                "Realign site rows JSON exceeds 48 MiB.",
            )?;
            &loaded_rows
        } else {
            &site.rows
        };
        validate_realign_rows(
            rows,
            &site.target,
            Some((pair.in_features, pair.out_features)),
        )?;

        let s1 = (f64::from(alpha) / usize_to_f64(pair.rank)).sqrt();
        let mut a_rows: Vec<Vec<f64>> = (0..pair.rank)
            .map(|k| {
                let off = k * pair.in_features;
                pair.lora_a_f32[off..off + pair.in_features]
                    .iter()
                    .map(|&v| f64::from(v) * s1)
                    .collect()
            })
            .collect();
        let mut b_cols: Vec<Vec<f64>> = (0..pair.rank)
            .map(|k| {
                (0..pair.out_features)
                    .map(|o| f64::from(pair.lora_b_f32[o * pair.rank + k]) * s1)
                    .collect()
            })
            .collect();

        let before = evaluate_realign_site_metrics(rows, &a_rows, &b_cols, &mut budget)?;
        let fit_rows: Vec<&RealignRow> = rows
            .iter()
            .filter(|r| r.split == RealignSplit::Fit)
            .collect();
        let dev_count = rows.len() - fit_rows.len();

        let input_spec = match config.mode {
            RealignMode::RealignB => None,
            RealignMode::RealignA | RealignMode::AlternatingAb => Some(
                precompute_input_gram_spectrum(&fit_rows, pair.in_features, &mut budget)?,
            ),
        };

        match (config.mode, &input_spec) {
            (RealignMode::RealignB, _) => {
                realign_b_step(&a_rows, &mut b_cols, &fit_rows, ridge, &mut budget)?;
            }
            (RealignMode::RealignA, Some(spec)) => {
                realign_a_step(&mut a_rows, &b_cols, &fit_rows, spec, ridge, &mut budget)?;
            }
            (RealignMode::AlternatingAb, Some(spec)) => {
                let steps = alternating_steps.unwrap_or(DEFAULT_ALTERNATING_STEPS);
                for _ in 0..steps {
                    realign_b_step(&a_rows, &mut b_cols, &fit_rows, ridge, &mut budget)?;
                    realign_a_step(&mut a_rows, &b_cols, &fit_rows, spec, ridge, &mut budget)?;
                }
            }
            (RealignMode::RealignA | RealignMode::AlternatingAb, None) => {
                return Err("Internal invariant error: missing input Gram spectrum.".into());
            }
        }

        let (lora_a_f32, lora_b_f32, eval_a, eval_b) =
            quantize_site_factors(&a_rows, &b_cols, alpha, out_dtype)?;
        let after = evaluate_realign_site_metrics(rows, &eval_a, &eval_b, &mut budget)?;

        let regressed = after.development.mean_squared_error
            > before.development.mean_squared_error * (1.0 + 1e-12);
        if regressed {
            regressed_sites.push(site.target.clone());
        }
        let accepted = config.allow_development_regression || !regressed;
        realigned_sites.push(site.target.clone());
        per_site_diagnostics.insert(
            site.target.clone(),
            json!({
                "target": site.target,
                "rank": pair.rank,
                "fitRows": fit_rows.len(),
                "developmentRows": dev_count,
                "regressed": regressed,
                "accepted": accepted,
                "before": before.to_json(),
                "after": after.to_json(),
            }),
        );

        updated_pairs.push(ExportTensorPair {
            gguf_base_name: pair.gguf_base_name,
            in_features: pair.in_features,
            out_features: pair.out_features,
            rank: pair.rank,
            lora_a_f32,
            lora_b_f32,
        });
    }

    if !config.allow_development_regression && !regressed_sites.is_empty() {
        return Ok(json!({
            "schemaVersion": 1,
            "operation": "realign-factors",
            "published": false,
            "status": "development-regression-rejected",
            "mode": config.mode.as_str(),
            "ridge": ridge,
            "alternatingSteps": alternating_steps,
            "allowDevelopmentRegression": config.allow_development_regression,
            "sourceAdapterPath": adapter_path,
            "sourceAdapterSha256": loaded.sha256,
            "exportDtype": out_dtype,
            "alpha": alpha,
            "trainingPromptSha256s": training_prompt_sha256s,
            "realignedSites": realigned_sites,
            "regressedSites": regressed_sites,
            "perSiteDiagnostics": per_site_diagnostics,
            "workUnits": budget.work,
        }));
    }

    // Re-encode any untouched pairs into the chosen output dtype cleanly.
    let mut final_pairs = Vec::with_capacity(updated_pairs.len() + pairs_by_target.len());
    final_pairs.extend(updated_pairs);
    for (_, pair) in pairs_by_target {
        if out_dtype == loaded.dtype {
            final_pairs.push(pair);
        } else {
            let mut a_f32 = Vec::with_capacity(pair.lora_a_f32.len());
            for v in pair.lora_a_f32 {
                a_f32.push(check_factor_value(
                    f64::from(v),
                    out_dtype,
                    "untouched lora_a",
                )?);
            }
            let mut b_f32 = Vec::with_capacity(pair.lora_b_f32.len());
            for v in pair.lora_b_f32 {
                b_f32.push(check_factor_value(
                    f64::from(v),
                    out_dtype,
                    "untouched lora_b",
                )?);
            }
            final_pairs.push(ExportTensorPair {
                gguf_base_name: pair.gguf_base_name,
                in_features: pair.in_features,
                out_features: pair.out_features,
                rank: pair.rank,
                lora_a_f32: a_f32,
                lora_b_f32: b_f32,
            });
        }
    }
    sort_pairs_canonically(&mut final_pairs);

    let max_rank = final_pairs.iter().map(|p| p.rank).max().unwrap_or(1);
    let mut families_used = BTreeSet::new();
    for pair in &final_pairs {
        let (_, _, family) = validate_target_linear_dims(&pair.gguf_base_name, base_dims)?;
        families_used.insert(family);
    }
    let (expected_env, _, _) = classify_families(&families_used);

    let mut meta: Vec<(String, String)> = loaded
        .provenance_meta
        .into_iter()
        .filter(|(k, _)| {
            k != "bonsai.composition.realign_parent_sha256"
                && k != "bonsai.composition.realign_method"
        })
        .collect();
    meta.push((
        "bonsai.composition.realign_parent_sha256".to_owned(),
        loaded.sha256.clone(),
    ));
    meta.push((
        "bonsai.composition.realign_method".to_owned(),
        config.mode.provenance_method().to_owned(),
    ));
    let meta_refs: Vec<(&str, &str)> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

    let written = write_lora_gguf(output_path, out_dtype, alpha, &meta_refs, &final_pairs)?;
    let verification = verify_published_adapter(
        output_path,
        &written.sha256,
        written.bytes,
        max_rank,
        final_pairs.len(),
        base_dims,
    )?;

    Ok(json!({
        "schemaVersion": 1,
        "operation": "realign-factors",
        "published": true,
        "status": "published-and-verified",
        "mode": config.mode.as_str(),
        "ridge": ridge,
        "alternatingSteps": alternating_steps,
        "allowDevelopmentRegression": config.allow_development_regression,
        "sourceAdapterPath": adapter_path,
        "sourceAdapterSha256": loaded.sha256,
        "outputPath": output_path,
        "targetEnvelope": expected_env,
        "exportDtype": out_dtype,
        "alpha": alpha,
        "minRank": verification["minRank"],
        "maxRank": verification["maxRank"],
        "bytes": written.bytes,
        "sha256": written.sha256,
        "tensorCount": written.tensor_count,
        "pairCount": final_pairs.len(),
        "trainingPromptSha256s": training_prompt_sha256s,
        "realignedSites": realigned_sites,
        "regressedSites": regressed_sites,
        "perSiteDiagnostics": per_site_diagnostics,
        "workUnits": budget.work,
        "verification": verification,
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use serde_json::{Value, json};

    use super::super::gguf_adapter::{
        ExportDtype, ExportTensorPair, read_lora_gguf_pairs_with_dims, write_lora_gguf,
    };
    use super::super::{Request, run, validate_request};
    use super::{
        RealignRow, RealignSplit, WorkBudget, f32_le_sha256, precompute_input_gram_spectrum,
        realign_a_step, realign_b_step, run_fold_patch_with_dims, run_realign_factors_with_dims,
        thin_qr_f64,
    };
    use crate::bounded_input::ScratchDir;

    fn toy_dims() -> BTreeMap<String, (usize, usize, &'static str)> {
        [
            ("blk.0.ffn_down.weight".to_owned(), (12, 8, "ffn_down")),
            ("blk.3.attn_k.weight".to_owned(), (10, 6, "attn_k")),
            ("blk.63.ffn_down.weight".to_owned(), (12, 8, "ffn_down")),
        ]
        .into_iter()
        .collect()
    }

    fn reconstruct_delta_f64(pair: &ExportTensorPair, alpha: f32) -> Vec<f64> {
        let scale = f64::from(alpha) / f64::from(u32::try_from(pair.rank).unwrap_or(1));
        let mut out = vec![0.0_f64; pair.out_features * pair.in_features];
        for row in 0..pair.out_features {
            for col in 0..pair.in_features {
                let mut acc = 0.0_f64;
                for k in 0..pair.rank {
                    acc += f64::from(pair.lora_b_f32[row * pair.rank + k])
                        * f64::from(pair.lora_a_f32[k * pair.in_features + col]);
                }
                out[row * pair.in_features + col] = scale * acc;
            }
        }
        out
    }

    fn write_weight_patch_json(
        path: &std::path::Path,
        tensor: &str,
        a: &[f32],
        b: &[f32],
        scale_bits: &str,
    ) -> crate::Result<()> {
        let artifact = json!({
            "schemaVersion": 1,
            "kind": "experimental-ffn-rank-one-patch",
            "enabled": false,
            "qualification": "unqualified-research-only",
            "site": {
                "tensor": tensor,
                "inputBasis": "original-ffn-hidden",
                "inputWidth": a.len(),
                "outputWidth": b.len()
            },
            "compositionOrder": "after-standard-adapter",
            "scaleBits": scale_bits,
            "factors": {
                "a": a,
                "b": b,
                "aSha256": f32_le_sha256(a)?,
                "bSha256": f32_le_sha256(b)?
            },
            "behaviorQualified": false
        });
        fs::write(
            path,
            serde_json::to_vec_pretty(&artifact).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    #[test]
    fn thin_qr_handles_full_and_rank_deficient_columns() -> crate::Result<()> {
        let cols = vec![
            vec![1.0, 2.0, -1.0, 0.5],
            vec![0.0, 1.0, 1.0, -2.0],
            vec![2.0, 4.0, -2.0, 1.0], // linearly dependent: 2 * col0
        ];
        let qr = thin_qr_f64(&cols)?;
        assert_eq!(qr.q_cols.len(), 2);
        // Orthonormality of Q
        for (i, qi) in qr.q_cols.iter().enumerate() {
            for (j, qj) in qr.q_cols.iter().enumerate() {
                let d: f64 = qi.iter().zip(qj).map(|(x, y)| x * y).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((d - expected).abs() < 1e-12);
            }
        }
        // Reconstruction M == Q * R
        for (j, col) in cols.iter().enumerate() {
            for (r, &expected) in col.iter().enumerate() {
                let mut got = 0.0_f64;
                for i in 0..qr.q_cols.len() {
                    got += qr.q_cols[i][r] * qr.r_mat[i * cols.len() + j];
                }
                assert!((got - expected).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn fold_patch_append_and_svd_match_exact_delta_composition() -> crate::Result<()> {
        let scratch = ScratchDir::new("fold-patch-math")?;
        let dims = toy_dims();
        let src = scratch.join("src-r2.gguf");

        let mut a0 = vec![0.0_f32; 2 * 12];
        let mut b0 = vec![0.0_f32; 8 * 2];
        for k in 0..2 {
            a0[k * 12 + k] = 0.5;
            a0[k * 12 + k + 4] = 0.25;
            b0[k * 2 + k] = 0.8;
            b0[(k + 2) * 2 + k] = -0.3;
        }
        let src_pair = ExportTensorPair {
            gguf_base_name: "blk.63.ffn_down.weight".to_owned(),
            in_features: 12,
            out_features: 8,
            rank: 2,
            lora_a_f32: a0,
            lora_b_f32: b0,
        };
        let untouched_pair = ExportTensorPair {
            gguf_base_name: "blk.3.attn_k.weight".to_owned(),
            in_features: 10,
            out_features: 6,
            rank: 2,
            lora_a_f32: vec![0.1_f32; 2 * 10],
            lora_b_f32: vec![0.2_f32; 6 * 2],
        };
        let src_delta = reconstruct_delta_f64(&src_pair, 2.0);
        let exp_3 = reconstruct_delta_f64(&untouched_pair, 2.0);
        write_lora_gguf(
            &src,
            ExportDtype::F32,
            2.0,
            &[("bonsai.composition.method", "fold-test-parent")],
            &[untouched_pair, src_pair],
        )?;

        let patch_path = scratch.join("weight-patch.json");
        let patch_a = vec![0.0, 0.0, 0.7, -0.4, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3];
        let patch_b = vec![0.0, 0.0, 0.0, 0.0, 0.9, -0.5, 0.2, 0.1];
        // scaleBits "3f800000" = 1.0f32, patchScale = -0.5 -> s_eff = -0.5
        write_weight_patch_json(
            &patch_path,
            "blk.63.ffn_down.weight",
            &patch_a,
            &patch_b,
            "3f800000",
        )?;

        let inline_a = vec![0.3, -0.1, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0];
        let inline_b = vec![0.1, 0.2, 0.0, 0.0, 0.0, 0.4, -0.3, 0.0];

        let diag_a = "11".repeat(32);
        let diag_b = "22".repeat(32);

        // Expected combined delta on blk.63: src_delta + (-0.5) * (patch_b * patch_a^T)
        let mut expected_63 = src_delta;
        for r in 0..8 {
            for c in 0..12 {
                expected_63[r * 12 + c] += -0.5 * f64::from(patch_b[r]) * f64::from(patch_a[c]);
            }
        }

        // 1. Append mode: combines patch into blk.63 (rank 2 -> 3) and creates new blk.0 (rank 1).
        let out_append = scratch.join("folded-append.gguf");
        let req_append: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "fold-patch",
            "adapterPath": src,
            "outputPath": out_append,
            "foldPatchConfig": {
                "mode": "append",
                "exportDtype": "f32",
                "diagnosticPromptSha256s": [diag_b, diag_a],
                "patches": [
                    {
                        "kind": "weight-patch-file",
                        "patchPath": patch_path,
                        "patchScale": -0.5
                    },
                    {
                        "kind": "inline-rank-one",
                        "target": "blk.0.ffn_down.weight",
                        "a": inline_a,
                        "b": inline_b,
                        "scale": 1.25
                    }
                ]
            }
        }))
        .map_err(|e| e.to_string())?;
        validate_request(&req_append)?;
        let res_append = run_fold_patch_with_dims(&req_append, &dims)?;
        assert_eq!(res_append["published"], true);
        assert_eq!(res_append["mode"], "append");
        assert_eq!(res_append["trainingPromptSha256s"], json!([diag_a, diag_b]));
        assert_eq!(
            res_append["foldedSites"],
            json!(["blk.0.ffn_down.weight", "blk.63.ffn_down.weight"])
        );
        assert_eq!(
            res_append["perSiteDiagnostics"]["blk.63.ffn_down.weight"]["finalRank"],
            3
        );
        assert_eq!(
            res_append["perSiteDiagnostics"]["blk.0.ffn_down.weight"]["createdSite"],
            true
        );
        assert_eq!(res_append["verification"]["verified"], true);
        assert_eq!(
            res_append["verification"]["provenanceMetadata"]["bonsai.composition.method"],
            "fold-test-parent"
        );
        assert_eq!(
            res_append["verification"]["provenanceMetadata"]["bonsai.composition.fold_method"],
            "activation-fold-append-v1"
        );
        assert!(
            res_append["verification"]["provenanceMetadata"]
                ["bonsai.composition.fold_parent_sha256"]
                .is_string()
        );

        let loaded_append = read_lora_gguf_pairs_with_dims(&out_append, &dims)?;
        assert_eq!(loaded_append.pairs.len(), 3);
        let p0 = loaded_append
            .pairs
            .iter()
            .find(|p| p.gguf_base_name == "blk.0.ffn_down.weight")
            .ok_or("Missing created blk.0")?;
        assert_eq!(p0.rank, 1);
        let got_0 = reconstruct_delta_f64(p0, loaded_append.alpha);
        for r in 0..8 {
            for c in 0..12 {
                let exp = 1.25 * f64::from(inline_b[r]) * f64::from(inline_a[c]);
                assert!((got_0[r * 12 + c] - exp).abs() < 1e-5);
            }
        }
        let p63 = loaded_append
            .pairs
            .iter()
            .find(|p| p.gguf_base_name == "blk.63.ffn_down.weight")
            .ok_or("Missing blk.63")?;
        assert_eq!(p63.rank, 3);
        let got_63 = reconstruct_delta_f64(p63, loaded_append.alpha);
        for (got, exp) in got_63.iter().zip(&expected_63) {
            assert!((got - exp).abs() < 1e-5);
        }

        // Untouched site blk.3 preserved identically.
        let p3 = loaded_append
            .pairs
            .iter()
            .find(|p| p.gguf_base_name == "blk.3.attn_k.weight")
            .ok_or("Missing untouched blk.3")?;
        let got_3 = reconstruct_delta_f64(p3, loaded_append.alpha);
        for (got, exp) in got_3.iter().zip(&exp_3) {
            assert!((got - exp).abs() < 1e-5);
        }

        // 2. FoldSvd mode: compress blk.63 back from rank 3 down to rank 2, check exact Frobenius error & energy fraction.
        let out_svd = scratch.join("folded-svd.gguf");
        let req_svd: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "fold-patch",
            "adapterPath": src,
            "outputPath": out_svd,
            "foldPatchConfig": {
                "mode": "fold-svd",
                "targetMaxRank": 2,
                "exportDtype": "f32",
                "diagnosticPromptSha256s": [diag_a],
                "patches": [
                    {
                        "kind": "weight-patch-file",
                        "patchPath": patch_path,
                        "patchScale": -0.5
                    }
                ]
            }
        }))
        .map_err(|e| e.to_string())?;
        let res_svd = run_fold_patch_with_dims(&req_svd, &dims)?;
        assert_eq!(res_svd["published"], true);
        assert_eq!(res_svd["mode"], "fold-svd");
        assert_eq!(res_svd["maxRank"], 2);
        let diag_63 = &res_svd["perSiteDiagnostics"]["blk.63.ffn_down.weight"];
        assert_eq!(diag_63["combinedRank"], 3);
        assert_eq!(diag_63["finalRank"], 2);
        let reported_err = diag_63["frobeniusReconstructionError"]
            .as_f64()
            .ok_or("Missing frobeniusReconstructionError")?;
        let retained = diag_63["retainedEnergyFraction"]
            .as_f64()
            .ok_or("Missing retainedEnergyFraction")?;
        let total_sq = diag_63["totalEnergyFrobeniusSquared"]
            .as_f64()
            .ok_or("Missing totalEnergyFrobeniusSquared")?;
        assert!(retained > 0.0 && retained < 1.0);
        let loaded_svd = read_lora_gguf_pairs_with_dims(&out_svd, &dims)?;
        let p63_svd = loaded_svd
            .pairs
            .iter()
            .find(|p| p.gguf_base_name == "blk.63.ffn_down.weight")
            .ok_or("Missing folded blk.63")?;
        let got_63_svd = reconstruct_delta_f64(p63_svd, loaded_svd.alpha);
        let actual_err = got_63_svd
            .iter()
            .zip(&expected_63)
            .map(|(g, e)| (g - e) * (g - e))
            .sum::<f64>()
            .sqrt();
        assert!(
            (actual_err - reported_err).abs() < 1e-5,
            "actual_err={actual_err} reported_err={reported_err}"
        );
        let expected_retained = 1.0 - (actual_err * actual_err) / total_sq;
        assert!((retained - expected_retained).abs() < 1e-5);

        // 3. FoldSvd at rank 3 preserves full rank-3 energy (error ~ 0, retained ~ 1.0).
        let out_svd3 = scratch.join("folded-svd-r3.gguf");
        let req_svd3: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "fold-patch",
            "adapterPath": src,
            "outputPath": out_svd3,
            "foldPatchConfig": {
                "mode": "fold-svd",
                "targetMaxRank": 3,
                "exportDtype": "f32",
                "diagnosticPromptSha256s": [diag_a],
                "patches": [
                    {
                        "kind": "weight-patch-file",
                        "patchPath": patch_path,
                        "patchScale": -0.5
                    }
                ]
            }
        }))
        .map_err(|e| e.to_string())?;
        let res_svd3 = run_fold_patch_with_dims(&req_svd3, &dims)?;
        let err3 = res_svd3["perSiteDiagnostics"]["blk.63.ffn_down.weight"]
            ["frobeniusReconstructionError"]
            .as_f64()
            .ok_or("Missing err3")?;
        let ret3 =
            res_svd3["perSiteDiagnostics"]["blk.63.ffn_down.weight"]["retainedEnergyFraction"]
                .as_f64()
                .ok_or("Missing ret3")?;
        assert!(err3 < 1e-10, "err3={err3}");
        assert!((ret3 - 1.0).abs() < 1e-10, "ret3={ret3}");
        Ok(())
    }

    #[test]
    fn fold_patch_creates_standalone_adapter_and_rejects_bad_inputs() -> crate::Result<()> {
        let scratch = ScratchDir::new("fold-patch-validation")?;
        let out_standalone = scratch.join("standalone.gguf");
        let a = vec![0.1_f32; 17_408];
        let b = vec![0.2_f32; 5_120];
        let patch_path = scratch.join("real-patch.json");
        write_weight_patch_json(&patch_path, "blk.63.ffn_down.weight", &a, &b, "3f800000")?;

        let diag = "aa".repeat(32);
        let res = run(json!({
            "schemaVersion": 1,
            "operation": "fold-patch",
            "outputPath": out_standalone,
            "foldPatchConfig": {
                "mode": "append",
                "exportDtype": "f16",
                "diagnosticPromptSha256s": [diag],
                "patches": [
                    { "kind": "weight-patch-file", "patchPath": patch_path }
                ]
            }
        }))?;
        assert_eq!(res["published"], true);
        assert_eq!(res["pairCount"], 1);
        assert_eq!(res["maxRank"], 1);
        assert_eq!(res["verification"]["verified"], true);
        assert_eq!(res["verification"]["detectedEnvelope"], "residual-writers");

        // Tampered aSha256 must fail.
        let bad_sha_patch = scratch.join("bad-sha.json");
        let tampered = json!({
            "schemaVersion": 1,
            "kind": "experimental-ffn-rank-one-patch",
            "site": { "tensor": "blk.0.ffn_down.weight", "inputWidth": 17_408, "outputWidth": 5_120 },
            "scaleBits": "3f800000",
            "factors": {
                "a": a,
                "b": b,
                "aSha256": "00".repeat(32),
                "bSha256": f32_le_sha256(&b)?
            }
        });
        fs::write(
            &bad_sha_patch,
            serde_json::to_vec(&tampered).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "fold-patch",
                "outputPath": scratch.join("should-not-exist-1.gguf"),
                "foldPatchConfig": {
                    "diagnosticPromptSha256s": [diag],
                    "patches": [{ "kind": "weight-patch-file", "patchPath": bad_sha_patch }]
                }
            }))
            .is_err()
        );

        // Missing targetMaxRank in fold-svd must fail.
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "fold-patch",
                "outputPath": scratch.join("should-not-exist-2.gguf"),
                "foldPatchConfig": {
                    "mode": "fold-svd",
                    "diagnosticPromptSha256s": [diag],
                    "patches": [{ "kind": "weight-patch-file", "patchPath": patch_path }]
                }
            }))
            .is_err()
        );

        // Empty diagnosticPromptSha256s must fail.
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "fold-patch",
                "outputPath": scratch.join("should-not-exist-3.gguf"),
                "foldPatchConfig": {
                    "mode": "append",
                    "diagnosticPromptSha256s": [],
                    "patches": [{ "kind": "weight-patch-file", "patchPath": patch_path }]
                }
            }))
            .is_err()
        );

        // Append exceeding targetMaxRank ceiling must fail.
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "fold-patch",
                "outputPath": scratch.join("should-not-exist-4.gguf"),
                "foldPatchConfig": {
                    "mode": "append",
                    "targetMaxRank": 1,
                    "diagnosticPromptSha256s": [diag],
                    "patches": [
                        { "kind": "weight-patch-file", "patchPath": patch_path },
                        { "kind": "inline-rank-one", "target": "blk.63.ffn_down.weight", "a": a, "b": b, "scale": 1.0 }
                    ]
                }
            }))
            .is_err()
        );

        // Unknown field inside FoldPatchEntry must fail (deny_unknown_fields).
        let bad_entry: Result<Value, _> = run(json!({
            "schemaVersion": 1,
            "operation": "fold-patch",
            "outputPath": scratch.join("should-not-exist-5.gguf"),
            "foldPatchConfig": {
                "diagnosticPromptSha256s": [diag],
                "patches": [
                    { "kind": "inline-rank-one", "target": "blk.0.ffn_down.weight", "a": a, "b": b, "scale": 1.0, "extraField": 123 }
                ]
            }
        }));
        assert!(bad_entry.is_err());

        Ok(())
    }

    fn make_realign_row(
        sample_id: &str,
        group_id: &str,
        token_index: u32,
        split: RealignSplit,
        input: Vec<f32>,
        target_output: Vec<f32>,
    ) -> RealignRow {
        RealignRow {
            sample_id: sample_id.to_owned(),
            group_id: group_id.to_owned(),
            token_index,
            split,
            input,
            target_output,
        }
    }

    fn eval_objective_a(
        a: &[Vec<f64>],
        a0: &[Vec<f64>],
        b: &[Vec<f64>],
        fit_rows: &[&RealignRow],
        ridge: f64,
    ) -> f64 {
        let rank = a.len();
        let out_features = b[0].len();
        let mut data_loss = 0.0_f64;
        for row in fit_rows {
            let mut h = vec![0.0_f64; rank];
            for (k, ak) in a.iter().enumerate() {
                h[k] = ak
                    .iter()
                    .zip(&row.input)
                    .map(|(&akc, &xc)| akc * f64::from(xc))
                    .sum();
            }
            for o in 0..out_features {
                let mut pred = 0.0_f64;
                for (k, bk) in b.iter().enumerate() {
                    pred += bk[o] * h[k];
                }
                let diff = pred - f64::from(row.target_output[o]);
                data_loss += diff * diff;
            }
        }
        let mut reg = 0.0_f64;
        for (ak, ak0) in a.iter().zip(a0) {
            for (&v, &v0) in ak.iter().zip(ak0) {
                let d = v - v0;
                reg += d * d;
            }
        }
        data_loss + ridge * reg
    }

    #[test]
    fn realign_a_and_b_match_exact_normal_equations_and_finite_differences() -> crate::Result<()> {
        let rank = 3;
        let in_f = 12;
        let out_f = 8;
        let ridge = 0.05;

        let a0: Vec<Vec<f64>> = (0..rank)
            .map(|k| {
                (0..in_f)
                    .map(|c| {
                        let kf = f64::from(u32::try_from(k + 1).unwrap_or(1));
                        let cf = f64::from(u32::try_from(c + 1).unwrap_or(1));
                        0.3 * (kf * 0.7 + cf * 0.4).sin()
                    })
                    .collect()
            })
            .collect();
        let b0: Vec<Vec<f64>> = (0..rank)
            .map(|k| {
                (0..out_f)
                    .map(|o| {
                        let kf = f64::from(u32::try_from(k + 1).unwrap_or(1));
                        let of = f64::from(u32::try_from(o + 1).unwrap_or(1));
                        0.4 * (kf * 0.5 - of * 0.3).cos()
                    })
                    .collect()
            })
            .collect();

        // Build 5 fit rows where row 4 is an exact linear combination of rows 0 and 1
        // so X_fit is intentionally rank-deficient to exercise the nullspace threshold.
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "Synthetic test vectors narrow exact deterministic values to f32."
        )]
        let mut inputs: Vec<Vec<f32>> = (0..4)
            .map(|j| {
                (0..in_f)
                    .map(|c| {
                        let jf = f64::from(u32::try_from(j + 1).unwrap_or(1));
                        let cf = f64::from(u32::try_from(c + 1).unwrap_or(1));
                        ((jf * 1.1 + cf * 0.37).sin() * 0.8) as f32
                    })
                    .collect()
            })
            .collect();
        let dep_row: Vec<f32> = inputs[0]
            .iter()
            .zip(&inputs[1])
            .map(|(&x0, &x1)| 1.5 * x0 - 0.5 * x1)
            .collect();
        inputs.push(dep_row);

        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "Synthetic test targets narrow deterministic values to f32."
        )]
        let targets: Vec<Vec<f32>> = (0..5)
            .map(|j| {
                (0..out_f)
                    .map(|o| {
                        let jf = f64::from(u32::try_from(j + 1).unwrap_or(1));
                        let of = f64::from(u32::try_from(o + 1).unwrap_or(1));
                        ((jf * 0.9 - of * 0.6).cos() * 1.2) as f32
                    })
                    .collect()
            })
            .collect();

        let owned_rows: Vec<RealignRow> = inputs
            .into_iter()
            .zip(targets)
            .enumerate()
            .map(|(j, (inp, tgt))| {
                make_realign_row(
                    &format!("s{j}"),
                    &format!("g{j}"),
                    0,
                    RealignSplit::Fit,
                    inp,
                    tgt,
                )
            })
            .collect();
        let fit_refs: Vec<&RealignRow> = owned_rows.iter().collect();

        // 1. Verify RealignA against exact normal equation and finite-difference gradient.
        let mut budget = WorkBudget::new(5_000)?;
        let spec = precompute_input_gram_spectrum(&fit_refs, in_f, &mut budget)?;
        let mut a_sol = a0.clone();
        realign_a_step(&mut a_sol, &b0, &fit_refs, &spec, ridge, &mut budget)?;

        // Check normal equation: (B^T B) ΔA (X X^T) + λ ΔA == B^T (Y - B A0 X) X^T
        let mut delta_a = vec![vec![0.0_f64; in_f]; rank];
        for k in 0..rank {
            for c in 0..in_f {
                delta_a[k][c] = a_sol[k][c] - a0[k][c];
            }
        }
        // Compute lhs = (B^T B) ΔA (X X^T) + λ ΔA and rhs = B^T (Y - B A0 X) X^T
        let mut lhs_mid = vec![vec![0.0_f64; fit_refs.len()]; rank];
        let mut rhs_mid = vec![vec![0.0_f64; fit_refs.len()]; rank];
        for (j, row) in fit_refs.iter().enumerate() {
            let mut b_da_x = vec![0.0_f64; out_f];
            let mut r0 = vec![0.0_f64; out_f];
            for o in 0..out_f {
                r0[o] = f64::from(row.target_output[o]);
            }
            for k in 0..rank {
                let da_x: f64 = delta_a[k]
                    .iter()
                    .zip(&row.input)
                    .map(|(&d, &x)| d * f64::from(x))
                    .sum();
                let a0_x: f64 = a0[k]
                    .iter()
                    .zip(&row.input)
                    .map(|(&a, &x)| a * f64::from(x))
                    .sum();
                for o in 0..out_f {
                    b_da_x[o] += b0[k][o] * da_x;
                    r0[o] -= b0[k][o] * a0_x;
                }
            }
            for k in 0..rank {
                lhs_mid[k][j] = b0[k].iter().zip(&b_da_x).map(|(&bv, &y)| bv * y).sum();
                rhs_mid[k][j] = b0[k].iter().zip(&r0).map(|(&bv, &y)| bv * y).sum();
            }
        }
        let mut max_ne_residual = 0.0_f64;
        for k in 0..rank {
            for c in 0..in_f {
                let mut lhs_val = ridge * delta_a[k][c];
                let mut rhs_val = 0.0_f64;
                for (j, row) in fit_refs.iter().enumerate() {
                    let xc = f64::from(row.input[c]);
                    lhs_val += lhs_mid[k][j] * xc;
                    rhs_val += rhs_mid[k][j] * xc;
                }
                max_ne_residual = max_ne_residual.max((lhs_val - rhs_val).abs());
            }
        }
        assert!(
            max_ne_residual < 1e-10,
            "RealignA normal equation residual={max_ne_residual}"
        );

        // Finite-difference optimality check around a_sol
        let h_fd = 1e-6_f64;
        let mut max_grad = 0.0_f64;
        for k in 0..rank {
            for c in 0..in_f {
                let mut plus = a_sol.clone();
                plus[k][c] += h_fd;
                let mut minus = a_sol.clone();
                minus[k][c] -= h_fd;
                let grad = (eval_objective_a(&plus, &a0, &b0, &fit_refs, ridge)
                    - eval_objective_a(&minus, &a0, &b0, &fit_refs, ridge))
                    / (2.0 * h_fd);
                max_grad = max_grad.max(grad.abs());
            }
        }
        assert!(max_grad < 1e-7, "RealignA finite-diff gradient={max_grad}");

        // 2. Verify RealignB against exact normal equation: (H H^T + λ I) ΔB^T = H (Y - B0 H)^T
        let mut b_sol = b0.clone();
        realign_b_step(&a0, &mut b_sol, &fit_refs, ridge, &mut budget)?;
        let h_mat = super::compute_hidden_activations(&a0, &fit_refs, &mut budget)?;
        let mut max_b_ne_residual = 0.0_f64;
        for o in 0..out_f {
            for i in 0..rank {
                let mut lhs_val = ridge * (b_sol[i][o] - b0[i][o]);
                let mut rhs_val = 0.0_f64;
                for (j, row) in fit_refs.iter().enumerate() {
                    let mut bsol_h = 0.0_f64;
                    let mut b0_h = 0.0_f64;
                    for k in 0..rank {
                        bsol_h += b_sol[k][o] * h_mat[k][j];
                        b0_h += b0[k][o] * h_mat[k][j];
                    }
                    let delta_pred = bsol_h - b0_h;
                    let r0 = f64::from(row.target_output[o]) - b0_h;
                    lhs_val += h_mat[i][j] * delta_pred;
                    rhs_val += h_mat[i][j] * r0;
                }
                max_b_ne_residual = max_b_ne_residual.max((lhs_val - rhs_val).abs());
            }
        }
        assert!(
            max_b_ne_residual < 1e-10,
            "RealignB normal equation residual={max_b_ne_residual}"
        );
        Ok(())
    }

    #[test]
    fn realign_factors_modes_rows_path_and_regression_gate() -> crate::Result<()> {
        let scratch = ScratchDir::new("realign-factors")?;
        let dims = toy_dims();
        let src = scratch.join("src-r3.gguf");

        let in_f = 12_usize;
        let out_f = 8_usize;
        let rank = 3_usize;
        let alpha = 3.0_f32;

        // Initial adapter factors (slightly perturbed from ground truth).
        let mut a_init = vec![0.0_f32; rank * in_f];
        let mut b_init = vec![0.0_f32; out_f * rank];
        for k in 0..rank {
            a_init[k * in_f + k] = 0.5;
            a_init[k * in_f + k + 3] = 0.2;
            b_init[k * rank + k] = 0.4;
            b_init[(k + 3) * rank + k] = 0.25;
        }
        let untouched_pair = ExportTensorPair {
            gguf_base_name: "blk.3.attn_k.weight".to_owned(),
            in_features: 10,
            out_features: 6,
            rank: 2,
            lora_a_f32: vec![0.125_f32; 2 * 10],
            lora_b_f32: vec![-0.25_f32; 6 * 2],
        };
        let untouched_expected = untouched_pair.lora_a_f32.clone();
        write_lora_gguf(
            &src,
            ExportDtype::F32,
            alpha,
            &[("bonsai.composition.method", "realign-parent-test")],
            &[
                untouched_pair,
                ExportTensorPair {
                    gguf_base_name: "blk.63.ffn_down.weight".to_owned(),
                    in_features: in_f,
                    out_features: out_f,
                    rank,
                    lora_a_f32: a_init,
                    lora_b_f32: b_init,
                },
            ],
        )?;

        // Ground-truth rank-3 linear target shifted from initial factors.
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "Synthetic test matrix narrows deterministic f64 values to f32."
        )]
        let eval_true_y = |x: &[f32]| -> Vec<f32> {
            let mut y = vec![0.0_f32; out_f];
            for k in 0..rank {
                let mut h = 0.0_f64;
                for (c, &xc) in x.iter().enumerate() {
                    let ac = if c == k {
                        0.9
                    } else if c == k + 3 {
                        -0.4
                    } else if c == k + 6 {
                        0.3
                    } else {
                        0.0
                    };
                    h += ac * f64::from(xc);
                }
                for (o, yo) in y.iter_mut().enumerate() {
                    let bo = if o == k {
                        1.1
                    } else if o == k + 3 {
                        0.6
                    } else {
                        0.05 * f64::from(u32::try_from(o + k + 1).unwrap_or(1))
                    };
                    *yo += (bo * h) as f32;
                }
            }
            y
        };

        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "Deterministic synthetic input vectors narrow f64 cosines/sines to f32."
        )]
        let make_input = |seed: usize| -> Vec<f32> {
            (0..in_f)
                .map(|c| {
                    let sf = f64::from(u32::try_from(seed + 1).unwrap_or(1));
                    let cf = f64::from(u32::try_from(c + 1).unwrap_or(1));
                    ((sf * 0.73 + cf * 1.19).sin() + 0.5 * (sf * 0.41 - cf * 0.29).cos()) as f32
                })
                .collect()
        };

        let mut rows = Vec::new();
        for idx in 0..8_usize {
            let x = make_input(idx);
            let y = eval_true_y(&x);
            rows.push(make_realign_row(
                &format!("fit-{idx}"),
                &format!("grp-fit-{idx}"),
                0,
                RealignSplit::Fit,
                x,
                y,
            ));
        }
        for idx in 8..11_usize {
            let x = make_input(idx);
            let y = eval_true_y(&x);
            rows.push(make_realign_row(
                &format!("dev-{idx}"),
                &format!("grp-dev-{idx}"),
                0,
                RealignSplit::Development,
                x,
                y,
            ));
        }

        let rows_path = scratch.join("site-rows.json");
        fs::write(
            &rows_path,
            serde_json::to_vec(&rows).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        let diag_a = "11".repeat(32);
        let diag_b = "22".repeat(32);

        // Run RealignB, RealignA, and AlternatingAb (via rowsPath) and verify MSE decreases cleanly.
        let mut alt_dev_mse = f64::INFINITY;
        for (mode, steps, tag) in [
            ("realign-b", None, "rb"),
            ("realign-a", None, "ra"),
            ("alternating-ab", Some(4_usize), "alt"),
        ] {
            let out_path = scratch.join(format!("realigned-{tag}.gguf"));
            let mut cfg = json!({
                "mode": mode,
                "ridge": 1e-6,
                "exportDtype": "f32",
                "diagnosticPromptSha256s": [diag_b, diag_a],
                "sites": [{
                    "target": "blk.63.ffn_down.weight",
                    "rowsPath": rows_path
                }]
            });
            if let Some(s) = steps {
                cfg["alternatingSteps"] = json!(s);
            }
            let req: Request = serde_json::from_value(json!({
                "schemaVersion": 1,
                "operation": "realign-factors",
                "adapterPath": src,
                "outputPath": out_path,
                "realignConfig": cfg
            }))
            .map_err(|e| e.to_string())?;
            validate_request(&req)?;
            let res = run_realign_factors_with_dims(&req, &dims)?;
            assert_eq!(res["published"], true);
            assert_eq!(res["status"], "published-and-verified");
            assert_eq!(res["mode"], mode);
            assert_eq!(res["trainingPromptSha256s"], json!([diag_a, diag_b]));
            assert_eq!(res["realignedSites"], json!(["blk.63.ffn_down.weight"]));
            assert_eq!(res["verification"]["verified"], true);
            assert_eq!(
                res["verification"]["provenanceMetadata"]["bonsai.composition.method"],
                "realign-parent-test"
            );
            assert!(
                res["verification"]["provenanceMetadata"]
                    ["bonsai.composition.realign_parent_sha256"]
                    .is_string()
            );

            let diag = &res["perSiteDiagnostics"]["blk.63.ffn_down.weight"];
            let before_fit_mse = diag["before"]["fit"]["meanSquaredError"]
                .as_f64()
                .ok_or("Missing before fit mse")?;
            let after_fit_mse = diag["after"]["fit"]["meanSquaredError"]
                .as_f64()
                .ok_or("Missing after fit mse")?;
            let before_dev_mse = diag["before"]["development"]["meanSquaredError"]
                .as_f64()
                .ok_or("Missing before dev mse")?;
            let after_dev_mse = diag["after"]["development"]["meanSquaredError"]
                .as_f64()
                .ok_or("Missing after dev mse")?;
            assert!(after_fit_mse < before_fit_mse * 0.25);
            assert!(after_dev_mse < before_dev_mse * 0.5);
            assert!(diag["after"]["development"]["meanCosineSimilarity"].is_f64());
            assert!(diag["after"]["development"]["minCosineSimilarity"].is_f64());
            assert!(diag["after"]["development"]["worstRowRelativeL2Error"].is_f64());
            assert!(diag["after"]["development"]["relativeSquaredError"].is_f64());

            if mode == "alternating-ab" {
                alt_dev_mse = after_dev_mse;
                let reloaded = read_lora_gguf_pairs_with_dims(&out_path, &dims)?;
                let p3 = reloaded
                    .pairs
                    .iter()
                    .find(|p| p.gguf_base_name == "blk.3.attn_k.weight")
                    .ok_or("Missing untouched blk.3")?;
                assert_eq!(p3.lora_a_f32, untouched_expected);
            }
        }
        assert!(alt_dev_mse < 1e-4, "alternating_ab dev_mse={alt_dev_mse}");

        // Fail-closed development regression gate: adversarial dev targets regress MSE!
        let mut reg_rows = rows;
        for row in &mut reg_rows {
            if row.split == RealignSplit::Development {
                for v in &mut row.target_output {
                    *v = -*v;
                }
            }
        }
        let reg_out = scratch.join("rejected-regression.gguf");
        let reg_req: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "realign-factors",
            "adapterPath": src,
            "outputPath": reg_out,
            "realignConfig": {
                "mode": "realign-b",
                "ridge": 1e-4,
                "allowDevelopmentRegression": false,
                "diagnosticPromptSha256s": [diag_a],
                "sites": [{
                    "target": "blk.63.ffn_down.weight",
                    "rows": reg_rows
                }]
            }
        }))
        .map_err(|e| e.to_string())?;
        let reg_res = run_realign_factors_with_dims(&reg_req, &dims)?;
        assert_eq!(reg_res["published"], false);
        assert_eq!(reg_res["status"], "development-regression-rejected");
        assert_eq!(reg_res["regressedSites"], json!(["blk.63.ffn_down.weight"]));
        assert!(
            !reg_out.exists(),
            "Rejected run must NOT write output GGUF file!"
        );

        // Same rows with allowDevelopmentRegression: true must publish!
        let allow_out = scratch.join("allowed-regression.gguf");
        let allow_req: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "realign-factors",
            "adapterPath": src,
            "outputPath": allow_out,
            "realignConfig": {
                "mode": "realign-b",
                "ridge": 1e-4,
                "allowDevelopmentRegression": true,
                "diagnosticPromptSha256s": [diag_a],
                "sites": [{
                    "target": "blk.63.ffn_down.weight",
                    "rows": reg_rows
                }]
            }
        }))
        .map_err(|e| e.to_string())?;
        let allow_res = run_realign_factors_with_dims(&allow_req, &dims)?;
        assert_eq!(allow_res["published"], true);
        assert!(allow_out.exists());
        Ok(())
    }

    #[test]
    fn realign_factors_rejects_split_leakage_and_invalid_configs() -> crate::Result<()> {
        let scratch = ScratchDir::new("realign-validation")?;
        let src = scratch.join("real-dims-src.gguf");
        let in_f = 17_408_usize;
        let out_f = 5_120_usize;
        let mut a = vec![0.0_f32; in_f];
        let mut b = vec![0.0_f32; out_f];
        a[0] = 0.25;
        b[0] = 0.5;
        write_lora_gguf(
            &src,
            ExportDtype::F16,
            1.0,
            &[],
            &[ExportTensorPair {
                gguf_base_name: "blk.63.ffn_down.weight".to_owned(),
                in_features: in_f,
                out_features: out_f,
                rank: 1,
                lora_a_f32: a,
                lora_b_f32: b,
            }],
        )?;

        let diag = "cc".repeat(32);
        let mk_vec = |len: usize, val: f32| -> Vec<f32> {
            let mut v = vec![0.0_f32; len];
            v[0] = val;
            v
        };
        let valid_rows = vec![
            make_realign_row(
                "s1",
                "g1",
                0,
                RealignSplit::Fit,
                mk_vec(in_f, 1.0),
                mk_vec(out_f, 0.4),
            ),
            make_realign_row(
                "s2",
                "g2",
                0,
                RealignSplit::Fit,
                mk_vec(in_f, 2.0),
                mk_vec(out_f, 0.8),
            ),
            make_realign_row(
                "s3",
                "g3",
                0,
                RealignSplit::Development,
                mk_vec(in_f, 1.5),
                mk_vec(out_f, 0.6),
            ),
        ];

        // Smoke test real Swift 17408 x 5120 F16 end-to-end through top-level lora_diff_audit::run!
        let real_out = scratch.join("real-realigned.gguf");
        let real_res = run(json!({
            "schemaVersion": 1,
            "operation": "realign-factors",
            "adapterPath": src,
            "outputPath": real_out,
            "realignConfig": {
                "mode": "alternating-ab",
                "alternatingSteps": 2,
                "ridge": 1e-3,
                "diagnosticPromptSha256s": [diag],
                "sites": [{
                    "target": "blk.63.ffn_down.weight",
                    "rows": valid_rows
                }]
            }
        }))?;
        assert_eq!(real_res["published"], true);
        assert_eq!(real_res["verification"]["verified"], true);

        // 1. SampleId crossing split
        let cross_sample = vec![
            make_realign_row(
                "s1",
                "g1",
                0,
                RealignSplit::Fit,
                mk_vec(in_f, 1.0),
                mk_vec(out_f, 0.4),
            ),
            make_realign_row(
                "s2",
                "g2",
                0,
                RealignSplit::Fit,
                mk_vec(in_f, 2.0),
                mk_vec(out_f, 0.8),
            ),
            make_realign_row(
                "s1",
                "g3",
                1,
                RealignSplit::Development,
                mk_vec(in_f, 1.5),
                mk_vec(out_f, 0.6),
            ),
        ];
        // 2. GroupId crossing split
        let cross_group = vec![
            make_realign_row(
                "s1",
                "g1",
                0,
                RealignSplit::Fit,
                mk_vec(in_f, 1.0),
                mk_vec(out_f, 0.4),
            ),
            make_realign_row(
                "s2",
                "g2",
                0,
                RealignSplit::Fit,
                mk_vec(in_f, 2.0),
                mk_vec(out_f, 0.8),
            ),
            make_realign_row(
                "s3",
                "g1",
                0,
                RealignSplit::Development,
                mk_vec(in_f, 1.5),
                mk_vec(out_f, 0.6),
            ),
        ];
        // 3. Identical F32 input crossing split
        let cross_input = vec![
            make_realign_row(
                "s1",
                "g1",
                0,
                RealignSplit::Fit,
                mk_vec(in_f, 1.0),
                mk_vec(out_f, 0.4),
            ),
            make_realign_row(
                "s2",
                "g2",
                0,
                RealignSplit::Fit,
                mk_vec(in_f, 2.0),
                mk_vec(out_f, 0.8),
            ),
            make_realign_row(
                "s3",
                "g3",
                0,
                RealignSplit::Development,
                mk_vec(in_f, 1.0),
                mk_vec(out_f, 0.4),
            ),
        ];
        // 4. Only 1 fit row (needs >= 2)
        let one_fit = vec![
            make_realign_row(
                "s1",
                "g1",
                0,
                RealignSplit::Fit,
                mk_vec(in_f, 1.0),
                mk_vec(out_f, 0.4),
            ),
            make_realign_row(
                "s2",
                "g2",
                0,
                RealignSplit::Development,
                mk_vec(in_f, 2.0),
                mk_vec(out_f, 0.8),
            ),
            make_realign_row(
                "s3",
                "g3",
                0,
                RealignSplit::Development,
                mk_vec(in_f, 3.0),
                mk_vec(out_f, 1.2),
            ),
        ];

        for (idx, bad_rows) in [cross_sample, cross_group, cross_input, one_fit]
            .into_iter()
            .enumerate()
        {
            assert!(
                run(json!({
                    "schemaVersion": 1,
                    "operation": "realign-factors",
                    "adapterPath": src,
                    "outputPath": scratch.join(format!("bad-rows-{idx}.gguf")),
                    "realignConfig": {
                        "mode": "realign-b",
                        "diagnosticPromptSha256s": [diag],
                        "sites": [{
                            "target": "blk.63.ffn_down.weight",
                            "rows": bad_rows
                        }]
                    }
                }))
                .is_err(),
                "Expected bad_rows[{idx}] to be rejected"
            );
        }

        // alternatingSteps forbidden unless alternating-ab
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "realign-factors",
                "adapterPath": src,
                "outputPath": scratch.join("bad-steps.gguf"),
                "realignConfig": {
                    "mode": "realign-b",
                    "alternatingSteps": 2,
                    "diagnosticPromptSha256s": [diag],
                    "sites": [{ "target": "blk.63.ffn_down.weight", "rows": valid_rows }]
                }
            }))
            .is_err()
        );

        // Both rows and rowsPath provided must fail
        let dummy_rows_path = scratch.join("dummy-rows.json");
        fs::write(&dummy_rows_path, b"[]").map_err(|e| e.to_string())?;
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "realign-factors",
                "adapterPath": src,
                "outputPath": scratch.join("both-rows.gguf"),
                "realignConfig": {
                    "mode": "realign-b",
                    "diagnosticPromptSha256s": [diag],
                    "sites": [{
                        "target": "blk.63.ffn_down.weight",
                        "rows": valid_rows,
                        "rowsPath": dummy_rows_path
                    }]
                }
            }))
            .is_err()
        );

        // Neither rows nor rowsPath provided must fail
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "realign-factors",
                "adapterPath": src,
                "outputPath": scratch.join("neither-rows.gguf"),
                "realignConfig": {
                    "mode": "realign-b",
                    "diagnosticPromptSha256s": [diag],
                    "sites": [{ "target": "blk.63.ffn_down.weight" }]
                }
            }))
            .is_err()
        );

        // Ridge out of bounds must fail
        assert!(
            run(json!({
                "schemaVersion": 1,
                "operation": "realign-factors",
                "adapterPath": src,
                "outputPath": scratch.join("bad-ridge.gguf"),
                "realignConfig": {
                    "mode": "realign-b",
                    "ridge": 1e-12,
                    "diagnosticPromptSha256s": [diag],
                    "sites": [{ "target": "blk.63.ffn_down.weight", "rows": valid_rows }]
                }
            }))
            .is_err()
        );

        // Unknown field rejected
        let bad_field: Result<Value, _> = run(json!({
            "schemaVersion": 1,
            "operation": "realign-factors",
            "adapterPath": src,
            "outputPath": scratch.join("bad-field.gguf"),
            "realignConfig": {
                "mode": "realign-b",
                "unknownOpt": true,
                "diagnosticPromptSha256s": [diag],
                "sites": [{ "target": "blk.63.ffn_down.weight", "rows": valid_rows }]
            }
        }));
        assert!(bad_field.is_err());

        Ok(())
    }
}
