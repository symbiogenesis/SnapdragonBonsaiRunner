//! Activation-guided pure-Rust multi-rank LoRA editing (`fold-patch`, `realign-factors`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::gguf_adapter::{
    ExportDtype, ExportTensorPair, classify_families, read_lora_gguf_pairs_with_dims,
    sort_pairs_canonically, swift_base_dims_map, write_lora_gguf,
};
use super::transform::{
    check_factor_value, effective_delta_frobenius_norm_f64, verify_published_adapter,
};
use super::{Operation, Request, usize_to_f32};

const MAX_FOLD_PATCHES: usize = 64;
const MAX_LORA_RANK: usize = 64;
const MAX_PATCH_FILE_BYTES: u64 = 4 * 1024 * 1024;
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use serde_json::{Value, json};

    use super::super::gguf_adapter::{
        ExportDtype, ExportTensorPair, read_lora_gguf_pairs_with_dims, write_lora_gguf,
    };
    use super::super::{Request, run, validate_request};
    use super::{f32_le_sha256, run_fold_patch_with_dims, thin_qr_f64};
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
}
