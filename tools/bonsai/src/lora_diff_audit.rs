//! Read-only BF16 weight-diff and SVD spectrum audit for Gate 1 (`DR1`).

mod safetensors;
mod svd;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use self::safetensors::{
    FloatDtype, SafetensorsIndex, ShardSource, TensorCategory, TensorDescriptor, TensorSpec,
    coalesce_small_spans, decode_scalar_f32, swift_tensor_inventory,
};
use self::svd::{CumulativeEnergy, StreamingTwoSidedSketch, SvdSpectrum, randomized_subspace_svd};

const DAISYROSE_REPO: &str = "RadixArk/DaisyRose-27B";
const DAISYROSE_REVISION: &str = "64a4ec665091bb0c1931babce4e97e686eef3c5c";
const QWEN38_REPO: &str = "Qwen/Qwen3.8-27B";
const QWEN38_REVISION: &str = "37561816e66b5b9153b1b64c10968c18c1e6f4a0";
const HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";
const VOCAB_CHUNK_ROWS: usize = 4_096;
const SMALL_TENSOR_MAX_BYTES: u64 = 4 * 1024 * 1024;
const STREAMING_THRESHOLD_ELEMENTS: usize = 100_000_000;

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Operation {
    Plan,
    Audit,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum SourceSpec {
    #[serde(rename = "pinned-daisyrose")]
    PinnedDaisyrose,
    LocalDirectories {
        #[serde(rename = "candidateDirectory")]
        candidate_directory: PathBuf,
        #[serde(rename = "baseDirectory")]
        base_directory: PathBuf,
    },
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct Scope {
    layers: Vec<u32>,
    include_output_norm: bool,
    include_vocab: bool,
    include_svd: bool,
    max_rank: usize,
    oversampling: usize,
    power_iterations: usize,
}

impl Default for Scope {
    fn default() -> Self {
        Self {
            layers: vec![0, 3, 31, 63],
            include_output_norm: true,
            include_vocab: true,
            include_svd: true,
            max_rank: 64,
            oversampling: 16,
            power_iterations: 2,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct Controls {
    time_budget_seconds: u64,
    workers: usize,
    http_concurrency: usize,
}

impl Default for Controls {
    fn default() -> Self {
        let parallel = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);
        Self {
            time_budget_seconds: 300,
            workers: parallel.clamp(1, 16),
            http_concurrency: 2,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operation: Operation,
    source: SourceSpec,
    #[serde(default)]
    scope: Scope,
    #[serde(default)]
    controls: Controls,
    output_path: Option<PathBuf>,
}

#[expect(
    clippy::as_conversions,
    reason = "Scalar integer counts convert cleanly to f64 for relative error and density metrics."
)]
#[inline]
const fn u64_to_f64(value: u64) -> f64 {
    value as f64
}

#[expect(
    clippy::as_conversions,
    reason = "Scalar element counts convert cleanly to f64 for mean absolute delta."
)]
#[inline]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

fn validate_output_path(path: &Path) -> crate::Result<()> {
    bonsai_artifacts::validate_external_path(path).map_err(|e| e.to_string())?;
    if fs::exists(path).map_err(|e| e.to_string())? {
        return Err("outputPath must name a new external file that does not already exist.".into());
    }
    if path.parent().is_none_or(|parent| !parent.is_dir()) {
        return Err("The outputPath parent directory must already exist.".into());
    }
    Ok(())
}

fn validate_request(request: &Request) -> crate::Result<()> {
    if request.schema_version != 1 {
        return Err("lora-diff-audit requires schemaVersion 1.".into());
    }
    let mut seen_layers = BTreeSet::new();
    for &layer in &request.scope.layers {
        if layer >= 64 {
            return Err("Selected layer index must be in 0..64.".into());
        }
        if !seen_layers.insert(layer) {
            return Err(format!("Duplicate layer index {layer} in scope.layers."));
        }
    }
    if request.scope.layers.is_empty()
        && !request.scope.include_output_norm
        && !request.scope.include_vocab
    {
        return Err("Audit scope selects zero tensors.".into());
    }
    if !(1..=64).contains(&request.scope.max_rank) {
        return Err("scope.maxRank must be in 1..=64.".into());
    }
    if request.scope.oversampling > 64 {
        return Err("scope.oversampling must be in 0..=64.".into());
    }
    if request.scope.power_iterations > 6 {
        return Err("scope.powerIterations must be in 0..=6.".into());
    }
    if !(10..=900).contains(&request.controls.time_budget_seconds) {
        return Err("controls.timeBudgetSeconds must be in 10..=900.".into());
    }
    if !(1..=32).contains(&request.controls.workers) {
        return Err("controls.workers must be in 1..=32.".into());
    }
    if !(1..=4).contains(&request.controls.http_concurrency) {
        return Err("controls.httpConcurrency must be in 1..=4.".into());
    }
    if let SourceSpec::LocalDirectories {
        candidate_directory,
        base_directory,
    } = &request.source
    {
        bonsai_artifacts::validate_external_path(candidate_directory).map_err(|e| e.to_string())?;
        bonsai_artifacts::validate_external_path(base_directory).map_err(|e| e.to_string())?;
        if !candidate_directory.is_dir() || !base_directory.is_dir() {
            return Err("Local candidateDirectory and baseDirectory must both exist.".into());
        }
    }
    if let Some(output) = &request.output_path {
        validate_output_path(output)?;
    }
    Ok(())
}

fn select_tensors(scope: &Scope) -> Vec<TensorSpec> {
    let layer_set: BTreeSet<u32> = scope.layers.iter().copied().collect();
    let inventory = swift_tensor_inventory();
    let mut selected = Vec::new();
    for spec in inventory {
        let include = match spec.layer {
            Some(layer) => layer_set.contains(&layer),
            None => match spec.category {
                TensorCategory::NonMatrix => scope.include_output_norm,
                TensorCategory::VocabMatrix => scope.include_vocab,
                TensorCategory::GdnSmallMatrix | TensorCategory::LinearMatrix => false,
            },
        };
        if include {
            selected.push(spec);
        }
    }
    // Order by increasing cost: non-matrix -> small GDN -> block linear -> vocab.
    selected.sort_by_key(|spec| (spec.category, spec.layer.unwrap_or(u32::MAX)));
    selected
}

fn estimated_bf16_bytes(spec: &TensorSpec) -> u64 {
    let elements: u64 = match spec.family {
        "token_embd" | "output" => 248_320 * 5_120,
        "norm" => {
            if spec.gguf_name.contains("q_norm") || spec.gguf_name.contains("k_norm") {
                256
            } else {
                5_120
            }
        }
        "ffn_gate" | "ffn_up" | "ffn_down" => 17_408 * 5_120,
        "attn_q" => 12_288 * 5_120,
        "attn_k" | "attn_v" => 1_024 * 5_120,
        "attn_output" | "attn_gate" | "ssm_out" => 6_144 * 5_120,
        "attn_qkv" => 10_240 * 5_120,
        "ssm_alpha" | "ssm_beta" => 48 * 5_120,
        "ssm_conv1d" => 10_240 * 4,
        "ssm_norm" => 128,
        "ssm_a" | "ssm_dt_bias" => 48,
        _ => 0,
    };
    elements * 2
}

fn source_summary(source: &SourceSpec) -> Value {
    match source {
        SourceSpec::PinnedDaisyrose => json!({
            "kind": "pinned-daisyrose",
            "candidate": {
                "repository": DAISYROSE_REPO,
                "revision": DAISYROSE_REVISION,
                "baseUrl": HUGGINGFACE_BASE_URL
            },
            "base": {
                "repository": QWEN38_REPO,
                "revision": QWEN38_REVISION,
                "baseUrl": HUGGINGFACE_BASE_URL
            }
        }),
        SourceSpec::LocalDirectories {
            candidate_directory,
            base_directory,
        } => json!({
            "kind": "local-directories",
            "candidateDirectory": candidate_directory,
            "baseDirectory": base_directory
        }),
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TensorReport {
    hf_name: String,
    gguf_name: String,
    layer: Option<u32>,
    family: String,
    category: String,
    dtype: String,
    shape: Vec<usize>,
    element_count: usize,
    byte_size_per_model: u64,
    base_frobenius_norm: f64,
    candidate_frobenius_norm: f64,
    delta_frobenius_energy: f64,
    delta_frobenius_norm: f64,
    relative_frobenius_delta: f64,
    mean_abs_delta: f64,
    max_abs_delta: f64,
    nonzero_delta_count: u64,
    nonzero_delta_fraction: f64,
    changed_rows: Option<usize>,
    total_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    svd: Option<SvdSpectrum>,
    #[serde(skip)]
    base_frobenius_energy: f64,
}

#[derive(Default)]
struct MetricsAccum {
    tensor_count: usize,
    total_elements: u64,
    total_base_energy: f64,
    total_delta_energy: f64,
    sum_rel_delta: f64,
    max_rel_delta: f64,
    max_abs_delta: f64,
    total_nonzero: u64,
    svd_weight_energy: f64,
    weighted_r1: f64,
    weighted_r4: f64,
    weighted_r8: f64,
    weighted_r16: f64,
    weighted_r32: f64,
    weighted_r64: f64,
}

impl MetricsAccum {
    fn add(&mut self, report: &TensorReport) {
        self.tensor_count += 1;
        self.total_elements += u64::try_from(report.element_count).unwrap_or(u64::MAX);
        self.total_base_energy += report.base_frobenius_energy;
        self.total_delta_energy += report.delta_frobenius_energy;
        self.sum_rel_delta += report.relative_frobenius_delta;
        self.max_rel_delta = self.max_rel_delta.max(report.relative_frobenius_delta);
        self.max_abs_delta = self.max_abs_delta.max(report.max_abs_delta);
        self.total_nonzero += report.nonzero_delta_count;
        if let Some(svd) = &report.svd
            && report.delta_frobenius_energy > 0.0
        {
            let w = report.delta_frobenius_energy;
            self.svd_weight_energy += w;
            self.weighted_r1 += w * svd.cumulative_energy.r1;
            self.weighted_r4 += w * svd.cumulative_energy.r4;
            self.weighted_r8 += w * svd.cumulative_energy.r8;
            self.weighted_r16 += w * svd.cumulative_energy.r16;
            self.weighted_r32 += w * svd.cumulative_energy.r32;
            self.weighted_r64 += w * svd.cumulative_energy.r64;
        }
    }

    fn aggregate_relative(&self) -> f64 {
        if self.total_base_energy > 0.0 {
            (self.total_delta_energy / self.total_base_energy).sqrt()
        } else {
            0.0
        }
    }

    fn mean_relative(&self) -> f64 {
        if self.tensor_count > 0 {
            self.sum_rel_delta / usize_to_f64(self.tensor_count)
        } else {
            0.0
        }
    }

    fn weighted_svd(&self) -> Option<CumulativeEnergy> {
        if self.svd_weight_energy <= 0.0 {
            return None;
        }
        let inv = 1.0 / self.svd_weight_energy;
        Some(CumulativeEnergy {
            r1: (self.weighted_r1 * inv).clamp(0.0, 1.0),
            r4: (self.weighted_r4 * inv).clamp(0.0, 1.0),
            r8: (self.weighted_r8 * inv).clamp(0.0, 1.0),
            r16: (self.weighted_r16 * inv).clamp(0.0, 1.0),
            r32: (self.weighted_r32 * inv).clamp(0.0, 1.0),
            r64: (self.weighted_r64 * inv).clamp(0.0, 1.0),
        })
    }

    fn to_json(&self, audited_energy: f64, non_vocab_energy: f64, linear_energy: f64) -> Value {
        let nonzero_fraction = if self.total_elements > 0 {
            u64_to_f64(self.total_nonzero) / u64_to_f64(self.total_elements)
        } else {
            0.0
        };
        let share_audited = if audited_energy > 0.0 {
            self.total_delta_energy / audited_energy
        } else {
            0.0
        };
        let share_non_vocab = if non_vocab_energy > 0.0 {
            self.total_delta_energy / non_vocab_energy
        } else {
            0.0
        };
        let share_linear = if linear_energy > 0.0 {
            self.total_delta_energy / linear_energy
        } else {
            0.0
        };
        json!({
            "tensorCount": self.tensor_count,
            "totalElements": self.total_elements,
            "totalBaseFrobeniusEnergy": self.total_base_energy,
            "totalDeltaFrobeniusEnergy": self.total_delta_energy,
            "aggregateRelativeFrobeniusDelta": self.aggregate_relative(),
            "meanTensorRelativeFrobeniusDelta": self.mean_relative(),
            "maxTensorRelativeFrobeniusDelta": self.max_rel_delta,
            "maxAbsDelta": self.max_abs_delta,
            "nonzeroDeltaFraction": nonzero_fraction,
            "shareOfAuditedEnergy": share_audited,
            "shareOfNonVocabEnergy": share_non_vocab,
            "shareOfLinearMatrixEnergy": share_linear,
            "energyWeightedCumulativeEnergy": self.weighted_svd(),
        })
    }
}

struct TensorDiffStats {
    base_energy: f64,
    cand_energy: f64,
    delta_energy: f64,
    sum_abs_delta: f64,
    max_abs_delta: f64,
    nonzero_count: u64,
    changed_rows: Option<usize>,
}

#[inline]
fn diff_chunk_rows(
    dtype: FloatDtype,
    cand_bytes: &[u8],
    base_bytes: &[u8],
    rows: usize,
    cols: usize,
    delta_out: &mut [f32],
    workers: usize,
) -> crate::Result<TensorDiffStats> {
    let width = dtype.byte_width();
    let expected_bytes = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(width))
        .ok_or("Diff chunk byte size overflow.")?;
    if cand_bytes.len() != expected_bytes
        || base_bytes.len() != expected_bytes
        || delta_out.len() != rows * cols
    {
        return Err("Diff chunk buffer length mismatch.".into());
    }
    if rows == 0 || cols == 0 {
        return Ok(TensorDiffStats {
            base_energy: 0.0,
            cand_energy: 0.0,
            delta_energy: 0.0,
            sum_abs_delta: 0.0,
            max_abs_delta: 0.0,
            nonzero_count: 0,
            changed_rows: Some(0),
        });
    }
    let worker_count = workers.clamp(1, rows);
    let rows_per_worker = rows.div_ceil(worker_count);
    let partials: Vec<crate::Result<TensorDiffStats>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (idx, d_part) in delta_out.chunks_mut(rows_per_worker * cols).enumerate() {
            let r_start = idx * rows_per_worker;
            let local_rows = d_part.len() / cols;
            let byte_start = r_start * cols * width;
            let byte_end = (r_start + local_rows) * cols * width;
            let c_part = &cand_bytes[byte_start..byte_end];
            let b_part = &base_bytes[byte_start..byte_end];
            handles.push(scope.spawn(move || {
                let mut base_energy = 0.0_f64;
                let mut cand_energy = 0.0_f64;
                let mut delta_energy = 0.0_f64;
                let mut sum_abs_delta = 0.0_f64;
                let mut max_abs_delta = 0.0_f64;
                let mut nonzero_count = 0_u64;
                let mut changed_rows = 0_usize;
                for r in 0..local_rows {
                    let mut row_changed = false;
                    let row_base = r * cols;
                    for c in 0..cols {
                        let elem_idx = row_base + c;
                        let byte_off = elem_idx * width;
                        let cw = decode_scalar_f32(dtype, &c_part[byte_off..byte_off + width])?;
                        let bw = decode_scalar_f32(dtype, &b_part[byte_off..byte_off + width])?;
                        let diff = cw - bw;
                        if !diff.is_finite() {
                            return Err("Weight delta overflowed F32.".into());
                        }
                        d_part[elem_idx] = diff;
                        let cf = f64::from(cw);
                        let bf = f64::from(bw);
                        let df = cf - bf;
                        base_energy += bf * bf;
                        cand_energy += cf * cf;
                        delta_energy += df * df;
                        let ad = df.abs();
                        sum_abs_delta += ad;
                        max_abs_delta = max_abs_delta.max(ad);
                        if cw != bw {
                            nonzero_count += 1;
                            row_changed = true;
                        }
                    }
                    if row_changed {
                        changed_rows += 1;
                    }
                }
                Ok(TensorDiffStats {
                    base_energy,
                    cand_energy,
                    delta_energy,
                    sum_abs_delta,
                    max_abs_delta,
                    nonzero_count,
                    changed_rows: Some(changed_rows),
                })
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("Diff worker thread panicked.".into()))
            })
            .collect()
    });

    let mut total = TensorDiffStats {
        base_energy: 0.0,
        cand_energy: 0.0,
        delta_energy: 0.0,
        sum_abs_delta: 0.0,
        max_abs_delta: 0.0,
        nonzero_count: 0,
        changed_rows: Some(0),
    };
    for part in partials {
        let part = part?;
        total.base_energy += part.base_energy;
        total.cand_energy += part.cand_energy;
        total.delta_energy += part.delta_energy;
        total.sum_abs_delta += part.sum_abs_delta;
        total.max_abs_delta = total.max_abs_delta.max(part.max_abs_delta);
        total.nonzero_count += part.nonzero_count;
        total.changed_rows = Some(total.changed_rows.unwrap_or(0) + part.changed_rows.unwrap_or(0));
    }
    Ok(total)
}

fn fetch_pair_slices(
    cand_src: &ShardSource,
    cand_shard: &str,
    cand_begin: u64,
    cand_len: u64,
    base_src: &ShardSource,
    base_shard: &str,
    base_begin: u64,
    base_len: u64,
    deadline: Instant,
) -> crate::Result<(Vec<u8>, Vec<u8>)> {
    std::thread::scope(|scope| {
        let c_handle =
            scope.spawn(move || cand_src.fetch_slice(cand_shard, cand_begin, cand_len, deadline));
        let b_res = base_src.fetch_slice(base_shard, base_begin, base_len, deadline);
        let c_res = c_handle
            .join()
            .unwrap_or_else(|_| Err("Candidate fetch thread panicked.".into()))?;
        let b_res = b_res?;
        Ok((c_res, b_res))
    })
}

fn fetch_coalesced_side(
    src: &ShardSource,
    items: &[(usize, &str, u64, u64)],
    deadline: Instant,
) -> crate::Result<BTreeMap<usize, Vec<u8>>> {
    let spans = coalesce_small_spans(items);
    let mut out = BTreeMap::new();
    for span in spans {
        if Instant::now() >= deadline {
            return Err("Audit time budget expired during small-tensor transfer.".into());
        }
        crate::startup_cancel::checkpoint()?;
        let buf = src.fetch_slice(
            &span.shard,
            span.abs_begin,
            span.abs_end - span.abs_begin,
            deadline,
        )?;
        for (idx, begin, end) in span.entries {
            let rel_begin = usize::try_from(begin - span.abs_begin).map_err(|e| e.to_string())?;
            let rel_end = usize::try_from(end - span.abs_begin).map_err(|e| e.to_string())?;
            out.insert(idx, buf[rel_begin..rel_end].to_vec());
        }
    }
    Ok(out)
}

fn build_tensor_report(
    spec: &TensorSpec,
    dtype: FloatDtype,
    shape: Vec<usize>,
    element_count: usize,
    byte_size: u64,
    stats: TensorDiffStats,
    svd: Option<SvdSpectrum>,
) -> TensorReport {
    let base_norm = stats.base_energy.sqrt();
    let cand_norm = stats.cand_energy.sqrt();
    let delta_norm = stats.delta_energy.sqrt();
    let rel = if base_norm > 0.0 {
        delta_norm / base_norm
    } else {
        0.0
    };
    let elem_f = usize_to_f64(element_count.max(1));
    TensorReport {
        hf_name: spec.hf_name.clone(),
        gguf_name: spec.gguf_name.clone(),
        layer: spec.layer,
        family: spec.family.to_owned(),
        category: spec.category.as_str().to_owned(),
        dtype: dtype.as_str().to_owned(),
        total_rows: (shape.len() == 2).then(|| shape[0]),
        shape,
        element_count,
        byte_size_per_model: byte_size,
        base_frobenius_norm: base_norm,
        candidate_frobenius_norm: cand_norm,
        delta_frobenius_energy: stats.delta_energy,
        delta_frobenius_norm: delta_norm,
        relative_frobenius_delta: rel,
        mean_abs_delta: stats.sum_abs_delta / elem_f,
        max_abs_delta: stats.max_abs_delta,
        nonzero_delta_count: stats.nonzero_count,
        nonzero_delta_fraction: u64_to_f64(stats.nonzero_count) / elem_f,
        changed_rows: stats.changed_rows,
        svd,
        base_frobenius_energy: stats.base_energy,
    }
}

struct PreparedTensor {
    spec: TensorSpec,
    cand_shard: String,
    base_shard: String,
    desc: TensorDescriptor,
    dtype: FloatDtype,
    element_count: usize,
    byte_size: u64,
}

fn resolve_and_validate_tensors(
    selected: Vec<TensorSpec>,
    cand_src: &ShardSource,
    cand_idx: &SafetensorsIndex,
    base_src: &ShardSource,
    base_idx: &SafetensorsIndex,
    deadline: Instant,
) -> crate::Result<Vec<PreparedTensor>> {
    let mut cand_shards = BTreeSet::new();
    let mut base_shards = BTreeSet::new();
    let mut shard_pairs = Vec::with_capacity(selected.len());
    for spec in selected {
        let cs = cand_idx
            .weight_map
            .get(&spec.hf_name)
            .ok_or_else(|| format!("Candidate index missing '{}'.", spec.hf_name))?
            .clone();
        let bs = base_idx
            .weight_map
            .get(&spec.hf_name)
            .ok_or_else(|| format!("Base index missing '{}'.", spec.hf_name))?
            .clone();
        cand_shards.insert(cs.clone());
        base_shards.insert(bs.clone());
        shard_pairs.push((spec, cs, bs));
    }

    std::thread::scope(|scope| {
        let c_handle = scope.spawn(move || -> crate::Result<()> {
            for shard in &cand_shards {
                crate::startup_cancel::checkpoint()?;
                cand_src.ensure_shard_header(shard, deadline)?;
            }
            Ok(())
        });
        for shard in &base_shards {
            crate::startup_cancel::checkpoint()?;
            base_src.ensure_shard_header(shard, deadline)?;
        }
        c_handle
            .join()
            .unwrap_or_else(|_| Err("Candidate header thread panicked.".into()))?;
        Ok::<_, String>(())
    })?;

    let mut prepared = Vec::with_capacity(shard_pairs.len());
    for (spec, cand_shard, base_shard) in shard_pairs {
        let c_desc = cand_src.descriptor(&cand_shard, &spec.hf_name)?;
        let b_desc = base_src.descriptor(&base_shard, &spec.hf_name)?;
        if c_desc.dtype != b_desc.dtype || c_desc.shape != b_desc.shape {
            return Err(format!(
                "Tensor '{}' schema mismatch: candidate {} {:?} vs base {} {:?}.",
                spec.hf_name, c_desc.dtype, c_desc.shape, b_desc.dtype, b_desc.shape
            ));
        }
        let dtype = c_desc.float_dtype()?;
        let element_count = c_desc.element_count()?;
        let byte_size = c_desc.abs_end - c_desc.abs_begin;
        if b_desc.abs_end - b_desc.abs_begin != byte_size {
            return Err(format!("Byte extent mismatch on '{}'.", spec.hf_name));
        }
        if matches!(
            spec.category,
            TensorCategory::GdnSmallMatrix
                | TensorCategory::LinearMatrix
                | TensorCategory::VocabMatrix
        ) && c_desc.shape.len() != 2
        {
            return Err(format!(
                "Expected 2D matrix shape for '{}', got {:?}.",
                spec.hf_name, c_desc.shape
            ));
        }
        prepared.push(PreparedTensor {
            spec,
            cand_shard,
            base_shard,
            desc: c_desc,
            dtype,
            element_count,
            byte_size,
        });
    }
    Ok(prepared)
}

fn evaluate_gate1(
    completed: bool,
    include_svd: bool,
    skipped_count: usize,
    all_audited: &MetricsAccum,
    non_vocab_all: &MetricsAccum,
    pure_non_matrix: &MetricsAccum,
    non_matrix_small_gdn: &MetricsAccum,
    linear_matrices: &MetricsAccum,
    ffn_and_full_attn: &MetricsAccum,
    cand_a_down_attn_o: &MetricsAccum,
    cand_b_gate_up_qkv: &MetricsAccum,
) -> Value {
    let mut reasons = Vec::new();
    let total_energy = all_audited.total_delta_energy;
    let non_vocab_energy = non_vocab_all.total_delta_energy;
    let non_matrix_small_gdn_share = if non_vocab_energy > 0.0 {
        non_matrix_small_gdn.total_delta_energy / non_vocab_energy
    } else {
        0.0
    };
    let pure_non_matrix_mean_rel = pure_non_matrix.mean_relative();
    let linear_mean_rel = linear_matrices.mean_relative();
    let ffn_attn_e32 = ffn_and_full_attn.weighted_svd().map_or(0.0, |e| e.r32);
    let ffn_attn_e16 = ffn_and_full_attn.weighted_svd().map_or(0.0, |e| e.r16);
    let cand_a_share = if linear_matrices.total_delta_energy > 0.0 {
        cand_a_down_attn_o.total_delta_energy / linear_matrices.total_delta_energy
    } else {
        0.0
    };
    let gate_up_qkv_share = if linear_matrices.total_delta_energy > 0.0 {
        cand_b_gate_up_qkv.total_delta_energy / linear_matrices.total_delta_energy
    } else {
        0.0
    };

    let (passed, verdict) = if total_energy == 0.0 {
        reasons.push("Zero difference detected across every audited tensor.".to_owned());
        (false, "no-go-zero-delta")
    } else if non_vocab_energy > 0.0 && non_matrix_small_gdn_share >= 0.50 {
        reasons.push(format!(
            "Non-matrix and small-GDN parameters hold {:.2}% (>= 50%) of non-vocab delta Frobenius energy.",
            non_matrix_small_gdn_share * 100.0
        ));
        (false, "no-go-non-matrix-dominant")
    } else if linear_mean_rel > 0.0 && pure_non_matrix_mean_rel > 5.0 * linear_mean_rel {
        reasons.push(format!(
            "Pure non-matrix mean relative Frobenius delta ({pure_non_matrix_mean_rel:.6}) exceeds 5.0x linear-matrix mean ({linear_mean_rel:.6}; ratio {:.2}x).",
            pure_non_matrix_mean_rel / linear_mean_rel
        ));
        (false, "no-go-non-matrix-dominant")
    } else if include_svd && ffn_and_full_attn.tensor_count > 0 && ffn_attn_e32 < 0.50 {
        reasons.push(format!(
            "Energy-weighted E(32) across FFN and full-attention matrices is {:.2}% (< 50% Gate 1 threshold).",
            ffn_attn_e32 * 100.0
        ));
        (false, "no-go-high-rank")
    } else if include_svd && ffn_and_full_attn.tensor_count > 0 && ffn_attn_e32 < 0.65 {
        reasons.push(format!(
            "Energy-weighted E(32) across FFN and full-attention matrices is {:.2}% (between 50% stop threshold and 65% low-rank knee; E(16)={:.2}%).",
            ffn_attn_e32 * 100.0,
            ffn_attn_e16 * 100.0
        ));
        (true, "borderline-needs-full-scan")
    } else if cand_a_share >= 0.40 || cand_a_share >= gate_up_qkv_share {
        reasons.push(format!(
            "FFN and full-attention linear matrices clear Gate 1 low-rank criteria (E(16)={:.2}%, E(32)={:.2}%) with Candidate A (ffn_down + attn_output) holding {:.2}% of block linear delta energy.",
            ffn_attn_e16 * 100.0,
            ffn_attn_e32 * 100.0,
            cand_a_share * 100.0
        ));
        (true, "pass-candidate-a")
    } else {
        reasons.push(format!(
            "FFN and full-attention linear matrices clear Gate 1 low-rank criteria (E(16)={:.2}%, E(32)={:.2}%), but input-side projections (ffn_gate/ffn_up/attn_qkv) hold {:.2}% vs {:.2}% in ffn_down+attn_output (Candidate B scope).",
            ffn_attn_e16 * 100.0,
            ffn_attn_e32 * 100.0,
            gate_up_qkv_share * 100.0,
            cand_a_share * 100.0
        ));
        (true, "pass-candidate-b")
    };

    if !completed {
        reasons.push(format!(
            "Audit stopped early before auditing {skipped_count} remaining tensor(s); evaluate skipped scope before drawing final Gate 1 conclusions."
        ));
    }

    json!({
        "passed": passed,
        "verdict": verdict,
        "reasons": reasons,
        "metrics": {
            "nonMatrixAndSmallGdnShareOfNonVocabEnergy": non_matrix_small_gdn_share,
            "pureNonMatrixMeanRelativeDelta": pure_non_matrix_mean_rel,
            "linearMatrixMeanRelativeDelta": linear_mean_rel,
            "ffnAndFullAttnEnergyWeightedE16": ffn_attn_e16,
            "ffnAndFullAttnEnergyWeightedE32": ffn_attn_e32,
            "candidateAShareOfBlockLinearEnergy": cand_a_share,
            "gateUpQkvShareOfBlockLinearEnergy": gate_up_qkv_share,
        }
    })
}

fn plan_response(request: &Request) -> crate::Result<Value> {
    let selected = select_tensors(&request.scope);
    let est_one: u64 = selected.iter().map(estimated_bf16_bytes).sum();
    let est_pair = est_one.saturating_mul(2);
    let receipt = json!({
        "schemaVersion": 1,
        "operation": "plan",
        "status": "planned-no-io",
        "source": source_summary(&request.source),
        "scope": request.scope,
        "controls": request.controls,
        "selectedTensorCount": selected.len(),
        "estimatedBf16BytesOneModel": est_one,
        "estimatedBf16BytesPair": est_pair,
        "gate1Criteria": {
            "passTargetKneeE32": 0.65,
            "hardStopE32Below": 0.50,
            "hardStopNonMatrixShareAtLeast": 0.50,
            "hardStopPureNonMatrixToLinearMeanRatioAbove": 5.0
        }
    });
    if let Some(output) = &request.output_path {
        crate::bounded_input::write_pretty_json_new(output, &receipt)?;
    }
    Ok(receipt)
}

#[expect(
    clippy::too_many_lines,
    reason = "Orchestrates read-only intake, small-tensor coalescing, block SVD, streaming vocab sketches, power endpoints, and Gate 1 aggregation."
)]
fn run_audit(request: &Request) -> crate::Result<Value> {
    let started = Instant::now();
    let power_before = crate::host_cpu::power_snapshot();
    let deadline = started
        .checked_add(Duration::from_secs(request.controls.time_budget_seconds))
        .ok_or("Audit deadline overflow.")?;

    let per_side_http = (request.controls.http_concurrency / 2).max(1);
    let (cand_src, base_src) = match &request.source {
        SourceSpec::PinnedDaisyrose => (
            ShardSource::https(
                HUGGINGFACE_BASE_URL.to_owned(),
                DAISYROSE_REPO.to_owned(),
                DAISYROSE_REVISION.to_owned(),
                per_side_http,
            )?,
            ShardSource::https(
                HUGGINGFACE_BASE_URL.to_owned(),
                QWEN38_REPO.to_owned(),
                QWEN38_REVISION.to_owned(),
                per_side_http,
            )?,
        ),
        SourceSpec::LocalDirectories {
            candidate_directory,
            base_directory,
        } => (
            ShardSource::local(candidate_directory.clone())?,
            ShardSource::local(base_directory.clone())?,
        ),
    };

    let (cand_idx, base_idx) = std::thread::scope(|scope| {
        let c_handle = scope.spawn(|| cand_src.fetch_index(deadline));
        let b_res = base_src.fetch_index(deadline);
        let c_res = c_handle
            .join()
            .unwrap_or_else(|_| Err("Candidate index thread panicked.".into()))?;
        let b_res = b_res?;
        Ok::<_, String>((c_res, b_res))
    })?;

    let selected = select_tensors(&request.scope);
    let prepared = resolve_and_validate_tensors(
        selected, &cand_src, &cand_idx, &base_src, &base_idx, deadline,
    )?;

    let mut small_indices = Vec::new();
    let mut medium_indices = Vec::new();
    let mut streaming_indices = Vec::new();
    for (idx, item) in prepared.iter().enumerate() {
        if item.element_count > STREAMING_THRESHOLD_ELEMENTS && item.desc.shape.len() == 2 {
            streaming_indices.push(idx);
        } else if item.byte_size <= SMALL_TENSOR_MAX_BYTES {
            small_indices.push(idx);
        } else {
            medium_indices.push(idx);
        }
    }

    let mut reports = Vec::with_capacity(prepared.len());
    let mut skipped_tensors = Vec::new();
    let mut budget_exceeded = false;
    let k_sketch = request
        .scope
        .max_rank
        .saturating_add(request.scope.oversampling);
    let workers = request.controls.workers;

    // Phase 1: Small coalesced tensors (non-matrix + ssm_alpha/ssm_beta).
    if !small_indices.is_empty() {
        if Instant::now() >= deadline {
            budget_exceeded = true;
            for &idx in &small_indices {
                skipped_tensors.push(prepared[idx].spec.gguf_name.clone());
            }
        } else {
            let cand_items: Vec<(usize, &str, u64, u64)> = small_indices
                .iter()
                .map(|&idx| {
                    let p = &prepared[idx];
                    (idx, p.cand_shard.as_str(), p.desc.abs_begin, p.desc.abs_end)
                })
                .collect();
            let base_items: Vec<(usize, &str, u64, u64)> = small_indices
                .iter()
                .filter_map(|&idx| {
                    let p = &prepared[idx];
                    base_src
                        .descriptor(&p.base_shard, &p.spec.hf_name)
                        .ok()
                        .map(|bd| (idx, p.base_shard.as_str(), bd.abs_begin, bd.abs_end))
                })
                .collect();
            if base_items.len() != cand_items.len() {
                return Err("Base descriptor lookup failed for small tensor batch.".into());
            }
            let (cand_bufs, base_bufs) = std::thread::scope(|scope| {
                let c_handle =
                    scope.spawn(|| fetch_coalesced_side(&cand_src, &cand_items, deadline));
                let b_res = fetch_coalesced_side(&base_src, &base_items, deadline);
                let c_res = c_handle
                    .join()
                    .unwrap_or_else(|_| Err("Candidate small-tensor thread panicked.".into()))?;
                let b_res = b_res?;
                Ok::<_, String>((c_res, b_res))
            })?;

            for &idx in &small_indices {
                let p = &prepared[idx];
                let cb = cand_bufs
                    .get(&idx)
                    .ok_or("Missing candidate small-tensor slice.")?;
                let bb = base_bufs
                    .get(&idx)
                    .ok_or("Missing base small-tensor slice.")?;
                let (rows, cols) = if p.desc.shape.len() == 2 {
                    (p.desc.shape[0], p.desc.shape[1])
                } else {
                    (1, p.element_count)
                };
                let mut delta = vec![0.0_f32; p.element_count];
                let mut stats = diff_chunk_rows(p.dtype, cb, bb, rows, cols, &mut delta, 1)?;
                if p.desc.shape.len() != 2 {
                    stats.changed_rows = None;
                }
                let svd = if request.scope.include_svd && p.desc.shape.len() == 2 {
                    Some(randomized_subspace_svd(
                        &p.spec.hf_name,
                        &delta,
                        rows,
                        cols,
                        k_sketch,
                        request.scope.power_iterations,
                        stats.delta_energy,
                        workers,
                    )?)
                } else {
                    None
                };
                reports.push(build_tensor_report(
                    &p.spec,
                    p.dtype,
                    p.desc.shape.clone(),
                    p.element_count,
                    p.byte_size,
                    stats,
                    svd,
                ));
            }
        }
    }

    // Phase 2: Medium block linear matrices.
    for (pos, &idx) in medium_indices.iter().enumerate() {
        if Instant::now() >= deadline {
            budget_exceeded = true;
            for &rem in &medium_indices[pos..] {
                skipped_tensors.push(prepared[rem].spec.gguf_name.clone());
            }
            break;
        }
        crate::startup_cancel::checkpoint()?;
        let p = &prepared[idx];
        let b_desc = base_src.descriptor(&p.base_shard, &p.spec.hf_name)?;
        let (cb, bb) = match fetch_pair_slices(
            &cand_src,
            &p.cand_shard,
            p.desc.abs_begin,
            p.byte_size,
            &base_src,
            &p.base_shard,
            b_desc.abs_begin,
            p.byte_size,
            deadline,
        ) {
            Ok(pair) => pair,
            Err(_) if Instant::now() >= deadline => {
                budget_exceeded = true;
                for &rem in &medium_indices[pos..] {
                    skipped_tensors.push(prepared[rem].spec.gguf_name.clone());
                }
                break;
            }
            Err(err) => return Err(err),
        };
        let (rows, cols) = if p.desc.shape.len() == 2 {
            (p.desc.shape[0], p.desc.shape[1])
        } else {
            (1, p.element_count)
        };
        let mut delta = Vec::new();
        delta
            .try_reserve_exact(p.element_count)
            .map_err(|e| format!("Cannot allocate delta matrix for '{}': {e}", p.spec.hf_name))?;
        delta.resize(p.element_count, 0.0_f32);
        let mut stats = diff_chunk_rows(p.dtype, &cb, &bb, rows, cols, &mut delta, workers)?;
        if p.desc.shape.len() != 2 {
            stats.changed_rows = None;
        }
        drop(cb);
        drop(bb);
        let svd = if request.scope.include_svd && p.desc.shape.len() == 2 {
            Some(randomized_subspace_svd(
                &p.spec.hf_name,
                &delta,
                rows,
                cols,
                k_sketch,
                request.scope.power_iterations,
                stats.delta_energy,
                workers,
            )?)
        } else {
            None
        };
        reports.push(build_tensor_report(
            &p.spec,
            p.dtype,
            p.desc.shape.clone(),
            p.element_count,
            p.byte_size,
            stats,
            svd,
        ));
    }

    // Phase 3: Huge vocabulary matrices via chunked streaming + single-pass two-sided sketch.
    for (pos, &idx) in streaming_indices.iter().enumerate() {
        if Instant::now() >= deadline {
            budget_exceeded = true;
            for &rem in &streaming_indices[pos..] {
                skipped_tensors.push(prepared[rem].spec.gguf_name.clone());
            }
            break;
        }
        let p = &prepared[idx];
        let b_desc = base_src.descriptor(&p.base_shard, &p.spec.hf_name)?;
        let rows = p.desc.shape[0];
        let cols = p.desc.shape[1];
        let row_bytes = u64::try_from(cols * p.dtype.byte_width()).map_err(|e| e.to_string())?;
        let mut sketch = if request.scope.include_svd {
            Some(StreamingTwoSidedSketch::new(
                &p.spec.hf_name,
                rows,
                cols,
                k_sketch,
            )?)
        } else {
            None
        };
        let mut total_stats = TensorDiffStats {
            base_energy: 0.0,
            cand_energy: 0.0,
            delta_energy: 0.0,
            sum_abs_delta: 0.0,
            max_abs_delta: 0.0,
            nonzero_count: 0,
            changed_rows: Some(0),
        };
        let mut aborted_mid_tensor = false;
        let mut row_start = 0_usize;
        while row_start < rows {
            if Instant::now() >= deadline {
                budget_exceeded = true;
                aborted_mid_tensor = true;
                break;
            }
            crate::startup_cancel::checkpoint()?;
            let chunk_rows = (rows - row_start).min(VOCAB_CHUNK_ROWS);
            let byte_offset = u64::try_from(row_start).map_err(|e| e.to_string())? * row_bytes;
            let chunk_bytes = u64::try_from(chunk_rows).map_err(|e| e.to_string())? * row_bytes;
            let (cb, bb) = match fetch_pair_slices(
                &cand_src,
                &p.cand_shard,
                p.desc.abs_begin + byte_offset,
                chunk_bytes,
                &base_src,
                &p.base_shard,
                b_desc.abs_begin + byte_offset,
                chunk_bytes,
                deadline,
            ) {
                Ok(pair) => pair,
                Err(_) if Instant::now() >= deadline => {
                    budget_exceeded = true;
                    aborted_mid_tensor = true;
                    break;
                }
                Err(err) => return Err(err),
            };
            let mut chunk_delta = Vec::new();
            chunk_delta
                .try_reserve_exact(chunk_rows * cols)
                .map_err(|e| format!("Cannot allocate vocab chunk buffer: {e}"))?;
            chunk_delta.resize(chunk_rows * cols, 0.0_f32);
            let chunk_stats = diff_chunk_rows(
                p.dtype,
                &cb,
                &bb,
                chunk_rows,
                cols,
                &mut chunk_delta,
                workers,
            )?;
            drop(cb);
            drop(bb);
            total_stats.base_energy += chunk_stats.base_energy;
            total_stats.cand_energy += chunk_stats.cand_energy;
            total_stats.delta_energy += chunk_stats.delta_energy;
            total_stats.sum_abs_delta += chunk_stats.sum_abs_delta;
            total_stats.max_abs_delta = total_stats.max_abs_delta.max(chunk_stats.max_abs_delta);
            total_stats.nonzero_count += chunk_stats.nonzero_count;
            total_stats.changed_rows =
                Some(total_stats.changed_rows.unwrap_or(0) + chunk_stats.changed_rows.unwrap_or(0));
            if let Some(sk) = sketch.as_mut() {
                sk.update_chunk(row_start, &chunk_delta, chunk_rows, workers)?;
            }
            row_start += chunk_rows;
        }
        if aborted_mid_tensor {
            for &rem in &streaming_indices[pos..] {
                skipped_tensors.push(prepared[rem].spec.gguf_name.clone());
            }
            break;
        }
        let svd = match sketch {
            Some(sk) => Some(sk.finish(total_stats.delta_energy, workers)?),
            None => None,
        };
        reports.push(build_tensor_report(
            &p.spec,
            p.dtype,
            p.desc.shape.clone(),
            p.element_count,
            p.byte_size,
            total_stats,
            svd,
        ));
    }

    let power_after = crate::host_cpu::power_snapshot();
    let power_conditions = bonsai_cpu_policy::power::compare_endpoints(&power_before, &power_after);

    let mut all_audited = MetricsAccum::default();
    let mut non_vocab_all = MetricsAccum::default();
    let mut pure_non_matrix = MetricsAccum::default();
    let mut non_matrix_small_gdn = MetricsAccum::default();
    let mut linear_matrices = MetricsAccum::default();
    let mut ffn_and_full_attn = MetricsAccum::default();
    let mut cand_a_ffn_down_only = MetricsAccum::default();
    let mut cand_a_down_attn_o = MetricsAccum::default();
    let mut cand_b_all = MetricsAccum::default();
    let mut cand_b_gate_up_qkv = MetricsAccum::default();
    let mut gdn_linear = MetricsAccum::default();
    let mut vocab_matrices = MetricsAccum::default();
    let mut by_family: BTreeMap<String, (String, MetricsAccum)> = BTreeMap::new();

    for report in &reports {
        all_audited.add(report);
        let fam_entry = by_family
            .entry(report.family.clone())
            .or_insert_with(|| (report.category.clone(), MetricsAccum::default()));
        fam_entry.1.add(report);

        match report.category.as_str() {
            "vocab-matrix" => vocab_matrices.add(report),
            "non-matrix" => {
                non_vocab_all.add(report);
                pure_non_matrix.add(report);
                non_matrix_small_gdn.add(report);
            }
            "gdn-small-matrix" => {
                non_vocab_all.add(report);
                non_matrix_small_gdn.add(report);
            }
            "linear-matrix" => {
                non_vocab_all.add(report);
                linear_matrices.add(report);
                match report.family.as_str() {
                    "ffn_down" => {
                        ffn_and_full_attn.add(report);
                        cand_a_ffn_down_only.add(report);
                        cand_a_down_attn_o.add(report);
                        cand_b_all.add(report);
                    }
                    "attn_output" => {
                        ffn_and_full_attn.add(report);
                        cand_a_down_attn_o.add(report);
                        cand_b_all.add(report);
                    }
                    "ffn_gate" | "ffn_up" | "attn_q" | "attn_k" | "attn_v" => {
                        ffn_and_full_attn.add(report);
                        cand_b_all.add(report);
                        cand_b_gate_up_qkv.add(report);
                    }
                    "attn_qkv" | "attn_gate" | "ssm_out" => {
                        gdn_linear.add(report);
                        if report.family != "ssm_out" {
                            cand_b_gate_up_qkv.add(report);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let audited_energy = all_audited.total_delta_energy;
    let non_vocab_energy = non_vocab_all.total_delta_energy;
    let linear_energy = linear_matrices.total_delta_energy;

    let mut family_json = serde_json::Map::new();
    for (family, (category, accum)) in &by_family {
        let mut entry = accum.to_json(audited_energy, non_vocab_energy, linear_energy);
        entry["category"] = json!(category);
        family_json.insert(family.clone(), entry);
    }

    let completed = !budget_exceeded && skipped_tensors.is_empty();
    let gate1_screen = evaluate_gate1(
        completed,
        request.scope.include_svd,
        skipped_tensors.len(),
        &all_audited,
        &non_vocab_all,
        &pure_non_matrix,
        &non_matrix_small_gdn,
        &linear_matrices,
        &ffn_and_full_attn,
        &cand_a_down_attn_o,
        &cand_b_gate_up_qkv,
    );

    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let cand_bytes_read = cand_src.transferred_bytes();
    let base_bytes_read = base_src.transferred_bytes();
    let receipt = json!({
        "schemaVersion": 1,
        "operation": "audit",
        "completed": completed,
        "budgetExceeded": budget_exceeded,
        "elapsedMilliseconds": elapsed_ms,
        "source": source_summary(&request.source),
        "scope": request.scope,
        "controls": request.controls,
        "candidateIndexSha256": cand_idx.sha256,
        "baseIndexSha256": base_idx.sha256,
        "transferredBytes": {
            "candidate": cand_bytes_read,
            "base": base_bytes_read,
            "total": cand_bytes_read.saturating_add(base_bytes_read)
        },
        "powerBefore": power_before,
        "powerAfter": power_after,
        "powerConditions": power_conditions,
        "auditedTensorCount": reports.len(),
        "skippedTensors": skipped_tensors,
        "byFamily": family_json,
        "aggregates": {
            "allAudited": all_audited.to_json(audited_energy, non_vocab_energy, linear_energy),
            "nonVocabAll": non_vocab_all.to_json(audited_energy, non_vocab_energy, linear_energy),
            "pureNonMatrix": pure_non_matrix.to_json(audited_energy, non_vocab_energy, linear_energy),
            "nonMatrixAndSmallGdn": non_matrix_small_gdn.to_json(audited_energy, non_vocab_energy, linear_energy),
            "linearMatrices": linear_matrices.to_json(audited_energy, non_vocab_energy, linear_energy),
            "ffnAndFullAttnLinearMatrices": ffn_and_full_attn.to_json(audited_energy, non_vocab_energy, linear_energy),
            "candidateA_ffnDownOnly": cand_a_ffn_down_only.to_json(audited_energy, non_vocab_energy, linear_energy),
            "candidateA_ffnDownAndAttnOutput": cand_a_down_attn_o.to_json(audited_energy, non_vocab_energy, linear_energy),
            "candidateB_allFfnAndFullAttn": cand_b_all.to_json(audited_energy, non_vocab_energy, linear_energy),
            "candidateB_gateUpQkvOnly": cand_b_gate_up_qkv.to_json(audited_energy, non_vocab_energy, linear_energy),
            "gdnLinearMatrices": gdn_linear.to_json(audited_energy, non_vocab_energy, linear_energy),
            "vocabMatrices": vocab_matrices.to_json(audited_energy, non_vocab_energy, linear_energy),
        },
        "gate1Screen": gate1_screen,
        "tensors": reports,
    });
    if let Some(output) = &request.output_path {
        crate::bounded_input::write_pretty_json_new(output, &receipt)?;
    }
    Ok(receipt)
}

/// Runs a read-only BF16 weight-diff and SVD spectrum audit (`DR1` Gate 1).
///
/// # Errors
/// Rejects unsupported schema versions, invalid scopes or budgets, unsafe or existing
/// output paths, mismatched safetensors layouts, nonfinite weights, or failed range I/O.
pub fn run(request: Value) -> crate::Result<Value> {
    let parsed: Request = serde_json::from_value(request)
        .map_err(|e| format!("Invalid lora-diff-audit request: {e}"))?;
    validate_request(&parsed)?;
    match parsed.operation {
        Operation::Plan => plan_response(&parsed),
        Operation::Audit => run_audit(&parsed),
    }
}
