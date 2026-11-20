use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write as _;
use std::path::Path;

use serde_json::{Value, json};

use super::gguf_adapter::{
    ExportDtype, ExportTensorPair, build_balanced_factors, encode_f16_le, write_lora_gguf,
};
use super::safetensors::{FloatDtype, TensorCategory, decode_scalar_f32, swift_tensor_inventory};
use super::svd::{
    StreamingTwoSidedSketch, exact_gram_svd, gavish_donoho_frobenius_shrink,
    jacobi_eigendecomp_f64, randomized_subspace_denoised_factors, randomized_subspace_svd,
};
use super::{DAISYROSE_REVISION, QWEN38_REVISION, Request, run, run_export_denoised_with_dims};
use crate::bounded_input::ScratchDir;

#[expect(
    clippy::as_conversions,
    reason = "Deterministic test fixture narrows f64 synthetic elements to f32 before BF16 quantization."
)]
#[inline]
const fn f64_to_f32(v: f64) -> f32 {
    v as f32
}

#[expect(
    clippy::as_conversions,
    reason = "Deterministic test fixture converts row/col/index integers to f64."
)]
#[inline]
const fn usize_to_f64(v: usize) -> f64 {
    v as f64
}

fn f32_to_bf16_le(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    // Round to nearest even in BF16.
    let rounding = 0x7FFF_u32 + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding);
    let hi = u16::try_from(rounded >> 16).unwrap_or(0);
    hi.to_le_bytes()
}

fn synthetic_low_rank_matrix(rows: usize, cols: usize, rank: usize, seed: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for k in 0..rank {
        let scale = 1.0 / usize_to_f64(k + 1);
        for r in 0..rows {
            let u = ((usize_to_f64((r + 1) * (k + 3) + seed) * 0.713).sin()) * scale;
            for c in 0..cols {
                let v = ((usize_to_f64((c + 1) * (k + 5) + seed * 7) * 0.371).cos()) * scale;
                out[r * cols + c] += f64_to_f32(u * v);
            }
        }
    }
    out
}

fn synthetic_flat_spectrum_matrix(rows: usize, cols: usize, value: f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for i in 0..rows.min(cols) {
        out[i * cols + i] = value;
    }
    out
}

fn energy_of(values: &[f32]) -> f64 {
    values.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
}

#[test]
fn svd_exact_and_randomized_recover_known_low_rank_spectra() -> crate::Result<()> {
    let rows = 96;
    let cols = 80;
    let rank4 = synthetic_low_rank_matrix(rows, cols, 4, 11);
    let total_energy = energy_of(&rank4);

    let exact = exact_gram_svd("test-exact", &rank4, rows, cols, total_energy, true, 32, 2)?;
    assert_eq!(exact.method, "exact-gram-jacobi-f64");
    assert!(exact.cumulative_energy.r4 > 0.999);

    let rand1 = randomized_subspace_svd(
        "test-rank4",
        &rank4,
        rows,
        cols,
        24,
        2,
        total_energy,
        true,
        16,
        2,
    )?;
    let rand1_repeat = randomized_subspace_svd(
        "test-rank4",
        &rank4,
        rows,
        cols,
        24,
        2,
        total_energy,
        true,
        16,
        2,
    )?;
    let rand2 = randomized_subspace_svd(
        "test-rank4",
        &rank4,
        rows,
        cols,
        24,
        2,
        total_energy,
        true,
        16,
        4,
    )?;
    assert!((rand1.cumulative_energy.r4 - exact.cumulative_energy.r4).abs() < 1e-4);
    assert!((rand1.cumulative_energy.r1 - exact.cumulative_energy.r1).abs() < 1e-4);
    assert_eq!(
        rand1.top_singular_values[0].to_bits(),
        rand1_repeat.top_singular_values[0].to_bits()
    );
    assert!((rand1.top_singular_values[0] - rand2.top_singular_values[0]).abs() < 1e-12);

    let mut sketch = StreamingTwoSidedSketch::new("test-stream-rank4", rows, cols, 24)?;
    sketch.update_chunk(0, &rank4[..48 * cols], 48, 2)?;
    sketch.update_chunk(48, &rank4[48 * cols..], 48, 2)?;
    let streamed = sketch.finish(total_energy, true, 16, 2)?;
    assert!(streamed.cumulative_energy.r4 > 0.99);

    let flat = synthetic_flat_spectrum_matrix(128, 128, 0.25);
    let flat_energy = energy_of(&flat);
    let flat_spec = randomized_subspace_svd(
        "test-flat",
        &flat,
        128,
        128,
        40,
        2,
        flat_energy,
        true,
        32,
        2,
    )?;
    assert!((flat_spec.cumulative_energy.r32 - 0.25).abs() < 1e-4);

    let zeros = vec![0.0_f32; 32 * 32];
    let zero_spec = randomized_subspace_svd("test-zero", &zeros, 32, 32, 16, 2, 0.0, true, 16, 2)?;
    assert_eq!(zero_spec.method, "zero-delta-skipped");
    assert_eq!(zero_spec.cumulative_energy.r32, 0.0);
    Ok(())
}

#[test]
fn hf_to_gguf_mapping_covers_all_851_swift_tensors_bijectively() -> crate::Result<()> {
    let inventory = swift_tensor_inventory();
    assert_eq!(inventory.len(), 851);
    let mut hf_seen = BTreeSet::new();
    let mut gguf_seen = BTreeSet::new();
    let mut full_attn_layers = BTreeSet::new();
    let mut gdn_layers = BTreeSet::new();
    for spec in &inventory {
        assert!(hf_seen.insert(spec.hf_name.as_str()));
        assert!(gguf_seen.insert(spec.gguf_name.as_str()));
        if spec.family == "attn_output" {
            let layer = spec.layer.ok_or("Missing layer index on attn_output.")?;
            assert_eq!((layer + 1) % 4, 0);
            full_attn_layers.insert(layer);
        }
        if spec.family == "ssm_out" {
            let layer = spec.layer.ok_or("Missing layer index on ssm_out.")?;
            assert_ne!((layer + 1) % 4, 0);
            gdn_layers.insert(layer);
        }
    }
    assert_eq!(full_attn_layers.len(), 16);
    assert_eq!(gdn_layers.len(), 48);
    assert!(full_attn_layers.is_disjoint(&gdn_layers));
    assert_eq!(
        inventory
            .iter()
            .filter(|s| s.category == TensorCategory::VocabMatrix)
            .count(),
        2
    );
    Ok(())
}

fn write_safetensors_dir(
    dir: &Path,
    shards: &[(&str, Vec<(&str, Vec<usize>, Vec<u8>)>)],
) -> crate::Result<()> {
    let mut weight_map = BTreeMap::new();
    for (shard_name, tensors) in shards {
        let mut header_map = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, shape, bytes) in tensors {
            let begin = u64::try_from(payload.len()).map_err(|e| e.to_string())?;
            payload.extend_from_slice(bytes);
            let end = u64::try_from(payload.len()).map_err(|e| e.to_string())?;
            header_map.insert(
                (*name).to_owned(),
                json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [begin, end],
                }),
            );
            weight_map.insert((*name).to_owned(), (*shard_name).to_owned());
        }
        let header_bytes =
            serde_json::to_vec(&Value::Object(header_map)).map_err(|e| e.to_string())?;
        let header_len = u64::try_from(header_bytes.len()).map_err(|e| e.to_string())?;
        let mut file = File::create(dir.join(shard_name)).map_err(|e| e.to_string())?;
        file.write_all(&header_len.to_le_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(&header_bytes).map_err(|e| e.to_string())?;
        file.write_all(&payload).map_err(|e| e.to_string())?;
    }
    let index = json!({
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    });
    fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn encode_bf16_tensor(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_bf16_le(v));
    }
    out
}

fn build_fixture_pair(
    base_dir: &Path,
    cand_dir: &Path,
    high_rank_linear: bool,
    dominant_norm: bool,
) -> crate::Result<()> {
    let inventory = swift_tensor_inventory();
    let mut base_shard0 = Vec::new();
    let mut cand_shard0 = Vec::new();
    let mut base_shard1 = Vec::new();
    let mut cand_shard1 = Vec::new();

    for spec in inventory {
        let include = spec.layer.is_none_or(|l| l == 0 || l == 3);
        if !include {
            continue;
        }
        let shape: Vec<usize> = match spec.category {
            TensorCategory::NonMatrix => vec![64],
            TensorCategory::GdnSmallMatrix => vec![8, 64],
            TensorCategory::LinearMatrix => {
                if spec.family == "ffn_down"
                    || spec.family == "attn_output"
                    || spec.family == "ssm_out"
                {
                    vec![128, 160]
                } else {
                    vec![160, 128]
                }
            }
            TensorCategory::VocabMatrix => vec![128, 64],
        };
        let count: usize = shape.iter().product();
        let base_vals = vec![1.0_f32; count];
        let delta_vals = match spec.category {
            TensorCategory::NonMatrix => {
                if dominant_norm {
                    vec![2.0_f32; count]
                } else {
                    vec![0.001_f32; count]
                }
            }
            TensorCategory::GdnSmallMatrix => synthetic_low_rank_matrix(shape[0], shape[1], 2, 3)
                .into_iter()
                .map(|v| v * 0.05)
                .collect(),
            TensorCategory::LinearMatrix => {
                if high_rank_linear {
                    synthetic_flat_spectrum_matrix(shape[0], shape[1], 1.0)
                } else if spec.family == "ffn_down" || spec.family == "attn_output" {
                    synthetic_low_rank_matrix(shape[0], shape[1], 2, 13)
                } else {
                    synthetic_low_rank_matrix(shape[0], shape[1], 2, 19)
                        .into_iter()
                        .map(|v| v * 0.1)
                        .collect()
                }
            }
            TensorCategory::VocabMatrix => synthetic_low_rank_matrix(shape[0], shape[1], 2, 23)
                .into_iter()
                .map(|v| v * 0.05)
                .collect(),
        };
        let cand_vals: Vec<f32> = base_vals
            .iter()
            .zip(&delta_vals)
            .map(|(&b, &d)| b + d)
            .collect();
        let b_bytes = encode_bf16_tensor(&base_vals);
        let c_bytes = encode_bf16_tensor(&cand_vals);
        if spec.layer == Some(3) {
            base_shard1.push((spec.hf_name.clone(), shape.clone(), b_bytes));
            cand_shard1.push((spec.hf_name, shape, c_bytes));
        } else {
            base_shard0.push((spec.hf_name.clone(), shape.clone(), b_bytes));
            cand_shard0.push((spec.hf_name, shape, c_bytes));
        }
    }

    fn to_refs(items: &[(String, Vec<usize>, Vec<u8>)]) -> Vec<(&str, Vec<usize>, Vec<u8>)> {
        items
            .iter()
            .map(|(n, s, b)| (n.as_str(), s.clone(), b.clone()))
            .collect()
    }
    write_safetensors_dir(
        base_dir,
        &[
            ("model-00001-of-00002.safetensors", to_refs(&base_shard0)),
            ("model-00002-of-00002.safetensors", to_refs(&base_shard1)),
        ],
    )?;
    write_safetensors_dir(
        cand_dir,
        &[
            ("model-00001-of-00002.safetensors", to_refs(&cand_shard0)),
            ("model-00002-of-00002.safetensors", to_refs(&cand_shard1)),
        ],
    )
}

#[test]
fn local_directory_audit_passes_low_rank_and_rejects_high_rank_or_norm_dominant_fixtures()
-> crate::Result<()> {
    let scratch = ScratchDir::new("Bonsai-LoraDiffAudit")?;
    let base_pass = scratch.join("base_pass");
    let cand_pass = scratch.join("cand_pass");
    fs::create_dir(&base_pass).map_err(|e| e.to_string())?;
    fs::create_dir(&cand_pass).map_err(|e| e.to_string())?;
    build_fixture_pair(&base_pass, &cand_pass, false, false)?;

    let out_file = scratch.join("pass-receipt.json");
    let pass_report = run(json!({
        "schemaVersion": 1,
        "operation": "audit",
        "source": {
            "kind": "local-directories",
            "candidateDirectory": cand_pass,
            "baseDirectory": base_pass,
        },
        "scope": {
            "layers": [0, 3],
            "includeOutputNorm": true,
            "includeVocab": true,
            "includeSvd": true,
            "maxRank": 32,
            "oversampling": 8,
            "powerIterations": 2
        },
        "controls": {
            "timeBudgetSeconds": 30,
            "workers": 2,
            "httpConcurrency": 1
        },
        "outputPath": out_file,
    }))?;
    assert_eq!(pass_report["completed"], true);
    assert_eq!(pass_report["gate1Screen"]["passed"], true);
    assert_eq!(pass_report["gate1Screen"]["verdict"], "pass-candidate-a");
    assert!(out_file.is_file());
    // Refuses overwriting existing outputPath.
    assert!(
        run(json!({
            "schemaVersion": 1,
            "operation": "plan",
            "source": { "kind": "pinned-daisyrose" },
            "outputPath": out_file,
        }))
        .is_err()
    );

    let base_hr = scratch.join("base_hr");
    let cand_hr = scratch.join("cand_hr");
    fs::create_dir(&base_hr).map_err(|e| e.to_string())?;
    fs::create_dir(&cand_hr).map_err(|e| e.to_string())?;
    build_fixture_pair(&base_hr, &cand_hr, true, false)?;
    let hr_report = run(json!({
        "schemaVersion": 1,
        "operation": "audit",
        "source": {
            "kind": "local-directories",
            "candidateDirectory": cand_hr,
            "baseDirectory": base_hr,
        },
        "scope": {
            "layers": [0, 3],
            "includeOutputNorm": true,
            "includeVocab": false,
            "includeSvd": true,
            "maxRank": 32,
            "oversampling": 8,
            "powerIterations": 2
        }
    }))?;
    assert_eq!(hr_report["gate1Screen"]["passed"], false);
    assert_eq!(hr_report["gate1Screen"]["verdict"], "no-go-high-rank");

    let base_norm = scratch.join("base_norm");
    let cand_norm = scratch.join("cand_norm");
    fs::create_dir(&base_norm).map_err(|e| e.to_string())?;
    fs::create_dir(&cand_norm).map_err(|e| e.to_string())?;
    build_fixture_pair(&base_norm, &cand_norm, false, true)?;
    let norm_report = run(json!({
        "schemaVersion": 1,
        "operation": "audit",
        "source": {
            "kind": "local-directories",
            "candidateDirectory": cand_norm,
            "baseDirectory": base_norm,
        },
        "scope": {
            "layers": [0, 3],
            "includeOutputNorm": true,
            "includeVocab": false,
            "includeSvd": true,
            "maxRank": 32,
            "oversampling": 8,
            "powerIterations": 2
        }
    }))?;
    assert_eq!(norm_report["gate1Screen"]["passed"], false);
    assert_eq!(
        norm_report["gate1Screen"]["verdict"],
        "no-go-non-matrix-dominant"
    );

    // Nonfinite BF16 (0x7FC0 NaN) must be rejected.
    let nan_dir = scratch.join("cand_nan");
    fs::create_dir(&nan_dir).map_err(|e| e.to_string())?;
    write_safetensors_dir(
        &nan_dir,
        &[(
            "model-00001-of-00001.safetensors",
            vec![(
                "model.language_model.norm.weight",
                vec![4],
                vec![0xc0, 0x7f, 0, 0, 0, 0, 0, 0],
            )],
        )],
    )?;
    let nan_base = scratch.join("base_nan");
    fs::create_dir(&nan_base).map_err(|e| e.to_string())?;
    write_safetensors_dir(
        &nan_base,
        &[(
            "model-00001-of-00001.safetensors",
            vec![(
                "model.language_model.norm.weight",
                vec![4],
                vec![0, 0, 0, 0, 0, 0, 0, 0],
            )],
        )],
    )?;
    assert!(
        run(json!({
            "schemaVersion": 1,
            "operation": "audit",
            "source": {
                "kind": "local-directories",
                "candidateDirectory": nan_dir,
                "baseDirectory": nan_base,
            },
            "scope": {
                "layers": [],
                "includeOutputNorm": true,
                "includeVocab": false,
                "includeSvd": false,
            }
        }))
        .is_err()
    );

    Ok(())
}

#[test]
fn plan_operation_reports_pinned_revisions_without_network() -> crate::Result<()> {
    let plan = run(json!({
        "schemaVersion": 1,
        "operation": "plan",
        "source": { "kind": "pinned-daisyrose" }
    }))?;
    assert_eq!(plan["status"], "planned-no-io");
    assert_eq!(plan["source"]["candidate"]["revision"], DAISYROSE_REVISION);
    assert_eq!(plan["source"]["base"]["revision"], QWEN38_REVISION);
    assert_eq!(plan["scope"]["layers"], json!([0, 3, 31, 63]));
    assert_eq!(plan["selectedTensorCount"], 50);
    Ok(())
}

fn rademacher_noise_matrix(rows: usize, cols: usize, amp: f32, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    let mut state = seed;
    for _ in 0..(rows * cols) {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        out.push(if (z & 1) == 1 { amp } else { -amp });
    }
    out
}

fn synthetic_orthonormal_rank_matrix(
    rows: usize,
    cols: usize,
    sigmas: &[f64],
    seed: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for (k, &sigma) in sigmas.iter().enumerate() {
        let mut u = vec![0.0_f64; rows];
        for (r, dst) in u.iter_mut().enumerate() {
            *dst = (usize_to_f64((r + 1) * (k + 3) + seed) * 0.713).sin();
        }
        let u_norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        for val in &mut u {
            *val /= u_norm;
        }
        let mut v = vec![0.0_f64; cols];
        for (c, dst) in v.iter_mut().enumerate() {
            *dst = (usize_to_f64((c + 1) * (k + 5) + seed * 7) * 0.371).cos();
        }
        let v_norm: f64 = v.iter().map(|val| val * val).sum::<f64>().sqrt();
        for val in &mut v {
            *val /= v_norm;
        }
        for r in 0..rows {
            for c in 0..cols {
                out[r * cols + c] += f64_to_f32(sigma * u[r] * v[c]);
            }
        }
    }
    out
}

fn frobenius_diff_sq(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum()
}

fn read_gguf_f32_or_f16_tensors(
    path: &Path,
) -> crate::Result<(f32, BTreeMap<String, (Vec<usize>, Vec<f32>)>)> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let mut pos = 0_usize;
    let take = |p: &mut usize, n: usize| -> crate::Result<&[u8]> {
        let end = p.checked_add(n).ok_or("GGUF read overflow.")?;
        let slice = bytes.get(*p..end).ok_or("Truncated GGUF test file.")?;
        *p = end;
        Ok(slice)
    };
    let read_u32 = |p: &mut usize| -> crate::Result<u32> {
        Ok(u32::from_le_bytes(
            take(p, 4)?.try_into().map_err(|_| "Bad U32.")?,
        ))
    };
    let read_u64 = |p: &mut usize| -> crate::Result<u64> {
        Ok(u64::from_le_bytes(
            take(p, 8)?.try_into().map_err(|_| "Bad U64.")?,
        ))
    };
    let read_str = |p: &mut usize| -> crate::Result<String> {
        let len = usize::try_from(read_u64(p)?).map_err(|e| e.to_string())?;
        String::from_utf8(take(p, len)?.to_vec()).map_err(|e| e.to_string())
    };
    assert_eq!(take(&mut pos, 4)?, b"GGUF");
    assert_eq!(read_u32(&mut pos)?, 3);
    let tensor_count = usize::try_from(read_u64(&mut pos)?).map_err(|e| e.to_string())?;
    let meta_count = usize::try_from(read_u64(&mut pos)?).map_err(|e| e.to_string())?;
    let mut alpha = 0.0_f32;
    for _ in 0..meta_count {
        let key = read_str(&mut pos)?;
        let kind = read_u32(&mut pos)?;
        match kind {
            4 => {
                let _ = read_u32(&mut pos)?;
            }
            6 => {
                let val =
                    f32::from_le_bytes(take(&mut pos, 4)?.try_into().map_err(|_| "Bad F32.")?);
                if key == "adapter.lora.alpha" {
                    alpha = val;
                }
            }
            8 => {
                let _ = read_str(&mut pos)?;
            }
            _ => return Err(format!("Unexpected metadata kind {kind} in test reader.")),
        }
    }
    let mut descs = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_str(&mut pos)?;
        let ndim = read_u32(&mut pos)?;
        assert_eq!(ndim, 2);
        let d0 = usize::try_from(read_u64(&mut pos)?).map_err(|e| e.to_string())?;
        let d1 = usize::try_from(read_u64(&mut pos)?).map_err(|e| e.to_string())?;
        let dtype_id = read_u32(&mut pos)?;
        let offset = usize::try_from(read_u64(&mut pos)?).map_err(|e| e.to_string())?;
        descs.push((name, vec![d0, d1], dtype_id, offset));
    }
    let data_start = pos.checked_next_multiple_of(32).ok_or("Align overflow.")?;
    let mut out = BTreeMap::new();
    for (name, dims, dtype_id, offset) in descs {
        let count = dims[0] * dims[1];
        let begin = data_start + offset;
        let vals = match dtype_id {
            0 => (0..count)
                .map(|i| {
                    let s = begin + i * 4;
                    Ok(f32::from_le_bytes(
                        bytes[s..s + 4].try_into().map_err(|_| "Bad F32 elem.")?,
                    ))
                })
                .collect::<crate::Result<Vec<f32>>>()?,
            1 => (0..count)
                .map(|i| {
                    let s = begin + i * 2;
                    decode_scalar_f32(FloatDtype::F16, &bytes[s..s + 2])
                })
                .collect::<crate::Result<Vec<f32>>>()?,
            other => return Err(format!("Unexpected tensor dtype {other}.")),
        };
        out.insert(name, (dims, vals));
    }
    Ok((alpha, out))
}

#[test]
fn jacobi_eigendecomp_and_f16_encoding_match_exact_identities() -> crate::Result<()> {
    let n = 8_usize;
    let a = synthetic_low_rank_matrix(n, n, 4, 29);
    let mut gram = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            gram[i * n + j] = (0..n)
                .map(|k| f64::from(a[k * n + i]) * f64::from(a[k * n + j]))
                .sum();
        }
    }
    let orig = gram.clone();
    let (evals, evecs) = jacobi_eigendecomp_f64(&mut gram, n)?;
    assert_eq!(evals.len(), n);
    for w in evals.windows(2) {
        assert!(w[0] >= w[1]);
    }
    for (j, &lambda) in evals.iter().enumerate() {
        let v = &evecs[j];
        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-11);
        for (k, other) in evecs.iter().enumerate() {
            if k != j {
                let dot: f64 = v.iter().zip(other).map(|(x, y)| x * y).sum();
                assert!(dot.abs() < 1e-11);
            }
        }
        for r in 0..n {
            let av_r: f64 = (0..n).map(|c| orig[r * n + c] * v[c]).sum();
            assert!((av_r - lambda * v[r]).abs() < 1e-11);
        }
    }

    for &val in &[0.0_f32, -0.0_f32, 1.0, -3.5, 65_504.0] {
        let bytes = encode_f16_le(val)?;
        let decoded = decode_scalar_f32(FloatDtype::F16, &bytes)?;
        assert_eq!(decoded.to_bits(), val.to_bits());
    }
    // 2^-25 exact tie rounds to even (0), slightly larger rounds to smallest subnormal (2^-24).
    let tie = 2.0_f32.powi(-25);
    assert_eq!(
        decode_scalar_f32(FloatDtype::F16, &encode_f16_le(tie)?)?,
        0.0
    );
    assert_eq!(
        decode_scalar_f32(FloatDtype::F16, &encode_f16_le(tie * 1.1)?)?,
        2.0_f32.powi(-24)
    );
    for &bad in &[65_520.0_f32, -65_520.0_f32, f32::NAN, f32::INFINITY] {
        assert!(encode_f16_le(bad).is_err());
    }
    Ok(())
}

#[test]
fn denoising_recovers_spiked_signal_buried_in_isotropic_bulk() -> crate::Result<()> {
    let rows = 256_usize;
    let cols = 320_usize;
    // Clean rank-3 signal (~13% energy) buried under ~87% Rademacher noise so raw E(32) < 50%.
    let signal = synthetic_orthonormal_rank_matrix(rows, cols, &[2.3, 1.7, 1.3], 41);
    let sig_energy = energy_of(&signal);
    let noise_amp = ((sig_energy * 6.5) / usize_to_f64(rows * cols)).sqrt();
    let noise = rademacher_noise_matrix(rows, cols, f64_to_f32(noise_amp), 0xfeed_face);
    let mixed: Vec<f32> = signal.iter().zip(&noise).map(|(&s, &n)| s + n).collect();
    let total_energy = energy_of(&mixed);

    let extracted = randomized_subspace_denoised_factors(
        "test-spiked-bulk",
        &mixed,
        rows,
        cols,
        48,
        2,
        total_energy,
        32,
        1.0,
        2,
    )?;
    assert!(extracted.spectrum.cumulative_energy.r32 < 0.50);
    let den = extracted
        .spectrum
        .denoised
        .as_ref()
        .ok_or("Missing denoised spectrum.")?;
    assert_eq!(den.above_bulk_rank, 3);
    assert!(!den.tail_saturated);
    assert!(den.sigma_max_to_bulk_edge_ratio > 2.5);
    assert_eq!(extracted.rank, 3);

    let mut shrunk_recon = vec![0.0_f32; rows * cols];
    let mut raw_recon = vec![0.0_f32; rows * cols];
    for k in 0..3 {
        let s_shrunk = extracted.shrunk_sigmas[k];
        let s_raw = extracted.spectrum.top_singular_values[k];
        assert!(
            (s_shrunk
                - gavish_donoho_frobenius_shrink(
                    s_raw,
                    den.effective_bulk_edge,
                    den.aspect_ratio_beta
                ))
            .abs()
                < 1e-9
        );
        for r in 0..rows {
            let u = f64::from(extracted.u_cols[k][r]);
            for c in 0..cols {
                let v = f64::from(extracted.v_cols[k][c]);
                shrunk_recon[r * cols + c] += f64_to_f32(s_shrunk * u * v);
                raw_recon[r * cols + c] += f64_to_f32(s_raw * u * v);
            }
        }
    }
    let shrunk_err = frobenius_diff_sq(&shrunk_recon, &signal);
    let raw_err = frobenius_diff_sq(&raw_recon, &signal);
    assert!(shrunk_err < raw_err);
    Ok(())
}

#[test]
fn denoising_preserves_noiseless_low_rank_and_rejects_flat_noise() -> crate::Result<()> {
    let rows = 96_usize;
    let cols = 80_usize;
    let rank4 = synthetic_orthonormal_rank_matrix(rows, cols, &[3.0, 2.0, 1.5, 1.0], 7);
    let e4 = energy_of(&rank4);
    let exact = exact_gram_svd("test-noiseless", &rank4, rows, cols, e4, true, 32, 2)?;
    let den4 = exact
        .denoised
        .as_ref()
        .ok_or("Missing denoised spectrum on noiseless rank-4.")?;
    assert_eq!(den4.above_bulk_rank, 4);
    assert!(!den4.tail_saturated);
    assert!(den4.estimated_bulk_energy_fraction < 0.02);

    let pure_noise = rademacher_noise_matrix(128, 160, 0.01, 0x1234_5678);
    let en = energy_of(&pure_noise);
    let noise_spec = randomized_subspace_svd(
        "test-pure-noise",
        &pure_noise,
        128,
        160,
        40,
        2,
        en,
        true,
        32,
        2,
    )?;
    let den_noise = noise_spec
        .denoised
        .as_ref()
        .ok_or("Missing denoised spectrum on noise matrix.")?;
    assert_eq!(den_noise.above_bulk_rank, 0);
    assert_eq!(den_noise.denoised_signal_energy, 0.0);
    Ok(())
}

#[test]
fn export_denoised_and_verify_adapter_roundtrip_f32_and_f16() -> crate::Result<()> {
    let scratch = ScratchDir::new("Bonsai-ExportDenoised")?;
    let base_dir = scratch.join("base");
    let cand_dir = scratch.join("cand");
    fs::create_dir(&base_dir).map_err(|e| e.to_string())?;
    fs::create_dir(&cand_dir).map_err(|e| e.to_string())?;

    // Synthetic layers [0, 3] restricted to npu-friendly targets with toy shapes (128x160 / 160x128).
    let mut toy_dims: BTreeMap<String, (usize, usize, &'static str)> = BTreeMap::new();
    let mut base_s0 = Vec::new();
    let mut cand_s0 = Vec::new();
    let mut base_s1 = Vec::new();
    let mut cand_s1 = Vec::new();
    let mut expected_targets: BTreeMap<String, (usize, usize, Vec<f32>)> = BTreeMap::new();

    for spec in swift_tensor_inventory() {
        let Some(layer) = spec.layer else {
            continue;
        };
        if (layer != 0 && layer != 3)
            || !matches!(
                spec.family,
                "ffn_down" | "attn_q" | "attn_k" | "attn_v" | "attn_output"
            )
        {
            continue;
        }
        let (rows, cols) = if matches!(spec.family, "ffn_down" | "attn_output") {
            (128_usize, 160_usize)
        } else {
            (160_usize, 128_usize)
        };
        toy_dims.insert(spec.gguf_name.clone(), (cols, rows, spec.family));

        let sigmas: &[f64] = if layer == 0 {
            &[2.0, 1.4]
        } else {
            &[2.2, 1.5, 1.1]
        };
        let signal = synthetic_orthonormal_rank_matrix(
            rows,
            cols,
            sigmas,
            usize::try_from(layer).unwrap_or(0) + 17,
        );
        let sig_e = energy_of(&signal);
        let noise_amp = ((sig_e * 3.0) / usize_to_f64(rows * cols)).sqrt();
        let noise =
            rademacher_noise_matrix(rows, cols, f64_to_f32(noise_amp), u64::from(layer) + 99);
        let delta: Vec<f32> = signal.iter().zip(&noise).map(|(&s, &n)| s + n).collect();
        let base_vals = vec![1.0_f32; rows * cols];
        let cand_vals: Vec<f32> = base_vals.iter().zip(&delta).map(|(&b, &d)| b + d).collect();
        let b_bytes = encode_bf16_tensor(&base_vals);
        let c_bytes = encode_bf16_tensor(&cand_vals);
        // Compute expected shrunk reconstruction directly from the BF16-encoded shard diff.
        let mut bf16_delta = Vec::with_capacity(rows * cols);
        for (c_elem, b_elem) in c_bytes.chunks_exact(2).zip(b_bytes.chunks_exact(2)) {
            let diff = decode_scalar_f32(FloatDtype::Bf16, c_elem)?
                - decode_scalar_f32(FloatDtype::Bf16, b_elem)?;
            bf16_delta.push(diff);
        }
        let factors = randomized_subspace_denoised_factors(
            &spec.hf_name,
            &bf16_delta,
            rows,
            cols,
            24,
            2,
            energy_of(&bf16_delta),
            16,
            1.0,
            2,
        )?;
        let mut target_recon = vec![0.0_f32; rows * cols];
        for k in 0..factors.rank {
            for r in 0..rows {
                for c in 0..cols {
                    target_recon[r * cols + c] += f64_to_f32(
                        factors.shrunk_sigmas[k]
                            * f64::from(factors.u_cols[k][r])
                            * f64::from(factors.v_cols[k][c]),
                    );
                }
            }
        }
        expected_targets.insert(spec.gguf_name.clone(), (rows, cols, target_recon));

        if layer == 0 {
            base_s0.push((spec.hf_name.clone(), vec![rows, cols], b_bytes));
            cand_s0.push((spec.hf_name, vec![rows, cols], c_bytes));
        } else {
            base_s1.push((spec.hf_name.clone(), vec![rows, cols], b_bytes));
            cand_s1.push((spec.hf_name, vec![rows, cols], c_bytes));
        }
    }
    fn to_refs(items: &[(String, Vec<usize>, Vec<u8>)]) -> Vec<(&str, Vec<usize>, Vec<u8>)> {
        items
            .iter()
            .map(|(n, s, b)| (n.as_str(), s.clone(), b.clone()))
            .collect()
    }
    write_safetensors_dir(
        &base_dir,
        &[
            ("model-00001-of-00002.safetensors", to_refs(&base_s0)),
            ("model-00002-of-00002.safetensors", to_refs(&base_s1)),
        ],
    )?;
    write_safetensors_dir(
        &cand_dir,
        &[
            ("model-00001-of-00002.safetensors", to_refs(&cand_s0)),
            ("model-00002-of-00002.safetensors", to_refs(&cand_s1)),
        ],
    )?;

    for (dtype_str, tol) in [("f32", 1e-4_f64), ("f16", 5e-3_f64)] {
        let out_gguf = scratch.join(format!("exported-{dtype_str}.gguf"));
        let req: Request = serde_json::from_value(json!({
            "schemaVersion": 1,
            "operation": "export-denoised",
            "source": {
                "kind": "local-directories",
                "candidateDirectory": cand_dir,
                "baseDirectory": base_dir,
            },
            "scope": {
                "layers": [0, 3],
                "targetEnvelope": "npu-friendly",
                "exportDtype": dtype_str,
                "maxRank": 16,
                "oversampling": 8,
                "powerIterations": 2
            },
            "controls": {
                "timeBudgetSeconds": 60,
                "workers": 2,
                "httpConcurrency": 1
            },
            "outputPath": out_gguf
        }))
        .map_err(|e| e.to_string())?;
        let receipt = run_export_denoised_with_dims(&req, &toy_dims)?;
        assert_eq!(receipt["published"], true);
        assert_eq!(receipt["pairCount"], 6);
        assert_eq!(receipt["verification"]["verified"], true);
        assert_eq!(receipt["verification"]["preservesHexagonFfnChains"], true);
        assert_eq!(receipt["verification"]["excludesGroupedSsmOut"], true);
        assert_eq!(
            receipt["verification"]["publisherEngineAlphaCompatible"],
            true
        );
        assert_eq!(
            receipt["verification"]["prismInferenceAlphaCompatible"],
            true
        );

        let (alpha, tensors) = read_gguf_f32_or_f16_tensors(&out_gguf)?;
        for target in ["blk.0.ffn_down.weight", "blk.3.attn_output.weight"] {
            let (rows, cols, expected) = expected_targets
                .get(target)
                .ok_or("Missing expected reconstruction.")?;
            let (a_dims, a_vals) = tensors
                .get(&format!("{target}.lora_a"))
                .ok_or("Missing exported lora_a.")?;
            let (b_dims, b_vals) = tensors
                .get(&format!("{target}.lora_b"))
                .ok_or("Missing exported lora_b.")?;
            let r = a_dims[1];
            assert_eq!(a_dims[0], *cols);
            assert_eq!(*b_dims, vec![r, *rows]);
            let scale = f64::from(alpha) / usize_to_f64(r);
            let mut actual = vec![0.0_f32; rows * cols];
            for row in 0..*rows {
                for col in 0..*cols {
                    let mut sum = 0.0_f64;
                    for k in 0..r {
                        sum += f64::from(b_vals[row * r + k]) * f64::from(a_vals[k * cols + col]);
                    }
                    actual[row * cols + col] = f64_to_f32(scale * sum);
                }
            }
            let rel_err = (frobenius_diff_sq(&actual, expected) / energy_of(expected)).sqrt();
            assert!(rel_err < tol, "{target} ({dtype_str}) rel_err={rel_err}");
        }
    }

    // Reject targetEnvelope: "all" and missing outputPath on export-denoised.
    assert!(
        run(json!({
            "schemaVersion": 1,
            "operation": "export-denoised",
            "source": { "kind": "local-directories", "candidateDirectory": cand_dir, "baseDirectory": base_dir },
            "scope": { "layers": [0], "targetEnvelope": "all" },
            "outputPath": scratch.join("should-fail.gguf")
        }))
        .is_err()
    );
    assert!(
        run(json!({
            "schemaVersion": 1,
            "operation": "export-denoised",
            "source": { "kind": "local-directories", "candidateDirectory": cand_dir, "baseDirectory": base_dir }
        }))
        .is_err()
    );

    // Public verify-adapter tests on real Swift-Bonsai-2 matrix dimensions (attn_k & attn_v [5120, 1024]).
    let make_real_pair =
        |name: &str, in_f: usize, out_f: usize, rank: usize| -> crate::Result<ExportTensorPair> {
            let u: Vec<Vec<f32>> = (0..rank)
                .map(|k| vec![0.01_f32 * f64_to_f32(usize_to_f64(k + 1)); out_f])
                .collect();
            let v: Vec<Vec<f32>> = (0..rank)
                .map(|k| vec![0.02_f32 * f64_to_f32(usize_to_f64(k + 1)); in_f])
                .collect();
            let sigmas = vec![0.5_f64; rank];
            let (a, b) = build_balanced_factors(&u, &v, &sigmas, 3.0, ExportDtype::F16)?;
            Ok(ExportTensorPair {
                gguf_base_name: name.to_owned(),
                in_features: in_f,
                out_features: out_f,
                rank,
                lora_a_f32: a,
                lora_b_f32: b,
            })
        };

    let real_ok = scratch.join("real-npu-friendly.gguf");
    let written_ok = write_lora_gguf(
        &real_ok,
        ExportDtype::F16,
        3.0,
        &[("bonsai.composition.method", "test")],
        &[
            make_real_pair("blk.3.attn_k.weight", 5_120, 1_024, 2)?,
            make_real_pair("blk.7.attn_v.weight", 5_120, 1_024, 3)?,
        ],
    )?;
    let verify_ok = run(json!({
        "schemaVersion": 1,
        "operation": "verify-adapter",
        "adapterPath": real_ok,
        "verifyPolicy": {
            "expectedSha256": written_ok.sha256,
            "expectedEnvelope": "npu-friendly",
            "requireNpuFriendly": true,
            "requireNoGroupedSsm": true,
            "maxAllowedRank": 3
        }
    }))?;
    assert_eq!(verify_ok["verified"], true);
    assert_eq!(verify_ok["detectedEnvelope"], "npu-friendly");
    assert_eq!(verify_ok["uniformRank"], false);
    assert_eq!(verify_ok["minRank"], 2);
    assert_eq!(verify_ok["maxRank"], 3);
    assert_eq!(verify_ok["preservesHexagonFfnChains"], true);
    assert_eq!(verify_ok["excludesGroupedSsmOut"], true);

    // Wrong expectedSha256 rejects with Err.
    assert!(
        run(json!({
            "schemaVersion": 1,
            "operation": "verify-adapter",
            "adapterPath": real_ok,
            "verifyPolicy": { "expectedSha256": "0".repeat(64) }
        }))
        .is_err()
    );

    // ssm_out fails requireNoGroupedSsm policy.
    let real_ssm = scratch.join("real-ssm.gguf");
    write_lora_gguf(
        &real_ssm,
        ExportDtype::F16,
        1.0,
        &[],
        &[make_real_pair("blk.0.ssm_out.weight", 6_144, 5_120, 1)?],
    )?;
    let verify_ssm = run(json!({
        "schemaVersion": 1,
        "operation": "verify-adapter",
        "adapterPath": real_ssm,
        "verifyPolicy": { "requireNoGroupedSsm": true }
    }))?;
    assert_eq!(verify_ssm["verified"], false);
    assert_eq!(verify_ssm["excludesGroupedSsmOut"], false);

    // ffn_gate fails requireNpuFriendly policy.
    let real_gate = scratch.join("real-gate.gguf");
    write_lora_gguf(
        &real_gate,
        ExportDtype::F16,
        1.0,
        &[],
        &[make_real_pair("blk.0.ffn_gate.weight", 5_120, 17_408, 1)?],
    )?;
    let verify_gate = run(json!({
        "schemaVersion": 1,
        "operation": "verify-adapter",
        "adapterPath": real_gate,
        "verifyPolicy": { "requireNpuFriendly": true }
    }))?;
    assert_eq!(verify_gate["verified"], false);
    assert_eq!(verify_gate["preservesHexagonFfnChains"], false);

    // Tampered NaN payload in F32 GGUF fails with Err.
    let nan_gguf = scratch.join("real-nan.gguf");
    write_lora_gguf(
        &nan_gguf,
        ExportDtype::F32,
        1.0,
        &[],
        &[ExportTensorPair {
            gguf_base_name: "blk.3.attn_k.weight".to_owned(),
            in_features: 5_120,
            out_features: 1_024,
            rank: 1,
            lora_a_f32: vec![0.1_f32; 5_120],
            lora_b_f32: vec![0.1_f32; 1_024],
        }],
    )?;
    let mut raw_nan = fs::read(&nan_gguf).map_err(|e| e.to_string())?;
    let tail_start = raw_nan.len() - 4;
    raw_nan[tail_start..].copy_from_slice(&f32::NAN.to_le_bytes());
    fs::write(&nan_gguf, &raw_nan).map_err(|e| e.to_string())?;
    assert!(
        run(json!({
            "schemaVersion": 1,
            "operation": "verify-adapter",
            "adapterPath": nan_gguf
        }))
        .is_err()
    );

    Ok(())
}
