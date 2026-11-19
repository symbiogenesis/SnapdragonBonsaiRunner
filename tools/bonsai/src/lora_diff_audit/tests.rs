use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write as _;
use std::path::Path;

use serde_json::{Value, json};

use super::safetensors::{TensorCategory, swift_tensor_inventory};
use super::svd::{StreamingTwoSidedSketch, exact_gram_svd, randomized_subspace_svd};
use super::{DAISYROSE_REVISION, QWEN38_REVISION, run};
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

    let exact = exact_gram_svd(&rank4, rows, cols, total_energy)?;
    assert_eq!(exact.method, "exact-gram-jacobi-f64");
    assert!(exact.cumulative_energy.r4 > 0.999);

    let rand1 = randomized_subspace_svd("test-rank4", &rank4, rows, cols, 24, 2, total_energy, 2)?;
    let rand1_repeat =
        randomized_subspace_svd("test-rank4", &rank4, rows, cols, 24, 2, total_energy, 2)?;
    let rand2 = randomized_subspace_svd("test-rank4", &rank4, rows, cols, 24, 2, total_energy, 4)?;
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
    let streamed = sketch.finish(total_energy, 2)?;
    assert!(streamed.cumulative_energy.r4 > 0.99);

    let flat = synthetic_flat_spectrum_matrix(128, 128, 0.25);
    let flat_energy = energy_of(&flat);
    let flat_spec = randomized_subspace_svd("test-flat", &flat, 128, 128, 40, 2, flat_energy, 2)?;
    assert!((flat_spec.cumulative_energy.r32 - 0.25).abs() < 1e-4);

    let zeros = vec![0.0_f32; 32 * 32];
    let zero_spec = randomized_subspace_svd("test-zero", &zeros, 32, 32, 16, 2, 0.0, 2)?;
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
