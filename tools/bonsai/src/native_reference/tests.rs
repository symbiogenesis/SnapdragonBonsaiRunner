//! Focused malformed-format and original-tolerance regression controls.

use super::{compare_bytes, decode, formats, native};

#[test]
fn native_source_manifest_extends_selection_without_relabelling_legacy() -> crate::Result<()> {
    let root = std::env::temp_dir().join(format!(
        "reference-source-{}",
        crate::startup_process::unique()
    ));
    std::fs::create_dir(&root).map_err(|error| error.to_string())?;
    let result = (|| -> crate::Result<()> {
        let repository = root.join("source");
        let evidence = root.join("evidence");
        std::fs::create_dir(&evidence).map_err(|error| error.to_string())?;
        for relative in super::public::ORIGINAL_SOURCES
            .iter()
            .chain(super::public::NATIVE_SOURCES)
        {
            let path = repository.join(relative);
            std::fs::create_dir_all(path.parent().ok_or("missing fixture parent")?)
                .map_err(|error| error.to_string())?;
            std::fs::write(path, format!("{relative}\r\n")).map_err(|error| error.to_string())?;
        }
        let revision = "a".repeat(40);
        let binding = super::public::save_sources(&repository, &evidence, &revision, false)?;
        let manifest =
            super::read_json(&evidence.join("replay-source/source-manifest.json"), 65_536)?;
        assert_eq!(manifest["schemaVersion"], 2);
        let selected = super::audit::sources(&evidence, &binding, &revision)?
            .ok_or("native source selection missing")?;
        assert_eq!(
            selected.len(),
            super::public::ORIGINAL_SOURCES.len() + super::public::NATIVE_SOURCES.len()
        );
        let native = evidence.join("replay-source/tools/bonsai/src/native_reference/native.rs");
        std::fs::write(native, b"changed").map_err(|error| error.to_string())?;
        assert!(super::audit::sources(&evidence, &binding, &revision).is_err());
        Ok(())
    })();
    std::fs::remove_dir_all(&root).map_err(|error| error.to_string())?;
    result
}
use serde_json::json;

#[test]
fn reference_json_keeps_root_zero_depth64_including_unused_config_metadata() -> crate::Result<()> {
    let nested =
        |leaf: &str, levels: usize| format!("{}{leaf}{}", "[".repeat(levels), "]".repeat(levels));
    for leaf in ["null", "{}", "[]", "\"leaf\"", "1", "true"] {
        decode(nested(leaf, 64).as_bytes())?;
        assert_eq!(
            decode(nested(leaf, 65).as_bytes()).err().as_deref(),
            Some("reference JSON nesting exceeds depth 64")
        );
    }
    assert!(decode(nested(r#"{"unused":null}"#, 64).as_bytes()).is_err());
    let config = json!({"schema_version":2,"model_type":"prism_hadamard_qwen35","base_model_type":"qwen3_5","tensor_namespace":"mlx-vlm-qwen3_5","gdn_activation_layout":"grouped","quantization":{"bits":2,"group_size":128,"mode":"affine"},"modules":super::public::blocks().map(|block|json!({"path":format!("model.layers.{block}.linear_attn.out_proj"),"block":1024,"dtype":"float16","embedding":false})).collect::<Vec<_>>()});
    let mut config = serde_json::to_string(&config).map_err(|error| error.to_string())?;
    config.pop();
    // The ignored field is one edge below the root. Its leaf may be at 64,
    // while the same valid configuration with a leaf at 65 must reject.
    let with_metadata =
        |levels| format!("{config},\"unusedMetadata\":{}}}", nested("null", levels));
    assert_eq!(
        super::public::config_sites(with_metadata(63).as_bytes())?
            .as_array()
            .map(Vec::len),
        Some(48)
    );
    assert_eq!(
        super::public::config_sites(with_metadata(64).as_bytes())
            .err()
            .as_deref(),
        Some("reference JSON nesting exceeds depth 64")
    );
    Ok(())
}

#[test]
fn rejects_ambiguous_json_and_every_tensor_extent() -> crate::Result<()> {
    for raw in [r#"{"x":1,"x":2}"#, r#"{"x":1e999}"#, "{}{}"] {
        assert!(decode(raw.as_bytes()).is_err());
    }
    assert!(decode(&[255]).is_err());
    let valid = json!({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}});
    let raw = serde_json::to_vec(&valid).map_err(|error| error.to_string())?;
    let bytes = u64::try_from(raw.len()).map_err(|error| error.to_string())? + 12;
    assert_eq!(
        formats::tensor_header(&raw, bytes)?["tensor"]["absoluteByteEnd"],
        bytes
    );
    for (field, value) in [
        ("dtype", json!("UNKNOWN")),
        ("shape", json!([-1])),
        ("shape", json!([true])),
        ("shape", json!([1_000_000_001])),
        ("data_offsets", json!([0, 5])),
    ] {
        let mut changed = valid.clone();
        changed["tensor"][field] = value;
        assert!(
            formats::tensor_header(
                &serde_json::to_vec(&changed).map_err(|error| error.to_string())?,
                bytes
            )
            .is_err()
        );
    }
    for raw in [
        json!({"a":valid["tensor"],"b":valid["tensor"]}),
        json!({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}),
    ] {
        let bytes = serde_json::to_vec(&raw).map_err(|error| error.to_string())?;
        assert!(
            formats::tensor_header(
                &bytes,
                u64::try_from(bytes.len()).map_err(|error| error.to_string())? + 16
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn keeps_original_tolerance_signed_zero_and_finite_controls() -> crate::Result<()> {
    let component = json!({"elements":2,"offset":0,"stride":2,"code":"f16"});
    let expected = [0, 60, 0, 128];
    let actual = [1.001_f32.to_le_bytes(), (-0.0_f32).to_le_bytes()].concat();
    assert_eq!(
        compare_bytes(&expected, &actual, &component, 0.001, 0.001)?["failedElements"],
        0
    );
    let actual = [1.003_f32.to_le_bytes(), 0.0_f32.to_le_bytes()].concat();
    let result = compare_bytes(&expected, &actual, &component, 0.001, 0.001)?;
    assert_eq!(result["failedElements"], 1);
    assert_eq!(result["byteExact"], false);
    let actual = [f32::NAN.to_le_bytes(), 0.0_f32.to_le_bytes()].concat();
    assert!(compare_bytes(&expected, &actual, &component, 0.001, 0.001).is_err());
    assert!(compare_bytes(&expected, &[0; 4], &component, 0.001, 0.001).is_err());
    for scale in [json!(true), json!(null), json!("0"), json!(-0.0)] {
        assert!(
            native::adapter_condition(
                &json!({"adapterLoaded":true,"adapterScale":scale}),
                "loaded-zero"
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn native_greedy_receipts_bind_retained_logits_and_first_maximum() -> crate::Result<()> {
    let logits = [-4.0_f32, -0.0, 0.0, 2.0, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    native::validate_greedy(&logits, 3)?;
    for claimed in [0, 1, 2, 4, 5, u64::MAX] {
        assert!(native::validate_greedy(&logits, claimed).is_err());
    }
    let zeros = [(-0.0_f32).to_le_bytes(), 0.0_f32.to_le_bytes()].concat();
    native::validate_greedy(&zeros, 0)?;
    assert!(native::validate_greedy(&zeros, 1).is_err());
    for malformed in [&[][..], &logits[..logits.len() - 1]] {
        assert!(native::validate_greedy(malformed, 3).is_err());
    }
    for index in 0..5 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut changed = logits.clone();
            changed[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
            assert!(native::validate_greedy(&changed, 3).is_err());
        }
    }
    Ok(())
}

#[test]
fn native_greedy_receipts_reject_agreeing_stale_tokens_despite_logit_tolerance() -> crate::Result<()>
{
    let native_logits = [1.0005_f32.to_le_bytes(), 1.0_f32.to_le_bytes()].concat();
    let rust_logits = [1.0_f32.to_le_bytes(), 1.0005_f32.to_le_bytes()].concat();
    let component = json!({"elements":2,"offset":0,"stride":4,"code":"f32"});
    assert_eq!(
        compare_bytes(&native_logits, &rust_logits, &component, 0.001, 0.001)?["failedElements"],
        0
    );
    native::validate_greedy(&native_logits, 0)?;
    native::validate_greedy(&rust_logits, 1)?;
    // A shared stale token must not turn different retained maxima into a match.
    for claimed in [0, 1, 2, u64::MAX] {
        assert!(
            native::validate_greedy(&native_logits, claimed).is_err()
                || native::validate_greedy(&rust_logits, claimed).is_err()
        );
    }
    Ok(())
}
