//! Frozen legacy policy parity and fail-closed artifact selection checks.

use super::*;

const FROZEN: &str = include_str!("../tests/fixtures/cpu-policy-v1.json");

fn profile(value: &Value) -> Result<Profile, String> {
    serde_json::from_value(value.clone()).map_err(|error| error.to_string())
}

#[test]
fn frozen_policies_keep_arguments_and_artifacts_with_reviewed_shorter_deadline()
-> Result<(), String> {
    let frozen = parse(FROZEN)?;
    let model = frozen["modelPath"]
        .as_str()
        .ok_or("Missing frozen model path.")?;
    let adapter = frozen["adapterPath"]
        .as_str()
        .ok_or("Missing frozen adapter path.")?;
    let cases = frozen["cases"].as_array().ok_or("Missing frozen cases.")?;
    assert_eq!(cases.len(), 5);
    for case in cases {
        let selected = profile(&case["profile"])?;
        let use_adapter = case["useAbliterateLora"]
            .as_bool()
            .ok_or("Missing adapter flag.")?;
        let use_ptq1 = case["useRepackedPtq1"]
            .as_bool()
            .ok_or("Missing PTQ1 flag.")?;
        let mut expected = case["policy"].clone();
        // Preserve the historical capture; only the owner-requested deadline changed.
        assert_eq!(expected["TimeoutSeconds"], 600);
        expected["TimeoutSeconds"] = json!(180);
        assert_eq!(policy(selected, use_adapter, use_ptq1)?, expected);
        let argv = check_arguments(model, selected, use_adapter, use_adapter.then_some(adapter))?;
        assert_eq!(json!(argv), case["arguments"]);
        let lock = parse(if selected == Profile::ReferenceCpu {
            REFERENCE_LOCK
        } else {
            SWIFT_LOCK
        })?;
        let derived = use_ptq1.then(|| parse(PTQ1_EVIDENCE)).transpose()?;
        assert_eq!(
            select_artifact(&lock, selected, use_ptq1, derived.as_ref())?,
            case["artifact"]
        );
    }
    Ok(())
}

#[test]
fn planning_selects_only_explicit_artifact_paths_and_keeps_null_adapter() -> Result<(), String> {
    let data = std::env::temp_dir().join("Bonsai policy fixture & spaces");
    let frozen = parse(FROZEN)?;
    let cases = frozen["cases"].as_array().ok_or("Missing frozen cases.")?;
    for case in cases {
        let output = run(json!({
            "operation": "plan", "dataDirectory": data,
            "profile": case["profile"],
            "useAbliterateLora": case["useAbliterateLora"],
            "useRepackedPtq1": case["useRepackedPtq1"],
        }))?;
        let mut expected = case["policy"].clone();
        assert_eq!(expected["TimeoutSeconds"], 600);
        expected["TimeoutSeconds"] = json!(180);
        assert_eq!(output["Policy"], expected);
        assert_eq!(output["Artifact"], case["artifact"]);
        let filename = case["artifact"]["Filename"]
            .as_str()
            .ok_or("Missing artifact filename.")?;
        let revision = output["Lock"]["model"]["revision"]
            .as_str()
            .ok_or("Missing source revision.")?;
        let expected_model = data.join("models").join(revision).join(filename);
        assert_eq!(output["ModelPath"], json!(expected_model));
        let mut expected_arguments = case["arguments"].clone();
        expected_arguments[1] = output["ModelPath"].clone();
        if case["useAbliterateLora"] == true {
            let expected_adapter = data
                .join("adapters")
                .join("703028bb7fba03496e796b144839b161cf0f5409")
                .join("bonsai-abliterate-lora.gguf");
            assert_eq!(output["AdapterPath"], json!(expected_adapter));
            assert_eq!(output["AdapterLock"], parse(ADAPTER_LOCK)?);
            let args = expected_arguments
                .as_array_mut()
                .ok_or("Missing frozen arguments.")?;
            let index = args
                .iter()
                .position(|arg| arg == "--lora")
                .ok_or("Missing frozen adapter argument.")?;
            args[index + 1] = output["AdapterPath"].clone();
        } else {
            assert_eq!(output["AdapterPath"], Value::Null);
            assert_eq!(output["AdapterLock"], Value::Null);
        }
        assert_eq!(output["Arguments"], expected_arguments);
        assert_eq!(output.as_object().map(serde_json::Map::len), Some(7));
    }
    Ok(())
}

#[test]
fn malformed_requests_and_swift_only_reference_options_are_rejected() -> Result<(), String> {
    let valid = json!({
        "operation": "plan", "dataDirectory": std::env::temp_dir(),
        "profile": "reference-cpu", "useAbliterateLora": false, "useRepackedPtq1": false,
    });
    for (key, value) in [
        ("operation", json!("launch")),
        ("dataDirectory", json!("relative")),
        ("dataDirectory", json!("C:\\%UNEXPANDED%\\models")),
        ("dataDirectory", json!("\0")),
        ("profile", json!("ReferenceCpu")),
        ("profile", Value::Null),
        ("useAbliterateLora", json!(true)),
        ("useRepackedPtq1", json!(true)),
        ("useAbliterateLora", json!("false")),
        ("useRepackedPtq1", json!(0)),
        ("arguments", json!(["--unsafe"])),
    ] {
        let mut changed = valid.clone();
        changed[key] = value;
        assert!(run(changed).is_err(), "accepted invalid {key}");
    }
    let keys: Vec<String> = valid
        .as_object()
        .ok_or("Invalid test fixture.")?
        .keys()
        .cloned()
        .collect();
    for key in keys {
        let mut missing = valid.clone();
        missing
            .as_object_mut()
            .ok_or("Invalid fixture.")?
            .remove(&key);
        assert!(run(missing).is_err(), "accepted missing {key}");
    }
    for invalid in [json!([]), Value::Null, json!(true), json!("plan")] {
        assert!(run(invalid).is_err());
    }
    assert!(policy(Profile::ReferenceCpu, true, false).is_err());
    assert!(policy(Profile::ReferenceCpu, false, true).is_err());
    for spelling in ["100% literal", "literal %%"] {
        let mut literal = valid.clone();
        literal["dataDirectory"] = json!(std::env::temp_dir().join(spelling));
        assert!(run(literal).is_ok());
    }
    Ok(())
}

#[test]
fn malformed_ptq1_evidence_is_rejected() -> Result<(), String> {
    let lock = parse(SWIFT_LOCK)?;
    let evidence = parse(PTQ1_EVIDENCE)?;
    let selected = Profile::SwiftPq2Assessment;
    assert!(select_artifact(&lock, selected, true, None).is_err());
    assert!(select_artifact(&lock, selected, false, Some(&evidence)).is_err());
    assert!(select_artifact(&lock, Profile::ReferenceCpu, true, Some(&evidence)).is_err());
    for (key, value) in [
        ("SourceRevision", json!("0".repeat(40))),
        ("SourceSha256", json!("0".repeat(64))),
        ("SourceBytes", json!(1)),
        ("SourceBytes", json!("7206168928")),
        ("SourceBytes", json!(7_206_168_928.0)),
        ("CodecCommit", json!("0".repeat(40))),
        ("OutputFilename", json!("../unreviewed.gguf")),
        ("OutputSha256", json!("invalid")),
        ("OutputSha256", json!("0".repeat(64))),
        ("OutputSha256", json!("A".repeat(64))),
        ("OutputBytes", json!(1)),
        ("OutputBytes", json!(-1)),
        ("OutputBytes", json!("5946648928")),
        ("ExactPq2ReconstructionVerified", json!(false)),
        ("ExactPq2ReconstructionVerified", json!("true")),
        ("NonPq2TensorBytesCopiedVerbatim", json!(false)),
        ("ConvertedTensorCount", json!(401)),
        ("ConvertedBlocks", json!(1)),
        ("WorkingBufferBytes", json!(0)),
        ("SavingBytes", json!(0)),
        ("SourceFilename", json!("other.gguf")),
        ("SourceRepository", json!("other/model")),
        ("TokenizerMetadataSha256", json!("invalid")),
        ("TokenizerMetadataSha256", json!("0".repeat(64))),
        ("ModelProcessStarted", json!(true)),
        ("SchemaVersion", json!(2)),
        ("extraArguments", json!(["--unsafe"])),
    ] {
        let mut changed = evidence.clone();
        changed[key] = value;
        assert!(
            select_artifact(&lock, selected, true, Some(&changed)).is_err(),
            "accepted invalid {key}"
        );
    }
    let keys: Vec<String> = evidence
        .as_object()
        .ok_or("Invalid evidence fixture.")?
        .keys()
        .cloned()
        .collect();
    for key in keys {
        let mut missing = evidence.clone();
        missing
            .as_object_mut()
            .ok_or("Invalid fixture.")?
            .remove(&key);
        assert!(
            select_artifact(&lock, selected, true, Some(&missing)).is_err(),
            "accepted missing {key}"
        );
    }
    for malformed in [json!([]), json!([1, 2, 3]), Value::Null, json!(true)] {
        assert!(select_artifact(&lock, selected, true, Some(&malformed)).is_err());
    }
    Ok(())
}

#[test]
fn source_identity_and_runtime_url_drift_are_rejected() -> Result<(), String> {
    let lock = parse(SWIFT_LOCK)?;
    let evidence = parse(PTQ1_EVIDENCE)?;
    let selected = Profile::SwiftPq2Assessment;
    for (pointer, replacement) in [
        ("/model/revision", json!("0".repeat(40))),
        ("/model/sha256", json!("0".repeat(64))),
        ("/model/filename", json!("../other.gguf")),
        ("/model/bytes", json!(7_206_168_929_u64)),
        ("/model/bytes", json!("7206168928")),
        ("/runtime/commit", json!("0".repeat(40))),
        ("/runtime/asset/bytes", json!(1)),
        (
            "/runtime/asset/url",
            json!("https://other.example/runtime.zip"),
        ),
        ("/schemaVersion", json!(2)),
    ] {
        let mut changed = lock.clone();
        *changed
            .pointer_mut(pointer)
            .ok_or("Missing fixture pointer.")? = replacement;
        assert!(
            select_artifact(&changed, selected, false, None).is_err(),
            "accepted invalid {pointer}"
        );
        assert!(
            select_artifact(&changed, selected, true, Some(&evidence)).is_err(),
            "accepted invalid PTQ source {pointer}"
        );
    }
    Ok(())
}

#[test]
fn adapter_compatibility_and_argument_pairing_fail_closed() -> Result<(), String> {
    let lock = parse(SWIFT_LOCK)?;
    let adapter = parse(ADAPTER_LOCK)?;
    assert!(validate_adapter(&adapter, &lock).is_ok());
    for (pointer, replacement) in [
        ("/compatibility/modelRevision", json!("0".repeat(40))),
        ("/compatibility/runtimeCommit", json!("0".repeat(40))),
        ("/compatibility/modelRepository", json!("other/model")),
        ("/adapter/bytes", json!(1)),
        ("/adapter/sha256", json!("0".repeat(64))),
        ("/adapter/filename", json!("../other.gguf")),
        ("/adapter/url", json!("https://other.example/adapter.gguf")),
    ] {
        let mut changed = adapter.clone();
        *changed
            .pointer_mut(pointer)
            .ok_or("Missing adapter fixture pointer.")? = replacement;
        assert!(
            validate_adapter(&changed, &lock).is_err(),
            "accepted invalid {pointer}"
        );
    }
    for (enabled, path) in [
        (true, None),
        (false, Some("adapter.gguf")),
        (true, Some("  ")),
        (true, Some("a\0b")),
    ] {
        assert!(check_arguments("model.gguf", Profile::SwiftPq2Assessment, enabled, path).is_err());
    }
    let model = "C:\\external model & fixtures\\Swift.gguf";
    let arguments = completion_arguments(model, Profile::SwiftPq2Assessment)?;
    assert_eq!(arguments[1], model);
    for flag in ["--temp", "--reasoning", "--n-predict", "--lora", "--prompt"] {
        assert!(!arguments.iter().any(|argument| argument == flag));
    }
    for invalid in ["", "  ", "model\0.gguf"] {
        assert!(completion_arguments(invalid, Profile::ReferenceCpu).is_err());
    }
    assert_eq!(
        Profile::ReferenceCpu.memory_limit_bytes(),
        MEMORY_LIMIT_BYTES
    );
    assert_eq!(
        Profile::ReferenceCpu.required_available_bytes(),
        REQUIRED_AVAILABLE_BYTES
    );
    assert_eq!(
        Profile::SwiftPq2Assessment.memory_limit_bytes(),
        12_348_030_976
    );
    Ok(())
}

#[test]
fn swift_development_budget_keeps_four_gib_reserve_for_every_artifact() -> Result<(), String> {
    let selected = Profile::SwiftPq2Assessment;
    let ceiling =
        u64::try_from(selected.memory_limit_bytes()).map_err(|error| error.to_string())?;
    assert_eq!(ceiling, 12_348_030_976);
    assert_eq!(selected.required_available_bytes(), 16_642_998_272);
    assert_eq!(
        selected.required_available_bytes() - ceiling,
        4 * 1024 * 1024 * 1024
    );
    for use_adapter in [false, true] {
        for use_ptq1 in [false, true] {
            let fixed = policy(selected, use_adapter, use_ptq1)?;
            assert_eq!(fixed["ServiceCeilingBytes"], ceiling);
            assert_eq!(fixed["RequiredAvailableBytes"], 16_642_998_272_u64);
            assert_eq!(fixed["AvailableMemoryFloorBytes"], 4_294_967_296_u64);
            assert_eq!(fixed["ContextTokens"], 8192);
            assert_eq!(fixed["BatchTokens"], 128);
            assert_eq!(fixed["MicrobatchTokens"], 64);
            assert_eq!(
                fixed["EstimatedRuntimeHeadroomBytes"].as_u64(),
                fixed["EstimatedKnownSubtotalBytes"]
                    .as_u64()
                    .map(|known| ceiling - known)
            );
        }
    }
    assert_eq!(Profile::ReferenceCpu.memory_limit_bytes(), 19_327_352_832);
    assert_eq!(
        Profile::ReferenceCpu.required_available_bytes(),
        23_622_320_128
    );
    Ok(())
}

#[test]
fn family_thread_budgets_and_speculation_arguments_stay_bounded() -> Result<(), String> {
    assert_eq!(
        hardware::SnapdragonFamily::XElite.default_thread_budget(),
        (8, 12)
    );
    assert_eq!(
        hardware::SnapdragonFamily::X2EliteExtreme.default_thread_budget(),
        (6, 6)
    );
    assert_eq!(
        hardware::SnapdragonFamily::XElite.default_speech_asr_budget(),
        (8, 10)
    );
    assert_eq!(
        hardware::SnapdragonFamily::X2EliteExtreme.default_speech_asr_budget(),
        (8, 16)
    );

    let x1_args = completion_arguments_for_family(
        "C:\\models\\Swift.gguf",
        Profile::SwiftPq2Assessment,
        hardware::SnapdragonFamily::XElite,
        SpeculationPolicy::Suffix { draft_max: 4 },
    )?;
    for (flag, expected) in [
        ("--threads", "8"),
        ("--threads-batch", "12"),
        ("--spec-type", "suffix"),
        ("--draft-max", "4"),
    ] {
        let index = x1_args
            .iter()
            .position(|argument| argument == flag)
            .ok_or_else(|| format!("missing flag {flag}"))?;
        assert_eq!(x1_args.get(index + 1).map(String::as_str), Some(expected));
    }

    let x2_none = completion_arguments_for_family(
        "C:\\models\\Swift.gguf",
        Profile::SwiftPq2Assessment,
        hardware::SnapdragonFamily::X2EliteExtreme,
        SpeculationPolicy::None,
    )?;
    for (flag, expected) in [
        ("--threads", "6"),
        ("--threads-batch", "6"),
        ("--spec-type", "none"),
    ] {
        let index = x2_none
            .iter()
            .position(|argument| argument == flag)
            .ok_or_else(|| format!("missing flag {flag}"))?;
        assert_eq!(x2_none.get(index + 1).map(String::as_str), Some(expected));
    }
    assert!(!x2_none.iter().any(|argument| argument == "--draft-max"));

    for bad_draft in [0_u8, 9_u8] {
        assert!(
            completion_arguments_for_family(
                "C:\\models\\Swift.gguf",
                Profile::SwiftPq2Assessment,
                hardware::SnapdragonFamily::XElite,
                SpeculationPolicy::Suffix {
                    draft_max: bad_draft,
                },
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn swift_mmproj_lock_and_server_arguments_enforce_identity_and_path_bounds() -> Result<(), String> {
    let data = std::env::temp_dir().join("Bonsai mmproj policy fixture");
    let artifact = swift_mmproj_artifact(&data)?;
    assert_eq!(artifact.bytes, MMPROJ_BYTES);
    assert_eq!(
        artifact.sha256,
        "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903"
    );
    assert_eq!(
        artifact.path,
        data.join("mmproj")
            .join("6ed5e12bf84b7a63069882c91dd9e9218647d17b")
            .join("Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf")
    );
    assert!(swift_mmproj_artifact(Path::new("relative\\data")).is_err());

    let text_args = server::arguments_with_mmproj(
        "C:\\models\\Swift.gguf",
        None,
        "C:\\keys\\k.key",
        8080,
        Profile::SwiftPq2Assessment,
    )?;
    assert!(text_args.iter().any(|arg| arg == "--no-mmproj"));
    assert!(!text_args.iter().any(|arg| arg == "--mmproj"));

    let mmproj_text = artifact.path.to_string_lossy().into_owned();
    let vision_args = server::arguments_with_mmproj(
        "C:\\models\\Swift.gguf",
        Some(&mmproj_text),
        "C:\\keys\\k.key",
        8080,
        Profile::SwiftPq2Assessment,
    )?;
    assert!(!vision_args.iter().any(|arg| arg == "--no-mmproj"));
    let index = vision_args
        .iter()
        .position(|arg| arg == "--mmproj")
        .ok_or_else(|| "missing --mmproj".to_owned())?;
    assert_eq!(vision_args.get(index + 1), Some(&mmproj_text));
    assert_eq!(
        vision_args.get(index + 2).map(String::as_str),
        Some("--no-mmproj-offload")
    );
    assert_eq!(
        vision_args.get(index + 3).map(String::as_str),
        Some("--image-max-tokens")
    );
    assert_eq!(vision_args.get(index + 4).map(String::as_str), Some("1024"));

    for bad_mmproj in [
        "",
        "  ",
        "-leading-switch.gguf",
        "%LOCALAPPDATA%\\m.gguf",
        "a\0b",
    ] {
        assert!(
            server::arguments_with_mmproj(
                "C:\\models\\Swift.gguf",
                Some(bad_mmproj),
                "C:\\keys\\k.key",
                8080,
                Profile::SwiftPq2Assessment,
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn weight_packing_ptq1_delta_and_default_shape_conservation_hold() -> Result<(), String> {
    assert_eq!(
        context::SwiftWeightPacking::Pq2_0.weight_bytes(),
        7_206_168_928
    );
    assert_eq!(
        context::SwiftWeightPacking::Ptq1_0.weight_bytes(),
        5_946_648_928
    );
    let delta = context::SwiftWeightPacking::Pq2_0.weight_bytes()
        - context::SwiftWeightPacking::Ptq1_0.weight_bytes();
    assert_eq!(delta, 1_259_520_000);

    let default_shape =
        context::SwiftExecutionShape::for_profile(context::SwiftContextProfile::NativeRsq4);
    assert_eq!(
        default_shape.weights_and_fixed_runtime_bytes,
        context::FIXED_RUNTIME_BYTES
    );
    assert_eq!(default_shape.evictable_mapped_weight_bytes, 0);
    assert_eq!(default_shape.live_kv_copies, 2);
    assert_eq!(default_shape.reuse_recurrent_copies, 1);

    let ptq1 = default_shape.with_weight_packing_calibrated(context::SwiftWeightPacking::Ptq1_0);
    assert_eq!(
        default_shape.weights_and_fixed_runtime_bytes - ptq1.weights_and_fixed_runtime_bytes,
        1_259_520_000
    );
    assert_eq!(
        ptq1.with_weight_packing_calibrated(context::SwiftWeightPacking::Pq2_0),
        default_shape
    );
    assert_eq!(
        ptq1.with_weight_packing_calibrated(context::SwiftWeightPacking::Ptq1_0),
        ptq1
    );
    Ok(())
}

#[test]
fn mapped_weights_commit_credit_keeps_physical_ram_honest_and_validates_bounds()
-> Result<(), String> {
    let gib = 1024 * 1024 * 1024_u64;
    let profile = context::SwiftContextProfile::NativeRsq4;
    let base = context::SwiftExecutionShape::for_profile(profile);
    let mapped_bytes = context::SwiftWeightPacking::Ptq1_0.weight_bytes();
    let credited = base
        .with_mapped_weights_commit_credit(mapped_bytes)
        .ok_or_else(|| "valid mapped weight credit must succeed".to_owned())?;

    assert!(
        base.with_mapped_weights_commit_credit(base.weights_and_fixed_runtime_bytes)
            .is_some()
    );
    assert!(
        base.with_mapped_weights_commit_credit(base.weights_and_fixed_runtime_bytes + 1)
            .is_none()
    );
    assert!(base.with_mapped_weights_commit_credit(u64::MAX).is_none());

    let base_bd = base
        .breakdown(8_192)
        .ok_or_else(|| "missing base breakdown".to_owned())?;
    let cred_bd = credited
        .breakdown(8_192)
        .ok_or_else(|| "missing credited breakdown".to_owned())?;
    assert_eq!(
        cred_bd.total_footprint_bytes(),
        base_bd.total_footprint_bytes()
    );
    assert_eq!(
        base_bd
            .job_commit_bytes()
            .and_then(|job| job.checked_sub(cred_bd.job_commit_bytes()?)),
        Some(mapped_bytes)
    );
    assert_eq!(
        base_bd.private_job_commit_with_mapped_credit(mapped_bytes),
        cred_bd.job_commit_bytes()
    );
    assert_eq!(
        cred_bd.private_job_commit_with_mapped_credit(0),
        base_bd.job_commit_bytes()
    );
    assert!(
        cred_bd
            .private_job_commit_with_mapped_credit(cred_bd.weights_and_fixed_runtime_bytes)
            .is_some()
    );
    assert!(
        cred_bd
            .private_job_commit_with_mapped_credit(cred_bd.weights_and_fixed_runtime_bytes + 1)
            .is_none()
    );
    assert!(
        cred_bd
            .private_job_commit_with_mapped_credit(u64::MAX)
            .is_none()
    );

    let (base_plan, _) = context::SwiftContextPlan::for_execution_shape(20 * gib, profile, &base)?;
    let (cred_plan, _) =
        context::SwiftContextPlan::for_execution_shape(20 * gib, profile, &credited)?;
    assert_eq!(cred_plan.context_tokens, base_plan.context_tokens);
    assert_eq!(
        cred_plan.required_available_bytes,
        base_plan.required_available_bytes
    );
    assert_eq!(
        base_plan
            .memory_limit_bytes
            .checked_sub(cred_plan.memory_limit_bytes)
            .and_then(|delta| u64::try_from(delta).ok()),
        Some(mapped_bytes)
    );
    Ok(())
}

#[test]
fn hexagon_host_deduplication_enforces_prefill_guard_and_conserves_physical_bytes()
-> Result<(), String> {
    assert_eq!(
        context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
        2_888_663_040
    );
    let profile = context::SwiftContextProfile::NativeRsq4;
    let base = context::SwiftExecutionShape::for_profile(profile);

    let tiled = base
        .with_tiled_prefill_tokens(8, 8_192)
        .ok_or_else(|| "tiled prefill shape must build".to_owned())?;
    assert!(
        tiled
            .with_hexagon_host_deduplication(
                context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
                context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
            )
            .is_none()
    );
    let mut manual_scratch = base;
    manual_scratch.tile_prefill_scratch_bytes = 1;
    assert!(
        manual_scratch
            .with_hexagon_host_deduplication(0, 0)
            .is_none()
    );

    assert!(
        base.with_hexagon_host_deduplication(
            context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES + 1,
            context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES + 1,
        )
        .is_none()
    );

    let mut small_weights = base;
    small_weights.weights_and_fixed_runtime_bytes =
        context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES - 1;
    assert!(
        small_weights
            .with_hexagon_host_deduplication(
                context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
                context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
            )
            .is_none()
    );

    let over_mapped = base
        .with_mapped_weights_commit_credit(base.weights_and_fixed_runtime_bytes)
        .ok_or_else(|| "full mapped credit".to_owned())?;
    assert!(
        over_mapped
            .with_hexagon_host_deduplication(
                context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
                context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
            )
            .is_none()
    );

    let mut overflow_arena = base;
    overflow_arena.accelerator_resident_arena_bytes = u64::MAX - 10;
    assert!(
        overflow_arena
            .with_hexagon_host_deduplication(11, 0)
            .is_none()
    );

    let arena_bytes = context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES + 64 * 1024 * 1024;
    let dedup = base
        .with_hexagon_host_deduplication(
            arena_bytes,
            context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
        )
        .ok_or_else(|| "valid hexagon dedup must succeed".to_owned())?;
    let base_bd = base
        .breakdown(8_192)
        .ok_or_else(|| "missing base breakdown".to_owned())?;
    let dedup_bd = dedup
        .breakdown(8_192)
        .ok_or_else(|| "missing dedup breakdown".to_owned())?;
    assert_eq!(
        base_bd
            .job_commit_bytes()
            .and_then(|job| job.checked_sub(dedup_bd.job_commit_bytes()?)),
        Some(context::HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES)
    );
    assert_eq!(dedup_bd.accelerator_bytes(), Some(arena_bytes));
    assert_eq!(
        dedup_bd
            .total_footprint_bytes()
            .and_then(|total| total.checked_sub(base_bd.total_footprint_bytes()?)),
        Some(64 * 1024 * 1024)
    );
    Ok(())
}

#[test]
fn multi_branch_checkpoint_slots_enforce_zero_to_eight_bounds_and_monotonicity()
-> Result<(), String> {
    assert_eq!(context::MAX_REUSABLE_BRANCH_CHECKPOINTS, 8);
    let gib = 1024 * 1024 * 1024_u64;
    let profile = context::SwiftContextProfile::NativeRsq4;
    let base = context::SwiftExecutionShape::for_profile(profile);

    assert!(base.with_branch_reuse_checkpoints(9).is_none());
    assert!(base.with_branch_reuse_checkpoints(u8::MAX).is_none());

    let mut prev_recurrent = 0_u64;
    let mut prev_tokens = usize::MAX;
    for slots in 0..=context::MAX_REUSABLE_BRANCH_CHECKPOINTS {
        let shaped = base
            .with_branch_reuse_checkpoints(slots)
            .ok_or_else(|| format!("branch slots {slots} must be valid"))?;
        assert_eq!(shaped.reuse_recurrent_copies, slots);
        let recurrent = shaped
            .recurrent_checkpoint_bytes()
            .ok_or_else(|| "recurrent checkpoint overflow".to_owned())?;
        assert_eq!(recurrent, (2 + u64::from(slots)) * context::RECURRENT_BYTES);
        if slots > 0 {
            assert_eq!(recurrent - prev_recurrent, context::RECURRENT_BYTES);
        }
        prev_recurrent = recurrent;

        let (plan, _) = context::SwiftContextPlan::for_execution_shape(18 * gib, profile, &shaped)?;
        assert!(plan.context_tokens <= prev_tokens);
        prev_tokens = plan.context_tokens;
    }
    Ok(())
}

#[test]
fn cow_kv_admission_saves_over_four_gib_at_262144_k4v4_and_preserves_reserves_and_state_caps()
-> Result<(), String> {
    let gib = 1024 * 1024 * 1024_u64;
    let full_profile = context::SwiftContextProfile::NativeRsq4Full;
    let rollback_shape = context::SwiftExecutionShape::for_profile(full_profile);
    let append_only_shape = rollback_shape.with_append_only_cow_kv_admission();
    assert_eq!(append_only_shape.live_kv_copies, 1);

    let rollback_bd = rollback_shape
        .breakdown(context::MODEL_CONTEXT_TOKENS)
        .ok_or_else(|| "missing rollback breakdown".to_owned())?;
    let append_bd = append_only_shape
        .breakdown(context::MODEL_CONTEXT_TOKENS)
        .ok_or_else(|| "missing append-only breakdown".to_owned())?;

    let rollback_tail_cow_saving = rollback_bd.live_kv_bytes - rollback_bd.physical_cow_kv_bytes;
    assert_eq!(
        rollback_tail_cow_saving,
        262_080 * context::RSQ4_KV_BYTES_PER_TOKEN
    );
    assert_eq!(rollback_tail_cow_saving, 4_428_103_680);

    let append_only_saving = rollback_bd.live_kv_bytes - append_bd.physical_cow_kv_bytes;
    assert_eq!(
        append_only_saving,
        262_144 * context::RSQ4_KV_BYTES_PER_TOKEN
    );
    assert_eq!(append_only_saving, 4_429_185_024);
    assert!(append_only_saving > 4 * gib);

    // At 20 GiB available RAM, 2x rollback cannot fit 262,144 tokens, whereas CoW admits full 262,144.
    let baseline_20g = context::SwiftContextPlan::for_profile(20 * gib, full_profile)?;
    assert!(baseline_20g.context_tokens < context::MODEL_CONTEXT_TOKENS);

    let cow_append_20g = context::SwiftContextPlan::for_append_only_cow_shape(
        20 * gib,
        full_profile,
        &append_only_shape,
    )?;
    assert_eq!(cow_append_20g.context_tokens, context::MODEL_CONTEXT_TOKENS);
    assert!(cow_append_20g.required_available_bytes + context::STARTUP_RESERVE_BYTES <= 20 * gib);

    let cow_rollback_20g = context::SwiftContextPlan::for_append_only_cow_shape(
        20 * gib,
        full_profile,
        &rollback_shape,
    )?;
    assert_eq!(
        cow_rollback_20g.context_tokens,
        context::MODEL_CONTEXT_TOKENS
    );
    assert_eq!(
        cow_rollback_20g.required_available_bytes - cow_append_20g.required_available_bytes,
        u64::try_from(context::KV_COW_PAGE_TOKENS).map_err(|error| error.to_string())?
            * context::RSQ4_KV_BYTES_PER_TOKEN
    );

    // Monotonicity across available RAM and enforcement of HOST_RESERVE_BYTES.
    let mut prev_ctx = 0_usize;
    for avail_gib in 16..=24_u64 {
        let plan = context::SwiftContextPlan::for_append_only_cow_shape(
            avail_gib * gib,
            full_profile,
            &append_only_shape,
        )?;
        assert!(plan.context_tokens >= prev_ctx);
        prev_ctx = plan.context_tokens;
        assert_eq!(
            plan.required_available_bytes
                - u64::try_from(plan.memory_limit_bytes).map_err(|error| error.to_string())?,
            context::HOST_RESERVE_BYTES
        );
        assert!(plan.required_available_bytes + context::STARTUP_RESERVE_BYTES <= avail_gib * gib);
    }

    // Ordinary NativeRsq4 retains its 4 GiB state cap (241,664 tokens) even under unlimited RAM.
    let capped = context::SwiftContextPlan::for_append_only_cow_shape(
        u64::MAX,
        context::SwiftContextProfile::NativeRsq4,
        &context::SwiftExecutionShape::for_profile(context::SwiftContextProfile::NativeRsq4)
            .with_append_only_cow_kv_admission(),
    )?;
    assert_eq!(capped.context_tokens, 241_664);

    // Minimum admission boundary and invalid zero-KV-copy rejection.
    let min_kv = append_only_shape
        .physical_cow_kv_bytes(context::MIN_CONTEXT_TOKENS)
        .ok_or_else(|| "missing min cow kv".to_owned())?;
    let min_fixed = append_only_shape
        .breakdown(0)
        .and_then(|bd| bd.physical_cow_footprint_bytes())
        .ok_or_else(|| "missing min fixed".to_owned())?;
    let minimum = context::HOST_RESERVE_BYTES + context::STARTUP_RESERVE_BYTES + min_fixed + min_kv;
    assert!(
        context::SwiftContextPlan::for_append_only_cow_shape(
            minimum - 1,
            full_profile,
            &append_only_shape,
        )
        .is_err()
    );
    let min_plan = context::SwiftContextPlan::for_append_only_cow_shape(
        minimum,
        full_profile,
        &append_only_shape,
    )?;
    assert_eq!(min_plan.context_tokens, context::MIN_CONTEXT_TOKENS);

    let mut zero_kv = append_only_shape;
    zero_kv.live_kv_copies = 0;
    assert!(
        context::SwiftContextPlan::for_append_only_cow_shape(24 * gib, full_profile, &zero_kv)
            .is_err()
    );
    Ok(())
}

#[test]
fn calibrated_cow_execution_matches_prism_accounting_across_packings_reuse_and_speculation()
-> Result<(), String> {
    use context::{
        CONTEXT_PAGE_TOKENS, FIXED_RUNTIME_BYTES, HOST_RESERVE_BYTES, RECURRENT_BYTES,
        STARTUP_RESERVE_BYTES, STATE_MEMORY_FLOOR_BYTES, SwiftCalibratedOptions,
        SwiftCalibratedReuse, SwiftContextPlan, SwiftContextProfile, SwiftWeightPacking,
        swift_selected_state_budget_bytes, swift_tiled_prefill_scratch_bytes,
        swift_tiled_verification_scratch_bytes,
    };

    let gib = 1024 * 1024 * 1024_u64;
    let pq2_options = SwiftCalibratedOptions {
        weight_packing: SwiftWeightPacking::Pq2_0,
        request_vision: true,
        prefill_tokens: None,
        chunkwise_gdn_wy_f32: false,
        speculation_tokens: None,
        prompt_reuse: SwiftCalibratedReuse::Disabled,
    };
    let pq2_24g = SwiftContextPlan::for_calibrated_execution(
        24 * gib,
        SwiftContextProfile::NativeRsq4Full,
        pq2_options,
    )?;
    assert_eq!(pq2_24g.context_plan.context_tokens, 262_144);
    assert!(pq2_24g.vision_admitted);
    let expected_cow_physical_pq2 = FIXED_RUNTIME_BYTES
        + 2 * RECURRENT_BYTES
        + 262_144 * 17_120
        + 64 * 16_896
        + MMPROJ_RESERVE_BYTES;
    assert_eq!(
        u64::try_from(pq2_24g.working_set_limit_bytes).map_err(|e| e.to_string())?,
        expected_cow_physical_pq2
    );
    assert_eq!(
        u64::try_from(pq2_24g.context_plan.memory_limit_bytes).map_err(|e| e.to_string())?,
        expected_cow_physical_pq2 - SWIFT_WEIGHT_BYTES
    );
    assert_eq!(
        pq2_24g.context_plan.required_available_bytes,
        expected_cow_physical_pq2 + HOST_RESERVE_BYTES
    );

    // PTQ1 calibrates fixed bytes down by 1_259_520_000 and credits PTQ1_WEIGHT_BYTES from Job commit.
    let ptq1_options = SwiftCalibratedOptions {
        weight_packing: SwiftWeightPacking::Ptq1_0,
        request_vision: false,
        prefill_tokens: Some(16),
        chunkwise_gdn_wy_f32: false,
        speculation_tokens: Some(4),
        prompt_reuse: SwiftCalibratedReuse::PromptEndAndLatest {
            checkpoints: 2,
            checkpoint_bytes: None,
        },
    };
    let ptq1_plan = SwiftContextPlan::for_calibrated_execution(
        24 * gib,
        SwiftContextProfile::NativeRsq4Full,
        ptq1_options,
    )?;
    assert!(!ptq1_plan.vision_admitted);
    let scratch = ptq1_plan
        .prefill_scratch_bytes
        .ok_or_else(|| "missing prefill scratch".to_owned())?;
    let (verify_scratch, checkpoint_bytes) = ptq1_plan
        .speculation_reserves
        .ok_or_else(|| "missing speculation reserves".to_owned())?;
    assert_eq!(verify_scratch, 7_946_432);
    assert_eq!(
        u64::try_from(scratch).ok(),
        swift_tiled_prefill_scratch_bytes(
            16,
            ptq1_plan.context_plan.context_tokens,
            context::SwiftKvCacheFormat::Rsq4,
        )
    );
    assert_eq!(
        u64::try_from(checkpoint_bytes).ok(),
        swift_selected_state_budget_bytes(
            ptq1_plan.context_plan.context_tokens,
            context::SwiftKvCacheFormat::Rsq4,
        )
    );
    assert!(
        checkpoint_bytes >= usize::try_from(STATE_MEMORY_FLOOR_BYTES).map_err(|e| e.to_string())?
    );
    let ctx = u64::try_from(ptq1_plan.context_plan.context_tokens).map_err(|e| e.to_string())?;
    // Base 3 recurrent copies (active + rollback + verify) + (2 + 1) reuse copies = 6 total.
    let expected_physical_ptq1 = (FIXED_RUNTIME_BYTES - (SWIFT_WEIGHT_BYTES - PTQ1_WEIGHT_BYTES))
        + 6 * RECURRENT_BYTES
        + ctx * 17_120
        + 64 * 16_896
        + u64::try_from(scratch).map_err(|e| e.to_string())?
        + swift_tiled_verification_scratch_bytes(16).ok_or_else(|| "missing verify".to_owned())?;
    assert_eq!(
        u64::try_from(ptq1_plan.working_set_limit_bytes).map_err(|e| e.to_string())?,
        expected_physical_ptq1
    );
    assert_eq!(
        u64::try_from(ptq1_plan.context_plan.memory_limit_bytes).map_err(|e| e.to_string())?,
        expected_physical_ptq1 - PTQ1_WEIGHT_BYTES
    );
    assert!(ptq1_plan.context_plan.required_available_bytes + STARTUP_RESERVE_BYTES <= 24 * gib);
    assert!(ptq1_plan.context_plan.context_tokens % CONTEXT_PAGE_TOKENS == 0);

    // Invalid option pairings reject cleanly.
    assert!(
        SwiftContextPlan::for_calibrated_execution(
            24 * gib,
            SwiftContextProfile::NativeRsq4Full,
            SwiftCalibratedOptions {
                chunkwise_gdn_wy_f32: true,
                prefill_tokens: Some(8),
                ..pq2_options
            },
        )
        .is_err()
    );
    assert!(
        SwiftContextPlan::for_calibrated_execution(
            24 * gib,
            SwiftContextProfile::NativeF16,
            SwiftCalibratedOptions {
                chunkwise_gdn_wy_f32: true,
                prefill_tokens: Some(8),
                ..pq2_options
            },
        )
        .is_ok()
    );
    Ok(())
}
