//! Capture identity, split isolation and artifact publication regression checks.

use super::*;

fn vector(width: usize, first: f32) -> Vec<f32> {
    let mut values = vec![0.0; width];
    values[0] = first;
    values
}

fn fixture() -> Value {
    json!({"operation":"fit","schemaVersion":1,
        "source":{"modelSha256":"a".repeat(64),"runtimeSha256":"b".repeat(64),
            "runtimeSourceCommit":"c".repeat(40),"adapterSha256":null,
            "adapterScaleBits":"00000000","weightRepresentation":"PQ2_0",
            "executionProfile":EXECUTION_PROFILE,"graphIdentity":"synthetic-unit-fixture"},
        "site":{"tensor":"blk.15.ffn_down.weight","inputBasis":"original-ffn-hidden",
            "inputWidth":INPUT_WIDTH,"outputWidth":OUTPUT_WIDTH},
        "rows":[
            {"sampleId":"fit-target","groupId":"fit-target-trace","tokenIndex":0,
                "split":"fit","label":"target","activation":vector(INPUT_WIDTH,1.0),
                "residual":vector(OUTPUT_WIDTH,1.0)},
            {"sampleId":"fit-preserve","groupId":"fit-preserve-trace","tokenIndex":0,
                "split":"fit","label":"preserve","activation":vector(INPUT_WIDTH,0.0),
                "residual":vector(OUTPUT_WIDTH,0.0)},
            {"sampleId":"holdout-target","groupId":"holdout-target-trace","tokenIndex":0,
                "split":"holdout","label":"target","activation":vector(INPUT_WIDTH,1.0),
                "residual":vector(OUTPUT_WIDTH,1.0)}],
        "direction":vector(OUTPUT_WIDTH,1.0),"directionSourceSha256":"d".repeat(64),
        "controls":{"lambda":1.0,"targetPenalty":2.0,"preservePenalty":2.0,
            "delta":1.0,"epsilon":0.0,"maxIterations":5000,
            "relativeGapTolerance":0.000_001,"timeBudgetMilliseconds":2000}})
}

#[test]
fn fit_binds_exported_factors_and_keeps_behavior_unqualified() -> crate::Result<()> {
    let result = run(fixture())?;
    assert_eq!(result["status"], "converged");
    assert_eq!(result["artifact"]["enabled"], false);
    assert_eq!(result["artifact"]["behaviorQualified"], false);
    assert_eq!(
        result["artifact"]["compositionOrder"],
        "after-standard-adapter"
    );
    assert_eq!(
        result["artifact"]["optimizationCertificate"]["certified"],
        true
    );
    let factors: Vec<f32> = serde_json::from_value(result["artifact"]["factors"]["a"].clone())
        .map_err(|error| error.to_string())?;
    assert_eq!(
        result["artifact"]["factors"]["aSha256"],
        factor_hash(&factors)?
    );
    assert_eq!(result["holdoutUsedForFit"], false);
    Ok(())
}

#[test]
fn direction_estimation_ignores_held_out_residuals() -> crate::Result<()> {
    let mut request = fixture();
    request["operation"] = json!("direction");
    request
        .as_object_mut()
        .ok_or("Fixture must be an object.")?
        .remove("direction");
    request
        .as_object_mut()
        .ok_or("Fixture must be an object.")?
        .remove("directionSourceSha256");
    let first = run(request.clone())?;
    request["rows"][2]["residual"] = json!(vector(OUTPUT_WIDTH, -500.0));
    let second = run(request)?;
    assert_eq!(first["direction"], second["direction"]);
    assert_eq!(
        first["directionSourceSha256"],
        second["directionSourceSha256"]
    );
    assert_eq!(first["holdoutUsed"], false);
    Ok(())
}

#[test]
fn fit_excludes_conflicting_holdout_rows_but_reports_their_effect() -> crate::Result<()> {
    let mut request = fixture();
    let first = run(request.clone())?;
    assert_eq!(first["status"], "converged");
    assert!(first["artifact"]["factors"].is_object());

    // This target contradicts the fitting target. Including it in the solver
    // would change both the optimum and its certificate, not just diagnostics.
    request["rows"][2]["activation"] = json!(vector(INPUT_WIDTH, -128.0));
    request["rows"][2]["residual"] = json!(vector(OUTPUT_WIDTH, -500.0));
    let second = run(request)?;
    assert_eq!(second["status"], "converged");
    for field in ["iterations", "optimizationCertificate"] {
        assert_eq!(first[field], second[field], "{field}");
    }
    assert_eq!(first["artifact"]["factors"], second["artifact"]["factors"]);
    assert_ne!(
        first["artifact"]["fitInputSha256"],
        second["artifact"]["fitInputSha256"]
    );
    assert_eq!(first["rows"][0], second["rows"][0]);
    assert_eq!(first["rows"][1], second["rows"][1]);
    assert_eq!(second["rows"][2]["split"], "holdout");
    assert!(
        second["rows"][2]["constraintViolation"]
            .as_f64()
            .is_some_and(|violation| violation > 100.0)
    );
    assert_ne!(first["rows"][2], second["rows"][2]);
    assert_eq!(second["holdoutUsedForFit"], false);
    Ok(())
}

#[test]
fn split_isolation_rejects_same_sample_in_distinct_trace_groups() {
    let mut request = fixture();
    // Distinct token positions and groups must not let one sample appear in
    // both fit and holdout; the existing group-level control cannot detect it.
    request["rows"][2]["sampleId"] = json!("fit-target");
    request["rows"][2]["tokenIndex"] = json!(1);
    assert_eq!(
        run(request).err().as_deref(),
        Some("A sample or trace group crosses the fit/holdout boundary.")
    );
}

#[test]
fn split_leakage_unknown_fields_and_bad_shapes_reject() {
    let mut request = fixture();
    request["rows"][2]["groupId"] = json!("fit-target-trace");
    assert!(run(request).is_err());
    let mut request = fixture();
    request["unexpected"] = json!(true);
    assert!(run(request).is_err());
    let mut request = fixture();
    request["site"]["inputBasis"] = json!("hadamard-folded");
    assert!(run(request).is_err());
    let mut request = fixture();
    request["direction"] = json!(vec![0.0; OUTPUT_WIDTH]);
    assert!(run(request).is_err());
    let mut request = fixture();
    request["source"]["adapterScaleBits"] = json!("7f800000");
    assert!(run(request).is_err());
}

#[test]
fn unconverged_fit_returns_no_artifact() -> crate::Result<()> {
    let mut request = fixture();
    request["controls"]["maxIterations"] = json!(1);
    let result = run(request)?;
    assert_ne!(result["status"], "converged");
    assert!(result["artifact"].is_null());
    assert_eq!(result["optimizationCertificate"]["certified"], false);
    Ok(())
}

#[test]
fn atomic_publication_never_overwrites_existing_output() -> crate::Result<()> {
    let directory =
        std::env::temp_dir().join(format!("bonsai-weight-patch-{}", std::process::id()));
    fs::create_dir(&directory).map_err(|error| error.to_string())?;
    let path = directory.join("patch.json");
    let result: crate::Result<()> = (|| {
        publish(&path, b"first")?;
        assert!(publish(&path, b"second").is_err());
        assert_eq!(
            fs::read(&path).map_err(|error| error.to_string())?,
            b"first"
        );
        Ok(())
    })();
    let cleanup = fs::remove_file(&path).and_then(|()| fs::remove_dir(&directory));
    result?;
    cleanup.map_err(|error| error.to_string())
}

#[test]
fn fit_input_hash_omits_output_path_receipt_omits_path_and_rows_cap_to_top_violations()
-> crate::Result<()> {
    let baseline = run(fixture())?;
    assert_eq!(baseline["published"], false);
    assert!(
        baseline
            .as_object()
            .ok_or("Missing patch report.")?
            .get("outputPath")
            .is_none()
    );
    assert_eq!(baseline["rowSummary"]["totalRows"], 3);
    assert_eq!(baseline["rowSummary"]["reportedRows"], 3);
    assert_eq!(baseline["rowSummary"]["omittedRows"], 0);
    assert!(
        baseline["rowSummary"]["factorsL2Norm"]
            .as_f64()
            .is_some_and(|v| v.is_finite())
    );
    assert!(
        baseline["rowSummary"]["factorsEnergyEntropy"]
            .as_f64()
            .is_some_and(|v| (0.0..=1.0).contains(&v))
    );
    assert!(
        baseline["rowSummary"]["factorsCoordinateCoverage"]
            .as_f64()
            .is_some_and(|v| (0.0..=1.0).contains(&v))
    );

    let dir = std::env::temp_dir().join(format!(
        "bonsai-weight-patch-pathfree-{}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let path = dir.join("patch.json");
    let mut with_output = fixture();
    with_output["outputPath"] = json!(path);
    let published = run(with_output);
    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir(&dir);
    let published = published?;
    assert_eq!(published["published"], true);
    assert!(
        published
            .as_object()
            .ok_or("Missing published patch report.")?
            .get("outputPath")
            .is_none()
    );
    assert_eq!(
        baseline["artifact"]["fitInputSha256"],
        published["artifact"]["fitInputSha256"]
    );

    let mut large = fixture();
    let mut rows = vec![
        json!({"sampleId":"fit-target","groupId":"fit-target-trace","tokenIndex":0,
            "split":"fit","label":"target","activation":vector(INPUT_WIDTH,1.0),
            "residual":vector(OUTPUT_WIDTH,1.0)}),
        json!({"sampleId":"fit-preserve","groupId":"fit-preserve-trace","tokenIndex":0,
            "split":"fit","label":"preserve","activation":vector(INPUT_WIDTH,0.0),
            "residual":vector(OUTPUT_WIDTH,0.0)}),
    ];
    for idx in 0..14_u32 {
        let (label, scale) = if idx >= 6 {
            (
                "target",
                f32::from(u8::try_from(idx + 2).map_err(|e| e.to_string())?) * -10.0,
            )
        } else {
            ("preserve", 0.0)
        };
        rows.push(json!({
            "sampleId": format!("holdout-{idx}"),
            "groupId": format!("holdout-group-{idx}"),
            "tokenIndex": 0,
            "split": "holdout",
            "label": label,
            "activation": vector(INPUT_WIDTH, scale),
            "residual": vector(OUTPUT_WIDTH, 1.0)
        }));
    }
    large["rows"] = Value::Array(rows);
    let capped = run(large)?;
    let reported = capped["rows"].as_array().ok_or("Missing rows array.")?;
    assert_eq!(reported.len(), MAX_VIOLATION_REPORT_ROWS);
    assert_eq!(capped["rowSummary"]["totalRows"], 16);
    assert_eq!(
        capped["rowSummary"]["reportedRows"],
        MAX_VIOLATION_REPORT_ROWS
    );
    assert_eq!(capped["rowSummary"]["omittedRows"], 8);
    assert_eq!(
        capped["rowSummary"]["reportedRowLimit"],
        MAX_VIOLATION_REPORT_ROWS
    );
    for entry in reported {
        assert_eq!(entry["split"], "holdout");
        assert!(
            entry["constraintViolation"]
                .as_f64()
                .is_some_and(|v| v > 1.0)
        );
    }
    Ok(())
}
