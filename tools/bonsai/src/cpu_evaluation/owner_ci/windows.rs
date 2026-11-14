//! Native X2 CI entry. No owner deadline, credentials or model Job passes through a shell.

use super::{
    BUDGET, RETENTION_RESERVE, Request, Result, Start, Value, decimal, json, require, validate,
    work,
};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

fn variable(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}
fn external(path: &Path) -> Result<()> {
    require(
        path.to_str()
            .is_some_and(|name| !name.contains(['\r', '\n', '\0'])),
        "Owner paths must be expanded single-line Unicode names.",
    )?;
    bonsai_artifacts::validate_external_path(path).map_err(|error| error.to_string())
}
fn read(path: &Path) -> Result<Value> {
    external(path)?;
    let file = crate::build_tools::input_file(path)?;
    let bytes = crate::bounded_input::read(file, 4_194_304)
        .map_err(|error| error.into_message("Owner input exceeds byte bound."))?;
    super::super::source_comparison::json_bytes(
        bytes.strip_prefix(&[239, 187, 191]).unwrap_or(&bytes),
    )
}
fn save(path: &Path, value: &Value, new: bool) -> Result<()> {
    external(path)?;
    let mut options = fs::OpenOptions::new();
    options.write(true);
    if new {
        options.create_new(true);
    } else {
        options.create(true).truncate(true);
    }
    let mut file = options.open(path).map_err(|error| error.to_string())?;
    file.write_all(&serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?)
        .and_then(|()| file.write_all(b"\n"))
        .and_then(|()| file.sync_all())
        .map_err(|error| error.to_string())
}
fn append(name: &str, line: &str) -> Result<()> {
    let path = PathBuf::from(std::env::var_os(name).ok_or_else(|| format!("Missing {name}."))?);
    external(&path)?;
    fs::OpenOptions::new()
        .append(true)
        .open(path)
        .and_then(|mut file| writeln!(file, "{line}"))
        .map_err(|error| error.to_string())
}
fn ticks() -> Result<(u64, u64)> {
    bonsai_windows_process::monotonic_ticks().map_err(|error| error.to_string())
}
fn unix() -> Result<u64> {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_millis(),
    )
    .map_err(|error| error.to_string())
}
fn guard() -> Result<()> {
    require(
        cfg!(target_arch = "aarch64"),
        "X2 owner execution requires native Windows ARM64.",
    )?;
    for (name, expected) in [
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REF", "refs/heads/main"),
        ("GITHUB_REPOSITORY", "symbiogenesis/SwiftBonsai2Runner"),
        ("RUNNER_ENVIRONMENT", "self-hosted"),
        ("RUNNER_ARCH", "ARM64"),
    ] {
        require(
            variable(name) == expected,
            "X2 owner execution requires its identified main-only CI lane.",
        )?;
    }
    require(
        decimal(&variable("GITHUB_RUN_ID"))
            && !variable("RUNNER_NAME").is_empty()
            && crate::identity::is_lower_hex(&variable("GITHUB_SHA"), 40),
        "Missing source-bound owner CI identity.",
    )
}
fn identity(revision: &str) -> Result<String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    external(&executable)?;
    let receipt = read(&executable.with_file_name("receipt.json"))?;
    require(
        receipt["SchemaVersion"].as_u64() == Some(3) && receipt["Revision"] == revision,
        "Checked native helper source identity differs.",
    )?;
    for (name, field) in [
        (executable.clone(), "Sha256"),
        (
            executable.with_file_name("bonsai-reference-numerics.exe"),
            "NumericSha256",
        ),
        (
            executable.with_file_name("bonsai-completion-test-probe.exe"),
            "ProbeSha256",
        ),
    ] {
        require(
            receipt[field]
                .as_str()
                .is_some_and(|value| crate::identity::is_lower_hex(value, 64))
                && crate::startup_cache::digest_file(&name)? == receipt[field],
            "Checked native helper bytes differ.",
        )?;
    }
    Ok(receipt["Sha256"]
        .as_str()
        .ok_or("Missing helper digest.")?
        .into())
}
fn original(path: &Path) -> Result<Start> {
    guard()?;
    let receipt: Start = serde_json::from_value(read(path)?).map_err(|error| error.to_string())?;
    let (now, frequency) = ticks()?;
    validate(
        &receipt,
        &variable("GITHUB_RUN_ID"),
        &variable("GITHUB_SHA"),
        now,
        frequency,
        unix()?,
    )?;
    require(
        identity(&receipt.runner_revision)? == receipt.helper_sha256,
        "Owner start and same-run helper identities differ.",
    )?;
    Ok(receipt)
}
fn matching(before: &Value, after: &Value) -> bool {
    super::super::bundle::power(before)
        .ok()
        .zip(super::super::bundle::power(after).ok())
        .is_some_and(|(before, after)| before == after)
}
fn create(output: &Path, run: &str, revision: &str) -> Result<Value> {
    let (started, frequency) = ticks()?;
    let utc = unix()?;
    let power_before = crate::host_cpu::power_snapshot();
    require(
        cfg!(target_arch = "aarch64")
            && decimal(run)
            && crate::identity::is_lower_hex(revision, 40),
        "Owner start requires native ARM64 and exact run/source identities.",
    )?;
    let digest = identity(revision)?;
    let receipt=Start{schema_version:1,kind:"runner-x2-cpu-owner-start".into(),run_id:run.into(),runner_revision:revision.into(),helper_sha256:digest,clock_frequency:frequency,started_tick:started,deadline_tick:started.checked_add(BUDGET*frequency).ok_or("Owner deadline overflow.")?,started_unix_milliseconds:utc,deadline_unix_milliseconds:utc+BUDGET*1000,budget_seconds:BUDGET,conservative_estimate_seconds:BUDGET,fixed_work:"One complete fixed four-sample X2 screen; bootstrap, checkout/tool intake, independent CIM/native conditions, artifact verification, four existing 45-second sample caps, cleanup and retention share 300 seconds. Reserve 70 seconds for a bounded one-minute upload and final power. Refuse when the complete screen cannot fit; no retry, reduced matrix or new clock.".into(),power_before};
    let value = serde_json::to_value(receipt).map_err(|error| error.to_string())?;
    save(output, &value, true)?;
    Ok(value)
}
fn admit(path: &Path) -> Result<Value> {
    let receipt = original(path)?;
    let power = crate::host_cpu::power_snapshot();
    let fit = work(&receipt, ticks()?.0);
    append("GITHUB_STEP_SUMMARY",&serde_json::to_string(&json!({"kind":"runner-x2-owner-admission","power_before":receipt.power_before,"power_at_admission":power,"complete_screen_fits":fit.is_ok(),"promotion_qualified":false})).map_err(|error|error.to_string())?)?;
    fit?;
    require(
        matching(&receipt.power_before, &power),
        "Setup power is unknown or changed; no X2 screen admitted.",
    )?;
    let stage = PathBuf::from(std::env::var_os("RUNNER_TEMP").ok_or("Missing RUNNER_TEMP.")?)
        .join("X2-Cpu-OwnerStart.json");
    save(
        &stage,
        &serde_json::to_value(receipt).map_err(|error| error.to_string())?,
        true,
    )?;
    append(
        "GITHUB_ENV",
        &format!("BONSAI_X2_OWNER_START={}", stage.display()),
    )?;
    Ok(json!({"Status":"admitted","StartReceipt":stage,"PowerAtAdmission":power}))
}
fn finalize(path: &Path) -> Result<Value> {
    let receipt = original(path)?;
    let (before, _) = ticks()?;
    let after = if before < receipt.deadline_tick {
        crate::host_cpu::power_snapshot()
    } else {
        bonsai_cpu_policy::power::unknown_snapshot()
    };
    let (now, frequency) = ticks()?;
    let eligible = matching(&receipt.power_before, &after) && now < receipt.deadline_tick;
    let result = json!({"kind":"runner-x2-cpu-post-upload-budget","run_id":receipt.run_id,"original_deadline_tick":receipt.deadline_tick,"completed_tick":now,"elapsed_ticks":now.saturating_sub(receipt.started_tick),"clock_frequency":frequency,"power_before":receipt.power_before,"power_after":after,"performance_comparison_eligible":eligible,"continuous_power_stability_proven":false,"promotion_qualified":false});
    append(
        "GITHUB_STEP_SUMMARY",
        &serde_json::to_string(&result).map_err(|error| error.to_string())?,
    )?;
    require(
        eligible,
        "Original X2 allowance expired or final power is unknown/mismatched; qualification remains pending.",
    )?;
    Ok(result)
}

fn capture(path: &Path, data: &Path, processors: &Value, attribution: bool) -> Result<Value> {
    let receipt = original(path)?;
    external(data)?;
    let before = crate::host_cpu::power_snapshot();
    require(
        matching(&receipt.power_before, &before),
        "Initial X2 power is unknown or changed.",
    )?;
    let evidence = std::env::temp_dir().join(format!(
        "Bonsai-X2Owner-{}",
        crate::startup_process::unique()
    ));
    external(&evidence)?;
    fs::create_dir(&evidence).map_err(|error| error.to_string())?;
    append(
        "GITHUB_ENV",
        &format!("BONSAI_X2_CPU_EVIDENCE={}", evidence.display()),
    )?;
    append("GITHUB_ENV", "BONSAI_X2_UPLOAD_ALLOWED=false")?;
    save(
        &evidence.join("owner-start.json"),
        &serde_json::to_value(&receipt).map_err(|error| error.to_string())?,
        true,
    )?;
    let working = evidence.join("working");
    fs::create_dir(&working).map_err(|error| error.to_string())?;
    let (seconds, deadline) = match work(&receipt, ticks()?.0) {
        Ok(allowance) => allowance,
        Err(reason) => {
            save(
                &evidence.join("owner-budget.json"),
                &json!({"kind":"runner-x2-cpu-owner-budget","status":"no-go","reason":reason,"original_start_tick":receipt.started_tick,"original_deadline_tick":receipt.deadline_tick,"completed_tick":ticks()?.0,"clock_frequency":receipt.clock_frequency,"budget_seconds":BUDGET,"retention_reserve_seconds":RETENTION_RESERVE,"power_before":receipt.power_before,"power_after":crate::host_cpu::power_snapshot(),"model_trials_requested":false,"promotion_qualified":false}),
                true,
            )?;
            fs::remove_dir(&working).map_err(|error| error.to_string())?;
            upload_window(&receipt)?;
            return Err(reason);
        }
    };
    let request = json!({"operation":"capture","dataDirectory":data,"cimProcessors":processors,"observeOutputProgress":attribution,"workBudgetSeconds":seconds,"workDeadlineTick":deadline,"evidenceDirectory":evidence.join("screen"),"parentProcessId":std::process::id(),"parentCreationTimeFileTime":bonsai_windows_process::current_process_creation_time().map_err(|error|error.to_string())?});
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let arguments = vec![
        "cpu-evaluation".into(),
        "--json".into(),
        serde_json::to_string(&request)
            .map_err(|error| error.to_string())?
            .into(),
    ];
    let removed = [
        "ACTIONS_RUNTIME_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "ACTIONS_RUNNER_INPUT_TOKEN",
    ]
    .map(str::to_owned);
    let result = crate::startup_process::run_strict_with_removed(
        executable.as_os_str(),
        &arguments,
        &working,
        &BTreeMap::new(),
        &removed,
        Duration::from_secs(seconds - 10),
        &working.join("owner-helper.log"),
    );
    let after = crate::host_cpu::power_snapshot();
    let (now, frequency) = ticks()?;
    let known = matching(&receipt.power_before, &after);
    let succeeded = result.as_ref().is_ok_and(|output| output.code == 0);
    if !succeeded {
        failure_diagnostic(&evidence, &working, &result)?;
    }
    save(
        &evidence.join("owner-budget.json"),
        &json!({"kind":"runner-x2-cpu-owner-budget","original_start_tick":receipt.started_tick,"original_deadline_tick":receipt.deadline_tick,"work_deadline_tick":deadline,"work_budget_seconds":seconds,"retention_reserve_seconds":RETENTION_RESERVE,"completed_tick":now,"clock_frequency":frequency,"power_before":receipt.power_before,"power_after":after,"power_endpoints_match":known,"worker_completed":succeeded,"continuous_power_stability_proven":false,"post_upload_acceptance":"pending","promotion_qualified":false}),
        true,
    )?;
    annotate_report(&evidence, &receipt, after, known)?;
    let cleanup = crate::release_tool::run(
        &json!({"operation":"cleanup","path":working,"parentDirectory":evidence,"leafName":"working","timeoutMilliseconds":5000}),
    );
    let upload = upload_window(&receipt)?;
    cleanup?;
    let output = result?;
    require(
        output.code == 0,
        "Native X2 worker failed; selected evidence retained.",
    )?;
    require(
        known && upload,
        "X2 power changed or original upload reserve no longer fits.",
    )?;
    serde_json::from_str(&output.stdout).map_err(|error| error.to_string())
}

fn failure_diagnostic(
    evidence: &Path,
    working: &Path,
    result: &Result<crate::startup_process::ProcessOutput>,
) -> Result<()> {
    let mut diagnostic = match &result {
        Ok(output) => format!(
            "Native X2 helper exit: {}\nstdout empty: {}\nstderr empty: {}\n",
            output.code,
            output.stdout.trim().is_empty(),
            output.stderr.trim().is_empty()
        ),
        Err(reason) => format!("Native X2 helper failed: {reason}\n"),
    };
    let path = working.join("owner-helper.log");
    if path.is_file() {
        let file = crate::build_tools::input_file(&path)?;
        let mut prefix = Vec::new();
        file.take(8192)
            .read_to_end(&mut prefix)
            .map_err(|error| error.to_string())?;
        let text = String::from_utf8_lossy(&prefix);
        if !text.trim().is_empty() {
            diagnostic.push_str(&text);
        }
    }
    fs::write(evidence.join("owner-helper-failure.txt"), diagnostic)
        .map_err(|error| error.to_string())?;
    Ok(())
}
fn annotate_report(evidence: &Path, receipt: &Start, after: Value, known: bool) -> Result<()> {
    let report_path = evidence.join("screen/cpu-evaluation.json");
    if report_path.exists() {
        let mut report = read(&report_path)?;
        report["OwnerPowerBeforeSetup"] = receipt.power_before.clone();
        report["OwnerPowerAfterWork"] = after;
        report["OwnerDeadlineTick"] = receipt.deadline_tick.into();
        if !known && let Some(comparisons) = report["Comparisons"].as_array_mut() {
            for comparison in comparisons {
                comparison["Comparable"] = false.into();
                for field in [
                    "CandidateOverBaselinePromptMilliseconds",
                    "CandidateOverBaselineDecodeMilliseconds",
                    "CandidateOverBaselineWallMilliseconds",
                ] {
                    comparison[field] = Value::Null;
                }
            }
        }
        save(&report_path, &report, false)?;
        append(
            "GITHUB_STEP_SUMMARY",
            &serde_json::to_string(&report["Comparisons"]).map_err(|error| error.to_string())?,
        )?;
    }
    Ok(())
}
fn upload_window(receipt: &Start) -> Result<bool> {
    let (now, frequency) = ticks()?;
    let upload = now
        .checked_add(RETENTION_RESERVE * frequency)
        .is_some_and(|end| end <= receipt.deadline_tick);
    append("GITHUB_ENV", &format!("BONSAI_X2_UPLOAD_ALLOWED={upload}"))?;
    Ok(upload)
}

pub(super) fn run(request: Request) -> Result<Value> {
    match request {
        Request::Create {
            output_path,
            run_id,
            runner_revision,
        } => create(&output_path, &run_id, &runner_revision),
        Request::Admit { start_receipt } => admit(&start_receipt),
        Request::Capture {
            start_receipt,
            data_directory,
            cim_processors,
            observe_output_progress,
            attribution_input,
        } => capture(
            &start_receipt,
            &data_directory,
            &cim_processors,
            super::attribution(observe_output_progress, attribution_input.as_deref())?,
        ),
        Request::Finalize { start_receipt } => finalize(&start_receipt),
    }
}
