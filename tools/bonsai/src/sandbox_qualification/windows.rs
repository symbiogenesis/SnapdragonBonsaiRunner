//! Hosted preparation and one original-budget native sandbox fixture sequence.

use super::{
    FILES, TOOLCHAIN, Workload, inventory, package_identity, remaining_fit, selected_artifacts,
    text,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs::{self, File},
    path::{Path, PathBuf},
    time::Duration,
};

fn env(name: &str) -> crate::Result<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("Missing sandbox environment: {name}"))
}
fn external(path: &Path) -> crate::Result<()> {
    bonsai_artifacts::validate_external_path(path).map_err(|error| error.to_string())
}
fn path(value: &Value, key: &str) -> crate::Result<PathBuf> {
    let path = PathBuf::from(text(value, key)?);
    external(&path)?;
    Ok(path)
}
fn clocks() -> crate::Result<(u64, u64)> {
    bonsai_windows_process::monotonic_ticks().map_err(|error| error.to_string())
}

fn write_json(path: &Path, value: &Value) -> crate::Result<()> {
    external(path)?;
    crate::bounded_input::write_pretty_json_atomic(path, value)
}

fn ci_guard(hosted: bool) -> crate::Result<()> {
    for (key, wanted) in [
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REPOSITORY", "symbiogenesis/SwiftBonsai2Runner"),
        ("GITHUB_REF", "refs/heads/main"),
        ("RUNNER_OS", "Windows"),
        ("RUNNER_ARCH", "ARM64"),
    ] {
        if env(key)? != wanted {
            return Err(
                "Sandbox qualification requires its exact main-only Windows ARM64 lane.".into(),
            );
        }
    }
    if std::env::consts::ARCH != "aarch64"
        || (hosted && env("RUNNER_ENVIRONMENT")? != "github-hosted")
    {
        return Err("Sandbox preparation requires native hosted Windows ARM64.".into());
    }
    Ok(())
}

fn digest(path: &Path) -> crate::Result<(u64, String)> {
    let mut file = crate::bounded_input::open_locked_file_sequential(path)
        .map_err(|error| error.to_string())?;
    let bytes = file.metadata().map_err(|error| error.to_string())?.len();
    if bytes == 0 || bytes > 134_217_728 {
        return Err("Sandbox packaged file exceeds its bound.".into());
    }
    let (_, hash) = crate::identity::hash_reader_checked(
        &mut file,
        Some(bytes),
        None,
        &mut crate::startup_cancel::checkpoint,
    )?;
    Ok((bytes, hash))
}

fn compile(
    repository: &Path,
    target: &Path,
    work: &Path,
    package: &str,
    test: &str,
    features: bool,
) -> crate::Result<String> {
    let mut arguments = vec![
        format!("+{TOOLCHAIN}").into(),
        "test".into(),
        "--locked".into(),
        "--release".into(),
        "--no-run".into(),
        "--message-format=json".into(),
        "--manifest-path".into(),
        repository.join("Cargo.toml").into_os_string(),
        "--target".into(),
        "aarch64-pc-windows-msvc".into(),
        "--package".into(),
        package.into(),
        "--test".into(),
        test.into(),
    ];
    if features {
        arguments.extend(["--features".into(), "test-probe".into()]);
    }
    let result = crate::startup_process::run(
        std::ffi::OsStr::new("cargo.exe"),
        &arguments,
        work,
        &BTreeMap::from([
            (
                "CARGO_TARGET_DIR".into(),
                target.to_string_lossy().into_owned(),
            ),
            ("CARGO_BUILD_FINGERPRINT".into(), "content".into()),
            ("CARGO_UNSTABLE_CHECKSUM_FRESHNESS".into(), "true".into()),
        ]),
        Duration::from_mins(15),
        &work.join(format!("build-{test}.log")),
    )?;
    if result.code != 0 {
        return Err(format!(
            "Sandbox hosted {test} build failed with exit code {}.",
            result.code
        ));
    }
    Ok(result.stdout)
}

fn clean_capture(work: &Path, stem: &str) -> crate::Result<()> {
    for entry in fs::read_dir(work).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if name.starts_with(&format!("{stem}."))
            && path.extension().is_some_and(|extension| {
                extension == "stdout" || extension == "stderr" || extension == "log"
            })
        {
            fs::remove_file(path).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn locked_mxc(repository: &Path, evidence: &Path) -> crate::Result<()> {
    let metadata = crate::startup_process::run(
        std::ffi::OsStr::new("cargo.exe"),
        &[
            format!("+{TOOLCHAIN}").into(),
            "metadata".into(),
            "--locked".into(),
            "--format-version".into(),
            "1".into(),
            "--filter-platform".into(),
            "aarch64-pc-windows-msvc".into(),
            "--manifest-path".into(),
            repository.join("Cargo.toml").into_os_string(),
        ],
        evidence,
        &BTreeMap::new(),
        Duration::from_mins(2),
        &evidence.join("metadata.log"),
    )?;
    if metadata.code != 0 {
        return Err("Sandbox locked dependency identity could not be read.".into());
    }
    let metadata: Value =
        serde_json::from_str(&metadata.stdout).map_err(|error| error.to_string())?;
    let versions = metadata["packages"]
        .as_array()
        .ok_or("Sandbox metadata package list missing.")?
        .iter()
        .filter(|package| package["name"] == "mxc-sdk")
        .map(|package| package["version"].clone())
        .collect::<Vec<_>>();
    if versions.as_slice() != [json!("1.0.1")] {
        return Err("Sandbox locked MXC dependency identity differs.".into());
    }
    clean_capture(evidence, "metadata")
}

fn copy_supporting_tools(output: &Path) -> crate::Result<()> {
    fs::copy(
        std::env::current_exe().map_err(|error| error.to_string())?,
        output.join("bonsai-tools.exe"),
    )
    .map_err(|error| error.to_string())?;
    fs::copy(
        PathBuf::from(env("BONSAI_CI_PATH")?),
        output.join("bonsai-ci.exe"),
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

fn prune_blank_logs(evidence: &Path) -> crate::Result<()> {
    for entry in fs::read_dir(evidence).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.extension().is_some_and(|extension| extension == "log") {
            let raw = crate::bounded_input::read(
                File::open(&path).map_err(|error| error.to_string())?,
                64 * 1024 * 1024,
            )
            .map_err(|error| {
                error.into_message("Sandbox diagnostic log exceeds its byte bound.")
            })?;
            if std::str::from_utf8(&raw).is_ok_and(|text| text.trim().is_empty()) {
                fs::remove_file(path).map_err(|error| error.to_string())?;
            }
        }
    }
    Ok(())
}

fn prepare(request: &Value) -> crate::Result<Value> {
    ci_guard(true)?;
    let workload = Workload::select(request.get("workload"))?;
    let repository = PathBuf::from(text(request, "repositoryRoot")?);
    let target = path(request, "buildDirectory")?;
    let output = path(request, "outputDirectory")?;
    let evidence = path(request, "evidenceDirectory")?;
    if !repository.is_absolute()
        || !repository.join("Cargo.toml").is_file()
        || output.exists()
        || evidence.exists()
    {
        return Err(
            "Sandbox preparation needs source and fresh external output/evidence directories."
                .into(),
        );
    }
    fs::create_dir_all(&target)
        .and_then(|()| fs::create_dir(&output))
        .and_then(|()| fs::create_dir(&evidence))
        .map_err(|error| error.to_string())?;
    let mut report = json!({"SchemaVersion":1,"Kind":"native-sandbox-hosted-preparation","Revision":env("GITHUB_SHA")?,"RunId":env("GITHUB_RUN_ID")?,"RunAttempt":env("GITHUB_RUN_ATTEMPT")?,"Workload":workload.name(),"Passed":false,"ModelExecuted":false,"OwnerDeviceExecuted":false,"Stage":"compile-ownership"});
    let result = (|| -> crate::Result<Value> {
        let ownership = compile(
            &repository,
            &target,
            &evidence,
            "bonsai-windows-process",
            "ownership",
            true,
        )?;
        let selected = selected_artifacts(
            &ownership,
            &[("ownership", "test"), ("bonsai-process-test-probe", "bin")],
        )?;
        fs::copy(&selected[0], output.join("sandbox-ownership.exe"))
            .and_then(|_| fs::copy(&selected[1], output.join("bonsai-process-test-probe.exe")))
            .map_err(|error| error.to_string())?;
        clean_capture(&evidence, "build-ownership")?;
        report["Stage"] = "compile-contained".into();
        let contained = compile(
            &repository,
            &target,
            &evidence,
            "bonsai-artifacts",
            "contained",
            false,
        )?;
        let selected = selected_artifacts(&contained, &[("contained", "test")])?;
        fs::copy(&selected[0], output.join("sandbox-contained.exe"))
            .map_err(|error| error.to_string())?;
        clean_capture(&evidence, "build-contained")?;
        copy_supporting_tools(&output)?;
        locked_mxc(&repository, &evidence)?;
        for binary in ["sandbox-ownership.exe", "sandbox-contained.exe"] {
            let listed = crate::startup_process::run(
                output.join(binary).as_os_str(),
                &["--list".into(), "--ignored".into()],
                &evidence,
                &BTreeMap::new(),
                Duration::from_secs(30),
                &evidence.join(format!("{binary}.log")),
            )?;
            if listed.code != 0 {
                return Err("Sandbox hosted case inventory failed.".into());
            }
            inventory(binary, &listed.stdout, workload)?;
            clean_capture(&evidence, binary)?;
        }
        let mut files = Vec::new();
        for name in FILES {
            let path = output.join(name);
            crate::startup_cache::assert_arm64(&path)?;
            let (bytes, hash) = digest(&path)?;
            files.push(json!({"Name":name,"Bytes":bytes,"Sha256":hash}));
        }
        let receipt = json!({"SchemaVersion":1,"Kind":"native-sandbox-prebuilt-package","Revision":env("GITHUB_SHA")?,"RunId":env("GITHUB_RUN_ID")?,"RunAttempt":env("GITHUB_RUN_ATTEMPT")?,"Toolchain":TOOLCHAIN,"MxcVersion":"1.0.1","Files":files,"Workload":workload.name(),"Cases":workload.cases(),"ModelExecuted":false});
        package_identity(
            &receipt,
            &env("GITHUB_SHA")?,
            &env("GITHUB_RUN_ID")?,
            &env("GITHUB_RUN_ATTEMPT")?,
            workload,
        )?;
        write_json(&output.join("receipt.json"), &receipt)?;
        report["Stage"] = "completed".into();
        report["Passed"] = true.into();
        Ok(json!({"PackageDirectory":output,"PreparedOnly":true,"ModelExecuted":false}))
    })();
    if let Err(error) = &result {
        report["FailureClass"] = "native-preparation-failed".into();
        report["Failure"] = error.clone().into();
    }
    prune_blank_logs(&evidence)?;
    write_json(&evidence.join("preparation.json"), &report)?;
    result
}

fn hold_package(
    root: &Path,
    deadline: u64,
    frequency: u64,
    workload: Workload,
) -> crate::Result<(Value, File, Vec<bonsai_artifacts::VerifiedFile>)> {
    if std::env::current_exe().map_err(|error| error.to_string())? != root.join("bonsai-tools.exe")
    {
        return Err("Sandbox execution must use its exact packaged native helper.".into());
    }
    let mut receipt = bonsai_artifacts::open_locked_file(&root.join("receipt.json"))
        .map_err(|error| error.to_string())?;
    let raw = crate::bounded_input::read(&mut receipt, 65_536)
        .map_err(|error| error.into_message("Sandbox package receipt exceeds its bound."))?;
    let value: Value = serde_json::from_slice(&raw).map_err(|error| error.to_string())?;
    package_identity(
        &value,
        &env("GITHUB_SHA")?,
        &env("GITHUB_RUN_ID")?,
        &env("GITHUB_RUN_ATTEMPT")?,
        workload,
    )?;
    let mut expected = FILES.iter().copied().collect::<BTreeSet<_>>();
    expected.insert("receipt.json");
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(root).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let metadata = fs::symlink_metadata(entry.path()).map_err(|error| error.to_string())?;
        if !metadata.is_file() || bonsai_artifacts::is_redirect(&metadata) {
            return Err("Sandbox package entry is redirected or nested.".into());
        }
        actual.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| "Sandbox package name must be Unicode.")?,
        );
    }
    if actual.iter().map(String::as_str).collect::<BTreeSet<_>>() != expected {
        return Err("Sandbox package member inventory differs.".into());
    }
    let mut files = Vec::new();
    for file in value["Files"].as_array().ok_or("Sandbox files missing.")? {
        files.push(
            bonsai_artifacts::VerifiedFile::verify_with(
                &root.join(text(file, "Name")?),
                file["Bytes"]
                    .as_u64()
                    .ok_or("Sandbox file length invalid.")?,
                text(file, "Sha256")?,
                || {
                    check_deadline(deadline, frequency)
                        .map_err(|_| bonsai_artifacts::ArtifactError::Cancelled)
                },
            )
            .map_err(|error| error.to_string())?,
        );
    }
    Ok((value, receipt, files))
}

fn check_deadline(deadline: u64, frequency: u64) -> crate::Result<()> {
    crate::startup_cancel::checkpoint()?;
    let (now, current) = clocks()?;
    if current != frequency || now >= deadline {
        return Err("Sandbox original aggregate allowance expired.".into());
    }
    Ok(())
}
fn child(
    root: &Path,
    binary: &str,
    arguments: Vec<OsString>,
    work: &Path,
    deadline: u64,
    frequency: u64,
) -> crate::Result<(i32, String, String)> {
    crate::acceptance_driver::supervised_tool(
        &root.join(binary),
        arguments,
        work,
        BTreeMap::from([
            (
                "BONSAI_SANDBOX_PROBE_PATH".into(),
                root.join("bonsai-process-test-probe.exe")
                    .to_string_lossy()
                    .into_owned(),
            ),
            ("TEMP".into(), work.to_string_lossy().into_owned()),
            ("TMP".into(), work.to_string_lossy().into_owned()),
        ]),
        deadline,
        frequency,
    )
}

fn budget(report: &mut Value) -> crate::Result<(u64, u64, u64)> {
    let (now, frequency) = clocks()?;
    let limits = crate::native_acceptance_budget::run(
        json!({"jobStartedUtc":env("BONSAI_SANDBOX_JOB_STARTED_UTC")?,"jobStartedTick":env("BONSAI_SANDBOX_JOB_STARTED_TICK")?.parse::<u64>().map_err(|_|"Sandbox original monotonic start is invalid.")?,"nowUtc":bonsai_windows_process::utc_now(),"nowTick":now,"tickFrequency":frequency,"runId":env("GITHUB_RUN_ID")?,"revision":env("GITHUB_SHA")?,"ownerDeadline":{"RunId":std::env::var("BONSAI_NATIVE_OWNER_RUN_ID").unwrap_or_default(),"Revision":std::env::var("BONSAI_NATIVE_OWNER_REVISION").unwrap_or_default(),"StartedTick":std::env::var("BONSAI_NATIVE_OWNER_STARTED_TICK").unwrap_or_default(),"StartedUtc":std::env::var("BONSAI_NATIVE_OWNER_STARTED_UTC").unwrap_or_default(),"WorkSeconds":std::env::var("BONSAI_NATIVE_OWNER_WORK_SECONDS").unwrap_or_default()}}),
    )?;
    let deadline = |key: &str| {
        limits[key]
            .as_u64()
            .and_then(|seconds| seconds.checked_mul(frequency))
            .and_then(|ticks| now.checked_add(ticks))
            .ok_or_else(|| "Sandbox process deadline overflow.".to_owned())
    };
    let hard = deadline("HardLimitSeconds")?;
    let work = deadline("CaseBudgetSeconds")?;
    report["Budget"] = limits;
    Ok((hard, work, frequency))
}

struct Execution<'a> {
    root: &'a Path,
    evidence: &'a Path,
    work: &'a Path,
    deadline: u64,
    work_deadline: u64,
    frequency: u64,
}

fn execute_case(
    execution: &Execution<'_>,
    binary: &str,
    name: &str,
    record: &mut Value,
) -> crate::Result<()> {
    let output = child(
        execution.root,
        binary,
        vec![
            name.into(),
            "--exact".into(),
            "--ignored".into(),
            "--nocapture".into(),
        ],
        execution.work,
        execution.work_deadline,
        execution.frequency,
    )?;
    record["ExitCode"] = output.0.into();
    let summaries = output
        .1
        .lines()
        .filter(|line| line.starts_with("test result:"))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let request = json!({"operation":"test-output","output":summaries,"exitCode":output.0,"expectedPassed":1});
    let proof = child(
        execution.root,
        "bonsai-ci.exe",
        vec![
            "native-acceptance-guard".into(),
            "--json".into(),
            serde_json::to_string(&request)
                .map_err(|error| error.to_string())?
                .into(),
        ],
        execution.work,
        execution.deadline,
        execution.frequency,
    )?;
    if proof.0 != 0 {
        let diagnostic = output.1 + &output.2;
        if !diagnostic.trim().is_empty() {
            fs::write(execution.evidence.join(format!("{name}.log")), diagnostic)
                .map_err(|error| error.to_string())?;
        }
        return Err(format!(
            "Native sandbox case failed or produced no exact positive test result: {name}"
        ));
    }
    Ok(())
}

fn native_host(report: &mut Value, target: &str) -> crate::Result<()> {
    let host = bonsai_windows_process::native_build_host().map_err(|error| error.to_string())?;
    report["Cpu"] = host.processor_name.clone().into();
    report["WindowsVersion"] = host.windows_version.into();
    report["Architecture"] = "Arm64".into();
    if target == "snapdragon"
        && bonsai_cpu_policy::hardware::SnapdragonFamily::from_processor_name(&host.processor_name)
            .is_none()
    {
        return Err(
            "Sandbox owner requires actual Snapdragon X Elite or X2 Elite Extreme identity.".into(),
        );
    }
    Ok(())
}

fn qualify(request: &Value) -> crate::Result<Value> {
    ci_guard(false)?;
    let workload = Workload::select(request.get("workload"))?;
    let root = path(request, "packageDirectory")?;
    let evidence = path(request, "evidenceDirectory")?;
    let work = path(request, "workingDirectory")?;
    if evidence.exists() || work.exists() {
        return Err(
            "Sandbox qualification needs fresh private working/evidence directories.".into(),
        );
    }
    fs::create_dir(&evidence)
        .and_then(|()| fs::create_dir(&work))
        .map_err(|error| error.to_string())?;
    let target = text(request, "target")?;
    if !["hosted-arm64", "snapdragon"].contains(&target) {
        return Err("Unknown sandbox qualification target.".into());
    }
    if env("RUNNER_ENVIRONMENT")?
        != if target == "hosted-arm64" {
            "github-hosted"
        } else {
            "self-hosted"
        }
    {
        return Err("Sandbox target differs from its admitted runner environment.".into());
    }
    let mut report = json!({"SchemaVersion":2,"Kind":"native-sandbox-prebuilt-qualification","Revision":env("GITHUB_SHA")?,"RunId":env("GITHUB_RUN_ID")?,"RunAttempt":env("GITHUB_RUN_ATTEMPT")?,"Toolchain":TOOLCHAIN,"Target":target,"Workload":workload.name(),"Tests":workload.cases().iter().map(|(_,name)|json!({"Name":name,"ExitCode":null,"Passed":false,"Status":"pending","PowerBefore":null,"PowerAfter":null})).collect::<Vec<_>>(),"Passed":false,"ActualModelQualified":false,"BrokerOnlyTransportQualified":false,"PerformanceMeasured":false,"PowerBefore":crate::host_cpu::power_snapshot(),"PowerAfter":null,"PreBootstrapPower":null,"WholeJobMaximumSeconds":300,"FixedCaseCount":workload.cases().len(),"EstimatedTotalSeconds":workload.estimated_total_seconds(),"Cleanup":{"WorkingDirectoryRemoved":false,"RetainedForDiagnosis":false},"Scope":workload.scope()});
    let result = (|| -> crate::Result<Value> {
        let (deadline, work_deadline, frequency) = budget(&mut report)?;
        let execution = Execution {
            root: &root,
            evidence: &evidence,
            work: &work,
            deadline,
            work_deadline,
            frequency,
        };
        let (receipt, _receipt_handle, _files) =
            hold_package(&root, deadline, frequency, workload)?;
        report["Package"] = receipt.clone();
        report["MxcVersion"] = receipt["MxcVersion"].clone();
        native_host(&mut report, target)?;
        for binary in ["sandbox-ownership.exe", "sandbox-contained.exe"] {
            let listed = child(
                &root,
                binary,
                vec!["--list".into(), "--ignored".into()],
                &work,
                work_deadline,
                frequency,
            )?;
            if listed.0 != 0 {
                return Err("Sandbox prebuilt case inventory failed.".into());
            }
            inventory(binary, &listed.1, workload)?;
        }
        for (index, &(binary, name)) in workload.cases().iter().enumerate() {
            check_deadline(work_deadline, frequency)?;
            remaining_fit(
                clocks()?.0,
                frequency,
                work_deadline,
                workload.cases().len() - index,
                workload,
            )?;
            report["Tests"][index]["Status"] = "running".into();
            report["Tests"][index]["PowerBefore"] = crate::host_cpu::power_snapshot();
            write_json(&evidence.join("qualification.json"), &report)?;
            let outcome = execute_case(&execution, binary, name, &mut report["Tests"][index]);
            report["Tests"][index]["PowerAfter"] = crate::host_cpu::power_snapshot();
            report["Tests"][index]["Passed"] = outcome.is_ok().into();
            report["Tests"][index]["Status"] =
                if outcome.is_ok() { "passed" } else { "failed" }.into();
            write_json(&evidence.join("qualification.json"), &report)?;
            outcome?;
        }
        check_deadline(deadline, frequency)?;
        external(&work)?;
        fs::remove_dir_all(&work).map_err(|error| error.to_string())?;
        report["Cleanup"]["WorkingDirectoryRemoved"] = true.into();
        check_deadline(deadline, frequency)?;
        report["Passed"] = true.into();
        Ok(
            json!({"Passed":true,"QualificationReceipt":evidence.join("qualification.json"),"ActualModelQualified":false,"PerformanceMeasured":false}),
        )
    })();
    report["PowerAfter"] = crate::host_cpu::power_snapshot();
    if let Err(error) = &result {
        report["FailureClass"] = "native-qualification-incomplete".into();
        report["Failure"] = error.clone().into();
        report["Cleanup"]["RetainedForDiagnosis"] = work.exists().into();
        if work.exists() {
            report["Cleanup"]["WorkingDirectory"] = json!(work);
        }
    }
    write_json(&evidence.join("qualification.json"), &report)?;
    result
}

pub(super) fn run(request: &Value) -> crate::Result<Value> {
    match text(request, "operation")? {
        "prepare" => prepare(request),
        "qualify" => qualify(request),
        _ => Err("Unknown native sandbox operation.".into()),
    }
}
