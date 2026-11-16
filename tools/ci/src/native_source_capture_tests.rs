//! Synthetic execution injection; these controls execute no native compiler.

use super::*;
use sha2::{Digest as _, Sha256};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

struct Scratch(crate::impact::workspace::ScratchDir);

impl Scratch {
    fn new() -> std::result::Result<Self, Box<dyn std::error::Error>> {
        Ok(Self(crate::impact::workspace::ScratchDir::new(
            "Bonsai-NativeExecution",
        )?))
    }

    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }

    fn request(&self) -> std::result::Result<Value, Box<dyn std::error::Error>> {
        fs::write(
            self.path("clang-cl.exe"),
            b"synthetic compiler identity; never executed",
        )?;
        fs::write(self.path("source.c"), b"int x = 7;\n")?;
        let (bytes, sha256) = digest(Path::new(&self.path("clang-cl.exe")), MAX_FILE)?;
        let output = self.0.join("out");
        let path = |name: &str| output.join(name).to_string_lossy().into_owned();
        Ok(json!({
            "schemaVersion":1, "kind":"explicit-clang-native-source-execution",
            "sourceRevision":"a".repeat(40), "target":"aarch64-pc-windows-msvc",
            "configuration":"synthetic execution injection; no compiler execution",
            "compiler":{"path":self.path("clang-cl.exe"), "bytes":bytes, "sha256":sha256},
            "outputDirectory":output, "environment":{},
            "clock":{"startedUnixMilliseconds":1000,"startedTick":1000,"tickFrequency":1000,"maximumSeconds":300,"estimatedTotalSeconds":240},
            "units":[{
                "source":self.path("source.c"), "workingDirectory":self.0.path(),
                "arguments":[self.path("clang-cl.exe"),"--target=aarch64-pc-windows-msvc","/MD","/c",
                    format!("/Fo{}",path("source.obj")), "/clang:-MD",
                    format!("/clang:-MF{}",path("source.d")),
                    format!("/clang:-MT{}",path("source.obj")), self.path("source.c")],
                "preprocessingArguments":[self.path("clang-cl.exe"),"--target=aarch64-pc-windows-msvc","/MD","/P",
                    format!("/Fi{}",path("source.i")),self.path("source.c")],
                "dependencies":path("source.d"),"preprocessed":path("source.i"),"object":path("source.obj"),
                "inputs":[{"path":self.path("source.c"),"origin":"source"}]
            }]
        }))
    }
}

fn fixture_execute(
    value: Value,
    invoke: impl FnMut(
        &CommandUnit,
        &[String],
        &str,
        Instant,
        &BTreeMap<String, String>,
        &Path,
    ) -> Result<u32>,
) -> Result<String> {
    let bytes = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
    execute(
        serde_json::from_value(value).map_err(|error| error.to_string())?,
        &bytes,
        &hex(&Sha256::digest(&bytes)),
        Instant::now() + Duration::from_secs(30),
        invoke,
    )
}

fn synthesize(
    unit: &CommandUnit,
    _: &[String],
    name: &str,
    _: Instant,
    _: &BTreeMap<String, String>,
    _: &Path,
) -> Result<u32> {
    if name.ends_with("compile") {
        fs::write(&unit.object, b"original selected object").map_err(|error| error.to_string())?;
        let escape = |path: &str| {
            path.strip_prefix("\\\\?\\")
                .unwrap_or(path)
                .replace('\\', "/")
                .replace('$', "$$")
                .replace(' ', "\\ ")
        };
        fs::write(
            &unit.dependencies,
            format!("{}: {}\n", escape(&unit.object), escape(&unit.source)),
        )
        .map_err(|error| error.to_string())?;
    } else {
        fs::write(&unit.preprocessed, b"int x = 7;\n").map_err(|error| error.to_string())?;
    }
    Ok(0)
}

#[test]
fn native_source_capture_preserves_original_argv_and_verifiable_outputs() -> TestResult {
    let scratch = Scratch::new()?;
    let request = scratch.request()?;
    let original = request["units"][0]["arguments"].clone();
    let mut invocations = Vec::new();
    let report: Value = serde_json::from_str(&fixture_execute(
        request,
        |unit, arguments, name, deadline, environment, root| {
            invocations.push(arguments.to_vec());
            synthesize(unit, arguments, name, deadline, environment, root)
        },
    )?)?;
    assert_eq!(invocations.len(), 2);
    assert_eq!(json!(invocations[0]), original);
    let database: Value =
        serde_json::from_slice(&fs::read(scratch.path("out/compile_commands.json"))?)?;
    assert_eq!(database[0]["arguments"], original);
    let manifest = fs::read(scratch.path("out/manifest.json"))?;
    let hash = report["manifestSha256"]
        .as_str()
        .ok_or("Missing manifest hash")?;
    assert_eq!(hex(&Sha256::digest(&manifest)), hash);
    let replay: Value = serde_json::from_str(&super::super::super::verify_manifest_bytes(
        &manifest, hash,
    )?)?;
    assert_eq!(replay["passed"], true);
    assert_eq!(report["nativeRetirementQualified"], false);
    assert_eq!(report["powerBefore"], Value::Null);
    Ok(())
}

#[test]
fn native_source_capture_failure_retains_original_status_and_stops_selection() -> TestResult {
    let scratch = Scratch::new()?;
    let request = scratch.request()?;
    let mut count = 0;
    assert!(
        fixture_execute(request, |_, _, _, _, _, _| {
            count += 1;
            Ok(37)
        })
        .is_err()
    );
    assert_eq!(count, 1);
    let report: Value = serde_json::from_slice(&fs::read(scratch.path("out/capture.json"))?)?;
    assert_eq!(report["passed"], false);
    assert_eq!(report["commands"][0]["exitCode"], 37);
    assert_eq!(
        report["commands"]
            .as_array()
            .ok_or("Missing command list")?
            .len(),
        2
    );
    assert_eq!(report["commands"][1]["disposition"], "unattempted");
    assert!(!Path::new(&scratch.path("out/manifest.json")).exists());
    assert!(!Path::new(&scratch.path("out/source.obj")).exists());
    assert!(!Path::new(&scratch.path("out/source.i")).exists());
    Ok(())
}

#[test]
fn native_source_capture_rejects_changed_inputs_and_preflight_configuration() -> TestResult {
    let scratch = Scratch::new()?;
    let request = scratch.request()?;
    assert!(
        fixture_execute(
            request,
            |unit, arguments, name, deadline, environment, root| {
                synthesize(unit, arguments, name, deadline, environment, root)?;
                fs::write(&unit.source, b"int x = 8;\n").map_err(|error| error.to_string())?;
                Ok(0)
            }
        )
        .is_err()
    );
    assert!(!Path::new(&scratch.path("out/manifest.json")).exists());
    let scratch = Scratch::new()?;
    let mut request = scratch.request()?;
    request["units"][0]["preprocessingArguments"][2] = json!("/MT");
    let mut called = false;
    assert!(
        fixture_execute(request, |_, _, _, _, _, _| {
            called = true;
            Ok(0)
        })
        .is_err()
    );
    assert!(!called);
    Ok(())
}

#[test]
fn native_source_capture_refuses_existing_evidence_and_unreviewed_request() -> TestResult {
    let scratch = Scratch::new()?;
    let request = scratch.request()?;
    fs::create_dir(scratch.path("out"))?;
    fs::write(scratch.path("out/owner.txt"), b"original evidence")?;
    assert!(
        fixture_execute(request.clone(), |_, _, _, _, _, _| Err(
            "must not execute".into()
        ))
        .is_err()
    );
    assert_eq!(
        fs::read(scratch.path("out/owner.txt"))?,
        b"original evidence"
    );
    let bytes = serde_json::to_vec(&request)?;
    fs::write(scratch.path("request.json"), &bytes)?;
    assert!(
        run(&[
            "native-closure-capture".into(),
            "--request".into(),
            scratch.path("request.json").into(),
            "--expected-sha256".into(),
            "0".repeat(64).into(),
        ])
        .is_err()
    );
    let mut changed = request;
    changed["units"][0]["compilerExitCode"] = json!(0);
    assert!(serde_json::from_value::<Execution>(changed).is_err());
    Ok(())
}

#[test]
fn native_source_capture_missing_outputs_and_foreign_cleanup_reject() -> TestResult {
    let scratch = Scratch::new()?;
    let request = scratch.request()?;
    assert!(fixture_execute(request, |_, _, _, _, _, _| Ok(0)).is_err());
    assert!(!Path::new(&scratch.path("out/manifest.json")).exists());
    let scratch = Scratch::new()?;
    let mut request = scratch.request()?;
    let foreign = scratch.path("owner-empty.txt");
    fs::write(&foreign, b"")?;
    request["units"][0]["object"] = json!(foreign);
    assert!(fixture_execute(request, |_, _, _, _, _, _| Err("must not execute".into())).is_err());
    assert!(Path::new(&foreign).is_file());
    let scratch = Scratch::new()?;
    let request = scratch.request()?;
    assert!(
        fixture_execute(request, |_, _, _, _, _, _| Err(
            "synthetic start/teardown failure".into()
        ))
        .is_err()
    );
    let report: Value = serde_json::from_slice(&fs::read(scratch.path("out/capture.json"))?)?;
    assert_eq!(report["commands"][0]["startObserved"], Value::Null);
    assert_eq!(report["commands"][1]["disposition"], "unattempted");
    assert_eq!(report["completedCompilerInvocationObserved"], false);
    Ok(())
}

#[test]
fn native_source_capture_original_clock_charges_setup_and_fractional_expiry() -> TestResult {
    let clock = Clock {
        started_unix_milliseconds: 1000,
        started_tick: 1000,
        tick_frequency: 1000,
        maximum_seconds: 300,
        estimated_total_seconds: 240,
        budget_reason: None,
    };
    assert_eq!(
        remaining(&clock, 2000, 31001, 1000)?,
        Duration::from_millis(239999)
    );
    assert!(remaining(&clock, 271000, 2000, 1000).is_err());
    assert!(remaining(&clock, 2000, 271001, 1000).is_err());
    assert!(remaining(&clock, 999, 2000, 1000).is_err());
    assert!(remaining(&clock, 2000, 999, 1000).is_err());
    assert!(remaining(&clock, 2000, 2000, 999).is_err());
    Ok(())
}

#[cfg(windows)]
#[test]
fn native_source_capture_supervision_preserves_failed_root_and_expired_start() -> TestResult {
    let scratch = Scratch::new()?;
    let request: Execution = serde_json::from_value(scratch.request()?)?;
    let executable = std::env::var_os("SystemRoot")
        .map(PathBuf::from)
        .ok_or("Missing SystemRoot")?
        .join("System32/where.exe");
    let arguments = vec![
        executable.to_string_lossy().into_owned(),
        "__bonsai_capture_missing_executable_fixture__".into(),
    ];
    assert_eq!(
        invoke(
            &request.units[0],
            &arguments,
            "failed-root",
            Instant::now() + Duration::from_secs(10),
            &BTreeMap::new(),
            &scratch.0
        )?,
        1
    );
    assert!(!scratch.0.join("failed-root.stdout").exists());
    assert!(scratch.0.join("failed-root.stderr").is_file());
    assert!(
        invoke(
            &request.units[0],
            &arguments,
            "expired",
            Instant::now(),
            &BTreeMap::new(),
            &scratch.0
        )
        .is_err()
    );
    assert!(!scratch.0.join("expired.stdout").exists());
    Ok(())
}

#[test]
fn native_source_capture_deadline_reaches_intake_collector_and_verifier() -> TestResult {
    let scratch = Scratch::new()?;
    let missing = scratch.0.join("must-not-open");
    let expired = Instant::now();
    let deadline_error =
        |result: Result<()>| result.is_err_and(|error| error.contains("original work deadline"));
    assert!(deadline_error(
        digest_until(&missing, MAX_FILE, Some(expired)).map(|_| ())
    ));
    assert!(deadline_error(
        crate::command::native_source_inventory::read_bounded_until(
            &missing,
            MAX_DOCUMENT,
            Some(expired)
        )
        .map(|_| ())
    ));
    let mut intake = Intake {
        deadline: Some(expired),
        ..Intake::default()
    };
    assert!(deadline_error(
        intake
            .identity(missing.to_string_lossy().into_owned(), MAX_FILE)
            .map(|_| ())
    ));
    let request: Execution = serde_json::from_value(scratch.request()?)?;
    assert!(deadline_error(preflight(&request, expired).map(|_| ())));
    assert!(deadline_error(
        assemble_until(
            Request {
                schema_version: 1,
                kind: "selected-clang-native-source-collection".into(),
                source_revision: "a".repeat(40),
                target: "aarch64-pc-windows-msvc".into(),
                configuration: "expired before any original file lookup".into(),
                compiler: missing.to_string_lossy().into_owned(),
                compilation_database: String::new(),
                units: Vec::new(),
            },
            Some(expired)
        )
        .map(|_| ())
    ));
    assert!(deadline_error(
        super::super::super::verify_manifest_bytes_until(
            b"not-json",
            &"0".repeat(64),
            Some(expired)
        )
        .map(|_| ())
    ));
    Ok(())
}

#[test]
fn native_source_capture_observed_invalid_retention_cannot_keep_success() -> TestResult {
    for (name, bytes) in [
        ("empty.obj", Some(b"".as_slice())),
        ("large.obj", Some(b"oversized".as_slice())),
        ("missing.obj", None),
    ] {
        let scratch = Scratch::new()?;
        let path = scratch.path(name);
        if let Some(bytes) = bytes {
            fs::write(&path, bytes)?;
        }
        let mut report = json!({"passed":true,"discardedOutputs":[]});
        let mut result = Ok(());
        retain_outputs(
            &scratch.0,
            vec![(path.clone(), 4)],
            &mut report,
            &mut result,
        )?;
        assert!(result.is_err());
        assert_eq!(report["passed"], false);
        assert!(!Path::new(&path).exists());
    }
    Ok(())
}
