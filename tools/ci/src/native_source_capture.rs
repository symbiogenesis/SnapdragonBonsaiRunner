//! Execute a reviewed, explicit selection; never reconstruct historical commands.

#[cfg(test)]
use super::super::digest;
use super::super::{
    Identity, MAX_DOCUMENT, MAX_FILE, Unit, arguments_valid, checkpoint, configuration,
    digest_until, hex, lower_hex, path_key, read_bounded,
};
use super::{CaptureInput, CaptureUnit, Intake, Request, assemble_until};
use crate::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs::{self, OpenOptions},
    io::Write as _,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const USAGE: &str =
    "Usage: bonsai-ci native-closure-capture --request PATH --expected-sha256 SHA256";
const REMOVED: &[&str] = &[
    "CL",
    "_CL_",
    "CCC_OVERRIDE_OPTIONS",
    "CCC_ADD_ARGS",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "GCC_EXEC_PREFIX",
    "COMPILER_PATH",
    "LIBRARY_PATH",
    "INCLUDE",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
];

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CommandUnit {
    source: String,
    working_directory: String,
    arguments: Vec<String>,
    preprocessing_arguments: Vec<String>,
    dependencies: String,
    preprocessed: String,
    object: String,
    inputs: Vec<CaptureInput>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Clock {
    started_unix_milliseconds: u64,
    started_tick: u64,
    tick_frequency: u64,
    maximum_seconds: u64,
    estimated_total_seconds: u64,
    budget_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Execution {
    schema_version: u32,
    kind: String,
    source_revision: String,
    target: String,
    configuration: String,
    compiler: Identity,
    output_directory: String,
    environment: BTreeMap<String, String>,
    clock: Clock,
    units: Vec<CommandUnit>,
}

fn now_milliseconds() -> Result<u64> {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_millis(),
    )
    .map_err(|error| error.to_string())
}

fn remaining(clock: &Clock, now: u64, tick: u64, frequency: u64) -> Result<Duration> {
    if !(31..=900).contains(&clock.maximum_seconds)
        || clock.tick_frequency == 0
        || frequency != clock.tick_frequency
        || clock.estimated_total_seconds == 0
        || clock.estimated_total_seconds > clock.maximum_seconds
        || (clock.maximum_seconds > 300
            && clock
                .budget_reason
                .as_deref()
                .is_none_or(|reason| reason.trim().is_empty()))
    {
        return Err("Capture requires the original bounded owner clock and frequency.".into());
    }
    let wall = now
        .checked_sub(clock.started_unix_milliseconds)
        .ok_or("Capture owner wall clock is in the future.")?;
    let ticks = tick
        .checked_sub(clock.started_tick)
        .ok_or("Capture owner monotonic clock is in the future.")?;
    // Integer ceilings charge partial elapsed milliseconds and never renew the
    // original clock. Thirty seconds remain for teardown, retention and exit.
    let monotonic = (u128::from(ticks) * 1000).div_ceil(u128::from(frequency));
    let elapsed = u128::from(wall).max(monotonic);
    let work = u128::from(clock.maximum_seconds - 30) * 1000;
    let left = work
        .checked_sub(elapsed)
        .filter(|left| *left > 0)
        .ok_or("Capture original aggregate work deadline expired.")?;
    Ok(Duration::from_millis(
        u64::try_from(left).map_err(|error| error.to_string())?,
    ))
}

fn write_new(path: &Path, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    write_new_bytes(path, &bytes)
}

fn write_new_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    if u64::try_from(bytes.len()).map_err(|error| error.to_string())? > MAX_DOCUMENT {
        return Err("Capture record exceeds its document bound.".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| error.to_string())
}

fn output(path: &str, root: &Path, selected: &mut BTreeSet<String>) -> Result<()> {
    let path = Path::new(path);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("Invalid capture output.")?;
    if path.parent() != Some(root)
        || name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
        || !selected.insert(name.to_ascii_lowercase())
    {
        return Err(
            "Capture outputs must be distinct direct files in its new output directory.".into(),
        );
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    path_key(path.to_str().ok_or("Invalid output encoding.")?)?;
    Ok(())
}

fn input_paths<'a>(input: &'a CaptureInput, depth: usize, paths: &mut Vec<&'a str>) -> Result<()> {
    if depth >= 64 || paths.len() >= 100_000 {
        return Err("Capture input inventory exceeds its depth/count bound.".into());
    }
    paths.push(&input.path);
    if let Some(generation) = &input.generation {
        paths.push(&generation.generator);
        for input in &generation.inputs {
            input_paths(input, depth + 1, paths)?;
        }
    }
    Ok(())
}

fn retain_outputs(
    root: &Path,
    outputs: Vec<(String, u64)>,
    report: &mut Value,
    result: &mut Result<()>,
) -> Result<()> {
    for (path, limit) in outputs {
        if Path::new(&path).parent() != Some(root) {
            continue;
        }
        let observation = fs::metadata(&path);
        let invalid = match &observation {
            Ok(metadata) => !metadata.is_file() || metadata.len() == 0 || metadata.len() > limit,
            Err(_) => true,
        };
        if invalid && result.is_ok() {
            *result = Err(
                "A required original output became missing, empty or oversized before retention."
                    .into(),
            );
        }
        if let Ok(metadata) = observation
            && metadata.is_file()
            && (metadata.len() == 0 || metadata.len() > limit)
        {
            report["discardedOutputs"].as_array_mut().ok_or("Missing artifact discard records.")?.push(json!({"path":path,"bytes":metadata.len(),"reason":"empty-or-oversized-output-without-replay-consumer"}));
            fs::remove_file(path).map_err(|error| error.to_string())?;
        }
    }
    if let Err(error) = result {
        report["passed"] = json!(false);
        report["failure"] = json!(error);
    }
    Ok(())
}

fn preflight(request: &Execution, deadline: Instant) -> Result<Vec<(String, u64, String)>> {
    checkpoint(Some(deadline))?;
    let compiler = &request.compiler.path;
    path_key(compiler)?;
    if digest_until(Path::new(compiler), MAX_FILE, Some(deadline))?
        != (request.compiler.bytes, request.compiler.sha256.clone())
        || !lower_hex(&request.compiler.sha256, 64)
        || !Path::new(compiler)
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                ["clang", "clang.exe", "clang-cl", "clang-cl.exe"]
                    .iter()
                    .any(|supported| name.eq_ignore_ascii_case(supported))
            })
    {
        return Err("Capture compiler differs from its reviewed Clang identity.".into());
    }
    let root = Path::new(&request.output_directory);
    let mut selected = BTreeSet::from([
        "capture.json".into(),
        "compile_commands.json".into(),
        "manifest.json".into(),
        "verification.json".into(),
        "request.json".into(),
    ]);
    let mut paths = vec![compiler.as_str()];
    let mut intake = Intake {
        deadline: Some(deadline),
        ..Intake::default()
    };
    for index in 0..request.units.len() {
        for suffix in ["compile", "preprocess"] {
            selected.insert(format!("{index}-{suffix}.json"));
            for extension in ["stdout", "stderr", "log"] {
                selected.insert(format!("{index}-{suffix}.{extension}"));
            }
        }
    }
    for unit in &request.units {
        path_key(&unit.working_directory)?;
        path_key(&unit.source)?;
        for path in [&unit.dependencies, &unit.preprocessed, &unit.object] {
            output(path, root, &mut selected)?;
        }
        let empty = |path: &str| Identity {
            path: path.into(),
            bytes: 0,
            sha256: "0".repeat(64),
        };
        let validation = Unit {
            source: unit.source.clone(),
            working_directory: unit.working_directory.clone(),
            arguments: unit.arguments.clone(),
            preprocessing_arguments: unit.preprocessing_arguments.clone(),
            compiler_exit_code: 0,
            preprocessor_exit_code: 0,
            dependencies: empty(&unit.dependencies),
            preprocessed: empty(&unit.preprocessed),
            object: empty(&unit.object),
            inputs: intake.inputs(unit.inputs.clone(), 0)?,
        };
        if !arguments_valid(&unit.arguments, compiler)
            || !arguments_valid(&unit.preprocessing_arguments, compiler)
            || configuration(&validation, false)? != configuration(&validation, true)?
        {
            return Err(
                "Capture needs matching explicit compile/preprocessing configuration.".into(),
            );
        }
        paths.push(&unit.source);
        for input in &unit.inputs {
            input_paths(input, 0, &mut paths)?;
        }
    }
    let mut identities = Vec::new();
    let mut seen = BTreeSet::new();
    for path in paths {
        checkpoint(Some(deadline))?;
        let key = path_key(path)?;
        if seen.insert(key) {
            let (bytes, sha256) = digest_until(Path::new(path), MAX_FILE, Some(deadline))?;
            identities.push((path.to_owned(), bytes, sha256));
        }
    }
    Ok(identities)
}

fn execute(
    request: Execution,
    request_bytes: &[u8],
    request_digest: &str,
    deadline: Instant,
    mut invoke: impl FnMut(
        &CommandUnit,
        &[String],
        &str,
        Instant,
        &BTreeMap<String, String>,
        &Path,
    ) -> Result<u32>,
) -> Result<String> {
    if hex(&Sha256::digest(request_bytes)) != request_digest
        || request.schema_version != 1
        || request.kind != "explicit-clang-native-source-execution"
        || request.target != "aarch64-pc-windows-msvc"
        || !lower_hex(&request.source_revision, 40)
        || !(1..=4096).contains(&request.units.len())
        || request.environment.iter().any(|(name, value)| {
            !["INCLUDE", "LIB", "PATH", "TEMP", "TMP", "SystemRoot"].contains(&name.as_str())
                || value.bytes().any(|byte| byte.is_ascii_control())
        })
    {
        return Err("Unsupported explicit native source execution selection.".into());
    }
    let root = PathBuf::from(&request.output_directory);
    if !root.is_absolute() || !request.output_directory.is_ascii() {
        return Err("Capture output directory must be an absolute ASCII path.".into());
    }
    path_key(
        root.parent()
            .and_then(Path::to_str)
            .ok_or("Capture output needs a parent.")?,
    )?;
    fs::create_dir(&root).map_err(|error| error.to_string())?;
    path_key(&request.output_directory)?;
    let outputs = request
        .units
        .iter()
        .flat_map(|unit| {
            [
                (&unit.dependencies, MAX_DOCUMENT),
                (&unit.preprocessed, MAX_FILE),
                (&unit.object, MAX_FILE),
            ]
        })
        .map(|(path, limit)| (path.clone(), limit))
        .collect::<Vec<_>>();
    let mut journals = Vec::new();
    let commands = request.units.iter().enumerate().flat_map(|(index, unit)| {
        [("compile", &unit.arguments), ("preprocess", &unit.preprocessing_arguments)].map(|(phase, arguments)| json!({
            "unit":index,"phase":phase,"arguments":arguments,"workingDirectory":unit.working_directory,
            "attempted":false,"startObserved":false,"completed":false,"exitCode":null,
            "disposition":"unattempted",
        }))
    }).collect::<Vec<_>>();
    let mut report = json!({
        "schemaVersion":1, "kind":"explicit-native-source-execution", "passed":false,
        "requestSha256":request_digest, "sourceRevision":request.source_revision,
        "target":request.target, "compiler":request.compiler, "environment":request.environment,
        "clock":request.clock,
        "commands":commands, "completedCompilerInvocationObserved":false, "discardedOutputs":[],
        "completeBuildClosureQualified":false, "effectiveConfigurationQualified":false,
        "binaryLinkageQualified":false, "nativeRetirementQualified":false,
        "powerBefore":null, "powerAfter":null,
        "powerScope":"external owner must retain aggregate and per-arm power observations",
    });
    let mut result: Result<()> = (|| {
        if Instant::now() >= deadline {
            return Err("Capture original deadline expired before intake.".into());
        }
        write_new_bytes(&root.join("request.json"), request_bytes)?;
        let identities = preflight(&request, deadline)?;
        let database = request
            .units
            .iter()
            .map(|unit| {
                json!({
                    "directory":unit.working_directory, "file":unit.source,
                    "arguments":unit.arguments, "output":unit.object,
                })
            })
            .collect::<Vec<_>>();
        write_new(&root.join("compile_commands.json"), &json!(database))?;
        let mut units = Vec::new();
        for (index, unit) in request.units.into_iter().enumerate() {
            for (phase_index, (phase, arguments)) in [
                ("compile", &unit.arguments),
                ("preprocess", &unit.preprocessing_arguments),
            ]
            .into_iter()
            .enumerate()
            {
                if Instant::now() >= deadline {
                    return Err("Capture original deadline expired before command.".into());
                }
                let record_index = index * 2 + phase_index;
                let mut record = report["commands"][record_index].clone();
                record["attempted"] = json!(true);
                record["startObserved"] = Value::Null;
                record["disposition"] = json!("dispatch-attempted-start-or-completion-unknown");
                // Each original argv is durable before the first child instruction.
                let journal = root.join(format!("{index}-{phase}.json"));
                write_new(&journal, &record)?;
                journals.push(journal);
                let outcome = invoke(
                    &unit,
                    arguments,
                    &format!("{index}-{phase}"),
                    deadline,
                    &request.environment,
                    &root,
                );
                if let Ok(code) = &outcome {
                    record["completed"] = json!(true);
                    record["startObserved"] = json!(true);
                    record["exitCode"] = json!(code);
                    record["disposition"] = json!("root-exit-and-owned-cleanup-observed");
                    report["completedCompilerInvocationObserved"] = json!(true);
                }
                report["commands"][record_index] = record;
                if outcome? != 0 {
                    return Err(
                        "Native source command failed; later work was not attempted.".into(),
                    );
                }
                if Instant::now() >= deadline {
                    return Err("Capture original deadline expired during command/cleanup.".into());
                }
            }
            units.push(CaptureUnit {
                source: unit.source,
                working_directory: unit.working_directory,
                arguments: unit.arguments,
                preprocessing_arguments: unit.preprocessing_arguments,
                compiler_exit_code: 0,
                preprocessor_exit_code: 0,
                dependencies: unit.dependencies,
                preprocessed: unit.preprocessed,
                object: unit.object,
                inputs: unit.inputs,
            });
        }
        for (path, bytes, sha256) in identities {
            if digest_until(Path::new(&path), MAX_FILE, Some(deadline))? != (bytes, sha256) {
                return Err(
                    "Capture deadline expired or an original input changed during execution."
                        .into(),
                );
            }
        }
        for (path, _) in &outputs {
            if !fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0) {
                return Err(
                    "Successful compiler invocation omitted a nonempty declared output.".into(),
                );
            }
        }
        let manifest = assemble_until(
            Request {
                schema_version: 1,
                kind: "selected-clang-native-source-collection".into(),
                source_revision: request.source_revision,
                target: request.target,
                configuration: request.configuration,
                compiler: request.compiler.path,
                compilation_database: root
                    .join("compile_commands.json")
                    .to_string_lossy()
                    .into_owned(),
                units,
            },
            Some(deadline),
        )?;
        let manifest_digest = hex(&Sha256::digest(manifest.as_bytes()));
        let verification = super::super::verify_manifest_bytes_until(
            manifest.as_bytes(),
            &manifest_digest,
            Some(deadline),
        )?;
        if Instant::now() >= deadline {
            return Err("Capture original deadline expired during verification.".into());
        }
        write_new_bytes(&root.join("manifest.json"), manifest.as_bytes())?;
        write_new(
            &root.join("verification.json"),
            &serde_json::from_str(&verification).map_err(|error| error.to_string())?,
        )?;
        checkpoint(Some(deadline))?;
        report["manifestSha256"] = json!(manifest_digest);
        report["passed"] = json!(true);
        Ok(())
    })();
    if let Err(error) = &result {
        report["failure"] = json!(error);
    }
    // Reserved empty compiler destinations are admission intermediates. They
    // establish no result and have no replay consumer after a failed command.
    retain_outputs(&root, outputs, &mut report, &mut result)?;
    write_new(&root.join("capture.json"), &report)?;
    // Invocation journals survive interrupted owners. Once the final receipt
    // durably owns every attempted command, retain its one concise selection.
    for journal in journals {
        fs::remove_file(journal).map_err(|error| error.to_string())?;
    }
    result.map(|()| report.to_string())
}

#[cfg(windows)]
fn invoke(
    unit: &CommandUnit,
    arguments: &[String],
    name: &str,
    deadline: Instant,
    environment: &BTreeMap<String, String>,
    root: &Path,
) -> Result<u32> {
    let spec = bonsai_windows_process::ToolProcessSpec {
        executable: PathBuf::from(&arguments[0]),
        arguments: arguments[1..].iter().map(OsString::from).collect(),
        working_directory: PathBuf::from(&unit.working_directory),
        stdout: root.join(format!("{name}.stdout")),
        stderr: root.join(format!("{name}.stderr")),
        environment: environment.clone(),
    };
    let log = root.join(format!("{name}.log"));
    let result = crate::command::checks::execute::command_with_deadline(
        &spec,
        Some(deadline),
        &REMOVED
            .iter()
            .map(|name| (*name).into())
            .collect::<Vec<_>>(),
        &[
            (unit.dependencies.clone().into(), MAX_DOCUMENT),
            (unit.preprocessed.clone().into(), MAX_FILE),
            (unit.object.clone().into(), MAX_FILE),
        ],
        true,
        &log,
    );
    // Passing compiler chatter has no replay consumer; failures retain only
    // nonblank captures. Exact original commands and statuses live in capture.json.
    for path in [&spec.stdout, &spec.stderr] {
        if path.is_file()
            && (result.as_ref().is_ok_and(|(code, _)| *code == 0)
                || read_bounded(path, 16 * 1024 * 1024)?
                    .iter()
                    .all(u8::is_ascii_whitespace))
        {
            fs::remove_file(path).map_err(|error| error.to_string())?;
        }
    }
    if log.is_file() {
        fs::remove_file(&log).map_err(|error| error.to_string())?;
    }
    result.map(|(code, _)| code)
}

pub(super) fn run(arguments: &[OsString]) -> Result<String> {
    if arguments.len() != 5 || arguments[1] != "--request" || arguments[3] != "--expected-sha256" {
        return Err(USAGE.into());
    }
    let expected = arguments[4]
        .to_str()
        .filter(|value| lower_hex(value, 64))
        .ok_or(USAGE)?;
    let path = arguments[2].to_str().ok_or(USAGE)?;
    path_key(path)?;
    let bytes = read_bounded(Path::new(path), MAX_DOCUMENT / 2)?;
    if hex(&Sha256::digest(&bytes)) != expected {
        return Err("Native execution request differs from its reviewed SHA-256.".into());
    }
    let request: Execution = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    #[cfg(windows)]
    {
        let admission = Instant::now();
        let (tick, frequency) =
            bonsai_windows_process::monotonic_ticks().map_err(|error| error.to_string())?;
        let left = remaining(&request.clock, now_milliseconds()?, tick, frequency)?;
        execute(request, &bytes, expected, admission + left, invoke)
    }
    #[cfg(not(windows))]
    {
        let _ = &request.clock;
        let _ = (
            request,
            now_milliseconds,
            remaining,
            execute::<
                fn(
                    &CommandUnit,
                    &[String],
                    &str,
                    Instant,
                    &BTreeMap<String, String>,
                    &Path,
                ) -> Result<u32>,
            >,
            REMOVED,
        );
        Err("Live native source capture requires native Windows supervision.".into())
    }
}

#[cfg(test)]
#[path = "native_source_capture_tests.rs"]
mod tests;
