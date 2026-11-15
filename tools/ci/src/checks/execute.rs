//! Windows Job ownership and retained Cargo check receipts.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead as _, BufReader, Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bonsai_artifacts::{hash_reader, validate_external_path};
use bonsai_windows_process::{OwnedProcess, ToolProcessSpec};
use serde_json::{Value, json};

use super::{Check, Options, Package};
use crate::Result;

const MAX_LOG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_CONSOLE_LOG_BYTES: u64 = 12 * 1024;
const MAX_HIGHLIGHT_BYTES: usize = 4 * 1024;

fn console_excerpt(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "  | (empty)\n".into();
    }
    let text = String::from_utf8_lossy(bytes);
    let mut excerpt = String::new();
    for line in text.lines() {
        // Captured output must not become a GitHub workflow command or emit
        // terminal control sequences. Preserve the original bytes in artifacts.
        excerpt.push_str("  | ");
        for character in line.chars() {
            excerpt.push(if character.is_control() && character != '\t' {
                '\u{fffd}'
            } else {
                character
            });
        }
        excerpt.push('\n');
    }
    excerpt
}

fn is_diagnostic_headline(trimmed: &str) -> bool {
    trimmed.contains("panicked at ")
        || (trimmed.starts_with("---- ") && trimmed.ends_with(" ----"))
        || trimmed.starts_with("error[")
        || trimmed.starts_with("error:")
        || trimmed == "failures:"
        || trimmed.starts_with("test result: FAILED")
}

pub(crate) fn extract_prefix_highlights(reader: impl std::io::Read) -> std::io::Result<Vec<u8>> {
    let mut buffered = BufReader::new(reader);
    let mut raw_line = Vec::new();
    let mut highlights = Vec::new();
    let mut context_remaining = 0_u8;
    while buffered.read_until(b'\n', &mut raw_line)? != 0 {
        let text = String::from_utf8_lossy(&raw_line);
        let trimmed = text.trim();
        let matched = is_diagnostic_headline(trimmed);
        if matched {
            context_remaining = 3;
        }
        if (matched || context_remaining > 0) && !trimmed.is_empty() {
            let remaining = MAX_HIGHLIGHT_BYTES.saturating_sub(highlights.len());
            if remaining == 0 {
                break;
            }
            let slice = if raw_line.len() <= remaining {
                &raw_line[..]
            } else {
                &raw_line[..remaining]
            };
            highlights.extend_from_slice(slice);
            if !highlights.ends_with(b"\n") {
                highlights.push(b'\n');
            }
            if !matched {
                context_remaining = context_remaining.saturating_sub(1);
            }
        }
        raw_line.clear();
    }
    Ok(highlights)
}

fn read_log_tail(path: &Path) -> std::io::Result<(Vec<u8>, bool, Vec<u8>)> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    let offset = length.saturating_sub(MAX_CONSOLE_LOG_BYTES);
    let highlights = if offset != 0 {
        let prefix = File::open(path)?.take(offset);
        extract_prefix_highlights(prefix).unwrap_or_default()
    } else {
        Vec::new()
    };
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = Vec::new();
    file.take(MAX_CONSOLE_LOG_BYTES).read_to_end(&mut bytes)?;
    Ok((highlights, offset != 0, bytes))
}

pub(super) fn write_failure_logs(output: &mut impl std::io::Write, spec: &ToolProcessSpec) {
    for (label, path) in [("stdout", &spec.stdout), ("stderr", &spec.stderr)] {
        let _ = writeln!(output, "{label}: {}", path.display());
        match read_log_tail(path) {
            Ok((highlights, truncated, bytes)) => {
                if !highlights.is_empty() {
                    let _ = writeln!(
                        output,
                        "  | [key diagnostic lines earlier in {label} before truncated tail]"
                    );
                    let _ = output.write_all(console_excerpt(&highlights).as_bytes());
                }
                if truncated {
                    let _ = writeln!(
                        output,
                        "  | [showing the final {MAX_CONSOLE_LOG_BYTES} bytes; full output retained]"
                    );
                }
                let _ = output.write_all(console_excerpt(&bytes).as_bytes());
            }
            Err(error) => {
                let _ = writeln!(output, "  | Cannot read diagnostic output: {error}");
            }
        }
    }
}

fn format_step_summary(report: &Value) -> String {
    let lane = std::env::var("RUST_LANE").unwrap_or_default();
    let heading = if lane.is_empty() {
        "### Rust Checks".to_owned()
    } else {
        format!("### Rust Checks (`{lane}`)")
    };
    let mut out = format!("{heading}\n\n");
    out.push_str("| Check | Phase | Packages | Status | Exit | Elapsed |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- |\n");
    if let Some(checks) = report["Checks"].as_array() {
        for check in checks {
            let name = check["Name"].as_str().unwrap_or("unknown");
            let phase = check["Phase"].as_str().unwrap_or("unknown");
            let packages = check["Packages"]
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            let status = if check["Passed"] == true {
                "PASS"
            } else {
                "FAIL"
            };
            let exit = check["ExitCode"]
                .as_u64()
                .map_or_else(|| "-".to_owned(), |code| code.to_string());
            let elapsed = check["ElapsedMilliseconds"].as_u64().map_or_else(
                || "-".to_owned(),
                |ms| format!("{}.{:02}s", ms / 1000, (ms % 1000) / 10),
            );
            let _ = writeln!(
                out,
                "| `{name}` | `{phase}` | `{packages}` | `{status}` | `{exit}` | `{elapsed}` |"
            );
        }
    }
    if let Some(error) = report["Error"].as_str() {
        let _ = writeln!(out, "\n**Error:** {error}");
    }
    out
}

fn external(path: &Path) -> Result<()> {
    validate_external_path(path).map_err(|error| error.to_string())
}

fn distinct(target: &Path, evidence: &Path) -> Result<()> {
    external(target)?;
    external(evidence)?;
    let normalized = |path: &Path| {
        path.to_string_lossy()
            .replace('/', "\\")
            .trim_end_matches('\\')
            .to_uppercase()
    };
    let target = normalized(target);
    let evidence = normalized(evidence);
    if target == evidence
        || target.starts_with(&(evidence.clone() + "\\"))
        || evidence.starts_with(&(target + "\\"))
    {
        return Err("Cargo targets must be separate from uploaded evidence.".into());
    }
    Ok(())
}

fn executable(name: &str) -> Result<PathBuf> {
    for directory in std::env::split_paths(&std::env::var_os("PATH").ok_or("PATH is unavailable.")?)
    {
        if directory.is_absolute() {
            let path = directory.join(name);
            if path.is_file() {
                return Ok(path);
            }
        }
    }
    Err(format!("Required executable is missing from PATH: {name}"))
}

fn digest(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    hash_reader(&mut file, length, &mut vec![0; 65536], &mut || Ok(()))
        .map_err(|error| error.to_string())
}

pub(super) fn read_log(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    file.take(MAX_LOG_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if u64::try_from(bytes.len()).map_err(|error| error.to_string())? > MAX_LOG_BYTES {
        return Err("Rust command output exceeded its byte limit.".into());
    }
    Ok(bytes)
}

pub(in crate::command) fn command(
    spec: &ToolProcessSpec,
    allowance: Option<Duration>,
    log: &Path,
) -> Result<(u32, String)> {
    command_with_deadline(
        spec,
        allowance.map(|duration| Instant::now() + duration),
        &[],
        &[],
        false,
        log,
    )
}

pub(in crate::command) fn command_with_deadline(
    spec: &ToolProcessSpec,
    deadline: Option<Instant>,
    removed: &[String],
    watched: &[(PathBuf, u64)],
    reject_survivors: bool,
    log: &Path,
) -> Result<(u32, String)> {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Err("Rust check aggregate deadline expired.".into());
    }
    let mut output = std::io::stderr().lock();
    let _ = writeln!(
        output,
        "Running {} {:?}; log: {}",
        spec.executable.display(),
        spec.arguments,
        log.display()
    );
    let child =
        OwnedProcess::start_tool_with_removed(spec, removed).map_err(|error| error.to_string())?;
    let code = loop {
        if let Some(code) = child.try_wait().map_err(|error| error.to_string())? {
            break code;
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline)
            || [&spec.stdout, &spec.stderr].into_iter().any(|path| {
                !fs::metadata(path).is_ok_and(|metadata| metadata.len() <= MAX_LOG_BYTES)
            })
            || watched.iter().any(|(path, limit)| {
                fs::metadata(path).is_ok_and(|metadata| metadata.len() > *limit)
            })
        {
            child
                .stop(Duration::from_secs(5))
                .map_err(|error| error.to_string())?;
            fs::write(log, "Rust command exceeded its deadline or output limit; partial stdout/stderr are retained beside this log.\n").map_err(|error| error.to_string())?;
            write_failure_logs(&mut output, spec);
            return Err("Rust command exceeded its deadline or output limit; owned descendants were stopped.".into());
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let survivors = if reject_survivors {
        // Root signaling can precede final Job accounting or normal child exit.
        // This short drain shares the original deadline; it is never a new
        // command allowance. Remaining members still force failure and teardown.
        let grace = Instant::now();
        Some(loop {
            let active = child
                .active_processes()
                .map_err(|error| error.to_string())?;
            if active == 0
                || grace.elapsed() >= Duration::from_millis(250)
                || deadline.is_some_and(|deadline| Instant::now() >= deadline)
            {
                break active;
            }
            std::thread::sleep(Duration::from_millis(10));
        })
    } else {
        None
    };
    child
        .stop(Duration::from_secs(5))
        .map_err(|error| error.to_string())?;
    if survivors.is_some_and(|count| count != 0) {
        return Err(
            "Compiler root exited with surviving owned descendants; capture rejected.".into(),
        );
    }
    if watched
        .iter()
        .any(|(path, limit)| fs::metadata(path).is_ok_and(|metadata| metadata.len() > *limit))
    {
        return Err(
            "Compiler output exceeded its declared artifact bound; capture rejected.".into(),
        );
    }
    let stdout = read_log(&spec.stdout)?;
    let stderr = read_log(&spec.stderr)?;
    let mut combined = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(log)
        .map_err(|error| error.to_string())?;
    combined
        .write_all(&stdout)
        .and_then(|()| combined.write_all(&stderr))
        .map_err(|error| error.to_string())?;
    if code != 0 {
        let _ = writeln!(
            output,
            "Command failed with exit code {code}; log: {}",
            log.display()
        );
        write_failure_logs(&mut output, spec);
    }
    Ok((code, String::from_utf8_lossy(&stdout).trim().into()))
}

fn receipt(path: &Path, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| error.to_string())
}

struct Execution<'a> {
    options: &'a Options,
    plan: &'a [Check],
    toolchain: &'a str,
    target: &'a Path,
    evidence: &'a Path,
    work: &'a Path,
    started: Instant,
    owner: bool,
}

impl Execution<'_> {
    fn allowance(&self, seconds: u64) -> Option<Duration> {
        if self.options.github_release {
            return None;
        }
        let per_command = Duration::from_secs(seconds);
        Some(if self.owner {
            per_command.min(Duration::from_secs(290).saturating_sub(self.started.elapsed()))
        } else {
            per_command
        })
    }

    fn spec(
        &self,
        executable: PathBuf,
        arguments: Vec<OsString>,
        name: &str,
    ) -> Result<ToolProcessSpec> {
        let text = |path: &Path| -> Result<String> {
            path.to_str()
                .map(String::from)
                .ok_or_else(|| "Rust check paths must be UTF-8.".into())
        };
        Ok(ToolProcessSpec {
            executable,
            arguments,
            working_directory: self.work.into(),
            stdout: self.evidence.join(format!("{name}.stdout")),
            stderr: self.evidence.join(format!("{name}.stderr")),
            environment: BTreeMap::from([
                ("CARGO_TARGET_DIR".into(), text(self.target)?),
                ("TEMP".into(), text(self.work)?),
                ("TMP".into(), text(self.work)?),
            ]),
        })
    }

    fn run(&self, report: &mut Value, receipt_path: &Path) -> Result<()> {
        let cargo = executable("cargo.exe")?;
        let rustc = executable("rustc.exe")?;
        let identity = command(
            &self.spec(
                rustc,
                vec![format!("+{}", self.toolchain).into(), "-vV".into()],
                "rustc",
            )?,
            self.allowance(60),
            &self.evidence.join("rustc.log"),
        )?;
        if identity.0 != 0 {
            return Err("Rust compiler identification failed.".into());
        }
        report["CompilerIdentity"] = identity.1.into();
        for (index, check) in self.plan.iter().enumerate() {
            let log = self.evidence.join(format!("{}.log", check.name));
            let started = Instant::now();
            let mut record = json!({"Name": check.name, "Phase": check.phase, "Packages": check.packages,
                "Executable": cargo, "Arguments": check.arguments, "Log": log,
                "ExitCode": null, "Passed": false, "ElapsedMilliseconds": null,
                "TestCase": check.test_case, "TestPassed": null, "Error": null});
            let result = command(
                &self.spec(
                    cargo.clone(),
                    check.arguments.iter().map(Into::into).collect(),
                    &check.name,
                )?,
                self.allowance(self.options.timeout_seconds),
                &log,
            );
            record["ElapsedMilliseconds"] = u64::try_from(started.elapsed().as_millis())
                .map_err(|error| error.to_string())?
                .into();
            if let Ok((code, stdout)) = &result {
                record["ExitCode"] = (*code).into();
                match super::focused_test_result(check, *code, stdout) {
                    Ok(passed) => {
                        record["Passed"] = (*code == 0).into();
                        record["TestPassed"] = passed.into();
                    }
                    Err(error) => record["Error"] = error.into(),
                }
            } else if let Err(error) = &result {
                record["Error"] = error.as_str().into();
            }
            let stop = super::record_check_result(
                self.owner,
                report,
                check,
                &self.plan[index + 1..],
                record,
            )?;
            receipt(receipt_path, report)?;
            result.map_err(|error| {
                format!(
                    "Rust {} check {} for {} could not complete: {error} Log: {}",
                    check.phase,
                    check.name,
                    check.packages.join(", "),
                    log.display()
                )
            })?;
            if stop {
                return Err(format!(
                    "Local focused check {} failed; remaining selections were not executed. Receipt: {}",
                    check.name,
                    receipt_path.display()
                ));
            }
        }
        if report["Checks"]
            .as_array()
            .ok_or("Invalid check receipt.")?
            .iter()
            .any(|record| record["Passed"] != true)
        {
            let failed = report["Checks"]
                .as_array()
                .ok_or("Invalid check receipt.")?
                .iter()
                .filter(|record| record["Passed"] != true)
                .filter_map(|record| record["Name"].as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "Rust checks failed: {failed}. Failure excerpts are above; complete logs and receipt: {}",
                receipt_path.display()
            ));
        }
        Ok(())
    }
}

pub(super) fn run(
    options: &Options,
    packages: &[&Package],
    toolchain: &str,
    plan: &[Check],
) -> Result<String> {
    let started = Instant::now();
    let target = options
        .target
        .as_deref()
        .ok_or("--target is required for rust-check.")?;
    let evidence = options
        .evidence
        .as_deref()
        .ok_or("--evidence is required for rust-check.")?;
    distinct(target, evidence)?;
    if evidence.exists() {
        return Err("Use a new Rust check evidence directory.".into());
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let work = std::env::temp_dir().join(format!(
        "SwiftBonsai-RustCheck-{}-{nonce}",
        std::process::id()
    ));
    external(&work)?;
    let workspace_hash = digest(&options.root.join("Cargo.toml"))?;
    let lock_hash = digest(&options.root.join("Cargo.lock"))?;
    let registry_hash = digest(&options.root.join("config/rust-packages.json"))?;
    fs::create_dir_all(target).map_err(|error| error.to_string())?;
    if let Some(parent) = evidence.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::create_dir(evidence).map_err(|error| error.to_string())?;
    fs::create_dir(&work).map_err(|error| error.to_string())?;
    let mut report = json!({
        "SchemaVersion": 2, "Revision": std::env::var("GITHUB_SHA").ok(), "Toolchain": toolchain,
        "CheckProfile": options.profile, "FingerprintMode": std::env::var("CARGO_BUILD_FINGERPRINT").unwrap_or_else(|_| "mtime".into()),
        "Packages": packages.iter().map(|package| &package.name).collect::<Vec<_>>(), "TargetDirectory": target,
        "WorkingDirectory": work, "CompilerIdentity": null, "Checks": [], "SkippedChecks": [], "Success": false, "Error": null,
        "PerformanceMeasured": false, "PlannedCargoInvocations": plan.len(), "UnbatchedCargoInvocations": packages.len() * options.phases.len(),
        "GitHubReleaseDeadline": options.github_release,
        "WorkspaceSha256": workspace_hash, "LockSha256": lock_hash, "RegistrySha256": registry_hash
    });
    let path = evidence.join("rust-checks.json");
    receipt(&path, &report)?;
    let execution = Execution {
        options,
        plan,
        toolchain,
        target,
        evidence,
        work: &work,
        started,
        owner: std::env::var("RUNNER_ENVIRONMENT").as_deref() != Ok("github-hosted"),
    };
    let result = execution.run(&mut report, &path);
    match &result {
        Ok(()) => report["Success"] = true.into(),
        Err(error) => report["Error"] = error.clone().into(),
    }
    receipt(&path, &report)?;
    let _ = crate::command::append_environment_file(
        "GITHUB_STEP_SUMMARY",
        &format_step_summary(&report),
    );
    result?;
    serde_json::to_string(&report).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_excerpts_preserve_unicode_and_neutralize_workflow_commands() {
        let excerpt = console_excerpt("::error::fixture\n\u{1b}[31m\rmalformed\t雪\n".as_bytes());
        assert_eq!(
            excerpt,
            "  | ::error::fixture\n  | \u{fffd}[31m\u{fffd}malformed\t雪\n"
        );
        assert_eq!(console_excerpt(&[0x80, b'\n']), "  | \u{fffd}\n");
        assert_eq!(console_excerpt(&[]), "  | (empty)\n");
    }

    #[test]
    fn prefix_highlights_retain_panic_and_compiler_errors_before_large_widget_dump()
    -> std::io::Result<()> {
        let mut prefix = String::new();
        prefix.push_str("running 1 test\n");
        prefix.push_str("---- window::interaction_tests::case stdout ----\n");
        prefix.push_str("thread 'main' panicked at src/window.rs:42:9:\n");
        prefix.push_str("assertion `left == right` failed\n");
        prefix.push_str("  left: 1\n");
        prefix.push_str(" right: 2\n");
        for _ in 0..400 {
            prefix.push_str("    role: Button, label: \"Make a plan\", hidden: false,\n");
        }
        let highlights = extract_prefix_highlights(prefix.as_bytes())?;
        let text = String::from_utf8_lossy(&highlights);
        assert!(text.contains("---- window::interaction_tests::case stdout ----"));
        assert!(text.contains("panicked at src/window.rs:42:9:"));
        assert!(text.contains("assertion `left == right` failed"));
        assert!(!text.contains("Make a plan"));
        Ok(())
    }

    #[test]
    fn targets_and_uploaded_evidence_cannot_overlap() {
        let root = std::env::temp_dir().join("SwiftBonsai-CheckPaths");
        assert!(distinct(&root.join("target"), &root.join("evidence")).is_ok());
        assert!(distinct(&root, &root).is_err());
        assert!(distinct(&root, &root.join("nested")).is_err());
        assert!(distinct(&root.join("nested"), &root).is_err());
        assert!(distinct(Path::new("relative"), &root).is_err());
    }
}
