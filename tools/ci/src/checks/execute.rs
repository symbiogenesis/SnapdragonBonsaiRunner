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
const MAX_SUMMARY_FAILURES: usize = 16;

pub(crate) fn describe_exit_code(code: u32) -> String {
    if code < 0x8000_0000 {
        return code.to_string();
    }
    let label = match code {
        0x8000_0003 => Some("STATUS_BREAKPOINT"),
        0xC000_0005 => Some("STATUS_ACCESS_VIOLATION"),
        0xC000_001D => Some("STATUS_ILLEGAL_INSTRUCTION"),
        0xC000_0094 => Some("STATUS_INTEGER_DIVIDE_BY_ZERO"),
        0xC000_00FD => Some("STATUS_STACK_OVERFLOW"),
        0xC000_0135 => Some("STATUS_DLL_NOT_FOUND"),
        0xC000_013A => Some("STATUS_CONTROL_C_EXIT"),
        0xC000_0142 => Some("STATUS_DLL_INIT_FAILED"),
        0xC000_0374 => Some("STATUS_HEAP_CORRUPTION"),
        0xC000_0409 => Some("STATUS_STACK_BUFFER_OVERRUN / __fastfail"),
        _ => None,
    };
    match label {
        Some(name) => format!("{code} (0x{code:08X}: {name})"),
        None => format!("{code} (0x{code:08X})"),
    }
}

fn sanitize_console_line(line: &str) -> String {
    let mut sanitized = String::with_capacity(line.len());
    for character in line.chars() {
        sanitized.push(if character.is_control() && character != '\t' {
            '\u{fffd}'
        } else {
            character
        });
    }
    sanitized
}

fn flush_repeated_line(excerpt: &mut String, previous: &str, count: usize) {
    if count == 2 {
        let _ = writeln!(excerpt, "  | {previous}");
    } else if count >= 3 {
        let _ = writeln!(
            excerpt,
            "  | [previous line repeated {} more times]",
            count - 1
        );
    }
}

fn console_excerpt(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "  | (empty)\n".into();
    }
    let text = String::from_utf8_lossy(bytes);
    let mut excerpt = String::new();
    let mut previous: Option<String> = None;
    let mut repeat_count = 0_usize;
    for line in text.lines() {
        // Captured output must not become a GitHub workflow command or emit
        // terminal control sequences. Preserve the original bytes in artifacts.
        let sanitized = sanitize_console_line(line);
        if previous.as_deref() == Some(sanitized.as_str()) {
            repeat_count += 1;
            continue;
        }
        if let Some(prev) = previous.take() {
            flush_repeated_line(&mut excerpt, &prev, repeat_count);
        }
        let _ = writeln!(excerpt, "  | {sanitized}");
        previous = Some(sanitized);
        repeat_count = 1;
    }
    if let Some(prev) = previous {
        flush_repeated_line(&mut excerpt, &prev, repeat_count);
    }
    excerpt
}

fn is_failed_test_line(trimmed: &str) -> bool {
    trimmed
        .strip_prefix("test ")
        .and_then(|rest| rest.split_once(" ... "))
        .is_some_and(|(_, outcome)| outcome == "FAILED" || outcome.starts_with("FAILED "))
}

fn is_diagnostic_headline(trimmed: &str) -> bool {
    trimmed.contains("panicked at ")
        || (trimmed.starts_with("---- ") && trimmed.ends_with(" ----"))
        || trimmed.starts_with("error[")
        || trimmed.starts_with("error:")
        || trimmed == "failures:"
        || trimmed.starts_with("test result: FAILED")
        || is_failed_test_line(trimmed)
}

fn summarize_test_failure(stdout: &str, code: u32) -> (Vec<String>, Option<String>) {
    let mut failed_tests = Vec::new();
    let mut suite_expected: Option<usize> = None;
    let mut suite_completed = 0_usize;
    let mut suite_last_test: Option<String> = None;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("running ")
            && let Some(count_str) = rest
                .strip_suffix(" tests")
                .or_else(|| rest.strip_suffix(" test"))
            && let Ok(expected) = count_str.parse::<usize>()
        {
            suite_expected = Some(expected);
            suite_completed = 0;
            suite_last_test = None;
            continue;
        }
        if trimmed.starts_with("test result:") {
            suite_expected = None;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("test ")
            && let Some((name, outcome)) = rest.split_once(" ... ")
        {
            if outcome == "FAILED" || outcome.starts_with("FAILED ") {
                if !failed_tests.iter().any(|existing| existing == name) {
                    failed_tests.push(name.to_owned());
                }
                suite_completed += 1;
                suite_last_test = Some(name.to_owned());
            } else if outcome == "ok"
                || outcome.starts_with("ok ")
                || outcome == "ignored"
                || outcome.starts_with("ignored ")
            {
                suite_completed += 1;
                suite_last_test = Some(name.to_owned());
            }
        }
    }
    let crash_diagnostic = suite_expected.map(|expected| {
        let last = suite_last_test
            .map_or_else(String::new, |name| format!(" (last completed: `{name}`)"));
        format!(
            "Test binary terminated before `test result:` ({}) after {suite_completed}/{expected} completed tests{last}; the crashing test was in-flight under concurrent libtest execution.",
            describe_exit_code(code)
        )
    });
    (failed_tests, crash_diagnostic)
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
            let exit = check["ExitCode"].as_u64().map_or_else(
                || "-".to_owned(),
                |code| u32::try_from(code).map_or_else(|_| code.to_string(), describe_exit_code),
            );
            let elapsed = check["ElapsedMilliseconds"].as_u64().map_or_else(
                || "-".to_owned(),
                |ms| format!("{}.{:02}s", ms / 1000, (ms % 1000) / 10),
            );
            let _ = writeln!(
                out,
                "| `{name}` | `{phase}` | `{packages}` | `{status}` | `{exit}` | `{elapsed}` |"
            );
        }
        for check in checks {
            let name = check["Name"].as_str().unwrap_or("unknown");
            if let Some(crash) = check["CrashDiagnostic"].as_str() {
                let _ = writeln!(out, "\n- **`{name}` crash:** {crash}");
            }
            if let Some(failed_tests) = check["FailedTests"].as_array()
                && !failed_tests.is_empty()
            {
                let shown: Vec<_> = failed_tests
                    .iter()
                    .filter_map(Value::as_str)
                    .take(MAX_SUMMARY_FAILURES)
                    .map(|case| format!("`{case}`"))
                    .collect();
                let extra = failed_tests.len().saturating_sub(shown.len());
                let suffix = if extra == 0 {
                    String::new()
                } else {
                    format!(" (+{extra} more)")
                };
                let _ = writeln!(
                    out,
                    "\n- **`{name}` failed tests:** {}{suffix}",
                    shown.join(", ")
                );
            }
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
    crate::impact::workspace::read_bounded_reader(
        file,
        MAX_LOG_BYTES,
        "Rust command output exceeded its byte limit.",
    )
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
    const OUTPUT_CHECK_INTERVAL: Duration = Duration::from_millis(20);
    let code = loop {
        let wait = match deadline {
            Some(deadline) => deadline
                .saturating_duration_since(Instant::now())
                .min(OUTPUT_CHECK_INTERVAL),
            None => OUTPUT_CHECK_INTERVAL,
        };
        if let Some(code) = child
            .wait_for_exit(wait)
            .map_err(|error| error.to_string())?
        {
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
            let step = if grace.elapsed() < Duration::from_millis(25) {
                Duration::from_millis(1)
            } else {
                Duration::from_millis(5)
            };
            std::thread::sleep(step);
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
            "Command failed with exit code {}; log: {}",
            describe_exit_code(code),
            log.display()
        );
        write_failure_logs(&mut output, spec);
    }
    Ok((code, String::from_utf8_lossy(&stdout).trim().into()))
}

fn is_transient_fs_error(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::PermissionDenied
        || error.kind() == std::io::ErrorKind::WouldBlock
        || matches!(error.raw_os_error(), Some(5 | 32 | 33))
}

fn rename_with_retry(source: &Path, destination: &Path) -> std::io::Result<()> {
    const MAX_ATTEMPTS: usize = 6;
    let mut delay = Duration::from_millis(if cfg!(test) { 1 } else { 10 });
    for attempt in 1..=MAX_ATTEMPTS {
        match fs::rename(source, destination) {
            Ok(()) => return Ok(()),
            Err(error) if attempt < MAX_ATTEMPTS && is_transient_fs_error(&error) => {
                std::thread::sleep(delay);
                delay = delay.saturating_mul(2);
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn receipt(path: &Path, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    let mut partial_name = path
        .file_name()
        .ok_or("Receipt path must include a file name.")?
        .to_os_string();
    partial_name.push(".partial");
    let partial_path = path.with_file_name(partial_name);
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&partial_path)
        .map_err(|error| error.to_string())?;
    if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = fs::remove_file(&partial_path);
        return Err(error.to_string());
    }
    drop(file);
    if let Err(error) = rename_with_retry(&partial_path, path) {
        let _ = fs::remove_file(&partial_path);
        return Err(error.to_string());
    }
    Ok(())
}

struct TempWorkDir(PathBuf);

impl TempWorkDir {
    fn create(path: PathBuf) -> Result<Self> {
        fs::create_dir(&path).map_err(|error| error.to_string())?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempWorkDir {
    fn drop(&mut self) {
        if self.0.starts_with(std::env::temp_dir())
            && fs::symlink_metadata(&self.0)
                .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
        {
            const MAX_ATTEMPTS: usize = 5;
            let mut delay = Duration::from_millis(if cfg!(test) { 1 } else { 10 });
            for attempt in 1..=MAX_ATTEMPTS {
                match fs::remove_dir_all(&self.0) {
                    Ok(()) => return,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
                    Err(error)
                        if attempt < MAX_ATTEMPTS
                            && (is_transient_fs_error(&error)
                                || error.kind() == std::io::ErrorKind::DirectoryNotEmpty
                                || error.raw_os_error() == Some(145)) =>
                    {
                        std::thread::sleep(delay);
                        delay = delay.saturating_mul(2);
                    }
                    Err(_) => return,
                }
            }
        }
    }
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
                if check.phase == "test" && *code != 0 {
                    let (failed_tests, crash_diagnostic) = summarize_test_failure(stdout, *code);
                    if !failed_tests.is_empty() {
                        record["FailedTests"] = failed_tests.into();
                    }
                    if let Some(crash) = crash_diagnostic {
                        let _ =
                            writeln!(std::io::stderr().lock(), "  | [crash diagnostic] {crash}");
                        record["CrashDiagnostic"] = crash.into();
                    }
                }
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
    let work = TempWorkDir::create(work)?;
    let mut report = json!({
        "SchemaVersion": 2, "Revision": std::env::var("GITHUB_SHA").ok(), "Toolchain": toolchain,
        "CheckProfile": options.profile, "FingerprintMode": std::env::var("CARGO_BUILD_FINGERPRINT").unwrap_or_else(|_| "mtime".into()),
        "Packages": packages.iter().map(|package| &package.name).collect::<Vec<_>>(), "TargetDirectory": target,
        "WorkingDirectory": work.path(), "CompilerIdentity": null, "Checks": [], "SkippedChecks": [], "Success": false, "Error": null,
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
        work: work.path(),
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

    #[test]
    fn exit_codes_decode_windows_ntstatus_and_preserve_small_codes() {
        assert_eq!(describe_exit_code(1), "1");
        assert_eq!(describe_exit_code(101), "101");
        assert_eq!(
            describe_exit_code(0xC000_0409),
            "3221226505 (0xC0000409: STATUS_STACK_BUFFER_OVERRUN / __fastfail)"
        );
        assert_eq!(
            describe_exit_code(0xC000_0005),
            "3221225477 (0xC0000005: STATUS_ACCESS_VIOLATION)"
        );
        assert_eq!(describe_exit_code(0xDEAD_BEEF), "3735928559 (0xDEADBEEF)");
    }

    #[test]
    fn console_excerpt_collapses_consecutive_duplicate_lines() {
        let mut noisy = String::new();
        noisy.push_str("alpha\n");
        noisy.push_str("pair\npair\n");
        for _ in 0..5 {
            noisy.push_str("Local model request failed: connection refused\n");
        }
        noisy.push_str("omega\n");
        assert_eq!(
            console_excerpt(noisy.as_bytes()),
            concat!(
                "  | alpha\n",
                "  | pair\n",
                "  | pair\n",
                "  | Local model request failed: connection refused\n",
                "  | [previous line repeated 4 more times]\n",
                "  | omega\n",
            )
        );
    }

    #[test]
    fn step_summary_surfaces_failed_tests_and_aborted_suite_crash() -> Result<()> {
        let stdout = concat!(
            "running 3 tests\n",
            "test alpha ... ok <0.001s>\n",
            "test beta ... FAILED <0.004s>\n",
            "test result: ok. 0 passed; 0 failed\n",
            "running 79 tests\n",
            "test asr::first ... ok <0.010s>\n",
            "test asr::seventy_eighth ... ok <0.015s>\n",
        );
        let (failed, crash) = summarize_test_failure(stdout, 0xC000_0409);
        assert_eq!(failed, vec!["beta".to_owned()]);
        let crash_text = crash.ok_or("expected mid-suite crash diagnostic")?;
        assert!(crash_text.contains("STATUS_STACK_BUFFER_OVERRUN"));
        assert!(crash_text.contains("2/79 completed tests"));
        assert!(crash_text.contains("asr::seventy_eighth"));

        let report = json!({
            "Checks": [{
                "Name": "bonsai-speech-asr-test",
                "Phase": "test",
                "Packages": ["bonsai-speech-asr"],
                "Passed": false,
                "ExitCode": 0xC000_0409_u32,
                "ElapsedMilliseconds": 41_910_u64,
                "FailedTests": failed,
                "CrashDiagnostic": crash_text,
            }],
            "Error": "Rust checks failed: bonsai-speech-asr-test."
        });
        let summary = format_step_summary(&report);
        assert!(
            summary.contains("3221226505 (0xC0000409: STATUS_STACK_BUFFER_OVERRUN / __fastfail)")
        );
        assert!(summary.contains("**`bonsai-speech-asr-test` crash:**"));
        assert!(summary.contains("**`bonsai-speech-asr-test` failed tests:** `beta`"));
        Ok(())
    }

    #[test]
    fn receipt_writes_atomically_and_temp_work_dir_cleans_up_on_drop() -> Result<()> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let scratch = std::env::temp_dir().join(format!(
            "SwiftBonsai-RustCheck-Test-{}-{nonce}",
            std::process::id()
        ));
        let receipt_path = scratch.join("rust-checks.json");
        let partial_path = scratch.join("rust-checks.json.partial");
        {
            let guard = TempWorkDir::create(scratch.clone())?;
            fs::write(guard.path().join("stale.txt"), b"scratch")
                .map_err(|error| error.to_string())?;
            receipt(
                &receipt_path,
                &json!({"SchemaVersion": 2, "Success": false}),
            )?;
            assert!(!partial_path.exists());
            receipt(&receipt_path, &json!({"SchemaVersion": 2, "Success": true}))?;
            assert!(!partial_path.exists());
            let parsed: Value = serde_json::from_slice(
                &fs::read(&receipt_path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            assert_eq!(parsed["Success"], json!(true));
        }
        assert!(
            !scratch.exists(),
            "TempWorkDir Drop guard must remove SwiftBonsai-RustCheck-* directory."
        );

        // is_transient_fs_error classification
        for code in [5_i32, 32, 33] {
            assert!(is_transient_fs_error(&std::io::Error::from_raw_os_error(
                code
            )));
        }
        assert!(is_transient_fs_error(&std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "denied"
        )));
        assert!(is_transient_fs_error(&std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "blocked"
        )));
        assert!(!is_transient_fs_error(&std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing"
        )));

        // receipt() error paths: missing filename and rename failure into existing directory
        let no_name = Path::new("C:\\");
        assert!(receipt(no_name, &json!({"Success": false})).is_err());

        let err_dir = std::env::temp_dir().join(format!(
            "SwiftBonsai-ReceiptErr-{}-{nonce}",
            std::process::id()
        ));
        let dest_dir = err_dir.join("occupied");
        let dest_partial = err_dir.join("occupied.partial");
        {
            let _guard = TempWorkDir::create(err_dir.clone())?;
            fs::create_dir(&dest_dir).map_err(|e| e.to_string())?;
            assert!(receipt(&dest_dir, &json!({"Success": true})).is_err());
            assert!(!dest_partial.exists());
            assert!(dest_dir.is_dir());

            // TempWorkDir::drop refuses non-temp path and refuses junction inside temp_dir
            let target_dir = err_dir.join("target-with-sentinel");
            let sentinel = target_dir.join("keep.txt");
            fs::create_dir(&target_dir).map_err(|e| e.to_string())?;
            fs::write(&sentinel, b"keep").map_err(|e| e.to_string())?;
            let junction = err_dir.join("junction");
            let status = std::process::Command::new("cmd")
                .args([
                    "/c",
                    "mklink",
                    "/J",
                    &junction.to_string_lossy(),
                    &target_dir.to_string_lossy(),
                ])
                .output()
                .map_err(|e| e.to_string())?;
            assert!(status.status.success(), "mklink /J failed");
            drop(TempWorkDir(junction.clone()));
            assert!(junction.exists());
            assert!(sentinel.exists());
            fs::remove_dir(&junction).map_err(|e| e.to_string())?;
        }
        let non_temp = PathBuf::from("non-temp-relative-guard-path");
        drop(TempWorkDir(non_temp));
        Ok(())
    }
}
