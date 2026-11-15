//! One bounded, offline rebuild of the source-sanity checker.
#![forbid(unsafe_code)]

use std::process::ExitCode;

#[cfg(any(all(windows, target_arch = "aarch64"), test))]
mod policy {
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};

    pub(super) const USAGE: &str = "bonsai-push-recovery --source PATH --target-directory PATH --toolchain nightly-YYYY-MM-DD-aarch64-pc-windows-msvc --bootstrap-milliseconds N";

    #[derive(Debug, PartialEq, Eq)]
    pub(super) struct Arguments {
        pub(super) source: PathBuf,
        pub(super) target: PathBuf,
        pub(super) toolchain: String,
        pub(super) elapsed: u64,
    }

    pub(super) fn toolchain(value: &str) -> bool {
        let Some(channel) = value.strip_suffix("-aarch64-pc-windows-msvc") else {
            return false;
        };
        let Some(date) = channel.strip_prefix("nightly-") else {
            return false;
        };
        matches!(
            date.as_bytes(),
            [
                y0, y1, y2, y3,
                b'-',
                m0, m1,
                b'-',
                d0, d1,
            ] if [y0, y1, y2, y3, m0, m1, d0, d1]
                .into_iter()
                .all(u8::is_ascii_digit)
        )
    }

    pub(super) fn arguments(values: &[OsString]) -> Result<Arguments, String> {
        let [
            flag_source,
            source_arg,
            flag_target,
            target_arg,
            flag_toolchain,
            toolchain_arg,
            flag_elapsed,
            elapsed_arg,
        ] = values
        else {
            return Err(USAGE.into());
        };
        if flag_source != "--source"
            || flag_target != "--target-directory"
            || flag_toolchain != "--toolchain"
            || flag_elapsed != "--bootstrap-milliseconds"
        {
            return Err(USAGE.into());
        }
        let source = PathBuf::from(source_arg);
        let target = PathBuf::from(target_arg);
        let selected = toolchain_arg.to_str().filter(|value| toolchain(value));
        if !source.is_absolute() || !target.is_absolute() || selected.is_none() {
            return Err(
                "Recovery requires absolute paths and the dated native ARM64 compiler pin.".into(),
            );
        }
        let elapsed = elapsed_arg
            .to_str()
            .ok_or(USAGE)?
            .parse::<u64>()
            .map_err(|_| USAGE)?;
        let toolchain = selected.ok_or(USAGE)?.to_owned();
        Ok(Arguments {
            source,
            target,
            toolchain,
            elapsed,
        })
    }

    pub(super) fn work_milliseconds(elapsed: u64) -> Result<u64, String> {
        // The following source checks keep at least thirty seconds; stopping
        // this owned compiler tree has a separate five-second reserve.
        let remaining = 270_000_u64
            .checked_sub(elapsed)
            .and_then(|remaining| remaining.checked_sub(35_000))
            .filter(|remaining| *remaining >= 1000)
            .ok_or("Push checker recovery cannot fit in the original allowance; push blocked.")?;
        Ok(remaining.min(90_000))
    }

    pub(super) fn cargo_arguments(
        arguments: &Arguments,
        manifest: &Path,
        target: &Path,
        nonce: &str,
    ) -> Vec<OsString> {
        vec![
            OsString::from(format!("+{}", arguments.toolchain)),
            "rustc".into(),
            "--quiet".into(),
            "--locked".into(),
            "--offline".into(),
            "--manifest-path".into(),
            manifest.into(),
            "--package".into(),
            "bonsai-ci".into(),
            "--bin".into(),
            "bonsai-ci".into(),
            "--target".into(),
            "aarch64-pc-windows-msvc".into(),
            "--target-dir".into(),
            target.into(),
            "--".into(),
            "-C".into(),
            format!("metadata=bonsai_push_recovery_{nonce}").into(),
        ]
    }

    pub(super) fn matching_pin(document: &str, selected: &str) -> bool {
        let mut in_toolchain = false;
        let mut channel = None;
        for line in document.lines() {
            let line = line.split('#').next().unwrap_or_default().trim();
            if line.starts_with('[') {
                in_toolchain = line == "[toolchain]";
                continue;
            }
            if in_toolchain
                && let Some((key, value)) = line.split_once('=')
                && key.trim() == "channel"
            {
                if channel.is_some() {
                    return false;
                }
                channel = value
                    .trim()
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'));
            }
        }
        channel.is_some_and(|channel| {
            selected.strip_suffix("-aarch64-pc-windows-msvc") == Some(channel)
        })
    }
}

#[cfg(all(windows, target_arch = "aarch64"))]
mod native {
    use super::policy;
    use bonsai_windows_process::{OwnedProcess, ToolProcessSpec};
    use std::collections::BTreeMap;
    use std::ffi::OsString;
    use std::fs;
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    const MAX_OUTPUT: u64 = 4 * 1024 * 1024;

    fn canonical(path: &Path) -> Result<PathBuf, String> {
        let path = fs::canonicalize(path).map_err(|error| error.to_string())?;
        let text = path.to_str().ok_or("Recovery paths must be UTF-8.")?;
        if text.starts_with("\\\\?\\UNC\\") {
            return Err("Recovery paths must be on a local Windows drive.".into());
        }
        Ok(PathBuf::from(text.strip_prefix("\\\\?\\").unwrap_or(text)))
    }

    fn cargo() -> Result<PathBuf, String> {
        std::env::split_paths(&std::env::var_os("PATH").ok_or("Recovery requires PATH.")?)
            .map(|directory| directory.join("cargo.exe"))
            .find(|path| path.is_absolute() && path.is_file())
            .ok_or_else(|| "Recovery requires cargo.exe on PATH.".into())
    }

    fn output_bytes(stdout: &Path, stderr: &Path) -> Result<u64, String> {
        fs::metadata(stdout)
            .and_then(|output| {
                fs::metadata(stderr).map(|errors| output.len().saturating_add(errors.len()))
            })
            .map_err(|error| error.to_string())
    }

    struct Work<'a> {
        source: &'a Path,
        scratch: &'a Path,
        executable: PathBuf,
        started: Instant,
        limit: Duration,
        output: u64,
    }

    impl Work<'_> {
        fn command(&mut self, name: &str, arguments: Vec<OsString>) -> Result<(), String> {
            if self.started.elapsed() >= self.limit {
                return Err("Push checker recovery exhausted its original allowance.".into());
            }
            let stdout = self.scratch.join(format!("{name}.stdout"));
            let stderr = self.scratch.join(format!("{name}.stderr"));
            let environment = BTreeMap::from([
                ("CARGO_UNSTABLE_CHECKSUM_FRESHNESS".into(), "true".into()),
                ("CARGO_BUILD_FINGERPRINT".into(), "content".into()),
                ("CARGO_INCREMENTAL".into(), "1".into()),
            ]);
            let child = OwnedProcess::start_tool(&ToolProcessSpec {
                executable: self.executable.clone(),
                arguments,
                working_directory: self.source.into(),
                stdout: stdout.clone(),
                stderr: stderr.clone(),
                environment,
            })
            .map_err(|error| format!("Cannot start owned checker recovery {name}: {error}"))?;
            let result: Result<u32, String> = (|| {
                loop {
                    let elapsed = self.started.elapsed();
                    if elapsed >= self.limit
                        || self.output.saturating_add(output_bytes(&stdout, &stderr)?) > MAX_OUTPUT
                    {
                        return Err("Push checker recovery exceeded its original time/output allowance; push blocked.".into());
                    }
                    let slice = (self.limit - elapsed).min(Duration::from_millis(20));
                    if let Some(code) = child
                        .wait_for_exit(slice)
                        .map_err(|error| error.to_string())?
                    {
                        return Ok(code);
                    }
                }
            })();
            let stopped = child
                .stop(Duration::from_secs(5))
                .map_err(|error| format!("Checker recovery tree cleanup failed: {error}"));
            drop(child);
            if let Err(error) = stopped {
                return Err(match result {
                    Err(primary) => format!("{primary}; {error}"),
                    Ok(_) => error,
                });
            }
            let code = result?;
            self.output = self.output.saturating_add(output_bytes(&stdout, &stderr)?);
            if self.output > MAX_OUTPUT {
                return Err(
                    "Push checker recovery exceeded its final output bound; push blocked.".into(),
                );
            }
            let output = fs::read(&stdout).map_err(|error| error.to_string())?;
            let errors = fs::read(&stderr).map_err(|error| error.to_string())?;
            if code != 0 {
                let detail = [output, errors].concat();
                let excerpt = if detail.is_empty() {
                    std::borrow::Cow::Borrowed("<empty stdout and stderr>")
                } else {
                    String::from_utf8_lossy(&detail[detail.len().saturating_sub(8192)..])
                };
                return Err(format!(
                    "Push checker recovery {name} failed with exit code {code}; push blocked.\n{excerpt}"
                ));
            }
            std::io::stdout()
                .lock()
                .write_all(&output)
                .map_err(|error| error.to_string())?;
            std::io::stderr()
                .lock()
                .write_all(&errors)
                .map_err(|error| error.to_string())?;
            Ok(())
        }
    }

    fn execute(
        arguments: &policy::Arguments,
        started: Instant,
        scratch: &Path,
    ) -> Result<(), String> {
        let source = canonical(&arguments.source)?;
        let manifest = source.join("Cargo.toml");
        if !manifest.is_file() {
            return Err("Recovery snapshot is missing Cargo.toml.".into());
        }
        let pin = fs::read_to_string(source.join("rust-toolchain.toml"))
            .map_err(|error| error.to_string())?;
        if !policy::matching_pin(&pin, &arguments.toolchain) {
            return Err("Recovery compiler differs from the committed snapshot's pin.".into());
        }
        fs::create_dir_all(&arguments.target).map_err(|error| error.to_string())?;
        let target = canonical(&arguments.target)?;
        if source == target || source.starts_with(&target) || target.starts_with(&source) {
            return Err(
                "Recovery build output must be separate from its committed snapshot.".into(),
            );
        }
        // Source intake and command composition share this invocation's start;
        // admitting Cargo never renews the recovery slice or aggregate clock.
        let work = Duration::from_millis(policy::work_milliseconds(arguments.elapsed)?);
        if started.elapsed() >= work {
            return Err("Push checker recovery setup exhausted its original allowance.".into());
        }
        let mut work = Work {
            source: &source,
            scratch,
            executable: cargo()?,
            started,
            limit: work,
            output: 0,
        };
        // A changed target-specific rustc argument dirties the checker binary
        // even when Cargo considers corrupted executable aliases fresh. Force
        // one relink without cleaning or blessing a legacy target directory;
        // dependency compilation receives no extra flags and remains cached.
        let nonce = format!(
            "{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_nanos(),
        );
        work.command(
            "relink",
            policy::cargo_arguments(arguments, &manifest, &target, &nonce),
        )
    }

    pub(super) fn run() -> Result<(), String> {
        let started = Instant::now();
        let arguments = policy::arguments(&std::env::args_os().skip(1).collect::<Vec<_>>())?;
        policy::work_milliseconds(arguments.elapsed)?;
        let temporary = canonical(&std::env::temp_dir())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let scratch = temporary.join(format!(
            "bonsai-push-recovery-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&scratch).map_err(|error| error.to_string())?;
        let result = execute(&arguments, started, &scratch);
        if scratch.parent() != Some(temporary.as_path()) {
            return Err("Checker recovery cleanup escaped temporary storage.".into());
        }
        let cleanup = fs::remove_dir_all(&scratch)
            .map_err(|error| format!("Checker recovery capture cleanup failed: {error}"));
        match (result, cleanup) {
            (Err(error), Err(cleanup)) => Err(format!("{error}; {cleanup}")),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }
}

fn main() -> ExitCode {
    #[cfg(all(windows, target_arch = "aarch64"))]
    let result = native::run();
    #[cfg(not(all(windows, target_arch = "aarch64")))]
    let result: Result<(), String> =
        Err("Push checker recovery requires native Windows ARM64.".into());
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ =
                std::io::Write::write_fmt(&mut std::io::stderr().lock(), format_args!("{error}\n"));
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::policy;
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn recovery_preserves_check_and_cleanup_reserves() {
        assert_eq!(policy::work_milliseconds(0), Ok(90_000));
        assert_eq!(policy::work_milliseconds(200_000), Ok(35_000));
        assert_eq!(policy::work_milliseconds(234_000), Ok(1000));
        for elapsed in [234_001, 235_000, 270_000, u64::MAX] {
            assert!(policy::work_milliseconds(elapsed).is_err());
        }
    }

    #[test]
    fn recovery_forces_only_the_checker_binary_to_relink() {
        let arguments = policy::Arguments {
            source: PathBuf::from("source"),
            target: PathBuf::from("target"),
            toolchain: "nightly-2026-11-12-aarch64-pc-windows-msvc".into(),
            elapsed: 0,
        };
        let first = policy::cargo_arguments(
            &arguments,
            &arguments.source.join("Cargo.toml"),
            &arguments.target,
            "first",
        );
        let second = policy::cargo_arguments(
            &arguments,
            &arguments.source.join("Cargo.toml"),
            &arguments.target,
            "second",
        );
        assert_eq!(first[1], "rustc");
        assert_eq!(first[2..5], ["--quiet", "--locked", "--offline"]);
        assert_eq!(
            first[7..11],
            ["--package", "bonsai-ci", "--bin", "bonsai-ci"]
        );
        assert_eq!(
            first[11..14],
            ["--target", "aarch64-pc-windows-msvc", "--target-dir"]
        );
        assert_eq!(first[15..17], ["--", "-C"]);
        assert_eq!(first[17], "metadata=bonsai_push_recovery_first");
        assert_eq!(second[17], "metadata=bonsai_push_recovery_second");
        assert_eq!(first[..17], second[..17]);
    }

    #[test]
    fn recovery_admits_only_ordered_native_compiler_arguments() -> Result<(), String> {
        let root = if cfg!(windows) {
            "C:/recovery"
        } else {
            "/recovery"
        };
        let values: Vec<OsString> = [
            "--source",
            root,
            "--target-directory",
            root,
            "--toolchain",
            "nightly-2026-11-12-aarch64-pc-windows-msvc",
            "--bootstrap-milliseconds",
            "17",
        ]
        .into_iter()
        .map(OsString::from)
        .collect();
        let parsed = policy::arguments(&values).map_err(|error| error.to_string())?;
        assert_eq!(parsed.source, PathBuf::from(root));
        assert_eq!(parsed.target, PathBuf::from(root));
        assert_eq!(parsed.elapsed, 17);
        assert_eq!(
            parsed.toolchain,
            "nightly-2026-11-12-aarch64-pc-windows-msvc"
        );
        for (index, poisoned) in [
            (0, "--target-directory"),
            (1, "relative"),
            (5, "stable-aarch64-pc-windows-msvc"),
            (7, "-1"),
        ] {
            let mut poisoned_values = values.clone();
            poisoned_values[index] = poisoned.into();
            assert!(policy::arguments(&poisoned_values).is_err());
        }
        assert!(policy::arguments(&values[..6]).is_err());
        Ok(())
    }

    #[test]
    fn recovery_requires_the_committed_compiler_pin() {
        let compiler = "nightly-2026-11-12-aarch64-pc-windows-msvc";
        assert!(policy::toolchain(compiler));
        assert!(policy::matching_pin(
            "[toolchain]\nchannel = \"nightly-2026-11-12\" # pin\nprofile = \"minimal\"",
            compiler
        ));
        for document in [
            "channel = \"nightly-2026-11-12\"",
            "[toolchain]\nchannel = \"nightly-2026-11-09\"",
            "[toolchain]\nchannel = \"nightly-2026-11-12\"\nchannel = \"nightly-2026-11-12\"",
            "[toolchain]\nchannel = \"nightly-2026-11-12\"\n[other]\nchannel = \"nightly-2026-11-09\"",
        ] {
            if document.contains("[other]") {
                assert!(policy::matching_pin(document, compiler));
            } else {
                assert!(!policy::matching_pin(document, compiler));
            }
        }
    }
}
