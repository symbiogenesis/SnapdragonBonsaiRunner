//! One bounded, offline rebuild of the source-sanity checker.
#![forbid(unsafe_code)]

use std::process::ExitCode;

#[cfg(any(all(windows, target_arch = "aarch64"), test))]
mod policy {
    use std::ffi::OsString;
    use std::path::PathBuf;

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
        channel.len() == 18
            && channel.starts_with("nightly-")
            && channel.bytes().enumerate().skip(8).all(|(index, byte)| {
                if matches!(index, 12 | 15) {
                    byte == b'-'
                } else {
                    byte.is_ascii_digit()
                }
            })
    }

    pub(super) fn arguments(values: &[OsString]) -> Result<Arguments, String> {
        if values.len() != 8
            || values[0] != "--source"
            || values[2] != "--target-directory"
            || values[4] != "--toolchain"
            || values[6] != "--bootstrap-milliseconds"
        {
            return Err(USAGE.into());
        }
        let source = PathBuf::from(&values[1]);
        let target = PathBuf::from(&values[3]);
        let selected = values[5].to_str().filter(|value| toolchain(value));
        if !source.is_absolute() || !target.is_absolute() || selected.is_none() {
            return Err(
                "Recovery requires absolute paths and the dated native ARM64 compiler pin.".into(),
            );
        }
        let elapsed = values[7]
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

    pub(super) fn matching_pin(document: &str, selected: &str) -> bool {
        let mut in_toolchain = false;
        let mut channel = None;
        for line in document.lines() {
            let line = line.split('#').next().unwrap_or_default().trim();
            if line.starts_with('[') {
                in_toolchain = line == "[toolchain]";
                continue;
            }
            if in_toolchain {
                if let Some((key, value)) = line.split_once('=') {
                    if key.trim() == "channel" {
                        if channel.is_some() {
                            return false;
                        }
                        channel = value
                            .trim()
                            .strip_prefix('"')
                            .and_then(|v| v.strip_suffix('"'));
                    }
                }
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
        let stdout = scratch.join("cargo.stdout");
        let stderr = scratch.join("cargo.stderr");
        let command = vec![
            OsString::from(format!("+{}", arguments.toolchain)),
            "build".into(),
            "--quiet".into(),
            "--locked".into(),
            "--offline".into(),
            "--manifest-path".into(),
            manifest.into_os_string(),
            "--package".into(),
            "bonsai-ci".into(),
            "--bin".into(),
            "bonsai-ci".into(),
            "--target".into(),
            "aarch64-pc-windows-msvc".into(),
            "--target-dir".into(),
            target.into_os_string(),
        ];
        let environment = BTreeMap::from([
            ("CARGO_UNSTABLE_CHECKSUM_FRESHNESS".into(), "true".into()),
            ("CARGO_BUILD_FINGERPRINT".into(), "content".into()),
            ("CARGO_INCREMENTAL".into(), "1".into()),
        ]);
        let child = OwnedProcess::start_tool(&ToolProcessSpec {
            executable: cargo()?,
            arguments: command,
            working_directory: source,
            stdout: stdout.clone(),
            stderr: stderr.clone(),
            environment,
        })
        .map_err(|error| format!("Cannot start owned checker recovery: {error}"))?;
        let result: Result<u32, String> = (|| {
            loop {
                if started.elapsed() >= work || output_bytes(&stdout, &stderr)? > MAX_OUTPUT {
                    return Err("Push checker recovery exceeded its original time/output allowance; push blocked.".into());
                }
                if let Some(code) = child.try_wait().map_err(|error| error.to_string())? {
                    return Ok(code);
                }
                std::thread::sleep(Duration::from_millis(20));
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
        if output_bytes(&stdout, &stderr)? > MAX_OUTPUT {
            return Err(
                "Push checker recovery exceeded its final output bound; push blocked.".into(),
            );
        }
        let output = fs::read(&stdout).map_err(|error| error.to_string())?;
        let errors = fs::read(&stderr).map_err(|error| error.to_string())?;
        if code != 0 {
            let detail = [output, errors].concat();
            let excerpt = String::from_utf8_lossy(&detail[detail.len().saturating_sub(8192)..]);
            return Err(format!(
                "Push checker recovery failed with exit code {code}; push blocked.\n{excerpt}"
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
            eprintln!("{error}");
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
    fn recovery_admits_only_ordered_native_compiler_arguments() {
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
        let parsed = policy::arguments(&values).expect("native arguments");
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
