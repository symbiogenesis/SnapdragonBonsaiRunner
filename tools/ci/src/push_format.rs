//! Cargo owns target discovery; bounded Rustfmt workers only split its check call.

use std::ffi::OsString;
use std::path::Path;
#[cfg(windows)]
use std::path::PathBuf;

use crate::Result;

pub(super) const IMAGE: &str = "BONSAI_PUSH_RUSTFMT_IMAGE";
#[cfg(windows)]
pub(super) const BUDGET: &str = "BONSAI_PUSH_RUSTFMT_MILLISECONDS";

pub(super) fn selected(arguments: &[OsString]) -> bool {
    std::env::var_os(IMAGE).is_some()
        && arguments
            .first()
            .is_some_and(|argument| Path::new(argument).is_absolute())
}

#[cfg(windows)]
pub(super) fn toolchain(root: &Path) -> Result<String> {
    let pin: toml::Value = toml::from_str(
        &std::fs::read_to_string(root.join("rust-toolchain.toml"))
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let channel = pin
        .get("toolchain")
        .and_then(|value| value.get("channel"))
        .and_then(toml::Value::as_str)
        .filter(|channel| {
            channel.len() == 18
                && channel.starts_with("nightly-")
                && channel.bytes().enumerate().skip(8).all(|(index, byte)| {
                    if matches!(index, 12 | 15) {
                        byte == b'-'
                    } else {
                        byte.is_ascii_digit()
                    }
                })
        })
        .ok_or("Push formatting requires the outgoing source's dated compiler pin.")?;
    Ok(format!("{channel}-aarch64-pc-windows-msvc"))
}

#[cfg(any(windows, test))]
fn groups(arguments: &[OsString], workers: usize) -> Result<Vec<Vec<OsString>>> {
    let suffix = arguments
        .len()
        .checked_sub(3)
        .filter(|count| (1..=4096).contains(count))
        .ok_or("Invalid Cargo Rustfmt check handoff.")?;
    if arguments[suffix] != "--edition"
        || !matches!(
            arguments[suffix + 1].to_str(),
            Some("2015" | "2018" | "2021" | "2024")
        )
        || arguments[suffix + 2] != "--check"
        || arguments[..suffix]
            .iter()
            .any(|argument| !Path::new(argument).is_absolute())
    {
        return Err("Unsupported Cargo Rustfmt check handoff; no input was skipped.".into());
    }
    let count = workers.clamp(1, 4).min(suffix);
    let mut grouped = vec![Vec::new(); count];
    for (index, argument) in arguments[..suffix].iter().enumerate() {
        grouped[index % count].push(argument.clone());
    }
    for group in &mut grouped {
        group.extend_from_slice(&arguments[suffix..]);
    }
    Ok(grouped)
}

pub(super) fn run(arguments: &[OsString]) -> Result<String> {
    #[cfg(windows)]
    {
        let image = std::env::var_os(IMAGE).ok_or("Missing native Rustfmt image.")?;
        let milliseconds = std::env::var(BUDGET)
            .map_err(|error| error.to_string())?
            .parse::<u64>()
            .map_err(|error| error.to_string())?;
        execute(&PathBuf::from(image), arguments, milliseconds)
    }
    #[cfg(not(windows))]
    {
        let _ = arguments;
        Err("Native push formatting requires Windows ARM64.".into())
    }
}

#[cfg(windows)]
fn execute(image: &Path, arguments: &[OsString], milliseconds: u64) -> Result<String> {
    use std::collections::BTreeMap;
    use std::fs;
    use std::io::Write as _;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use bonsai_windows_process::{OwnedProcess, ToolProcessSpec};

    const MAX_OUTPUT: u64 = 4 * 1024 * 1024;
    let started = Instant::now();
    if !cfg!(target_arch = "aarch64")
        || !image.is_absolute()
        || !image.is_file()
        || image
            .file_name()
            .is_none_or(|name| !name.eq_ignore_ascii_case("rustfmt.exe"))
        || !(6000..=270_000).contains(&milliseconds)
    {
        return Err("Invalid native Rustfmt image or remaining allowance.".into());
    }
    let work_limit = Duration::from_millis(milliseconds - 5000);
    let temporary = super::push_sanity::canonical_path(std::env::temp_dir())?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let directory = temporary.join(format!("bonsai-push-format-{}-{nonce}", std::process::id()));
    let working = super::push_sanity::canonical_path(
        std::env::current_dir().map_err(|error| error.to_string())?,
    )?;
    let workers = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    // Preserve Rustfmt's single-session ordering for external targets or local
    // configuration overrides. The normal workspace shares its root config.
    let groups = if shared_configuration(arguments, &working) {
        groups(arguments, workers)?
    } else {
        groups(arguments, 1)?
    };
    fs::create_dir(&directory).map_err(|error| error.to_string())?;
    let mut jobs = Vec::<(OwnedProcess, PathBuf, PathBuf, Option<u32>)>::new();
    let result: Result<()> = (|| {
        for (number, arguments) in groups.iter().enumerate() {
            if started.elapsed() >= work_limit {
                return Err("Native Rustfmt startup exhausted its remaining allowance.".into());
            }
            let stdout = directory.join(format!("{number}.stdout"));
            let stderr = directory.join(format!("{number}.stderr"));
            let child = OwnedProcess::start_tool(&ToolProcessSpec {
                executable: image.into(),
                arguments: arguments.clone(),
                working_directory: working.clone(),
                stdout: stdout.clone(),
                stderr: stderr.clone(),
                environment: BTreeMap::new(),
            })
            .map_err(|error| format!("Cannot start native Rustfmt: {error}"))?;
            jobs.push((child, stdout, stderr, None));
        }
        loop {
            let total = jobs.iter().try_fold(0_u64, |sum, (_, stdout, stderr, _)| {
                let bytes = fs::metadata(stdout)
                    .and_then(|output| {
                        fs::metadata(stderr).map(|errors| output.len().saturating_add(errors.len()))
                    })
                    .map_err(|error| error.to_string())?;
                sum.checked_add(bytes)
                    .ok_or_else(|| "Rustfmt output length overflow.".to_owned())
            })?;
            if started.elapsed() >= work_limit || total > MAX_OUTPUT {
                return Err("Native Rustfmt exceeded its shared allowance or output bound.".into());
            }
            for (child, _, _, code) in &mut jobs {
                if code.is_none() {
                    *code = child.try_wait().map_err(|error| error.to_string())?;
                }
            }
            if jobs
                .iter()
                .any(|(_, _, _, code)| code.is_some_and(|code| code != 0))
                || jobs.iter().all(|(_, _, _, code)| code.is_some())
            {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    })();
    let cleanup_started = Instant::now();
    let mut cleanup = Ok(());
    for (child, _, _, _) in &jobs {
        let stop = Duration::from_secs(5)
            .checked_sub(cleanup_started.elapsed())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| "Native Rustfmt cleanup reserve exhausted.".to_owned())
            .and_then(|remaining| child.stop(remaining).map_err(|error| error.to_string()));
        if let Err(error) = stop {
            cleanup = Err(error);
        }
    }
    let output: Result<String> = (|| {
        match (result, cleanup) {
            (Err(error), Err(cleanup)) => {
                return Err(format!("{error}; cleanup also failed: {cleanup}"));
            }
            (Err(error), Ok(())) | (Ok(()), Err(error)) => return Err(error),
            (Ok(()), Ok(())) => {}
        }
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        for (_, output, errors, _) in &jobs {
            if fs::metadata(output)
                .map_err(|error| error.to_string())?
                .len()
                .saturating_add(
                    fs::metadata(errors)
                        .map_err(|error| error.to_string())?
                        .len(),
                )
                > MAX_OUTPUT
            {
                return Err("Native Rustfmt exceeded its final output bound.".into());
            }
            stdout.extend(fs::read(output).map_err(|error| error.to_string())?);
            stderr.extend(fs::read(errors).map_err(|error| error.to_string())?);
            if stdout.len().saturating_add(stderr.len())
                > usize::try_from(MAX_OUTPUT).map_err(|error| error.to_string())?
            {
                return Err("Native Rustfmt exceeded its final output bound.".into());
            }
        }
        std::io::stderr()
            .lock()
            .write_all(&stderr)
            .map_err(|error| error.to_string())?;
        if jobs
            .iter()
            .any(|(_, _, _, code)| code.is_some_and(|code| code != 0))
        {
            return Err(format!(
                "Native Rustfmt rejected an outgoing target.\n{}",
                String::from_utf8_lossy(&stdout)
            ));
        }
        String::from_utf8(stdout).map_err(|error| error.to_string())
    })();
    if directory.parent() != Some(temporary.as_path()) {
        return Err("Native Rustfmt cleanup escaped temporary storage.".into());
    }
    fs::remove_dir_all(&directory)
        .map_err(|error| format!("Native Rustfmt cleanup failed: {error}"))?;
    output
}

#[cfg(windows)]
fn shared_configuration(arguments: &[OsString], working: &Path) -> bool {
    arguments[..arguments.len() - 3].iter().all(|argument| {
        let Ok(root) = super::push_sanity::canonical_path(PathBuf::from(argument)) else {
            return false;
        };
        if !root.starts_with(working) {
            return false;
        }
        root.ancestors()
            .skip(1)
            .take_while(|parent| *parent != working)
            .all(|parent| {
                !["rustfmt.toml", ".rustfmt.toml"]
                    .iter()
                    .any(|name| parent.join(name).is_file())
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_format_covers_every_cargo_root_with_identical_flags() -> Result<()> {
        let roots: Vec<_> = (0..11)
            .map(|index| {
                std::env::temp_dir()
                    .join(format!("root {index}.rs"))
                    .into_os_string()
            })
            .collect();
        let mut arguments = roots.clone();
        arguments.extend(["--edition".into(), "2024".into(), "--check".into()]);
        let planned = groups(&arguments, 12)?;
        assert_eq!(planned.len(), 4);
        let mut observed = Vec::new();
        for group in planned {
            assert_eq!(&group[group.len() - 3..], &arguments[roots.len()..]);
            observed.extend_from_slice(&group[..group.len() - 3]);
        }
        observed.sort();
        let mut expected = roots;
        expected.sort();
        assert_eq!(observed, expected);
        for option in ["--emit=files", "--skip-children", "--help"] {
            let mut poisoned = arguments.clone();
            poisoned.push(option.into());
            assert!(groups(&poisoned, 4).is_err());
        }
        assert!(
            groups(
                &[
                    "relative.rs".into(),
                    "--edition".into(),
                    "2024".into(),
                    "--check".into()
                ],
                4
            )
            .is_err()
        );
        Ok(())
    }
}
