//! Windows Job supervision, temporary-index checkout and source checks.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bonsai_windows_process::{OwnedProcess, ToolProcessSpec};

use super::{Update, selection};
use crate::Result;

struct Work {
    directory: PathBuf,
    started: Instant,
    sequence: usize,
}

impl Work {
    fn command(
        &mut self,
        root: &Path,
        program: &str,
        arguments: &[OsString],
        environment: BTreeMap<String, String>,
    ) -> Result<Vec<u8>> {
        if self.started.elapsed() >= Duration::from_secs(270) {
            return Err("Push sanity aggregate allowance exhausted; push blocked.".into());
        }
        let executable =
            std::env::split_paths(&std::env::var_os("PATH").ok_or("PATH is unavailable")?)
                .map(|directory| directory.join(program))
                .find(|path| path.is_absolute() && path.is_file())
                .ok_or_else(|| format!("Push sanity requires {program} on PATH."))?;
        let name = self.sequence;
        self.sequence += 1;
        let stdout = self.directory.join(format!("{name}.stdout"));
        let stderr = self.directory.join(format!("{name}.stderr"));
        let child = OwnedProcess::start_tool(&ToolProcessSpec {
            executable,
            arguments: arguments.into(),
            working_directory: root.into(),
            stdout: stdout.clone(),
            stderr: stderr.clone(),
            environment,
        })
        .map_err(|error| error.to_string())?;
        let code = loop {
            if let Some(code) = child.try_wait().map_err(|error| error.to_string())? {
                break code;
            }
            if self.started.elapsed() >= Duration::from_secs(270)
                || [&stdout, &stderr].into_iter().any(|path| {
                    fs::metadata(path).is_ok_and(|metadata| metadata.len() > 4 * 1024 * 1024)
                })
            {
                child
                    .stop(Duration::from_secs(5))
                    .map_err(|error| error.to_string())?;
                return Err(
                    "Push sanity exceeded its aggregate allowance or output bound; push blocked."
                        .into(),
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        child
            .stop(Duration::from_secs(5))
            .map_err(|error| error.to_string())?;
        drop(child);
        let output = fs::read(&stdout).map_err(|error| error.to_string())?;
        if code != 0 {
            let mut detail = output;
            detail.extend(fs::read(&stderr).map_err(|error| error.to_string())?);
            let excerpt = String::from_utf8_lossy(&detail[detail.len().saturating_sub(8192)..]);
            return Err(format!(
                "Push blocked: {program} {arguments:?} exited {code}.\n{excerpt}"
            ));
        }
        Ok(output)
    }

    fn git(&mut self, root: &Path, arguments: &[&str]) -> Result<Vec<u8>> {
        self.command(
            root,
            "git.exe",
            &arguments.iter().map(OsString::from).collect::<Vec<_>>(),
            BTreeMap::new(),
        )
    }

    fn script(
        &mut self,
        source: &Path,
        policy: &Path,
        name: &str,
        arguments: &[OsString],
    ) -> Result<()> {
        let mut command = vec![
            "-NoLogo".into(),
            "-NoProfile".into(),
            "-NonInteractive".into(),
            "-File".into(),
            policy.join("tools/ci").join(name).into_os_string(),
        ];
        command.extend_from_slice(arguments);
        self.command(source, "pwsh.exe", &command, BTreeMap::new())?;
        Ok(())
    }

    fn snapshot(&mut self, root: &Path, commit: &str, number: usize) -> Result<PathBuf> {
        let source = self.directory.join(format!("source-{number}"));
        fs::create_dir(&source).map_err(|error| error.to_string())?;
        let index = self.directory.join(format!("index-{number}"));
        let environment = BTreeMap::from([(
            "GIT_INDEX_FILE".into(),
            index.to_string_lossy().into_owned(),
        )]);
        self.command(
            root,
            "git.exe",
            &["read-tree".into(), commit.into()],
            environment.clone(),
        )?;
        self.command(
            root,
            "git.exe",
            &[
                "checkout-index".into(),
                "--all".into(),
                format!("--prefix={}/", source.display()).into(),
            ],
            environment,
        )?;
        Ok(source)
    }

    fn checks(&mut self, root: &Path, source: &Path, paths: &[String]) -> Result<()> {
        let plan = selection(paths);
        self.script(
            source,
            root,
            "check_owned_languages.ps1",
            &["-Root".into(), source.into()],
        )?;
        let powershell: Vec<_> = plan
            .powershell
            .into_iter()
            .filter(|path| source.join(path).is_file())
            .collect();
        if !powershell.is_empty() {
            let paths_file = self.directory.join("powershell-paths.json");
            fs::write(
                &paths_file,
                serde_json::to_vec(&powershell).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            self.script(
                source,
                root,
                "check_powershell_syntax.ps1",
                &[
                    "-Root".into(),
                    source.into(),
                    "-PathsFile".into(),
                    paths_file.into_os_string(),
                ],
            )?;
        }
        if plan.rust {
            self.command(
                source,
                "cargo.exe",
                &["fmt".into(), "--all".into(), "--".into(), "--check".into()],
                BTreeMap::new(),
            )?;
        }
        if plan.workflows {
            self.script(
                source,
                root,
                "check_workflow_startup.ps1",
                &["-Root".into(), source.into()],
            )?;
            let mut arguments = vec!["-Root".into(), source.into()];
            if let Some(path) = std::env::var_os("BONSAI_ACTIONLINT_PATH") {
                arguments.extend(["-ActionlintPath".into(), path]);
            }
            self.script(source, root, "check_workflow_syntax.ps1", &arguments)?;
        }
        // Execute only named, model-free controls for the failure-producing adapters.
        // Fixtures read adjacent manifests, so use the immutable snapshot's script.
        for (selected, script, cases) in [
            (
                plan.cache_fixture,
                "test_tools_cache_inputs.ps1",
                &[
                    "mixed_case_names_have_same_build_identity",
                    "environment_fixture_preserves_absent_and_empty_values",
                ][..],
            ),
            (
                plan.source_fixture,
                "test_source_checks.ps1",
                &["main_push_always_records_source_verdict"][..],
            ),
        ] {
            if selected {
                for case in cases {
                    self.script(source, source, script, &["-Case".into(), (*case).into()])?;
                }
            }
        }
        Ok(())
    }
}

fn paths(raw: &[u8]) -> Result<Vec<String>> {
    raw.split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            std::str::from_utf8(path)
                .map(str::to_owned)
                .map_err(|error| error.to_string())
        })
        .collect()
}

pub(super) fn check(
    root: &Path,
    policy: &Path,
    updates: &[Update],
    bootstrap_milliseconds: u64,
) -> Result<String> {
    if !cfg!(target_arch = "aarch64") {
        return Err("Push sanity requires native Windows ARM64.".into());
    }
    if bootstrap_milliseconds >= 270_000 {
        return Err("Push sanity bootstrap exhausted the shared allowance; push blocked.".into());
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let temporary = super::canonical_path(std::env::temp_dir())?;
    let directory = temporary.join(format!("bonsai-push-sanity-{}-{nonce}", std::process::id()));
    fs::create_dir(&directory).map_err(|error| error.to_string())?;
    let before = bonsai_windows_process::power_status();
    let mut work = Work {
        directory,
        started: Instant::now()
            .checked_sub(Duration::from_millis(bootstrap_milliseconds))
            .ok_or("Push bootstrap start predates the monotonic clock.")?,
        sequence: 0,
    };
    let result = (|| {
        let mut commits = BTreeMap::<String, BTreeSet<String>>::new();
        for update in updates {
            let raw = work.git(
                root,
                &[
                    "rev-parse",
                    "--verify",
                    &format!("{}^{{commit}}", update.tip),
                ],
            )?;
            let commit = std::str::from_utf8(&raw)
                .map_err(|error| error.to_string())?
                .trim()
                .to_owned();
            let base = if update.base.bytes().all(|byte| byte == b'0') {
                None
            } else {
                work.git(
                    root,
                    &[
                        "rev-parse",
                        "--verify",
                        &format!("{}^{{commit}}", update.base),
                    ],
                )
                .ok()
            };
            let changed = if let Some(base) = base {
                let base = std::str::from_utf8(&base)
                    .map_err(|error| error.to_string())?
                    .trim();
                work.git(
                    root,
                    &[
                        "diff",
                        "--no-renames",
                        "--name-only",
                        "-z",
                        base,
                        &commit,
                        "--",
                    ],
                )?
            } else {
                work.git(root, &["ls-tree", "-r", "--name-only", "-z", &commit])?
            };
            commits.entry(commit).or_default().extend(paths(&changed)?);
        }
        for (number, (commit, paths)) in commits.iter().enumerate() {
            let source = work.snapshot(root, commit, number)?;
            work.checks(policy, &source, &paths.iter().cloned().collect::<Vec<_>>())?;
        }
        Ok(format!(
            "Push sanity passed for {} immutable outgoing source tip(s); hosted compilation, tests and release acceptance remain separate.",
            commits.len()
        ))
    })();
    let after = bonsai_windows_process::power_status();
    let mut output = std::io::stderr().lock();
    let _ = writeln!(
        output,
        "Push sanity native-work power before={before:?}; after={after:?}; pre-bootstrap power unknown; elapsed={:?}.",
        work.started.elapsed()
    );
    // Only remove this invocation's verified direct child of system temporary storage.
    if work.directory.parent() != Some(temporary.as_path()) {
        return Err("Push sanity scratch ownership changed.".into());
    }
    fs::remove_dir_all(&work.directory)
        .map_err(|error| format!("Push sanity cleanup failed: {error}"))?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_sanity_snapshot_is_independent_of_index_and_worktree() -> Result<()> {
        let temporary = super::super::canonical_path(std::env::temp_dir())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let scratch = temporary.join(format!("bonsai-push-fixture-{nonce}"));
        fs::create_dir(&scratch).map_err(|error| error.to_string())?;
        let mut work = Work {
            directory: scratch.clone(),
            started: Instant::now(),
            sequence: 0,
        };
        let result = (|| {
            let root = scratch.join("repository");
            fs::create_dir(&root).map_err(|error| error.to_string())?;
            work.git(&root, &["init", "--quiet"])?;
            fs::write(root.join("file with spaces.ps1"), "committed")
                .map_err(|error| error.to_string())?;
            work.git(&root, &["add", "--all"])?;
            work.git(
                &root,
                &[
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "core.hooksPath=",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--quiet",
                    "-m",
                    "baseline",
                ],
            )?;
            fs::write(root.join("file with spaces.ps1"), "staged repair")
                .map_err(|error| error.to_string())?;
            work.git(&root, &["add", "--all"])?;
            fs::write(root.join("file with spaces.ps1"), "unstaged repair")
                .map_err(|error| error.to_string())?;
            let index = work.git(&root, &["ls-files", "--stage", "-z"])?;
            let snapshot = work.snapshot(&root, "HEAD", 0)?;
            assert_eq!(
                fs::read_to_string(snapshot.join("file with spaces.ps1"))
                    .map_err(|error| error.to_string())?,
                "committed"
            );
            assert_eq!(
                fs::read_to_string(root.join("file with spaces.ps1"))
                    .map_err(|error| error.to_string())?,
                "unstaged repair"
            );
            assert_eq!(work.git(&root, &["ls-files", "--stage", "-z"])?, index);
            Ok(())
        })();
        if scratch.parent() != Some(temporary.as_path()) {
            return Err("Push fixture cleanup escaped temp.".into());
        }
        fs::remove_dir_all(scratch).map_err(|error| error.to_string())?;
        result
    }
}
