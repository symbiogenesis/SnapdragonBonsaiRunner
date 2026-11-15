//! Windows Job supervision, temporary-index checkout and source checks.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bonsai_windows_process::{OwnedProcess, ToolProcessSpec};

use super::{Selection, Update, selection};
use crate::Result;

#[derive(serde::Serialize)]
struct ScriptCall {
    script: &'static str,
    parameters: BTreeMap<&'static str, serde_json::Value>,
}

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
        let command_started = Instant::now();
        if program == "pwsh.exe" {
            // Packaged PowerShell rejects suspended direct creation. CMD starts
            // it normally inside our already-owned Job; no breakaway is enabled.
            let image =
                std::env::var_os("BONSAI_PUSH_PWSH_PATH").unwrap_or_else(|| "pwsh.exe".into());
            let mut words = vec![image];
            words.extend_from_slice(arguments);
            let words = words
                .iter()
                .map(|word| {
                    let text = word
                        .to_str()
                        .ok_or("PowerShell startup arguments must be UTF-8.")?;
                    if text.contains(['\"', '\r', '\n', '\0']) {
                        return Err("Unsupported PowerShell startup argument.".into());
                    }
                    Ok(format!("\"{}\"", text.replace('%', "%%")))
                })
                .collect::<Result<Vec<_>>>()?;
            let batch = self.directory.join(format!("shell-{}.cmd", self.sequence));
            fs::write(
                &batch,
                format!(
                    "@echo off\r\nchcp 65001 >nul\r\nset \"ERRORLEVEL=\"\r\n{}\r\nexit /b %errorlevel%\r\n",
                    words.join(" ")
                ),
            )
            .map_err(|error| error.to_string())?;
            return self.command(
                root,
                "cmd.exe",
                &["/d".into(), "/c".into(), batch.into_os_string()],
                environment,
            );
        }
        let search = std::env::var_os("PATH").unwrap_or_default();
        let executable = std::env::split_paths(&search)
            .map(|directory| directory.join(program))
            .find(|path| path.is_absolute() && path.is_file())
            .ok_or_else(|| format!("Push sanity requires {program} on PATH."))?;
        if !executable.is_absolute() || !executable.is_file() {
            return Err(format!(
                "Invalid native tool image: {}",
                executable.display()
            ));
        }
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
        .map_err(|error| format!("Cannot start {program} in {}: {error}", root.display()))?;
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
        if [&stdout, &stderr]
            .into_iter()
            .any(|path| fs::metadata(path).is_ok_and(|metadata| metadata.len() > 4 * 1024 * 1024))
        {
            return Err("Push command output exceeded its final byte bound; push blocked.".into());
        }
        let output = fs::read(&stdout).map_err(|error| error.to_string())?;
        if std::env::var("BONSAI_PUSH_TIMINGS").as_deref() == Ok("1") {
            let _ = writeln!(
                std::io::stderr().lock(),
                "Push sanity timing: {program} {arguments:?}: {:?}.",
                command_started.elapsed()
            );
        }
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
        if !plan.markdown.is_empty() {
            super::super::push_markdown::check(source, &plan.markdown)?;
        }
        if plan.workspace_registry
            && (source.join("config/rust-packages.json").exists()
                || paths.iter().any(|path| {
                    path == "config/rust-packages.json" || path == ".github/ci/impact.json"
                }))
        {
            super::super::checks::verify_workspace_registry(source)?;
        }
        if plan.native_boundary
            && (source.join("tools/ci/native-boundary.runner.json").exists()
                || paths
                    .iter()
                    .any(|path| path == "tools/ci/native-boundary.runner.json"))
        {
            super::super::native_boundary::verify_in_tree(source)?;
        }
        if plan.hexagon_locks {
            verify_hexagon_locks(source, paths)?;
        }
        let mut calls = vec![ScriptCall {
            script: "check_owned_languages.ps1",
            parameters: BTreeMap::from([("Root", serde_json::json!(source))]),
        }];
        let powershell: Vec<_> = plan
            .powershell
            .iter()
            .filter(|path| source.join(path.as_str()).is_file())
            .cloned()
            .collect();
        if !powershell.is_empty() {
            let paths_file = self.directory.join("powershell-paths.json");
            fs::write(
                &paths_file,
                serde_json::to_vec(&powershell).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            calls.push(ScriptCall {
                script: "check_powershell_syntax.ps1",
                parameters: BTreeMap::from([
                    ("Root", serde_json::json!(source)),
                    ("PathsFile", serde_json::json!(paths_file)),
                ]),
            });
        }
        if plan.workflows {
            calls.push(ScriptCall {
                script: "check_workflow_startup.ps1",
                parameters: BTreeMap::from([("Root", serde_json::json!(source))]),
            });
            let mut parameters = BTreeMap::from([("Root", serde_json::json!(source))]);
            if let Some(path) = std::env::var_os("BONSAI_ACTIONLINT_PATH") {
                let path = path
                    .into_string()
                    .map_err(|_| "Push sanity actionlint path must be UTF-8.")?;
                parameters.insert("ActionlintPath", serde_json::json!(path));
            }
            calls.push(ScriptCall {
                script: "check_workflow_syntax.ps1",
                parameters,
            });
        }
        calls.extend(Self::adapter_calls(source, &plan));
        // Adapter controls execute candidate bodies. Inspect Rust formatting
        // before those bodies can modify the materialized committed snapshot.
        if plan.rust {
            self.format_check(source)?;
        }
        self.script_calls(root, source, &calls)?;
        #[cfg(not(test))]
        if plan.rust || plan.workspace_registry || plan.hexagon_locks {
            self.cargo_checks(source, paths, &plan)?;
        }
        Ok(())
    }

    fn format_check(&mut self, source: &Path) -> Result<()> {
        let toolchain = super::super::push_format::toolchain(source)?;
        #[cfg(test)]
        let environment = BTreeMap::new();
        #[cfg(not(test))]
        let environment = {
            let image = self.command(
                source,
                "rustup.exe",
                &[
                    "which".into(),
                    "--toolchain".into(),
                    toolchain.clone().into(),
                    "rustfmt".into(),
                ],
                BTreeMap::new(),
            )?;
            let image = std::str::from_utf8(&image)
                .map_err(|error| error.to_string())?
                .trim();
            let adapter = std::env::current_exe().map_err(|error| error.to_string())?;
            let remaining = 270_000_u128
                .checked_sub(self.started.elapsed().as_millis())
                .filter(|remaining| *remaining >= 6000)
                .ok_or("Push sanity has no formatting work and cleanup allowance remaining.")?;
            BTreeMap::from([
                (
                    "RUSTFMT".into(),
                    adapter
                        .to_str()
                        .ok_or("Formatter adapter path must be UTF-8.")?
                        .into(),
                ),
                (super::super::push_format::IMAGE.into(), image.into()),
                (
                    super::super::push_format::BUDGET.into(),
                    remaining.to_string(),
                ),
            ])
        };
        self.command(
            source,
            "cargo.exe",
            &[
                format!("+{toolchain}").into(),
                "fmt".into(),
                "--all".into(),
                "--".into(),
                "--check".into(),
            ],
            environment,
        )?;
        Ok(())
    }

    fn script_calls(&mut self, policy: &Path, source: &Path, calls: &[ScriptCall]) -> Result<()> {
        let calls_file = self
            .directory
            .join(format!("script-calls-{}.json", self.sequence));
        fs::write(
            &calls_file,
            serde_json::to_vec(calls).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        self.script(
            source,
            policy,
            "check_push_snapshot.ps1",
            &["-CallsFile".into(), calls_file.into_os_string()],
        )
    }

    fn adapter_calls(source: &Path, plan: &Selection) -> Vec<ScriptCall> {
        // Assertions and support come from the committed checking policy. Inputs
        // come from this outgoing snapshot even when it is not checking HEAD.
        // Candidate fixture scripts are never used, so later controls can also
        // exercise historical adapters that did not yet contain those fixtures.
        let mut calls = Vec::new();
        for action in &plan.cache_actions {
            for case in [
                "mixed_case_names_have_same_build_identity",
                "environment_fixture_preserves_absent_and_empty_values",
            ] {
                calls.push(ScriptCall {
                    script: "test_tools_cache_inputs.ps1",
                    parameters: BTreeMap::from([
                        ("Case", serde_json::json!(case)),
                        ("CandidateRoot", serde_json::json!(source)),
                        ("EnvironmentAction", serde_json::json!(action)),
                    ]),
                });
            }
        }
        if plan.cache_closures
            && source
                .join(".github/actions/check-rust/action.yml")
                .is_file()
            && source
                .join(".github/actions/bonsai-tools/action.yml")
                .is_file()
            && source.join("Cargo.lock").is_file()
        {
            for case in [
                "lane_dependency_and_embedded_closures_hashed",
                "real_dependency_embedded_closure_hashed",
            ] {
                calls.push(ScriptCall {
                    script: "test_tools_cache_inputs.ps1",
                    parameters: BTreeMap::from([
                        ("Case", serde_json::json!(case)),
                        ("CandidateRoot", serde_json::json!(source)),
                    ]),
                });
            }
        }
        if plan.source_fixture {
            calls.push(ScriptCall {
                script: "test_source_checks.ps1",
                parameters: BTreeMap::from([
                    (
                        "Case",
                        serde_json::json!("main_push_always_records_source_verdict"),
                    ),
                    ("CandidateRoot", serde_json::json!(source)),
                ]),
            });
        }
        calls
    }

    #[cfg(not(test))]
    fn cargo_checks(&mut self, source: &Path, paths: &[String], plan: &Selection) -> Result<()> {
        if !source.join("config/rust-packages.json").is_file()
            || !source.join(".github/ci/impact.json").is_file()
        {
            return Ok(());
        }
        let impact_bytes =
            fs::read(source.join(".github/ci/impact.json")).map_err(|error| error.to_string())?;
        let impact_config: serde_json::Value =
            serde_json::from_slice(&impact_bytes).map_err(|error| error.to_string())?;
        let impact_plan = crate::impact::select(source, paths, false, &impact_config, "pre-push")?;
        let affected: BTreeSet<String> = impact_plan.packages.into_iter().collect();
        let toolchain = super::super::push_format::toolchain(source)?;
        let target_dir = std::env::var_os("BONSAI_PUSH_CARGO_TARGET_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| self.directory.join("cargo-target"));
        fs::create_dir_all(&target_dir).map_err(|error| error.to_string())?;
        let manifest = source.join("Cargo.toml");

        let base_cargo_args = |subcommand: &str, target: &str| -> Vec<OsString> {
            vec![
                format!("+{toolchain}").into(),
                subcommand.into(),
                "--quiet".into(),
                "--locked".into(),
                "--offline".into(),
                "--manifest-path".into(),
                manifest.as_os_str().into(),
                "--target".into(),
                target.into(),
                "--target-dir".into(),
                target_dir.as_os_str().into(),
            ]
        };

        if affected.contains("bonsai-ci") || plan.workspace_registry {
            let mut routing_args = base_cargo_args("test", "aarch64-pc-windows-msvc");
            routing_args.extend([
                "-p".into(),
                "bonsai-ci".into(),
                "--test".into(),
                "routing".into(),
            ]);
            self.command(source, "cargo.exe", &routing_args, BTreeMap::new())?;

            let mut lib_args = base_cargo_args("test", "aarch64-pc-windows-msvc");
            lib_args.extend([
                "-p".into(),
                "bonsai-ci".into(),
                "--lib".into(),
                "--".into(),
                "--skip".into(),
                "push_sanity_".into(),
            ]);
            self.command(source, "cargo.exe", &lib_args, BTreeMap::new())?;
        }

        if plan.hexagon_locks
            || (affected.contains("bonsai-tools")
                && paths.iter().any(|path| path.contains("hexagon")))
        {
            let mut hex_args = base_cargo_args("test", "aarch64-pc-windows-msvc");
            hex_args.extend([
                "-p".into(),
                "bonsai-tools".into(),
                "--lib".into(),
                "hexagon_tools::public_control".into(),
            ]);
            self.command(source, "cargo.exe", &hex_args, BTreeMap::new())?;
        }

        if affected.contains("bonsai-tools")
            && paths.iter().any(|path| {
                path == "tools/bonsai/src/release_package.rs"
                    || path.starts_with("scripts/release/")
            })
        {
            let mut rel_args = base_cargo_args("test", "aarch64-pc-windows-msvc");
            rel_args.extend([
                "-p".into(),
                "bonsai-tools".into(),
                "--lib".into(),
                "release_package".into(),
            ]);
            self.command(source, "cargo.exe", &rel_args, BTreeMap::new())?;
        }

        let windows_standard: Vec<_> = affected
            .iter()
            .filter(|package| package.as_str() != "bonsai-speech-asr")
            .cloned()
            .collect();
        if !windows_standard.is_empty() {
            let mut args = base_cargo_args("check", "aarch64-pc-windows-msvc");
            args.push("--all-targets".into());
            for package in &windows_standard {
                args.push("-p".into());
                args.push(package.into());
            }
            self.command(source, "cargo.exe", &args, BTreeMap::new())?;
        }

        if affected.contains("bonsai-speech-asr") {
            let mut args = base_cargo_args("check", "aarch64-pc-windows-msvc");
            args.extend([
                "-p".into(),
                "bonsai-speech-asr".into(),
                "--no-default-features".into(),
                "--features".into(),
                "rust-qualification".into(),
                "--all-targets".into(),
            ]);
            self.command(source, "cargo.exe", &args, BTreeMap::new())?;
        }

        const PORTABLE_LINUX_CRATES: [&str; 6] = [
            "bonsai-artifacts",
            "bonsai-audio",
            "bonsai-cpu-policy",
            "bonsai-broker",
            "bonsai-markdown",
            "bonsai-speech-service",
        ];
        let linux_affected: Vec<_> = affected
            .iter()
            .filter(|package| PORTABLE_LINUX_CRATES.contains(&package.as_str()))
            .cloned()
            .collect();
        if !linux_affected.is_empty() {
            let installed = self.command(
                source,
                "rustup.exe",
                &[
                    "target".into(),
                    "list".into(),
                    "--installed".into(),
                    "--toolchain".into(),
                    toolchain.clone().into(),
                ],
                BTreeMap::new(),
            )?;
            let installed_text =
                std::str::from_utf8(&installed).map_err(|error| error.to_string())?;
            if installed_text
                .lines()
                .any(|line| line.trim() == "aarch64-unknown-linux-gnu")
            {
                let mut args = base_cargo_args("check", "aarch64-unknown-linux-gnu");
                args.push("--lib".into());
                for package in &linux_affected {
                    args.push("-p".into());
                    args.push(package.into());
                }
                self.command(source, "cargo.exe", &args, BTreeMap::new())?;
            }
        }

        Ok(())
    }

    #[cfg(test)]
    fn adapter_checks(&mut self, policy: &Path, source: &Path, plan: &Selection) -> Result<()> {
        self.script_calls(policy, source, &Self::adapter_calls(source, plan))
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

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn verify_lockfiles_in_dir(dir: &Path, prefix: Option<&str>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.ends_with(".lock.json") {
            continue;
        }
        if let Some(expected_prefix) = prefix
            && !name.starts_with(expected_prefix)
        {
            continue;
        }
        if entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_file()
        {
            let bytes = fs::read(&path).map_err(|error| error.to_string())?;
            let _: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
                format!(
                    "Push blocked: invalid Hexagon lockfile JSON {}: {error}",
                    path.display()
                )
            })?;
        }
    }
    Ok(())
}

fn verify_hexagon_locks(source: &Path, paths: &[String]) -> Result<()> {
    verify_lockfiles_in_dir(&source.join("tools/hexagon"), None)?;
    verify_lockfiles_in_dir(&source.join("config"), Some("hexagon-"))?;

    let lock_rel = "tools/hexagon/windows-queue-abi.lock.json";
    let c_rel = "tools/hexagon/windows_queue_abi.c";
    let lock_path = source.join(lock_rel);
    let c_path = source.join(c_rel);
    if !lock_path.is_file() {
        if paths.iter().any(|path| path == lock_rel || path == c_rel) || c_path.is_file() {
            return Err(format!("Push blocked: missing {lock_rel}."));
        }
        return Ok(());
    }

    let raw = fs::read(&lock_path).map_err(|error| error.to_string())?;
    let document: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|error| format!("Push blocked: invalid {lock_rel}: {error}"))?;
    if document
        .get("schemaVersion")
        .and_then(serde_json::Value::as_u64)
        != Some(1)
    {
        return Err(format!(
            "Push blocked: {lock_rel} must declare integer schemaVersion 1."
        ));
    }
    let inputs = document
        .get("publicInputs")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("Push blocked: {lock_rel} must declare publicInputs array."))?;
    let mut pinned = BTreeSet::new();
    for input in inputs {
        let object = input.as_object().ok_or_else(|| {
            format!("Push blocked: {lock_rel} publicInputs entries must be JSON objects.")
        })?;
        let name = object
            .get("name")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("Push blocked: {lock_rel} publicInput missing name."))?;
        if !pinned.insert(name.to_owned()) {
            return Err(format!(
                "Push blocked: duplicate publicInput '{name}' in {lock_rel}."
            ));
        }
        let bytes = object
            .get("bytes")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("Push blocked: {lock_rel} entry '{name}' invalid bytes."))?;
        let _ = bytes;
        let sha256 = object
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|value| is_lower_hex_digest(value))
            .ok_or_else(|| format!("Push blocked: {lock_rel} entry '{name}' invalid sha256."))?;
        let _ = sha256;
        let revision = object
            .get("revision")
            .and_then(serde_json::Value::as_str)
            .filter(|value| {
                matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            .ok_or_else(|| format!("Push blocked: {lock_rel} entry '{name}' invalid revision."))?;
        let _ = revision;
        let url = object
            .get("url")
            .and_then(serde_json::Value::as_str)
            .filter(|value| value.starts_with("https://"))
            .ok_or_else(|| format!("Push blocked: {lock_rel} entry '{name}' invalid url."))?;
        let _ = url;
    }

    if !pinned.contains("AEEStdDef.h") {
        return Err(format!(
            "Push blocked: {lock_rel} must pin transitive public header AEEStdDef.h."
        ));
    }

    if c_path.is_file() {
        let c_text = fs::read_to_string(&c_path).map_err(|error| error.to_string())?;
        for (index, line) in c_text.lines().enumerate() {
            let trimmed = line.trim_start();
            if let Some(after_include) = trimmed.strip_prefix("#include") {
                let after = after_include.trim_start();
                if let Some(quoted) = after.strip_prefix('"')
                    && let Some((header, _)) = quoted.split_once('"')
                    && !pinned.contains(header)
                {
                    return Err(format!(
                        "Push blocked: {c_rel}:{} includes \"{header}\" which is missing from {lock_rel}.",
                        index + 1
                    ));
                }
            }
        }
    }

    Ok(())
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

    fn preparation_case(case: &str) -> Result<()> {
        let temporary = super::super::canonical_path(std::env::temp_dir())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let scratch = temporary.join(format!("bonsai-push-preparation-{nonce}"));
        fs::create_dir(&scratch).map_err(|error| error.to_string())?;
        let mut work = Work {
            directory: scratch.clone(),
            started: Instant::now(),
            sequence: 0,
        };
        // This bootstrap adapter requires Windows. Foundation's native CI
        // library lane owns it; the Linux planner does not exercise WinForms.
        let policy =
            super::super::canonical_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))?;
        let result = work.script(
            &policy,
            &policy,
            "test_push_sanity.ps1",
            &["-Case".into(), case.into()],
        );
        if scratch.parent() != Some(temporary.as_path()) {
            return Err("Push preparation fixture cleanup escaped temp.".into());
        }
        fs::remove_dir_all(scratch).map_err(|error| error.to_string())?;
        result
    }

    #[test]
    fn push_sanity_fresh_preparation_reaches_committed_bootstrap() -> Result<()> {
        preparation_case("fresh_preparation_reaches_committed_bootstrap")
    }

    #[test]
    fn push_sanity_failed_preparation_invalidates_existing_receipt() -> Result<()> {
        preparation_case("failed_preparation_invalidates_existing_receipt")
    }

    #[test]
    fn push_sanity_ordinary_push_archives_only_committed_policy() -> Result<()> {
        preparation_case("ordinary_push_archives_only_committed_policy")
    }

    #[test]
    fn push_sanity_stale_checker_recovers_from_committed_source() -> Result<()> {
        preparation_case("stale_checker_recovers_from_committed_source")
    }

    #[test]
    fn push_sanity_malformed_checker_receipt_recovers_from_committed_source() -> Result<()> {
        preparation_case("malformed_checker_receipt_recovers_from_committed_source")
    }

    #[test]
    fn push_sanity_corrupt_checker_recovers_from_committed_source() -> Result<()> {
        preparation_case("corrupt_checker_recovers_from_committed_source")
    }

    #[test]
    fn push_sanity_missing_checker_receipt_recovers_from_committed_source() -> Result<()> {
        preparation_case("missing_checker_receipt_recovers_from_committed_source")
    }

    #[test]
    fn push_sanity_unverified_supervisor_blocks_recovery() -> Result<()> {
        preparation_case("unverified_supervisor_blocks_recovery")
    }

    #[test]
    fn push_sanity_failed_automatic_recovery_invalidates_existing_receipt() -> Result<()> {
        preparation_case("failed_automatic_recovery_invalidates_existing_receipt")
    }

    #[test]
    fn push_sanity_collection_checker_source_recovers_from_committed_source() -> Result<()> {
        preparation_case("collection_checker_source_recovers_from_committed_source")
    }

    #[test]
    fn push_sanity_collection_checker_digest_recovers_from_committed_source() -> Result<()> {
        preparation_case("collection_checker_digest_recovers_from_committed_source")
    }

    #[test]
    fn push_sanity_collection_recovery_source_blocks_recovery() -> Result<()> {
        preparation_case("collection_recovery_source_blocks_recovery")
    }

    #[test]
    fn push_sanity_collection_recovery_digest_blocks_recovery() -> Result<()> {
        preparation_case("collection_recovery_digest_blocks_recovery")
    }

    #[test]
    fn push_sanity_script_batch_preserves_scopes_and_rejections() -> Result<()> {
        let temporary = super::super::canonical_path(std::env::temp_dir())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let scratch = temporary.join(format!("bonsai-push-script-batch-{nonce}"));
        fs::create_dir(&scratch).map_err(|error| error.to_string())?;
        let mut work = Work {
            directory: scratch.clone(),
            started: Instant::now(),
            sequence: 0,
        };
        let policy =
            super::super::canonical_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))?;
        let result = work.script(&policy, &policy, "test_push_snapshot.ps1", &[]);
        if scratch.parent() != Some(temporary.as_path()) {
            return Err("Push batch fixture cleanup escaped temp.".into());
        }
        fs::remove_dir_all(scratch).map_err(|error| error.to_string())?;
        result
    }

    #[test]
    fn push_sanity_checks_format_before_candidate_adapter_execution() -> Result<()> {
        let temporary = super::super::canonical_path(std::env::temp_dir())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let scratch = temporary.join(format!("bonsai-push-format-order-{nonce}"));
        fs::create_dir(&scratch).map_err(|error| error.to_string())?;
        let mut work = Work {
            directory: scratch.clone(),
            started: Instant::now(),
            sequence: 0,
        };
        let result = (|| {
            let policy =
                super::super::canonical_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))?;
            let source = scratch.join("candidate");
            fs::create_dir_all(source.join("src")).map_err(|error| error.to_string())?;
            fs::write(
                source.join("Cargo.toml"),
                "[package]\nname = \"push-format-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[workspace]\n",
            )
            .map_err(|error| error.to_string())?;
            fs::copy(
                policy.join("rust-toolchain.toml"),
                source.join("rust-toolchain.toml"),
            )
            .map_err(|error| error.to_string())?;
            let rust_source = source.join("src/main.rs");
            let unformatted = "fn main( ){ }\n";
            fs::write(&rust_source, unformatted).map_err(|error| error.to_string())?;
            let escaped = rust_source
                .to_str()
                .ok_or("Format fixture path must be UTF-8.")?
                .replace('\'', "''");
            for action in ["bonsai-tools", "ci-tool"] {
                let relative = format!(".github/actions/{action}/action.yml");
                let text = fs::read_to_string(policy.join(&relative))
                    .map_err(|error| error.to_string())?;
                // Retain the harness's exact extraction start, then repair the
                // Rust file without changing the original environment hash.
                let mutated = text.replace(
                    "$buildInputs = @(",
                    &format!(
                        "$buildInputs = $null; [IO.File]::WriteAllText('{escaped}', 'fn main() {{}}' + [Environment]::NewLine); $buildInputs = @("
                    ),
                );
                assert_ne!(mutated, text, "Cache mutation must change {action}.");
                let target = source.join(relative);
                fs::create_dir_all(target.parent().ok_or("Missing action parent.")?)
                    .map_err(|error| error.to_string())?;
                fs::write(target, mutated).map_err(|error| error.to_string())?;
            }
            let failure = work
                .checks(
                    &policy,
                    &source,
                    &[
                        "src/main.rs".into(),
                        "tools/ci/test_tools_cache_inputs.ps1".into(),
                    ],
                )
                .err()
                .ok_or("Committed formatting must fail before candidate repair.")?;
            assert!(failure.contains("cargo.exe"), "{failure}");
            assert_eq!(
                fs::read_to_string(rust_source).map_err(|error| error.to_string())?,
                unformatted
            );
            Ok(())
        })();
        if scratch.parent() != Some(temporary.as_path()) {
            return Err("Push format fixture cleanup escaped temp.".into());
        }
        fs::remove_dir_all(scratch).map_err(|error| error.to_string())?;
        result
    }

    #[test]
    fn push_sanity_trusted_controls_reject_non_head_candidate_adapters() -> Result<()> {
        let temporary = super::super::canonical_path(std::env::temp_dir())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let scratch = temporary.join(format!("bonsai-push-adapters-{nonce}"));
        fs::create_dir(&scratch).map_err(|error| error.to_string())?;
        let mut work = Work {
            directory: scratch.clone(),
            started: Instant::now(),
            sequence: 0,
        };
        let result = (|| {
            let policy =
                super::super::canonical_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))?;
            let root = scratch.join("repository");
            fs::create_dir(&root).map_err(|error| error.to_string())?;
            work.git(&root, &["init", "--quiet"])?;
            let adapters = [
                ".github/actions/bonsai-tools/action.yml",
                ".github/actions/ci-tool/action.yml",
                "tools/ci/select_source_checks.ps1",
            ];
            let mut originals = Vec::new();
            for name in adapters {
                let text =
                    fs::read_to_string(policy.join(name)).map_err(|error| error.to_string())?;
                let broken = if name.ends_with("action.yml") {
                    text.replace(
                        "$buildInputs = @(",
                        "$buildInputs = @('constant'); $ignoredInputs = @(",
                    )
                } else {
                    text.replace(
                        "$selected.audit=$eligible.audit -and",
                        "$selected.audit=$eligible.audit -or",
                    )
                };
                assert_ne!(text, broken, "The adapter mutation must change {name}.");
                let path = root.join(name);
                fs::create_dir_all(path.parent().ok_or("Missing adapter parent.")?)
                    .map_err(|error| error.to_string())?;
                fs::write(&path, broken).map_err(|error| error.to_string())?;
                originals.push((path, text));
            }
            for name in ["source-checks", "foundation"] {
                let relative = format!(".github/workflows/{name}.yml");
                let path = root.join(&relative);
                fs::create_dir_all(path.parent().ok_or("Missing workflow parent.")?)
                    .map_err(|error| error.to_string())?;
                fs::copy(policy.join(relative), path).map_err(|error| error.to_string())?;
            }
            // An outgoing candidate cannot replace trusted fixture assertions
            // or support with its own successful or failing script.
            for name in [
                "test_tools_cache_inputs",
                "test_source_checks",
                "test_support",
            ] {
                fs::write(
                    root.join(format!("tools/ci/{name}.ps1")),
                    "throw 'Candidate fixture must never execute.'",
                )
                .map_err(|error| error.to_string())?;
            }
            work.git(&root, &["add", "--all"])?;
            let commit = [
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
                "adapter fixture",
            ];
            work.git(&root, &commit)?;
            for (path, text) in originals {
                fs::write(path, text).map_err(|error| error.to_string())?;
            }
            work.git(&root, &["add", "--all"])?;
            work.git(&root, &commit)?;
            let broken = work.snapshot(&root, "HEAD~1", 0)?;
            for path in adapters {
                assert!(
                    work.adapter_checks(&policy, &broken, &selection(&[path.into()]))
                        .is_err(),
                    "Trusted controls must reject the non-HEAD candidate's {path}."
                );
            }
            let repaired = work.snapshot(&root, "HEAD", 1)?;
            work.adapter_checks(&policy, &repaired, &selection(&adapters.map(String::from)))
        })();
        if scratch.parent() != Some(temporary.as_path()) {
            return Err("Push adapter fixture cleanup escaped temp.".into());
        }
        fs::remove_dir_all(scratch).map_err(|error| error.to_string())?;
        result
    }

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

    #[test]
    fn push_sanity_verify_hexagon_locks_rejects_unpinned_includes() -> Result<()> {
        let temporary = super::super::canonical_path(std::env::temp_dir())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let scratch = temporary.join(format!("bonsai-push-hexlock-{nonce}"));
        fs::create_dir(&scratch).map_err(|error| error.to_string())?;
        let result = (|| {
            let policy =
                super::super::canonical_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))?;
            fs::create_dir_all(scratch.join("tools/hexagon")).map_err(|error| error.to_string())?;
            fs::copy(
                policy.join("tools/hexagon/windows-queue-abi.lock.json"),
                scratch.join("tools/hexagon/windows-queue-abi.lock.json"),
            )
            .map_err(|error| error.to_string())?;
            fs::copy(
                policy.join("tools/hexagon/windows_queue_abi.c"),
                scratch.join("tools/hexagon/windows_queue_abi.c"),
            )
            .map_err(|error| error.to_string())?;
            verify_hexagon_locks(&scratch, &["tools/hexagon/windows_queue_abi.c".into()])?;

            fs::write(
                scratch.join("tools/hexagon/windows_queue_abi.c"),
                "#include \"dspqueue.h\"\n#include \"unpinned_extra.h\"\n",
            )
            .map_err(|error| error.to_string())?;
            let error =
                verify_hexagon_locks(&scratch, &["tools/hexagon/windows_queue_abi.c".into()])
                    .err()
                    .ok_or("Expected unpinned include failure.")?;
            assert!(error.contains("unpinned_extra.h"), "{error}");
            Ok(())
        })();
        if scratch.parent() != Some(temporary.as_path()) {
            return Err("Push hexlock fixture cleanup escaped temp.".into());
        }
        fs::remove_dir_all(scratch).map_err(|error| error.to_string())?;
        result
    }
}
