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

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct CachedBlob {
    blob: String,
    len: u64,
    modified_ns: u128,
}

const PUSH_CACHE_INCREMENTAL_LIMIT_BYTES: u64 = 6 * 1024 * 1024 * 1024;
const PUSH_CACHE_HARD_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

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
        const AGGREGATE_BUDGET: Duration = Duration::from_secs(270);
        const OUTPUT_CHECK_INTERVAL: Duration = Duration::from_millis(25);
        let code = loop {
            let wait = AGGREGATE_BUDGET
                .saturating_sub(self.started.elapsed())
                .min(OUTPUT_CHECK_INTERVAL);
            if let Some(code) = child
                .wait_for_exit(wait)
                .map_err(|error| error.to_string())?
            {
                break code;
            }
            if self.started.elapsed() >= AGGREGATE_BUDGET
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
            let tail_start = detail.len().saturating_sub(8192);
            let highlights = if tail_start > 0 {
                super::super::checks::execute::extract_prefix_highlights(&detail[..tail_start])
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let prefix = if highlights.is_empty() {
                String::new()
            } else {
                format!(
                    "[key diagnostic lines before truncated tail]\n{}\n[final 8192 bytes]\n",
                    String::from_utf8_lossy(&highlights).trim_end()
                )
            };
            let excerpt = String::from_utf8_lossy(&detail[tail_start..]);
            return Err(format!(
                "Push blocked: {program} {arguments:?} exited {code} after {:.2?}.\n{prefix}{excerpt}",
                command_started.elapsed()
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
        self.cargo_checks(source, paths, &plan)?;
        Ok(())
    }

    #[cfg(not(test))]
    fn toolchain_root(toolchain: &str) -> Option<PathBuf> {
        let rustup_home = std::env::var_os("RUSTUP_HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("USERPROFILE")
                    .filter(|value| !value.is_empty())
                    .map(|home| PathBuf::from(home).join(".rustup"))
            })?;
        let toolchains = rustup_home.join("toolchains");
        [
            toolchains.join(format!("{toolchain}-aarch64-pc-windows-msvc")),
            toolchains.join(toolchain),
        ]
        .into_iter()
        .find(|candidate| candidate.is_dir())
    }

    #[expect(
        clippy::permissions_set_readonly_false,
        reason = "Windows-only helper clearing FILE_ATTRIBUTE_READONLY before replacing or removing cached files"
    )]
    fn clear_readonly_metadata(path: &Path, metadata: &fs::Metadata) -> Result<()> {
        if metadata.file_type().is_symlink() {
            return Ok(());
        }
        let mut permissions = metadata.permissions();
        if permissions.readonly() {
            permissions.set_readonly(false);
            fs::set_permissions(path, permissions).map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn remove_entry(path: &Path) -> Result<()> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.to_string()),
        };
        Self::clear_readonly_metadata(path, &metadata)?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            return fs::remove_dir(path)
                .or_else(|_| fs::remove_file(path))
                .map_err(|error| error.to_string());
        }
        if !file_type.is_dir() {
            return fs::remove_file(path).map_err(|error| error.to_string());
        }
        let mut stack = vec![path.to_path_buf()];
        let mut dirs = Vec::new();
        while let Some(dir) = stack.pop() {
            dirs.push(dir.clone());
            for entry in fs::read_dir(&dir).map_err(|error| error.to_string())? {
                let entry = entry.map_err(|error| error.to_string())?;
                let child = entry.path();
                let child_meta = entry.metadata().map_err(|error| error.to_string())?;
                Self::clear_readonly_metadata(&child, &child_meta)?;
                let child_type = child_meta.file_type();
                if child_type.is_symlink() {
                    fs::remove_dir(&child)
                        .or_else(|_| fs::remove_file(&child))
                        .map_err(|error| error.to_string())?;
                } else if child_type.is_dir() {
                    stack.push(child);
                } else {
                    fs::remove_file(&child).map_err(|error| error.to_string())?;
                }
            }
        }
        while let Some(dir) = dirs.pop() {
            fs::remove_dir(&dir).map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn files_equal(left: &Path, left_len: u64, right: &Path, right_len: u64) -> Result<bool> {
        if left_len != right_len {
            return Ok(false);
        }
        let mut left_file = fs::File::open(left).map_err(|error| error.to_string())?;
        let mut right_file = fs::File::open(right).map_err(|error| error.to_string())?;
        let mut left_buf = [0_u8; 32_768];
        let mut right_buf = [0_u8; 32_768];
        loop {
            let left_read = std::io::Read::read(&mut left_file, &mut left_buf)
                .map_err(|error| error.to_string())?;
            if left_read == 0 {
                let mut trailing = [0_u8; 1];
                let extra = std::io::Read::read(&mut right_file, &mut trailing)
                    .map_err(|error| error.to_string())?;
                return Ok(extra == 0);
            }
            let mut right_filled = 0;
            while right_filled < left_read {
                let Some(dest_slice) = right_buf.get_mut(right_filled..left_read) else {
                    return Err("Internal buffer slice out of bounds during comparison.".into());
                };
                let chunk = std::io::Read::read(&mut right_file, dest_slice)
                    .map_err(|error| error.to_string())?;
                if chunk == 0 {
                    return Ok(false);
                }
                right_filled += chunk;
            }
            if left_buf.get(..left_read) != right_buf.get(..left_read) {
                return Ok(false);
            }
        }
    }

    fn sync_tree(source: &Path, destination: &Path) -> Result<()> {
        let source_meta = fs::symlink_metadata(source).map_err(|error| error.to_string())?;
        if source_meta.file_type().is_symlink() || !source_meta.is_dir() {
            return Err(format!(
                "Push blocked: source tree root '{}' must be a directory.",
                source.display()
            ));
        }
        match fs::symlink_metadata(destination) {
            Ok(dest_meta) => {
                if dest_meta.file_type().is_symlink() || !dest_meta.is_dir() {
                    Self::remove_entry(destination)?;
                    fs::create_dir_all(destination).map_err(|error| error.to_string())?;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(destination).map_err(|error| error.to_string())?;
            }
            Err(error) => return Err(error.to_string()),
        }
        let mut pending = vec![(source.to_path_buf(), destination.to_path_buf())];
        while let Some((src_dir, dst_dir)) = pending.pop() {
            let mut src_entries: BTreeMap<OsString, (PathBuf, bool, u64)> = BTreeMap::new();
            for entry in fs::read_dir(&src_dir).map_err(|error| error.to_string())? {
                let entry = entry.map_err(|error| error.to_string())?;
                let file_type = entry.file_type().map_err(|error| error.to_string())?;
                if file_type.is_symlink() {
                    return Err(format!(
                        "Push blocked: staged source entry '{}' cannot be a symlink.",
                        entry.path().display()
                    ));
                }
                let len = if file_type.is_dir() {
                    0
                } else if file_type.is_file() {
                    entry.metadata().map_err(|error| error.to_string())?.len()
                } else {
                    return Err(format!(
                        "Push blocked: unsupported staged source entry '{}'.",
                        entry.path().display()
                    ));
                };
                src_entries.insert(entry.file_name(), (entry.path(), file_type.is_dir(), len));
            }
            let mut kept: BTreeMap<OsString, fs::Metadata> = BTreeMap::new();
            for entry in fs::read_dir(&dst_dir).map_err(|error| error.to_string())? {
                let entry = entry.map_err(|error| error.to_string())?;
                let name = entry.file_name();
                let dst_path = entry.path();
                let dst_meta = entry.metadata().map_err(|error| error.to_string())?;
                let dst_type = dst_meta.file_type();
                let matches = src_entries.get(&name).is_some_and(|(_, is_dir, _)| {
                    !dst_type.is_symlink()
                        && if *is_dir {
                            dst_type.is_dir()
                        } else {
                            dst_type.is_file()
                        }
                });
                if matches {
                    kept.insert(name, dst_meta);
                } else {
                    Self::remove_entry(&dst_path)?;
                }
            }
            for (name, (src_path, is_dir, src_len)) in src_entries {
                let dst_path = dst_dir.join(&name);
                let kept_meta = kept.get(&name);
                if is_dir {
                    if kept_meta.is_none() {
                        fs::create_dir_all(&dst_path).map_err(|error| error.to_string())?;
                    }
                    pending.push((src_path, dst_path));
                } else {
                    let unchanged = if let Some(dst_meta) = kept_meta {
                        Self::files_equal(&dst_path, dst_meta.len(), &src_path, src_len)?
                    } else {
                        false
                    };
                    if !unchanged {
                        if let Some(dst_meta) = kept_meta {
                            Self::clear_readonly_metadata(&dst_path, dst_meta)?;
                        }
                        fs::copy(&src_path, &dst_path).map_err(|error| error.to_string())?;
                    }
                }
            }
        }
        Ok(())
    }

    fn stable_source_dir(target_dir: &Path) -> Result<PathBuf> {
        let leaf = target_dir
            .file_name()
            .ok_or("Push cargo target directory must have a directory name.")?;
        let mut name = leaf.to_os_string();
        name.push("-source");
        Ok(target_dir.with_file_name(name))
    }

    fn modified_ns(metadata: &fs::Metadata) -> Option<u128> {
        metadata
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|duration| duration.as_nanos())
    }

    fn commit_tree_blobs(&mut self, root: &Path, commit: &str) -> Result<BTreeMap<String, String>> {
        let raw = self.git(root, &["ls-tree", "-r", "-z", "--full-tree", commit])?;
        let mut blobs = BTreeMap::new();
        for record in raw
            .split(|byte| *byte == 0)
            .filter(|record| !record.is_empty())
        {
            let text = std::str::from_utf8(record).map_err(|error| error.to_string())?;
            let Some((header, path)) = text.split_once('\t') else {
                return Err("Malformed git ls-tree record.".into());
            };
            let mut parts = header.split_whitespace();
            let (Some(mode), Some(kind), Some(oid), None) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                return Err("Malformed git ls-tree header.".into());
            };
            if mode == "120000" {
                return Err(format!(
                    "Push blocked: staged source entry '{path}' cannot be a symlink."
                ));
            }
            if kind != "blob" {
                return Err(format!(
                    "Push blocked: unsupported staged source entry '{path}'."
                ));
            }
            blobs.insert(path.to_owned(), format!("{mode}:{oid}"));
        }
        Ok(blobs)
    }

    fn record_stable_blobs(
        stable_source: &Path,
        commit_blobs: &BTreeMap<String, String>,
    ) -> BTreeMap<String, CachedBlob> {
        let mut recorded = BTreeMap::new();
        let mut pending = vec![(String::new(), stable_source.to_path_buf())];
        while let Some((rel_dir, dir_path)) = pending.pop() {
            let Ok(entries) = fs::read_dir(&dir_path) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(meta) = entry.metadata() else {
                    continue;
                };
                let name_os = entry.file_name();
                let Some(name) = name_os.to_str() else {
                    continue;
                };
                let rel_path = if rel_dir.is_empty() {
                    name.to_owned()
                } else {
                    format!("{rel_dir}/{name}")
                };
                if meta.is_dir() {
                    pending.push((rel_path, entry.path()));
                } else if meta.is_file()
                    && let (Some(blob), Some(modified_ns)) =
                        (commit_blobs.get(&rel_path), Self::modified_ns(&meta))
                {
                    recorded.insert(
                        rel_path,
                        CachedBlob {
                            blob: blob.clone(),
                            len: meta.len(),
                            modified_ns,
                        },
                    );
                }
            }
        }
        recorded
    }

    fn sync_commit(
        &mut self,
        root: &Path,
        commit: &str,
        number: usize,
        target_dir: &Path,
        stable_source: &Path,
    ) -> Result<()> {
        let state_path = target_dir.join(".bonsai-source-blobs.json");
        let cached: BTreeMap<String, CachedBlob> = fs::read(&state_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        let commit_blobs = self.commit_tree_blobs(root, commit)?;

        let mut expected_dirs: BTreeMap<String, BTreeMap<OsString, bool>> = BTreeMap::new();
        expected_dirs.insert(String::new(), BTreeMap::new());
        for path in commit_blobs.keys() {
            let parts: Vec<&str> = path.split('/').collect();
            let mut parent_rel = String::new();
            for (idx, part) in parts.iter().enumerate() {
                let is_dir = idx + 1 < parts.len();
                expected_dirs
                    .entry(parent_rel.clone())
                    .or_default()
                    .insert(OsString::from(*part), is_dir);
                if is_dir {
                    parent_rel = if parent_rel.is_empty() {
                        (*part).to_owned()
                    } else {
                        format!("{parent_rel}/{part}")
                    };
                    expected_dirs.entry(parent_rel.clone()).or_default();
                }
            }
        }

        let mut verified: BTreeMap<String, CachedBlob> = BTreeMap::new();
        let stable_exists = match fs::symlink_metadata(stable_source) {
            Ok(meta) => {
                if meta.file_type().is_symlink() || !meta.is_dir() {
                    Self::remove_entry(stable_source)?;
                    false
                } else {
                    true
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.to_string()),
        };

        if stable_exists && !cached.is_empty() {
            let mut pending = vec![(String::new(), stable_source.to_path_buf())];
            while let Some((rel_dir, dst_dir)) = pending.pop() {
                let Some(expected_children) = expected_dirs.get(&rel_dir) else {
                    continue;
                };
                for entry in fs::read_dir(&dst_dir).map_err(|error| error.to_string())? {
                    let entry = entry.map_err(|error| error.to_string())?;
                    let name_os = entry.file_name();
                    let dst_path = entry.path();
                    let dst_meta = entry.metadata().map_err(|error| error.to_string())?;
                    let dst_type = dst_meta.file_type();
                    let matches = expected_children.get(&name_os).is_some_and(|is_dir| {
                        !dst_type.is_symlink()
                            && if *is_dir {
                                dst_type.is_dir()
                            } else {
                                dst_type.is_file()
                            }
                    });
                    if !matches {
                        Self::remove_entry(&dst_path)?;
                        continue;
                    }
                    let Some(name) = name_os.to_str() else {
                        Self::remove_entry(&dst_path)?;
                        continue;
                    };
                    let rel_path = if rel_dir.is_empty() {
                        name.to_owned()
                    } else {
                        format!("{rel_dir}/{name}")
                    };
                    if dst_type.is_dir() {
                        pending.push((rel_path, dst_path));
                    } else if let (Some(expected_blob), Some(prev), Some(modified_ns)) = (
                        commit_blobs.get(&rel_path),
                        cached.get(&rel_path),
                        Self::modified_ns(&dst_meta),
                    ) && prev.blob == *expected_blob
                        && prev.len == dst_meta.len()
                        && prev.modified_ns == modified_ns
                    {
                        verified.insert(rel_path, prev.clone());
                    }
                }
            }
        }

        let dirty: Vec<String> = commit_blobs
            .keys()
            .filter(|path| !verified.contains_key(*path))
            .cloned()
            .collect();

        if dirty.len() > 64 {
            let staged = self.snapshot(root, commit, number)?;
            Self::sync_tree(&staged, stable_source)?;
            Self::remove_entry(&staged)?;
            verified = Self::record_stable_blobs(stable_source, &commit_blobs);
        } else if !dirty.is_empty() {
            fs::create_dir_all(stable_source).map_err(|error| error.to_string())?;
            let staged = self.directory.join(format!("source-{number}"));
            fs::create_dir(&staged).map_err(|error| error.to_string())?;
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
            let mut checkout_args: Vec<OsString> = vec![
                "checkout-index".into(),
                "--force".into(),
                format!("--prefix={}/", staged.display()).into(),
                "--".into(),
            ];
            checkout_args.extend(dirty.iter().map(OsString::from));
            self.command(root, "git.exe", &checkout_args, environment)?;
            for rel_path in &dirty {
                let src_path = staged.join(rel_path);
                let dst_path = stable_source.join(rel_path);
                let src_meta =
                    fs::symlink_metadata(&src_path).map_err(|error| error.to_string())?;
                if src_meta.file_type().is_symlink() || !src_meta.is_file() {
                    return Err(format!(
                        "Push blocked: staged source entry '{}' must be a regular file.",
                        src_path.display()
                    ));
                }
                if let Some(parent) = dst_path.parent() {
                    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
                }
                let existing_meta = fs::symlink_metadata(&dst_path).ok();
                let unchanged = if let Some(dst_meta) = existing_meta.as_ref()
                    && !dst_meta.file_type().is_symlink()
                    && dst_meta.is_file()
                {
                    Self::files_equal(&dst_path, dst_meta.len(), &src_path, src_meta.len())?
                } else {
                    false
                };
                if !unchanged {
                    if let Some(dst_meta) = existing_meta.as_ref() {
                        Self::clear_readonly_metadata(&dst_path, dst_meta)?;
                    }
                    fs::copy(&src_path, &dst_path).map_err(|error| error.to_string())?;
                }
                let final_meta =
                    fs::symlink_metadata(&dst_path).map_err(|error| error.to_string())?;
                if let (Some(blob), Some(modified_ns)) =
                    (commit_blobs.get(rel_path), Self::modified_ns(&final_meta))
                {
                    verified.insert(
                        rel_path.clone(),
                        CachedBlob {
                            blob: blob.clone(),
                            len: final_meta.len(),
                            modified_ns,
                        },
                    );
                }
            }
            Self::remove_entry(&staged)?;
        }

        if let Ok(encoded) = serde_json::to_vec(&verified) {
            let _ = fs::write(&state_path, encoded);
        }
        Ok(())
    }

    fn lock_source_cache(&self, target_dir: &Path) -> Result<fs::File> {
        use std::os::windows::fs::OpenOptionsExt;
        fs::create_dir_all(target_dir).map_err(|error| error.to_string())?;
        let lock_path = target_dir.join(".bonsai-source.lock");
        loop {
            match fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .share_mode(0)
                .open(&lock_path)
            {
                Ok(file) => return Ok(file),
                Err(error)
                    if matches!(error.raw_os_error(), Some(32 | 33))
                        || error.kind() == std::io::ErrorKind::WouldBlock =>
                {
                    if self.started.elapsed() >= Duration::from_secs(270) {
                        return Err(
                            "Push blocked: timed out waiting for the source cache lock.".into()
                        );
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(error) => return Err(error.to_string()),
            }
        }
    }

    fn directory_bytes(root: &Path) -> u64 {
        let mut total = 0_u64;
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(meta) = entry.metadata() else {
                    continue;
                };
                let kind = meta.file_type();
                if kind.is_symlink() {
                    continue;
                }
                if kind.is_dir() {
                    stack.push(entry.path());
                } else if kind.is_file() {
                    total = total.saturating_add(meta.len());
                }
            }
        }
        total
    }

    fn prune_target_cache(
        target_dir: &Path,
        incremental_limit: u64,
        hard_limit: u64,
    ) -> Result<()> {
        let initial = Self::directory_bytes(target_dir);
        if initial <= incremental_limit {
            return Ok(());
        }
        for relative in [
            "debug/incremental",
            "aarch64-pc-windows-msvc/debug/incremental",
            "aarch64-unknown-linux-gnu/debug/incremental",
        ] {
            Self::remove_entry(&target_dir.join(relative))?;
        }
        let after_incremental = Self::directory_bytes(target_dir);
        if after_incremental <= hard_limit {
            return Ok(());
        }
        // Multi-toolchain accumulation exceeded the hard bound: drop cached
        // dependency units while preserving verified helper binaries, receipts,
        // source blob state and the held `.bonsai-source.lock` file.
        for relative in [
            "debug",
            "aarch64-unknown-linux-gnu",
            "aarch64-pc-windows-msvc/debug/deps",
            "aarch64-pc-windows-msvc/debug/build",
            "aarch64-pc-windows-msvc/debug/examples",
            "aarch64-pc-windows-msvc/debug/.fingerprint",
        ] {
            Self::remove_entry(&target_dir.join(relative))?;
        }
        Ok(())
    }

    fn format_check(&mut self, source: &Path) -> Result<()> {
        let toolchain = super::super::push_format::toolchain(source)?;
        #[cfg(test)]
        let environment = BTreeMap::new();
        #[cfg(not(test))]
        let environment = {
            let direct = Self::toolchain_root(&toolchain)
                .map(|root| root.join("bin").join("rustfmt.exe"))
                .filter(|path| path.is_file());
            let image = if let Some(path) = direct {
                super::canonical_path(path)?
                    .into_os_string()
                    .into_string()
                    .map_err(|_| "Formatter binary path must be UTF-8.".to_string())?
            } else {
                let output = self.command(
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
                std::str::from_utf8(&output)
                    .map_err(|error| error.to_string())?
                    .trim()
                    .to_string()
            };
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
                (super::super::push_format::IMAGE.into(), image),
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
        if plan.cache_closures
            && source.join(".github/actions/ci-tool/action.yml").is_file()
            && source.join("Cargo.lock").is_file()
        {
            calls.push(ScriptCall {
                script: "test_tools_cache_inputs.ps1",
                parameters: BTreeMap::from([
                    (
                        "Case",
                        serde_json::json!("ci_tool_dependency_and_embedded_closure_hashed"),
                    ),
                    ("CandidateRoot", serde_json::json!(source)),
                ]),
            });
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
        if plan.foundation_handoff
            && [
                ".github/workflows/foundation.yml",
                ".github/workflows/release.yml",
                ".github/ci/impact.json",
                "config/rust-packages.json",
                ".github/actions/check-rust/action.yml",
                ".github/actions/use-bonsai-tools/action.yml",
                "tools/ci/src/checks/mod.rs",
            ]
            .into_iter()
            .all(|path| source.join(path).is_file())
        {
            calls.push(ScriptCall {
                script: "test_foundation_handoff.ps1",
                parameters: BTreeMap::from([("CandidateRoot", serde_json::json!(source))]),
            });
        }
        if plan.device_concurrency
            && [
                ".github/workflows/native-runtime-acceptance.yml",
                ".github/workflows/native-ptq1-numerics.yml",
                ".github/workflows/x2-cpu-evaluation.yml",
                ".github/workflows/cpu-bundle-comparison.yml",
                ".github/workflows/tts-evaluation.yml",
            ]
            .into_iter()
            .all(|path| source.join(path).is_file())
        {
            calls.push(ScriptCall {
                script: "test_device_concurrency.ps1",
                parameters: BTreeMap::from([("CandidateRoot", serde_json::json!(source))]),
            });
        }
        if plan.ci_bootstrap_cache && source.join(".github/actions/ci-tool/action.yml").is_file() {
            calls.push(ScriptCall {
                script: "test_native_ci_cache.ps1",
                parameters: BTreeMap::from([
                    (
                        "Case",
                        serde_json::json!("bootstrap_codegen_overrides_stay_in_coordinator_build"),
                    ),
                    ("CandidateRoot", serde_json::json!(source)),
                ]),
            });
        }
        if plan.rust_toolchain_fixture
            && source
                .join(".github/actions/rust-toolchain/action.yml")
                .is_file()
        {
            calls.push(ScriptCall {
                script: "test_rust_toolchain.ps1",
                parameters: BTreeMap::from([("CandidateRoot", serde_json::json!(source))]),
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
        if affected.is_empty() && !plan.workspace_registry && !plan.hexagon_locks {
            return Ok(());
        }
        let toolchain = super::super::push_format::toolchain(source)?;
        let target_dir = std::env::var_os("BONSAI_PUSH_CARGO_TARGET_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| self.directory.join("cargo-target"));
        fs::create_dir_all(&target_dir).map_err(|error| error.to_string())?;
        let target_dir = super::canonical_path(target_dir)?;
        let manifest = source.join("Cargo.toml");
        let cargo_env = BTreeMap::from([
            (
                "CARGO_UNSTABLE_CHECKSUM_FRESHNESS".to_string(),
                "true".to_string(),
            ),
            ("CARGO_BUILD_FINGERPRINT".to_string(), "content".to_string()),
            ("CARGO_INCREMENTAL".to_string(), "1".to_string()),
        ]);

        let workspace = crate::impact::workspace::read(
            &source.canonicalize().map_err(|error| error.to_string())?,
        )?;
        let debug_dir = target_dir.join("aarch64-pc-windows-msvc").join("debug");
        let saved_helpers: Vec<(PathBuf, Vec<u8>)> = [
            ("push-checker.json", "bonsai-ci.exe"),
            ("push-recovery.json", "bonsai-push-recovery.exe"),
        ]
        .into_iter()
        .filter_map(|(receipt, binary)| {
            if target_dir.join(receipt).is_file() {
                let path = debug_dir.join(binary);
                fs::read(&path).ok().map(|bytes| (path, bytes))
            } else {
                None
            }
        })
        .collect();

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

        let result = (|| -> Result<()> {
            if affected.contains("bonsai-ci") || plan.workspace_registry {
                let mut test_args = base_cargo_args("test", "aarch64-pc-windows-msvc");
                test_args.extend([
                    "-p".into(),
                    "bonsai-ci".into(),
                    "--lib".into(),
                    "--test".into(),
                    "routing".into(),
                    "--".into(),
                    "--skip".into(),
                    "push_sanity_".into(),
                ]);
                self.command(source, "cargo.exe", &test_args, cargo_env.clone())?;
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
                self.command(source, "cargo.exe", &hex_args, cargo_env.clone())?;
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
                self.command(source, "cargo.exe", &rel_args, cargo_env.clone())?;
            }

            // Always check the complete standard Windows workspace set when any
            // standard package is affected so Cargo resolves one invariant
            // unified feature graph and never thrashes cached dependency units.
            if affected
                .iter()
                .any(|package| package.as_str() != "bonsai-speech-asr")
            {
                let mut args = base_cargo_args("clippy", "aarch64-pc-windows-msvc");
                args.push("--all-features".into());
                args.push("--all-targets".into());
                for package in workspace
                    .packages
                    .keys()
                    .filter(|package| package.as_str() != "bonsai-speech-asr")
                {
                    args.push("-p".into());
                    args.push(package.into());
                }
                args.extend(["--".into(), "-D".into(), "warnings".into()]);
                self.command(source, "cargo.exe", &args, cargo_env.clone())?;
            }

            if affected.contains("bonsai-speech-asr") {
                let mut args = base_cargo_args("clippy", "aarch64-pc-windows-msvc");
                args.extend([
                    "-p".into(),
                    "bonsai-speech-asr".into(),
                    "--no-default-features".into(),
                    "--features".into(),
                    "rust-qualification".into(),
                    "--all-targets".into(),
                    "--".into(),
                    "-D".into(),
                    "warnings".into(),
                ]);
                self.command(source, "cargo.exe", &args, cargo_env.clone())?;
            }

            const PORTABLE_LINUX_CRATES: [&str; 6] = [
                "bonsai-artifacts",
                "bonsai-audio",
                "bonsai-cpu-policy",
                "bonsai-broker",
                "bonsai-markdown",
                "bonsai-speech-service",
            ];
            let linux_portable_affected = affected
                .iter()
                .any(|package| PORTABLE_LINUX_CRATES.contains(&package.as_str()));
            let linux_ci_affected = (affected.contains("bonsai-ci") || plan.workspace_registry)
                && workspace.packages.contains_key("bonsai-ci");
            let linux_affected = linux_portable_affected || linux_ci_affected;
            let has_linux_target = if !linux_affected {
                false
            } else if let Some(rustlib) = Self::toolchain_root(&toolchain)
                .map(|root| root.join("lib").join("rustlib"))
                .filter(|path| path.is_dir())
            {
                rustlib.join("aarch64-unknown-linux-gnu").is_dir()
            } else {
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
                installed_text
                    .lines()
                    .any(|line| line.trim() == "aarch64-unknown-linux-gnu")
            };
            if has_linux_target {
                if linux_portable_affected {
                    let mut args = base_cargo_args("clippy", "aarch64-unknown-linux-gnu");
                    args.push("--all-targets".into());
                    for package in PORTABLE_LINUX_CRATES
                        .into_iter()
                        .filter(|package| workspace.packages.contains_key(*package))
                    {
                        args.push("-p".into());
                        args.push(package.into());
                    }
                    args.extend(["--".into(), "-D".into(), "warnings".into()]);
                    self.command(source, "cargo.exe", &args, cargo_env.clone())?;
                }
                if linux_ci_affected {
                    let mut args = base_cargo_args("clippy", "aarch64-unknown-linux-gnu");
                    args.extend([
                        "-p".into(),
                        "bonsai-ci".into(),
                        "--no-default-features".into(),
                        "--all-targets".into(),
                        "--".into(),
                        "-D".into(),
                        "warnings".into(),
                    ]);
                    self.command(source, "cargo.exe", &args, cargo_env)?;
                }
            }

            Ok(())
        })();

        for (path, bytes) in saved_helpers {
            let _ = fs::write(path, bytes);
        }

        result
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
            .filter(|value| crate::impact::workspace::is_lower_hex(value, 64))
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
        let cache_target = std::env::var_os("BONSAI_PUSH_CARGO_TARGET_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        for (number, (commit, paths)) in commits.iter().enumerate() {
            let changed_paths: Vec<String> = paths.iter().cloned().collect();
            if let Some(target_dir) = cache_target.as_ref() {
                let _cache_lock = work.lock_source_cache(target_dir)?;
                if number == 0 {
                    Work::prune_target_cache(
                        target_dir,
                        PUSH_CACHE_INCREMENTAL_LIMIT_BYTES,
                        PUSH_CACHE_HARD_LIMIT_BYTES,
                    )?;
                }
                let stable_source = Work::stable_source_dir(target_dir)?;
                work.sync_commit(root, commit, number, target_dir, &stable_source)?;
                work.checks(policy, &stable_source, &changed_paths)?;
            } else {
                let staged = work.snapshot(root, commit, number)?;
                work.checks(policy, &staged, &changed_paths)?;
            }
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
        let policy = super::super::canonical_path(crate::command::test_workspace_root())?;
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
    fn push_sanity_unverified_supervisor_triggers_preparation() -> Result<()> {
        preparation_case("unverified_supervisor_triggers_preparation")
    }

    #[test]
    fn push_sanity_missing_recovery_receipt_triggers_preparation() -> Result<()> {
        preparation_case("missing_recovery_receipt_triggers_preparation")
    }

    #[test]
    fn push_sanity_missing_recovery_helper_triggers_preparation() -> Result<()> {
        preparation_case("missing_recovery_helper_triggers_preparation")
    }

    #[test]
    fn push_sanity_stale_recovery_source_triggers_preparation() -> Result<()> {
        preparation_case("stale_recovery_source_triggers_preparation")
    }

    #[test]
    fn push_sanity_corrupt_recovery_helper_triggers_preparation() -> Result<()> {
        preparation_case("corrupt_recovery_helper_triggers_preparation")
    }

    #[test]
    fn push_sanity_automatic_preparation_invokes_committed_helper_once() -> Result<()> {
        preparation_case("automatic_preparation_invokes_committed_helper_once")
    }

    #[test]
    fn push_sanity_automatic_preparation_rejects_invalid_prepared_outputs() -> Result<()> {
        preparation_case("automatic_preparation_rejects_invalid_prepared_outputs")
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
    fn push_sanity_collection_recovery_source_triggers_preparation() -> Result<()> {
        preparation_case("collection_recovery_source_triggers_preparation")
    }

    #[test]
    fn push_sanity_collection_recovery_digest_triggers_preparation() -> Result<()> {
        preparation_case("collection_recovery_digest_triggers_preparation")
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
        let policy = super::super::canonical_path(crate::command::test_workspace_root())?;
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
            let policy = super::super::canonical_path(crate::command::test_workspace_root())?;
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
            let policy = super::super::canonical_path(crate::command::test_workspace_root())?;
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
                "test_native_ci_cache",
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
            let policy = super::super::canonical_path(crate::command::test_workspace_root())?;
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

    #[test]
    fn push_sanity_sync_tree_preserves_unchanged_files_and_repairs_mutations() -> Result<()> {
        let temporary = std::env::temp_dir();
        let scratch = temporary.join(format!("bonsai-push-sync-test-{}", std::process::id()));
        if scratch.exists() {
            Work::remove_entry(&scratch)?;
        }
        let result = (|| -> Result<()> {
            let src = scratch.join("src");
            let dst = scratch.join("dst");
            fs::create_dir_all(src.join("nested")).map_err(|error| error.to_string())?;
            fs::write(src.join("keep.txt"), b"unchanged").map_err(|error| error.to_string())?;
            fs::write(src.join("nested").join("modify.txt"), b"expected")
                .map_err(|error| error.to_string())?;
            fs::write(src.join("RenameMe.txt"), b"case").map_err(|error| error.to_string())?;

            Work::sync_tree(&src, &dst)?;

            let old_time = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
            {
                let keep_file = fs::OpenOptions::new()
                    .write(true)
                    .open(dst.join("keep.txt"))
                    .map_err(|error| error.to_string())?;
                keep_file
                    .set_times(fs::FileTimes::new().set_modified(old_time))
                    .map_err(|error| error.to_string())?;
            }

            fs::write(dst.join("nested").join("modify.txt"), b"corrupted")
                .map_err(|error| error.to_string())?;
            let mut lock_ro = fs::metadata(dst.join("nested").join("modify.txt"))
                .map_err(|error| error.to_string())?
                .permissions();
            lock_ro.set_readonly(true);
            fs::set_permissions(dst.join("nested").join("modify.txt"), lock_ro)
                .map_err(|error| error.to_string())?;

            fs::write(dst.join("extra.txt"), b"extra").map_err(|error| error.to_string())?;
            fs::create_dir_all(dst.join("extra_dir")).map_err(|error| error.to_string())?;
            fs::write(dst.join("extra_dir").join("stale.txt"), b"stale")
                .map_err(|error| error.to_string())?;
            fs::remove_file(dst.join("RenameMe.txt")).map_err(|error| error.to_string())?;
            fs::write(dst.join("renameme.txt"), b"case").map_err(|error| error.to_string())?;

            Work::sync_tree(&src, &dst)?;

            let observed_time = fs::metadata(dst.join("keep.txt"))
                .map_err(|error| error.to_string())?
                .modified()
                .map_err(|error| error.to_string())?;
            if observed_time != old_time {
                return Err("Unchanged file modified time was altered during sync_tree.".into());
            }
            let repaired = fs::read(dst.join("nested").join("modify.txt"))
                .map_err(|error| error.to_string())?;
            if repaired != b"expected" {
                return Err("Mutated readonly file was not repaired during sync_tree.".into());
            }
            if dst.join("extra.txt").exists() || dst.join("extra_dir").exists() {
                return Err("Extra stale entries were not removed during sync_tree.".into());
            }
            let mut names = Vec::new();
            for entry in fs::read_dir(&dst).map_err(|error| error.to_string())? {
                names.push(entry.map_err(|error| error.to_string())?.file_name());
            }
            if !names.contains(&OsString::from("RenameMe.txt"))
                || names.contains(&OsString::from("renameme.txt"))
            {
                return Err(format!("Case-only rename was not repaired: {names:?}"));
            }
            Ok(())
        })();
        if scratch.parent() != Some(temporary.as_path()) {
            return Err("Push sync test fixture cleanup escaped temp.".into());
        }
        if scratch.exists() {
            Work::remove_entry(&scratch)?;
        }
        result
    }

    #[test]
    fn prune_target_cache_trims_incremental_before_deps_and_keeps_receipts() -> Result<()> {
        let temporary = std::env::temp_dir();
        let scratch = temporary.join(format!("bonsai-push-prune-test-{}", std::process::id()));
        if scratch.exists() {
            Work::remove_entry(&scratch)?;
        }
        let result = (|| -> Result<()> {
            let win_debug = scratch.join("aarch64-pc-windows-msvc/debug");
            fs::create_dir_all(win_debug.join("incremental/session-1"))
                .map_err(|error| error.to_string())?;
            fs::create_dir_all(win_debug.join("deps")).map_err(|error| error.to_string())?;
            fs::create_dir_all(scratch.join("debug/incremental"))
                .map_err(|error| error.to_string())?;
            fs::create_dir_all(scratch.join("aarch64-unknown-linux-gnu/debug/deps"))
                .map_err(|error| error.to_string())?;

            fs::write(scratch.join(".bonsai-source.lock"), b"lock")
                .map_err(|error| error.to_string())?;
            fs::write(scratch.join(".bonsai-source-blobs.json"), b"{}")
                .map_err(|error| error.to_string())?;
            fs::write(scratch.join("push-checker.json"), b"{\"SchemaVersion\":1}")
                .map_err(|error| error.to_string())?;
            fs::write(scratch.join("push-recovery.json"), b"{\"SchemaVersion\":1}")
                .map_err(|error| error.to_string())?;
            fs::write(win_debug.join("bonsai-ci.exe"), b"checker-bytes")
                .map_err(|error| error.to_string())?;
            fs::write(
                win_debug.join("bonsai-push-recovery.exe"),
                b"recovery-bytes",
            )
            .map_err(|error| error.to_string())?;
            fs::write(win_debug.join("deps/libdep.rmeta"), &[b'd'; 400])
                .map_err(|error| error.to_string())?;
            fs::write(
                win_debug.join("incremental/session-1/work.bin"),
                &[b'i'; 700],
            )
            .map_err(|error| error.to_string())?;
            fs::write(scratch.join("debug/incremental/host.bin"), &[b'h'; 200])
                .map_err(|error| error.to_string())?;
            fs::write(
                scratch.join("aarch64-unknown-linux-gnu/debug/deps/linux.rmeta"),
                &[b'l'; 300],
            )
            .map_err(|error| error.to_string())?;

            // Stage 1: exceeds incremental_limit (1000), drops incremental/ but stays <= hard_limit (1200).
            Work::prune_target_cache(&scratch, 1000, 1200)?;
            assert!(!win_debug.join("incremental").exists());
            assert!(!scratch.join("debug/incremental").exists());
            assert!(win_debug.join("deps/libdep.rmeta").is_file());
            assert!(
                scratch
                    .join("aarch64-unknown-linux-gnu/debug/deps/linux.rmeta")
                    .is_file()
            );
            assert!(win_debug.join("bonsai-ci.exe").is_file());
            assert!(scratch.join("push-checker.json").is_file());

            // Stage 2: tightening hard_limit (200) drops deps/host/linux caches while keeping helpers & receipts.
            Work::prune_target_cache(&scratch, 100, 200)?;
            assert!(!win_debug.join("deps").exists());
            assert!(!scratch.join("debug").exists());
            assert!(!scratch.join("aarch64-unknown-linux-gnu").exists());
            assert_eq!(
                fs::read(win_debug.join("bonsai-ci.exe")).map_err(|error| error.to_string())?,
                b"checker-bytes"
            );
            assert_eq!(
                fs::read(win_debug.join("bonsai-push-recovery.exe"))
                    .map_err(|error| error.to_string())?,
                b"recovery-bytes"
            );
            assert!(scratch.join("push-checker.json").is_file());
            assert!(scratch.join("push-recovery.json").is_file());
            assert!(scratch.join(".bonsai-source.lock").is_file());
            assert!(scratch.join(".bonsai-source-blobs.json").is_file());
            Ok(())
        })();
        if scratch.parent() != Some(temporary.as_path()) {
            return Err("Push prune test fixture cleanup escaped temp.".into());
        }
        if scratch.exists() {
            Work::remove_entry(&scratch)?;
        }
        result
    }
}
