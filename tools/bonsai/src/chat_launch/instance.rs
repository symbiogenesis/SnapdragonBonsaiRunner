//! Per-user launcher admission retained until the owned app tree has exited.

#[cfg(windows)]
use std::time::{Duration, Instant};
use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::Read as _,
    path::{Path, PathBuf},
};

use bonsai_artifacts::is_redirect;
#[cfg(windows)]
use bonsai_windows_process::ProcessIdentity;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(windows)]
const REPLACE_ALLOWANCE: Duration = Duration::from_secs(10);
const MAX_INSTANCE_RECORD_BYTES: u64 = 4096;
const MAX_PACKAGE_METADATA_BYTES: u64 = 32_768;
const MAX_BUNDLE_EXECUTABLE_BYTES: u64 = 6 * 512 * 1024 * 1024 + 50 * 1024 * 1024;

/// Retained single-instance admission guard for `chat.lock` and its owner metadata.
pub(super) struct Guard {
    _lock: File,
    metadata_path: PathBuf,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if super::external(&self.metadata_path).is_ok()
            && fs::symlink_metadata(&self.metadata_path)
                .is_ok_and(|metadata| metadata.is_file() && !is_redirect(&metadata))
        {
            let _ = fs::remove_file(&self.metadata_path);
        }
        // `_lock` drops after `metadata_path` is removed so a departing owner
        // never removes a newer owner's metadata after releasing `chat.lock`.
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct InstanceVersion(u16, u16, u16);

impl InstanceVersion {
    fn parse(text: &str) -> crate::Result<Self> {
        let mut parts = text.split('.');
        let (Some(major), Some(minor), Some(patch), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(format!("Invalid application version: {text}"));
        };
        let parse_part = |part: &str| -> crate::Result<u16> {
            if part.is_empty()
                || (part.len() > 1 && part.starts_with('0'))
                || !part.bytes().all(|byte| byte.is_ascii_digit())
            {
                return Err(format!("Invalid application version component: {part}"));
            }
            part.parse::<u16>()
                .map_err(|_| format!("Application version component out of range: {part}"))
        };
        Ok(Self(
            parse_part(major)?,
            parse_part(minor)?,
            parse_part(patch)?,
        ))
    }

    fn current() -> crate::Result<Self> {
        Self::parse(env!("CARGO_PKG_VERSION"))
    }
}

impl fmt::Display for InstanceVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.0, self.1, self.2)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InstanceRecord {
    schema_version: u32,
    version: String,
    pid: u32,
    creation_time_file_time: u64,
}

pub(super) fn acquire(directory: &Path) -> crate::Result<Option<Guard>> {
    acquire_for_version(directory, InstanceVersion::current()?)
}

fn acquire_for_version(directory: &Path, current: InstanceVersion) -> crate::Result<Option<Guard>> {
    super::external(directory)?;
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let lock_path = directory.join("chat.lock");
    let meta_path = directory.join("chat.instance.json");
    super::external(&lock_path)?;
    super::external(&meta_path)?;

    if let Some(lock) = try_open_lock(&lock_path)? {
        return write_metadata_and_guard(lock, &meta_path, current).map(Some);
    }

    #[cfg(windows)]
    {
        acquire_contended(directory, &lock_path, &meta_path, current)
    }
    #[cfg(not(windows))]
    {
        let _ = (current, &meta_path);
        Ok(None)
    }
}

fn try_open_lock(path: &Path) -> crate::Result<Option<File>> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        // Keep the sentinel at one path, with no delete-sharing. Open the leaf
        // itself so a replacement cannot redirect the ordinary-file check.
        options.share_mode(0).custom_flags(0x0020_0000);
    }
    let guard = match options.open(path) {
        Ok(guard) => guard,
        #[cfg(windows)]
        Err(error) if matches!(error.raw_os_error(), Some(32 | 33)) => return Ok(None),
        Err(error) => return Err(format!("Cannot acquire native chat admission: {error}")),
    };
    let metadata = guard.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file() || is_redirect(&metadata) {
        return Err("Native chat admission requires an ordinary lock file.".into());
    }
    match guard.try_lock() {
        Ok(()) => Ok(Some(guard)),
        Err(fs::TryLockError::WouldBlock) => Ok(None),
        Err(fs::TryLockError::Error(error)) => {
            Err(format!("Cannot lock native chat admission: {error}"))
        }
    }
}

fn write_metadata_and_guard(
    lock: File,
    meta_path: &Path,
    current: InstanceVersion,
) -> crate::Result<Guard> {
    super::external(meta_path)?;
    match fs::symlink_metadata(meta_path) {
        Ok(metadata) if !metadata.is_file() || is_redirect(&metadata) => {
            return Err("Native chat instance metadata must be an ordinary file.".into());
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("Cannot inspect native chat metadata: {error}")),
    }

    #[cfg(windows)]
    let creation_time_file_time = bonsai_windows_process::current_process_creation_time()
        .map_err(|error| format!("Cannot query current process creation time: {error}"))?;
    #[cfg(not(windows))]
    let creation_time_file_time = 0_u64;

    let record = InstanceRecord {
        schema_version: 1,
        version: current.to_string(),
        pid: std::process::id(),
        creation_time_file_time,
    };
    let bytes = serde_json::to_vec(&record).map_err(|error| error.to_string())?;
    crate::fs_guard::write_atomic_bytes(meta_path, &bytes)
        .map_err(|error| format!("Cannot write native chat instance metadata: {error}"))?;

    Ok(Guard {
        _lock: lock,
        metadata_path: meta_path.to_path_buf(),
    })
}

fn open_shared_readonly(path: &Path) -> Option<File> {
    super::external(path).ok()?;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        // Allow concurrent readers, writers, and delete/rename replacement while
        // inspecting the leaf without following reparse points.
        options.share_mode(7).custom_flags(0x0020_0000);
    }
    let file = options.open(path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.is_file() || is_redirect(&metadata) {
        return None;
    }
    Some(file)
}

fn read_bounded_bytes(file: File, limit: u64) -> Option<Vec<u8>> {
    let length = file.metadata().ok()?.len();
    if length > limit {
        return None;
    }
    let mut bytes = Vec::with_capacity(usize::try_from(length).ok()?);
    file.take(limit + 1).read_to_end(&mut bytes).ok()?;
    if u64::try_from(bytes.len()).ok()? > limit {
        return None;
    }
    Some(bytes)
}

#[cfg(any(windows, test))]
fn read_instance_record(path: &Path) -> Option<InstanceRecord> {
    let file = open_shared_readonly(path)?;
    let bytes = read_bounded_bytes(file, MAX_INSTANCE_RECORD_BYTES)?;
    let record: InstanceRecord = serde_json::from_slice(&bytes).ok()?;
    if record.schema_version != 1 {
        return None;
    }
    InstanceVersion::parse(&record.version).ok()?;
    Some(record)
}

fn parse_package_version_bytes(bytes: &[u8]) -> Option<InstanceVersion> {
    let trimmed = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    let value: Value = serde_json::from_slice(trimmed).ok()?;
    if value["kind"].as_str() != Some("bonsai-chat-package") {
        return None;
    }
    InstanceVersion::parse(value["version"].as_str()?).ok()
}

#[cfg(any(windows, test))]
fn read_bounded_package_version(package_json_path: &Path) -> Option<InstanceVersion> {
    let file = open_shared_readonly(package_json_path)?;
    let bytes = read_bounded_bytes(file, MAX_PACKAGE_METADATA_BYTES)?;
    parse_package_version_bytes(&bytes)
}

#[cfg(any(windows, test))]
fn version_from_bundle_executable(exe_path: &Path) -> Option<InstanceVersion> {
    let file = open_shared_readonly(exe_path)?;
    if file.metadata().ok()?.len() > MAX_BUNDLE_EXECUTABLE_BYTES {
        return None;
    }
    let mut archive = zip::ZipArchive::new(file).ok()?;
    let prefix = format!("{}:", crate::chat_package::bundle::LAUNCHER_PROTOCOL);
    if !archive.comment().starts_with(prefix.as_bytes()) {
        return None;
    }
    let entry = archive.by_name("package.json").ok()?;
    let size = entry.size();
    if size == 0 || size > MAX_PACKAGE_METADATA_BYTES {
        return None;
    }
    let mut bytes = Vec::with_capacity(usize::try_from(size).ok()?);
    entry
        .take(MAX_PACKAGE_METADATA_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if u64::try_from(bytes.len()).ok()? > MAX_PACKAGE_METADATA_BYTES {
        return None;
    }
    parse_package_version_bytes(&bytes)
}

#[cfg(any(windows, test))]
fn version_from_package_layout(exe_path: &Path) -> Option<InstanceVersion> {
    let name = exe_path.file_name().and_then(|value| value.to_str())?;
    if !name.eq_ignore_ascii_case("bonsai-tools.exe")
        && !name.eq_ignore_ascii_case("bonsai-launcher.exe")
        && !name.eq_ignore_ascii_case("bonsai-chat.exe")
    {
        return None;
    }
    let bin_dir = exe_path.parent()?;
    if !bin_dir
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|dir| dir.eq_ignore_ascii_case("bin"))
    {
        return None;
    }
    let package_root = bin_dir.parent()?;
    read_bounded_package_version(&package_root.join("package.json"))
}

#[cfg(windows)]
fn is_recognized_bonsai_executable(exe_path: &Path) -> bool {
    let Some(name) = exe_path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower == "bonsai-launcher.exe"
        || lower == "bonsai-tools.exe"
        || lower == "bonsai-chat.exe"
        || ((lower.starts_with("swiftbonsai2runner")
            || lower.starts_with("snapdragonbonsairunner"))
            && lower.ends_with(".exe"))
}

#[cfg(windows)]
fn version_from_pe_resource(exe_path: &Path) -> Option<InstanceVersion> {
    if !is_recognized_bonsai_executable(exe_path) {
        return None;
    }
    let (major, minor, patch, _) = bonsai_windows_process::file_fixed_version(exe_path)
        .ok()
        .flatten()?;
    if (major, minor, patch) == (0, 0, 0) {
        return None;
    }
    Some(InstanceVersion(major, minor, patch))
}

#[cfg(windows)]
fn version_from_legacy_chat_child(holder: &ProcessIdentity) -> Option<InstanceVersion> {
    let exe_name = holder
        .executable_path
        .as_deref()
        .and_then(|path| path.file_name())
        .and_then(|value| value.to_str())?;
    if !exe_name.eq_ignore_ascii_case("bonsai-tools.exe")
        && !exe_name.eq_ignore_ascii_case("bonsai-launcher.exe")
    {
        return None;
    }
    let children =
        bonsai_windows_process::process_direct_children(holder.pid, holder.creation_file_time)
            .ok()?;
    for child in children {
        let Some(child_exe) = child.executable_path.as_deref() else {
            continue;
        };
        let Some(child_name) = child_exe.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !child_name.eq_ignore_ascii_case("bonsai-chat.exe") {
            continue;
        }
        if let Some(version) =
            version_from_package_layout(child_exe).or_else(|| version_from_pe_resource(child_exe))
        {
            return Some(version);
        }
    }
    None
}

#[cfg(windows)]
fn resolve_holder_version(
    holder: &ProcessIdentity,
    record: Option<&InstanceRecord>,
) -> Option<InstanceVersion> {
    if let Some(record) = record
        && record.pid == holder.pid
        && record.creation_time_file_time == holder.creation_file_time
        && let Ok(version) = InstanceVersion::parse(&record.version)
    {
        return Some(version);
    }
    let exe_path = holder.executable_path.as_deref()?;
    version_from_bundle_executable(exe_path)
        .or_else(|| version_from_package_layout(exe_path))
        .or_else(|| version_from_pe_resource(exe_path))
        .or_else(|| version_from_legacy_chat_child(holder))
}

#[cfg(windows)]
fn collect_older_descendants(
    older_holders: &[&ProcessIdentity],
) -> std::collections::BTreeSet<(u32, u64)> {
    const MAX_DEPTH: usize = 8;
    const MAX_DESCENDANTS: usize = 64;

    let mut descendants = std::collections::BTreeSet::new();
    let mut queue: std::collections::VecDeque<(u32, u64, usize)> = older_holders
        .iter()
        .map(|holder| (holder.pid, holder.creation_file_time, 0))
        .collect();
    while let Some((pid, creation, depth)) = queue.pop_front() {
        if depth >= MAX_DEPTH {
            continue;
        }
        let Ok(children) = bonsai_windows_process::process_direct_children(pid, creation) else {
            continue;
        };
        for child in children {
            if descendants.len() >= MAX_DESCENDANTS {
                return descendants;
            }
            if descendants.insert((child.pid, child.creation_file_time)) {
                queue.push_back((child.pid, child.creation_file_time, depth + 1));
            }
        }
    }
    descendants
}

#[cfg(windows)]
fn acquire_contended(
    directory: &Path,
    lock_path: &Path,
    meta_path: &Path,
    current: InstanceVersion,
) -> crate::Result<Option<Guard>> {
    const MAX_ROUNDS: usize = 4;

    let start = Instant::now();
    for _round in 0..MAX_ROUNDS {
        let Ok(mut holders) = bonsai_windows_process::file_lock_holders(lock_path) else {
            return Ok(None);
        };
        if holders.is_empty() {
            for _ in 0..10 {
                if let Some(lock) = try_open_lock(lock_path)? {
                    return write_metadata_and_guard(lock, meta_path, current).map(Some);
                }
                if start.elapsed() >= REPLACE_ALLOWANCE {
                    return Ok(None);
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            return Ok(None);
        }

        let self_pid = std::process::id();
        if holders.iter().any(|holder| holder.pid == self_pid) {
            return Ok(None);
        }

        let mut record = read_instance_record(meta_path);
        let record_matches_any = |rec: &InstanceRecord, list: &[ProcessIdentity]| {
            list.iter().any(|holder| {
                rec.pid == holder.pid && rec.creation_time_file_time == holder.creation_file_time
            })
        };
        if record
            .as_ref()
            .is_none_or(|rec| !record_matches_any(rec, &holders))
        {
            // Give a simultaneous peer launcher a brief moment to finish publishing `chat.instance.json`.
            std::thread::sleep(Duration::from_millis(50));
            if let Some(lock) = try_open_lock(lock_path)? {
                return write_metadata_and_guard(lock, meta_path, current).map(Some);
            }
            holders = match bonsai_windows_process::file_lock_holders(lock_path) {
                Ok(list) if !list.is_empty() => list,
                Ok(_) => {
                    if let Some(lock) = try_open_lock(lock_path)? {
                        return write_metadata_and_guard(lock, meta_path, current).map(Some);
                    }
                    return Ok(None);
                }
                Err(_) => return Ok(None),
            };
            if holders.iter().any(|holder| holder.pid == self_pid) {
                return Ok(None);
            }
            record = read_instance_record(meta_path);
        }

        let resolved: Vec<(&ProcessIdentity, Option<InstanceVersion>)> = holders
            .iter()
            .map(|holder| (holder, resolve_holder_version(holder, record.as_ref())))
            .collect();
        let older_holders: Vec<&ProcessIdentity> = resolved
            .iter()
            .filter_map(|(holder, version)| {
                version.is_some_and(|ver| ver < current).then_some(*holder)
            })
            .collect();
        if older_holders.is_empty() {
            return Ok(None);
        }
        let older_descendants = collect_older_descendants(&older_holders);
        let all_killable = resolved.iter().all(|(holder, version)| match version {
            Some(ver) => *ver < current,
            None => older_descendants.contains(&(holder.pid, holder.creation_file_time)),
        });
        if !all_killable {
            return Ok(None);
        }

        for holder in &older_holders {
            if older_descendants.contains(&(holder.pid, holder.creation_file_time)) {
                continue;
            }
            let Some(remaining) = REPLACE_ALLOWANCE
                .checked_sub(start.elapsed())
                .filter(|remaining| !remaining.is_zero())
            else {
                return Err(format!(
                    "Timed out replacing an older running instance in {} before opening version {current}.",
                    directory.display()
                ));
            };
            bonsai_windows_process::terminate_process_tree(
                holder.pid,
                holder.creation_file_time,
                remaining,
            )
            .map_err(|error| {
                format!(
                    "Swift Bonsai 2 found an older running instance (PID {}), but could not stop it: {error}",
                    holder.pid
                )
            })?;
        }

        let poll_deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < poll_deadline && start.elapsed() < REPLACE_ALLOWANCE {
            if let Some(lock) = try_open_lock(lock_path)? {
                return write_metadata_and_guard(lock, meta_path, current).map(Some);
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    if let Some(lock) = try_open_lock(lock_path)? {
        return write_metadata_and_guard(lock, meta_path, current).map(Some);
    }
    Err(format!(
        "Stopped an older running Swift Bonsai 2 instance, but could not acquire {} within {} seconds.",
        lock_path.display(),
        REPLACE_ALLOWANCE.as_secs()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use zip::{CompressionMethod, ZipWriter, write::SimpleFileOptions};

    fn directory(name: &str) -> crate::Result<PathBuf> {
        let directory = std::env::temp_dir().join(format!(
            "SwiftBonsai-ChatInstance-{name}-{}",
            crate::startup_process::unique()
        ));
        super::super::external(&directory)?;
        fs::create_dir(&directory).map_err(|error| error.to_string())?;
        Ok(directory)
    }

    #[test]
    fn singleton_retains_admission_until_drop_and_keeps_persistent_sentinel() -> crate::Result<()> {
        let directory = directory("lifetime")?;
        let owner = acquire(&directory)?.ok_or("First launcher was rejected.")?;
        assert!(acquire(&directory)?.is_none());
        assert!(directory.join("chat.lock").is_file());
        assert!(directory.join("chat.instance.json").is_file());
        let record = read_instance_record(&directory.join("chat.instance.json"))
            .ok_or("Expected instance metadata while lock is held.")?;
        assert_eq!(record.pid, std::process::id());
        assert_eq!(record.version, env!("CARGO_PKG_VERSION"));
        drop(owner);
        assert!(!directory.join("chat.instance.json").exists());
        let next = acquire(&directory)?.ok_or("Admission survived owner teardown.")?;
        assert!(directory.join("chat.lock").is_file());
        assert!(directory.join("chat.instance.json").is_file());
        drop(next);
        assert!(!directory.join("chat.instance.json").exists());
        fs::remove_dir_all(&directory).map_err(|error| error.to_string())
    }

    #[test]
    fn singleton_storage_errors_are_not_reported_as_duplicates() -> crate::Result<()> {
        let directory = directory("storage-error")?;
        fs::create_dir(directory.join("chat.lock")).map_err(|error| error.to_string())?;
        assert!(acquire(&directory).is_err());
        assert!(acquire(Path::new("relative-storage")).is_err());
        fs::remove_dir_all(&directory).map_err(|error| error.to_string())
    }

    #[test]
    fn instance_version_parses_canonical_semver_and_orders_numerically() -> crate::Result<()> {
        let v038 = InstanceVersion::parse("0.3.8")?;
        let v039 = InstanceVersion::parse("0.3.9")?;
        let v0310 = InstanceVersion::parse("0.3.10")?;
        let v040 = InstanceVersion::parse("0.4.0")?;
        let v100 = InstanceVersion::parse("1.0.0")?;
        assert!(v038 < v039);
        assert!(v039 < v0310);
        assert!(v0310 < v040);
        assert!(v040 < v100);
        assert_eq!(v039.to_string(), "0.3.9");

        for invalid in ["", "1", "1.2", "1.2.3.4", "01.2.3", "1.02.3", "1.2.3-rc1"] {
            assert!(InstanceVersion::parse(invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn legacy_version_resolves_from_bundle_executable_and_package_layout() -> crate::Result<()> {
        let root = directory("legacy-version")?;

        let bundle_path = root.join("renamed-portable.exe");
        {
            let file = File::create(&bundle_path).map_err(|error| error.to_string())?;
            let mut writer = ZipWriter::new(file);
            writer
                .set_comment(format!(
                    "{}:{}",
                    crate::chat_package::bundle::LAUNCHER_PROTOCOL,
                    "a".repeat(64)
                ))
                .map_err(|error| error.to_string())?;
            writer
                .start_file(
                    "package.json",
                    SimpleFileOptions::default().compression_method(CompressionMethod::Deflated),
                )
                .map_err(|error| error.to_string())?;
            writer
                .write_all(br#"{"kind":"bonsai-chat-package","version":"0.3.4"}"#)
                .map_err(|error| error.to_string())?;
            writer.finish().map_err(|error| error.to_string())?;
        }
        assert_eq!(
            version_from_bundle_executable(&bundle_path),
            Some(InstanceVersion(0, 3, 4))
        );

        let wrong_bundle = root.join("not-bonsai.exe");
        {
            let file = File::create(&wrong_bundle).map_err(|error| error.to_string())?;
            let mut writer = ZipWriter::new(file);
            writer
                .start_file("package.json", SimpleFileOptions::default())
                .map_err(|error| error.to_string())?;
            writer
                .write_all(br#"{"kind":"bonsai-chat-package","version":"0.3.4"}"#)
                .map_err(|error| error.to_string())?;
            writer.finish().map_err(|error| error.to_string())?;
        }
        assert_eq!(version_from_bundle_executable(&wrong_bundle), None);

        let pkg_bin = root.join("installed/bin");
        fs::create_dir_all(&pkg_bin).map_err(|error| error.to_string())?;
        fs::write(
            root.join("installed/package.json"),
            br#"{"kind":"bonsai-chat-package","version":"0.3.5"}"#,
        )
        .map_err(|error| error.to_string())?;
        let tools_exe = pkg_bin.join("bonsai-tools.exe");
        fs::write(&tools_exe, b"").map_err(|error| error.to_string())?;
        assert_eq!(
            version_from_package_layout(&tools_exe),
            Some(InstanceVersion(0, 3, 5))
        );

        fs::remove_dir_all(&root).map_err(|error| error.to_string())
    }

    #[cfg(windows)]
    fn spawn_lock_holder(
        lock_path: &Path,
        command_line: &str,
    ) -> crate::Result<std::process::Child> {
        use std::os::windows::fs::OpenOptionsExt as _;
        use std::os::windows::process::CommandExt as _;
        use std::process::{Command, Stdio};

        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .share_mode(0)
            .custom_flags(0x0020_0000)
            .open(lock_path)
            .map_err(|error| error.to_string())?;
        let comspec = std::env::var_os("ComSpec").ok_or("Missing ComSpec")?;
        Command::new(comspec)
            .args(["/d", "/c", command_line])
            .creation_flags(0x0800_0000)
            .stdin(Stdio::from(lock))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| error.to_string())
    }

    #[cfg(windows)]
    fn write_test_record(
        directory: &Path,
        version: &str,
        pid: u32,
        creation_time_file_time: u64,
    ) -> crate::Result<()> {
        let record = InstanceRecord {
            schema_version: 1,
            version: version.into(),
            pid,
            creation_time_file_time,
        };
        let bytes = serde_json::to_vec(&record).map_err(|error| error.to_string())?;
        crate::fs_guard::write_atomic_bytes(&directory.join("chat.instance.json"), &bytes)
            .map_err(|error| error.to_string())
    }

    #[cfg(windows)]
    #[test]
    fn newer_version_terminates_older_process_tree_and_takes_over() -> crate::Result<()> {
        let dir = directory("replace-older")?;
        let lock_path = dir.join("chat.lock");
        let mut child = spawn_lock_holder(&lock_path, "ping -n 30 127.0.0.1 >nul")?;
        let child_pid = child.id();
        let child_creation = bonsai_windows_process::process_creation_file_time(child_pid)
            .map_err(|error| error.to_string())?;

        let result = (|| -> crate::Result<()> {
            let mut grandchildren = Vec::new();
            for _ in 0..40 {
                grandchildren =
                    bonsai_windows_process::process_direct_children(child_pid, child_creation)
                        .map_err(|error| error.to_string())?;
                if !grandchildren.is_empty() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            assert!(!grandchildren.is_empty());

            write_test_record(&dir, "0.3.8", child_pid, child_creation)?;
            let guard = acquire_for_version(&dir, InstanceVersion(0, 3, 9))?
                .ok_or("Expected newer version to terminate older holder and take over.")?;

            let status = child.try_wait().map_err(|error| error.to_string())?;
            assert!(status.is_some());

            let updated = read_instance_record(&dir.join("chat.instance.json"))
                .ok_or("Missing updated instance record.")?;
            assert_eq!(updated.version, "0.3.9");
            assert_eq!(updated.pid, std::process::id());

            drop(guard);
            assert!(!dir.join("chat.instance.json").exists());
            Ok(())
        })();

        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_dir_all(&dir);
        result
    }

    #[cfg(windows)]
    #[test]
    fn same_or_newer_version_keeps_existing_process() -> crate::Result<()> {
        let dir = directory("keep-same-or-newer")?;
        let lock_path = dir.join("chat.lock");
        let mut child = spawn_lock_holder(&lock_path, "ping -n 30 127.0.0.1 >nul")?;
        let child_pid = child.id();
        let child_creation = bonsai_windows_process::process_creation_file_time(child_pid)
            .map_err(|error| error.to_string())?;

        let result = (|| -> crate::Result<()> {
            write_test_record(&dir, "0.3.9", child_pid, child_creation)?;
            assert!(acquire_for_version(&dir, InstanceVersion(0, 3, 9))?.is_none());
            assert!(
                child
                    .try_wait()
                    .map_err(|error| error.to_string())?
                    .is_none()
            );

            write_test_record(&dir, "0.4.0", child_pid, child_creation)?;
            assert!(acquire_for_version(&dir, InstanceVersion(0, 3, 9))?.is_none());
            assert!(
                child
                    .try_wait()
                    .map_err(|error| error.to_string())?
                    .is_none()
            );
            Ok(())
        })();

        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_dir_all(&dir);
        result
    }

    #[cfg(windows)]
    #[test]
    fn unknown_holder_or_tampered_metadata_never_terminates_existing_process() -> crate::Result<()>
    {
        use std::os::windows::process::CommandExt as _;
        use std::process::{Command, Stdio};

        let dir = directory("unknown-or-tampered")?;
        let lock_path = dir.join("chat.lock");
        let mut holder = spawn_lock_holder(&lock_path, "ping -n 30 127.0.0.1 >nul")?;
        let comspec = std::env::var_os("ComSpec").ok_or("Missing ComSpec")?;
        let mut decoy = Command::new(comspec)
            .args(["/d", "/c", "ping -n 30 127.0.0.1 >nul"])
            .creation_flags(0x0800_0000)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| error.to_string())?;

        let result = (|| -> crate::Result<()> {
            // No metadata and non-Bonsai executable (`cmd.exe`) must keep holder alive.
            assert!(acquire_for_version(&dir, InstanceVersion(0, 3, 9))?.is_none());
            assert!(
                holder
                    .try_wait()
                    .map_err(|error| error.to_string())?
                    .is_none()
            );

            // Tampered metadata pointing to a separate decoy PID must kill neither decoy nor holder.
            let decoy_creation = bonsai_windows_process::process_creation_file_time(decoy.id())
                .map_err(|error| error.to_string())?;
            write_test_record(&dir, "0.1.0", decoy.id(), decoy_creation)?;
            assert!(acquire_for_version(&dir, InstanceVersion(0, 3, 9))?.is_none());
            assert!(
                holder
                    .try_wait()
                    .map_err(|error| error.to_string())?
                    .is_none()
            );
            assert!(
                decoy
                    .try_wait()
                    .map_err(|error| error.to_string())?
                    .is_none()
            );
            Ok(())
        })();

        let _ = holder.kill();
        let _ = holder.wait();
        let _ = decoy.kill();
        let _ = decoy.wait();
        let _ = fs::remove_dir_all(&dir);
        result
    }
}
