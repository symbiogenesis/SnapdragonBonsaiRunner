//! Default per-user data directory resolution and one-time legacy AppData migration.

use std::{
    ffi::OsString,
    fs::{self, File},
    io::{self, Read as _},
    path::{Path, PathBuf},
};

use bonsai_artifacts::{is_redirect, validate_external_path};

pub(crate) const DEFAULT_DIR_NAME: &str = "SnapdragonBonsai";
pub(crate) const LEGACY_DIR_NAME: &str = "SwiftBonsai2";
const TRANSIENT_DIRS: &[&str] = &["build", "staging", "application-bundles"];
const LOCK_FILES: &[&str] = &["chat.lock", "chat.instance.json"];
const MAX_DEDUP_COMPARE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct MigrationSummary {
    pub(crate) moved_entries: usize,
    pub(crate) merged_dirs: usize,
    pub(crate) removed_transient_dirs: usize,
    pub(crate) deduplicated_files: usize,
    pub(crate) retained_conflicts: usize,
    pub(crate) legacy_removed: bool,
}

pub(crate) fn default_data_directory() -> crate::Result<PathBuf> {
    let local_app_data = std::env::var_os("LOCALAPPDATA")
        .filter(|value| !value.is_empty())
        .ok_or("LOCALAPPDATA is required.")?;
    Ok(PathBuf::from(local_app_data).join(DEFAULT_DIR_NAME))
}

pub(crate) fn legacy_data_directory() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .filter(|value| !value.is_empty())
        .map(|local_app_data| PathBuf::from(local_app_data).join(LEGACY_DIR_NAME))
}

pub(crate) fn default_subpath(relative: &str) -> crate::Result<PathBuf> {
    Ok(default_data_directory()?.join(relative))
}

pub(crate) fn legacy_subpath(relative: &str) -> Option<PathBuf> {
    legacy_data_directory().map(|dir| dir.join(relative))
}

fn normalize_path_text(path: &Path) -> String {
    let raw = path.to_string_lossy();
    let stripped = raw.strip_prefix(r"\\?\UNC\").map_or_else(
        || raw.strip_prefix(r"\\?\").unwrap_or(&raw).to_owned(),
        |unc| format!(r"\\{unc}"),
    );
    let mut normalized = stripped.replace('\\', "/");
    while normalized.len() > 1 && normalized.ends_with('/') && !normalized.ends_with(":/") {
        normalized.pop();
    }
    normalized
}

fn paths_match_normalized(a: &Path, b: &Path) -> bool {
    let left = normalize_path_text(a);
    let right = normalize_path_text(b);
    if cfg!(windows) {
        left.eq_ignore_ascii_case(&right)
    } else {
        left == right
    }
}

pub(crate) fn is_default_data_directory(candidate: &Path, default_dir: &Path) -> bool {
    if paths_match_normalized(candidate, default_dir) {
        return true;
    }
    if let (Ok(left), Ok(right)) = (fs::canonicalize(candidate), fs::canonicalize(default_dir)) {
        return left == right;
    }
    false
}

pub(crate) fn legacy_dir_exists(legacy: &Path) -> crate::Result<bool> {
    match fs::symlink_metadata(legacy) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "Cannot inspect legacy data directory {}: {error}",
            legacy.display()
        )),
        Ok(metadata) if is_redirect(&metadata) => Err(format!(
            "Refusing to migrate redirected legacy data directory: {}",
            legacy.display()
        )),
        Ok(metadata) => Ok(metadata.is_dir()),
    }
}

fn sorted_dir_entries(dir: &Path) -> crate::Result<Vec<(OsString, PathBuf)>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(dir)
        .map_err(|error| format!("Cannot read directory {}: {error}", dir.display()))?
    {
        let entry = entry
            .map_err(|error| format!("Cannot enumerate directory {}: {error}", dir.display()))?;
        entries.push((entry.file_name(), entry.path()));
    }
    entries.sort_by(|(left, _), (right, _)| left.cmp(right));
    Ok(entries)
}

fn is_named(name: &OsString, candidates: &[&str]) -> bool {
    name.to_str().is_some_and(|text| {
        candidates
            .iter()
            .any(|candidate| text.eq_ignore_ascii_case(candidate))
    })
}

fn read_chunk(file: &mut File, buf: &mut [u8]) -> io::Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        let count = file.read(&mut buf[total..])?;
        if count == 0 {
            break;
        }
        total += count;
    }
    Ok(total)
}

fn files_equal(left_path: &Path, right_path: &Path) -> io::Result<bool> {
    let mut left = File::open(left_path)?;
    let mut right = File::open(right_path)?;
    let mut left_buf = vec![0_u8; 65_536];
    let mut right_buf = vec![0_u8; 65_536];
    loop {
        let left_len = read_chunk(&mut left, &mut left_buf)?;
        let right_len = read_chunk(&mut right, &mut right_buf)?;
        if left_len != right_len || left_buf[..left_len] != right_buf[..right_len] {
            return Ok(false);
        }
        if left_len == 0 {
            return Ok(true);
        }
    }
}

fn move_or_merge_entry(
    src: &Path,
    dest: &Path,
    summary: &mut MigrationSummary,
) -> crate::Result<()> {
    crate::startup_cancel::checkpoint()?;
    let src_meta = fs::symlink_metadata(src)
        .map_err(|error| format!("Cannot inspect legacy entry {}: {error}", src.display()))?;
    if is_redirect(&src_meta) {
        return Err(format!(
            "Refusing to migrate redirected entry: {}",
            src.display()
        ));
    }

    match fs::symlink_metadata(dest) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            crate::fs_guard::rename_with_retry(src, dest).map_err(|error| {
                format!(
                    "Cannot move {} to {}: {error}",
                    src.display(),
                    dest.display()
                )
            })?;
            summary.moved_entries += 1;
            Ok(())
        }
        Err(error) => Err(format!(
            "Cannot inspect target entry {}: {error}",
            dest.display()
        )),
        Ok(dest_meta) if is_redirect(&dest_meta) => Err(format!(
            "Refusing to merge into redirected target entry: {}",
            dest.display()
        )),
        Ok(dest_meta) if src_meta.is_dir() && dest_meta.is_dir() => {
            for (child_name, child_src) in sorted_dir_entries(src)? {
                move_or_merge_entry(&child_src, &dest.join(&child_name), summary)?;
            }
            let mut remaining = fs::read_dir(src).map_err(|error| {
                format!(
                    "Cannot inspect merged legacy directory {}: {error}",
                    src.display()
                )
            })?;
            if remaining.next().is_none() {
                crate::fs_guard::remove_dir_with_retry(src).map_err(|error| {
                    format!(
                        "Cannot remove empty merged legacy directory {}: {error}",
                        src.display()
                    )
                })?;
                summary.merged_dirs += 1;
            }
            Ok(())
        }
        Ok(dest_meta) if src_meta.is_file() && dest_meta.is_file() => {
            if src_meta.len() == dest_meta.len()
                && src_meta.len() <= MAX_DEDUP_COMPARE_BYTES
                && files_equal(src, dest).unwrap_or(false)
            {
                crate::fs_guard::remove_file_with_retry(src).map_err(|error| {
                    format!(
                        "Cannot remove duplicate legacy file {}: {error}",
                        src.display()
                    )
                })?;
                summary.deduplicated_files += 1;
            } else {
                summary.retained_conflicts += 1;
            }
            Ok(())
        }
        Ok(_) => {
            summary.retained_conflicts += 1;
            Ok(())
        }
    }
}

fn dir_contains_exe(dir: &Path, active_exe: Option<&Path>, canonical_exe: Option<&Path>) -> bool {
    if let (Some(exe), Ok(canon_dir)) = (canonical_exe, fs::canonicalize(dir)) {
        return exe.starts_with(&canon_dir);
    }
    let Some(exe) = active_exe else {
        return false;
    };
    let dir_norm = normalize_path_text(dir);
    let exe_norm = normalize_path_text(exe);
    let prefix = if dir_norm.ends_with('/') {
        dir_norm
    } else {
        format!("{dir_norm}/")
    };
    if cfg!(windows) {
        exe_norm.len() >= prefix.len() && exe_norm[..prefix.len()].eq_ignore_ascii_case(&prefix)
    } else {
        exe_norm.starts_with(&prefix)
    }
}

fn is_dir_not_empty(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::DirectoryNotEmpty {
        return true;
    }
    #[cfg(windows)]
    {
        matches!(error.raw_os_error(), Some(145))
    }
    #[cfg(not(windows))]
    {
        false
    }
}

pub(crate) fn migrate_directory_with_lock_release(
    legacy: &Path,
    target: &Path,
    active_exe: Option<&Path>,
    release_legacy_lock: impl FnOnce(),
) -> crate::Result<MigrationSummary> {
    validate_external_path(legacy).map_err(|error| error.to_string())?;
    validate_external_path(target).map_err(|error| error.to_string())?;

    let legacy_meta = match fs::symlink_metadata(legacy) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            release_legacy_lock();
            return Ok(MigrationSummary::default());
        }
        Err(error) => {
            return Err(format!(
                "Cannot inspect legacy data directory {}: {error}",
                legacy.display()
            ));
        }
        Ok(meta) => meta,
    };
    if is_redirect(&legacy_meta) {
        return Err(format!(
            "Refusing to migrate redirected legacy data directory: {}",
            legacy.display()
        ));
    }
    if !legacy_meta.is_dir() {
        return Err(format!(
            "Legacy data path is not a directory: {}",
            legacy.display()
        ));
    }
    if is_default_data_directory(legacy, target) {
        return Err("Legacy and target data directories must differ.".into());
    }

    fs::create_dir_all(target).map_err(|error| {
        format!(
            "Cannot create target data directory {}: {error}",
            target.display()
        )
    })?;
    validate_external_path(target).map_err(|error| error.to_string())?;
    let target_meta = fs::symlink_metadata(target).map_err(|error| {
        format!(
            "Cannot inspect target data directory {}: {error}",
            target.display()
        )
    })?;
    if is_redirect(&target_meta) || !target_meta.is_dir() {
        return Err(format!(
            "Target data path must be an ordinary directory: {}",
            target.display()
        ));
    }

    let mut summary = MigrationSummary::default();
    let mut transient_dirs = Vec::new();

    for (name, src) in sorted_dir_entries(legacy)? {
        crate::startup_cancel::checkpoint()?;
        let meta = fs::symlink_metadata(&src)
            .map_err(|error| format!("Cannot inspect legacy entry {}: {error}", src.display()))?;
        if is_redirect(&meta) {
            return Err(format!(
                "Refusing to migrate redirected legacy entry: {}",
                src.display()
            ));
        }
        if is_named(&name, LOCK_FILES) {
            continue;
        }
        if is_named(&name, TRANSIENT_DIRS) && meta.is_dir() {
            transient_dirs.push(src);
            continue;
        }
        move_or_merge_entry(&src, &target.join(&name), &mut summary)?;
    }

    let canonical_exe = active_exe.and_then(|path| fs::canonicalize(path).ok());
    for dir in transient_dirs {
        crate::startup_cancel::checkpoint()?;
        if dir_contains_exe(&dir, active_exe, canonical_exe.as_deref()) {
            continue;
        }
        match crate::fs_guard::remove_dir_all_with_retry(&dir) {
            Ok(()) => summary.removed_transient_dirs += 1,
            Err(error) if crate::fs_guard::is_windows_sharing_or_access(&error) => {}
            Err(error) => {
                return Err(format!(
                    "Cannot remove transient legacy directory {}: {error}",
                    dir.display()
                ));
            }
        }
    }

    release_legacy_lock();

    for lock_name in LOCK_FILES {
        let lock_path = legacy.join(lock_name);
        if let Ok(meta) = fs::symlink_metadata(&lock_path)
            && meta.is_file()
            && !is_redirect(&meta)
            && let Err(error) = crate::fs_guard::remove_file_with_retry(&lock_path)
            && !crate::fs_guard::is_windows_sharing_or_access(&error)
        {
            return Err(format!(
                "Cannot remove legacy lock file {}: {error}",
                lock_path.display()
            ));
        }
    }

    match crate::fs_guard::remove_dir_with_retry(legacy) {
        Ok(()) => summary.legacy_removed = true,
        Err(error)
            if is_dir_not_empty(&error)
                || crate::fs_guard::is_windows_sharing_or_access(&error) =>
        {
            summary.legacy_removed = false;
        }
        Err(error) => {
            return Err(format!(
                "Cannot remove empty legacy data directory {}: {error}",
                legacy.display()
            ));
        }
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs_guard::TempDirGuard;

    fn scratch_root(label: &str) -> crate::Result<TempDirGuard> {
        let root = std::env::temp_dir().join(format!(
            "Bonsai-DataDir-{label}-{}",
            crate::startup_process::unique()
        ));
        validate_external_path(&root).map_err(|error| error.to_string())?;
        fs::create_dir(&root).map_err(|error| error.to_string())?;
        Ok(TempDirGuard::new(root))
    }

    #[test]
    fn clean_migration_moves_persistent_data_drops_transients_and_removes_legacy_dir()
    -> crate::Result<()> {
        let mut guard = scratch_root("clean")?;
        let legacy = guard.path().join(LEGACY_DIR_NAME);
        let target = guard.path().join(DEFAULT_DIR_NAME);

        // Pre-populate legacy directory with persistent files, nested dirs, transient caches, and locks.
        fs::create_dir_all(legacy.join("models/rev1")).map_err(|e| e.to_string())?;
        fs::create_dir_all(legacy.join("speech/native")).map_err(|e| e.to_string())?;
        fs::create_dir_all(legacy.join("updates/0.3.9")).map_err(|e| e.to_string())?;
        fs::create_dir_all(legacy.join("build/maintenance")).map_err(|e| e.to_string())?;
        fs::create_dir_all(legacy.join("staging/tmp")).map_err(|e| e.to_string())?;
        fs::create_dir_all(legacy.join("application-bundles/abc")).map_err(|e| e.to_string())?;

        fs::write(legacy.join("chat-recovery.dpapi"), b"encrypted-chats")
            .map_err(|e| e.to_string())?;
        fs::write(legacy.join("chat-recovery.lock"), b"1").map_err(|e| e.to_string())?;
        fs::write(legacy.join("chat-profile.dpapi"), b"encrypted-voice")
            .map_err(|e| e.to_string())?;
        fs::write(legacy.join("chat-window.json"), b"{\"version\":1}")
            .map_err(|e| e.to_string())?;
        fs::write(legacy.join("chat-appearance.json"), b"{\"version\":1}")
            .map_err(|e| e.to_string())?;
        fs::write(legacy.join("models/rev1/model.gguf"), b"weights").map_err(|e| e.to_string())?;
        fs::write(legacy.join("speech/native/tts.gguf"), b"tts").map_err(|e| e.to_string())?;
        fs::write(legacy.join("updates/0.3.9/installer.exe"), b"exe").map_err(|e| e.to_string())?;
        fs::write(
            legacy.join("build/maintenance/bonsai-tools.exe"),
            b"old-build",
        )
        .map_err(|e| e.to_string())?;
        fs::write(legacy.join("staging/tmp/part"), b"partial").map_err(|e| e.to_string())?;
        fs::write(legacy.join("application-bundles/abc/active.json"), b"{}")
            .map_err(|e| e.to_string())?;
        fs::write(legacy.join("chat.lock"), b"").map_err(|e| e.to_string())?;
        fs::write(legacy.join("chat.instance.json"), b"{}").map_err(|e| e.to_string())?;

        // Target already exists with its own new lock and maintenance build directory.
        fs::create_dir_all(target.join("build/maintenance")).map_err(|e| e.to_string())?;
        fs::write(target.join("chat.lock"), b"").map_err(|e| e.to_string())?;
        fs::write(
            target.join("build/maintenance/bonsai-tools.exe"),
            b"new-build",
        )
        .map_err(|e| e.to_string())?;

        let mut released = false;
        let summary = migrate_directory_with_lock_release(&legacy, &target, None, || {
            released = true;
        })?;
        assert!(released);
        assert!(summary.legacy_removed);
        assert_eq!(summary.removed_transient_dirs, 3);
        assert_eq!(summary.retained_conflicts, 0);
        assert!(!legacy.exists());

        assert_eq!(
            fs::read(target.join("chat-recovery.dpapi")).map_err(|e| e.to_string())?,
            b"encrypted-chats"
        );
        assert_eq!(
            fs::read(target.join("chat-recovery.lock")).map_err(|e| e.to_string())?,
            b"1"
        );
        assert_eq!(
            fs::read(target.join("chat-profile.dpapi")).map_err(|e| e.to_string())?,
            b"encrypted-voice"
        );
        assert_eq!(
            fs::read(target.join("chat-window.json")).map_err(|e| e.to_string())?,
            b"{\"version\":1}"
        );
        assert_eq!(
            fs::read(target.join("chat-appearance.json")).map_err(|e| e.to_string())?,
            b"{\"version\":1}"
        );
        assert_eq!(
            fs::read(target.join("models/rev1/model.gguf")).map_err(|e| e.to_string())?,
            b"weights"
        );
        assert_eq!(
            fs::read(target.join("speech/native/tts.gguf")).map_err(|e| e.to_string())?,
            b"tts"
        );
        assert_eq!(
            fs::read(target.join("updates/0.3.9/installer.exe")).map_err(|e| e.to_string())?,
            b"exe"
        );
        assert_eq!(
            fs::read(target.join("build/maintenance/bonsai-tools.exe"))
                .map_err(|e| e.to_string())?,
            b"new-build"
        );
        assert!(!target.join("staging").exists());
        assert!(!target.join("application-bundles").exists());

        guard.cleanup().map_err(|e| e.to_string())?;
        Ok(())
    }

    #[test]
    fn merge_deduplicates_identical_files_and_preserves_differing_conflicts() -> crate::Result<()> {
        let mut guard = scratch_root("merge")?;
        let legacy = guard.path().join(LEGACY_DIR_NAME);
        let target = guard.path().join(DEFAULT_DIR_NAME);

        fs::create_dir_all(legacy.join("updates/0.3.10")).map_err(|e| e.to_string())?;
        fs::create_dir_all(target.join("updates/0.3.10")).map_err(|e| e.to_string())?;

        // Identical file in nested directory -> deduplicated cleanly.
        fs::write(legacy.join("updates/0.3.10/a.json"), b"same-content")
            .map_err(|e| e.to_string())?;
        fs::write(target.join("updates/0.3.10/a.json"), b"same-content")
            .map_err(|e| e.to_string())?;
        // New file in nested directory -> moved cleanly.
        fs::write(legacy.join("updates/0.3.10/b.json"), b"only-in-legacy")
            .map_err(|e| e.to_string())?;
        // Differing file -> target preserved, legacy copy retained.
        fs::write(legacy.join("chat-window.json"), b"legacy-window").map_err(|e| e.to_string())?;
        fs::write(target.join("chat-window.json"), b"newer-target-window")
            .map_err(|e| e.to_string())?;

        let summary = migrate_directory_with_lock_release(&legacy, &target, None, || {})?;
        assert_eq!(summary.deduplicated_files, 1);
        assert_eq!(summary.moved_entries, 1);
        assert_eq!(summary.merged_dirs, 2); // updates/0.3.10 and updates
        assert_eq!(summary.retained_conflicts, 1);
        assert!(!summary.legacy_removed);

        assert_eq!(
            fs::read(target.join("chat-window.json")).map_err(|e| e.to_string())?,
            b"newer-target-window"
        );
        assert_eq!(
            fs::read(legacy.join("chat-window.json")).map_err(|e| e.to_string())?,
            b"legacy-window"
        );
        assert_eq!(
            fs::read(target.join("updates/0.3.10/b.json")).map_err(|e| e.to_string())?,
            b"only-in-legacy"
        );
        assert!(!legacy.join("updates").exists());

        // Resolving the conflict manually allows subsequent migration to clean up the empty legacy dir.
        fs::remove_file(legacy.join("chat-window.json")).map_err(|e| e.to_string())?;
        let rerun = migrate_directory_with_lock_release(&legacy, &target, None, || {})?;
        assert!(rerun.legacy_removed);
        assert!(!legacy.exists());

        guard.cleanup().map_err(|e| e.to_string())?;
        Ok(())
    }

    #[test]
    fn active_executable_inside_legacy_transient_dir_skips_deleting_that_dir() -> crate::Result<()>
    {
        let mut guard = scratch_root("active-exe")?;
        let legacy = guard.path().join(LEGACY_DIR_NAME);
        let target = guard.path().join(DEFAULT_DIR_NAME);

        let exe = legacy.join("build/maintenance/bonsai-tools.exe");
        fs::create_dir_all(exe.parent().ok_or("missing parent")?).map_err(|e| e.to_string())?;
        fs::write(&exe, b"running").map_err(|e| e.to_string())?;
        fs::write(legacy.join("chat-recovery.dpapi"), b"chats").map_err(|e| e.to_string())?;
        fs::create_dir_all(legacy.join("staging")).map_err(|e| e.to_string())?;

        let summary = migrate_directory_with_lock_release(&legacy, &target, Some(&exe), || {})?;
        assert!(!summary.legacy_removed);
        assert_eq!(summary.removed_transient_dirs, 1); // staging removed, build skipped
        assert!(exe.exists());
        assert!(!legacy.join("staging").exists());
        assert_eq!(
            fs::read(target.join("chat-recovery.dpapi")).map_err(|e| e.to_string())?,
            b"chats"
        );

        guard.cleanup().map_err(|e| e.to_string())?;
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn migration_refuses_windows_junction_without_touching_junction_target() -> crate::Result<()> {
        let mut guard = scratch_root("junction")?;
        let real_target = guard.path().join("real-target");
        let sentinel = real_target.join("keep.txt");
        let legacy = guard.path().join(LEGACY_DIR_NAME);
        let target = guard.path().join(DEFAULT_DIR_NAME);
        fs::create_dir(&real_target).map_err(|e| e.to_string())?;
        fs::write(&sentinel, b"do-not-touch").map_err(|e| e.to_string())?;

        let status = std::process::Command::new("cmd")
            .args([
                "/c",
                "mklink",
                "/J",
                &legacy.to_string_lossy(),
                &real_target.to_string_lossy(),
            ])
            .output()
            .map_err(|e| e.to_string())?;
        assert!(status.status.success(), "mklink /J failed");

        assert!(legacy_dir_exists(&legacy).is_err());
        assert!(migrate_directory_with_lock_release(&legacy, &target, None, || {}).is_err());
        assert_eq!(
            fs::read(&sentinel).map_err(|e| e.to_string())?,
            b"do-not-touch"
        );

        fs::remove_dir(&legacy).map_err(|e| e.to_string())?;
        guard.cleanup().map_err(|e| e.to_string())?;
        Ok(())
    }

    #[test]
    fn default_data_directory_matching_handles_normalization_and_custom_dirs() {
        let default_dir = Path::new(r"C:\Users\Owner\AppData\Local\SnapdragonBonsai");
        assert!(is_default_data_directory(
            Path::new(r"C:\Users\Owner\AppData\Local\SnapdragonBonsai"),
            default_dir,
        ));
        assert!(is_default_data_directory(
            Path::new("C:/Users/Owner/AppData/Local/SnapdragonBonsai/"),
            default_dir,
        ));
        if cfg!(windows) {
            assert!(is_default_data_directory(
                Path::new(r"\\?\c:\users\owner\appdata\local\snapdragonbonsai"),
                default_dir,
            ));
        }
        assert!(!is_default_data_directory(
            Path::new(r"D:\CustomBonsaiData"),
            default_dir,
        ));
    }

    #[test]
    fn migration_is_noop_when_legacy_directory_is_absent() -> crate::Result<()> {
        let mut guard = scratch_root("noop")?;
        let legacy = guard.path().join(LEGACY_DIR_NAME);
        let target = guard.path().join(DEFAULT_DIR_NAME);
        assert!(!legacy_dir_exists(&legacy)?);

        let mut released = false;
        let summary = migrate_directory_with_lock_release(&legacy, &target, None, || {
            released = true;
        })?;
        assert!(released);
        assert_eq!(summary, MigrationSummary::default());
        assert!(!target.exists());

        guard.cleanup().map_err(|e| e.to_string())?;
        Ok(())
    }
}
