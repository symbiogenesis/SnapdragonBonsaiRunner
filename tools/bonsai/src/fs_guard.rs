//! Shared drop guards and Windows sharing/access retry helpers for local filesystem paths.

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

/// Returns `true` when `error` is a Windows sharing, lock, or access-denied error
/// (Win32 error codes 5, 32, 33) or a permission-denied I/O error.
pub(crate) fn is_windows_sharing_or_access(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::PermissionDenied {
        return true;
    }
    #[cfg(windows)]
    {
        matches!(error.raw_os_error(), Some(5 | 32 | 33))
    }
    #[cfg(not(windows))]
    {
        false
    }
}

#[cfg(windows)]
pub(crate) fn clear_readonly(path: &Path) {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return;
    };
    if bonsai_artifacts::is_redirect(&metadata) || !metadata.permissions().readonly() {
        return;
    }
    let mut permissions = metadata.permissions();
    #[expect(
        clippy::permissions_set_readonly_false,
        reason = "Windows-only cleanup clears the DOS readonly attribute; this branch never changes Unix permissions."
    )]
    permissions.set_readonly(false);
    let _ = fs::set_permissions(path, permissions);
}

#[cfg(windows)]
pub(crate) fn clear_readonly_tree(path: &Path) {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return;
    };
    if bonsai_artifacts::is_redirect(&metadata) {
        return;
    }
    if metadata.is_dir()
        && let Ok(entries) = fs::read_dir(path)
    {
        for entry in entries.flatten() {
            clear_readonly_tree(&entry.path());
        }
    }
    if metadata.permissions().readonly() {
        let mut permissions = metadata.permissions();
        #[expect(
            clippy::permissions_set_readonly_false,
            reason = "Windows-only cleanup clears the DOS readonly attribute; this branch never changes Unix permissions."
        )]
        permissions.set_readonly(false);
        let _ = fs::set_permissions(path, permissions);
    }
}

fn retry_windows_sharing<T>(
    mut op: impl FnMut() -> io::Result<T>,
    mut prepare_retry: impl FnMut(),
) -> io::Result<T> {
    const RETRIES: usize = 7;
    for attempt in 0..RETRIES {
        match op() {
            Ok(value) => return Ok(value),
            Err(error) if is_windows_sharing_or_access(&error) => {
                prepare_retry();
                let shift = u32::try_from(attempt.min(3)).unwrap_or(3);
                std::thread::sleep(Duration::from_millis(20 << shift));
            }
            Err(error) => return Err(error),
        }
    }
    op()
}

/// Renames `from` to `to`, retrying on transient Windows sharing/access errors (5, 32, 33).
pub(crate) fn rename_with_retry(from: &Path, to: &Path) -> io::Result<()> {
    retry_windows_sharing(
        || fs::rename(from, to),
        || {
            #[cfg(windows)]
            {
                clear_readonly(from);
                clear_readonly(to);
            }
        },
    )
}

/// Removes `path` as a file, clearing DOS read-only attributes and retrying on Windows locks.
pub(crate) fn remove_file_with_retry(path: &Path) -> io::Result<()> {
    retry_windows_sharing(
        || match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        },
        || {
            #[cfg(windows)]
            clear_readonly(path);
        },
    )
}

/// Recursively removes `path` as a non-redirected directory, clearing read-only attributes and retrying on Windows locks.
pub(crate) fn remove_dir_all_with_retry(path: &Path) -> io::Result<()> {
    retry_windows_sharing(
        || match fs::symlink_metadata(path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
            Ok(metadata) if bonsai_artifacts::is_redirect(&metadata) => {
                Err(io::Error::other("Refusing to remove redirected directory."))
            }
            Ok(_) => match fs::remove_dir_all(path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(error),
            },
        },
        || {
            #[cfg(windows)]
            clear_readonly_tree(path);
        },
    )
}

/// Removes an empty non-redirected directory `path`, clearing read-only attributes and retrying on Windows locks.
pub(crate) fn remove_dir_with_retry(path: &Path) -> io::Result<()> {
    retry_windows_sharing(
        || match fs::symlink_metadata(path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
            Ok(metadata) if bonsai_artifacts::is_redirect(&metadata) => {
                Err(io::Error::other("Refusing to remove redirected directory."))
            }
            Ok(_) => match fs::remove_dir(path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(error),
            },
        },
        || {
            #[cfg(windows)]
            clear_readonly(path);
        },
    )
}

/// Removes a temporary or `.partial` file on drop unless explicitly disarmed.
pub(crate) struct TempFileGuard(Option<PathBuf>);

impl TempFileGuard {
    /// Arms a guard that deletes `path` when dropped unless disarmed.
    pub(crate) fn new(path: PathBuf) -> Self {
        Self(Some(path))
    }

    /// Disarms the guard after the file has been published or intentionally retained.
    pub(crate) fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = remove_file_with_retry(&path);
        }
    }
}

/// Removes a temporary or staging directory on drop unless explicitly disarmed.
pub(crate) struct TempDirGuard {
    path: PathBuf,
    armed: bool,
}

impl TempDirGuard {
    /// Arms a guard that removes `path` when dropped unless disarmed.
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    /// Returns the guarded directory path.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Disarms the guard after the directory has been published or intentionally exported.
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }

    /// Explicitly removes the directory with retry and disarms the guard.
    #[cfg(test)]
    pub(crate) fn cleanup(&mut self) -> io::Result<()> {
        self.armed = false;
        remove_dir_all_with_retry(&self.path)
    }
}

impl std::ops::Deref for TempDirGuard {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

impl AsRef<Path> for TempDirGuard {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = remove_dir_all_with_retry(&self.path);
        }
    }
}

/// Atomically writes `bytes` to `path` via a guarded `.partial` sibling and Windows rename retry.
pub(crate) fn write_atomic_bytes(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Output path requires a file name.",
        )
    })?;
    let temporary = path.with_file_name(format!(
        "{}.{}.partial",
        name.to_string_lossy(),
        crate::startup_process::unique()
    ));
    let mut guard = TempFileGuard::new(temporary.clone());
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    use io::Write as _;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    rename_with_retry(&temporary, path)?;
    guard.disarm();
    Ok(())
}

/// Atomically creates a new file at `path` via a guarded `.partial` sibling and hard-link publication.
pub(crate) fn write_new_atomic_bytes(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Output path requires a file name.",
        )
    })?;
    let temporary = path.with_file_name(format!(
        "{}.{}.partial",
        name.to_string_lossy(),
        crate::startup_process::unique()
    ));
    let mut guard = TempFileGuard::new(temporary.clone());
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    use io::Write as _;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::hard_link(&temporary, path)?;
    remove_file_with_retry(&temporary)?;
    guard.disarm();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_guards_clean_up_on_drop_and_skip_when_disarmed() -> crate::Result<()> {
        let root = std::env::temp_dir().join(format!(
            "Bonsai-FsGuard-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&root).map_err(|error| error.to_string())?;
        let mut _root_guard = TempDirGuard::new(root.clone());

        let file_drop = root.join("leaked.partial");
        {
            let _guard = TempFileGuard::new(file_drop.clone());
            fs::write(&file_drop, b"partial").map_err(|error| error.to_string())?;
            #[cfg(windows)]
            {
                let mut perms = fs::metadata(&file_drop)
                    .map_err(|error| error.to_string())?
                    .permissions();
                perms.set_readonly(true);
                fs::set_permissions(&file_drop, perms).map_err(|error| error.to_string())?;
            }
            assert!(file_drop.exists());
        }
        assert!(!file_drop.exists());

        let file_kept = root.join("kept.json");
        {
            let mut guard = TempFileGuard::new(file_kept.clone());
            fs::write(&file_kept, b"{}").map_err(|error| error.to_string())?;
            guard.disarm();
        }
        assert!(file_kept.is_file());

        let dir_drop = root.join("staging");
        {
            let _guard = TempDirGuard::new(dir_drop.clone());
            fs::create_dir(&dir_drop).map_err(|error| error.to_string())?;
            let nested = dir_drop.join("nested.bin");
            fs::write(&nested, b"bytes").map_err(|error| error.to_string())?;
            #[cfg(windows)]
            {
                let mut perms = fs::metadata(&nested)
                    .map_err(|error| error.to_string())?
                    .permissions();
                perms.set_readonly(true);
                fs::set_permissions(&nested, perms).map_err(|error| error.to_string())?;
            }
        }
        assert!(!dir_drop.exists());

        let dir_kept = root.join("exported");
        {
            let mut guard = TempDirGuard::new(dir_kept.clone());
            fs::create_dir(&dir_kept).map_err(|error| error.to_string())?;
            guard.disarm();
        }
        assert!(dir_kept.is_dir());

        _root_guard.cleanup().map_err(|error| error.to_string())?;
        assert!(!root.exists());
        Ok(())
    }

    #[test]
    fn atomic_writes_publish_cleanly_without_partial_leaks() -> crate::Result<()> {
        let root = std::env::temp_dir().join(format!(
            "Bonsai-AtomicWrite-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&root).map_err(|error| error.to_string())?;
        let mut root_guard = TempDirGuard::new(root.clone());

        let exclusive = root.join("exclusive.json");
        write_new_atomic_bytes(&exclusive, b"{\"v\":1}\n").map_err(|error| error.to_string())?;
        assert_eq!(
            fs::read(&exclusive).map_err(|error| error.to_string())?,
            b"{\"v\":1}\n"
        );
        assert!(write_new_atomic_bytes(&exclusive, b"{\"v\":2}\n").is_err());

        let manifest = root.join("SHA256SUMS.txt");
        write_atomic_bytes(&manifest, b"first\n").map_err(|error| error.to_string())?;
        write_atomic_bytes(&manifest, b"second\n").map_err(|error| error.to_string())?;
        assert_eq!(
            fs::read(&manifest).map_err(|error| error.to_string())?,
            b"second\n"
        );
        let partial_count = fs::read_dir(&root)
            .map_err(|error| error.to_string())?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "partial"))
            .count();
        assert_eq!(partial_count, 0);

        root_guard.cleanup().map_err(|error| error.to_string())?;
        Ok(())
    }

    #[test]
    fn retry_windows_sharing_exhausts_bounded_attempts() {
        let mut attempts = 0usize;
        let result: io::Result<()> = retry_windows_sharing(
            || {
                attempts += 1;
                Err(io::Error::from(io::ErrorKind::PermissionDenied))
            },
            || {},
        );
        assert_eq!(attempts, 8);
        assert!(result.is_err());

        let mut immediate_attempts = 0usize;
        let non_retryable: io::Result<()> = retry_windows_sharing(
            || {
                immediate_attempts += 1;
                Err(io::Error::from(io::ErrorKind::NotFound))
            },
            || {},
        );
        assert_eq!(immediate_attempts, 1);
        assert!(non_retryable.is_err());
    }

    #[test]
    fn fs_guard_rejects_junctions_clears_readonly_on_rename_and_cleans_atomic_errors()
    -> crate::Result<()> {
        let root = std::env::temp_dir().join(format!(
            "Bonsai-FsGuardEdge-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&root).map_err(|e| e.to_string())?;
        let mut root_guard = TempDirGuard::new(root.clone());

        // 1. is_windows_sharing_or_access raw codes 5, 32, 33 vs 2 (NotFound)
        for code in [5_i32, 32, 33] {
            assert!(is_windows_sharing_or_access(&io::Error::from_raw_os_error(
                code
            )));
        }
        assert!(!is_windows_sharing_or_access(
            &io::Error::from_raw_os_error(2)
        ));

        // 2. remove_file_with_retry returns Ok(()) on NotFound
        remove_file_with_retry(&root.join("missing.tmp")).map_err(|e| e.to_string())?;

        // 3. remove_dir_all_with_retry refuses a Windows directory junction and preserves target
        let target = root.join("target");
        let sentinel = target.join("sentinel.txt");
        let junction = root.join("junction");
        fs::create_dir(&target).map_err(|e| e.to_string())?;
        fs::write(&sentinel, b"keep").map_err(|e| e.to_string())?;
        let mklink = std::process::Command::new("cmd")
            .args([
                "/c",
                "mklink",
                "/J",
                &junction.to_string_lossy(),
                &target.to_string_lossy(),
            ])
            .output()
            .map_err(|e| e.to_string())?;
        assert!(mklink.status.success(), "mklink /J failed");
        let j_err = remove_dir_all_with_retry(&junction)
            .err()
            .ok_or("junction removal unexpectedly succeeded")?;
        assert!(j_err.to_string().contains("redirected directory"));
        assert!(sentinel.exists());
        fs::remove_dir(&junction).map_err(|e| e.to_string())?;

        // 4. rename_with_retry clears DOS read-only on `to` and succeeds
        let from_file = root.join("from.json");
        let to_file = root.join("to.json");
        fs::write(&from_file, b"updated").map_err(|e| e.to_string())?;
        fs::write(&to_file, b"original").map_err(|e| e.to_string())?;
        let mut perms = fs::metadata(&to_file)
            .map_err(|e| e.to_string())?
            .permissions();
        perms.set_readonly(true);
        fs::set_permissions(&to_file, perms).map_err(|e| e.to_string())?;
        rename_with_retry(&from_file, &to_file).map_err(|e| e.to_string())?;
        assert_eq!(fs::read(&to_file).map_err(|e| e.to_string())?, b"updated");

        // 5. write_atomic_bytes & write_new_atomic_bytes error paths leave 0 .partial files
        assert_eq!(
            write_atomic_bytes(Path::new("C:\\"), b"x")
                .err()
                .map(|e| e.kind()),
            Some(io::ErrorKind::InvalidInput)
        );
        assert_eq!(
            write_new_atomic_bytes(Path::new("C:\\"), b"x")
                .err()
                .map(|e| e.kind()),
            Some(io::ErrorKind::InvalidInput)
        );

        let occupied_dir = root.join("occupied-dir");
        fs::create_dir(&occupied_dir).map_err(|e| e.to_string())?;
        assert!(write_atomic_bytes(&occupied_dir, b"fail").is_err());

        assert!(write_new_atomic_bytes(&to_file, b"collide").is_err());
        assert_eq!(fs::read(&to_file).map_err(|e| e.to_string())?, b"updated");

        let partials = fs::read_dir(&root)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "partial"))
            .count();
        assert_eq!(partials, 0);

        root_guard.cleanup().map_err(|e| e.to_string())?;
        Ok(())
    }
}
