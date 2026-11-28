//! Runtime extraction shares the verifier's archive policy.

use super::{
    ArtifactError, HASH_BUFFER_BYTES, Installation, RuntimeSpec, VerifiedRuntimeFile, check_locked,
    copy_and_hash_bounded, enumerate_installation, hold_ancestors, inspect_archive,
    open_locked_file, validate_external_path, verify_enumerated_tree, verify_pe_if_binary,
};
use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek};
use std::path::Path;
use zip::ZipArchive;

/// A staging failure, including any failure to remove its own incomplete files.
#[derive(Debug)]
pub struct RuntimeStagingError {
    error: ArtifactError,
    cleanup: Option<ArtifactError>,
}

impl RuntimeStagingError {
    /// Original validation, extraction or verification failure.
    #[must_use]
    pub const fn error(&self) -> ArtifactError {
        self.error
    }

    /// Failure to clean this attempt's incomplete staging, when cleanup was needed.
    #[must_use]
    pub const fn cleanup_error(&self) -> Option<ArtifactError> {
        self.cleanup
    }
}

impl From<ArtifactError> for RuntimeStagingError {
    fn from(error: ArtifactError) -> Self {
        Self {
            error,
            cleanup: None,
        }
    }
}

impl fmt::Display for RuntimeStagingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, formatter)?;
        if let Some(cleanup) = self.cleanup {
            write!(formatter, " Incomplete staging cleanup failed: {cleanup}")?;
        }
        Ok(())
    }
}

impl std::error::Error for RuntimeStagingError {}

/// Stage a locked ARM64 runtime ZIP in a fresh directory without executing it.
///
/// The caller supplies a fresh staging sibling, retained on failure. The archive
/// remains guarded through extraction and verification. Staging file guards are
/// released when this operation returns. The caller must publish without replacing
/// an existing destination, then verify the published runtime before executing it.
/// This operation neither publishes nor starts a native process.
///
/// # Errors
/// Rejects unsupported platforms, unsafe paths or archive entries, identity or
/// architecture drift, failed writes, and existing staging or destination paths.
pub fn stage_runtime(
    spec: &RuntimeSpec,
    staging: &Path,
) -> Result<Vec<VerifiedRuntimeFile>, ArtifactError> {
    stage_runtime_inner(spec, staging, false).map_err(|error| error.error)
}

/// Stage a locked runtime and remove this attempt's incomplete files on failure.
///
/// Cleanup runs after extraction and verification handles are released. It uses
/// the same admitted ZIP inventory, rejects redirects and unknown entries, and
/// never sweeps a preexisting path or another attempt. The pinned archive remains
/// available to reproduce the error. Successful staging is retained for publication.
///
/// # Errors
/// Returns the original staging error and any failed cleanup. Preflight failures
/// create no staging directory and never authorize deleting an existing path.
pub fn stage_runtime_cleaned(
    spec: &RuntimeSpec,
    staging: &Path,
) -> Result<Vec<VerifiedRuntimeFile>, RuntimeStagingError> {
    stage_runtime_inner(spec, staging, true)
}

fn stage_runtime_inner(
    spec: &RuntimeSpec,
    staging: &Path,
    cleanup: bool,
) -> Result<Vec<VerifiedRuntimeFile>, RuntimeStagingError> {
    if !cfg!(windows) {
        return Err(ArtifactError::Configuration.into());
    }
    validate_external_path(&spec.runtime_directory)?;
    validate_external_path(staging)?;
    if staging == spec.runtime_directory || staging.parent() != spec.runtime_directory.parent() {
        return Err(ArtifactError::Path.into());
    }
    absent(staging)?;
    absent(&spec.runtime_directory)?;

    let mut archive_handles = Vec::new();
    let mut archive_held = BTreeSet::new();
    hold_ancestors(
        &spec.archive_path,
        &mut archive_handles,
        &mut archive_held,
        &mut || Ok(()),
    )?;
    let mut archive = open_locked_file(&spec.archive_path)?;
    let mut buffer = vec![0; HASH_BUFFER_BYTES];
    check_locked(
        &mut archive,
        spec.archive_bytes,
        &spec.archive_sha256,
        &mut buffer,
        &mut || Ok(()),
    )?;
    let mut zip = ZipArchive::new(archive).map_err(|_| ArtifactError::Archive)?;
    // Complete policy inspection precedes creation of even the staging directory.
    let installation = inspect_archive(&mut zip, staging, &mut || Ok(()))?;
    let parent = staging.parent().ok_or(ArtifactError::Path)?;
    fs::create_dir_all(parent).map_err(|_| ArtifactError::Path)?;
    hold_ancestors(
        &spec.runtime_directory,
        &mut archive_handles,
        &mut archive_held,
        &mut || Ok(()),
    )?;
    fs::create_dir(staging).map_err(|_| ArtifactError::Path)?;
    let inventory = cleanup.then(|| {
        (
            installation.files.keys().cloned().collect::<BTreeSet<_>>(),
            installation.directories.clone(),
        )
    });
    let result = (|| {
        let (files, _staging_handles) = extract(&mut zip, staging, &installation, &mut buffer)?;
        verify_enumerated_tree(
            staging,
            &installation.files,
            &installation.directories,
            &mut || Ok(()),
        )?;
        Ok(files)
    })();
    result.map_err(|error| RuntimeStagingError {
        error,
        cleanup: inventory
            .and_then(|(files, directories)| cleanup_staging(staging, &files, &directories).err()),
    })
}

fn remove_dir_all_with_retry(path: &Path) -> std::io::Result<()> {
    const MAX_ATTEMPTS: usize = 6;
    const BASE_DELAY_MS: u64 = 5;
    let mut attempt = 0_usize;
    loop {
        match fs::remove_dir_all(path) {
            Ok(()) => return Ok(()),
            Err(error)
                if cfg!(windows)
                    && attempt + 1 < MAX_ATTEMPTS
                    && matches!(error.raw_os_error(), Some(5 | 32 | 33 | 145)) =>
            {
                let shift = u32::try_from(attempt).unwrap_or(u32::MAX);
                let delay_ms =
                    BASE_DELAY_MS.saturating_mul(1_u64.checked_shl(shift).unwrap_or(u64::MAX));
                std::thread::sleep(std::time::Duration::from_millis(delay_ms.min(160)));
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

fn cleanup_staging(
    staging: &Path,
    files: &BTreeSet<String>,
    directories: &BTreeSet<String>,
) -> Result<(), ArtifactError> {
    let mut installed_files = BTreeSet::new();
    let mut installed_directories = BTreeSet::new();
    enumerate_installation(
        staging,
        staging,
        &mut installed_files,
        &mut installed_directories,
        &mut || Ok(()),
    )?;
    if !installed_files.is_subset(files) || !installed_directories.is_subset(directories) {
        return Err(ArtifactError::Installation);
    }
    remove_dir_all_with_retry(staging).map_err(|_| ArtifactError::Installation)
}

fn absent(path: &Path) -> Result<(), ArtifactError> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        _ => Err(ArtifactError::Path),
    }
}

fn extract<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    staging: &Path,
    installation: &Installation,
    buffer: &mut [u8],
) -> Result<(Vec<VerifiedRuntimeFile>, Vec<File>), ArtifactError> {
    for directory in &installation.directories {
        let path = staging.join(directory);
        validate_external_path(&path)?;
        fs::create_dir_all(path).map_err(|_| ArtifactError::Installation)?;
    }
    let mut files = Vec::with_capacity(installation.files.len());
    let mut handles = Vec::with_capacity(installation.files.len());
    for entry in installation.files.values() {
        let path = staging.join(&entry.name);
        validate_external_path(&path)?;
        let mut source = archive
            .by_index(entry.index)
            .map_err(|_| ArtifactError::Archive)?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt as _;
            options.share_mode(0).custom_flags(0x0020_0000);
        }
        let mut output = options
            .open(&path)
            .map_err(|_| ArtifactError::Installation)?;
        let sha256 = copy_and_hash_bounded(
            &mut source,
            &mut output,
            entry.bytes,
            buffer,
            &mut || Ok(()),
        )?;
        verify_pe_if_binary(&path, &mut output)?;
        files.push(VerifiedRuntimeFile {
            path,
            bytes: entry.bytes,
            sha256,
        });
        handles.push(output);
    }
    for handle in &handles {
        handle.sync_all().map_err(|_| ArtifactError::Installation)?;
    }
    Ok((files, handles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn remove_dir_all_with_retry_recovers_from_transient_lock_and_fails_on_not_found()
    -> std::io::Result<()> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(std::io::Error::other)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-InstallRetry-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&root)?;
        let locked_path = root.join("locked.bin");
        fs::write(&locked_path, b"locked")?;

        let missing = root.join("nonexistent-dir");
        let missing_err = remove_dir_all_with_retry(&missing)
            .err()
            .ok_or_else(|| std::io::Error::other("missing directory removal succeeded"))?;
        assert_eq!(missing_err.kind(), std::io::ErrorKind::NotFound);

        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt as _;
            let locked = OpenOptions::new()
                .read(true)
                .share_mode(0)
                .open(&locked_path)?;
            let releaser = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(15));
                drop(locked);
            });
            remove_dir_all_with_retry(&root)?;
            releaser
                .join()
                .map_err(|_| std::io::Error::other("releaser panicked"))?;
            assert!(!root.exists());
        }
        #[cfg(not(windows))]
        {
            fs::remove_dir_all(&root)?;
        }
        Ok(())
    }
}
