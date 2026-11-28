//! Publish complete settings or export bytes through a synced, create-new sibling file.

use std::{
    fs::{self, File, Metadata, OpenOptions},
    io::{self, Write as _},
    os::windows::fs::{MetadataExt as _, OpenOptionsExt as _},
    path::{Component, Path, PathBuf, Prefix},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
const MAX_CREATE_ATTEMPTS: usize = 4;
const TRANSIENT_BACKOFF_MS: [u64; 3] = [10, 25, 50];

static PENDING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Ordered phases reported when callers surface stage-specific persistence failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Initial validation of the target file path.
    ValidatePath,
    /// Path generation or validation for the temporary sibling file.
    ValidateTemporary,
    /// Opening the exclusive create-new sibling file.
    CreateTemporary,
    /// Writing bytes and flushing the sibling file to disk.
    WriteTemporary,
    /// Re-validating the target file path right before replacement.
    RevalidatePath,
    /// Atomically renaming the temporary file over the target path.
    Replace,
}

/// Atomic file replacement failure tagged with its failing phase.
pub(crate) struct StagedError {
    /// Phase in which the failure occurred.
    pub(crate) stage: Stage,
    /// Underlying I/O or validation error.
    pub(crate) error: io::Error,
}

fn pending_path(path: &Path) -> io::Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_nanos();
    let sequence = PENDING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    // Append the unique suffix rather than replacing multi-dot target extensions.
    Ok(path.with_added_extension(format!("{}.{stamp}.{sequence}.pending", std::process::id())))
}

fn is_transient_windows_file_error(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(5 | 32 | 33))
}

fn retry_transient_io(mut op: impl FnMut() -> io::Result<()>) -> io::Result<()> {
    for delay_ms in TRANSIENT_BACKOFF_MS {
        match op() {
            Ok(()) => return Ok(()),
            Err(error) if is_transient_windows_file_error(&error) => {
                thread::sleep(Duration::from_millis(delay_ms));
            }
            Err(error) => return Err(error),
        }
    }
    op()
}

fn open_exclusive_temp(
    path: &Path,
    validate_temp: fn(&Path) -> io::Result<()>,
    mut make_path: impl FnMut(&Path) -> io::Result<PathBuf>,
) -> Result<(PathBuf, File), StagedError> {
    for attempt in 0..MAX_CREATE_ATTEMPTS {
        let temporary = make_path(path).map_err(|error| StagedError {
            stage: Stage::ValidateTemporary,
            error,
        })?;
        validate_stage(Stage::ValidateTemporary, validate_temp, &temporary)?;
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .share_mode(0)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(&temporary)
        {
            Ok(file) => return Ok((temporary, file)),
            Err(error)
                if error.kind() == io::ErrorKind::AlreadyExists
                    && attempt + 1 < MAX_CREATE_ATTEMPTS => {}
            Err(error) => {
                return Err(StagedError {
                    stage: Stage::CreateTemporary,
                    error,
                });
            }
        }
    }
    Err(StagedError {
        stage: Stage::CreateTemporary,
        error: io::Error::new(
            io::ErrorKind::AlreadyExists,
            "Temporary staging path remained occupied.",
        ),
    })
}

const fn is_allowed_leaf_attributes(is_file: bool, attributes: u32) -> bool {
    is_file && attributes & FILE_ATTRIBUTE_REPARSE_POINT == 0
}

fn ensure_regular_non_reparse(metadata: &Metadata) -> io::Result<()> {
    if !is_allowed_leaf_attributes(metadata.is_file(), metadata.file_attributes()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Target must be an ordinary non-reparse file.",
        ));
    }
    Ok(())
}

fn validate_export_leaf(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => ensure_regular_non_reparse(&metadata),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn is_reserved_dos_name(component: &str) -> bool {
    let stem = component.split('.').next().unwrap_or(component);
    if ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"]
        .iter()
        .any(|reserved| stem.eq_ignore_ascii_case(reserved))
    {
        return true;
    }
    stem.len() == 4
        && (stem[..3].eq_ignore_ascii_case("COM") || stem[..3].eq_ignore_ascii_case("LPT"))
        && matches!(stem.as_bytes()[3], b'1'..=b'9')
}

/// Validates a user-chosen Save Dialog export path while allowing OneDrive cloud-reparse ancestors,
/// `.git` directories, and `%` or multiple dots in filenames, yet still rejecting UNC/device paths
/// and leaf symlinks/reparse points.
pub(crate) fn validate_user_export_path(path: &Path) -> io::Result<()> {
    let raw = path.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Export path must be valid UTF-8.",
        )
    })?;
    if raw.is_empty() || raw.ends_with(['\\', '/']) || raw.chars().any(char::is_control) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Export path must name a file.",
        ));
    }
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(Component::Prefix(prefix)), Some(Component::RootDir))
            if matches!(prefix.kind(), Prefix::Disk(_)) => {}
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Export paths must target an ordinary local drive.",
            ));
        }
    }
    let mut normal_count = 0usize;
    for component in components {
        match component {
            Component::Normal(part) => {
                normal_count += 1;
                let text = part.to_str().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Export path must be valid UTF-8.",
                    )
                })?;
                if text.is_empty()
                    || text.contains([':', '/', '\\'])
                    || text.ends_with([' ', '.'])
                    || is_reserved_dos_name(text)
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Export path component is invalid.",
                    ));
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Export path contains unsupported navigation components.",
                ));
            }
        }
    }
    if normal_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Export paths must name a file.",
        ));
    }
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Export path requires a parent directory.",
        )
    })?;
    // Do not reject ancestor reparse attributes (`IO_REPARSE_TAG_CLOUD` on Windows 11 OneDrive).
    if !fs::metadata(parent)?.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Export parent must be an existing directory.",
        ));
    }
    validate_export_leaf(path)
}

fn validate_stage(
    stage: Stage,
    validator: fn(&Path) -> io::Result<()>,
    path: &Path,
) -> Result<(), StagedError> {
    validator(path).map_err(|error| StagedError { stage, error })
}

fn replace_with(
    path: &Path,
    bytes: &[u8],
    validate_target: fn(&Path) -> io::Result<()>,
    validate_temp: fn(&Path) -> io::Result<()>,
    revalidate_target: fn(&Path) -> io::Result<()>,
) -> Result<(), StagedError> {
    validate_stage(Stage::ValidatePath, validate_target, path)?;
    let (temporary, mut file) = open_exclusive_temp(path, validate_temp, pending_path)?;
    let result = (|| {
        file.metadata()
            .and_then(|meta| ensure_regular_non_reparse(&meta))
            .map_err(|error| StagedError {
                stage: Stage::CreateTemporary,
                error,
            })?;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| StagedError {
                stage: Stage::WriteTemporary,
                error,
            })?;
        drop(file);
        validate_stage(Stage::RevalidatePath, revalidate_target, path)?;
        retry_transient_io(|| fs::rename(&temporary, path)).map_err(|error| StagedError {
            stage: Stage::Replace,
            error,
        })
    })();
    if result.is_err() {
        let _removed = retry_transient_io(|| fs::remove_file(&temporary));
    }
    result
}

/// Validates strict external path containment and atomically replaces `path`, reporting stage.
// Serialization, encryption and storage limits belong to the individual stores.
pub(crate) fn replace_staged(path: &Path, bytes: &[u8]) -> Result<(), StagedError> {
    replace_with(
        path,
        bytes,
        bonsai_windows_credentials::validate_external_path,
        bonsai_windows_credentials::validate_external_path,
        bonsai_windows_credentials::validate_external_path,
    )
}

/// Validates strict external path containment and atomically writes `bytes` to `path`.
pub(crate) fn replace(path: &Path, bytes: &[u8]) -> io::Result<()> {
    replace_staged(path, bytes).map_err(|failure| failure.error)
}

/// Atomically writes user-chosen Save Dialog exports without triggering strict app-private path
/// rejections on OneDrive cloud-reparse ancestors, `.git` paths, or `%` filenames.
pub(crate) fn replace_user_export_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    replace_with(
        path,
        bytes,
        validate_user_export_path,
        validate_export_leaf,
        validate_export_leaf,
    )
    .map_err(|failure| failure.error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_path_preserves_multi_dot_extensions() -> io::Result<()> {
        let path = Path::new(r"C:\exports\diagram.v2.png");
        let pending = pending_path(path)?;
        let name = pending
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| io::Error::other("missing pending filename"))?;
        assert!(name.starts_with("diagram.v2.png."));
        assert!(name.ends_with(".pending"));
        Ok(())
    }

    #[test]
    fn export_path_validation_allows_git_percent_and_dots_but_rejects_unc_devices_and_leaf_reparse()
    -> io::Result<()> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "BonsaiExportValidate-{}-{stamp}",
            std::process::id()
        ));
        let git_dir = root.join(".git");
        fs::create_dir_all(&git_dir)?;

        let allowed = git_dir.join("budget 100%.v2.png");
        replace_user_export_file(&allowed, b"export bytes")?;
        assert_eq!(fs::read(&allowed)?, b"export bytes");

        for rejected in [
            r"\\server\share\diagram.png",
            r"\\?\UNC\server\share\diagram.png",
            r"\\?\C:\exports\diagram.png",
            r"\\.\COM1",
            r"relative\diagram.png",
            r"C:\",
            &root.join("..").join("diagram.png").to_string_lossy(),
            &root.join("diagram.png:stream").to_string_lossy(),
            &root.join("nul.png").to_string_lossy(),
            &root.join("COM1.svg").to_string_lossy(),
            &root.join("diagram.png.").to_string_lossy(),
        ] {
            assert!(
                validate_user_export_path(Path::new(rejected)).is_err(),
                "expected rejection for {rejected}"
            );
        }

        // Reject existing directory leaves and reparse-tagged leaves cleanly.
        assert!(replace_user_export_file(&git_dir, b"dir leaf").is_err());
        assert!(is_allowed_leaf_attributes(true, 0));
        assert!(!is_allowed_leaf_attributes(false, 0));
        assert!(!is_allowed_leaf_attributes(
            true,
            FILE_ATTRIBUTE_REPARSE_POINT
        ));

        fs::remove_dir_all(root)
    }

    #[test]
    fn settings_replace_preserves_previous_bytes_and_cleans_up_after_failed_publication()
    -> io::Result<()> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "BonsaiAtomicSettings-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&directory)?;
        let path = directory.join("settings.json");
        replace(&path, b"original")?;
        replace(&path, b"complete replacement")?;
        assert_eq!(fs::read(&path)?, b"complete replacement");
        // Deny delete sharing so publication fails after the sibling is written.
        let retained = OpenOptions::new().read(true).share_mode(1).open(&path)?;
        assert!(replace(&path, b"must not replace the retained file").is_err());
        assert_eq!(fs::read(&path)?, b"complete replacement");
        assert_eq!(fs::read_dir(&directory)?.count(), 1);
        drop(retained);
        fs::remove_file(path)?;
        fs::remove_dir(directory)
    }

    #[test]
    fn pending_path_sequence_and_transient_retry_recover_collisions_and_os_locks() -> io::Result<()>
    {
        let target = Path::new(r"C:\exports\config.json");
        assert_ne!(pending_path(target)?, pending_path(target)?);

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("BonsaiAtomicRetry-{}-{stamp}", std::process::id()));
        fs::create_dir(&dir)?;
        let existing = dir.join("collision.pending");
        fs::write(&existing, b"occupied")?;
        let next = dir.join("fresh.pending");
        let mut calls = 0_usize;
        let (created_path, file) = open_exclusive_temp(
            &dir.join("target.json"),
            |_| Ok(()),
            |_| {
                calls += 1;
                Ok(if calls == 1 {
                    existing.clone()
                } else {
                    next.clone()
                })
            },
        )
        .map_err(|err| err.error)?;
        drop(file);
        assert_eq!(calls, 2);
        assert_eq!(created_path, next);
        assert_eq!(fs::read(&existing)?, b"occupied");

        let mut os_attempts = 0_usize;
        retry_transient_io(|| {
            os_attempts += 1;
            if os_attempts < 3 {
                Err(io::Error::from_raw_os_error(32))
            } else {
                Ok(())
            }
        })?;
        assert_eq!(os_attempts, 3);

        let mut fatal_attempts = 0_usize;
        assert!(
            retry_transient_io(|| {
                fatal_attempts += 1;
                Err(io::Error::new(io::ErrorKind::NotFound, "missing"))
            })
            .is_err()
        );
        assert_eq!(fatal_attempts, 1);

        // is_transient_windows_file_error recognizes 5 and 33 as well as 32.
        for code in [5_i32, 32, 33] {
            assert!(is_transient_windows_file_error(
                &io::Error::from_raw_os_error(code)
            ));
        }
        assert!(!is_transient_windows_file_error(
            &io::Error::from_raw_os_error(2)
        ));

        // retry_transient_io exhausts after 4 attempts (3 backoffs + 1 final attempt).
        let mut exhausted_attempts = 0_usize;
        let exhausted = retry_transient_io(|| {
            exhausted_attempts += 1;
            Err(io::Error::from_raw_os_error(5))
        });
        assert_eq!(
            exhausted.as_ref().err().and_then(io::Error::raw_os_error),
            Some(5)
        );
        assert_eq!(exhausted_attempts, TRANSIENT_BACKOFF_MS.len() + 1);

        // open_exclusive_temp stages: make_path error, validate_temp error, non-AlreadyExists error,
        // and 4-collision AlreadyExists exhaustion.
        let make_err = open_exclusive_temp(
            &dir.join("target.json"),
            |_| Ok(()),
            |_| Err(io::Error::new(io::ErrorKind::InvalidInput, "bad name")),
        )
        .err()
        .ok_or_else(|| io::Error::other("expected make_path error"))?;
        assert_eq!(make_err.stage, Stage::ValidateTemporary);

        let val_err = open_exclusive_temp(
            &dir.join("target.json"),
            |_| Err(io::Error::new(io::ErrorKind::InvalidInput, "rejected temp")),
            pending_path,
        )
        .err()
        .ok_or_else(|| io::Error::other("expected validate_temp error"))?;
        assert_eq!(val_err.stage, Stage::ValidateTemporary);

        let missing_dir = dir.join("no-such-parent").join("target.json");
        let mut non_exist_calls = 0_usize;
        let create_err = open_exclusive_temp(
            &missing_dir,
            |_| Ok(()),
            |p| {
                non_exist_calls += 1;
                pending_path(p)
            },
        )
        .err()
        .ok_or_else(|| io::Error::other("expected CreateTemporary error"))?;
        assert_eq!(create_err.stage, Stage::CreateTemporary);
        assert_eq!(non_exist_calls, 1);

        let mut col_calls = 0_usize;
        let exhausted_col = open_exclusive_temp(
            &dir.join("target.json"),
            |_| Ok(()),
            |_| {
                col_calls += 1;
                Ok(existing.clone())
            },
        )
        .err()
        .ok_or_else(|| io::Error::other("expected collision exhaustion"))?;
        assert_eq!(exhausted_col.stage, Stage::CreateTemporary);
        assert_eq!(exhausted_col.error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(col_calls, MAX_CREATE_ATTEMPTS);

        // replace_with failure during Replace removes its temporary staging file cleanly.
        let occupied_dir_target = dir.join("dir-target");
        fs::create_dir(&occupied_dir_target)?;
        let rep_err = replace_with(
            &occupied_dir_target,
            b"payload",
            |_| Ok(()),
            |_| Ok(()),
            |_| Ok(()),
        )
        .err()
        .ok_or_else(|| io::Error::other("expected Replace error on directory target"))?;
        assert_eq!(rep_err.stage, Stage::Replace);
        let leftovers: Vec<_> = fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains("dir-target."))
            .collect();
        assert!(leftovers.is_empty());

        fs::remove_dir_all(dir)
    }
}
