//! Publish complete settings or export bytes through a synced, create-new sibling file.

use std::{
    fs::{self, Metadata, OpenOptions},
    io::{self, Write as _},
    os::windows::fs::{MetadataExt as _, OpenOptionsExt as _},
    path::{Component, Path, PathBuf, Prefix},
    time::{SystemTime, UNIX_EPOCH},
};

const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;

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
    // Append the unique suffix rather than replacing multi-dot target extensions.
    Ok(path.with_added_extension(format!("{}.{stamp}.pending", std::process::id())))
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

fn replace_with(
    path: &Path,
    bytes: &[u8],
    validate_target: fn(&Path) -> Result<(), StagedError>,
    validate_temp: fn(&Path) -> Result<(), StagedError>,
    revalidate_target: fn(&Path) -> Result<(), StagedError>,
) -> Result<(), StagedError> {
    validate_target(path)?;
    let temporary = pending_path(path).map_err(|error| StagedError {
        stage: Stage::ValidateTemporary,
        error,
    })?;
    validate_temp(&temporary)?;
    let mut created = false;
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .share_mode(0)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(&temporary)
            .map_err(|error| StagedError {
                stage: Stage::CreateTemporary,
                error,
            })?;
        created = true;
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
        revalidate_target(path)?;
        fs::rename(&temporary, path).map_err(|error| StagedError {
            stage: Stage::Replace,
            error,
        })
    })();
    // A create-new collision belongs to somebody else and must survive cleanup.
    if result.is_err() && created {
        let _removed = fs::remove_file(temporary);
    }
    result
}

fn validate_external_target(path: &Path) -> Result<(), StagedError> {
    bonsai_windows_credentials::validate_external_path(path).map_err(|error| StagedError {
        stage: Stage::ValidatePath,
        error,
    })
}

fn validate_external_temp(path: &Path) -> Result<(), StagedError> {
    bonsai_windows_credentials::validate_external_path(path).map_err(|error| StagedError {
        stage: Stage::ValidateTemporary,
        error,
    })
}

fn revalidate_external_target(path: &Path) -> Result<(), StagedError> {
    bonsai_windows_credentials::validate_external_path(path).map_err(|error| StagedError {
        stage: Stage::RevalidatePath,
        error,
    })
}

/// Validates strict external path containment and atomically replaces `path`, reporting stage.
// Serialization, encryption and storage limits belong to the individual stores.
pub(crate) fn replace_staged(path: &Path, bytes: &[u8]) -> Result<(), StagedError> {
    replace_with(
        path,
        bytes,
        validate_external_target,
        validate_external_temp,
        revalidate_external_target,
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
        |target| {
            validate_user_export_path(target).map_err(|error| StagedError {
                stage: Stage::ValidatePath,
                error,
            })
        },
        |temp| {
            validate_export_leaf(temp).map_err(|error| StagedError {
                stage: Stage::ValidateTemporary,
                error,
            })
        },
        |target| {
            validate_export_leaf(target).map_err(|error| StagedError {
                stage: Stage::RevalidatePath,
                error,
            })
        },
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
}
