//! Byte-limited reads; callers retain file admission, decoding and handle ownership.
#![expect(
    clippy::redundant_pub_crate,
    reason = "Private maintenance modules expose only a crate-local implementation API."
)]

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write as _};
use std::path::{Component, Path, PathBuf};

#[derive(Debug)]
pub(super) enum Error {
    Bound,
    Io(io::Error),
}

impl Error {
    pub(super) fn into_message(self, exceeded: &str) -> String {
        match self {
            Self::Bound => exceeded.into(),
            Self::Io(error) => error.to_string(),
        }
    }
}

pub(super) fn read(reader: impl Read, limit: u64) -> Result<Vec<u8>, Error> {
    // The extra byte detects growth after the caller's metadata admission. Reject
    // an unrepresentable probe instead of wrapping or accepting a prefix.
    let probe = limit.checked_add(1).ok_or(Error::Bound)?;
    let mut bytes = Vec::new();
    reader
        .take(probe)
        .read_to_end(&mut bytes)
        .map_err(Error::Io)?;
    if u64::try_from(bytes.len()).map_err(|_| Error::Bound)? > limit {
        return Err(Error::Bound);
    }
    Ok(bytes)
}

pub(super) fn strip_utf8_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes)
}

pub(super) fn read_json_bytes<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> crate::Result<T> {
    serde_json::from_slice(strip_utf8_bom(bytes)).map_err(|error| error.to_string())
}

pub(super) fn validate_plain_absolute(path: &Path) -> crate::Result<()> {
    if !path.is_absolute()
        || path
            .to_str()
            .is_none_or(|value| value.contains(['%', '\0']))
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
    {
        return Err("Input paths must use expanded absolute paths.".into());
    }
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(metadata) if bonsai_artifacts::is_redirect(&metadata) => {
                return Err("Input paths must not traverse redirected paths.".into());
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(())
}

fn open_regular_file_with(path: &Path, sequential: bool) -> crate::Result<File> {
    let leaf = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !leaf.is_file() || bonsai_artifacts::is_redirect(&leaf) {
        return Err("Input path must refer to an ordinary regular file.".into());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        options
            .share_mode(1)
            .custom_flags(0x0020_0000 | if sequential { 0x0800_0000 } else { 0 });
    }
    #[cfg(not(windows))]
    let _ = sequential;
    let file = options.open(path).map_err(|error| error.to_string())?;
    // Verify the opened handle after disabling reparse-point traversal.
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file() || bonsai_artifacts::is_redirect(&metadata) {
        return Err("Input path must refer to an ordinary regular file.".into());
    }
    Ok(file)
}

pub(super) fn open_regular_file(path: &Path) -> crate::Result<File> {
    open_regular_file_with(path, false)
}

pub(super) fn open_sequential_file(path: &Path) -> crate::Result<File> {
    open_regular_file_with(path, true)
}

pub(super) fn open_plain_regular_file(path: &Path) -> crate::Result<File> {
    validate_plain_absolute(path)?;
    open_regular_file(path)
}

pub(super) fn read_open_file(file: File, limit: u64, exceeded: &str) -> crate::Result<Vec<u8>> {
    if file.metadata().map_err(|error| error.to_string())?.len() > limit {
        return Err(exceeded.into());
    }
    read(file, limit).map_err(|error| error.into_message(exceeded))
}

pub(super) fn read_regular_file(path: &Path, limit: u64, exceeded: &str) -> crate::Result<Vec<u8>> {
    read_open_file(open_regular_file(path)?, limit, exceeded)
}

pub(super) fn read_regular_json<T: serde::de::DeserializeOwned>(
    path: &Path,
    limit: u64,
    exceeded: &str,
) -> crate::Result<T> {
    read_json_bytes(&read_regular_file(path, limit, exceeded)?)
}

pub(super) fn hold_directory_chain(
    path: &Path,
    handles: &mut BTreeMap<PathBuf, File>,
    include_leaf: bool,
    share_mode: u32,
) -> crate::Result<()> {
    let skip = usize::from(!include_leaf);
    let ancestors = path.ancestors().skip(skip).collect::<Vec<_>>();
    for ancestor in ancestors.into_iter().rev() {
        if handles.contains_key(ancestor) {
            continue;
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt as _;
            options
                .access_mode(0x81)
                .share_mode(share_mode)
                .custom_flags(0x0220_0000);
        }
        #[cfg(not(windows))]
        let _ = share_mode;
        let directory = options.open(ancestor).map_err(|error| error.to_string())?;
        let metadata = directory.metadata().map_err(|error| error.to_string())?;
        if !metadata.is_dir() || bonsai_artifacts::is_redirect(&metadata) {
            return Err("Directory ancestry must contain ordinary directories.".into());
        }
        handles.insert(ancestor.to_owned(), directory);
    }
    Ok(())
}

pub(super) fn write_new_synced(path: &Path, bytes: &[u8]) -> crate::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        options.share_mode(0).custom_flags(0x0020_0000);
    }
    let mut file = options.open(path).map_err(|error| error.to_string())?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| error.to_string())
}

pub(super) fn pretty_json_bytes(value: &impl serde::Serialize) -> crate::Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub(super) fn write_pretty_json_new(
    path: &Path,
    value: &impl serde::Serialize,
) -> crate::Result<()> {
    write_new_synced(path, &pretty_json_bytes(value)?)
}

pub(super) fn write_pretty_json_atomic(
    path: &Path,
    value: &impl serde::Serialize,
) -> crate::Result<()> {
    write_atomic_synced(path, &pretty_json_bytes(value)?)
}

struct TemporaryFile(PathBuf);

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub(super) fn write_atomic_synced(path: &Path, bytes: &[u8]) -> crate::Result<()> {
    let name = path
        .file_name()
        .ok_or("Output path requires a file name.")?;
    let temporary = TemporaryFile(path.with_file_name(format!(
        "{}.{}.tmp",
        name.to_string_lossy(),
        crate::startup_process::unique()
    )));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        options.share_mode(0).custom_flags(0x0020_0000);
    }
    let mut file = options
        .open(&temporary.0)
        .map_err(|error| error.to_string())?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| error.to_string())?;
    drop(file);
    fs::rename(&temporary.0, path).map_err(|error| error.to_string())
}

#[cfg(test)]
pub(crate) struct ScratchDir(PathBuf);

#[cfg(test)]
impl ScratchDir {
    pub(crate) fn new(prefix: &str) -> crate::Result<Self> {
        let path =
            std::env::temp_dir().join(format!("{prefix}-{}", crate::startup_process::unique()));
        bonsai_artifacts::validate_external_path(&path).map_err(|error| error.to_string())?;
        fs::create_dir(&path).map_err(|error| error.to_string())?;
        Ok(Self(path))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

#[cfg(test)]
impl AsRef<Path> for ScratchDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

#[cfg(test)]
impl std::ops::Deref for ScratchDir {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Ok(parent) = fs::canonicalize(std::env::temp_dir())
            && let Ok(target) = fs::canonicalize(&self.0)
            && target.parent() == Some(parent.as_path())
            && fs::symlink_metadata(&self.0)
                .is_ok_and(|metadata| !bonsai_artifacts::is_redirect(&metadata))
        {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor, Read as _};

    use super::{
        Error, ScratchDir, read, read_json_bytes, read_regular_json, strip_utf8_bom,
        write_atomic_synced,
    };

    #[test]
    fn bounded_input_accepts_exact_and_empty_inputs_and_keeps_the_reader() -> io::Result<()> {
        assert_eq!(read(&b""[..], 0).ok(), Some(Vec::new()));
        let mut reader = Cursor::new(b"abc");
        assert_eq!(read(&mut reader, 3).ok(), Some(b"abc".to_vec()));
        assert_eq!(reader.position(), 3);
        assert_eq!(reader.read(&mut [0; 1])?, 0);
        Ok(())
    }

    #[test]
    fn bounded_input_rejects_growth_after_one_probe_byte() {
        let mut reader = Cursor::new(b"abcde");
        assert!(matches!(read(&mut reader, 2), Err(Error::Bound)));
        assert_eq!(reader.position(), 3);
        assert!(matches!(read(&b"x"[..], 0), Err(Error::Bound)));
    }

    #[test]
    fn bounded_input_rejects_limit_overflow_before_reading() {
        let mut reader = Cursor::new(b"abc");
        assert!(matches!(read(&mut reader, u64::MAX), Err(Error::Bound)));
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn bounded_input_preserves_io_failures_and_caller_bound_messages() {
        struct Failed;
        impl io::Read for Failed {
            fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("read failed"))
            }
        }
        assert!(
            matches!(read(Failed, 3), Err(Error::Io(error)) if error.to_string() == "read failed")
        );
        assert_eq!(Error::Bound.into_message("caller bound"), "caller bound");
        assert_eq!(
            Error::Io(io::Error::other("read failed")).into_message("caller bound"),
            "read failed"
        );
    }

    #[test]
    fn regular_json_write_bom_and_scratch_dir_round_trip() -> crate::Result<()> {
        let scratch = ScratchDir::new("Bonsai-BoundedInput")?;
        let root = scratch.path().to_path_buf();
        assert!(root.is_dir());
        assert_eq!(
            strip_utf8_bom(b"\xef\xbb\xbf{\"ok\":true}"),
            b"{\"ok\":true}"
        );
        assert_eq!(
            read_json_bytes::<serde_json::Value>(b"\xef\xbb\xbf{\"ok\":1}")?,
            serde_json::json!({"ok": 1})
        );
        let json_path = scratch.join("state.json");
        write_atomic_synced(&json_path, b"\xef\xbb\xbf{\"count\":2}")?;
        let parsed: serde_json::Value = read_regular_json(&json_path, 64, "too big")?;
        assert_eq!(parsed, serde_json::json!({"count": 2}));
        assert!(super::open_regular_file(&root).is_err());
        assert!(super::open_sequential_file(&root).is_err());
        drop(scratch);
        assert!(!root.exists());
        Ok(())
    }
}
