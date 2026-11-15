//! Bounded local UTF-8 text and source-file attachment reader.
#![forbid(unsafe_code)]

use std::{
    fs::{File, OpenOptions},
    io::Read as _,
    os::windows::fs::{MetadataExt as _, OpenOptionsExt as _},
    path::{Component, Path, Prefix},
};

use super::{Attachment, MAX_TEXT_FILE_BYTES, display_name};

const SOURCE_FAILURE: &str = "Choose a nonempty regular UTF-8 text file up to 256 KiB on a local drive. Your draft was kept.";

struct OpenTextSelection {
    _ancestors: Vec<File>,
    file: File,
}

impl OpenTextSelection {
    fn open(path: &Path) -> Result<Self, String> {
        if !path.is_absolute()
            || !matches!(
                path.components().next(),
                Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_))
            )
            || path
                .components()
                .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
        {
            return Err(SOURCE_FAILURE.to_owned());
        }
        bonsai_windows_process::require_local_data_drive(path)
            .map_err(|_| SOURCE_FAILURE.to_owned())?;
        let mut ancestors = Vec::new();
        for ancestor in path
            .ancestors()
            .skip(1)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            let directory = OpenOptions::new()
                .read(true)
                // FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES prevents ancestor rename/rebind.
                .access_mode(0x81)
                .share_mode(3)
                .custom_flags(0x0220_0000)
                .open(ancestor)
                .map_err(|_| SOURCE_FAILURE.to_owned())?;
            let metadata = directory
                .metadata()
                .map_err(|_| SOURCE_FAILURE.to_owned())?;
            if !metadata.is_dir() || metadata.file_attributes() & 0x400 != 0 {
                return Err(SOURCE_FAILURE.to_owned());
            }
            ancestors.push(directory);
        }
        let file = OpenOptions::new()
            .read(true)
            .share_mode(1)
            .custom_flags(0x0020_0000)
            .open(path)
            .map_err(|_| SOURCE_FAILURE.to_owned())?;
        let metadata = file.metadata().map_err(|_| SOURCE_FAILURE.to_owned())?;
        if !metadata.is_file() || metadata.file_attributes() & 0x400 != 0 {
            return Err(SOURCE_FAILURE.to_owned());
        }
        let len = usize::try_from(metadata.len()).map_err(|_| SOURCE_FAILURE.to_owned())?;
        if len == 0 {
            return Err(format!(
                "\"{}\" is empty. Your draft was kept.",
                display_name(path)
            ));
        }
        if len > MAX_TEXT_FILE_BYTES {
            return Err(format!(
                "\"{}\" exceeds the 256 KiB text file limit. Your draft was kept.",
                display_name(path)
            ));
        }
        Ok(Self {
            _ancestors: ancestors,
            file,
        })
    }
}

pub(super) fn import_text_file(path: &Path, remaining_bytes: usize) -> Result<Attachment, String> {
    let selection = OpenTextSelection::open(path)?;
    let max_read = u64::try_from(MAX_TEXT_FILE_BYTES + 1).map_err(|_| SOURCE_FAILURE.to_owned())?;
    let mut raw = Vec::new();
    selection
        .file
        .take(max_read)
        .read_to_end(&mut raw)
        .map_err(|_| SOURCE_FAILURE.to_owned())?;
    if raw.len() > MAX_TEXT_FILE_BYTES {
        return Err(format!(
            "\"{}\" exceeds the 256 KiB text file limit. Your draft was kept.",
            display_name(path)
        ));
    }
    let text = normalize_text_bytes(&raw, remaining_bytes, &display_name(path))?;
    Ok(Attachment::Text {
        name: display_name(path),
        text,
    })
}

pub(super) fn normalize_text_bytes(
    raw: &[u8],
    remaining_bytes: usize,
    name: &str,
) -> Result<String, String> {
    if raw.len() > MAX_TEXT_FILE_BYTES {
        return Err(format!(
            "\"{name}\" exceeds the 256 KiB text file limit. Your draft was kept."
        ));
    }
    let without_bom = raw.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(raw);
    if without_bom.is_empty() {
        return Err(format!("\"{name}\" is empty. Your draft was kept."));
    }
    let utf8 = std::str::from_utf8(without_bom)
        .map_err(|_| format!("\"{name}\" is not valid UTF-8 text. Your draft was kept."))?;
    if utf8.contains('\0') {
        return Err(format!(
            "\"{name}\" contains NUL bytes and cannot be attached as text. Your draft was kept."
        ));
    }
    let normalized = utf8.replace("\r\n", "\n");
    if normalized.len() > remaining_bytes {
        return Err(
            "Attached text files exceed the 512 KiB per-message text budget. Your draft was kept."
                .to_owned(),
        );
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::{MAX_TEXT_FILE_BYTES, normalize_text_bytes};

    #[test]
    fn text_validation_strips_bom_normalizes_crlf_and_rejects_binary_or_oversized() {
        let ok = normalize_text_bytes(b"\xEF\xBB\xBFalpha\r\nbeta\r\n", 4096, "sample.txt");
        assert_eq!(ok.as_deref(), Ok("alpha\nbeta\n"));

        assert!(normalize_text_bytes(b"\xEF\xBB\xBF", 4096, "bom-only.txt").is_err());
        assert!(normalize_text_bytes(b"", 4096, "empty.txt").is_err());
        assert!(normalize_text_bytes(b"\xFF\xFEbad", 4096, "utf16.txt").is_err());
        assert!(normalize_text_bytes(b"ab\0cd", 4096, "bin.dat").is_err());
        assert!(normalize_text_bytes(b"hello world", 5, "tight.txt").is_err());
        let oversized = vec![b'a'; MAX_TEXT_FILE_BYTES + 1];
        assert!(normalize_text_bytes(&oversized, MAX_TEXT_FILE_BYTES * 2, "big.txt").is_err());
    }
}
