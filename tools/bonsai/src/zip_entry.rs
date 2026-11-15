//! Shared ZIP archive entry validation before inspection or extraction.

use std::io::Read;

use zip::CompressionMethod;
use zip::read::ZipFile;

pub(crate) struct ZipEntryPolicy<'a> {
    pub(crate) allow_directories: bool,
    pub(crate) min_size: u64,
    pub(crate) max_size: u64,
    pub(crate) error: &'a str,
}

pub(crate) fn validate_zip_entry<R: Read>(
    entry: &ZipFile<'_, R>,
    policy: &ZipEntryPolicy<'_>,
) -> crate::Result<String> {
    let raw = entry.name().map_err(|error| error.to_string())?;
    let trimmed = if entry.is_dir() {
        raw.strip_suffix('/').unwrap_or(&raw)
    } else {
        raw.as_ref()
    };
    if entry.encrypted()
        || entry.is_symlink()
        || !matches!(
            entry.compression(),
            CompressionMethod::Stored | CompressionMethod::Deflated
        )
        || (entry.compression() == CompressionMethod::Stored
            && entry.compressed_size() != entry.size())
        || trimmed.is_empty()
        || raw.starts_with('/')
        || (!entry.is_dir() && raw.ends_with('/'))
        || trimmed.contains(['\\', ':', '\0'])
        || trimmed
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."))
        || entry.unix_mode().is_some_and(|mode| {
            let kind = mode & 0o170_000;
            kind != 0 && kind != if entry.is_dir() { 0o040_000 } else { 0o100_000 }
        })
    {
        return Err(policy.error.into());
    }
    if entry.is_dir() {
        if !policy.allow_directories || entry.size() != 0 {
            return Err(policy.error.into());
        }
    } else if !entry.is_file() || entry.size() < policy.min_size || entry.size() > policy.max_size {
        return Err(policy.error.into());
    }
    Ok(raw.into_owned())
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write as _};

    use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

    use super::{ZipEntryPolicy, validate_zip_entry};

    fn policy(allow_directories: bool) -> ZipEntryPolicy<'static> {
        ZipEntryPolicy {
            allow_directories,
            min_size: 1,
            max_size: 1024,
            error: "rejected",
        }
    }

    #[test]
    fn validate_zip_entry_accepts_files_and_optional_directories() -> crate::Result<()> {
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .add_directory("pkg/", SimpleFileOptions::default())
            .map_err(|error| error.to_string())?;
        writer
            .start_file(
                "pkg/bin.exe",
                SimpleFileOptions::default().compression_method(CompressionMethod::Deflated),
            )
            .map_err(|error| error.to_string())?;
        writer.write_all(b"MZ").map_err(|error| error.to_string())?;
        let bytes = writer
            .finish()
            .map_err(|error| error.to_string())?
            .into_inner();
        let mut archive = ZipArchive::new(Cursor::new(bytes)).map_err(|error| error.to_string())?;

        let dir_entry = archive.by_index(0).map_err(|error| error.to_string())?;
        assert!(validate_zip_entry(&dir_entry, &policy(false)).is_err());
        assert_eq!(validate_zip_entry(&dir_entry, &policy(true))?, "pkg/");
        drop(dir_entry);

        let file_entry = archive.by_index(1).map_err(|error| error.to_string())?;
        assert_eq!(
            validate_zip_entry(&file_entry, &policy(false))?,
            "pkg/bin.exe"
        );
        Ok(())
    }

    #[test]
    fn validate_zip_entry_rejects_symlinks_traversal_and_unsafe_modes() -> crate::Result<()> {
        for (name, symlink) in [
            ("../escape", false),
            ("a\\b", false),
            ("C:bad", false),
            ("a//b", false),
            ("ok/symlink", true),
        ] {
            let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
            if symlink {
                writer
                    .add_symlink(name, "target", SimpleFileOptions::default())
                    .map_err(|error| error.to_string())?;
            } else {
                writer
                    .start_file(name, SimpleFileOptions::default())
                    .map_err(|error| error.to_string())?;
                writer.write_all(b"x").map_err(|error| error.to_string())?;
            }
            let bytes = writer
                .finish()
                .map_err(|error| error.to_string())?
                .into_inner();
            let mut archive =
                ZipArchive::new(Cursor::new(bytes)).map_err(|error| error.to_string())?;
            let entry = archive.by_index(0).map_err(|error| error.to_string())?;
            assert!(
                validate_zip_entry(&entry, &policy(true)).is_err(),
                "expected rejection for {name}"
            );
        }

        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .start_file(
                "ok/fifo",
                SimpleFileOptions::default().unix_permissions(0o644),
            )
            .map_err(|error| error.to_string())?;
        writer.write_all(b"x").map_err(|error| error.to_string())?;
        let base = writer
            .finish()
            .map_err(|error| error.to_string())?
            .into_inner();
        let cd = base
            .windows(4)
            .position(|window| window == b"PK\x01\x02")
            .ok_or("missing central directory header")?;
        let mut fifo = base.clone();
        fifo[cd + 5] = 3;
        fifo[cd + 38..cd + 42].copy_from_slice(&(0o010_644_u32 << 16).to_le_bytes());
        let mut fifo_archive =
            ZipArchive::new(Cursor::new(fifo)).map_err(|error| error.to_string())?;
        let fifo_entry = fifo_archive
            .by_index(0)
            .map_err(|error| error.to_string())?;
        assert_eq!(fifo_entry.unix_mode(), Some(0o010_644));
        assert!(validate_zip_entry(&fifo_entry, &policy(true)).is_err());

        let mut encrypted = base;
        encrypted[cd + 8] |= 1;
        let mut encrypted_archive =
            ZipArchive::new(Cursor::new(encrypted)).map_err(|error| error.to_string())?;
        if let Ok(encrypted_entry) = encrypted_archive.by_index(0) {
            assert!(validate_zip_entry(&encrypted_entry, &policy(true)).is_err());
        }
        Ok(())
    }
}
