//! Atomic publication uses the verified open file rather than its former name.

use std::{
    fs::File,
    io,
    mem::offset_of,
    os::windows::{ffi::OsStrExt as _, io::AsRawHandle as _},
    path::Path,
    ptr,
};

use windows_sys::Win32::Storage::FileSystem::{
    FILE_RENAME_INFO, FileRenameInfo, SetFileInformationByHandle,
};

pub(crate) fn replace(file: &File, target: &Path) -> io::Result<()> {
    replace_with_retry_sleep(file, target, std::thread::sleep)
}

fn replace_with_retry_sleep(
    file: &File,
    target: &Path,
    sleep: impl FnMut(std::time::Duration),
) -> io::Result<()> {
    let mut name: Vec<_> = target.as_os_str().encode_wide().collect();
    if !target.is_absolute() || name.is_empty() || name.len() > 32_767 || name.contains(&0) {
        return Err(super::invalid(
            "File replacement requires an absolute destination without NUL characters.",
        ));
    }
    let name_bytes = name
        .len()
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| super::invalid("Replacement name length overflow."))?;
    // FILE_RENAME_INFO documents FileName as NUL-terminated even though its
    // separate byte count excludes the terminator. Supply both explicitly:
    // relying on trailing allocation padding can publish a different filename.
    name.push(0);
    let encoded_bytes = name_bytes + size_of::<u16>();
    let name_offset = offset_of!(FILE_RENAME_INFO, FileName);
    let required = name_offset
        .checked_add(encoded_bytes)
        .ok_or_else(|| super::invalid("Replacement buffer size overflow."))?
        .max(size_of::<FILE_RENAME_INFO>());
    // Typed allocation supplies the SDK structure's alignment on every Windows
    // architecture; its trailing filename may extend past the first element.
    let mut information =
        vec![FILE_RENAME_INFO::default(); required.div_ceil(size_of::<FILE_RENAME_INFO>())];
    information[0].Anonymous.ReplaceIfExists = true;
    information[0].RootDirectory = ptr::null_mut();
    information[0].FileNameLength = u32::try_from(name_bytes)
        .map_err(|_| super::invalid("Replacement name exceeds Windows bounds."))?;
    let destination = information
        .as_mut_ptr()
        .cast::<u8>()
        .wrapping_add(name_offset);
    // SAFETY: information owns at least name_offset + encoded_bytes initialized
    // bytes, with the SDK's alignment for FileName. Copying the native UTF-16
    // representation as bytes requires no pointer alignment increase. The
    // separate source allocation cannot overlap this destination.
    unsafe { ptr::copy_nonoverlapping(name.as_ptr().cast::<u8>(), destination, encoded_bytes) };
    let required = u32::try_from(required)
        .map_err(|_| super::invalid("Replacement buffer exceeds Windows bounds."))?;
    crate::retry_transient_win32_with(
        false,
        || {
            // SAFETY: The borrowed File and aligned information buffer remain alive
            // throughout this synchronous call. FileNameLength counts exactly the
            // copied UTF-16 path bytes, excluding the explicit terminator; RootDirectory is null for the
            // absolute path. Windows checks DELETE access and all sharing constraints.
            if unsafe {
                SetFileInformationByHandle(
                    file.as_raw_handle(),
                    FileRenameInfo,
                    information.as_ptr().cast(),
                    required,
                )
            } == 0
            {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        },
        sleep,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, OpenOptions};
    use std::os::windows::fs::OpenOptionsExt as _;
    use std::time::{SystemTime, UNIX_EPOCH};
    use windows_sys::Win32::Storage::FileSystem::{DELETE, FILE_SHARE_READ};

    #[test]
    fn replace_open_file_retries_transient_sharing_conflict() -> io::Result<()> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "bonsai-replace-retry-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&root)?;
        let source_path = root.join("new.bin");
        let dest_path = root.join("target.bin");
        fs::write(&source_path, b"new-bytes")?;
        fs::write(&dest_path, b"old-bytes")?;

        // Open source with DELETE access so FileRenameInfo can replace dest_path.
        let source_file = OpenOptions::new()
            .read(true)
            .write(true)
            .share_mode(FILE_SHARE_READ)
            .access_mode(
                windows_sys::Win32::Storage::FileSystem::FILE_GENERIC_READ
                    | windows_sys::Win32::Storage::FileSystem::FILE_GENERIC_WRITE
                    | DELETE,
            )
            .open(&source_path)?;

        // Hold an unshared lock on dest_path so the first SetFileInformationByHandle hits
        // ERROR_SHARING_VIOLATION, then release on retry backoff.
        let mut lock = Some(
            OpenOptions::new()
                .read(true)
                .share_mode(0)
                .open(&dest_path)?,
        );
        let mut retries = 0_usize;
        replace_with_retry_sleep(&source_file, &dest_path, |_| {
            retries += 1;
            drop(lock.take());
        })?;
        drop(source_file);
        assert_eq!(retries, 1);
        assert_eq!(fs::read(&dest_path)?, b"new-bytes");
        assert!(!source_path.exists());
        fs::remove_dir_all(&root)?;
        Ok(())
    }
}
