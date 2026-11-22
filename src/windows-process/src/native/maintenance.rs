//! Small operating-system observations for native maintenance commands.

use std::io;
use std::path::Path;

use windows_sys::Win32::Foundation::SYSTEMTIME;
use windows_sys::Win32::Storage::FileSystem::{GetDiskFreeSpaceExW, GetDriveTypeW};
use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows_sys::Win32::System::SystemInformation::GetSystemTime;

pub(crate) fn monotonic_ticks() -> io::Result<(u64, u64)> {
    let mut ticks = 0;
    let mut frequency = 0;
    // SAFETY: Both calls receive live writable i64 outputs and retain no pointers.
    super::check(unsafe { QueryPerformanceCounter(&raw mut ticks) })?;
    // SAFETY: The frequency output remains writable for the complete synchronous call.
    super::check(unsafe { QueryPerformanceFrequency(&raw mut frequency) })?;
    let ticks = u64::try_from(ticks).map_err(|_| super::invalid("Negative monotonic counter."))?;
    let frequency = u64::try_from(frequency)
        .ok()
        .filter(|value| *value != 0)
        .ok_or_else(|| super::invalid("Invalid monotonic frequency."))?;
    Ok((ticks, frequency))
}

pub(crate) fn has_running_process(names: &[&str]) -> io::Result<bool> {
    if names.len() > 64
        || names
            .iter()
            .any(|name| name.is_empty() || !name.is_ascii() || name.len() > 260)
    {
        return Err(super::invalid(
            "Process-name selection exceeds its fixed bound.",
        ));
    }
    let mut matched = false;
    super::for_each_process("Process inventory exceeds its fixed bound.", |entry| {
        let exe_units = super::utf16_z(&entry.szExeFile, "Unterminated process name.")?;
        if names
            .iter()
            .any(|selected| super::utf16_eq_ignore_ascii_case(exe_units, selected))
        {
            matched = true;
            return Ok(false);
        }
        Ok(true)
    })?;
    Ok(matched)
}

pub(crate) fn current_creation_time() -> io::Result<u64> {
    super::current_creation_time()
}

pub(crate) fn utc_now() -> String {
    let mut time = SYSTEMTIME::default();
    // SAFETY: The initialized structure is writable for the complete call.
    unsafe { GetSystemTime(&raw mut time) };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}0000Z",
        time.wYear,
        time.wMonth,
        time.wDay,
        time.wHour,
        time.wMinute,
        time.wSecond,
        time.wMilliseconds
    )
}

pub(crate) fn local_drive(path: &Path) -> io::Result<()> {
    let text = path
        .to_str()
        .ok_or_else(|| super::invalid("Use a Unicode local drive path."))?;
    let bytes = text.as_bytes();
    if bytes.len() <= 3
        || path.parent().is_none()
        || !bytes[0].is_ascii_alphabetic()
        || bytes[1] != b':'
        || !matches!(bytes[2], b'/' | b'\\')
    {
        return Err(super::invalid(
            "Use a dedicated absolute local drive directory.",
        ));
    }
    let root = [u16::from(bytes[0]), u16::from(b':'), u16::from(b'\\'), 0];
    // SAFETY: root is a terminated three-character drive-root string.
    let kind = unsafe { GetDriveTypeW(root.as_ptr()) };
    if !matches!(kind, 2 | 3 | 6) {
        return Err(super::invalid(
            "The directory must be on a fixed, removable or RAM disk.",
        ));
    }
    Ok(())
}

pub(crate) fn available_disk_bytes(path: &Path) -> io::Result<u64> {
    let path = super::wide(path.as_os_str());
    if path[..path.len() - 1].contains(&0) {
        return Err(super::invalid("Disk query path contains a NUL."));
    }
    let mut available = 0;
    // SAFETY: path is terminated and available points to a writable u64; optional outputs are null.
    if unsafe {
        GetDiskFreeSpaceExW(
            path.as_ptr(),
            &raw mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(available)
}
