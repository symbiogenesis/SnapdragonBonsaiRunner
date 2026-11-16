//! Read-only file mapping with sequential/random virtual memory prefetch.

use std::fs::File;
use std::io;
use std::os::windows::fs::MetadataExt as _;
use std::os::windows::io::{AsRawHandle as _, OwnedHandle};
use std::ptr::{NonNull, null};

use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READONLY,
    PrefetchVirtualMemory, UnmapViewOfFile, WIN32_MEMORY_RANGE_ENTRY,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const MAX_MAPPED_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const REPARSE_ATTRIBUTE: u32 = 0x400;

pub(crate) struct NativeMappedFile {
    ptr: NonNull<u8>,
    len: usize,
    _section: OwnedHandle,
}

// SAFETY: The mapping is created read-only (`PAGE_READONLY` + `FILE_MAP_READ`),
// backed by the retained section handle `_section` for the lifetime of `self`.
unsafe impl Send for NativeMappedFile {}

// SAFETY: Concurrent shared reads (`&[u8]`) across worker threads require no
// synchronization over an immutable read-only section view.
unsafe impl Sync for NativeMappedFile {}

impl Drop for NativeMappedFile {
    fn drop(&mut self) {
        let view = MEMORY_MAPPED_VIEW_ADDRESS {
            Value: self.ptr.as_ptr().cast(),
        };
        // SAFETY: `self.ptr` was returned by `MapViewOfFile` and has not yet
        // been unmapped; dropping closes `_section` afterward.
        let _ = unsafe { UnmapViewOfFile(view) };
    }
}

impl NativeMappedFile {
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` points to a valid read-only mapped view of `self.len`
        // bytes that remains mapped and immutable for the lifetime of `&self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

pub(crate) fn map_readonly_file(file: &File) -> io::Result<NativeMappedFile> {
    let metadata = file.metadata()?;
    if !metadata.is_file() || (metadata.file_attributes() & REPARSE_ATTRIBUTE) != 0 {
        return Err(super::invalid(
            "Read-only file mapping requires an ordinary non-reparse file.",
        ));
    }
    let file_len = metadata.len();
    if file_len == 0 || file_len > MAX_MAPPED_BYTES {
        return Err(super::invalid(
            "Mapped file length is empty or exceeds bound.",
        ));
    }
    let len = usize::try_from(file_len)
        .map_err(|_| super::invalid("Mapped file length exceeds address space."))?;
    // SAFETY: `file` is a valid open handle borrowed for this synchronous call;
    // security attributes and name are null and size zero maps the full file.
    let raw_section =
        unsafe { CreateFileMappingW(file.as_raw_handle(), null(), PAGE_READONLY, 0, 0, null()) };
    let section = super::owned(raw_section)?;
    // SAFETY: `section` is an owned read-only file mapping handle with offset 0
    // and length 0 (entire section).
    let view = unsafe { MapViewOfFile(section.as_raw_handle(), FILE_MAP_READ, 0, 0, 0) };
    let Some(ptr) = NonNull::new(view.Value.cast::<u8>()) else {
        return Err(io::Error::last_os_error());
    };
    let mapped = NativeMappedFile {
        ptr,
        len,
        _section: section,
    };
    let entry = WIN32_MEMORY_RANGE_ENTRY {
        VirtualAddress: mapped.ptr.as_ptr().cast(),
        NumberOfBytes: mapped.len,
    };
    // SAFETY: `GetCurrentProcess` returns the pseudo-handle for this process.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: `mapped` owns the live mapped view covering `[VirtualAddress, +len)`.
    // Prefetch is advisory only; unsupported volumes ignore failure safely.
    let _ = unsafe { PrefetchVirtualMemory(process, 1, &raw const entry, 0) };
    Ok(mapped)
}
