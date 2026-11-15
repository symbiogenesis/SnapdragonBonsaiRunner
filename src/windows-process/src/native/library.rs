//! Bounded `System32` native library loading with RAII lifetime ownership.

use std::ffi::{CStr, OsString, c_void};
use std::io;
use std::os::windows::ffi::{OsStrExt as _, OsStringExt as _};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;

use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

pub(super) struct SystemLibrary {
    module: HMODULE,
    path: PathBuf,
}

impl Drop for SystemLibrary {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns the successful LoadLibraryExW module
        // and drops only after resolved function pointers are no longer used.
        let _ = unsafe { FreeLibrary(self.module) };
    }
}

impl SystemLibrary {
    pub(super) fn load_system32(
        dll_name: &str,
        unavailable_message: &'static str,
    ) -> io::Result<Self> {
        const {
            assert!(size_of::<HMODULE>() == size_of::<*mut c_void>());
        }
        let path = system_directory()?.join(dll_name);
        let wide_path: Vec<_> = path.as_os_str().encode_wide().chain([0]).collect();
        // SAFETY: An absolute NUL-terminated system-directory path is selected,
        // with search flags restricting dependencies to the loader's directory and
        // System32.
        let module = unsafe {
            LoadLibraryExW(
                wide_path.as_ptr(),
                null_mut(),
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        };
        if module.is_null() {
            return Err(io::Error::other(unavailable_message));
        }
        Ok(Self { module, path })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn symbol(&self, name: &CStr) -> io::Result<unsafe extern "system" fn() -> isize> {
        // SAFETY: This module is owned and live; `name` is NUL-terminated and
        // any resolved pointer is used only while this wrapper remains live.
        unsafe { GetProcAddress(self.module, name.as_ptr().cast()) }
            .ok_or_else(|| super::invalid("required system library symbol is unavailable"))
    }
}

fn system_directory() -> io::Result<PathBuf> {
    const MAX_DIRECTORY: usize = 32_768;
    let mut stack = [0_u16; 260];
    // SAFETY: The stack UTF-16 buffer has its advertised element capacity and stays live.
    let count = unsafe { GetSystemDirectoryW(stack.as_mut_ptr(), 260) };
    let count = usize::try_from(count).map_err(|_| super::invalid("System directory overflow."))?;
    if count == 0 {
        return Err(io::Error::last_os_error());
    }
    if count < stack.len() {
        return Ok(PathBuf::from(OsString::from_wide(&stack[..count])));
    }
    if count > MAX_DIRECTORY {
        return Err(super::invalid("System directory exceeds its bound."));
    }
    let mut heap = vec![0_u16; count];
    let capacity =
        u32::try_from(heap.len()).map_err(|_| super::invalid("System directory overflow."))?;
    // SAFETY: Heap UTF-16 storage matches the advertised element capacity.
    let actual = unsafe { GetSystemDirectoryW(heap.as_mut_ptr(), capacity) };
    let actual =
        usize::try_from(actual).map_err(|_| super::invalid("System directory overflow."))?;
    if actual == 0 || actual >= heap.len() {
        return Err(io::Error::last_os_error());
    }
    Ok(PathBuf::from(OsString::from_wide(&heap[..actual])))
}
