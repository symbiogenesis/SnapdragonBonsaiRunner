//! Bounded `System32` native library loading with RAII lifetime ownership.

use std::ffi::{CStr, c_void};
use std::io;
use std::path::{Path, PathBuf};
use std::ptr::null_mut;

use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

macro_rules! load_symbol {
    ($library:expr, $name:literal, $kind:ty) => {{
        let pointer = $library.symbol($name)?;
        // SAFETY: Caller supplies the exact documented C ABI function-pointer type
        // for `$name` on this still-owned `SystemLibrary`.
        unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, $kind>(pointer) }
    }};
    ($library:expr, $name:literal, $kind:ty, $map_err:expr) => {{
        let pointer = $library.symbol($name).map_err($map_err)?;
        // SAFETY: Caller supplies the exact documented C ABI function-pointer type
        // for `$name` on this still-owned `SystemLibrary`.
        unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, $kind>(pointer) }
    }};
}
pub(crate) use load_symbol;

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
        let wide_path = super::wide(&path);
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
    super::query_system_dir("System directory overflow.", |buf, cap| {
        // SAFETY: Caller provides writable UTF-16 buffer of `cap` elements.
        unsafe { GetSystemDirectoryW(buf, cap) }
    })
}
