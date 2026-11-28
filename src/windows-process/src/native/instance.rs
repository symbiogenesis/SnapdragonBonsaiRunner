//! Audited Win32 process-lock inspection, file version query, and tree termination.

use std::collections::{BTreeSet, VecDeque};
use std::ffi::c_void;
use std::io;
use std::os::windows::ffi::OsStringExt as _;
use std::os::windows::io::{AsRawHandle as _, OwnedHandle};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{
    ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER, ERROR_MORE_DATA, FILETIME, WAIT_FAILED,
    WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows_sys::Win32::Storage::FileSystem::{FILE_VER_GET_NEUTRAL, VS_FIXEDFILEINFO};
use windows_sys::Win32::System::Threading::{
    OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
    QueryFullProcessImageNameW, TerminateProcess, WaitForSingleObject,
};

use super::library::{SystemLibrary, load_symbol};
use super::{check, creation_file_time, invalid, owned, wide};
use crate::ProcessIdentity;

const CCH_RM_SESSION_KEY: usize = 32;
const MAX_LOCK_HOLDERS: usize = 64;
const MAX_RM_RETRIES: usize = 4;
const MAX_IMAGE_PATH_UNITS: usize = 32_768;
const MAX_VERSION_BYTES: u32 = 4 * 1024 * 1024;
const MAX_TREE_DEPTH: usize = 8;
const MAX_TREE_DESCENDANTS: usize = 64;
const VS_FFI_SIGNATURE_U32: u32 = 0xFEEF_04BD;

const ERROR_RESOURCE_DATA_NOT_FOUND: i32 = 1812;
const ERROR_RESOURCE_TYPE_NOT_FOUND: i32 = 1813;
const ERROR_RESOURCE_NAME_NOT_FOUND: i32 = 1814;
const ERROR_RESOURCE_LANG_NOT_FOUND: i32 = 1815;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RmUniqueProcess {
    dw_process_id: u32,
    process_start_time: FILETIME,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RmProcessInfo {
    process: RmUniqueProcess,
    str_app_name: [u16; 256],
    str_service_short_name: [u16; 64],
    application_type: i32,
    app_status: u32,
    ts_session_id: u32,
    b_restartable: i32,
}

impl Default for RmProcessInfo {
    fn default() -> Self {
        Self {
            process: RmUniqueProcess::default(),
            str_app_name: [0; 256],
            str_service_short_name: [0; 64],
            application_type: 0,
            app_status: 0,
            ts_session_id: 0,
            b_restartable: 0,
        }
    }
}

type RmStartSessionFn = unsafe extern "system" fn(*mut u32, u32, *mut u16) -> u32;
type RmRegisterResourcesFn = unsafe extern "system" fn(
    u32,
    u32,
    *const *const u16,
    u32,
    *const RmUniqueProcess,
    u32,
    *const *const u16,
) -> u32;
type RmGetListFn =
    unsafe extern "system" fn(u32, *mut u32, *mut u32, *mut RmProcessInfo, *mut u32) -> u32;
type RmEndSessionFn = unsafe extern "system" fn(u32) -> u32;

type GetFileVersionInfoSizeExWFn =
    unsafe extern "system" fn(u32, *const u16, *mut u32) -> u32;
type GetFileVersionInfoExWFn =
    unsafe extern "system" fn(u32, *const u16, u32, u32, *mut c_void) -> i32;
type VerQueryValueWFn =
    unsafe extern "system" fn(*const c_void, *const u16, *mut *mut c_void, *mut u32) -> i32;

struct RmSession<'a> {
    handle: u32,
    end_session: RmEndSessionFn,
    _library: &'a SystemLibrary,
}

impl Drop for RmSession<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.handle` is a live session handle created by `RmStartSession`
        // on the still-loaded `rstrtmgr.dll` library.
        let _ = unsafe { (self.end_session)(self.handle) };
    }
}

fn win32_error(code: u32, fallback: &'static str) -> io::Error {
    i32::try_from(code).map_or_else(|_| invalid(fallback), io::Error::from_raw_os_error)
}

fn filetime_to_u64(time: FILETIME) -> u64 {
    (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime)
}

fn validated_wide(path: &Path, message: &'static str) -> io::Result<Vec<u16>> {
    let units = wide(path.as_os_str());
    if units[..units.len() - 1].contains(&0) {
        return Err(invalid(message));
    }
    Ok(units)
}

fn process_exited(handle: &OwnedHandle) -> io::Result<bool> {
    // SAFETY: `handle` is a live synchronization process handle and timeout 0 never blocks.
    match unsafe { WaitForSingleObject(handle.as_raw_handle(), 0) } {
        WAIT_OBJECT_0 => Ok(true),
        WAIT_TIMEOUT => Ok(false),
        WAIT_FAILED => Err(io::Error::last_os_error()),
        _ => Err(io::Error::other("unexpected process wait status")),
    }
}

fn wait_for_handle(handle: &OwnedHandle, deadline: Instant) -> io::Result<bool> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    let milliseconds = u32::try_from(remaining.as_millis())
        .unwrap_or(u32::MAX - 1)
        .min(u32::MAX - 1);
    // SAFETY: `handle` is a live synchronization process handle and `milliseconds < u32::MAX`.
    match unsafe { WaitForSingleObject(handle.as_raw_handle(), milliseconds) } {
        WAIT_OBJECT_0 => Ok(true),
        WAIT_TIMEOUT => Ok(false),
        WAIT_FAILED => Err(io::Error::last_os_error()),
        _ => Err(io::Error::other("unexpected process wait status")),
    }
}

fn query_image_path(handle: &OwnedHandle) -> Option<PathBuf> {
    let mut capacity = 512_usize;
    for _ in 0..8 {
        let mut buffer = vec![0_u16; capacity];
        let mut length = u32::try_from(buffer.len()).ok()?;
        // SAFETY: `handle` is live with `PROCESS_QUERY_LIMITED_INFORMATION`, `buffer`
        // is writable for `length` UTF-16 code units, and `length` receives the written count.
        let ok = unsafe {
            QueryFullProcessImageNameW(
                handle.as_raw_handle(),
                0,
                buffer.as_mut_ptr(),
                &raw mut length,
            )
        };
        if ok != 0 {
            let count = usize::try_from(length).ok()?;
            return Some(PathBuf::from(std::ffi::OsString::from_wide(
                &buffer[..count],
            )));
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(122) || capacity >= MAX_IMAGE_PATH_UNITS {
            return None;
        }
        capacity = (capacity * 2).min(MAX_IMAGE_PATH_UNITS);
    }
    None
}

fn open_query_sync(pid: u32) -> io::Result<Option<OwnedHandle>> {
    if pid == 0 {
        return Ok(None);
    }
    // SAFETY: `OpenProcess` requests only query and synchronization rights on a numeric PID.
    let raw = unsafe {
        OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE,
            0,
            pid,
        )
    };
    if raw.is_null() {
        let error = io::Error::last_os_error();
        if error.raw_os_error()
            == i32::try_from(ERROR_INVALID_PARAMETER)
                .ok()
        {
            return Ok(None);
        }
        return Err(error);
    }
    owned(raw).map(Some)
}

pub(crate) fn file_lock_holders(path: &Path) -> io::Result<Vec<ProcessIdentity>> {
    let path_wide = validated_wide(path, "Lock path contains an interior NUL.")?;
    let library =
        SystemLibrary::load_system32("rstrtmgr.dll", "Windows Restart Manager is unavailable.")?;
    let start_session = load_symbol!(&library, c"RmStartSession", RmStartSessionFn);
    let register_resources = load_symbol!(&library, c"RmRegisterResources", RmRegisterResourcesFn);
    let get_list = load_symbol!(&library, c"RmGetList", RmGetListFn);
    let end_session = load_symbol!(&library, c"RmEndSession", RmEndSessionFn);

    let mut handle = 0_u32;
    let mut session_key = [0_u16; CCH_RM_SESSION_KEY + 1];
    // SAFETY: `handle` and `session_key` are writable buffers matching the Restart Manager contract.
    let status = unsafe { start_session(&raw mut handle, 0, session_key.as_mut_ptr()) };
    if status != 0 {
        return Err(win32_error(status, "Restart Manager session failed."));
    }
    let session = RmSession {
        handle,
        end_session,
        _library: &library,
    };

    let resource_ptrs = [path_wide.as_ptr()];
    // SAFETY: `session.handle` is live and `resource_ptrs` points to one NUL-terminated UTF-16 path.
    let status = unsafe {
        register_resources(
            session.handle,
            1,
            resource_ptrs.as_ptr(),
            0,
            std::ptr::null(),
            0,
            std::ptr::null(),
        )
    };
    if status != 0 {
        return Err(win32_error(status, "Restart Manager resource registration failed."));
    }

    let mut needed = 0_u32;
    let mut count = 16_u32;
    let mut list = Vec::new();
    for _ in 0..MAX_RM_RETRIES {
        let capacity = usize::try_from(count)
            .map_err(|_| invalid("Restart Manager buffer count overflow."))?
            .min(MAX_LOCK_HOLDERS);
        list.resize_with(capacity, RmProcessInfo::default);
        let mut requested = u32::try_from(list.len())
            .map_err(|_| invalid("Restart Manager buffer count overflow."))?;
        let mut reboot_reasons = 0_u32;
        // SAFETY: `list` holds `requested` writable `RmProcessInfo` entries and `needed`/`reboot_reasons` are live.
        let status = unsafe {
            get_list(
                session.handle,
                &raw mut needed,
                &raw mut requested,
                list.as_mut_ptr(),
                &raw mut reboot_reasons,
            )
        };
        if status == 0 {
            let returned = usize::try_from(requested)
                .map_err(|_| invalid("Restart Manager count overflow."))?
                .min(list.len());
            list.truncate(returned);
            break;
        }
        if status == ERROR_MORE_DATA {
            if usize::try_from(needed).is_ok_and(|n| n > MAX_LOCK_HOLDERS) {
                return Err(invalid("Lock holder inventory exceeds its fixed bound."));
            }
            count = needed.max(1);
            continue;
        }
        return Err(win32_error(status, "Restart Manager query failed."));
    }

    let mut holders = Vec::new();
    let mut seen = BTreeSet::new();
    for entry in &list {
        let pid = entry.process.dw_process_id;
        if pid == 0 {
            continue;
        }
        let rm_creation = filetime_to_u64(entry.process.process_start_time);
        match open_query_sync(pid) {
            Ok(None) => continue,
            Ok(Some(process)) => {
                if process_exited(&process).unwrap_or(false) {
                    continue;
                }
                let Ok(actual_creation) = creation_file_time(&process) else {
                    if rm_creation != 0 && seen.insert((pid, rm_creation)) {
                        holders.push(ProcessIdentity {
                            pid,
                            creation_file_time: rm_creation,
                            executable_path: None,
                        });
                    }
                    continue;
                };
                if rm_creation != 0 && actual_creation != rm_creation {
                    continue;
                }
                if seen.insert((pid, actual_creation)) {
                    holders.push(ProcessIdentity {
                        pid,
                        creation_file_time: actual_creation,
                        executable_path: query_image_path(&process),
                    });
                }
            }
            Err(error)
                if error.raw_os_error() == i32::try_from(ERROR_ACCESS_DENIED).ok()
                    && rm_creation != 0 =>
            {
                if seen.insert((pid, rm_creation)) {
                    holders.push(ProcessIdentity {
                        pid,
                        creation_file_time: rm_creation,
                        executable_path: None,
                    });
                }
            }
            Err(_) => continue,
        }
    }
    Ok(holders)
}

pub(crate) fn process_direct_children(
    parent_pid: u32,
    parent_creation_file_time: u64,
) -> io::Result<Vec<ProcessIdentity>> {
    if parent_pid == 0 || parent_creation_file_time == 0 {
        return Err(invalid("Parent process identity must be nonzero."));
    }
    let mut candidate_pids = Vec::new();
    super::for_each_process("Process inventory exceeds its fixed bound.", |entry| {
        if entry.th32ParentProcessID == parent_pid
            && entry.th32ProcessID != 0
            && entry.th32ProcessID != parent_pid
        {
            if candidate_pids.len() >= MAX_TREE_DESCENDANTS {
                return Err(invalid("Direct child inventory exceeds its fixed bound."));
            }
            candidate_pids.push(entry.th32ProcessID);
        }
        Ok(true)
    })?;

    let mut children = Vec::new();
    for pid in candidate_pids {
        let Ok(Some(handle)) = open_query_sync(pid) else {
            continue;
        };
        if process_exited(&handle).unwrap_or(true) {
            continue;
        }
        let Ok(creation) = creation_file_time(&handle) else {
            continue;
        };
        if creation < parent_creation_file_time {
            continue;
        }
        children.push(ProcessIdentity {
            pid,
            creation_file_time: creation,
            executable_path: query_image_path(&handle),
        });
    }
    Ok(children)
}

pub(crate) fn file_fixed_version(path: &Path) -> io::Result<Option<(u16, u16, u16, u16)>> {
    let path_wide = validated_wide(path, "Version query path contains an interior NUL.")?;
    let library =
        SystemLibrary::load_system32("version.dll", "Windows file version API is unavailable.")?;
    let get_size = load_symbol!(
        &library,
        c"GetFileVersionInfoSizeExW",
        GetFileVersionInfoSizeExWFn
    );
    let get_info = load_symbol!(&library, c"GetFileVersionInfoExW", GetFileVersionInfoExWFn);
    let ver_query = load_symbol!(&library, c"VerQueryValueW", VerQueryValueWFn);

    let mut zero_handle = 0_u32;
    // SAFETY: `path_wide` is NUL-terminated and `zero_handle` is a writable u32 output.
    let size = unsafe { get_size(FILE_VER_GET_NEUTRAL, path_wide.as_ptr(), &raw mut zero_handle) };
    if size == 0 {
        let error = io::Error::last_os_error();
        if matches!(
            error.raw_os_error(),
            None | Some(
                0 | ERROR_RESOURCE_DATA_NOT_FOUND
                    | ERROR_RESOURCE_TYPE_NOT_FOUND
                    | ERROR_RESOURCE_NAME_NOT_FOUND
                    | ERROR_RESOURCE_LANG_NOT_FOUND
            )
        ) {
            return Ok(None);
        }
        return Err(error);
    }
    if size > MAX_VERSION_BYTES {
        return Err(invalid("Version resource exceeds its fixed bound."));
    }
    let words = usize::try_from(size)
        .map_err(|_| invalid("Version resource size overflow."))?
        .div_ceil(size_of::<u32>());
    let mut storage = vec![0_u32; words];
    // SAFETY: `storage` provides at least `size` writable 4-byte-aligned bytes for `GetFileVersionInfoExW`.
    check(unsafe {
        get_info(
            FILE_VER_GET_NEUTRAL,
            path_wide.as_ptr(),
            0,
            size,
            storage.as_mut_ptr().cast(),
        )
    })?;

    let sub_block = wide("\\");
    let mut value_ptr: *mut c_void = null_mut();
    let mut value_len = 0_u32;
    // SAFETY: `storage` holds the loaded version info resource block and `sub_block` is NUL-terminated.
    let queried = unsafe {
        ver_query(
            storage.as_ptr().cast(),
            sub_block.as_ptr(),
            &raw mut value_ptr,
            &raw mut value_len,
        )
    };
    if queried == 0 || value_ptr.is_null() {
        return Ok(None);
    }
    let len = usize::try_from(value_len).map_err(|_| invalid("Version value length overflow."))?;
    if len < size_of::<VS_FIXEDFILEINFO>() {
        return Ok(None);
    }
    // SAFETY: `VerQueryValueW` returned a live pointer into `storage` covering at least `VS_FIXEDFILEINFO`.
    let info = unsafe { std::ptr::read_unaligned(value_ptr.cast::<VS_FIXEDFILEINFO>()) };
    if info.dwSignature != VS_FFI_SIGNATURE_U32 {
        return Ok(None);
    }
    let major = u16::try_from((info.dwFileVersionMS >> 16) & 0xFFFF)
        .map_err(|_| invalid("Version major overflow."))?;
    let minor = u16::try_from(info.dwFileVersionMS & 0xFFFF)
        .map_err(|_| invalid("Version minor overflow."))?;
    let patch = u16::try_from((info.dwFileVersionLS >> 16) & 0xFFFF)
        .map_err(|_| invalid("Version patch overflow."))?;
    let build = u16::try_from(info.dwFileVersionLS & 0xFFFF)
        .map_err(|_| invalid("Version build overflow."))?;
    Ok(Some((major, minor, patch, build)))
}

struct LiveDescendant {
    pid: u32,
    creation_file_time: u64,
    handle: OwnedHandle,
}

fn collect_descendants(root_pid: u32, root_creation: u64) -> io::Result<Vec<LiveDescendant>> {
    let mut edges = Vec::new();
    super::for_each_process("Process inventory exceeds its fixed bound.", |entry| {
        if entry.th32ProcessID != 0 && entry.th32ProcessID != entry.th32ParentProcessID {
            edges.push((entry.th32ProcessID, entry.th32ParentProcessID));
        }
        Ok(true)
    })?;

    let mut descendants = Vec::new();
    let mut visited = BTreeSet::new();
    visited.insert(root_pid);
    let mut queue = VecDeque::new();
    queue.push_back((root_pid, root_creation, 0_usize));

    while let Some((parent_pid, parent_creation, depth)) = queue.pop_front() {
        if depth >= MAX_TREE_DEPTH {
            continue;
        }
        for &(child_pid, candidate_parent) in &edges {
            if candidate_parent != parent_pid || !visited.insert(child_pid) {
                continue;
            }
            let Ok(Some(handle)) = open_query_sync(child_pid) else {
                continue;
            };
            if process_exited(&handle).unwrap_or(true) {
                continue;
            }
            let Ok(child_creation) = creation_file_time(&handle) else {
                continue;
            };
            if child_creation < parent_creation {
                continue;
            }
            if descendants.len() >= MAX_TREE_DESCENDANTS {
                return Err(invalid("Target process tree exceeds its fixed bound."));
            }
            descendants.push(LiveDescendant {
                pid: child_pid,
                creation_file_time: child_creation,
                handle,
            });
            queue.push_back((child_pid, child_creation, depth + 1));
        }
    }
    Ok(descendants)
}

fn open_terminate(pid: u32) -> io::Result<Option<OwnedHandle>> {
    // SAFETY: `OpenProcess` opens `pid` with terminate, sync, and query rights without inheritance.
    let raw = unsafe {
        OpenProcess(
            PROCESS_TERMINATE | PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
            0,
            pid,
        )
    };
    if raw.is_null() {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == i32::try_from(ERROR_INVALID_PARAMETER).ok() {
            return Ok(None);
        }
        return Err(error);
    }
    owned(raw).map(Some)
}

fn terminate_verified(handle: &OwnedHandle, sync_handle: &OwnedHandle) -> io::Result<()> {
    // SAFETY: `handle` is an owned process handle with `PROCESS_TERMINATE` verified against creation time.
    let terminated = unsafe { TerminateProcess(handle.as_raw_handle(), 1) };
    if terminated == 0 {
        let error = io::Error::last_os_error();
        if process_exited(sync_handle)? {
            return Ok(());
        }
        return Err(error);
    }
    Ok(())
}

pub(crate) fn terminate_process_tree(
    pid: u32,
    expected_creation_file_time: u64,
    timeout: Duration,
) -> io::Result<()> {
    if pid == 0 || pid == std::process::id() || expected_creation_file_time == 0 {
        return Err(invalid("Refusing to terminate invalid or self process."));
    }
    let valid_timeout = u32::try_from(timeout.as_millis())
        .ok()
        .is_some_and(|ms| ms > 0 && ms < u32::MAX);
    let Some(deadline) = valid_timeout
        .then(|| Instant::now().checked_add(timeout))
        .flatten()
    else {
        return Err(invalid(
            "Process termination timeout must be positive and within the bounded native limit.",
        ));
    };

    let Some(root_handle) = open_terminate(pid)? else {
        return Ok(());
    };
    if creation_file_time(&root_handle)? != expected_creation_file_time {
        return Ok(());
    }
    if process_exited(&root_handle)? {
        return Ok(());
    }

    let descendants = collect_descendants(pid, expected_creation_file_time)?;
    terminate_verified(&root_handle, &root_handle)?;
    if !wait_for_handle(&root_handle, deadline)? {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "Timed out waiting for older process to exit.",
        ));
    }

    let grace_limit = Duration::from_secs(2).min(deadline.saturating_duration_since(Instant::now()));
    let grace_deadline = Instant::now() + grace_limit;
    let mut survivors = Vec::new();
    for descendant in descendants {
        if !wait_for_handle(&descendant.handle, grace_deadline)? {
            survivors.push(descendant);
        }
    }

    for descendant in survivors {
        if process_exited(&descendant.handle)? {
            continue;
        }
        if let Some(term_handle) = open_terminate(descendant.pid)?
            && creation_file_time(&term_handle)? == descendant.creation_file_time
        {
            terminate_verified(&term_handle, &descendant.handle)?;
        }
        if !wait_for_handle(&descendant.handle, deadline)? {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Timed out waiting for older process descendants to exit.",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::os::windows::fs::OpenOptionsExt as _;
    use std::os::windows::process::CommandExt as _;
    use std::process::{Command, Stdio};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_dir(name: &str) -> io::Result<PathBuf> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "Bonsai-InstanceNative-{name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&dir)?;
        Ok(dir)
    }

    fn spawn_holding_lock(lock_path: &Path, command_line: &str) -> io::Result<std::process::Child> {
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .share_mode(0)
            .custom_flags(0x0020_0000)
            .open(lock_path)?;
        let comspec =
            std::env::var_os("ComSpec").ok_or_else(|| io::Error::other("Missing ComSpec"))?;
        Command::new(comspec)
            .args(["/d", "/c", command_line])
            .creation_flags(0x0800_0000)
            .stdin(Stdio::from(lock))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
    }

    #[test]
    fn file_lock_holders_detects_exclusive_holder_and_clears_after_exit() -> io::Result<()> {
        let dir = temp_dir("lock-holders")?;
        let lock_path = dir.join("chat.lock");
        let mut child = spawn_holding_lock(&lock_path, "ping -n 30 127.0.0.1 >nul")?;
        let child_pid = child.id();
        let child_creation = super::super::process_creation_time(child_pid)?;

        let result = (|| -> io::Result<()> {
            let holders = file_lock_holders(&lock_path)?;
            let matched = holders
                .iter()
                .find(|entry| entry.pid == child_pid)
                .ok_or_else(|| io::Error::other("Missing exclusive lock holder in RmGetList"))?;
            assert_eq!(matched.creation_file_time, child_creation);
            assert!(matched.executable_path.is_some());
            Ok(())
        })();

        let _ = child.kill();
        let _ = child.wait();
        if result.is_ok() {
            let holders_after = file_lock_holders(&lock_path)?;
            assert!(holders_after.iter().all(|entry| entry.pid != child_pid));
        }
        let _ = fs::remove_dir_all(&dir);
        result
    }

    #[test]
    fn file_fixed_version_reads_pe_version_and_handles_non_pe() -> io::Result<()> {
        let comspec =
            std::env::var_os("ComSpec").ok_or_else(|| io::Error::other("Missing ComSpec"))?;
        let version = file_fixed_version(Path::new(&comspec))?
            .ok_or_else(|| io::Error::other("Expected ComSpec to have fixed version info"))?;
        assert!(version.0 > 0);

        let dir = temp_dir("fixed-version")?;
        let plain = dir.join("plain.txt");
        fs::write(&plain, b"not a PE image")?;
        assert_eq!(file_fixed_version(&plain)?, None);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn terminate_process_tree_guards_identity_and_kills_descendants() -> io::Result<()> {
        let self_creation = super::super::current_creation_time()?;
        assert!(terminate_process_tree(std::process::id(), self_creation, Duration::from_secs(1)).is_err());
        assert!(terminate_process_tree(0, 1, Duration::from_secs(1)).is_err());

        let dir = temp_dir("terminate-tree")?;
        let lock_path = dir.join("chat.lock");
        let mut child = spawn_holding_lock(&lock_path, "cmd /d /c ping -n 30 127.0.0.1 >nul")?;
        let child_pid = child.id();
        let child_creation = super::super::process_creation_time(child_pid)?;

        // Wait briefly for the nested `cmd` / `ping` descendant to appear.
        let mut descendants = Vec::new();
        for _ in 0..40 {
            descendants = collect_descendants(child_pid, child_creation)?;
            if !descendants.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        assert!(!descendants.is_empty());

        // Mismatched creation time must leave the process untouched.
        terminate_process_tree(child_pid, child_creation + 1, Duration::from_secs(1))?;
        assert!(child.try_wait()?.is_none());

        terminate_process_tree(child_pid, child_creation, Duration::from_secs(5))?;
        let status = child.wait()?;
        assert!(!status.success());
        for descendant in &descendants {
            assert!(process_exited(&descendant.handle)?);
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
