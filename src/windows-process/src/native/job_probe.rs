//! Test-feature-only bounded failure context; never a serving admission rule.

use std::collections::BTreeMap;
use std::io;
use std::os::windows::io::AsRawHandle as _;
use std::path::Path;
use std::ptr;

use windows_sys::Win32::System::JobObjects::{
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JobObjectBasicAccountingInformation,
    JobObjectBasicProcessIdList, QueryInformationJobObject,
};
use windows_sys::Win32::System::Threading::{GetExitCodeProcess, GetProcessId};

pub(super) fn observe(process: &super::NativeProcess) -> String {
    let mut accounting = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
    // SAFETY: The exact live Job and correctly sized writable native structure
    // remain owned for the complete synchronous query; no pointers escape.
    let accounting_result = super::check(unsafe {
        QueryInformationJobObject(
            process.job.as_raw_handle(),
            JobObjectBasicAccountingInformation,
            (&raw mut accounting).cast(),
            super::win_size::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>(),
            ptr::null_mut(),
        )
    });
    let accounting_observation = match accounting_result {
        Ok(()) => format!(
            "total={},active={},limit_terminated={}",
            accounting.TotalProcesses,
            accounting.ActiveProcesses,
            accounting.TotalTerminatedProcesses
        ),
        Err(error) => format!("accounting_error={:?}", error.raw_os_error()),
    };
    // Independent stages preserve earlier counters and raw IDs even if strict
    // cleanup enumeration cannot reconcile assigned/listed counts or open a PID.
    let (raw_observation, mut ids) = raw_members(process);
    let strict_result = retained_members(process);
    if let Ok(members) = &strict_result {
        for (id, _) in members {
            if ids.len() < 64 && !ids.contains(id) {
                ids.push(*id);
            }
        }
    }
    let metadata_result = snapshot_metadata(&ids);
    format!(
        "{accounting_observation}; raw_job_list={raw_observation}; retained_pid_exit={strict_result:?}; snapshot_pid_parent_basename={metadata_result:?}"
    )
}

fn raw_members(process: &super::NativeProcess) -> (String, Vec<u32>) {
    const CAPACITY: usize = 64;
    let header_units = super::job_process_list_header_units();
    let mut storage = vec![0_usize; header_units + CAPACITY];
    let Ok(bytes) = u32::try_from(size_of_val(storage.as_slice())) else {
        return (
            "query_ok=false,win32=None,header_valid=false".into(),
            Vec::new(),
        );
    };
    // SAFETY: The usize-aligned initialized allocation contains the
    // JOBOBJECT_BASIC_PROCESS_ID_LIST header plus exactly CAPACITY pointer-sized IDs.
    // Only one bounded query is made; ERROR_MORE_DATA preserves partial diagnostic data.
    let result = unsafe {
        QueryInformationJobObject(
            process.job.as_raw_handle(),
            JobObjectBasicProcessIdList,
            storage.as_mut_ptr().cast(),
            bytes,
            ptr::null_mut(),
        )
    };
    let error = (result == 0).then(io::Error::last_os_error);
    let error_code = error.as_ref().and_then(io::Error::raw_os_error);
    let valid = result != 0 || error_code == i32::try_from(super::ERROR_MORE_DATA).ok();
    let (assigned, listed) = super::decode_job_process_list_header(&storage);
    let count = if valid {
        usize::try_from(listed).unwrap_or(CAPACITY).min(CAPACITY)
    } else {
        0
    };
    let raw_ids = &storage[header_units..header_units + count];
    let ids = raw_ids
        .iter()
        .filter_map(|id| u32::try_from(*id).ok())
        .collect();
    (
        format!(
            "query_ok={},win32={error_code:?},header_valid={valid},assigned={:?},listed={:?},raw_pids={raw_ids:?},listed_exceeds_buffer={}",
            result != 0,
            valid.then_some(assigned),
            valid.then_some(listed),
            valid && listed > 64
        ),
        ids,
    )
}

fn retained_members(process: &super::NativeProcess) -> io::Result<Vec<(u32, u32)>> {
    // Reuse cleanup's bounded enumeration, retained query handles and exact Job
    // membership verification. These handles are never used to terminate by PID.
    let handles = process.members()?;
    if handles.len() > 64 {
        return Err(super::invalid(
            "Fixture Job members exceed their diagnostic bound.",
        ));
    }
    let mut members = Vec::with_capacity(handles.len());
    for handle in &handles {
        // SAFETY: The retained queryable process handle stays live throughout
        // both calls. The exit-code output is correctly sized and writable.
        let id = unsafe { GetProcessId(handle.as_raw_handle()) };
        if id == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut exit_code = 0;
        // SAFETY: The same retained handle and writable DWORD remain live.
        super::check(unsafe { GetExitCodeProcess(handle.as_raw_handle(), &raw mut exit_code) })?;
        members.push((id, exit_code));
    }
    Ok(members)
}

fn snapshot_metadata(ids: &[u32]) -> io::Result<BTreeMap<u32, (u32, String)>> {
    let mut metadata = BTreeMap::new();
    super::for_each_process(
        "Fixture process snapshot exceeds its diagnostic bound.",
        |entry| {
            if ids.contains(&entry.th32ProcessID) {
                let exe_units =
                    super::utf16_z(&entry.szExeFile, "Unterminated fixture process basename.")?;
                let name = String::from_utf16_lossy(exe_units);
                let basename = Path::new(&name)
                    .file_name()
                    .ok_or_else(|| super::invalid("Missing fixture process basename."))?
                    .to_string_lossy()
                    .into_owned();
                metadata.insert(entry.th32ProcessID, (entry.th32ParentProcessID, basename));
            }
            Ok(true)
        },
    )?;
    Ok(metadata)
}
