//! Read-only parent identity for direct native maintenance invocations.

use std::io;

use windows_sys::Wdk::System::Threading::{NtQueryInformationProcess, ProcessBasicInformation};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, PROCESS_BASIC_INFORMATION};

pub(crate) fn identity() -> io::Result<(u32, u64)> {
    let mut info = PROCESS_BASIC_INFORMATION::default();
    let mut returned = 0_u32;
    // SAFETY: GetCurrentProcess returns the current-process pseudo handle without ownership.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: `process` is valid for basic-information query, `info` is writable for its
    // exact byte size, and `returned` is a separate live output slot.
    let status = unsafe {
        NtQueryInformationProcess(
            process,
            ProcessBasicInformation,
            (&raw mut info).cast(),
            super::win_size::<PROCESS_BASIC_INFORMATION>(),
            &raw mut returned,
        )
    };
    if status < 0 || returned != super::win_size::<PROCESS_BASIC_INFORMATION>() {
        return Err(io::Error::other(
            "Current process basic information query failed.",
        ));
    }
    let id = u32::try_from(info.InheritedFromUniqueProcessId)
        .map_err(|_| super::invalid("Caller parent PID exceeds 32-bit range."))?;
    let creation = super::process_creation_time(id)?;
    let own_creation = super::current_creation_time()?;
    if creation > own_creation {
        return Err(super::invalid("The original caller PID has been reused."));
    }
    // Verify liveness and exact creation under a retained handle before returning;
    // command admission immediately retains its own guard.
    let parent = super::ParentProcess::open(id, creation)?;
    parent.ensure_alive()?;
    Ok((id, creation))
}
