//! Shared path locking, private scratch, quarantine slots, and piped launch supervision for isolated workers.
#![forbid(unsafe_code)]

use std::{
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io,
    os::windows::fs::{MetadataExt as _, OpenOptionsExt as _},
    path::{Component, Path, PathBuf, Prefix},
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, RecvTimeoutError},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use bonsai_windows_credentials::PrivateDirectory;
use bonsai_windows_process::{OwnedProcess, ProcessSpec, SandboxNetwork, SandboxPolicy};

/// Opened directory ancestors plus leaf handle preventing path rebinding through worker teardown.
pub(crate) struct PathLock {
    _ancestors: Vec<File>,
    file: File,
    len: u64,
}

impl PathLock {
    /// Opens ancestor directories from drive root down and locks a local regular file for reading.
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        if !path.is_absolute()
            || !matches!(
                path.components().next(),
                Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_))
            )
            || path
                .components()
                .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
        {
            return Err(io::Error::other("Invalid local file path"));
        }
        bonsai_windows_process::require_local_data_drive(path)?;
        let mut ancestors = Vec::new();
        for ancestor in path
            .ancestors()
            .skip(1)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            // FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES blocks ancestor rename/rebind.
            let directory = OpenOptions::new()
                .read(true)
                .access_mode(0x81)
                .share_mode(3)
                .custom_flags(0x0220_0000)
                .open(ancestor)?;
            let metadata = directory.metadata()?;
            if !metadata.is_dir() || metadata.file_attributes() & 0x400 != 0 {
                return Err(io::Error::other("Redirected file ancestry"));
            }
            ancestors.push(directory);
        }
        let file = OpenOptions::new()
            .read(true)
            .share_mode(1)
            .custom_flags(0x0020_0000)
            .open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.file_attributes() & 0x400 != 0 {
            return Err(io::Error::other("Invalid regular file"));
        }
        Ok(Self {
            _ancestors: ancestors,
            file,
            len: metadata.len(),
        })
    }

    /// Locks a nonempty local regular file within an optional byte cap.
    pub(crate) fn open_nonempty(path: &Path, max_bytes: Option<u64>) -> io::Result<Self> {
        let lock = Self::open(path)?;
        if lock.len == 0 || max_bytes.is_some_and(|cap| lock.len > cap) {
            return Err(io::Error::other("Invalid file size"));
        }
        Ok(lock)
    }

    pub(crate) const fn len(&self) -> u64 {
        self.len
    }

    pub(crate) const fn file(&self) -> &File {
        &self.file
    }
}

/// Private per-launch temporary directory with known fixed child entries.
pub(crate) struct WorkerScratch {
    directory: Option<PrivateDirectory>,
    stderr_file: Option<&'static str>,
}

impl WorkerScratch {
    /// Creates a private scratch directory containing a `worker` writable subdirectory.
    pub(crate) fn create(stderr_file: Option<&'static str>) -> io::Result<Self> {
        let directory = PrivateDirectory::create(&std::env::temp_dir())?;
        if let Err(error) = fs::create_dir(directory.path().join("worker")) {
            let path = directory.path().to_owned();
            drop(directory);
            let _removed = fs::remove_dir(path);
            return Err(error);
        }
        Ok(Self {
            directory: Some(directory),
            stderr_file,
        })
    }

    pub(crate) fn path(&self) -> io::Result<&Path> {
        self.directory
            .as_ref()
            .map(PrivateDirectory::path)
            .ok_or_else(|| io::Error::other("Worker scratch already released"))
    }

    pub(crate) fn worker_dir(&self) -> io::Result<PathBuf> {
        self.path().map(|path| path.join("worker"))
    }

    /// Removes known scratch entries and the private directory without traversing unknown files.
    pub(crate) fn cleanup(&mut self) -> io::Result<()> {
        let Some(directory) = self.directory.take() else {
            return Ok(());
        };
        let path = directory.path().to_owned();
        let entry_result = (|| -> io::Result<()> {
            if let Some(name) = self.stderr_file {
                let stderr = path.join(name);
                if stderr.try_exists()? {
                    fs::remove_file(stderr)?;
                }
            }
            let worker = path.join("worker");
            if self.stderr_file.is_none() || worker.try_exists()? {
                fs::remove_dir(worker)?;
            }
            Ok(())
        })();
        drop(directory);
        entry_result?;
        fs::remove_dir(path)
    }
}

impl Drop for WorkerScratch {
    fn drop(&mut self) {
        let _removed = self.cleanup();
    }
}

struct Quarantined {
    _source: Option<PathLock>,
    _executable: PathLock,
    _scratch: WorkerScratch,
    _reader: Option<JoinHandle<()>>,
}

/// Per-subsystem single-active admission gate and permanent post-failure quarantine.
pub(crate) struct WorkerSlot {
    active: AtomicBool,
    admission_closed: AtomicBool,
    quarantine: OnceLock<Quarantined>,
}

/// Admission rejection reason for a [`WorkerSlot`].
#[derive(Clone, Copy)]
pub(crate) enum SlotError {
    Active,
    Closed,
}

/// Releases active-slot ownership when dropped.
pub(crate) struct WorkerGuard<'a>(&'a WorkerSlot);

impl Drop for WorkerGuard<'_> {
    fn drop(&mut self) {
        self.0.active.store(false, Ordering::Release);
    }
}

impl WorkerSlot {
    pub(crate) const fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            admission_closed: AtomicBool::new(false),
            quarantine: OnceLock::new(),
        }
    }

    pub(crate) fn acquire(&self) -> Result<WorkerGuard<'_>, SlotError> {
        self.active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| SlotError::Active)?;
        let guard = WorkerGuard(self);
        if self.is_closed() {
            return Err(SlotError::Closed);
        }
        Ok(guard)
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.admission_closed.load(Ordering::Acquire)
    }

    pub(crate) fn close_admission(&self) {
        self.admission_closed.store(true, Ordering::Release);
    }

    pub(crate) fn quarantine(
        &self,
        source: Option<PathLock>,
        executable: PathLock,
        scratch: WorkerScratch,
        reader: Option<JoinHandle<()>>,
    ) {
        self.close_admission();
        self.quarantine.get_or_init(|| Quarantined {
            _source: source,
            _executable: executable,
            _scratch: scratch,
            _reader: reader,
        });
    }
}

/// Locked source/executable handles, private scratch, and process spec for an offline worker launch.
pub(crate) struct SandboxedLaunch {
    source: PathLock,
    executable: PathLock,
    scratch: WorkerScratch,
    specification: ProcessSpec,
    policy: SandboxPolicy,
}

impl SandboxedLaunch {
    #[expect(
        clippy::too_many_arguments,
        reason = "Captures the exact per-importer path caps, scratch filenames, memory limit, arguments, and error texts."
    )]
    pub(crate) fn prepare(
        path: &Path,
        max_source_bytes: Option<u64>,
        stdout_file: &'static str,
        stderr_file: &'static str,
        memory_limit_bytes: usize,
        arguments: Vec<OsString>,
        source_error: &'static str,
        isolation_error: &'static str,
    ) -> Result<Self, &'static str> {
        let source = PathLock::open_nonempty(path, max_source_bytes).map_err(|_| source_error)?;
        let executable_path = std::env::current_exe().map_err(|_| isolation_error)?;
        let executable =
            PathLock::open_nonempty(&executable_path, None).map_err(|_| isolation_error)?;
        let scratch = WorkerScratch::create(Some(stderr_file)).map_err(|_| isolation_error)?;
        let directory = scratch.path().map_err(|_| isolation_error)?;
        let working_directory = scratch.worker_dir().map_err(|_| isolation_error)?;
        let specification = ProcessSpec {
            executable: executable_path.clone(),
            arguments,
            working_directory: working_directory.clone(),
            stdout: directory.join(stdout_file),
            stderr: directory.join(stderr_file),
            memory_limit_bytes,
            active_process_limit: 1,
        };
        let policy = SandboxPolicy {
            readonly_paths: vec![executable_path, path.to_owned()],
            writable_directory: working_directory,
        };
        Ok(Self {
            source,
            executable,
            scratch,
            specification,
            policy,
        })
    }
}

/// Diagnostic error strings mapped during [`run_sandboxed_piped`].
#[derive(Clone, Copy)]
pub(crate) struct WorkerErrors {
    pub(crate) isolation: &'static str,
    pub(crate) reboot_pending: &'static str,
    pub(crate) unavailable: &'static str,
    pub(crate) protocol: &'static str,
    pub(crate) cleanup: &'static str,
}

/// Launches an offline piped worker, coordinates host IPC, and quarantines on unconfirmed cleanup.
pub(crate) fn run_sandboxed_piped<R, E, H>(
    slot: &'static WorkerSlot,
    mut launch: SandboxedLaunch,
    cleanup_limit: Duration,
    errors: WorkerErrors,
    start_reader: impl FnOnce(File, File) -> io::Result<(JoinHandle<()>, H)>,
    relay: impl FnOnce(H) -> Result<R, E>,
) -> Result<R, E>
where
    E: From<&'static str>,
{
    let (process, input, output) = match OwnedProcess::start_sandboxed_piped(
        &launch.specification,
        &launch.policy,
        SandboxNetwork::Offline,
    ) {
        Ok(started) => started,
        Err(error) if !bonsai_windows_process::process_start_cleanup_unconfirmed(&error) => {
            if error.kind() == io::ErrorKind::Unsupported {
                return Err(E::from(
                    if bonsai_windows_process::windows_reboot_pending() {
                        errors.reboot_pending
                    } else {
                        errors.unavailable
                    },
                ));
            }
            return Err(E::from(errors.isolation));
        }
        Err(_) => {
            slot.quarantine(Some(launch.source), launch.executable, launch.scratch, None);
            return Err(E::from(errors.cleanup));
        }
    };
    let Ok((reader, host_state)) = start_reader(input, output) else {
        if process.stop(cleanup_limit).is_err() {
            slot.quarantine(Some(launch.source), launch.executable, launch.scratch, None);
            drop(process);
            return Err(E::from(errors.cleanup));
        }
        drop(process);
        return Err(E::from(errors.isolation));
    };
    let result = relay(host_state);
    // `host_state` (channel endpoints) is dropped before exit/stop so the reader wakes cleanly.
    let teardown = finish_worker_teardown(&result, &process, cleanup_limit);
    if matches!(teardown, TeardownOutcome::QuarantineNeeded) {
        slot.quarantine(
            Some(launch.source),
            launch.executable,
            launch.scratch,
            Some(reader),
        );
        drop(process);
        return Err(E::from(errors.cleanup));
    }
    reader.join().map_err(|_| E::from(errors.protocol))?;
    drop(process);
    if launch.scratch.cleanup().is_err() {
        slot.close_admission();
        return Err(E::from(errors.cleanup));
    }
    match teardown {
        TeardownOutcome::Confirmed => result,
        TeardownOutcome::FailedExit => Err(E::from(errors.protocol)),
        TeardownOutcome::QuarantineNeeded => Err(E::from(errors.cleanup)),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TeardownOutcome {
    Confirmed,
    FailedExit,
    QuarantineNeeded,
}

fn decide_teardown(
    relay_ok: bool,
    wait_clean: bool,
    stop_process: impl FnOnce() -> bool,
) -> TeardownOutcome {
    if !relay_ok {
        return if stop_process() {
            TeardownOutcome::Confirmed
        } else {
            TeardownOutcome::QuarantineNeeded
        };
    }
    if wait_clean {
        return TeardownOutcome::Confirmed;
    }
    if stop_process() {
        TeardownOutcome::FailedExit
    } else {
        TeardownOutcome::QuarantineNeeded
    }
}

fn finish_worker_teardown<R, E>(
    result: &Result<R, E>,
    process: &OwnedProcess,
    cleanup_limit: Duration,
) -> TeardownOutcome {
    let relay_ok = result.is_ok();
    let wait_clean = relay_ok
        && matches!(
            process.wait_for_exit(cleanup_limit),
            Ok(Some(0)) if matches!(process.active_processes(), Ok(0))
        );
    decide_teardown(relay_ok, wait_clean, || process.stop(cleanup_limit).is_ok())
}

/// Polls a bounded worker channel until an item arrives, `deadline` expires, or `cancelled` is set.
pub(crate) fn recv_until<T, E>(
    receiver: &Receiver<Result<T, E>>,
    cancelled: &AtomicBool,
    deadline: Instant,
    poll: Duration,
    on_cancel: impl Fn() -> E,
    on_timeout: impl FnOnce() -> E,
    on_disconnect: impl FnOnce() -> E,
) -> Result<T, E> {
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err(on_cancel());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(on_timeout());
        }
        match receiver.recv_timeout(poll.min(remaining)) {
            Ok(result) => {
                return if cancelled.load(Ordering::Acquire) {
                    Err(on_cancel())
                } else {
                    result
                };
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Err(on_disconnect()),
        }
    }
}

pub(crate) fn le_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    offset
        .checked_add(4)
        .and_then(|end| bytes.get(offset..end))
        .and_then(|slice| slice.try_into().ok())
        .map(u32::from_le_bytes)
}

pub(crate) fn le_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    offset
        .checked_add(8)
        .and_then(|end| bytes.get(offset..end))
        .and_then(|slice| slice.try_into().ok())
        .map(u64::from_le_bytes)
}

#[cfg(test)]
mod tests {
    use super::{TeardownOutcome, decide_teardown};

    #[test]
    fn teardown_stops_worker_before_deciding_quarantine_on_unclean_exit() {
        let mut stop_calls = 0_usize;
        assert_eq!(
            decide_teardown(true, true, || {
                stop_calls += 1;
                true
            }),
            TeardownOutcome::Confirmed
        );
        assert_eq!(stop_calls, 0);

        assert_eq!(
            decide_teardown(true, false, || {
                stop_calls += 1;
                true
            }),
            TeardownOutcome::FailedExit
        );
        assert_eq!(stop_calls, 1);

        assert_eq!(
            decide_teardown(true, false, || {
                stop_calls += 1;
                false
            }),
            TeardownOutcome::QuarantineNeeded
        );
        assert_eq!(stop_calls, 2);

        assert_eq!(
            decide_teardown(false, false, || {
                stop_calls += 1;
                true
            }),
            TeardownOutcome::Confirmed
        );
        assert_eq!(stop_calls, 3);

        assert_eq!(
            decide_teardown(false, false, || {
                stop_calls += 1;
                false
            }),
            TeardownOutcome::QuarantineNeeded
        );
        assert_eq!(stop_calls, 4);

        // Failed relay with wait_clean = true still invokes stop_process before deciding.
        assert_eq!(
            decide_teardown(false, true, || {
                stop_calls += 1;
                true
            }),
            TeardownOutcome::Confirmed
        );
        assert_eq!(stop_calls, 5);
    }
}
