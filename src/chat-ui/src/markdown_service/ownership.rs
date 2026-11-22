//! Immutable executable grants and private scratch retained through worker death.
use super::SLOT;
use crate::isolated_worker::{PathLock, WorkerScratch};
use bonsai_windows_process::{ProcessSpec, SandboxPolicy};
use std::{ffi::OsString, io, path::Path, thread::JoinHandle};

const PROCESS_COMMIT_BYTES: usize = 256 * 1024 * 1024;

pub(super) struct Prepared {
    pub(super) specification: ProcessSpec,
    pub(super) policy: SandboxPolicy,
    executable: PathLock,
    scratch: WorkerScratch,
}

impl Prepared {
    pub(super) fn new(executable_path: &Path) -> io::Result<Self> {
        let executable = PathLock::open_nonempty(executable_path, None)?;
        let scratch = WorkerScratch::create(None)?;
        let path = scratch.path()?.to_owned();
        let worker = scratch.worker_dir()?;
        let specification = ProcessSpec {
            executable: executable_path.to_owned(),
            arguments: vec![OsString::from("--markdown-worker")],
            working_directory: worker.clone(),
            stdout: path.join("markdown.stdout"),
            stderr: path.join("markdown.stderr"),
            memory_limit_bytes: PROCESS_COMMIT_BYTES,
            active_process_limit: 1,
        };
        let policy = SandboxPolicy {
            readonly_paths: vec![executable_path.to_owned()],
            writable_directory: worker,
        };
        Ok(Self {
            specification,
            policy,
            executable,
            scratch,
        })
    }

    pub(super) fn cleanup(&mut self) -> io::Result<()> {
        self.scratch.cleanup()
    }
}

pub(super) fn quarantine(prepared: Prepared, reader: Option<JoinHandle<()>>) {
    SLOT.quarantine(None, prepared.executable, prepared.scratch, reader);
}
