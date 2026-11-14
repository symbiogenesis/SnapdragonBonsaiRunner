//! Immutable executable grants and private scratch retained through worker death.
use super::{ADMISSION_CLOSED, Ordering};
use bonsai_windows_credentials::PrivateDirectory;
use bonsai_windows_process::{ProcessSpec, SandboxPolicy};
use std::{
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io,
    os::windows::fs::{MetadataExt as _, OpenOptionsExt as _},
    path::{Component, Path, Prefix},
    sync::Mutex,
    thread::JoinHandle,
};

const PROCESS_COMMIT_BYTES: usize = 256 * 1024 * 1024;

struct Executable {
    _ancestors: Vec<File>,
    _file: File,
}

impl Executable {
    fn open(path: &Path) -> io::Result<Self> {
        if !path.is_absolute()
            || !matches!(path.components().next(), Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_)))
            || path
                .components()
                .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
        {
            return Err(io::Error::other("Invalid Markdown worker image path"));
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
            let directory = OpenOptions::new()
                .read(true)
                .access_mode(0x81)
                .share_mode(3)
                .custom_flags(0x0220_0000)
                .open(ancestor)?;
            let metadata = directory.metadata()?;
            if !metadata.is_dir() || metadata.file_attributes() & 0x400 != 0 {
                return Err(io::Error::other(
                    "Redirected Markdown worker image ancestry",
                ));
            }
            ancestors.push(directory);
        }
        let file = OpenOptions::new()
            .read(true)
            .share_mode(1)
            .custom_flags(0x0020_0000)
            .open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() == 0 || metadata.file_attributes() & 0x400 != 0 {
            return Err(io::Error::other("Invalid Markdown worker image"));
        }
        Ok(Self {
            _ancestors: ancestors,
            _file: file,
        })
    }
}

pub(super) struct Prepared {
    pub(super) specification: ProcessSpec,
    pub(super) policy: SandboxPolicy,
    _executable: Executable,
    scratch: Option<PrivateDirectory>,
}

impl Prepared {
    pub(super) fn new() -> io::Result<Self> {
        let executable = std::env::current_exe()?;
        let retained = Executable::open(&executable)?;
        let scratch = PrivateDirectory::create(&std::env::temp_dir())?;
        let worker = scratch.path().join("worker");
        if let Err(error) = fs::create_dir(&worker) {
            let path = scratch.path().to_owned();
            drop(scratch);
            let _removed = fs::remove_dir(path);
            return Err(error);
        }
        let specification = ProcessSpec {
            executable: executable.clone(),
            arguments: vec![OsString::from("--markdown-worker")],
            working_directory: worker.clone(),
            stdout: scratch.path().join("markdown.stdout"),
            stderr: scratch.path().join("markdown.stderr"),
            memory_limit_bytes: PROCESS_COMMIT_BYTES,
            active_process_limit: 1,
        };
        let policy = SandboxPolicy {
            readonly_paths: vec![executable],
            writable_directory: worker,
        };
        Ok(Self {
            specification,
            policy,
            _executable: retained,
            scratch: Some(scratch),
        })
    }

    pub(super) fn cleanup(&mut self) -> io::Result<()> {
        let Some(scratch) = self.scratch.take() else {
            return Ok(());
        };
        let path = scratch.path().to_owned();
        // Unknown worker-created entries remain private and close admission;
        // cleanup never traverses potentially hostile filesystem contents.
        let result = fs::remove_dir(path.join("worker"));
        drop(scratch);
        result?;
        fs::remove_dir(path)
    }
}

impl Drop for Prepared {
    fn drop(&mut self) {
        let _removed = self.cleanup();
    }
}

struct Quarantine {
    _prepared: Prepared,
    _reader: Option<JoinHandle<()>>,
}

static QUARANTINE: Mutex<Option<Quarantine>> = Mutex::new(None);

pub(super) fn quarantine(prepared: Prepared, reader: Option<JoinHandle<()>>) {
    ADMISSION_CLOSED.store(true, Ordering::Release);
    *QUARANTINE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Quarantine {
        _prepared: prepared,
        _reader: reader,
    });
}
