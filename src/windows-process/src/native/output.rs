//! Parent-owned, size-bounded captures for untrusted worker standard streams.

use std::fs::File;
use std::io::{self, Read, Write};
use std::os::windows::io::{AsRawHandle as _, OwnedHandle};
use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use windows_sys::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation};
use windows_sys::Win32::Storage::FileSystem::{CREATE_NEW, FILE_SHARE_READ};
use windows_sys::Win32::System::Threading::TerminateProcess;

use super::{anonymous_pipe, check, standard_file};

// This is a hard retained-file bound, independently of how often the broker
// samples resources. The pipe and stack buffer provide bounded backpressure.
pub(super) const MAX_LOG_BYTES: usize = 16 * 1024 * 1024;

const PENDING: u8 = 0;
const COMPLETE: u8 = 1;
const READ_FAILED: u8 = 2;
const WRITE_FAILED: u8 = 3;
const LIMIT_EXCEEDED: u8 = 4;
const THREAD_FAILED: u8 = 5;

struct PendingCapture {
    reader: File,
    file: File,
    label: &'static str,
}

pub(super) struct PendingCaptures([PendingCapture; 2]);

struct Capture {
    result: Arc<AtomicU8>,
    thread: Mutex<Option<JoinHandle<()>>>,
    label: &'static str,
}

pub(super) struct Captures([Capture; 2]);

pub(super) fn prepare(
    stdout: &Path,
    stderr: &Path,
) -> io::Result<([OwnedHandle; 2], PendingCaptures)> {
    let (output, stdout) = pending(stdout, "stdout")?;
    let (error, stderr) = pending(stderr, "stderr")?;
    Ok(([output, error], PendingCaptures([stdout, stderr])))
}

fn pending(path: &Path, label: &'static str) -> io::Result<(OwnedHandle, PendingCapture)> {
    let file = standard_file(path.as_os_str(), 0x4000_0000, FILE_SHARE_READ, CREATE_NEW)?;
    // SAFETY: This fresh parent-owned file is live; clearing inheritance keeps
    // the untrusted worker from receiving a seekable, unbounded disk sink.
    check(unsafe { SetHandleInformation(file.as_raw_handle(), HANDLE_FLAG_INHERIT, 0) })?;
    let (reader, writer) = anonymous_pipe(false)?;
    Ok((
        writer,
        PendingCapture {
            reader: reader.into(),
            file: file.into(),
            label,
        },
    ))
}

impl PendingCaptures {
    pub(super) fn start(self, process: &OwnedHandle) -> io::Result<Captures> {
        let [stdout, stderr] = self.0;
        // Finish fallible handle duplication before any reader thread exists.
        let stdout_process = process.try_clone()?;
        let stderr_process = process.try_clone()?;
        Ok(Captures([
            stdout.start(stdout_process),
            stderr.start(stderr_process),
        ]))
    }
}

impl PendingCapture {
    fn start(self, process: OwnedHandle) -> Capture {
        let result = Arc::new(AtomicU8::new(PENDING));
        let published = Arc::clone(&result);
        let label = self.label;
        let thread = thread::Builder::new()
            .name(format!("sandbox-{label}-capture"))
            .spawn(move || {
                let Self {
                    mut reader,
                    mut file,
                    label: _,
                } = self;
                let status = copy_bounded(&mut reader, &mut file, MAX_LOG_BYTES);
                // Release the parent file and read endpoint before publishing
                // completion; receipt consumers may immediately reopen them.
                drop(file);
                drop(reader);
                if status != COMPLETE {
                    // SAFETY: This duplicate retains the exact newly created
                    // one-process sandbox worker against PID reuse. A process
                    // handle, never a Job clone, preserves kill-on-owner-loss.
                    let _ = unsafe { TerminateProcess(process.as_raw_handle(), 1) };
                }
                drop(process);
                published.store(status, Ordering::Release);
            });
        let thread = if let Ok(thread) = thread {
            Some(thread)
        } else {
            result.store(THREAD_FAILED, Ordering::Release);
            None
        };
        Capture {
            result,
            thread: Mutex::new(thread),
            label,
        }
    }
}

impl Captures {
    pub(super) fn finished(&self) -> io::Result<bool> {
        // Always visit both captures, even if the first is still draining.
        let stdout = self.0[0].finished()?;
        let stderr = self.0[1].finished()?;
        Ok(stdout && stderr)
    }

    pub(super) fn check(&self) -> io::Result<()> {
        // Observe a panicked drain even while its worker is still alive. This
        // joins only already-finished threads and never waits for pipe EOF.
        let _ = self.finished()?;
        for capture in &self.0 {
            let cause = match capture.result.load(Ordering::Acquire) {
                PENDING | COMPLETE => continue,
                READ_FAILED => "could not read the worker pipe",
                WRITE_FAILED => "could not retain the worker log",
                LIMIT_EXCEEDED => "exceeded the 16 MiB retained-log bound",
                _ => "capture thread could not start or complete",
            };
            return Err(io::Error::other(format!(
                "sandbox {} {cause}",
                capture.label
            )));
        }
        Ok(())
    }
}

impl Capture {
    fn finished(&self) -> io::Result<bool> {
        let mut handle = self
            .thread
            .lock()
            .map_err(|_| io::Error::other("sandbox capture ownership was poisoned"))?;
        if handle.as_ref().is_some_and(|thread| !thread.is_finished()) {
            return Ok(false);
        }
        if let Some(thread) = handle.take()
            && thread.join().is_err()
        {
            self.result.store(THREAD_FAILED, Ordering::Release);
        }
        drop(handle);
        Ok(true)
    }
}

fn copy_bounded(reader: &mut impl Read, writer: &mut impl Write, limit: usize) -> u8 {
    let mut retained = 0;
    let mut buffer = [0_u8; 16_384];
    loop {
        let count = match reader.read(&mut buffer) {
            Ok(0) => return COMPLETE,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == io::ErrorKind::BrokenPipe => return COMPLETE,
            Err(_) => return READ_FAILED,
        };
        let admitted = count.min(limit - retained);
        if writer.write_all(&buffer[..admitted]).is_err() {
            return WRITE_FAILED;
        }
        retained += admitted;
        if admitted != count {
            return LIMIT_EXCEEDED;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn sandbox_capture_retains_exact_prefix_and_rejects_first_excess_byte() {
        for limit in [0, 1, 16_383, 16_384, 16_385] {
            let source = (0_u8..=255).cycle().take(limit + 1).collect::<Vec<_>>();
            for length in [limit, limit + 1] {
                let mut retained = Vec::new();
                assert_eq!(
                    copy_bounded(&mut Cursor::new(&source[..length]), &mut retained, limit),
                    if length == limit {
                        COMPLETE
                    } else {
                        LIMIT_EXCEEDED
                    }
                );
                assert_eq!(retained, source[..limit]);
            }
        }
    }

    struct FailedRead;

    impl Read for FailedRead {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("synthetic read failure"))
        }
    }

    struct FailedWrite;

    impl Write for FailedWrite {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::StorageFull,
                "synthetic disk exhaustion",
            ))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn sandbox_capture_distinguishes_read_and_disk_exhaustion_failures() {
        let mut retained = Vec::new();
        assert_eq!(copy_bounded(&mut FailedRead, &mut retained, 4), READ_FAILED);
        assert!(retained.is_empty());
        assert_eq!(
            copy_bounded(&mut Cursor::new(b"bytes"), &mut FailedWrite, 4),
            WRITE_FAILED
        );
    }

    #[test]
    fn sandbox_capture_observes_panicked_drain_before_worker_exit() -> io::Result<()> {
        let thread = thread::spawn(|| {
            std::panic::resume_unwind(Box::new("synthetic capture panic"));
        });
        let captures = Captures([
            Capture {
                result: Arc::new(AtomicU8::new(PENDING)),
                thread: Mutex::new(Some(thread)),
                label: "stdout",
            },
            Capture {
                result: Arc::new(AtomicU8::new(COMPLETE)),
                thread: Mutex::new(None),
                label: "stderr",
            },
        ]);
        let started = std::time::Instant::now();
        while captures.check().is_ok() {
            assert!(started.elapsed() < std::time::Duration::from_secs(5));
            thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(captures.finished()?);
        assert_eq!(captures.0[0].result.load(Ordering::Acquire), THREAD_FAILED);
        Ok(())
    }
}
