//! Parent-owned, size-bounded captures for untrusted worker standard streams.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::windows::fs::OpenOptionsExt as _;
use std::os::windows::io::{AsRawHandle as _, OwnedHandle};
use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use windows_sys::Win32::Storage::FileSystem::{FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_READ};
use windows_sys::Win32::System::Threading::TerminateProcess;

use super::anonymous_pipe;

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

pub(super) struct PendingCaptures(Vec<PendingCapture>);

struct DoneSignal(Arc<(Mutex<bool>, Condvar)>);

impl DoneSignal {
    fn new() -> Arc<(Mutex<bool>, Condvar)> {
        Arc::new((Mutex::new(false), Condvar::new()))
    }

    fn signal(target: &(Mutex<bool>, Condvar)) {
        if let Ok(mut finished) = target.0.lock() {
            *finished = true;
            target.1.notify_all();
        }
    }
}

impl Drop for DoneSignal {
    fn drop(&mut self) {
        Self::signal(&self.0);
    }
}

struct Capture {
    result: Arc<AtomicU8>,
    done: Arc<(Mutex<bool>, Condvar)>,
    thread: Mutex<Option<JoinHandle<()>>>,
    label: &'static str,
}

pub(super) struct Captures(Vec<Capture>);

pub(super) fn prepare(
    stdout: &Path,
    stderr: &Path,
) -> io::Result<([OwnedHandle; 2], PendingCaptures)> {
    let (output, stdout) = pending(stdout, "stdout")?;
    let (error, stderr) = pending(stderr, "stderr")?;
    Ok(([output, error], PendingCaptures(vec![stdout, stderr])))
}

pub(super) fn prepare_stderr(
    stdout: &Path,
    stderr: &Path,
) -> io::Result<(OwnedHandle, PendingCaptures)> {
    // Keep a 0-byte stdout sentinel on disk while protocol traffic uses pipes.
    drop(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .share_mode(FILE_SHARE_READ)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(stdout)?,
    );
    let (error, stderr) = pending(stderr, "stderr")?;
    Ok((error, PendingCaptures(vec![stderr])))
}

fn pending(path: &Path, label: &'static str) -> io::Result<(OwnedHandle, PendingCapture)> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .share_mode(FILE_SHARE_READ)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)?;
    let (reader, writer) = anonymous_pipe(false)?;
    Ok((
        writer,
        PendingCapture {
            reader: reader.into(),
            file,
            label,
        },
    ))
}

impl PendingCaptures {
    pub(super) fn start(self, process: &OwnedHandle) -> io::Result<Captures> {
        // Finish fallible handle duplication before any reader thread exists.
        let mut paired = Vec::with_capacity(self.0.len());
        for capture in self.0 {
            paired.push((capture, process.try_clone()?));
        }
        Ok(Captures(
            paired
                .into_iter()
                .map(|(capture, cloned)| capture.start(cloned))
                .collect(),
        ))
    }
}

impl PendingCapture {
    fn start(self, process: OwnedHandle) -> Capture {
        let result = Arc::new(AtomicU8::new(PENDING));
        let done = DoneSignal::new();
        let published = Arc::clone(&result);
        let done_worker = Arc::clone(&done);
        let label = self.label;
        let thread = thread::Builder::new()
            .name(format!("sandbox-{label}-capture"))
            .spawn(move || {
                let _signal = DoneSignal(done_worker);
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
        let thread = match thread {
            Ok(thread) => Some(thread),
            Err(_) => {
                result.store(THREAD_FAILED, Ordering::Release);
                DoneSignal::signal(&done);
                None
            }
        };
        Capture {
            result,
            done,
            thread: Mutex::new(thread),
            label,
        }
    }
}

impl Captures {
    pub(super) fn finished(&self) -> io::Result<bool> {
        // Always visit every capture, even if an earlier one is still draining.
        let mut all_finished = true;
        for capture in &self.0 {
            if !capture.finished()? {
                all_finished = false;
            }
        }
        Ok(all_finished)
    }

    pub(super) fn wait_until(&self, deadline: Instant) -> io::Result<bool> {
        for capture in &self.0 {
            if !capture.wait_until(deadline)? {
                return Ok(false);
            }
        }
        Ok(true)
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

    fn wait_until(&self, deadline: Instant) -> io::Result<bool> {
        {
            let (lock, signal) = &*self.done;
            let mut finished = lock
                .lock()
                .map_err(|_| io::Error::other("sandbox capture completion lock was poisoned"))?;
            while !*finished {
                let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                    return Ok(false);
                };
                if remaining.is_zero() {
                    return Ok(false);
                }
                let (guard, timeout) = signal.wait_timeout(finished, remaining).map_err(|_| {
                    io::Error::other("sandbox capture completion lock was poisoned")
                })?;
                finished = guard;
                if timeout.timed_out() && !*finished {
                    return Ok(false);
                }
            }
        }
        let mut handle = self
            .thread
            .lock()
            .map_err(|_| io::Error::other("sandbox capture ownership was poisoned"))?;
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
        assert_eq!(retained, Vec::<u8>::new());
        assert_eq!(
            copy_bounded(&mut Cursor::new(b"bytes"), &mut FailedWrite, 4),
            WRITE_FAILED
        );
    }

    #[test]
    fn sandbox_capture_observes_panicked_drain_before_worker_exit() -> io::Result<()> {
        let done = DoneSignal::new();
        let done_worker = Arc::clone(&done);
        let thread = thread::spawn(move || {
            let _signal = DoneSignal(done_worker);
            std::panic::resume_unwind(Box::new("synthetic capture panic"));
        });
        let captures = Captures(vec![
            Capture {
                result: Arc::new(AtomicU8::new(PENDING)),
                done,
                thread: Mutex::new(Some(thread)),
                label: "stdout",
            },
            Capture {
                result: Arc::new(AtomicU8::new(COMPLETE)),
                done: Arc::new((Mutex::new(true), Condvar::new())),
                thread: Mutex::new(None),
                label: "stderr",
            },
        ]);
        assert!(captures.wait_until(Instant::now() + std::time::Duration::from_secs(5))?);
        assert!(captures.check().is_err());
        assert!(captures.finished()?);
        assert_eq!(captures.0[0].result.load(Ordering::Acquire), THREAD_FAILED);
        Ok(())
    }
}
