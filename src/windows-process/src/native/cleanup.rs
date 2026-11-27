//! Keep an explicit native stop's allowance spent when its owner is dropped.

use std::{
    io,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

#[derive(Default)]
pub(super) struct StopAttempt(AtomicBool);

impl StopAttempt {
    pub(super) fn run(
        &self,
        timeout: Duration,
        stop: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        let valid_millis = u32::try_from(timeout.as_millis())
            .ok()
            .is_some_and(|ms| ms > 0 && ms < u32::MAX);
        if !valid_millis || Instant::now().checked_add(timeout).is_none() {
            return Err(super::invalid(
                "cleanup timeout must be positive and within the bounded native limit",
            ));
        }
        // Spend the allowance before entering native calls, including failures
        // and unwinding. A later Drop cannot turn either into another wait.
        self.0.store(true, Ordering::Relaxed);
        stop()
    }

    #[cfg(feature = "containment")]
    pub(super) fn on_drop(&self, stop: impl FnOnce() -> io::Result<()>) {
        if !self.0.swap(true, Ordering::Relaxed) {
            let _stopped = stop();
        }
    }
}

/// Removes newly created `CREATE_NEW` output files if launch fails before completion.
pub(super) struct CreatedOutputs<'a> {
    paths: Vec<&'a Path>,
    armed: bool,
}

impl<'a> CreatedOutputs<'a> {
    pub(super) fn new() -> Self {
        Self {
            paths: Vec::new(),
            armed: true,
        }
    }

    pub(super) fn record(&mut self, path: &'a Path) {
        self.paths.push(path);
    }

    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CreatedOutputs<'_> {
    fn drop(&mut self) {
        if self.armed {
            for path in self.paths.drain(..) {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "containment")]
    #[test]
    fn explicit_cleanup_failure_never_renews_the_drop_allowance() {
        for result in [
            Ok(()),
            Err(io::ErrorKind::TimedOut),
            Err(io::ErrorKind::Other),
        ] {
            let attempt = StopAttempt::default();
            let mut stops = Vec::new();
            let outcome = attempt.run(Duration::from_secs(2), || {
                assert!(attempt.0.load(Ordering::Relaxed));
                stops.push("explicit");
                result.map_err(io::Error::from)
            });
            assert_eq!(outcome.is_ok(), result.is_ok());
            attempt.on_drop(|| {
                stops.push("drop");
                Ok(())
            });
            assert_eq!(stops, ["explicit"]);
        }

        // Invalid input never starts native teardown; the fallback still owns
        // exactly one attempt and cannot restart itself after that attempt fails.
        let attempt = StopAttempt::default();
        let mut stops = 0;
        assert!(
            attempt
                .run(Duration::ZERO, || {
                    stops += 1;
                    Ok(())
                })
                .is_err()
        );
        for _ in 0..2 {
            attempt.on_drop(|| {
                stops += 1;
                Err(io::Error::from(io::ErrorKind::TimedOut))
            });
        }
        assert_eq!(stops, 1);
    }

    #[test]
    fn invalid_timeouts_never_spend_stop_attempt() {
        for invalid_timeout in [
            Duration::ZERO,
            Duration::from_nanos(1),
            Duration::from_millis(u64::from(u32::MAX)),
            Duration::MAX,
        ] {
            let attempt = StopAttempt::default();
            let result = attempt.run(invalid_timeout, || Ok(()));
            assert!(result.is_err());
            assert!(!attempt.0.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn created_outputs_removes_recorded_files_unless_disarmed() -> io::Result<()> {
        let dir =
            std::env::temp_dir().join(format!("bonsai-created-outputs-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let orphan = dir.join("orphan.log");
        let kept = dir.join("kept.log");
        std::fs::write(&orphan, b"")?;
        std::fs::write(&kept, b"")?;

        {
            let mut guard = CreatedOutputs::new();
            guard.record(&orphan);
        }
        assert!(!orphan.exists());

        {
            let mut guard = CreatedOutputs::new();
            guard.record(&kept);
            guard.disarm();
        }
        assert!(kept.exists());

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
