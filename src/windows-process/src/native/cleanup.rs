//! Keep an explicit native stop's allowance spent when its owner is dropped.

use std::{
    io,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

#[derive(Default)]
pub(super) struct StopAttempt(AtomicBool);

impl StopAttempt {
    pub(super) fn run(
        &self,
        timeout: Duration,
        stop: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        if timeout.is_zero() {
            return Err(super::invalid("cleanup timeout must be positive"));
        }
        // Spend the allowance before entering native calls, including failures
        // and unwinding. A later Drop cannot turn either into another wait.
        self.0.store(true, Ordering::Relaxed);
        stop()
    }

    pub(super) fn on_drop(&self, stop: impl FnOnce() -> io::Result<()>) {
        if !self.0.swap(true, Ordering::Relaxed) {
            let _stopped = stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
