//! Coalescing single-slot background persistence worker for device-local state.

#![forbid(unsafe_code)]

use std::{
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, SyncSender, TrySendError},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

/// Single-slot coalescing background writer retaining only the latest queued state.
pub(super) struct CoalescingStore<T> {
    latest: Option<T>,
    pending: Arc<Mutex<Option<T>>>,
    wake: Option<SyncSender<()>>,
    failed: Arc<AtomicBool>,
    retry: Arc<AtomicBool>,
    retry_cooldown: Option<Duration>,
    last_attempt: Instant,
    stopped_message: &'static str,
    worker: Option<JoinHandle<()>>,
}

impl<T: Clone + Send + 'static> CoalescingStore<T> {
    /// Starts the dedicated worker thread with optional initial baseline and retry cooldown.
    pub(super) fn spawn(
        thread_name: &str,
        stopped_message: &'static str,
        initial: Option<T>,
        retry_cooldown: Option<Duration>,
        mut save: impl FnMut(T) -> io::Result<()> + Send + 'static,
    ) -> io::Result<Self> {
        let pending = Arc::new(Mutex::new(None));
        let queued = Arc::clone(&pending);
        let failed = Arc::new(AtomicBool::new(false));
        let reported = Arc::clone(&failed);
        let retry = Arc::new(AtomicBool::new(false));
        let retry_requested = Arc::clone(&retry);
        let (wake, incoming) = mpsc::sync_channel(1);
        let worker = std::thread::Builder::new()
            .name(thread_name.to_owned())
            .spawn(move || {
                while incoming.recv().is_ok() {
                    let value = queued.lock().ok().and_then(|mut slot| slot.take());
                    if let Some(value) = value
                        && save(value).is_err()
                    {
                        reported.store(true, Ordering::Release);
                        retry_requested.store(true, Ordering::Release);
                    }
                }
            })?;
        Ok(Self {
            latest: initial,
            pending,
            wake: Some(wake),
            failed,
            retry,
            retry_cooldown,
            last_attempt: Instant::now(),
            stopped_message,
            worker: Some(worker),
        })
    }

    /// Replaces the baseline without enqueueing a disk write.
    pub(super) fn set_latest(&mut self, latest: Option<T>) {
        self.latest = latest;
    }

    /// Queues `value` only when it differs from the latest state or the retry cooldown elapsed.
    pub(super) fn queue(&mut self, value: T) -> io::Result<()>
    where
        T: PartialEq,
    {
        self.queue_by(
            &value,
            |latest, candidate| latest == candidate,
            Clone::clone,
        )
    }

    /// Coalesced enqueue using a borrowed query so unchanged frames avoid cloning/allocating.
    pub(super) fn queue_by<Q: ?Sized>(
        &mut self,
        query: &Q,
        same: impl FnOnce(&T, &Q) -> bool,
        to_owned: impl FnOnce(&Q) -> T,
    ) -> io::Result<()> {
        let retry_due = self.retry_cooldown.is_some_and(|cooldown| {
            self.retry.load(Ordering::Acquire) && self.last_attempt.elapsed() >= cooldown
        });
        if self
            .latest
            .as_ref()
            .is_some_and(|latest| same(latest, query))
            && !retry_due
        {
            return Ok(());
        }
        let owned = to_owned(query);
        let latest = owned.clone();
        self.retry.store(false, Ordering::Release);
        *self
            .pending
            .lock()
            .map_err(|_| io::Error::other(self.stopped_message))? = Some(owned);
        if let Some(wake) = &self.wake {
            match wake.try_send(()) {
                Ok(()) | Err(TrySendError::Full(())) => {}
                Err(TrySendError::Disconnected(())) => {
                    return Err(io::Error::other(self.stopped_message));
                }
            }
        }
        self.latest = Some(latest);
        self.last_attempt = Instant::now();
        Ok(())
    }

    /// Returns whether the background writer failed since the previous check, clearing the flag.
    pub(super) fn take_error(&self) -> bool {
        self.failed.swap(false, Ordering::Acquire)
    }
}

impl<T> Drop for CoalescingStore<T> {
    fn drop(&mut self) {
        self.wake.take();
        if let Some(worker) = self.worker.take() {
            let _joined = worker.join();
        }
    }
}
