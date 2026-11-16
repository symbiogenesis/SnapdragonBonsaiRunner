//! Spin-then-park worker pool using the calling thread as worker 0.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::io;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

const MAX_POOL_WORKERS: usize = 32;
const SPIN_ITERATIONS: usize = 512;

thread_local! {
    static IN_ROW_POOL: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone, Copy)]
struct ChunkDesc {
    offset: usize,
    ptr: *mut f32,
    len: usize,
}

impl Default for ChunkDesc {
    fn default() -> Self {
        Self {
            offset: 0,
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }
}

type DispatchFn = unsafe fn(*const (), usize, &mut [f32]);

#[derive(Clone, Copy)]
struct TaskPtr {
    call: DispatchFn,
    env: *const (),
}

impl Default for TaskPtr {
    fn default() -> Self {
        unsafe fn noop(_: *const (), _: usize, _: &mut [f32]) {}
        Self {
            call: noop,
            env: std::ptr::null(),
        }
    }
}

unsafe fn call_closure<F: Fn(usize, &mut [f32]) + Sync>(
    env: *const (),
    offset: usize,
    slice: &mut [f32],
) {
    // SAFETY: `env` points to the caller-owned `F` which remains live until
    // `ScopeGuard::drop` confirms `workers_remaining == 0`.
    let func = unsafe { &*env.cast::<F>() };
    func(offset, slice);
}

struct Shared {
    worker_count: usize,
    generation: AtomicUsize,
    next_chunk: AtomicUsize,
    num_chunks: AtomicUsize,
    workers_remaining: AtomicUsize,
    parked_count: AtomicUsize,
    shutdown: AtomicBool,
    park_mutex: Mutex<()>,
    park_cv: Condvar,
    done_mutex: Mutex<()>,
    done_cv: Condvar,
    dispatch_mutex: Mutex<()>,
    panic_slot: Mutex<Option<Box<dyn Any + Send + 'static>>>,
    task: UnsafeCell<TaskPtr>,
    chunks: [UnsafeCell<ChunkDesc>; MAX_POOL_WORKERS],
}

// SAFETY: `Shared` synchronizes chunk publication via `generation` (SeqCst/Release)
// and exclusive atomic chunk claims via `next_chunk`; each chunk descriptor is
// claimed by at most one thread over disjoint mutable sub-slices, and `ScopeGuard`
// keeps all referenced stack data alive until `workers_remaining == 0`.
unsafe impl Send for Shared {}

// SAFETY: See `Send` contract above; all concurrent mutations happen through
// atomics, mutexes, or disjoint uniquely claimed chunk slots.
unsafe impl Sync for Shared {}

impl Shared {
    fn new(worker_count: usize) -> Self {
        Self {
            worker_count,
            generation: AtomicUsize::new(0),
            next_chunk: AtomicUsize::new(0),
            num_chunks: AtomicUsize::new(0),
            workers_remaining: AtomicUsize::new(0),
            parked_count: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            park_mutex: Mutex::new(()),
            park_cv: Condvar::new(),
            done_mutex: Mutex::new(()),
            done_cv: Condvar::new(),
            dispatch_mutex: Mutex::new(()),
            panic_slot: Mutex::new(None),
            task: UnsafeCell::new(TaskPtr::default()),
            chunks: std::array::from_fn(|_| UnsafeCell::new(ChunkDesc::default())),
        }
    }

    fn drain_chunks(&self, task: TaskPtr) {
        let count = self.num_chunks.load(Ordering::Acquire);
        loop {
            let idx = self.next_chunk.fetch_add(1, Ordering::AcqRel);
            if idx >= count {
                break;
            }
            // SAFETY: `idx < count <= MAX_POOL_WORKERS`, initialized before the
            // generation bump, and exclusively claimed by this thread via `fetch_add`.
            let desc = unsafe { *self.chunks[idx].get() };
            if desc.len == 0 {
                continue;
            }
            // SAFETY: `desc.ptr..desc.ptr + desc.len` is a valid disjoint mutable
            // chunk from the caller's slice held live through `ScopeGuard::drop`.
            let slice = unsafe { std::slice::from_raw_parts_mut(desc.ptr, desc.len) };
            // SAFETY: `task.call` was paired with `&F` (`task.env`), kept live
            // and immutable across all workers for the active scope.
            unsafe { (task.call)(task.env, desc.offset, slice) };
        }
    }
}

fn worker_loop(shared: &Shared) {
    IN_ROW_POOL.with(|flag| flag.set(true));
    let mut last_gen = 0_usize;
    loop {
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
        let mut current_gen = shared.generation.load(Ordering::Acquire);
        if current_gen == last_gen {
            for _ in 0..SPIN_ITERATIONS {
                if shared.shutdown.load(Ordering::Acquire) {
                    return;
                }
                current_gen = shared.generation.load(Ordering::Acquire);
                if current_gen != last_gen {
                    break;
                }
                std::hint::spin_loop();
            }
        }
        if current_gen == last_gen {
            shared.parked_count.fetch_add(1, Ordering::SeqCst);
            let mut guard = shared
                .park_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while !shared.shutdown.load(Ordering::SeqCst) {
                current_gen = shared.generation.load(Ordering::SeqCst);
                if current_gen != last_gen {
                    break;
                }
                guard = shared
                    .park_cv
                    .wait(guard)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            shared.parked_count.fetch_sub(1, Ordering::SeqCst);
            drop(guard);
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
        }
        last_gen = current_gen;
        // SAFETY: `task` was published prior to incrementing `generation` and
        // stays valid until `workers_remaining` drops to zero below.
        let task = unsafe { *shared.task.get() };
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            shared.drain_chunks(task);
        }));
        if let Err(payload) = result {
            let mut panic_guard = shared
                .panic_slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if panic_guard.is_none() {
                *panic_guard = Some(payload);
            }
        }
        if shared.workers_remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
            let _done_guard = shared
                .done_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            shared.done_cv.notify_all();
        }
    }
}

struct ScopeGuard<'a> {
    shared: &'a Shared,
    previous_in_pool: bool,
    chunks_written: usize,
}

impl Drop for ScopeGuard<'_> {
    fn drop(&mut self) {
        if self.shared.workers_remaining.load(Ordering::Acquire) != 0 {
            for _ in 0..SPIN_ITERATIONS {
                if self.shared.workers_remaining.load(Ordering::Acquire) == 0 {
                    break;
                }
                std::hint::spin_loop();
            }
            if self.shared.workers_remaining.load(Ordering::Acquire) != 0 {
                let mut guard = self
                    .shared
                    .done_mutex
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                while self.shared.workers_remaining.load(Ordering::Acquire) != 0 {
                    guard = self
                        .shared
                        .done_cv
                        .wait(guard)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }
        }
        self.shared.num_chunks.store(0, Ordering::Release);
        for slot in &self.shared.chunks[..self.chunks_written] {
            // SAFETY: All background workers exited this generation (`workers_remaining == 0`).
            unsafe { *slot.get() = ChunkDesc::default() };
        }
        // SAFETY: No worker can access `task` after reporting completion.
        unsafe { *self.shared.task.get() = TaskPtr::default() };
        IN_ROW_POOL.with(|flag| flag.set(self.previous_in_pool));
        if !std::thread::panicking() {
            let panic_payload = self
                .shared
                .panic_slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(payload) = panic_payload {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

pub(crate) struct NativeRowWorkerPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl NativeRowWorkerPool {
    pub(crate) fn new(worker_count: usize, name_prefix: &str) -> io::Result<Self> {
        let count = worker_count.clamp(1, MAX_POOL_WORKERS);
        let shared = Arc::new(Shared::new(count));
        let mut pool = Self {
            shared: Arc::clone(&shared),
            workers: Vec::with_capacity(count.saturating_sub(1)),
        };
        for idx in 1..count {
            let worker_shared = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("{name_prefix}-{idx}"))
                .spawn(move || worker_loop(&worker_shared))?;
            pool.workers.push(handle);
        }
        Ok(pool)
    }

    pub(crate) fn worker_count(&self) -> usize {
        self.shared.worker_count
    }

    pub(crate) fn run_aligned<F>(
        &self,
        out: &mut [f32],
        active_workers: usize,
        alignment: usize,
        compute: &F,
    ) where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        if out.is_empty() {
            return;
        }
        let nested = IN_ROW_POOL.with(Cell::get);
        let bg_count = self.workers.len();
        let active = active_workers.clamp(1, self.shared.worker_count);
        let align = alignment.max(1).min(out.len());
        let groups = out.len().div_ceil(align);
        if nested || bg_count == 0 || active <= 1 || groups < 2 {
            compute(0, out);
            return;
        }

        let target_chunks = active.min(groups).min(MAX_POOL_WORKERS);
        let _dispatch = self
            .shared
            .dispatch_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let chunk_elems = groups.div_ceil(target_chunks) * align;
        let mut chunk_count = 0_usize;
        let mut offset = 0_usize;
        for sub in out.chunks_mut(chunk_elems) {
            // SAFETY: Coordinator owns `&Shared` exclusively for the generation
            // before bumping `generation` (`workers_remaining` is 0).
            unsafe {
                *self.shared.chunks[chunk_count].get() = ChunkDesc {
                    offset,
                    ptr: sub.as_mut_ptr(),
                    len: sub.len(),
                };
            }
            offset += sub.len();
            chunk_count += 1;
        }

        let previous_in_pool = IN_ROW_POOL.with(|flag| {
            let prev = flag.get();
            flag.set(true);
            prev
        });

        let task = TaskPtr {
            call: call_closure::<F>,
            env: (&raw const *compute).cast::<()>(),
        };
        // SAFETY: Previous generation completed (`workers_remaining == 0`).
        unsafe { *self.shared.task.get() = task };
        self.shared.num_chunks.store(chunk_count, Ordering::Release);
        self.shared.next_chunk.store(0, Ordering::Release);
        self.shared
            .workers_remaining
            .store(bg_count, Ordering::SeqCst);

        let _guard = ScopeGuard {
            shared: &self.shared,
            previous_in_pool,
            chunks_written: chunk_count,
        };

        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        if self.shared.parked_count.load(Ordering::SeqCst) > 0 {
            let _park = self
                .shared
                .park_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.shared.park_cv.notify_all();
        }

        self.shared.drain_chunks(task);
    }
}

impl Drop for NativeRowWorkerPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        {
            let _guard = self
                .shared
                .park_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.shared.park_cv.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}
