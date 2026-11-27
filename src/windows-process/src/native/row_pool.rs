//! Spin-then-park worker pool using the calling thread as worker 0.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::io;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn wait_unpoisoned<'a, T>(cv: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    cv.wait(guard)
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

const MAX_POOL_WORKERS: usize = 32;
const SPIN_ITERATIONS: usize = 12_288;
const GEN_SHIFT: u32 = 8;
const DIRECT_FLAG: usize = 1 << (GEN_SHIFT - 1);
const SLOT_MASK: usize = DIRECT_FLAG - 1;
const GEN_MASK: usize = usize::MAX >> GEN_SHIFT;

const _: () = assert!(MAX_POOL_WORKERS <= SLOT_MASK);

#[repr(align(64))]
struct CachePadded<T>(T);

#[repr(align(128))]
struct WorkerWakeSlot {
    wake_gen: AtomicUsize,
    parked: AtomicBool,
    park_mutex: Mutex<()>,
    park_cv: Condvar,
}

impl WorkerWakeSlot {
    fn new() -> Self {
        Self {
            wake_gen: AtomicUsize::new(0),
            parked: AtomicBool::new(false),
            park_mutex: Mutex::new(()),
            park_cv: Condvar::new(),
        }
    }
}

thread_local! {
    static IN_ROW_POOL: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone, Copy)]
struct ChunkDesc {
    offset: usize,
    ptr: *mut u8,
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

type DispatchFn = unsafe fn(*const (), usize, *mut u8, usize);

#[derive(Clone, Copy)]
struct TaskPtr {
    call: DispatchFn,
    env: *const (),
}

impl Default for TaskPtr {
    fn default() -> Self {
        unsafe fn noop(_: *const (), _: usize, _: *mut u8, _: usize) {}
        Self {
            call: noop,
            env: std::ptr::null(),
        }
    }
}

unsafe fn call_closure<T: Send, F: Fn(usize, &mut [T]) + Sync>(
    env: *const (),
    offset: usize,
    ptr: *mut u8,
    len: usize,
) {
    // SAFETY: `env` points to the caller-owned `F` which remains live until
    // `ScopeGuard::drop` confirms every active worker updated `done_gen`.
    let func = unsafe { &*env.cast::<F>() };
    // SAFETY: `ptr..ptr + len` is a valid, properly aligned, disjoint mutable
    // sub-slice of `&mut [T]` held live through `ScopeGuard::drop`.
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<T>(), len) };
    func(offset, slice);
}

struct Shared {
    worker_count: usize,
    num_chunks: AtomicUsize,
    shutdown: AtomicBool,
    has_panic: AtomicBool,
    next_gen: CachePadded<AtomicUsize>,
    dispatch_state: CachePadded<AtomicUsize>,
    next_chunk: CachePadded<AtomicUsize>,
    done_gen: [CachePadded<AtomicUsize>; MAX_POOL_WORKERS],
    done_waiting: CachePadded<AtomicBool>,
    wake_slots: [WorkerWakeSlot; MAX_POOL_WORKERS],
    done_mutex: Mutex<()>,
    done_cv: Condvar,
    dispatch_mutex: Mutex<()>,
    panic_slot: Mutex<Option<Box<dyn Any + Send + 'static>>>,
    task: UnsafeCell<TaskPtr>,
    chunks: [UnsafeCell<ChunkDesc>; MAX_POOL_WORKERS],
}

// SAFETY: `Shared` synchronizes chunk publication via each active worker's
// `wake_slots[worker_id].wake_gen` (SeqCst/Acquire) and per-worker direct slot
// assignment or exclusive atomic chunk claims; each chunk descriptor is claimed
// by at most one thread over disjoint mutable sub-slices, and `ScopeGuard` keeps
// all referenced stack data alive until every active worker publishes
// completion in `done_gen`.
unsafe impl Send for Shared {}

// SAFETY: See `Send` contract above; all concurrent mutations happen through
// atomics, mutexes, or disjoint uniquely claimed chunk slots.
unsafe impl Sync for Shared {}

impl Shared {
    fn new(worker_count: usize) -> Self {
        Self {
            worker_count,
            num_chunks: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            has_panic: AtomicBool::new(false),
            next_gen: CachePadded(AtomicUsize::new(0)),
            dispatch_state: CachePadded(AtomicUsize::new(0)),
            next_chunk: CachePadded(AtomicUsize::new(0)),
            done_gen: std::array::from_fn(|_| CachePadded(AtomicUsize::new(0))),
            done_waiting: CachePadded(AtomicBool::new(false)),
            wake_slots: std::array::from_fn(|_| WorkerWakeSlot::new()),
            done_mutex: Mutex::new(()),
            done_cv: Condvar::new(),
            dispatch_mutex: Mutex::new(()),
            panic_slot: Mutex::new(None),
            task: UnsafeCell::new(TaskPtr::default()),
            chunks: std::array::from_fn(|_| UnsafeCell::new(ChunkDesc::default())),
        }
    }

    #[cfg(test)]
    fn parked_worker_count(&self) -> usize {
        self.wake_slots[1..self.worker_count]
            .iter()
            .filter(|slot| slot.parked.load(Ordering::Acquire))
            .count()
    }

    fn try_claim_slot_for_gen(&self, gen_id: usize, worker_id: usize) -> ClaimedWork {
        let mut state = self.dispatch_state.0.load(Ordering::Acquire);
        loop {
            let state_gen = state >> GEN_SHIFT;
            if state_gen != gen_id {
                return ClaimedWork::None;
            }
            if (state & DIRECT_FLAG) != 0 {
                let chunk_count = state & SLOT_MASK;
                return if worker_id < chunk_count {
                    ClaimedWork::Direct
                } else {
                    ClaimedWork::None
                };
            }
            let remaining = state & SLOT_MASK;
            if remaining == 0 {
                return ClaimedWork::None;
            }
            let next = (gen_id << GEN_SHIFT) | (remaining - 1);
            match self.dispatch_state.0.compare_exchange_weak(
                state,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return ClaimedWork::Dynamic,
                Err(actual) => state = actual,
            }
        }
    }

    #[inline]
    fn run_chunk(&self, task: TaskPtr, idx: usize) {
        // SAFETY: `idx < num_chunks <= MAX_POOL_WORKERS`, initialized before the
        // generation bump, and exclusively claimed by this thread for the generation.
        let desc = unsafe { *self.chunks[idx].get() };
        if desc.len == 0 {
            return;
        }
        // SAFETY: `task.call` was paired with `&F` (`task.env`) and `desc.ptr..desc.ptr + desc.len`
        // from `&mut [T]`, kept live and immutable across all workers for the active scope.
        unsafe { (task.call)(task.env, desc.offset, desc.ptr, desc.len) };
    }

    fn drain_chunks(&self, task: TaskPtr) {
        let count = self.num_chunks.load(Ordering::Acquire);
        loop {
            let idx = self.next_chunk.0.fetch_add(1, Ordering::AcqRel);
            if idx >= count {
                break;
            }
            self.run_chunk(task, idx);
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ClaimedWork {
    None,
    Direct,
    Dynamic,
}

fn worker_loop(shared: &Shared, worker_id: usize) {
    IN_ROW_POOL.with(|flag| flag.set(true));
    let slot = &shared.wake_slots[worker_id];
    let mut last_gen = 0_usize;
    loop {
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
        let mut current_gen = slot.wake_gen.load(Ordering::Acquire);
        if current_gen == last_gen {
            for _ in 0..SPIN_ITERATIONS {
                if shared.shutdown.load(Ordering::Relaxed) {
                    return;
                }
                if slot.wake_gen.load(Ordering::Relaxed) != last_gen {
                    current_gen = slot.wake_gen.load(Ordering::Acquire);
                    break;
                }
                std::hint::spin_loop();
            }
        }
        if current_gen == last_gen {
            let mut guard = lock_unpoisoned(&slot.park_mutex);
            slot.parked.store(true, Ordering::SeqCst);
            while !shared.shutdown.load(Ordering::SeqCst) {
                current_gen = slot.wake_gen.load(Ordering::SeqCst);
                if current_gen != last_gen {
                    break;
                }
                guard = wait_unpoisoned(&slot.park_cv, guard);
            }
            slot.parked.store(false, Ordering::Release);
            drop(guard);
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
        }
        last_gen = current_gen;
        let claim = shared.try_claim_slot_for_gen(current_gen, worker_id);
        if claim == ClaimedWork::None {
            continue;
        }
        // SAFETY: `task` was published prior to incrementing `wake_gen` and
        // stays valid until every active worker publishes `done_gen == current_gen`.
        let task = unsafe { *shared.task.get() };
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| match claim {
            ClaimedWork::Direct => shared.run_chunk(task, worker_id),
            ClaimedWork::Dynamic => shared.drain_chunks(task),
            ClaimedWork::None => {}
        }));
        if let Err(payload) = result {
            let mut panic_guard = lock_unpoisoned(&shared.panic_slot);
            if panic_guard.is_none() {
                *panic_guard = Some(payload);
            }
            shared.has_panic.store(true, Ordering::Release);
        }
        shared.done_gen[worker_id]
            .0
            .store(current_gen, Ordering::Release);
        std::sync::atomic::fence(Ordering::SeqCst);
        if shared.done_waiting.0.load(Ordering::Relaxed) {
            let _done_guard = lock_unpoisoned(&shared.done_mutex);
            shared.done_cv.notify_all();
        }
    }
}

struct ScopeGuard<'a> {
    shared: &'a Shared,
    previous_in_pool: bool,
    gen_id: usize,
    active_workers: usize,
}

impl Drop for ScopeGuard<'_> {
    fn drop(&mut self) {
        let mut idx = 1_usize;
        for _ in 0..SPIN_ITERATIONS {
            while idx < self.active_workers
                && self.shared.done_gen[idx].0.load(Ordering::Acquire) == self.gen_id
            {
                idx += 1;
            }
            if idx == self.active_workers {
                break;
            }
            std::hint::spin_loop();
        }
        if idx < self.active_workers {
            let mut guard = lock_unpoisoned(&self.shared.done_mutex);
            self.shared.done_waiting.0.store(true, Ordering::SeqCst);
            while idx < self.active_workers {
                if self.shared.done_gen[idx].0.load(Ordering::SeqCst) == self.gen_id {
                    idx += 1;
                } else {
                    guard = wait_unpoisoned(&self.shared.done_cv, guard);
                }
            }
            self.shared.done_waiting.0.store(false, Ordering::Release);
        }
        IN_ROW_POOL.with(|flag| flag.set(self.previous_in_pool));
        if self.shared.has_panic.load(Ordering::Acquire) {
            self.shared.has_panic.store(false, Ordering::Release);
            let panic_payload = lock_unpoisoned(&self.shared.panic_slot).take();
            if !std::thread::panicking()
                && let Some(payload) = panic_payload
            {
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
                .spawn(move || worker_loop(&worker_shared, idx))?;
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
        self.run_slice_aligned(out, active_workers, alignment, compute);
    }

    pub(crate) fn run_slice_aligned<T: Send, F>(
        &self,
        out: &mut [T],
        active_workers: usize,
        alignment: usize,
        compute: &F,
    ) where
        F: Fn(usize, &mut [T]) + Sync,
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
        let _dispatch = lock_unpoisoned(&self.shared.dispatch_mutex);
        let base_groups = groups / target_chunks;
        let extra_chunks = groups % target_chunks;
        let mut rest = out;
        let mut chunk_count = 0_usize;
        let mut offset = 0_usize;
        for idx in 0..target_chunks {
            let chunk_groups = base_groups + usize::from(idx < extra_chunks);
            let take = (chunk_groups * align).min(rest.len());
            if take == 0 {
                break;
            }
            let (sub, tail) = rest.split_at_mut(take);
            rest = tail;
            // SAFETY: Coordinator owns `&Shared` exclusively for the generation
            // before bumping `generation` (all previous workers completed).
            unsafe {
                *self.shared.chunks[chunk_count].get() = ChunkDesc {
                    offset,
                    ptr: sub.as_mut_ptr().cast::<u8>(),
                    len: sub.len(),
                };
            }
            offset += take;
            chunk_count += 1;
        }

        let previous_in_pool = IN_ROW_POOL.with(|flag| {
            let prev = flag.get();
            flag.set(true);
            prev
        });

        let active_bg = chunk_count.saturating_sub(1).min(bg_count);
        let direct = chunk_count <= self.shared.worker_count;
        let next_gen = match self
            .shared
            .next_gen
            .0
            .load(Ordering::Relaxed)
            .wrapping_add(1)
            & GEN_MASK
        {
            0 => 1,
            g => g,
        };
        self.shared.next_gen.0.store(next_gen, Ordering::Relaxed);
        let task = TaskPtr {
            call: call_closure::<T, F>,
            env: (&raw const *compute).cast::<()>(),
        };
        // SAFETY: Previous generation completed across all workers.
        unsafe { *self.shared.task.get() = task };
        let dispatch_word = if direct {
            (next_gen << GEN_SHIFT) | DIRECT_FLAG | chunk_count
        } else {
            self.shared.num_chunks.store(chunk_count, Ordering::Release);
            self.shared.next_chunk.0.store(0, Ordering::Release);
            (next_gen << GEN_SHIFT) | active_bg
        };
        self.shared
            .dispatch_state
            .0
            .store(dispatch_word, Ordering::Release);

        let _guard = ScopeGuard {
            shared: &self.shared,
            previous_in_pool,
            gen_id: next_gen,
            active_workers: chunk_count,
        };

        let wake_end = chunk_count.min(self.shared.worker_count);
        for slot in &self.shared.wake_slots[1..wake_end] {
            slot.wake_gen.store(next_gen, Ordering::SeqCst);
            if slot.parked.load(Ordering::SeqCst) {
                let _park = lock_unpoisoned(&slot.park_mutex);
                slot.park_cv.notify_one();
            }
        }

        if direct {
            self.shared.run_chunk(task, 0);
        } else {
            self.shared.drain_chunks(task);
        }
    }

    pub(crate) fn try_run_aligned<F>(
        &self,
        out: &mut [f32],
        active_workers: usize,
        alignment: usize,
        compute: &F,
    ) -> io::Result<()>
    where
        F: Fn(usize, &mut [f32]) -> io::Result<()> + Sync,
    {
        let failure: Mutex<Option<(usize, io::Error)>> = Mutex::new(None);
        self.run_aligned(out, active_workers, alignment, &|offset, slice| {
            if let Err(error) = compute(offset, slice) {
                let mut guard = lock_unpoisoned(&failure);
                if guard.as_ref().is_none_or(|(prev, _)| offset < *prev) {
                    *guard = Some((offset, error));
                }
            }
        });
        match failure
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            Some((_, error)) => Err(error),
            None => Ok(()),
        }
    }
}

impl Drop for NativeRowWorkerPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        for slot in &self.shared.wake_slots[1..self.shared.worker_count] {
            if slot.parked.load(Ordering::SeqCst) {
                let _guard = lock_unpoisoned(&slot.park_mutex);
                slot.park_cv.notify_one();
            }
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_hot_atomics_are_cache_line_padded() {
        assert_eq!(align_of::<CachePadded<AtomicUsize>>(), 64);
        assert_eq!(align_of::<CachePadded<AtomicBool>>(), 64);
        assert_eq!(align_of::<WorkerWakeSlot>(), 128);
    }

    #[test]
    fn partial_active_workers_and_park_wake_cycles_complete_accurately() -> io::Result<()> {
        let pool = NativeRowWorkerPool::new(8, "test-partial-row")?;
        let mut out = vec![0.0_f32; 128];
        for active in [1_usize, 2, 3, 5, 8, 2, 4] {
            out.fill(0.0);
            pool.run_aligned(&mut out, active, 4, &|offset, slice| {
                for (i, slot) in slice.iter_mut().enumerate() {
                    *slot = u16::try_from(offset + i + active).map_or(0.0, f32::from);
                }
            });
            for (idx, &val) in out.iter().enumerate() {
                let expected = u16::try_from(idx + active).map_or(0.0, f32::from);
                assert_eq!(val.to_bits(), expected.to_bits());
            }
        }
        // Let workers park, then wake only a subset.
        let park_deadline = std::time::Instant::now() + std::time::Duration::from_millis(200);
        while pool.shared.parked_worker_count() < 7 && std::time::Instant::now() < park_deadline {
            std::thread::yield_now();
        }
        out.fill(0.0);
        pool.run_aligned(&mut out, 2, 8, &|offset, slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = u16::try_from(offset + i + 10).map_or(0.0, f32::from);
            }
        });
        for (idx, &val) in out.iter().enumerate() {
            let expected = u16::try_from(idx + 10).map_or(0.0, f32::from);
            assert_eq!(val.to_bits(), expected.to_bits());
        }
        Ok(())
    }

    #[test]
    fn worker_panic_sets_has_panic_and_recovers_cleanly() -> io::Result<()> {
        let pool = NativeRowWorkerPool::new(4, "test-panic-row")?;
        let coordinator_id = std::thread::current().id();
        let mut out = vec![0.0_f32; 32];
        // Warm up workers so they are spinning, then force a background worker to run a chunk and panic.
        pool.run_aligned(&mut out, 4, 8, &|_, _| {});
        // Worker-only panic: coordinator returns normally into ScopeGuard::drop and resumes the worker panic.
        let worker_res = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let shared_panic = &pool.shared.has_panic;
            pool.run_aligned(&mut out, 4, 8, &|offset, _slice| {
                if std::thread::current().id() == coordinator_id {
                    let deadline =
                        std::time::Instant::now() + std::time::Duration::from_millis(200);
                    while !shared_panic.load(Ordering::Acquire)
                        && std::time::Instant::now() < deadline
                    {
                        std::thread::yield_now();
                    }
                } else {
                    assert!(
                        std::thread::current().id() == coordinator_id,
                        "deliberate worker panic at offset {offset}"
                    );
                }
            });
        }));
        assert!(worker_res.is_err());
        assert!(!pool.shared.has_panic.load(Ordering::Acquire));

        // Coordinator + worker panic: ScopeGuard::drop still clears has_panic/panic_slot when unwinding.
        let coord_res = std::panic::catch_unwind(AssertUnwindSafe(|| {
            pool.run_aligned(&mut out, 4, 8, &|offset, _slice| {
                assert!(offset == 0, "deliberate panic at offset {offset}");
            });
        }));
        assert!(coord_res.is_err());
        assert!(!pool.shared.has_panic.load(Ordering::Acquire));

        out.fill(0.0);
        pool.run_aligned(&mut out, 4, 8, &|offset, slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = u16::try_from(offset + i).map_or(0.0, f32::from);
            }
        });
        for (idx, &val) in out.iter().enumerate() {
            assert_eq!(
                val.to_bits(),
                u16::try_from(idx).map_or(0.0, f32::from).to_bits()
            );
        }
        Ok(())
    }

    #[test]
    fn direct_dispatch_runs_without_next_chunk_fetch_and_supports_run_slice_aligned()
    -> io::Result<()> {
        use std::collections::HashSet;

        let pool = NativeRowWorkerPool::new(4, "test-direct-row")?;
        let coordinator_id = std::thread::current().id();
        let mut out = vec![0_u64; 32];

        // Full-team aligned workload: active == worker_count == chunk_count (4).
        // Marker 0xDEAD_BEEF in next_chunk must remain untouched because direct-dispatch
        // bypasses CAS and next_chunk.fetch_add.
        pool.shared
            .next_chunk
            .0
            .store(0xDEAD_BEEF, Ordering::Release);
        let chunk_threads: Mutex<Vec<(usize, std::thread::ThreadId)>> = Mutex::new(Vec::new());
        pool.run_slice_aligned(&mut out, 4, 8, &|offset, slice| {
            lock_unpoisoned(&chunk_threads).push((offset, std::thread::current().id()));
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = u64::try_from(offset + i).unwrap_or(0) * 3 + 1;
            }
        });
        assert_eq!(
            pool.shared.next_chunk.0.load(Ordering::Acquire),
            0xDEAD_BEEF
        );
        assert_ne!(
            pool.shared.dispatch_state.0.load(Ordering::Acquire) & DIRECT_FLAG,
            0
        );
        let recorded = lock_unpoisoned(&chunk_threads);
        assert_eq!(recorded.len(), 4);
        assert!(
            recorded
                .iter()
                .any(|(offset, tid)| *offset == 0 && *tid == coordinator_id)
        );
        let unique_threads: HashSet<_> = recorded.iter().map(|(_, tid)| *tid).collect();
        assert_eq!(unique_threads.len(), 4);
        drop(recorded);
        for (idx, &val) in out.iter().enumerate() {
            assert_eq!(val, u64::try_from(idx).unwrap_or(0) * 3 + 1);
        }

        // Partial-team wake (`chunk_count < worker_count`) also uses direct-dispatch
        // without CAS or `next_chunk` atomic RMW.
        let mut partial = vec![0_u64; 16];
        let partial_threads: Mutex<Vec<(usize, std::thread::ThreadId)>> = Mutex::new(Vec::new());
        pool.run_slice_aligned(&mut partial, 2, 4, &|offset, slice| {
            lock_unpoisoned(&partial_threads).push((offset, std::thread::current().id()));
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = u64::try_from(offset + i).unwrap_or(0) + 100;
            }
        });
        let partial_state = pool.shared.dispatch_state.0.load(Ordering::Acquire);
        assert_ne!(partial_state & DIRECT_FLAG, 0);
        assert_eq!(partial_state & SLOT_MASK, 2);
        assert_eq!(
            pool.shared.next_chunk.0.load(Ordering::Acquire),
            0xDEAD_BEEF
        );
        let partial_recorded = lock_unpoisoned(&partial_threads);
        assert_eq!(partial_recorded.len(), 2);
        assert!(
            partial_recorded
                .iter()
                .any(|(offset, tid)| *offset == 0 && *tid == coordinator_id)
        );
        drop(partial_recorded);
        for (idx, &val) in partial.iter().enumerate() {
            assert_eq!(val, u64::try_from(idx).unwrap_or(0) + 100);
        }
        Ok(())
    }

    #[test]
    fn balanced_chunking_uses_all_18_workers_on_1024_elements_aligned_8() -> io::Result<()> {
        use std::collections::HashSet;

        let pool = NativeRowWorkerPool::new(18, "test-18w-balanced")?;
        let mut out = vec![0.0_f32; 1024];
        pool.shared
            .next_chunk
            .0
            .store(0xDEAD_BEEF, Ordering::Release);
        let chunks: Mutex<Vec<(usize, usize, std::thread::ThreadId)>> = Mutex::new(Vec::new());
        pool.run_aligned(&mut out, 18, 8, &|offset, slice| {
            lock_unpoisoned(&chunks).push((offset, slice.len(), std::thread::current().id()));
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = u16::try_from(offset + i).map_or(0.0, f32::from);
            }
        });

        let state = pool.shared.dispatch_state.0.load(Ordering::Acquire);
        assert_ne!(state & DIRECT_FLAG, 0);
        assert_eq!(state & SLOT_MASK, 18);
        assert_eq!(
            pool.shared.next_chunk.0.load(Ordering::Acquire),
            0xDEAD_BEEF
        );

        let mut recorded = lock_unpoisoned(&chunks).clone();
        recorded.sort_by_key(|(offset, _, _)| *offset);
        assert_eq!(recorded.len(), 18);
        let unique_threads: HashSet<_> = recorded.iter().map(|(_, _, tid)| *tid).collect();
        assert_eq!(unique_threads.len(), 18);

        let mut expected_offset = 0_usize;
        for (idx, &(offset, len, _)) in recorded.iter().enumerate() {
            assert_eq!(offset, expected_offset);
            let expected_len = if idx < 2 { 64 } else { 56 };
            assert_eq!(len, expected_len);
            expected_offset += len;
        }
        assert_eq!(expected_offset, 1024);
        for (idx, &val) in out.iter().enumerate() {
            assert_eq!(
                val.to_bits(),
                u16::try_from(idx).map_or(0.0, f32::from).to_bits()
            );
        }
        Ok(())
    }

    #[test]
    fn partial_team_steady_state_keeps_idle_workers_parked() -> io::Result<()> {
        let pool = NativeRowWorkerPool::new(16, "test-steady-park")?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
        while pool.shared.parked_worker_count() < 15 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(pool.shared.parked_worker_count(), 15);

        let mut out = vec![0_usize; 96];
        for iter in 0..64_usize {
            out.fill(0);
            pool.run_slice_aligned(&mut out, 6, 16, &|offset, slice| {
                for (i, slot) in slice.iter_mut().enumerate() {
                    *slot = offset + i + iter;
                }
            });
            for (idx, &val) in out.iter().enumerate() {
                assert_eq!(val, idx + iter);
            }
            for w in 6..16 {
                assert_eq!(
                    pool.shared.wake_slots[w].wake_gen.load(Ordering::Acquire),
                    0
                );
                assert!(pool.shared.wake_slots[w].parked.load(Ordering::Acquire));
            }
        }
        Ok(())
    }

    #[test]
    fn rapid_active_worker_oscillation_completes_accurately() -> io::Result<()> {
        use std::collections::HashSet;

        let pool = NativeRowWorkerPool::new(16, "test-oscillate")?;
        let mut out = vec![0_usize; 256];
        let sequence = [1_usize, 6, 8, 16, 2, 6];
        for round in 0..4_usize {
            if round == 2 {
                let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
                while pool.shared.parked_worker_count() < 15 && std::time::Instant::now() < deadline
                {
                    std::thread::yield_now();
                }
            }
            for (step, &active) in sequence.iter().enumerate() {
                out.fill(0);
                let threads: Mutex<HashSet<std::thread::ThreadId>> = Mutex::new(HashSet::new());
                pool.run_slice_aligned(&mut out, active, 16, &|offset, slice| {
                    lock_unpoisoned(&threads).insert(std::thread::current().id());
                    for (i, slot) in slice.iter_mut().enumerate() {
                        *slot = offset + i + active + step;
                    }
                });
                assert_eq!(lock_unpoisoned(&threads).len(), active);
                for (idx, &val) in out.iter().enumerate() {
                    assert_eq!(val, idx + active + step);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn partial_team_panic_propagates_and_leaves_idle_workers_parked() -> io::Result<()> {
        let pool = NativeRowWorkerPool::new(8, "test-partial-panic")?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
        while pool.shared.parked_worker_count() < 7 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(pool.shared.parked_worker_count(), 7);

        let coordinator_id = std::thread::current().id();
        let mut out = vec![0.0_f32; 48];
        let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let shared_panic = &pool.shared.has_panic;
            pool.run_aligned(&mut out, 3, 16, &|offset, _slice| {
                if std::thread::current().id() == coordinator_id {
                    let wait_deadline =
                        std::time::Instant::now() + std::time::Duration::from_millis(200);
                    while !shared_panic.load(Ordering::Acquire)
                        && std::time::Instant::now() < wait_deadline
                    {
                        std::thread::yield_now();
                    }
                } else {
                    assert!(
                        std::thread::current().id() == coordinator_id,
                        "deliberate partial worker panic at {offset}"
                    );
                }
            });
        }));
        assert!(res.is_err());
        assert!(!pool.shared.has_panic.load(Ordering::Acquire));
        for w in 3..8 {
            assert_eq!(
                pool.shared.wake_slots[w].wake_gen.load(Ordering::Acquire),
                0
            );
            assert!(pool.shared.wake_slots[w].parked.load(Ordering::Acquire));
        }
        Ok(())
    }

    #[test]
    fn shutdown_with_mixed_parked_and_spinning_workers_joins_cleanly() -> io::Result<()> {
        let pool = NativeRowWorkerPool::new(12, "test-mixed-drop")?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
        while pool.shared.parked_worker_count() < 11 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(pool.shared.parked_worker_count(), 11);

        let mut out = vec![0_usize; 64];
        pool.run_slice_aligned(&mut out, 4, 16, &|offset, slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = offset + i;
            }
        });
        for w in 4..12 {
            assert!(pool.shared.wake_slots[w].parked.load(Ordering::Acquire));
        }
        drop(pool);
        Ok(())
    }
}
