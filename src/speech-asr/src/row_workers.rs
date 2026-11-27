//! Bounded, calling-thread-owned worker pool for aligned Rust ASR slices.
#![forbid(unsafe_code)]

use crate::threads::MAX_WORKERS;
#[cfg(windows)]
use bonsai_windows_process::RowWorkerPool;
#[cfg(not(windows))]
use std::sync::Mutex;
use std::{cell::RefCell, io};

thread_local! {
    // Each calling serving thread owns its worker team across requests; dropping
    // that thread or switching worker count releases the previous team first.
    #[cfg(windows)]
    static TEAM: RefCell<Option<CachedTeam>> = const { RefCell::new(None) };
    static IN_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(windows)]
enum CachedTeam {
    Ready(OwnedTeam),
    Unavailable(usize),
}

#[cfg(windows)]
impl CachedTeam {
    fn get_or_build(slot: &mut Option<Self>, worker_count: usize) -> Option<&OwnedTeam> {
        // Polonius lets a warm hit lend the cached team directly while a
        // rebuild ends the borrow to drop the old workers before spawning.
        if let Some(cached) = slot.as_ref() {
            match cached {
                Self::Ready(team) if team.worker_count() >= worker_count => return Some(team),
                Self::Unavailable(count) if *count == worker_count => return None,
                Self::Ready(_) | Self::Unavailable(_) => {}
            }
        }
        // Release any prior team only when growing beyond its worker count
        // or changing unavailable fallback requests, keeping warm batch pools across decode.
        *slot = None;
        match slot.insert(
            OwnedTeam::build(worker_count)
                .map_or_else(|_| Self::Unavailable(worker_count), Self::Ready),
        ) {
            Self::Ready(team) => Some(team),
            Self::Unavailable(_) => None,
        }
    }
}

#[cfg(windows)]
struct OwnedTeam {
    pool: RowWorkerPool,
}

#[cfg(windows)]
impl OwnedTeam {
    fn build(worker_count: usize) -> io::Result<Self> {
        Ok(Self {
            pool: RowWorkerPool::new(worker_count, "asr-row")?,
        })
    }

    fn worker_count(&self) -> usize {
        self.pool.worker_count()
    }
}

struct WorkerFlagGuard(bool);

impl WorkerFlagGuard {
    fn enter() -> Self {
        Self(IN_WORKER.with(|flag| flag.replace(true)))
    }
}

impl Drop for WorkerFlagGuard {
    fn drop(&mut self) {
        IN_WORKER.with(|flag| flag.set(self.0));
    }
}

#[cfg(windows)]
fn with_team<R>(
    worker_count: usize,
    total_len: usize,
    alignment: usize,
    action: impl FnOnce(&RowWorkerPool, usize) -> R,
) -> Option<R> {
    let worker_count = worker_count.clamp(1, MAX_WORKERS);
    let alignment = alignment.max(1);
    if worker_count <= 1 || total_len == 0 || IN_WORKER.with(std::cell::Cell::get) {
        return None;
    }
    let groups = total_len.div_ceil(alignment);
    if groups < 2 {
        return None;
    }
    let active_workers = worker_count.min(groups);
    TEAM.with(|slot| {
        let Ok(mut team) = slot.try_borrow_mut() else {
            return None;
        };
        let owned = CachedTeam::get_or_build(&mut team, worker_count)?;
        Some(action(&owned.pool, active_workers))
    })
}

#[cfg(not(windows))]
fn non_windows_chunk_len(worker_count: usize, total_len: usize, alignment: usize) -> Option<usize> {
    let worker_count = worker_count.clamp(1, MAX_WORKERS);
    let alignment = alignment.max(1);
    if worker_count <= 1 || total_len == 0 || IN_WORKER.with(std::cell::Cell::get) {
        return None;
    }
    let groups = total_len.div_ceil(alignment);
    if groups < 2 {
        return None;
    }
    let active_workers = worker_count.min(groups);
    Some(groups.div_ceil(active_workers) * alignment)
}

/// Execute `compute_slice(offset, chunk)` over `alignment`-aligned contiguous
/// slices of `out` for arbitrary `Send` element types, falling back to serial
/// execution on single-worker, small, nested, or unspawnable pools.
pub(super) fn run_slice_aligned<T: Send, F>(
    out: &mut [T],
    worker_count: usize,
    alignment: usize,
    compute_slice: &F,
) where
    F: Fn(usize, &mut [T]) + Sync,
{
    #[cfg(windows)]
    {
        let eff_align = alignment.max(1);
        let done = with_team(
            worker_count,
            out.len(),
            eff_align,
            |pool, active_workers| {
                let run = |offset: usize, slice: &mut [T]| {
                    let _guard = WorkerFlagGuard::enter();
                    compute_slice(offset, slice);
                };
                pool.run_slice_aligned(out, active_workers, eff_align, &run);
            },
        );
        if done.is_none() {
            compute_slice(0, out);
        }
    }
    #[cfg(not(windows))]
    {
        let Some(chunk_len) = non_windows_chunk_len(worker_count, out.len(), alignment) else {
            compute_slice(0, out);
            return;
        };
        std::thread::scope(|scope| {
            let mut chunks = out.chunks_mut(chunk_len).enumerate();
            let Some((first_index, first_slice)) = chunks.next() else {
                return;
            };
            for (index, slice) in chunks {
                scope.spawn(move || {
                    let _guard = WorkerFlagGuard::enter();
                    compute_slice(index * chunk_len, slice);
                });
            }
            let _guard = WorkerFlagGuard::enter();
            compute_slice(first_index * chunk_len, first_slice);
        });
    }
}

/// Execute `compute_slice(offset, chunk)` over `alignment`-aligned contiguous
/// slices of `out`, falling back to serial execution on single-worker, small,
/// nested, or unspawnable pools.
pub(super) fn run_aligned<F>(
    out: &mut [f32],
    worker_count: usize,
    alignment: usize,
    compute_slice: &F,
) where
    F: Fn(usize, &mut [f32]) + Sync,
{
    run_slice_aligned(out, worker_count, alignment, compute_slice);
}

/// Fallible counterpart to [`run_aligned`], joining every task and returning
/// the smallest-`offset` error deterministically when one or more slices fail.
pub(super) fn try_run_aligned<F>(
    out: &mut [f32],
    worker_count: usize,
    alignment: usize,
    compute_slice: &F,
) -> io::Result<()>
where
    F: Fn(usize, &mut [f32]) -> io::Result<()> + Sync,
{
    #[cfg(windows)]
    {
        let eff_align = alignment.max(1);
        if let Some(result) = with_team(
            worker_count,
            out.len(),
            eff_align,
            |pool, active_workers| {
                let run = |offset: usize, slice: &mut [f32]| {
                    let _guard = WorkerFlagGuard::enter();
                    compute_slice(offset, slice)
                };
                pool.try_run_aligned(out, active_workers, eff_align, &run)
            },
        ) {
            result
        } else {
            compute_slice(0, out)
        }
    }
    #[cfg(not(windows))]
    {
        let Some(chunk_len) = non_windows_chunk_len(worker_count, out.len(), alignment) else {
            return compute_slice(0, out);
        };
        let failure: Mutex<Option<(usize, io::Error)>> = Mutex::new(None);
        let record_error = |offset: usize, error: io::Error| {
            let mut guard = failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if guard.as_ref().is_none_or(|(prev, _)| offset < *prev) {
                *guard = Some((offset, error));
            }
        };
        std::thread::scope(|scope| {
            let mut chunks = out.chunks_mut(chunk_len).enumerate();
            let Some((first_index, first_slice)) = chunks.next() else {
                return;
            };
            for (index, slice) in chunks {
                let failure_ref = &record_error;
                scope.spawn(move || {
                    let _guard = WorkerFlagGuard::enter();
                    let offset = index * chunk_len;
                    if let Err(error) = compute_slice(offset, slice) {
                        failure_ref(offset, error);
                    }
                });
            }
            let _guard = WorkerFlagGuard::enter();
            let offset = first_index * chunk_len;
            if let Err(error) = compute_slice(offset, first_slice) {
                record_error(offset, error);
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

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use super::{CachedTeam, OwnedTeam, TEAM};
    use super::{run_aligned, try_run_aligned};
    #[cfg(windows)]
    use std::{
        collections::HashSet,
        sync::Barrier,
        thread::{self, ThreadId},
    };
    use std::{io, sync::Mutex};

    #[test]
    fn asr_workers_preserve_group_alignment_and_partial_tails() {
        for alignment in [0, 1, 4, 16, 64] {
            let ranges = Mutex::new(Vec::new());
            let mut output = [0.0; 53];
            run_aligned(&mut output, 3, alignment, &|offset, rows| {
                let effective_alignment = alignment.max(1);
                assert_eq!(offset % effective_alignment, 0);
                if offset + rows.len() < 53 {
                    assert_eq!(rows.len() % effective_alignment, 0);
                }
                ranges
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push((offset, rows.len()));
                for (local, value) in rows.iter_mut().enumerate() {
                    *value = f32::from(u16::try_from(offset + local).unwrap_or(0));
                }
            });
            assert_eq!(
                output,
                std::array::from_fn(|index| f32::from(u16::try_from(index).unwrap_or(0)))
            );
            let mut ranges = ranges
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(!ranges.is_empty() && ranges.len() <= 3);
            ranges.sort_unstable();
            assert_eq!(ranges[0].0, 0);
            for [left, right] in ranges.array_windows() {
                assert_eq!(left.0 + left.1, right.0);
            }
            assert_eq!(ranges.last().map(|(start, len)| start + len), Some(53));
        }
    }

    #[cfg(windows)]
    #[test]
    fn asr_workers_reuse_threads_handle_nesting_and_order_errors() -> io::Result<()> {
        fn team_ids() -> io::Result<HashSet<ThreadId>> {
            let barrier = Barrier::new(4);
            let ids = Mutex::new(HashSet::new());
            let mut rows = [0.0; 16];
            run_aligned(&mut rows, 4, 1, &|_, output| {
                ids.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(thread::current().id());
                barrier.wait();
                output.fill(1.0);
            });
            assert_eq!(rows, [1.0; 16]);
            ids.into_inner()
                .map_err(|_| io::Error::other("Thread IDs poisoned."))
        }

        let team = OwnedTeam::build(4).map_err(io::Error::other)?;
        TEAM.with(|slot| *slot.borrow_mut() = Some(CachedTeam::Ready(team)));
        let first = team_ids()?;
        assert_eq!(first.len(), 4);
        assert_eq!(first, team_ids()?);

        // Nested call inside worker pool falls back serially without deadlock.
        let mut outer = [0.0; 12];
        run_aligned(&mut outer, 4, 3, &|offset, slice| {
            let mut nested = [0.0; 8];
            run_aligned(&mut nested, 4, 2, &|inner_off, inner| {
                assert_eq!(inner_off, 0);
                inner.fill(5.0);
            });
            assert_eq!(nested, [5.0; 8]);
            for (i, v) in slice.iter_mut().enumerate() {
                *v = f32::from(u16::try_from(offset + i).unwrap_or(0));
            }
        });
        assert_eq!(
            outer,
            std::array::from_fn(|i| f32::from(u16::try_from(i).unwrap_or(0)))
        );

        // Deterministic smallest-offset error reporting across failing chunks.
        let mut buf = [0.0; 16];
        let err = try_run_aligned(&mut buf, 4, 4, &|offset, _| {
            if offset == 8 {
                Err(io::Error::new(io::ErrorKind::InvalidData, "chunk-8"))
            } else if offset == 4 {
                Err(io::Error::new(io::ErrorKind::Interrupted, "chunk-4"))
            } else {
                Ok(())
            }
        });
        assert_eq!(
            err.err().map(|e| e.kind()),
            Some(io::ErrorKind::Interrupted)
        );

        // Unavailable team falls back serially without failing.
        TEAM.with(|slot| *slot.borrow_mut() = Some(CachedTeam::Unavailable(4)));
        let mut fallback = [0.0; 9];
        run_aligned(&mut fallback, 4, 1, &|offset, slice| {
            assert_eq!(offset, 0);
            assert_eq!(slice.len(), 9);
            slice.fill(3.0);
        });
        assert_eq!(fallback, [3.0; 9]);
        TEAM.with(|slot| {
            assert!(matches!(*slot.borrow(), Some(CachedTeam::Unavailable(4))));
        });
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn asr_workers_retain_batch_pool_during_smaller_decode_slices() -> io::Result<()> {
        fn team_ids(worker_count: usize) -> io::Result<HashSet<ThreadId>> {
            let barrier = Barrier::new(worker_count);
            let ids = Mutex::new(HashSet::new());
            let mut rows = vec![0.0_f32; worker_count * 4];
            run_aligned(&mut rows, worker_count, 4, &|_, output| {
                ids.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(thread::current().id());
                barrier.wait();
                output.fill(1.0);
            });
            ids.into_inner()
                .map_err(|_| io::Error::other("Thread IDs poisoned."))
        }

        let team = OwnedTeam::build(8).map_err(io::Error::other)?;
        TEAM.with(|slot| *slot.borrow_mut() = Some(CachedTeam::Ready(team)));
        let initial = team_ids(8)?;
        assert_eq!(initial.len(), 8);

        let mut decode_buf = [0.0_f32; 16];
        let chunks = Mutex::new(Vec::new());
        run_aligned(&mut decode_buf, 4, 2, &|offset, slice| {
            chunks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((offset, slice.len(), thread::current().id()));
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = f32::from(u16::try_from(offset + i).unwrap_or(0));
            }
        });
        let chunks = chunks
            .into_inner()
            .map_err(|_| io::Error::other("Chunks poisoned."))?;
        assert!(!chunks.is_empty() && chunks.len() <= 4);
        assert!(chunks.iter().all(|(_, _, id)| initial.contains(id)));
        assert_eq!(team_ids(8)?, initial);
        Ok(())
    }
}
