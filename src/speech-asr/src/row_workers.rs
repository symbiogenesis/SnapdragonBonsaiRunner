//! Bounded, calling-thread-owned worker pool for aligned Rust ASR slices.
#![forbid(unsafe_code)]

use crate::threads::MAX_WORKERS;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use std::{cell::RefCell, io, sync::Mutex, thread::JoinHandle};

thread_local! {
    // Each calling serving thread owns its worker team across requests; dropping
    // that thread or switching worker count releases the previous team first.
    static TEAM: RefCell<Option<CachedTeam>> = const { RefCell::new(None) };
}

enum CachedTeam {
    Ready(OwnedTeam),
    Unavailable(usize),
}

impl CachedTeam {
    const fn worker_count(&self) -> usize {
        match self {
            Self::Ready(team) => team.workers.len(),
            Self::Unavailable(count) => *count,
        }
    }
}

struct OwnedTeam {
    pool: Option<ThreadPool>,
    workers: Vec<JoinHandle<()>>,
}

impl OwnedTeam {
    fn build(worker_count: usize) -> Result<Self, rayon::ThreadPoolBuildError> {
        let mut team = Self {
            pool: None,
            workers: Vec::with_capacity(worker_count),
        };
        let pool = ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .spawn_handler(|worker| {
                let handle = std::thread::Builder::new()
                    .name(format!("asr-row-{}", worker.index()))
                    .spawn(move || worker.run())?;
                team.workers.push(handle);
                Ok(())
            })
            .build()?;
        team.pool = Some(pool);
        Ok(team)
    }
}

impl Drop for OwnedTeam {
    fn drop(&mut self) {
        // Dropping the pool signals every worker thread to exit before joining.
        drop(self.pool.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn with_team<R>(
    worker_count: usize,
    total_len: usize,
    alignment: usize,
    action: impl FnOnce(&ThreadPool, usize) -> R,
) -> Option<R> {
    let worker_count = worker_count.clamp(1, MAX_WORKERS);
    let alignment = alignment.max(1);
    if worker_count <= 1 || total_len == 0 || rayon::current_thread_index().is_some() {
        return None;
    }
    let groups = total_len.div_ceil(alignment);
    if groups < 2 {
        return None;
    }
    let chunk_len = groups.div_ceil(worker_count) * alignment;
    TEAM.with(|slot| {
        let Ok(mut team) = slot.try_borrow_mut() else {
            return None;
        };
        if team
            .as_ref()
            .is_none_or(|cached| cached.worker_count() != worker_count)
        {
            // Release any prior team before constructing its replacement.
            *team = None;
            *team = Some(
                OwnedTeam::build(worker_count)
                    .map_or_else(|_| CachedTeam::Unavailable(worker_count), CachedTeam::Ready),
            );
        }
        let CachedTeam::Ready(owned) = team.as_ref()? else {
            return None;
        };
        let pool = owned.pool.as_ref()?;
        Some(action(pool, chunk_len))
    })
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
    let done = with_team(worker_count, out.len(), alignment, |pool, chunk_len| {
        pool.install(|| {
            out.par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(index, slice)| compute_slice(index * chunk_len, slice));
        });
    });
    if done.is_none() {
        compute_slice(0, out);
    }
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
    if let Some(result) = with_team(worker_count, out.len(), alignment, |pool, chunk_len| {
        let failure: Mutex<Option<(usize, io::Error)>> = Mutex::new(None);
        pool.install(|| {
            out.par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(index, slice)| {
                    let offset = index * chunk_len;
                    if let Err(error) = compute_slice(offset, slice) {
                        let mut guard = failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if guard.as_ref().is_none_or(|(prev, _)| offset < *prev) {
                            *guard = Some((offset, error));
                        }
                    }
                });
        });
        match failure
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            Some((_, error)) => Err(error),
            None => Ok(()),
        }
    }) {
        result
    } else {
        compute_slice(0, out)
    }
}

#[cfg(test)]
mod tests {
    use super::{CachedTeam, OwnedTeam, TEAM, run_aligned, try_run_aligned};
    use std::{
        collections::HashSet,
        io,
        sync::{Barrier, Mutex},
        thread::{self, ThreadId},
    };

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
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].0 + pair[0].1, pair[1].0);
            }
            assert_eq!(ranges.last().map(|(start, len)| start + len), Some(53));
        }
    }

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
        assert!(rayon::current_thread_index().is_none());
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
}
