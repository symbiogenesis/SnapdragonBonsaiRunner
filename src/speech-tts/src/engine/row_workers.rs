//! Bounded, reusable projection workers with synchronous borrowed row jobs.
#![forbid(unsafe_code)]

#[cfg(not(windows))]
use std::thread::JoinHandle;
use std::{cell::RefCell, io};

#[cfg(windows)]
use bonsai_windows_process::RowWorkerPool;
#[cfg(not(windows))]
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

thread_local! {
    // The serving thread owns its team through voice/model switches. Dropping
    // that thread or changing its parallel team size releases the old workers.
    static TEAM: RefCell<Option<CachedTeam>> = const { RefCell::new(None) };
}

enum CachedTeam {
    Ready(OwnedTeam),
    Unavailable(usize),
}

impl CachedTeam {
    fn get_or_build(slot: &mut Option<Self>, requested: usize) -> Option<&OwnedTeam> {
        // Polonius lets a warm hit lend the cached team directly while a
        // rebuild ends the borrow to drop the old workers before spawning.
        if let Some(cached) = slot.as_ref() {
            match cached {
                Self::Ready(team) if team.worker_count() >= requested => return Some(team),
                Self::Unavailable(count) if *count == requested => return None,
                Self::Ready(_) | Self::Unavailable(_) => {}
            }
        }
        // Release the old team only when growing beyond its worker count
        // or changing unavailable fallback requests, keeping warm larger pools.
        *slot = None;
        match slot.insert(
            OwnedTeam::build(requested).map_or_else(|_| Self::Unavailable(requested), Self::Ready),
        ) {
            Self::Ready(team) => Some(team),
            Self::Unavailable(_) => None,
        }
    }
}

struct OwnedTeam {
    #[cfg(windows)]
    pool: RowWorkerPool,
    #[cfg(windows)]
    worker_count: usize,
    #[cfg(not(windows))]
    pool: Option<ThreadPool>,
    #[cfg(not(windows))]
    workers: Vec<JoinHandle<()>>,
}

impl OwnedTeam {
    #[cfg(windows)]
    const fn worker_count(&self) -> usize {
        self.worker_count
    }

    #[cfg(not(windows))]
    const fn worker_count(&self) -> usize {
        self.workers.len()
    }
    #[cfg(windows)]
    fn build(worker_count: usize) -> io::Result<Self> {
        Ok(Self {
            pool: RowWorkerPool::new(worker_count, "tts-row")?,
            worker_count,
        })
    }

    #[cfg(not(windows))]
    fn build(worker_count: usize) -> io::Result<Self> {
        let mut team = Self {
            pool: None,
            workers: Vec::with_capacity(worker_count),
        };
        let pool = ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .spawn_handler(|worker| {
                let handle = std::thread::Builder::new()
                    .name(format!("tts-row-{}", worker.index()))
                    .spawn(move || worker.run())?;
                team.workers.push(handle);
                Ok(())
            })
            .build()
            .map_err(io::Error::other)?;
        team.pool = Some(pool);
        Ok(team)
    }
}

#[cfg(not(windows))]
impl Drop for OwnedTeam {
    fn drop(&mut self) {
        // Rayon signals termination when the pool drops. Join even a partial
        // construction before leaving or creating a replacement team.
        drop(self.pool.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

pub(super) fn run<F>(out: &mut [f32], worker_count: usize, compute_slice: &F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    run_aligned(out, worker_count, 1, compute_slice);
}

// Frame/channel kernels must keep their indivisible output groups together.
// Reuse the projection team instead of creating another pool or per-call threads.
#[expect(
    clippy::redundant_pub_crate,
    reason = "Share aligned jobs with the sibling codec without exposing the worker module outside this crate."
)]
pub(crate) fn run_aligned<F>(
    out: &mut [f32],
    worker_count: usize,
    alignment: usize,
    compute_slice: &F,
) where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let requested = worker_count.clamp(1, 18);
    #[cfg(not(windows))]
    if rayon::current_thread_index().is_some() {
        compute_slice(0, out);
        return;
    }
    if requested <= 1 || out.len() < 4 {
        compute_slice(0, out);
        return;
    }
    let alignment = alignment.max(1).min(out.len());
    let groups = out.len().div_ceil(alignment);
    if groups < 2 {
        compute_slice(0, out);
        return;
    }
    #[cfg(not(windows))]
    let chunk_rows = groups.div_ceil(requested.min(groups)) * alignment;

    let completed = TEAM.with(|slot| {
        let Ok(mut team) = slot.try_borrow_mut() else {
            return false;
        };
        let Some(team) = CachedTeam::get_or_build(&mut team, requested) else {
            return false;
        };
        #[cfg(windows)]
        {
            team.pool
                .run_aligned(out, requested, alignment, compute_slice);
        }
        #[cfg(not(windows))]
        {
            let Some(pool) = team.pool.as_ref() else {
                return false;
            };
            pool.install(|| {
                out.par_chunks_mut(chunk_rows)
                    .enumerate()
                    .for_each(|(chunk, rows)| compute_slice(chunk * chunk_rows, rows));
            });
        }
        true
    });
    if !completed {
        // Thread creation failure never removes speech availability, and no
        // partial job has been submitted before taking this serial fallback.
        compute_slice(0, out);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        io,
        sync::{Barrier, Mutex},
        thread::{self, ThreadId},
    };

    use super::{CachedTeam, OwnedTeam, TEAM, run, run_aligned};

    #[test]
    fn projection_workers_keep_frame_groups_whole_and_cover_partial_tail() {
        for alignment in [0, 1, 4, 16, 64] {
            let ranges = Mutex::new(Vec::new());
            let mut output = [0.0; 53];
            run_aligned(&mut output, 3, alignment, &|offset, rows| {
                let alignment = alignment.max(1);
                assert_eq!(offset % alignment, 0);
                if offset + rows.len() < 53 {
                    assert_eq!(rows.len() % alignment, 0);
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

    #[test]
    fn projection_workers_reuse_threads_and_cover_uneven_rows() -> io::Result<()> {
        fn team_ids() -> io::Result<HashSet<ThreadId>> {
            let barrier = Barrier::new(4);
            let ids = Mutex::new(HashSet::new());
            let mut rows = [0.0; 16];
            run(&mut rows, 4, &|_, output| {
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

        // Return construction failure before entering a barrier: the public
        // helper's serial fallback must never make this control hang.
        let team = OwnedTeam::build(4)?;
        TEAM.with(|slot| *slot.borrow_mut() = Some(CachedTeam::Ready(team)));
        #[cfg(not(windows))]
        assert!(rayon::current_thread_index().is_none());
        let first = team_ids()?;
        assert_eq!(first.len(), 4);
        assert_eq!(first, team_ids()?);
        let mut rows = [0.0; 17];
        run(&mut rows, 4, &|offset, output| {
            for (index, value) in output.iter_mut().enumerate() {
                *value = f32::from(u16::try_from(offset + index).unwrap_or(0));
            }
        });
        assert_eq!(
            rows,
            std::array::from_fn(|index| { f32::from(u16::try_from(index).unwrap_or(0)) })
        );
        Ok(())
    }

    #[test]
    fn projection_workers_handle_nested_and_serial_jobs() {
        let mut rows = [0.0; 9];
        run(&mut rows, 3, &|offset, output| {
            let mut nested = [0.0; 5];
            run(&mut nested, 3, &|start, nested_output| {
                assert_eq!(start, 0);
                nested_output.fill(2.0);
            });
            assert_eq!(nested, [2.0; 5]);
            for (index, value) in output.iter_mut().enumerate() {
                *value = f32::from(u16::try_from(offset + index).unwrap_or(0));
            }
        });
        assert_eq!(rows, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        run(&mut rows, 1, &|offset, output| {
            assert_eq!(offset, 0);
            output.fill(3.0);
        });
        assert_eq!(rows, [3.0; 9]);

        TEAM.with(|slot| *slot.borrow_mut() = Some(CachedTeam::Unavailable(3)));
        run(&mut rows, 3, &|offset, output| {
            assert_eq!(offset, 0);
            assert_eq!(output.len(), 9);
            output.fill(4.0);
        });
        assert_eq!(rows, [4.0; 9]);
        TEAM.with(|slot| {
            assert!(matches!(*slot.borrow(), Some(CachedTeam::Unavailable(3))));
        });
    }

    #[test]
    fn projection_workers_retain_pool_across_smaller_slices_and_group_counts() -> io::Result<()> {
        fn team_ids(worker_count: usize) -> io::Result<HashSet<ThreadId>> {
            let barrier = Barrier::new(worker_count);
            let ids = Mutex::new(HashSet::new());
            let mut rows = vec![0.0_f32; worker_count * 4];
            run(&mut rows, worker_count, &|_, output| {
                ids.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(thread::current().id());
                barrier.wait();
                output.fill(1.0);
            });
            ids.into_inner()
                .map_err(|_| io::Error::other("Thread IDs poisoned."))
        }

        let team = OwnedTeam::build(8)?;
        TEAM.with(|slot| *slot.borrow_mut() = Some(CachedTeam::Ready(team)));
        let initial = team_ids(8)?;
        assert_eq!(initial.len(), 8);

        for alignment in [2_usize, 4] {
            let mut small = [0.0_f32; 6];
            let seen = Mutex::new(HashSet::new());
            run_aligned(&mut small, 8, alignment, &|offset, slice| {
                seen.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(thread::current().id());
                for (i, slot) in slice.iter_mut().enumerate() {
                    *slot = f32::from(u16::try_from(offset + i).unwrap_or(0));
                }
            });
            assert_eq!(small, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
            let seen = seen
                .into_inner()
                .map_err(|_| io::Error::other("Seen IDs poisoned."))?;
            assert!(seen.is_subset(&initial));
        }

        assert_eq!(team_ids(8)?, initial);
        Ok(())
    }
}
