//! Bounded CPU thread budgets for Snapdragon X1/X2 Qwen3-ASR serving and qualification.
#![forbid(unsafe_code)]

use std::io;

/// Maximum explicit `--threads` count admitted across speech services.
const MAX_THREADS: usize = 256;

/// Maximum worker threads spawned in one persistent ASR worker pool.
pub(super) const MAX_WORKERS: usize = 18;

const DECODE_CLUSTER_CAP: usize = 8;

fn invalid_threads() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "Invalid speech ASR CPU thread count.",
    )
}

pub(super) fn validate_thread_count(threads: usize) -> io::Result<usize> {
    if (1..=MAX_THREADS).contains(&threads) {
        Ok(threads)
    } else {
        Err(invalid_threads())
    }
}

/// Select `(decode, batch)` thread budgets for Rust ASR stages.
///
/// Explicit `--threads` values (`1..=256`) pin both decode and batch execution
/// so `--threads 1` enforces single-threaded serial execution. Without an
/// explicit override, two logical cores are reserved for UI and capture (`1..=18`),
/// yielding `(8, 10)` on 12-core Snapdragon X1 Elite and `(8, 16)` on 18-core X2.
pub(super) fn select_asr_threads(
    requested: Option<usize>,
    available: usize,
) -> io::Result<(usize, usize)> {
    if let Some(count) = requested {
        let count = validate_thread_count(count)?;
        return Ok((count, count));
    }
    let batch = available.saturating_sub(2).clamp(1, MAX_WORKERS);
    let decode = batch.min(DECODE_CLUSTER_CAP);
    Ok((decode, batch))
}

#[cfg(any(feature = "native-engine", test))]
pub(super) fn select_native_threads(requested: Option<usize>, available: usize) -> io::Result<i32> {
    let (decode, _) = select_asr_threads(requested, available)?;
    i32::try_from(decode).map_err(|_| invalid_threads())
}

#[cfg(test)]
mod tests {
    use super::{MAX_THREADS, MAX_WORKERS, select_asr_threads, select_native_threads};
    use std::io;

    #[test]
    fn asr_thread_budget_matches_x1_x2_defaults_and_bounds() -> io::Result<()> {
        // 12-core Snapdragon X Elite -> 8 decode (2 Oryon clusters), 10 batch (12 - 2).
        assert_eq!(select_asr_threads(None, 12)?, (8, 10));
        assert_eq!(select_native_threads(None, 12)?, 8);
        // 18-core Snapdragon X2 Elite Extreme -> 8 decode, 16 batch (18 - 2).
        assert_eq!(select_asr_threads(None, 18)?, (8, 16));
        assert_eq!(select_native_threads(None, 18)?, 8);
        // Clamped extremes.
        assert_eq!(select_asr_threads(None, 0)?, (1, 1));
        assert_eq!(select_asr_threads(None, 3)?, (1, 1));
        assert_eq!(select_asr_threads(None, 64)?, (8, MAX_WORKERS));
        // Explicit override pins both stages symmetrically, including serial control `1`.
        for count in [1, 4, 8, 12, MAX_THREADS] {
            assert_eq!(select_asr_threads(Some(count), 12)?, (count, count));
            assert_eq!(
                usize::try_from(select_native_threads(Some(count), 12)?).ok(),
                Some(count)
            );
        }
        for invalid in [0, MAX_THREADS + 1, usize::MAX] {
            assert_eq!(
                select_asr_threads(Some(invalid), 12)
                    .err()
                    .map(|error| error.kind()),
                Some(io::ErrorKind::InvalidInput)
            );
            assert_eq!(
                select_native_threads(Some(invalid), 12)
                    .err()
                    .map(|error| error.kind()),
                Some(io::ErrorKind::InvalidInput)
            );
        }
        Ok(())
    }
}
