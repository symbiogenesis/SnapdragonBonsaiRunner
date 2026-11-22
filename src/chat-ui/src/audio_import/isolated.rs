//! One offline native decoder process per explicitly selected recording.
//!
//! Only bounded canonical WAV segments cross this boundary. The selected source
//! and its ancestors stay locked through confirmed process cleanup; neither the
//! ASR service nor this pipe protocol can request another pathname.
#![forbid(unsafe_code)]

use std::{
    io::{self, Read, Write},
    path::Path,
    sync::mpsc::{self, Receiver, SyncSender},
    thread,
    time::{Duration, Instant},
};

use crate::isolated_worker::{
    SandboxedLaunch, SlotError, WorkerErrors, WorkerSlot, le_u32, le_u64, recv_until,
    run_sandboxed_piped,
};

use super::{
    CANCELLED, CHUNK_TOO_LONG, CHUNK_WORKING_BYTES, CODEC_FAILURE, CONVERSION_LIMIT, Chunk,
    ImportResult, MAX_SAMPLES, MEMORY_FAILURE, OsString, POLL, Shared, await_advance, check_memory,
    convert_with, sample_duration,
};

const HEADER_BYTES: usize = 48;
const MAGIC: &[u8; 8] = b"BSAUD001";
const FLAG_LAST: u32 = 1;
const FLAG_ERROR: u32 = 2;
const NEXT: u8 = b'N';
const PROCESS_COMMIT_LIMIT: u64 = 256 * 1024 * 1024;
// Charge the complete helper allowance in addition to the existing host/audio
// pipeline buffer ledger and both requested 64 KiB kernel pipe buffers.
const ISOLATED_WORKING_BYTES: u64 = PROCESS_COMMIT_LIMIT + CHUNK_WORKING_BYTES + 128 * 1024;
const CLEANUP_LIMIT: Duration = Duration::from_secs(2);
const ACTIVE_FAILURE: &str =
    "Another audio import is still active. Cancel it before selecting a new file.";
const ISOLATION_FAILURE: &str = "Windows could not start the isolated audio decoder. Transcription stopped. Your draft was kept.";
const SANDBOX_REBOOT_PENDING_FAILURE: &str = "Windows Update has a pending restart required for the isolated audio decoder. Restart Windows, then reopen Swift. Your draft was kept.";
const SANDBOX_UNAVAILABLE_FAILURE: &str = "This Windows build lacks the required Windows sandbox (PSEC 1.1) for the isolated audio decoder. Install the latest Windows updates, restart Windows, and reopen Swift. Your draft was kept.";
const PROTOCOL_FAILURE: &str = "The isolated audio decoder returned invalid or incomplete audio. Transcription stopped. Your draft was kept.";
const TIMEOUT: &str = "Windows did not finish decoding the next audio segment. Try converting the recording to WAV. Your draft was kept.";
const CLEANUP_FAILURE: &str = "Windows could not confirm isolated audio decoder cleanup. Restart the app before importing another file. Your draft was kept.";
const SOURCE_FAILURE: &str = "Choose a nonempty regular audio file on a local drive. The file must stay unchanged until transcription finishes. Your draft was kept.";
const ERRORS: WorkerErrors = WorkerErrors {
    isolation: ISOLATION_FAILURE,
    reboot_pending: SANDBOX_REBOOT_PENDING_FAILURE,
    unavailable: SANDBOX_UNAVAILABLE_FAILURE,
    protocol: PROTOCOL_FAILURE,
    cleanup: CLEANUP_FAILURE,
};
static SLOT: WorkerSlot = WorkerSlot::new();

fn prepare(path: &Path) -> Result<SandboxedLaunch, &'static str> {
    check_memory(ISOLATED_WORKING_BYTES)?;
    SandboxedLaunch::prepare(
        path,
        None,
        "decoder.stdout",
        "decoder.stderr",
        usize::try_from(PROCESS_COMMIT_LIMIT).map_err(|_| ISOLATION_FAILURE)?,
        vec![
            OsString::from("--audio-import-worker"),
            path.as_os_str().to_owned(),
        ],
        SOURCE_FAILURE,
        ISOLATION_FAILURE,
    )
}

pub(super) fn convert(
    path: &Path,
    shared: &Shared,
    sender: &SyncSender<ImportResult>,
    next: &Receiver<()>,
) -> Result<(), &'static str> {
    let _active = SLOT.acquire().map_err(|err| match err {
        SlotError::Active => ACTIVE_FAILURE,
        SlotError::Closed => CLEANUP_FAILURE,
    })?;
    if shared.cancelled.load(std::sync::atomic::Ordering::Acquire) {
        return Err(CANCELLED);
    }
    let launch = prepare(path)?;
    let deadline = Instant::now() + CONVERSION_LIMIT;
    let final_chunk = run_sandboxed_piped(
        &SLOT,
        launch,
        CLEANUP_LIMIT,
        ERRORS,
        |input, output| {
            let (frames, received) = mpsc::sync_channel(1);
            let (advance, permits) = mpsc::sync_channel(1);
            let reader = thread::Builder::new()
                .name("bonsai-audio-decoder-pipe".to_owned())
                .spawn(move || read_frames(input, output, &frames, &permits))?;
            Ok((reader, (advance, received)))
        },
        |(advance, received)| relay(&advance, &received, shared, sender, next, deadline),
    )?;
    // Final output is admitted only after verified native teardown and scratch
    // cleanup. This prevents a terminal audio result from hiding failed cleanup.
    if !shared.cancelled.load(std::sync::atomic::Ordering::Acquire) {
        let _sent = sender.send(Ok(Some(final_chunk)));
    }
    Ok(())
}

fn read_frames(
    mut input: impl Write,
    mut output: impl Read,
    frames: &SyncSender<Result<Chunk, &'static str>>,
    permits: &Receiver<()>,
) {
    let mut expected_start = Duration::ZERO;
    loop {
        let result = read_frame(&mut output, expected_start).and_then(|chunk| {
            if chunk.last {
                let mut surplus = [0];
                if output.read(&mut surplus).map_err(|_| PROTOCOL_FAILURE)? != 0 {
                    return Err(PROTOCOL_FAILURE);
                }
            }
            Ok(chunk)
        });
        let terminal = result.as_ref().ok().is_none_or(|chunk| {
            expected_start = chunk.end;
            chunk.last
        });
        if frames.send(result).is_err() || terminal {
            break;
        }
        if permits.recv().is_err() || input.write_all(&[NEXT]).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod protocol_tests;

fn relay(
    advance: &SyncSender<()>,
    frames: &Receiver<Result<Chunk, &'static str>>,
    shared: &Shared,
    sender: &SyncSender<ImportResult>,
    next: &Receiver<()>,
    initial_deadline: Instant,
) -> Result<Chunk, &'static str> {
    let mut deadline = initial_deadline;
    loop {
        let chunk = receive_frame(frames, shared, deadline)?;
        if chunk.last {
            return Ok(chunk);
        }
        if sender.send(Ok(Some(chunk))).is_err() {
            return Err(CANCELLED);
        }
        await_advance(shared, next)?;
        if shared.cancelled.load(std::sync::atomic::Ordering::Acquire) {
            return Err(CANCELLED);
        }
        // Native writes stay on the supervised IO thread too: a compromised
        // helper may produce frames without consuming its stdin permits.
        advance.try_send(()).map_err(|_| PROTOCOL_FAILURE)?;
        deadline = Instant::now() + CONVERSION_LIMIT;
    }
}

fn receive_frame(
    frames: &Receiver<Result<Chunk, &'static str>>,
    shared: &Shared,
    deadline: Instant,
) -> Result<Chunk, &'static str> {
    recv_until(
        frames,
        &shared.cancelled,
        deadline,
        POLL,
        || CANCELLED,
        || TIMEOUT,
        || PROTOCOL_FAILURE,
    )
}

fn read_frame(read: &mut impl Read, expected_start: Duration) -> Result<Chunk, &'static str> {
    let mut header = [0; HEADER_BYTES];
    read.read_exact(&mut header).map_err(|_| PROTOCOL_FAILURE)?;
    if &header[..8] != MAGIC || header[44..] != [0; 4] {
        return Err(PROTOCOL_FAILURE);
    }
    let length = usize::try_from(u32_at(&header, 8)?).map_err(|_| PROTOCOL_FAILURE)?;
    let flags = u32_at(&header, 12)?;
    if flags == FLAG_ERROR && length == 0 {
        let code = u32_at(&header, 16)?;
        if code > 1 || header[20..].iter().any(|byte| *byte != 0) {
            return Err(PROTOCOL_FAILURE);
        }
        return Err(if code == 1 {
            MEMORY_FAILURE
        } else {
            CODEC_FAILURE
        });
    }
    if flags > FLAG_LAST
        || !(46..=MAX_SAMPLES * 2 + 44).contains(&length)
        || !length.is_multiple_of(2)
    {
        return Err(PROTOCOL_FAILURE);
    }
    check_memory(CHUNK_WORKING_BYTES)?;
    let start = sample_duration(u64_at(&header, 16)?)?;
    let end = sample_duration(u64_at(&header, 24)?)?;
    let nanos = u32_at(&header, 40)?;
    if nanos >= 1_000_000_000 {
        return Err(PROTOCOL_FAILURE);
    }
    let total = Duration::new(u64_at(&header, 32)?, nanos);
    let samples = u64::try_from((length - 44) / 2).map_err(|_| PROTOCOL_FAILURE)?;
    if start != expected_start
        || end <= start
        || end.checked_sub(start) != Some(sample_duration(samples)?)
        || (flags == FLAG_LAST && total != end)
    {
        return Err(PROTOCOL_FAILURE);
    }
    let mut wav = Vec::new();
    wav.try_reserve_exact(length).map_err(|_| MEMORY_FAILURE)?;
    wav.resize(length, 0);
    read.read_exact(&mut wav).map_err(|_| PROTOCOL_FAILURE)?;
    if bonsai_audio::mono_pcm16_data(
        &wav,
        16_000,
        u32::try_from(MAX_SAMPLES).map_err(|_| PROTOCOL_FAILURE)?,
    )
    .is_none()
    {
        return Err(PROTOCOL_FAILURE);
    }
    Ok(Chunk {
        wav,
        start,
        end,
        total,
        last: flags == FLAG_LAST,
    })
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, &'static str> {
    le_u32(bytes, offset).ok_or(PROTOCOL_FAILURE)
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, &'static str> {
    le_u64(bytes, offset).ok_or(PROTOCOL_FAILURE)
}

fn samples(duration: Duration) -> Result<u64, &'static str> {
    duration
        .as_secs()
        .checked_mul(16_000)
        .and_then(|whole| whole.checked_add(u64::from(duration.subsec_nanos() / 62_500)))
        .ok_or(CHUNK_TOO_LONG)
}

fn write_frame(write: &mut impl Write, chunk: &Chunk) -> Result<(), &'static str> {
    let mut header = [0; HEADER_BYTES];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(
        &u32::try_from(chunk.wav.len())
            .map_err(|_| CHUNK_TOO_LONG)?
            .to_le_bytes(),
    );
    header[12..16].copy_from_slice(&u32::from(chunk.last).to_le_bytes());
    header[16..24].copy_from_slice(&samples(chunk.start)?.to_le_bytes());
    header[24..32].copy_from_slice(&samples(chunk.end)?.to_le_bytes());
    header[32..40].copy_from_slice(&chunk.total.as_secs().to_le_bytes());
    header[40..44].copy_from_slice(&chunk.total.subsec_nanos().to_le_bytes());
    write.write_all(&header).map_err(|_| CANCELLED)?;
    write.write_all(&chunk.wav).map_err(|_| CANCELLED)?;
    // StdoutLock can buffer a small PCM tail. Flush one complete frame before
    // waiting for the next permit, including recordings with no newline bytes.
    write.flush().map_err(|_| CANCELLED)
}

pub(super) fn run_worker(arguments: &[OsString]) -> std::process::ExitCode {
    // A direct command-line invocation never activates codecs without actual
    // kernel containment. This entry point exposes no picker/device operation.
    if arguments.len() != 2
        || arguments[0] != "--audio-import-worker"
        || !bonsai_windows_credentials::is_app_container().unwrap_or(false)
    {
        return std::process::ExitCode::FAILURE;
    }
    let source = Path::new(&arguments[1]);
    let shared = Shared::default();
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let result = convert_with(source, &shared, &mut |chunk| {
        write_frame(&mut output, &chunk)?;
        if chunk.last {
            return Ok(false);
        }
        let mut command = [0];
        input.read_exact(&mut command).map_err(|_| CANCELLED)?;
        if command[0] != NEXT {
            return Err(CANCELLED);
        }
        Ok(true)
    });
    if let Err(message) = result {
        let mut header = [0; HEADER_BYTES];
        header[..8].copy_from_slice(MAGIC);
        header[12..16].copy_from_slice(&FLAG_ERROR.to_le_bytes());
        header[16..20].copy_from_slice(&u32::from(message == MEMORY_FAILURE).to_le_bytes());
        let _reported = output.write_all(&header);
        let _flushed = output.flush();
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::{FLAG_LAST, HEADER_BYTES, MAGIC, PROTOCOL_FAILURE, read_frame, write_frame};
    use crate::{
        audio_import::{Chunk, MAX_SAMPLES, sample_duration},
        isolated_worker::PathLock,
    };
    use std::{
        fs, io,
        path::PathBuf,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    fn chunk(last: bool) -> io::Result<Chunk> {
        let mut wav = bonsai_audio::mono_pcm16_header(16_000, 2)
            .ok_or_else(|| io::Error::other("header"))?
            .to_vec();
        wav.extend_from_slice(&[1, 2, 3, 4]);
        let end = sample_duration(2).map_err(io::Error::other)?;
        Ok(Chunk {
            wav,
            start: Duration::ZERO,
            end,
            total: if last { end } else { Duration::from_secs(60) },
            last,
        })
    }

    #[test]
    fn isolated_frames_preserve_pcm_and_reject_changed_identity_and_truncation() -> io::Result<()> {
        let expected = chunk(true)?;
        let mut wire = Vec::new();
        write_frame(&mut wire, &expected).map_err(io::Error::other)?;
        let actual = read_frame(&mut wire.as_slice(), Duration::ZERO).map_err(io::Error::other)?;
        assert_eq!(actual.wav, expected.wav);
        assert_eq!(actual.end, expected.end);
        assert!(actual.last);
        for offset in [0, 8, 12, 16, 24, 32, 40, 44, HEADER_BYTES] {
            let mut changed = wire.clone();
            changed[offset] ^= 0x80;
            assert!(read_frame(&mut changed.as_slice(), Duration::ZERO).is_err());
        }
        for length in [0, 7, HEADER_BYTES - 1, HEADER_BYTES, wire.len() - 1] {
            assert!(read_frame(&mut &wire[..length], Duration::ZERO).is_err());
        }
        assert!(read_frame(&mut wire.as_slice(), Duration::from_secs(1)).is_err());
        Ok(())
    }

    #[test]
    fn isolated_frames_reject_excess_bytes_before_allocating_the_payload() -> io::Result<()> {
        let mut header = [0; HEADER_BYTES];
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(
            &u32::try_from(MAX_SAMPLES * 2 + 46)
                .map_err(io::Error::other)?
                .to_le_bytes(),
        );
        header[12..16].copy_from_slice(&FLAG_LAST.to_le_bytes());
        assert!(matches!(
            read_frame(&mut header.as_slice(), Duration::ZERO),
            Err(PROTOCOL_FAILURE)
        ));
        Ok(())
    }

    #[test]
    fn isolated_frames_flush_small_recordings_before_waiting_for_permit() -> io::Result<()> {
        // The stdio boundary can buffer an entire short clip without a newline.
        // Its frame must be visible before the decoder blocks for its next permit.
        let mut output = io::BufWriter::with_capacity(4_096, Vec::new());
        let expected = chunk(false)?;
        write_frame(&mut output, &expected).map_err(io::Error::other)?;
        assert_eq!(output.buffer(), b"");
        assert_eq!(output.get_ref().len(), HEADER_BYTES + expected.wav.len());
        Ok(())
    }

    struct Fixture(PathBuf);

    impl Drop for Fixture {
        fn drop(&mut self) {
            for name in ["input.wav", "sibling"] {
                let _removed = fs::remove_file(self.0.join(name));
            }
            let _removed = fs::remove_dir(&self.0);
        }
    }

    #[test]
    fn isolated_selection_retains_file_and_ancestors_without_blocking_sibling_writes()
    -> io::Result<()> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?;
        let path = std::env::temp_dir().join(format!(
            "Bonsai-SelectedAudio-{}-{}",
            std::process::id(),
            stamp.as_nanos()
        ));
        fs::create_dir(&path)?;
        let mut fixture = Fixture(path.clone());
        let input = path.join("input.wav");
        fs::write(&input, [1, 2, 3, 4])?;
        let selected = PathLock::open_nonempty(&input, None)?;
        assert!(fs::write(&input, [9, 9]).is_err());
        let moved = path.with_extension("moved");
        assert!(fs::rename(&path, &moved).is_err());
        fs::write(path.join("sibling"), [5, 6])?;
        assert_eq!(fs::read(&input)?, [1, 2, 3, 4]);
        drop(selected);
        fs::rename(&path, &moved)?;
        fixture.0 = moved;
        Ok(())
    }
}
