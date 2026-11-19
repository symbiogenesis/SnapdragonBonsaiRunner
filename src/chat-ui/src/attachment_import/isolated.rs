//! Offline AppContainer host supervision and binary packet framing for image/video decoding.
#![forbid(unsafe_code)]

use std::{
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::windows::fs::{MetadataExt as _, OpenOptionsExt as _},
    path::{Component, Path, Prefix},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use bonsai_windows_credentials::PrivateDirectory;
use bonsai_windows_process::{OwnedProcess, ProcessSpec, SandboxNetwork, SandboxPolicy};

use super::{
    Attachment, MAX_IMAGE_EDGE, MAX_MODEL_PNG_BYTES, MAX_SOURCE_MEDIA_BYTES, MAX_THUMBNAIL_EDGE,
    MAX_THUMBNAIL_PNG_BYTES, MAX_TOTAL_VIDEO_PNG_BYTES, MAX_VIDEO_FRAME_EDGE, MAX_VIDEO_KEYFRAMES,
    MediaKind, POLL, Shared, VideoFrame, display_name, image,
};

pub(super) const MAGIC: &[u8; 8] = b"BSATT001";
pub(super) const HEADER_BYTES: usize = 32;
pub(super) const KIND_IMAGE: u32 = 1;
pub(super) const KIND_VIDEO: u32 = 2;
pub(super) const KIND_ERROR: u32 = 3;
pub(super) const ERR_DECODE: u64 = 0;
pub(super) const ERR_MEMORY: u64 = 1;
pub(super) const ERR_NO_VIDEO: u64 = 2;

const PROCESS_COMMIT_LIMIT: usize = 512 * 1024 * 1024;
const CLEANUP_LIMIT: Duration = Duration::from_secs(2);
const ISOLATION_FAILURE: &str =
    "Windows could not start the isolated attachment decoder. Your draft was kept.";
const SANDBOX_REBOOT_PENDING_FAILURE: &str = "Windows Update has a pending restart required for the isolated attachment decoder. Restart Windows, then reopen Swift. Your draft was kept.";
const SANDBOX_UNAVAILABLE_FAILURE: &str = "This Windows build lacks the required Windows sandbox (PSEC 1.1) for the isolated attachment decoder. Install the latest Windows updates, restart Windows, and reopen Swift. Your draft was kept.";
const PROTOCOL_FAILURE: &str =
    "The isolated attachment decoder returned invalid media data. Your draft was kept.";
const TIMEOUT_FAILURE: &str =
    "Windows did not finish decoding this media file in time. Your draft was kept.";
const CLEANUP_FAILURE: &str = "Windows could not confirm isolated media decoder cleanup. Restart the app before importing more media.";
const SOURCE_FAILURE: &str =
    "Choose a nonempty image or video file up to 64 MiB on a local drive. Your draft was kept.";
const DECODE_FAILURE: &str =
    "Windows could not decode this image or video file. Your draft was kept.";
const MEMORY_FAILURE: &str =
    "This media file exceeds memory limits after decoding. Choose a smaller file.";
const NO_VIDEO_FAILURE: &str =
    "No decodable video stream was found in this file. Your draft was kept.";

static ADMISSION_CLOSED: AtomicBool = AtomicBool::new(false);
static ACTIVE: AtomicBool = AtomicBool::new(false);
static QUARANTINE: OnceLock<Quarantine> = OnceLock::new();

struct ActiveImport<'a>(&'a AtomicBool);

impl<'a> ActiveImport<'a> {
    fn acquire(slot: &'a AtomicBool) -> Result<Self, String> {
        slot.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| Self(slot))
            .map_err(|_| {
                "Another media attachment import is still active. Wait or cancel it first."
                    .to_owned()
            })
    }
}

impl Drop for ActiveImport<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

#[allow(dead_code, reason = "Retained for process-lifetime cleanup safety.")]
struct Quarantine {
    _source: Selection,
    _executable: Selection,
    _scratch: Scratch,
    _reader: Option<JoinHandle<()>>,
}

fn quarantine(
    source: Selection,
    executable: Selection,
    scratch: Scratch,
    reader: Option<JoinHandle<()>>,
) {
    ADMISSION_CLOSED.store(true, Ordering::Release);
    QUARANTINE.get_or_init(|| Quarantine {
        _source: source,
        _executable: executable,
        _scratch: scratch,
        _reader: reader,
    });
}

#[allow(
    dead_code,
    reason = "Retained directory and file locks prevent source replacement."
)]
struct Selection {
    _ancestors: Vec<File>,
    _file: File,
}

impl Selection {
    fn open(path: &Path, enforce_media_cap: bool) -> io::Result<Self> {
        if !path.is_absolute()
            || !matches!(
                path.components().next(),
                Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_))
            )
            || path
                .components()
                .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
        {
            return Err(io::Error::other(SOURCE_FAILURE));
        }
        bonsai_windows_process::require_local_data_drive(path)?;
        let mut ancestors = Vec::new();
        for ancestor in path
            .ancestors()
            .skip(1)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            let directory = OpenOptions::new()
                .read(true)
                .access_mode(0x81)
                .share_mode(3)
                .custom_flags(0x0220_0000)
                .open(ancestor)?;
            let metadata = directory.metadata()?;
            if !metadata.is_dir() || metadata.file_attributes() & 0x400 != 0 {
                return Err(io::Error::other(SOURCE_FAILURE));
            }
            ancestors.push(directory);
        }
        let file = OpenOptions::new()
            .read(true)
            .share_mode(1)
            .custom_flags(0x0020_0000)
            .open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file()
            || metadata.len() == 0
            || metadata.file_attributes() & 0x400 != 0
            || (enforce_media_cap && metadata.len() > MAX_SOURCE_MEDIA_BYTES)
        {
            return Err(io::Error::other(SOURCE_FAILURE));
        }
        Ok(Self {
            _ancestors: ancestors,
            _file: file,
        })
    }
}

struct Scratch(Option<PrivateDirectory>);

impl Scratch {
    fn create() -> io::Result<Self> {
        PrivateDirectory::create(&std::env::temp_dir()).map(|directory| Self(Some(directory)))
    }

    fn path(&self) -> io::Result<&Path> {
        self.0
            .as_ref()
            .map(PrivateDirectory::path)
            .ok_or_else(|| io::Error::other(CLEANUP_FAILURE))
    }

    fn cleanup(&mut self) -> io::Result<()> {
        let Some(directory) = self.0.take() else {
            return Ok(());
        };
        let path = directory.path().to_owned();
        let stderr = path.join("attachment.stderr");
        if stderr.try_exists()? {
            fs::remove_file(stderr)?;
        }
        let worker = path.join("worker");
        if worker.try_exists()? {
            fs::remove_dir(worker)?;
        }
        drop(directory);
        fs::remove_dir(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _removed = self.cleanup();
    }
}

struct Prepared {
    source: Selection,
    executable_guard: Selection,
    scratch: Scratch,
    specification: ProcessSpec,
    policy: SandboxPolicy,
}

fn prepare(path: &Path, kind: MediaKind) -> Result<Prepared, String> {
    let source = Selection::open(path, true).map_err(|_| SOURCE_FAILURE.to_owned())?;
    let executable = std::env::current_exe().map_err(|_| ISOLATION_FAILURE.to_owned())?;
    let executable_guard =
        Selection::open(&executable, false).map_err(|_| ISOLATION_FAILURE.to_owned())?;
    let scratch = Scratch::create().map_err(|_| ISOLATION_FAILURE.to_owned())?;
    let directory = scratch.path().map_err(|_| ISOLATION_FAILURE.to_owned())?;
    let working_directory = directory.join("worker");
    fs::create_dir(&working_directory).map_err(|_| ISOLATION_FAILURE.to_owned())?;
    let specification = ProcessSpec {
        executable: executable.clone(),
        arguments: vec![
            OsString::from("--attachment-import-worker"),
            OsString::from(kind.as_arg()),
            path.as_os_str().to_owned(),
        ],
        working_directory: working_directory.clone(),
        stdout: directory.join("attachment.stdout"),
        stderr: directory.join("attachment.stderr"),
        memory_limit_bytes: PROCESS_COMMIT_LIMIT,
        active_process_limit: 1,
    };
    let policy = SandboxPolicy {
        readonly_paths: vec![executable, path.to_owned()],
        writable_directory: working_directory,
    };
    Ok(Prepared {
        source,
        executable_guard,
        scratch,
        specification,
        policy,
    })
}

pub(super) fn import_media(
    path: &Path,
    kind: MediaKind,
    shared: &Shared,
    timeout: Duration,
) -> Result<Attachment, String> {
    let _active = ActiveImport::acquire(&ACTIVE)?;
    if ADMISSION_CLOSED.load(Ordering::Acquire) {
        return Err(CLEANUP_FAILURE.to_owned());
    }
    if shared.cancelled.load(Ordering::Acquire) {
        return Err("Attachment import cancelled.".to_owned());
    }
    let name = display_name(path);
    let Prepared {
        source,
        executable_guard,
        mut scratch,
        specification,
        policy,
    } = prepare(path, kind)?;
    let deadline = Instant::now() + timeout;
    let (process, input, mut output) =
        match OwnedProcess::start_sandboxed_piped(&specification, &policy, SandboxNetwork::Offline)
        {
            Ok(started) => started,
            Err(error) if !bonsai_windows_process::process_start_cleanup_unconfirmed(&error) => {
                if error.kind() == io::ErrorKind::Unsupported {
                    return Err(if bonsai_windows_process::windows_reboot_pending() {
                        SANDBOX_REBOOT_PENDING_FAILURE
                    } else {
                        SANDBOX_UNAVAILABLE_FAILURE
                    }
                    .to_owned());
                }
                return Err(ISOLATION_FAILURE.to_owned());
            }
            Err(_) => {
                quarantine(source, executable_guard, scratch, None);
                return Err(CLEANUP_FAILURE.to_owned());
            }
        };
    drop(input);
    let (done_tx, done_rx) = mpsc::sync_channel(1);
    let reader_name = name.clone();
    let Ok(reader) = thread::Builder::new()
        .name("bonsai-attachment-pipe".to_owned())
        .spawn(move || {
            let result = read_packet_and_eof(&mut output, &reader_name, kind);
            let _sent = done_tx.try_send(result);
        })
    else {
        if process.stop(CLEANUP_LIMIT).is_err() {
            quarantine(source, executable_guard, scratch, None);
            drop(process);
            return Err(CLEANUP_FAILURE.to_owned());
        }
        drop(process);
        return Err(ISOLATION_FAILURE.to_owned());
    };
    let result = await_packet(&done_rx, shared, deadline);
    drop(done_rx);
    let clean = match &result {
        Ok(_) => process
            .wait_for_exit(CLEANUP_LIMIT)
            .and_then(|code| match code {
                Some(0) if process.active_processes()? == 0 => Ok(()),
                _ => Err(io::Error::other(CLEANUP_FAILURE)),
            }),
        Err(_) => process.stop(CLEANUP_LIMIT),
    };
    if clean.is_err() {
        quarantine(source, executable_guard, scratch, Some(reader));
        drop(process);
        return Err(CLEANUP_FAILURE.to_owned());
    }
    reader.join().map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    drop(process);
    if scratch.cleanup().is_err() {
        ADMISSION_CLOSED.store(true, Ordering::Release);
        return Err(CLEANUP_FAILURE.to_owned());
    }
    result
}

fn await_packet(
    receiver: &Receiver<Result<Attachment, String>>,
    shared: &Shared,
    deadline: Instant,
) -> Result<Attachment, String> {
    loop {
        if shared.cancelled.load(Ordering::Acquire) {
            return Err("Attachment import cancelled.".to_owned());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(TIMEOUT_FAILURE.to_owned());
        }
        match receiver.recv_timeout(POLL.min(remaining)) {
            Ok(result) => {
                return if shared.cancelled.load(Ordering::Acquire) {
                    Err("Attachment import cancelled.".to_owned())
                } else {
                    result
                };
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err(PROTOCOL_FAILURE.to_owned()),
        }
    }
}

#[allow(dead_code, reason = "Used across host pipe reader and unit tests.")]
fn read_packet_and_eof(
    read: &mut impl Read,
    name: &str,
    expected_kind: MediaKind,
) -> Result<Attachment, String> {
    let attachment = read_packet(read, name, expected_kind)?;
    let mut surplus = [0_u8; 1];
    if read
        .read(&mut surplus)
        .map_err(|_| PROTOCOL_FAILURE.to_owned())?
        != 0
    {
        return Err(PROTOCOL_FAILURE.to_owned());
    }
    Ok(attachment)
}

#[allow(dead_code, reason = "Used across host pipe reader and unit tests.")]
pub(super) fn read_packet(
    read: &mut impl Read,
    name: &str,
    expected_kind: MediaKind,
) -> Result<Attachment, String> {
    let mut header = [0_u8; HEADER_BYTES];
    read.read_exact(&mut header)
        .map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    if &header[..8] != MAGIC {
        return Err(PROTOCOL_FAILURE.to_owned());
    }
    let kind = u32_at(&header, 8)?;
    let width = u32_at(&header, 12)?;
    let height = u32_at(&header, 16)?;
    let frame_count = u32_at(&header, 20)?;
    let duration_or_code = u64_at(&header, 24)?;
    if kind == KIND_ERROR {
        if width != 0 || height != 0 || frame_count != 0 {
            return Err(PROTOCOL_FAILURE.to_owned());
        }
        return Err(match duration_or_code {
            ERR_MEMORY => MEMORY_FAILURE.to_owned(),
            ERR_NO_VIDEO => NO_VIDEO_FAILURE.to_owned(),
            _ => DECODE_FAILURE.to_owned(),
        });
    }
    let expected_code = match expected_kind {
        MediaKind::Image => KIND_IMAGE,
        MediaKind::Video => KIND_VIDEO,
    };
    if kind != expected_code {
        return Err(PROTOCOL_FAILURE.to_owned());
    }
    let max_edge = match expected_kind {
        MediaKind::Image => MAX_IMAGE_EDGE,
        MediaKind::Video => MAX_VIDEO_FRAME_EDGE,
    };
    if width == 0 || height == 0 || width > max_edge || height > max_edge {
        return Err(PROTOCOL_FAILURE.to_owned());
    }
    let expected_frames = usize::try_from(frame_count).map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    let max_frames = match expected_kind {
        MediaKind::Image => 1,
        MediaKind::Video => MAX_VIDEO_KEYFRAMES,
    };
    if expected_frames == 0 || expected_frames > max_frames {
        return Err(PROTOCOL_FAILURE.to_owned());
    }
    let thumbnail_png =
        read_length_prefixed_png(read, MAX_THUMBNAIL_EDGE, MAX_THUMBNAIL_PNG_BYTES)?;
    match expected_kind {
        MediaKind::Image => {
            if duration_or_code != 0 {
                return Err(PROTOCOL_FAILURE.to_owned());
            }
            let (timestamp_ms, png_bytes) = read_frame_png(read, width, height, max_edge)?;
            if timestamp_ms != 0 {
                return Err(PROTOCOL_FAILURE.to_owned());
            }
            let data_url = image::png_to_data_url(&png_bytes);
            Ok(Attachment::Image {
                name: name.to_owned(),
                width,
                height,
                data_url: Arc::from(data_url),
                thumbnail_png: Arc::from(thumbnail_png),
            })
        }
        MediaKind::Video => {
            let mut frames = Vec::with_capacity(expected_frames);
            let mut total_png_bytes = 0_usize;
            let mut prev_ts = 0_u64;
            for index in 0..expected_frames {
                let (timestamp_ms, png_bytes) = read_frame_png(read, width, height, max_edge)?;
                if index > 0 && timestamp_ms < prev_ts {
                    return Err(PROTOCOL_FAILURE.to_owned());
                }
                prev_ts = timestamp_ms;
                total_png_bytes = total_png_bytes
                    .checked_add(png_bytes.len())
                    .ok_or_else(|| PROTOCOL_FAILURE.to_owned())?;
                if total_png_bytes > MAX_TOTAL_VIDEO_PNG_BYTES {
                    return Err(MEMORY_FAILURE.to_owned());
                }
                frames.push(VideoFrame {
                    timestamp_ms,
                    data_url: Arc::from(image::png_to_data_url(&png_bytes)),
                });
            }
            Ok(Attachment::Video {
                name: name.to_owned(),
                width,
                height,
                duration_ms: duration_or_code,
                frames: Arc::from(frames),
                thumbnail_png: Arc::from(thumbnail_png),
            })
        }
    }
}

#[allow(dead_code, reason = "Used across host pipe reader and unit tests.")]
fn read_length_prefixed_png(
    read: &mut impl Read,
    max_edge: u32,
    max_bytes: usize,
) -> Result<Vec<u8>, String> {
    let mut len_buf = [0_u8; 4];
    read.read_exact(&mut len_buf)
        .map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    let len =
        usize::try_from(u32::from_le_bytes(len_buf)).map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    if len == 0 || len > max_bytes {
        return Err(PROTOCOL_FAILURE.to_owned());
    }
    let mut png = vec![0_u8; len];
    read.read_exact(&mut png)
        .map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    let _dims = image::validate_png_bytes(&png, max_edge, max_bytes)
        .map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    Ok(png)
}

#[allow(dead_code, reason = "Used across host pipe reader and unit tests.")]
fn read_frame_png(
    read: &mut impl Read,
    expected_w: u32,
    expected_h: u32,
    max_edge: u32,
) -> Result<(u64, Vec<u8>), String> {
    let mut meta = [0_u8; 12];
    read.read_exact(&mut meta)
        .map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    let timestamp_ms = u64_at(&meta, 0)?;
    let len = usize::try_from(u32_at(&meta, 8)?).map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    if len == 0 || len > MAX_MODEL_PNG_BYTES {
        return Err(PROTOCOL_FAILURE.to_owned());
    }
    let mut png = vec![0_u8; len];
    read.read_exact(&mut png)
        .map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    let dims = image::validate_png_bytes(&png, max_edge, MAX_MODEL_PNG_BYTES)
        .map_err(|_| PROTOCOL_FAILURE.to_owned())?;
    if dims != (expected_w, expected_h) {
        return Err(PROTOCOL_FAILURE.to_owned());
    }
    Ok((timestamp_ms, png))
}

pub(super) fn write_error_packet(write: &mut impl Write, code: u64) -> io::Result<()> {
    let mut header = [0_u8; HEADER_BYTES];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&KIND_ERROR.to_le_bytes());
    header[24..32].copy_from_slice(&code.to_le_bytes());
    write.write_all(&header)?;
    write.flush()
}

pub(super) struct EncodedFrame {
    pub(super) timestamp_ms: u64,
    pub(super) png_bytes: Vec<u8>,
}

pub(super) fn write_media_packet(
    write: &mut impl Write,
    kind: MediaKind,
    width: u32,
    height: u32,
    duration_ms: u64,
    thumbnail_png: &[u8],
    frames: &[EncodedFrame],
) -> io::Result<()> {
    let frame_count =
        u32::try_from(frames.len()).map_err(|_| io::Error::other("Too many frames"))?;
    let thumb_len =
        u32::try_from(thumbnail_png.len()).map_err(|_| io::Error::other("Thumbnail too large"))?;
    let kind_code = match kind {
        MediaKind::Image => KIND_IMAGE,
        MediaKind::Video => KIND_VIDEO,
    };
    let mut header = [0_u8; HEADER_BYTES];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&kind_code.to_le_bytes());
    header[12..16].copy_from_slice(&width.to_le_bytes());
    header[16..20].copy_from_slice(&height.to_le_bytes());
    header[20..24].copy_from_slice(&frame_count.to_le_bytes());
    header[24..32].copy_from_slice(&duration_or_zero(kind, duration_ms).to_le_bytes());
    write.write_all(&header)?;
    write.write_all(&thumb_len.to_le_bytes())?;
    write.write_all(thumbnail_png)?;
    for frame in frames {
        let len =
            u32::try_from(frame.png_bytes.len()).map_err(|_| io::Error::other("PNG too large"))?;
        write.write_all(&frame.timestamp_ms.to_le_bytes())?;
        write.write_all(&len.to_le_bytes())?;
        write.write_all(&frame.png_bytes)?;
    }
    write.flush()
}

const fn duration_or_zero(kind: MediaKind, duration_ms: u64) -> u64 {
    match kind {
        MediaKind::Image => 0,
        MediaKind::Video => duration_ms,
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, String> {
    bytes
        .get(offset..offset + 4)
        .and_then(|slice| slice.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| PROTOCOL_FAILURE.to_owned())
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, String> {
    bytes
        .get(offset..offset + 8)
        .and_then(|slice| slice.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| PROTOCOL_FAILURE.to_owned())
}

#[cfg(test)]
mod tests {
    use super::{
        ERR_NO_VIDEO, EncodedFrame, NO_VIDEO_FAILURE, read_packet_and_eof, write_error_packet,
        write_media_packet,
    };
    use crate::attachment_import::{Attachment, MediaKind, image::encode_rgba_png};
    use std::io::{self, Cursor};

    #[test]
    fn packet_roundtrip_verifies_image_video_errors_and_rejects_surplus_bytes() -> io::Result<()> {
        let rgba_2x2 = vec![255_u8; 2 * 2 * 4];
        let full_png = encode_rgba_png(&rgba_2x2, 2, 2, 64 * 1024).unwrap_or_default();
        let thumb_png = encode_rgba_png(&rgba_2x2, 2, 2, 64 * 1024).unwrap_or_default();

        let mut wire = Vec::new();
        assert!(
            write_media_packet(
                &mut wire,
                MediaKind::Video,
                2,
                2,
                2_400,
                &thumb_png,
                &[
                    EncodedFrame {
                        timestamp_ms: 400,
                        png_bytes: full_png.clone(),
                    },
                    EncodedFrame {
                        timestamp_ms: 1_800,
                        png_bytes: full_png,
                    },
                ],
            )
            .is_ok()
        );
        let parsed = read_packet_and_eof(&mut Cursor::new(&wire), "clip.mp4", MediaKind::Video)
            .map_err(io::Error::other)?;
        let Attachment::Video {
            name,
            width,
            height,
            duration_ms,
            frames,
            ..
        } = parsed
        else {
            return Err(io::Error::other(format!(
                "Expected video attachment, got {parsed:?}"
            )));
        };
        assert_eq!(name, "clip.mp4");
        assert_eq!((width, height, duration_ms), (2, 2, 2_400));
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].timestamp_ms, 400);
        assert_eq!(frames[1].timestamp_ms, 1_800);
        assert!(frames[0].data_url.starts_with("data:image/png;base64,"));

        // Trailing byte after valid packet must be rejected.
        wire.push(0);
        assert!(
            read_packet_and_eof(&mut Cursor::new(&wire), "clip.mp4", MediaKind::Video).is_err()
        );

        // Error packet surfaces clean reason.
        let mut err_wire = Vec::new();
        assert!(write_error_packet(&mut err_wire, ERR_NO_VIDEO).is_ok());
        let err = read_packet_and_eof(&mut Cursor::new(&err_wire), "audio.mp4", MediaKind::Video);
        assert_eq!(err, Err(NO_VIDEO_FAILURE.to_owned()));
        Ok(())
    }
}
