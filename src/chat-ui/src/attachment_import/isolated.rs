//! Offline AppContainer host supervision and binary packet framing for image/video decoding.
#![forbid(unsafe_code)]

use std::{
    ffi::OsString,
    io::{self, Read, Write},
    path::Path,
    sync::{Arc, atomic::Ordering, mpsc},
    thread,
    time::{Duration, Instant},
};

use crate::isolated_worker::{
    SandboxedLaunch, SlotError, WorkerErrors, WorkerSlot, le_u32, le_u64, recv_until,
    run_sandboxed_piped,
};

use super::{
    Attachment, MAX_IMAGE_EDGE, MAX_MODEL_PNG_BYTES, MAX_SOURCE_MEDIA_BYTES, MAX_THUMBNAIL_EDGE,
    MAX_THUMBNAIL_PNG_BYTES, MAX_TOTAL_VIDEO_PNG_BYTES, MAX_VIDEO_FRAME_EDGE, MAX_VIDEO_KEYFRAMES,
    MediaKind, POLL, PrehashedPng, Shared, VideoFrame, display_name, image,
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
const ACTIVE_FAILURE: &str =
    "Another media attachment import is still active. Wait or cancel it first.";
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
const ERRORS: WorkerErrors = WorkerErrors {
    isolation: ISOLATION_FAILURE,
    reboot_pending: SANDBOX_REBOOT_PENDING_FAILURE,
    unavailable: SANDBOX_UNAVAILABLE_FAILURE,
    protocol: PROTOCOL_FAILURE,
    cleanup: CLEANUP_FAILURE,
};

static SLOT: WorkerSlot = WorkerSlot::new();

pub(super) fn import_media(
    path: &Path,
    kind: MediaKind,
    shared: &Shared,
    timeout: Duration,
) -> Result<Attachment, String> {
    let _active = SLOT.acquire().map_err(|err| {
        match err {
            SlotError::Active => ACTIVE_FAILURE,
            SlotError::Closed => CLEANUP_FAILURE,
        }
        .to_owned()
    })?;
    if shared.cancelled.load(Ordering::Acquire) {
        return Err("Attachment import cancelled.".to_owned());
    }
    let name = display_name(path);
    let launch = SandboxedLaunch::prepare(
        path,
        Some(MAX_SOURCE_MEDIA_BYTES),
        "attachment.stdout",
        "attachment.stderr",
        PROCESS_COMMIT_LIMIT,
        vec![
            OsString::from("--attachment-import-worker"),
            OsString::from(kind.as_arg()),
            path.as_os_str().to_owned(),
        ],
        SOURCE_FAILURE,
        ISOLATION_FAILURE,
    )
    .map_err(str::to_owned)?;
    let deadline = Instant::now() + timeout;
    run_sandboxed_piped(
        &SLOT,
        launch,
        CLEANUP_LIMIT,
        ERRORS,
        |input, mut output| {
            drop(input);
            let (done_tx, done_rx) = mpsc::sync_channel(1);
            let reader = thread::Builder::new()
                .name("bonsai-attachment-pipe".to_owned())
                .spawn(move || {
                    let result = read_packet_and_eof(&mut output, &name, kind);
                    let _sent = done_tx.try_send(result);
                })?;
            Ok((reader, done_rx))
        },
        |done_rx| {
            recv_until(
                &done_rx,
                &shared.cancelled,
                deadline,
                POLL,
                || "Attachment import cancelled.".to_owned(),
                || TIMEOUT_FAILURE.to_owned(),
                || PROTOCOL_FAILURE.to_owned(),
            )
        },
    )
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
                thumbnail_png: PrehashedPng::from(thumbnail_png),
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
                thumbnail_png: PrehashedPng::from(thumbnail_png),
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
    le_u32(bytes, offset).ok_or_else(|| PROTOCOL_FAILURE.to_owned())
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, String> {
    le_u64(bytes, offset).ok_or_else(|| PROTOCOL_FAILURE.to_owned())
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
