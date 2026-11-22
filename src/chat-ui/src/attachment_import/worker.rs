//! Isolated AppContainer WIC image and Media Foundation synchronous video keyframe decoder.

use std::{ffi::OsString, io, path::Path, ptr, time::Duration};
use windows::{
    Win32::{
        Foundation::GENERIC_READ,
        Graphics::Imaging::{
            CLSID_WICImagingFactory, GUID_WICPixelFormat32bppRGBA, IWICBitmapSource,
            IWICImagingFactory, WICBitmapDitherTypeNone, WICBitmapInterpolationModeFant,
            WICBitmapPaletteTypeMedianCut, WICDecodeMetadataCacheOnDemand,
        },
        Media::MediaFoundation::{
            IMFAttributes, IMFMediaBuffer, IMFSample, IMFSourceReader, MF_MT_DEFAULT_STRIDE,
            MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_PD_DURATION,
            MF_SOURCE_READER_ALL_STREAMS, MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,
            MF_SOURCE_READER_FIRST_VIDEO_STREAM, MF_SOURCE_READER_MEDIASOURCE,
            MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED, MF_SOURCE_READERF_ENDOFSTREAM,
            MF_SOURCE_READERF_ERROR, MF_VERSION, MFCreateAttributes, MFCreateMediaType,
            MFCreateSourceReaderFromURL, MFMediaType_Video, MFSTARTUP_FULL, MFShutdown, MFStartup,
            MFVideoFormat_RGB32,
        },
        System::{
            Com::StructuredStorage::PROPVARIANT,
            Com::{CLSCTX_INPROC_SERVER, CoCreateInstance},
        },
    },
    core::{GUID, HSTRING, Interface as _, PCWSTR},
};

use super::{
    MAX_IMAGE_EDGE, MAX_MODEL_PNG_BYTES, MAX_THUMBNAIL_EDGE, MAX_THUMBNAIL_PNG_BYTES,
    MAX_TOTAL_VIDEO_PNG_BYTES, MAX_VIDEO_FRAME_EDGE, MAX_VIDEO_KEYFRAMES, MediaKind, image,
    isolated::{
        ERR_DECODE, ERR_MEMORY, ERR_NO_VIDEO, EncodedFrame, write_error_packet, write_media_packet,
    },
};

const ALL_STREAMS: u32 = MF_SOURCE_READER_ALL_STREAMS.0.cast_unsigned();
const VIDEO_STREAM: u32 = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0.cast_unsigned();
const MEDIA_SOURCE: u32 = MF_SOURCE_READER_MEDIASOURCE.0.cast_unsigned();
const MAX_VIDEO_DIM: u32 = 8_192;
const MAX_VIDEO_READ_STEPS: usize = 48;

struct DecodedMedia {
    width: u32,
    height: u32,
    duration_ms: u64,
    thumbnail_png: Vec<u8>,
    frames: Vec<EncodedFrame>,
}

struct Platform;

impl Platform {
    fn start() -> Result<Self, u64> {
        // SAFETY: Matches Media Foundation process-wide init/shutdown pairing inside this worker.
        unsafe { MFStartup(MF_VERSION, MFSTARTUP_FULL) }.map_err(|_| ERR_DECODE)?;
        Ok(Self)
    }
}

impl Drop for Platform {
    fn drop(&mut self) {
        // SAFETY: Pairs with the successful MFStartup call that constructed Platform.
        let _ = unsafe { MFShutdown() };
    }
}

struct LockedBuffer<'a> {
    buffer: &'a IMFMediaBuffer,
    bytes: &'a [u8],
}

impl<'a> LockedBuffer<'a> {
    fn lock(buffer: &'a IMFMediaBuffer) -> Result<Self, u64> {
        let mut data = ptr::null_mut();
        let mut current_length = 0_u32;
        // SAFETY: Valid COM buffer locked until Unlock runs in Drop.
        unsafe { buffer.Lock(&raw mut data, None, Some(&raw mut current_length)) }
            .map_err(|_| ERR_DECODE)?;
        if data.is_null() {
            // SAFETY: Unlock pairs immediately with the Lock call above.
            let _ = unsafe { buffer.Unlock() };
            return Err(ERR_DECODE);
        }
        let len = usize::try_from(current_length).map_err(|_| {
            // SAFETY: Unlock pairs immediately with the Lock call above.
            let _ = unsafe { buffer.Unlock() };
            ERR_DECODE
        })?;
        // SAFETY: Media Foundation guarantees `data` is readable for `current_length` bytes while locked.
        let bytes = unsafe { std::slice::from_raw_parts(data, len) };
        Ok(Self { buffer, bytes })
    }
}

impl Drop for LockedBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: Paired with Lock on the same IMFMediaBuffer reference.
        let _ = unsafe { self.buffer.Unlock() };
    }
}

pub(super) fn run_worker(arguments: &[OsString]) -> std::process::ExitCode {
    if arguments.len() != 3 || arguments[0] != "--attachment-import-worker" {
        return std::process::ExitCode::FAILURE;
    }
    let Some(kind_str) = arguments[1].to_str() else {
        return std::process::ExitCode::FAILURE;
    };
    let Some(kind) = MediaKind::from_arg(kind_str) else {
        return std::process::ExitCode::FAILURE;
    };
    if !bonsai_windows_credentials::is_app_container().unwrap_or(false) {
        return std::process::ExitCode::FAILURE;
    }
    let path = Path::new(&arguments[2]);
    let mut output = io::stdout().lock();
    let decoded = match kind {
        MediaKind::Image => decode_image(path),
        MediaKind::Video => decode_video(path),
    };
    match decoded {
        Ok(media) => {
            if write_media_packet(
                &mut output,
                kind,
                media.width,
                media.height,
                media.duration_ms,
                &media.thumbnail_png,
                &media.frames,
            )
            .is_err()
            {
                return std::process::ExitCode::FAILURE;
            }
            std::process::ExitCode::SUCCESS
        }
        Err(code) => {
            let _ = write_error_packet(&mut output, code);
            std::process::ExitCode::FAILURE
        }
    }
}

fn decode_image(path: &Path) -> Result<DecodedMedia, u64> {
    let _apartment = crate::com_dialog::Apartment::mta().map_err(|_| ERR_DECODE)?;
    // SAFETY: In-process COM activation of WIC imaging factory on initialized MTA thread.
    let factory: IWICImagingFactory =
        unsafe { CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER) }
            .map_err(|_| ERR_DECODE)?;
    let wide_path = HSTRING::from(path.as_os_str());
    // SAFETY: `wide_path` is a valid NUL-terminated UTF-16 string opened read-only by WIC.
    let decoder = unsafe {
        factory.CreateDecoderFromFilename(
            PCWSTR(wide_path.as_ptr()),
            None,
            GENERIC_READ,
            WICDecodeMetadataCacheOnDemand,
        )
    }
    .map_err(|_| ERR_DECODE)?;
    // SAFETY: Valid IWICBitmapDecoder fetching primary frame 0.
    let frame = unsafe { decoder.GetFrame(0) }.map_err(|_| ERR_DECODE)?;
    let mut src_w = 0_u32;
    let mut src_h = 0_u32;
    // SAFETY: Valid stack output pointers for image frame dimensions.
    unsafe { frame.GetSize(&raw mut src_w, &raw mut src_h) }.map_err(|_| ERR_DECODE)?;
    let (dst_w, dst_h) =
        image::fit_dimensions(src_w, src_h, MAX_IMAGE_EDGE).map_err(|_| ERR_DECODE)?;

    // SAFETY: Creates an owned WIC format converter targeting packed 32bpp RGBA.
    let converter = unsafe { factory.CreateFormatConverter() }.map_err(|_| ERR_DECODE)?;
    // SAFETY: Converts `frame` into straight 32bpp RGBA with no palette or dither.
    unsafe {
        converter.Initialize(
            &frame,
            &GUID_WICPixelFormat32bppRGBA,
            WICBitmapDitherTypeNone,
            None,
            0.0,
            WICBitmapPaletteTypeMedianCut,
        )
    }
    .map_err(|_| ERR_DECODE)?;

    let source: IWICBitmapSource = if dst_w != src_w || dst_h != src_h {
        // SAFETY: Creates an owned WIC bitmap scaler so giant images downscale during streaming copy.
        let scaler = unsafe { factory.CreateBitmapScaler() }.map_err(|_| ERR_DECODE)?;
        // SAFETY: Initializes scaler from live converter with bounded target dimensions.
        unsafe { scaler.Initialize(&converter, dst_w, dst_h, WICBitmapInterpolationModeFant) }
            .map_err(|_| ERR_DECODE)?;
        scaler.cast().map_err(|_| ERR_DECODE)?
    } else {
        converter.cast().map_err(|_| ERR_DECODE)?
    };

    let stride = dst_w.checked_mul(4).ok_or(ERR_MEMORY)?;
    let total_bytes = usize::try_from(stride)
        .ok()
        .and_then(|s| usize::try_from(dst_h).ok().and_then(|h| s.checked_mul(h)))
        .ok_or(ERR_MEMORY)?;
    let mut rgba = vec![0_u8; total_bytes];
    // SAFETY: `rgba` is sized cleanly to `stride * dst_h` writable bytes for this owned source.
    unsafe { source.CopyPixels(ptr::null(), stride, &mut rgba) }.map_err(|_| ERR_DECODE)?;

    let full_png =
        image::encode_rgba_png(&rgba, dst_w, dst_h, MAX_MODEL_PNG_BYTES).map_err(|_| ERR_MEMORY)?;
    let thumbnail_png = build_thumbnail(&rgba, dst_w, dst_h)?;
    Ok(DecodedMedia {
        width: dst_w,
        height: dst_h,
        duration_ms: 0,
        thumbnail_png,
        frames: vec![EncodedFrame {
            timestamp_ms: 0,
            png_bytes: full_png,
        }],
    })
}

fn decode_video(path: &Path) -> Result<DecodedMedia, u64> {
    let _apartment = crate::com_dialog::Apartment::mta().map_err(|_| ERR_DECODE)?;
    let _platform = Platform::start()?;
    let mut attributes: Option<IMFAttributes> = None;
    // SAFETY: Valid stack output pointer receiving a new owned attribute store.
    unsafe { MFCreateAttributes(&raw mut attributes, 1) }.map_err(|_| ERR_DECODE)?;
    let attributes = attributes.ok_or(ERR_DECODE)?;
    // SAFETY: Enables Media Foundation video processor for software YUV->RGB32 conversion.
    unsafe { attributes.SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1) }
        .map_err(|_| ERR_DECODE)?;
    let wide_path = HSTRING::from(path.as_os_str());
    // SAFETY: Synchronous reader for local read-only file opened without network access.
    let reader =
        unsafe { MFCreateSourceReaderFromURL(&wide_path, &attributes) }.map_err(|_| ERR_DECODE)?;

    // SAFETY: Select only the first video stream on this exclusively owned reader.
    unsafe { reader.SetStreamSelection(ALL_STREAMS, false) }.map_err(|_| ERR_NO_VIDEO)?;
    // SAFETY: Sentinel targets the container's primary video stream.
    unsafe { reader.SetStreamSelection(VIDEO_STREAM, true) }.map_err(|_| ERR_NO_VIDEO)?;
    // SAFETY: Creates a clean empty media type specifying unpacked 32-bit RGB frames.
    let wanted = unsafe { MFCreateMediaType() }.map_err(|_| ERR_DECODE)?;
    // SAFETY: Static GUID pointers configure video RGB32 output format.
    unsafe { wanted.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video) }.map_err(|_| ERR_DECODE)?;
    // SAFETY: Static subtype GUID configures BGRX 32bpp pixels.
    unsafe { wanted.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32) }.map_err(|_| ERR_DECODE)?;
    // SAFETY: One-time synchronous output type negotiation before reading samples.
    unsafe { reader.SetCurrentMediaType(VIDEO_STREAM, None, &wanted) }.map_err(|_| ERR_NO_VIDEO)?;

    let (mut src_w, mut src_h, mut stride) = query_video_geometry(&reader)?;
    let (dst_w, dst_h) =
        image::fit_dimensions(src_w, src_h, MAX_VIDEO_FRAME_EDGE).map_err(|_| ERR_DECODE)?;
    let duration_hns = query_duration_hns(&reader);
    let duration_ms = hns_to_ms(duration_hns.unwrap_or(0));
    let targets = duration_hns.map_or_else(Vec::new, sample_targets_hns);

    let mut frames = Vec::new();
    let mut thumbnail_png = None;
    let mut cumulative_png_bytes = 0_usize;

    let mut prev_ts = None;
    let mut append_sample = |timestamp_ms: u64,
                             sample: &IMFSample,
                             src_w: u32,
                             src_h: u32,
                             stride: i32,
                             fail_on_limit: bool|
     -> Result<bool, u64> {
        let (thumb, encoded) = encode_video_sample(
            sample,
            src_w,
            src_h,
            stride,
            dst_w,
            dst_h,
            timestamp_ms,
            thumbnail_png.is_none(),
        )?;
        thumbnail_png = thumbnail_png.take().or(thumb);
        let next_total = cumulative_png_bytes.saturating_add(encoded.png_bytes.len());
        if next_total > MAX_TOTAL_VIDEO_PNG_BYTES {
            return if fail_on_limit {
                Err(ERR_MEMORY)
            } else {
                Ok(false)
            };
        }
        cumulative_png_bytes = next_total;
        frames.push(encoded);
        Ok(true)
    };

    if targets.is_empty() {
        if let Some((ts_hns, sample)) =
            read_next_video_sample(&reader, &mut src_w, &mut src_h, &mut stride)?
        {
            append_sample(hns_to_ms(ts_hns), &sample, src_w, src_h, stride, true)?;
        }
    } else {
        for &target_hns in &targets {
            if !seek_hns(&reader, target_hns) {
                break;
            }
            let Some((ts_hns, sample)) =
                read_next_video_sample(&reader, &mut src_w, &mut src_h, &mut stride)?
            else {
                continue;
            };
            let raw_ms = hns_to_ms(ts_hns);
            let timestamp_ms = prev_ts.map_or(raw_ms, |prev| raw_ms.max(prev));
            if !append_sample(timestamp_ms, &sample, src_w, src_h, stride, false)? {
                break;
            }
            prev_ts = Some(timestamp_ms);
        }
        // Fallback when container index rejects random seeks: rewind to start and decode first keyframe.
        if prev_ts.is_none()
            && seek_hns(&reader, 0)
            && let Some((ts_hns, sample)) =
                read_next_video_sample(&reader, &mut src_w, &mut src_h, &mut stride)?
        {
            let _ = append_sample(hns_to_ms(ts_hns), &sample, src_w, src_h, stride, false)?;
        }
    }

    if frames.is_empty() {
        return Err(ERR_NO_VIDEO);
    }
    let Some(thumbnail_png) = thumbnail_png else {
        return Err(ERR_NO_VIDEO);
    };
    Ok(DecodedMedia {
        width: dst_w,
        height: dst_h,
        duration_ms,
        thumbnail_png,
        frames,
    })
}

fn query_video_geometry(reader: &IMFSourceReader) -> Result<(u32, u32, i32), u64> {
    // SAFETY: Queries the live negotiated video output media type.
    let actual = unsafe { reader.GetCurrentMediaType(VIDEO_STREAM) }.map_err(|_| ERR_DECODE)?;
    // SAFETY: MF_MT_FRAME_SIZE packs (width << 32) | height into a single UINT64 attribute.
    let frame_size = unsafe { actual.GetUINT64(&MF_MT_FRAME_SIZE) }.map_err(|_| ERR_DECODE)?;
    let width = u32::try_from(frame_size >> 32).map_err(|_| ERR_DECODE)?;
    let height = u32::try_from(frame_size & u64::from(u32::MAX)).map_err(|_| ERR_DECODE)?;
    if width == 0 || height == 0 || width > MAX_VIDEO_DIM || height > MAX_VIDEO_DIM {
        return Err(ERR_DECODE);
    }
    // SAFETY: Optional INT32 stride stored as raw 32-bit attribute bits.
    let stride = match unsafe { actual.GetUINT32(&MF_MT_DEFAULT_STRIDE) } {
        Ok(bits) => i32::from_ne_bytes(bits.to_ne_bytes()),
        Err(_) => i32::try_from(width.checked_mul(4).ok_or(ERR_MEMORY)?).map_err(|_| ERR_MEMORY)?,
    };
    Ok((width, height, stride))
}

fn query_duration_hns(reader: &IMFSourceReader) -> Option<i64> {
    // SAFETY: Retrieves presentation duration metadata from the owned media source.
    let prop = unsafe { reader.GetPresentationAttribute(MEDIA_SOURCE, &MF_PD_DURATION) }.ok()?;
    let duration = i64::try_from(&prop).ok()?;
    (duration > 0).then_some(duration)
}

fn seek_hns(reader: &IMFSourceReader, position_hns: i64) -> bool {
    let position = PROPVARIANT::from(position_hns.max(0));
    let guid = GUID::zeroed();
    // SAFETY: `guid` (TIME_FORMAT_NONE = 100ns units) and `position` remain valid for SetCurrentPosition.
    unsafe { reader.SetCurrentPosition(&guid, &position) }.is_ok()
}

fn read_next_video_sample(
    reader: &IMFSourceReader,
    src_w: &mut u32,
    src_h: &mut u32,
    stride: &mut i32,
) -> Result<Option<(i64, IMFSample)>, u64> {
    let eof_flag = MF_SOURCE_READERF_ENDOFSTREAM.0.cast_unsigned();
    let error_flag = MF_SOURCE_READERF_ERROR.0.cast_unsigned();
    let type_changed_flag = MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0.cast_unsigned();
    for _ in 0..MAX_VIDEO_READ_STEPS {
        let mut flags = 0_u32;
        let mut timestamp_hns = 0_i64;
        let mut sample: Option<IMFSample> = None;
        // SAFETY: Synchronous ReadSample passes valid stack pointers for status, timestamp and output sample.
        unsafe {
            reader.ReadSample(
                VIDEO_STREAM,
                0,
                None,
                Some(&raw mut flags),
                Some(&raw mut timestamp_hns),
                Some(&raw mut sample),
            )
        }
        .map_err(|_| ERR_DECODE)?;
        if flags & error_flag != 0 {
            return Err(ERR_DECODE);
        }
        if flags & type_changed_flag != 0 {
            (*src_w, *src_h, *stride) = query_video_geometry(reader)?;
        }
        if let Some(sample) = sample {
            return Ok(Some((timestamp_hns, sample)));
        }
        if flags & eof_flag != 0 {
            return Ok(None);
        }
    }
    Ok(None)
}

#[allow(
    clippy::too_many_arguments,
    reason = "Direct sample conversion inputs keep state flat."
)]
fn encode_video_sample(
    sample: &IMFSample,
    src_w: u32,
    src_h: u32,
    stride: i32,
    dst_w: u32,
    dst_h: u32,
    timestamp_ms: u64,
    with_thumbnail: bool,
) -> Result<(Option<Vec<u8>>, EncodedFrame), u64> {
    // SAFETY: Valid live IMFSample converting to a contiguous locked buffer.
    let buffer = unsafe { sample.ConvertToContiguousBuffer() }.map_err(|_| ERR_DECODE)?;
    let locked = LockedBuffer::lock(&buffer)?;
    let scaled =
        image::downscale_bgrx_stride_to_rgba(locked.bytes, src_w, src_h, stride, dst_w, dst_h)
            .map_err(|_| ERR_DECODE)?;
    drop(locked);
    let png_bytes = image::encode_rgba_png(&scaled, dst_w, dst_h, MAX_MODEL_PNG_BYTES)
        .map_err(|_| ERR_MEMORY)?;
    let thumbnail = if with_thumbnail {
        Some(build_thumbnail(&scaled, dst_w, dst_h)?)
    } else {
        None
    };
    Ok((
        thumbnail,
        EncodedFrame {
            timestamp_ms,
            png_bytes,
        },
    ))
}

fn build_thumbnail(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, u64> {
    let (thumb_w, thumb_h) =
        image::fit_dimensions(width, height, MAX_THUMBNAIL_EDGE).map_err(|_| ERR_DECODE)?;
    let thumb_rgba =
        image::downscale_rgba(rgba, width, height, thumb_w, thumb_h).map_err(|_| ERR_DECODE)?;
    image::encode_rgba_png(&thumb_rgba, thumb_w, thumb_h, MAX_THUMBNAIL_PNG_BYTES)
        .map_err(|_| ERR_MEMORY)
}

fn sample_targets_hns(duration_hns: i64) -> Vec<i64> {
    if duration_hns <= 0 {
        return Vec::new();
    }
    let seconds = duration_hns / 10_000_000;
    let count = if seconds <= 1 {
        2
    } else if seconds <= 4 {
        3
    } else if seconds <= 10 {
        4
    } else {
        i64::try_from(MAX_VIDEO_KEYFRAMES).unwrap_or(6)
    };
    (0..count)
        .map(|index| {
            duration_hns
                .saturating_mul(2 * index + 1)
                .checked_div(2 * count)
                .unwrap_or(0)
        })
        .collect()
}

fn hns_to_ms(hns: i64) -> u64 {
    if hns <= 0 {
        return 0;
    }
    let ms = Duration::from_nanos(u64::try_from(hns).unwrap_or(0).saturating_mul(100)).as_millis();
    u64::try_from(ms).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{hns_to_ms, sample_targets_hns};

    #[test]
    fn keyframe_targets_scale_with_duration_and_avoid_endpoints() {
        let short = sample_targets_hns(10_000_000);
        assert_eq!(short, vec![2_500_000, 7_500_000]);

        let medium = sample_targets_hns(30_000_000);
        assert_eq!(medium.len(), 3);
        assert!(medium.windows(2).all(|pair| pair[0] < pair[1]));

        let long = sample_targets_hns(300_000_000);
        assert_eq!(long.len(), 6);
        assert!(long[0] > 0 && long[5] < 300_000_000);
        assert_eq!(hns_to_ms(25_000_000), 2_500);
        assert_eq!(hns_to_ms(-5), 0);
    }
}
