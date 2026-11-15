//! Bounded RGBA scaling, stride normalization and PNG validation helpers.
#![forbid(unsafe_code)]

use std::io::Cursor;

use base64ct::{Base64, Encoding as _};

const MAX_SOURCE_DIMENSION: u32 = 16_384;
pub(super) const INVALID_IMAGE: &str = "Could not process this image or video frame.";

pub(super) fn fit_dimensions(
    width: u32,
    height: u32,
    max_edge: u32,
) -> Result<(u32, u32), &'static str> {
    if width == 0
        || height == 0
        || max_edge == 0
        || width > MAX_SOURCE_DIMENSION
        || height > MAX_SOURCE_DIMENSION
    {
        return Err(INVALID_IMAGE);
    }
    if width <= max_edge && height <= max_edge {
        return Ok((width, height));
    }
    if width >= height {
        let scaled_h = ((u64::from(height) * u64::from(max_edge) + u64::from(width) / 2)
            / u64::from(width))
        .max(1);
        let out_h = u32::try_from(scaled_h).map_err(|_| INVALID_IMAGE)?;
        Ok((max_edge, out_h))
    } else {
        let scaled_w = ((u64::from(width) * u64::from(max_edge) + u64::from(height) / 2)
            / u64::from(height))
        .max(1);
        let out_w = u32::try_from(scaled_w).map_err(|_| INVALID_IMAGE)?;
        Ok((out_w, max_edge))
    }
}

pub(super) fn downscale_rgba(
    rgba: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, &'static str> {
    if src_w == 0
        || src_h == 0
        || dst_w == 0
        || dst_h == 0
        || dst_w > src_w
        || dst_h > src_h
        || src_w > MAX_SOURCE_DIMENSION
        || src_h > MAX_SOURCE_DIMENSION
    {
        return Err(INVALID_IMAGE);
    }
    let src_w_usize = usize::try_from(src_w).map_err(|_| INVALID_IMAGE)?;
    let src_h_usize = usize::try_from(src_h).map_err(|_| INVALID_IMAGE)?;
    let dst_w_usize = usize::try_from(dst_w).map_err(|_| INVALID_IMAGE)?;
    let dst_h_usize = usize::try_from(dst_h).map_err(|_| INVALID_IMAGE)?;
    let expected = src_w_usize
        .checked_mul(src_h_usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(INVALID_IMAGE)?;
    if rgba.len() != expected {
        return Err(INVALID_IMAGE);
    }
    if dst_w == src_w && dst_h == src_h {
        return Ok(rgba.to_vec());
    }
    let out_len = dst_w_usize
        .checked_mul(dst_h_usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(INVALID_IMAGE)?;
    let mut output = Vec::with_capacity(out_len);
    for dy in 0..dst_h {
        let sy0 = usize::try_from(u64::from(dy) * u64::from(src_h) / u64::from(dst_h))
            .map_err(|_| INVALID_IMAGE)?;
        let sy1_raw =
            usize::try_from(((u64::from(dy) + 1) * u64::from(src_h)).div_ceil(u64::from(dst_h)))
                .map_err(|_| INVALID_IMAGE)?;
        let sy1 = sy1_raw.max(sy0 + 1).min(src_h_usize);
        for dx in 0..dst_w {
            let sx0 = usize::try_from(u64::from(dx) * u64::from(src_w) / u64::from(dst_w))
                .map_err(|_| INVALID_IMAGE)?;
            let sx1_raw = usize::try_from(
                ((u64::from(dx) + 1) * u64::from(src_w)).div_ceil(u64::from(dst_w)),
            )
            .map_err(|_| INVALID_IMAGE)?;
            let sx1 = sx1_raw.max(sx0 + 1).min(src_w_usize);
            let mut sum = [0_u64; 4];
            let mut count = 0_u64;
            for sy in sy0..sy1 {
                let row_offset = sy * src_w_usize * 4;
                for sx in sx0..sx1 {
                    let base = row_offset + sx * 4;
                    sum[0] += u64::from(rgba[base]);
                    sum[1] += u64::from(rgba[base + 1]);
                    sum[2] += u64::from(rgba[base + 2]);
                    sum[3] += u64::from(rgba[base + 3]);
                    count += 1;
                }
            }
            for channel in sum {
                let averaged =
                    u8::try_from((channel + count / 2) / count).map_err(|_| INVALID_IMAGE)?;
                output.push(averaged);
            }
        }
    }
    Ok(output)
}

#[allow(dead_code, reason = "Used by AppContainer worker and unit tests.")]
pub(super) fn bgrx_stride_to_rgba(
    src: &[u8],
    width: u32,
    height: u32,
    stride: i32,
) -> Result<Vec<u8>, &'static str> {
    if width == 0 || height == 0 || width > MAX_SOURCE_DIMENSION || height > MAX_SOURCE_DIMENSION {
        return Err(INVALID_IMAGE);
    }
    let w = usize::try_from(width).map_err(|_| INVALID_IMAGE)?;
    let h = usize::try_from(height).map_err(|_| INVALID_IMAGE)?;
    let row_bytes = w.checked_mul(4).ok_or(INVALID_IMAGE)?;
    let abs_stride = usize::try_from(stride.unsigned_abs()).map_err(|_| INVALID_IMAGE)?;
    if abs_stride < row_bytes {
        return Err(INVALID_IMAGE);
    }
    let required = abs_stride
        .checked_mul(h.saturating_sub(1))
        .and_then(|tail| tail.checked_add(row_bytes))
        .ok_or(INVALID_IMAGE)?;
    if src.len() < required {
        return Err(INVALID_IMAGE);
    }
    let mut out = Vec::with_capacity(row_bytes.checked_mul(h).ok_or(INVALID_IMAGE)?);
    let bottom_up = stride < 0;
    for y in 0..h {
        let src_row_index = if bottom_up { h - 1 - y } else { y };
        let start = src_row_index * abs_stride;
        let row = &src[start..start + row_bytes];
        for pixel in row.chunks_exact(4) {
            out.push(pixel[2]);
            out.push(pixel[1]);
            out.push(pixel[0]);
            out.push(255);
        }
    }
    Ok(out)
}

#[allow(dead_code, reason = "Used by AppContainer worker and unit tests.")]
pub(super) fn encode_rgba_png(
    rgba: &[u8],
    width: u32,
    height: u32,
    max_bytes: usize,
) -> Result<Vec<u8>, &'static str> {
    if width == 0 || height == 0 || width > MAX_SOURCE_DIMENSION || height > MAX_SOURCE_DIMENSION {
        return Err(INVALID_IMAGE);
    }
    let expected = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(INVALID_IMAGE)?;
    if rgba.len() != expected {
        return Err(INVALID_IMAGE);
    }
    let mut output = Vec::new();
    let mut encoder = png::Encoder::new(&mut output, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|_| INVALID_IMAGE)?;
    writer.write_image_data(rgba).map_err(|_| INVALID_IMAGE)?;
    writer.finish().map_err(|_| INVALID_IMAGE)?;
    if output.is_empty() || output.len() > max_bytes {
        return Err(INVALID_IMAGE);
    }
    Ok(output)
}

pub(super) fn validate_png_bytes(
    bytes: &[u8],
    max_edge: u32,
    max_bytes: usize,
) -> Result<(u32, u32), &'static str> {
    if bytes.is_empty() || bytes.len() > max_bytes {
        return Err(INVALID_IMAGE);
    }
    let reader = png::Decoder::new(Cursor::new(bytes))
        .read_info()
        .map_err(|_| INVALID_IMAGE)?;
    let info = reader.info();
    if info.width == 0
        || info.height == 0
        || info.width > max_edge
        || info.height > max_edge
        || info.color_type != png::ColorType::Rgba
        || info.bit_depth != png::BitDepth::Eight
    {
        return Err(INVALID_IMAGE);
    }
    Ok((info.width, info.height))
}

pub(super) fn png_to_data_url(png_bytes: &[u8]) -> String {
    format!("data:image/png;base64,{}", Base64::encode_string(png_bytes))
}

#[cfg(test)]
mod tests {
    use super::{
        bgrx_stride_to_rgba, downscale_rgba, encode_rgba_png, fit_dimensions, png_to_data_url,
        validate_png_bytes,
    };

    #[test]
    fn downscaling_preserves_aspect_ratio_and_produces_valid_data_url_png() {
        assert_eq!(fit_dimensions(1920, 1080, 1280), Ok((1280, 720)));
        assert_eq!(fit_dimensions(800, 1600, 768), Ok((384, 768)));
        assert_eq!(fit_dimensions(320, 240, 1280), Ok((320, 240)));
        assert!(fit_dimensions(0, 100, 256).is_err());
        assert!(fit_dimensions(20_000, 100, 1280).is_err());

        let mut src = Vec::with_capacity(4 * 4 * 4);
        for index in 0..16_u8 {
            src.extend_from_slice(&[index * 10, 128, 255 - index * 10, 255]);
        }
        let scaled = downscale_rgba(&src, 4, 4, 2, 2).unwrap_or_default();
        assert_eq!(scaled.len(), 2 * 2 * 4);
        let png = encode_rgba_png(&scaled, 2, 2, 64 * 1024).unwrap_or_default();
        assert_eq!(validate_png_bytes(&png, 256, 64 * 1024), Ok((2, 2)));
        assert!(validate_png_bytes(&png, 1, 64 * 1024).is_err());
        let data_url = png_to_data_url(&png);
        assert!(data_url.starts_with("data:image/png;base64,iVBOR"));
    }

    #[test]
    fn negative_stride_bgrx_flips_rows_and_forces_opaque_alpha() {
        // 2x2 image stored bottom-up with 4 padding bytes per row (stride = -12).
        let src = [
            // Bottom row in file (top row of output): blue=(255,0,0,0), green=(0,255,0,0), pad(0..4)
            255, 0, 0, 0, 0, 255, 0, 0, 99, 99, 99, 99,
            // Top row in file (bottom row of output): red=(0,0,255,0), white=(255,255,255,0)
            0, 0, 255, 0, 255, 255, 255, 0,
        ];
        let rgba = bgrx_stride_to_rgba(&src, 2, 2, -12).unwrap_or_default();
        assert_eq!(
            rgba,
            [
                255, 0, 0, 255, 255, 255, 255, 255, // top row: red, white
                0, 0, 255, 255, 0, 255, 0, 255, // bottom row: blue, green
            ]
        );
    }
}
