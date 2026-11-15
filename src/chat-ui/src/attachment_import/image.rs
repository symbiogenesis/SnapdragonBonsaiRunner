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
    let (src_pixels, _) = rgba.as_chunks::<4>();
    let mut x_spans = Vec::with_capacity(dst_w_usize);
    for dx in 0..dst_w {
        let sx0 = usize::try_from(u64::from(dx) * u64::from(src_w) / u64::from(dst_w))
            .map_err(|_| INVALID_IMAGE)?;
        let sx1_raw =
            usize::try_from(((u64::from(dx) + 1) * u64::from(src_w)).div_ceil(u64::from(dst_w)))
                .map_err(|_| INVALID_IMAGE)?;
        let sx1 = sx1_raw.max(sx0 + 1).min(src_w_usize);
        let span_w = u64::try_from(sx1 - sx0).map_err(|_| INVALID_IMAGE)?;
        x_spans.push((sx0, sx1, span_w));
    }

    let mut output = vec![0_u8; out_len];
    let dst_row_bytes = dst_w_usize * 4;
    let mut row_sums = vec![[0_u64; 4]; dst_w_usize];
    for (dy, dst_row) in output.chunks_exact_mut(dst_row_bytes).enumerate() {
        let dy_u64 = u64::try_from(dy).map_err(|_| INVALID_IMAGE)?;
        let sy0 = usize::try_from(dy_u64 * u64::from(src_h) / u64::from(dst_h))
            .map_err(|_| INVALID_IMAGE)?;
        let sy1_raw = usize::try_from(((dy_u64 + 1) * u64::from(src_h)).div_ceil(u64::from(dst_h)))
            .map_err(|_| INVALID_IMAGE)?;
        let sy1 = sy1_raw.max(sy0 + 1).min(src_h_usize);
        row_sums.fill([0_u64; 4]);
        for sy in sy0..sy1 {
            let src_row = &src_pixels[sy * src_w_usize..(sy + 1) * src_w_usize];
            for (&(sx0, sx1, _), acc) in x_spans.iter().zip(&mut row_sums) {
                let mut span_sum = [0_u32; 4];
                for &[r, g, b, a] in &src_row[sx0..sx1] {
                    span_sum[0] += u32::from(r);
                    span_sum[1] += u32::from(g);
                    span_sum[2] += u32::from(b);
                    span_sum[3] += u32::from(a);
                }
                for c in 0..4 {
                    acc[c] += u64::from(span_sum[c]);
                }
            }
        }
        let span_h = u64::try_from(sy1 - sy0).map_err(|_| INVALID_IMAGE)?;
        let (dst_pixels, _) = dst_row.as_chunks_mut::<4>();
        for (dst_pixel, (acc, &(_, _, span_w))) in
            dst_pixels.iter_mut().zip(row_sums.iter().zip(&x_spans))
        {
            let count = span_h * span_w;
            let half = count / 2;
            *dst_pixel = [
                u8::try_from((acc[0] + half) / count).map_err(|_| INVALID_IMAGE)?,
                u8::try_from((acc[1] + half) / count).map_err(|_| INVALID_IMAGE)?,
                u8::try_from((acc[2] + half) / count).map_err(|_| INVALID_IMAGE)?,
                u8::try_from((acc[3] + half) / count).map_err(|_| INVALID_IMAGE)?,
            ];
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
    let mut out = vec![0_u8; row_bytes.checked_mul(h).ok_or(INVALID_IMAGE)?];
    let bottom_up = stride < 0;
    for (y, dst_row) in out.chunks_exact_mut(row_bytes).enumerate() {
        let src_row_index = if bottom_up { h - 1 - y } else { y };
        let start = src_row_index * abs_stride;
        let (src_pixels, _) = src[start..start + row_bytes].as_chunks::<4>();
        let (dst_pixels, _) = dst_row.as_chunks_mut::<4>();
        for (dst, &[b, g, r, _]) in dst_pixels.iter_mut().zip(src_pixels) {
            *dst = [r, g, b, 255];
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

    fn direct_box_average_reference(
        rgba: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Vec<u8> {
        let mut expected = Vec::with_capacity(usize::try_from(dst_w * dst_h * 4).unwrap_or(0));
        let src_w_usize = usize::try_from(src_w).unwrap_or(0);
        let src_h_usize = usize::try_from(src_h).unwrap_or(0);
        for dy in 0..dst_h {
            let sy0 =
                usize::try_from(u64::from(dy) * u64::from(src_h) / u64::from(dst_h)).unwrap_or(0);
            let sy1 = usize::try_from(
                ((u64::from(dy) + 1) * u64::from(src_h)).div_ceil(u64::from(dst_h)),
            )
            .unwrap_or(0)
            .max(sy0 + 1)
            .min(src_h_usize);
            for dx in 0..dst_w {
                let sx0 = usize::try_from(u64::from(dx) * u64::from(src_w) / u64::from(dst_w))
                    .unwrap_or(0);
                let sx1 = usize::try_from(
                    ((u64::from(dx) + 1) * u64::from(src_w)).div_ceil(u64::from(dst_w)),
                )
                .unwrap_or(0)
                .max(sx0 + 1)
                .min(src_w_usize);
                let mut sum = [0_u64; 4];
                let mut count = 0_u64;
                for sy in sy0..sy1 {
                    for sx in sx0..sx1 {
                        let base = (sy * src_w_usize + sx) * 4;
                        for c in 0..4 {
                            sum[c] += u64::from(rgba[base + c]);
                        }
                        count += 1;
                    }
                }
                for channel in sum {
                    expected.push(u8::try_from((channel + count / 2) / count).unwrap_or(0));
                }
            }
        }
        expected
    }

    #[test]
    fn scanline_downscaling_matches_direct_2d_box_average_bit_for_bit() {
        for (src_w, src_h, dst_w, dst_h) in [
            (37_u32, 29_u32, 11_u32, 7_u32),
            (53, 41, 13, 9),
            (192, 108, 64, 36),
        ] {
            let mut src = Vec::with_capacity(usize::try_from(src_w * src_h * 4).unwrap_or(0));
            for idx in 0..(src_w * src_h) {
                let r = u8::try_from((idx.wrapping_mul(73) ^ (idx >> 3)) & 0xff).unwrap_or(0);
                let g = u8::try_from((idx.wrapping_mul(139) + 17) & 0xff).unwrap_or(0);
                let b = u8::try_from((255 - (idx % 256)) ^ 0x5a).unwrap_or(0);
                let a = u8::try_from((idx.wrapping_mul(31) + 200) & 0xff).unwrap_or(0);
                src.extend_from_slice(&[r, g, b, a]);
            }
            let actual = downscale_rgba(&src, src_w, src_h, dst_w, dst_h).unwrap_or_default();
            let expected = direct_box_average_reference(&src, src_w, src_h, dst_w, dst_h);
            assert_eq!(actual, expected, "{src_w}x{src_h} -> {dst_w}x{dst_h}");
        }
    }
}
