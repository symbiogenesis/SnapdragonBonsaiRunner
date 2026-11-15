//! Trusted validation for untrusted offline `BMG1` SVG rasterization packets.
#![forbid(unsafe_code)]

const RASTER_MAGIC: &[u8; 4] = b"BMG1";
const RASTER_HEADER_BYTES: usize = 13;

/// Maximum rasterized edge length accepted from the offline worker.
pub(crate) const MAX_RASTER_DIMENSION: u32 = 2_560;
/// Maximum total rasterized pixel count (`2048 x 2048`).
pub(crate) const MAX_RASTER_PIXELS: usize = 4_194_304;
/// Maximum bounded `BSMR0001` payload length for a `BMG1` bitmap packet.
pub(super) const MAX_RASTER_OUTPUT_BYTES: usize = RASTER_HEADER_BYTES + MAX_RASTER_PIXELS * 4;

/// Validated premultiplied RGBA8 bitmap decoded from an untrusted `BMG1` packet.
pub(crate) struct DecodedRaster<'a> {
    /// Image width in pixels (`1..=2_560`).
    pub(crate) width: u32,
    /// Image height in pixels (`1..=2_560`).
    pub(crate) height: u32,
    /// Exact `width * height * 4` premultiplied RGBA8 pixels (`r <= a`, `g <= a`, `b <= a`).
    pub(crate) premultiplied_rgba: &'a [u8],
}

/// Returned when a `BMG1` response violates framing, dimension, length, or pixel invariants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InvalidRasterWire;

/// Validates an untrusted `BMG1` packet returned by the offline Markdown worker.
///
/// Returns `Ok(None)` when the worker explicitly reported an unrenderable SVG
/// (`ok == 0` with zero dimensions and no pixel payload), allowing the UI to fall
/// back to source without tearing down the worker session.
pub(crate) fn decode_raster(bytes: &[u8]) -> Result<Option<DecodedRaster<'_>>, InvalidRasterWire> {
    let (header, payload) = bytes
        .split_first_chunk::<RASTER_HEADER_BYTES>()
        .ok_or(InvalidRasterWire)?;
    if &header[..4] != RASTER_MAGIC {
        return Err(InvalidRasterWire);
    }
    let ok = header[4];
    let width = u32::from_le_bytes([header[5], header[6], header[7], header[8]]);
    let height = u32::from_le_bytes([header[9], header[10], header[11], header[12]]);
    match ok {
        0 => {
            if width == 0 && height == 0 && payload.is_empty() {
                Ok(None)
            } else {
                Err(InvalidRasterWire)
            }
        }
        1 => {
            if !(1..=MAX_RASTER_DIMENSION).contains(&width)
                || !(1..=MAX_RASTER_DIMENSION).contains(&height)
            {
                return Err(InvalidRasterWire);
            }
            let width_px = usize::try_from(width).map_err(|_| InvalidRasterWire)?;
            let height_px = usize::try_from(height).map_err(|_| InvalidRasterWire)?;
            let pixels = width_px.checked_mul(height_px).ok_or(InvalidRasterWire)?;
            if pixels > MAX_RASTER_PIXELS {
                return Err(InvalidRasterWire);
            }
            let expected_len = pixels.checked_mul(4).ok_or(InvalidRasterWire)?;
            if payload.len() != expected_len {
                return Err(InvalidRasterWire);
            }
            for chunk in payload.chunks_exact(4) {
                let alpha = chunk[3];
                if chunk[0] > alpha || chunk[1] > alpha || chunk[2] > alpha {
                    return Err(InvalidRasterWire);
                }
            }
            Ok(Some(DecodedRaster {
                width,
                height,
                premultiplied_rgba: payload,
            }))
        }
        _ => Err(InvalidRasterWire),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(ok: u8, width: u32, height: u32, payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(RASTER_HEADER_BYTES + payload.len());
        bytes.extend_from_slice(RASTER_MAGIC);
        bytes.push(ok);
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn valid_unrenderable_packet_returns_ok_none() -> Result<(), InvalidRasterWire> {
        assert!(decode_raster(&packet(0, 0, 0, &[]))?.is_none());
        Ok(())
    }

    #[test]
    fn valid_premultiplied_packet_decodes_pixels() -> Result<(), &'static str> {
        let raw = [
            0_u8, 0, 0, 0, 128, 64, 32, 200, 255, 255, 255, 255, 10, 10, 10, 10,
        ];
        let bytes = packet(1, 2, 2, &raw);
        let decoded = decode_raster(&bytes)
            .map_err(|_| "valid packet rejected")?
            .ok_or("expected bitmap")?;
        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.premultiplied_rgba, &raw);
        Ok(())
    }

    #[test]
    fn rejects_malformed_headers_non_premultiplied_pixels_and_bad_lengths() {
        let bad_magic = b"NOPE\0\0\0\0\0\0\0\0\0";
        assert_eq!(decode_raster(bad_magic).err(), Some(InvalidRasterWire));
        assert_eq!(
            decode_raster(&packet(0, 1, 0, &[])).err(),
            Some(InvalidRasterWire)
        );
        assert_eq!(
            decode_raster(&packet(0, 0, 0, &[1])).err(),
            Some(InvalidRasterWire)
        );
        assert_eq!(
            decode_raster(&packet(2, 0, 0, &[])).err(),
            Some(InvalidRasterWire)
        );
        assert_eq!(
            decode_raster(&packet(1, 0, 1, &[])).err(),
            Some(InvalidRasterWire)
        );
        assert_eq!(
            decode_raster(&packet(1, MAX_RASTER_DIMENSION + 1, 1, &[])).err(),
            Some(InvalidRasterWire)
        );
        assert_eq!(
            decode_raster(&packet(1, 2_560, 2_560, &[])).err(),
            Some(InvalidRasterWire)
        );
        assert_eq!(
            decode_raster(&packet(1, 1, 1, &[1, 2, 3])).err(),
            Some(InvalidRasterWire)
        );
        // Non-premultiplied channel exceeds alpha.
        assert_eq!(
            decode_raster(&packet(1, 1, 1, &[200, 10, 10, 100])).err(),
            Some(InvalidRasterWire)
        );
    }
}
