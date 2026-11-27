//! Fixed binary framing for the persistent offline parser and SVG rasterizer worker.
use std::io::{self, Read, Write};

const MARKDOWN_MAGIC: &[u8; 8] = b"BSMD0001";
const RASTER_MAGIC: &[u8; 8] = b"BSMG0001";
const RESPONSE_MAGIC: &[u8; 8] = b"BSMR0001";
const HEADER_BYTES: usize = 20;
const SCRATCH_LIMIT_BYTES: usize = 64 * 1024;
const MAX_OUTPUT_BYTES: usize = 2 * 1024 * 1024;
const MAX_RASTER_REQUEST_BYTES: usize = 8 + super::svg::MAX_SVG_BYTES;
// Four bytes per pixel (premultiplied RGBA8) plus the 13-byte BMG1 header.
const MAX_RASTER_OUTPUT_BYTES: usize =
    super::svg::RASTER_HEADER_BYTES + super::svg::MAX_RASTER_PIXELS * 4;

enum RequestKind {
    Markdown(String),
    Rasterize {
        max_width: u32,
        max_height: u32,
        svg: String,
    },
}

fn write_frame(
    output: &mut impl Write,
    ticket: u64,
    bytes: &[u8],
    scratch: &mut Vec<u8>,
) -> io::Result<()> {
    let length = u32::try_from(bytes.len()).map_err(io::Error::other)?;
    if bytes.len() > SCRATCH_LIMIT_BYTES {
        if scratch.capacity() > HEADER_BYTES + SCRATCH_LIMIT_BYTES {
            scratch.shrink_to(HEADER_BYTES + SCRATCH_LIMIT_BYTES);
        }
        let mut header = [0_u8; HEADER_BYTES];
        header[..8].copy_from_slice(RESPONSE_MAGIC);
        header[8..16].copy_from_slice(&ticket.to_le_bytes());
        header[16..20].copy_from_slice(&length.to_le_bytes());
        output.write_all(&header)?;
        output.write_all(bytes)?;
        return output.flush();
    }
    scratch.clear();
    scratch.reserve(HEADER_BYTES + bytes.len());
    scratch.extend_from_slice(RESPONSE_MAGIC);
    scratch.extend_from_slice(&ticket.to_le_bytes());
    scratch.extend_from_slice(&length.to_le_bytes());
    scratch.extend_from_slice(bytes);
    output.write_all(scratch)?;
    output.flush()
}

fn read_frame(input: &mut impl Read) -> io::Result<(u64, RequestKind)> {
    let mut header = [0; HEADER_BYTES];
    input.read_exact(&mut header)?;
    let Some((magic, rest)) = header.split_first_chunk::<8>() else {
        return Err(io::Error::other("Invalid Markdown frame identity"));
    };
    let is_markdown = magic[..] == MARKDOWN_MAGIC[..];
    let is_raster = !is_markdown && magic[..] == RASTER_MAGIC[..];
    if !is_markdown && !is_raster {
        return Err(io::Error::other("Invalid Markdown frame identity"));
    }
    let Some((identity, rest)) = rest.split_first_chunk::<8>() else {
        return Err(io::Error::other("Invalid Markdown frame identity"));
    };
    let ticket = u64::from_le_bytes(*identity);
    if ticket == 0 {
        return Err(io::Error::other("Invalid Markdown request identity"));
    }
    let Some((length, &[])) = rest.split_first_chunk::<4>() else {
        return Err(io::Error::other("Invalid Markdown frame length"));
    };
    let length = usize::try_from(u32::from_le_bytes(*length)).map_err(io::Error::other)?;
    let limit = if is_markdown {
        super::MAX_MARKDOWN_BYTES
    } else {
        MAX_RASTER_REQUEST_BYTES
    };
    if length > limit || (is_raster && length < 8) {
        return Err(io::Error::other("Oversized Markdown frame"));
    }
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes)?;
    if is_markdown {
        let source = String::from_utf8(bytes).map_err(io::Error::other)?;
        Ok((ticket, RequestKind::Markdown(source)))
    } else {
        let Some((dimensions, svg_bytes)) = bytes.split_first_chunk::<8>() else {
            return Err(io::Error::other("Invalid SVG raster frame"));
        };
        let Some((width_bytes, rest)) = dimensions.split_first_chunk::<4>() else {
            return Err(io::Error::other("Invalid SVG raster frame"));
        };
        let Some((height_bytes, _)) = rest.split_first_chunk::<4>() else {
            return Err(io::Error::other("Invalid SVG raster frame"));
        };
        let max_width = u32::from_le_bytes(*width_bytes);
        let max_height = u32::from_le_bytes(*height_bytes);
        let svg = std::str::from_utf8(svg_bytes)
            .map_err(io::Error::other)?
            .to_owned();
        Ok((
            ticket,
            RequestKind::Rasterize {
                max_width,
                max_height,
                svg,
            },
        ))
    }
}

/// Serves bounded binary Markdown and SVG raster frames until input, output or parsing fails.
///
/// The caller must establish native containment and own its pipes and lifetime
/// before invoking this function. A zero-ticket empty response announces ready;
/// each nonzero request receives exactly one matching response and a flush.
///
/// # Errors
/// Rejects invalid identity, zero request tickets, oversized or truncated frames,
/// invalid UTF-8, oversized results and failed pipe reads, writes or flushes.
pub fn serve(input: &mut impl Read, output: &mut impl Write) -> io::Result<()> {
    let mut scratch = Vec::with_capacity(HEADER_BYTES);
    write_frame(output, 0, &[], &mut scratch)?;
    loop {
        let (ticket, request) = read_frame(input)?;
        match request {
            RequestKind::Markdown(source) => {
                let rendered = super::parse(&source);
                if rendered.len() > MAX_OUTPUT_BYTES {
                    return Err(io::Error::other("Oversized Markdown result"));
                }
                write_frame(output, ticket, &rendered, &mut scratch)?;
            }
            RequestKind::Rasterize {
                max_width,
                max_height,
                svg,
            } => {
                let raster = super::svg::rasterize(&svg, max_width, max_height);
                if raster.len() > MAX_RASTER_OUTPUT_BYTES {
                    return Err(io::Error::other("Oversized SVG raster result"));
                }
                write_frame(output, ticket, &raster, &mut scratch)?;
            }
        }
    }
}
