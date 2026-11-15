//! Fixed binary framing for the persistent offline worker.
use std::io::{self, Read, Write};

const REQUEST_MAGIC: &[u8; 8] = b"BSMD0001";
const RESPONSE_MAGIC: &[u8; 8] = b"BSMR0001";
const HEADER_BYTES: usize = 20;
const MAX_OUTPUT_BYTES: usize = 2 * 1024 * 1024;

fn write_frame(output: &mut impl Write, ticket: u64, bytes: &[u8]) -> io::Result<()> {
    let length = u32::try_from(bytes.len()).map_err(io::Error::other)?;
    output.write_all(RESPONSE_MAGIC)?;
    output.write_all(&ticket.to_le_bytes())?;
    output.write_all(&length.to_le_bytes())?;
    output.write_all(bytes)?;
    output.flush()
}

fn read_frame(input: &mut impl Read) -> io::Result<(u64, Vec<u8>)> {
    let mut header = [0; HEADER_BYTES];
    input.read_exact(&mut header)?;
    let Some((magic, rest)) = header.split_first_chunk::<8>() else {
        return Err(io::Error::other("Invalid Markdown frame identity"));
    };
    if magic[..] != REQUEST_MAGIC[..] {
        return Err(io::Error::other("Invalid Markdown frame identity"));
    }
    let Some((identity, rest)) = rest.split_first_chunk::<8>() else {
        return Err(io::Error::other("Invalid Markdown frame identity"));
    };
    let Some((length, &[])) = rest.split_first_chunk::<4>() else {
        return Err(io::Error::other("Invalid Markdown frame length"));
    };
    let length = usize::try_from(u32::from_le_bytes(*length)).map_err(io::Error::other)?;
    if length > super::MAX_MARKDOWN_BYTES {
        return Err(io::Error::other("Oversized Markdown frame"));
    }
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes)?;
    Ok((u64::from_le_bytes(*identity), bytes))
}

/// Serves bounded binary Markdown frames until input, output or parsing fails.
///
/// The caller must establish native containment and own its pipes and lifetime
/// before invoking this function. A zero-ticket empty response announces ready;
/// each nonzero request receives exactly one matching response and a flush.
///
/// # Errors
/// Rejects invalid identity, zero request tickets, oversized or truncated frames,
/// invalid UTF-8, oversized results and failed pipe reads, writes or flushes.
pub fn serve(input: &mut impl Read, output: &mut impl Write) -> io::Result<()> {
    write_frame(output, 0, &[])?;
    loop {
        let (ticket, bytes) = read_frame(input)?;
        if ticket == 0 {
            return Err(io::Error::other("Invalid Markdown request identity"));
        }
        let source = std::str::from_utf8(&bytes).map_err(io::Error::other)?;
        let rendered = super::parse(source);
        if rendered.len() > MAX_OUTPUT_BYTES {
            return Err(io::Error::other("Oversized Markdown result"));
        }
        write_frame(output, ticket, &rendered)?;
    }
}
