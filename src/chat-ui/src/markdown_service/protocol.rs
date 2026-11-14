//! Fixed bounded framing, independent of the untrusted parser's output format.
use super::{MAX_OUTPUT_BYTES, Request, Ticket};
use std::{
    io::{self, Read, Write},
    sync::mpsc::{Receiver, SyncSender},
};

const REQUEST_MAGIC: &[u8; 8] = b"BSMD0001";
const RESPONSE_MAGIC: &[u8; 8] = b"BSMR0001";
const HEADER_BYTES: usize = 20;

pub(super) enum Event {
    Ready,
    Response(Ticket, Vec<u8>),
}

fn write_frame(
    output: &mut impl Write,
    magic: &[u8; 8],
    ticket: Ticket,
    bytes: &[u8],
) -> io::Result<()> {
    let length = u32::try_from(bytes.len()).map_err(io::Error::other)?;
    output.write_all(magic)?;
    output.write_all(&ticket.0.to_le_bytes())?;
    output.write_all(&length.to_le_bytes())?;
    output.write_all(bytes)?;
    output.flush()
}

fn read_frame(
    input: &mut impl Read,
    magic: &[u8; 8],
    limit: usize,
) -> io::Result<(Ticket, Vec<u8>)> {
    let mut header = [0; HEADER_BYTES];
    input.read_exact(&mut header)?;
    if header[..8] != magic[..] {
        return Err(io::Error::other("Invalid Markdown frame identity"));
    }
    let mut identity = [0; 8];
    identity.copy_from_slice(&header[8..16]);
    let mut length = [0; 4];
    length.copy_from_slice(&header[16..]);
    let length = usize::try_from(u32::from_le_bytes(length)).map_err(io::Error::other)?;
    if length > limit {
        return Err(io::Error::other("Oversized Markdown frame"));
    }
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes)?;
    Ok((Ticket(u64::from_le_bytes(identity)), bytes))
}

pub(super) fn exchange(
    mut input: impl Write,
    mut output: impl Read,
    commands: &Receiver<Request>,
    events: &SyncSender<Event>,
) {
    let Ok((Ticket(0), ready)) = read_frame(&mut output, RESPONSE_MAGIC, 0) else {
        return;
    };
    if !ready.is_empty() || events.send(Event::Ready).is_err() {
        return;
    }
    while let Ok(request) = commands.recv() {
        if write_frame(
            &mut input,
            REQUEST_MAGIC,
            request.ticket,
            request.source.as_bytes(),
        )
        .is_err()
        {
            return;
        }
        let Ok((ticket, bytes)) = read_frame(&mut output, RESPONSE_MAGIC, MAX_OUTPUT_BYTES) else {
            return;
        };
        if ticket != request.ticket || events.send(Event::Response(ticket, bytes)).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Cursor,
        sync::{Arc, mpsc},
    };

    fn header(magic: &[u8; 8], ticket: u64, length: u32) -> Vec<u8> {
        let mut bytes = Vec::from(*magic);
        bytes.extend_from_slice(&ticket.to_le_bytes());
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes
    }

    #[test]
    fn markdown_protocol_rejects_bad_identity_oversize_and_truncated_frames() {
        let bad = header(b"MALFORM!", 1, 0);
        assert!(read_frame(&mut Cursor::new(bad), RESPONSE_MAGIC, MAX_OUTPUT_BYTES).is_err());
        let oversized = header(RESPONSE_MAGIC, 1, 2_097_153);
        let mut oversized = Cursor::new(oversized);
        assert!(read_frame(&mut oversized, RESPONSE_MAGIC, MAX_OUTPUT_BYTES).is_err());
        assert_eq!(oversized.position(), 20);
        let mut short = header(RESPONSE_MAGIC, 1, 2);
        short.push(7);
        assert!(
            read_frame(&mut Cursor::new(short), RESPONSE_MAGIC, MAX_OUTPUT_BYTES)
                .is_err_and(|error| error.kind() == io::ErrorKind::UnexpectedEof)
        );
    }

    #[test]
    fn markdown_protocol_binds_response_to_exact_request_without_parser_execution() -> io::Result<()>
    {
        let mut responses = header(RESPONSE_MAGIC, 0, 0);
        // A well-framed response for a different request must never be published.
        responses.extend_from_slice(&header(RESPONSE_MAGIC, 2, 1));
        responses.push(0);
        let (commands, receiver) = mpsc::sync_channel(1);
        commands
            .send(Request {
                ticket: Ticket(1),
                source: Arc::from("界"),
            })
            .map_err(io::Error::other)?;
        drop(commands);
        let (events, received) = mpsc::sync_channel(2);
        let mut requests = Vec::new();
        exchange(
            &mut requests,
            &mut Cursor::new(responses),
            &receiver,
            &events,
        );
        assert!(matches!(received.try_recv(), Ok(Event::Ready)));
        assert!(received.try_recv().is_err());
        let mut expected = header(REQUEST_MAGIC, 1, 3);
        expected.extend_from_slice("界".as_bytes());
        assert_eq!(requests, expected);
        Ok(())
    }
}
