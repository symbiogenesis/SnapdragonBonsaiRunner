use super::{MAX_MARKDOWN_BYTES, parse, serve};
use std::io::{self, Cursor, Read as _};

fn frame(ticket: u64, bytes: &[u8]) -> io::Result<Vec<u8>> {
    let mut frame = b"BSMD0001".to_vec();
    frame.extend_from_slice(&ticket.to_le_bytes());
    frame.extend_from_slice(
        &u32::try_from(bytes.len())
            .map_err(io::Error::other)?
            .to_le_bytes(),
    );
    frame.extend_from_slice(bytes);
    Ok(frame)
}

#[test]
fn persistent_worker_preserves_unicode_packet_and_request_identity() -> io::Result<()> {
    let mut requests = frame(7, "**界**".as_bytes())?;
    requests.extend(frame(19, "**界**".as_bytes())?);
    let mut output = Vec::new();
    assert_eq!(
        serve(&mut Cursor::new(requests), &mut output)
            .err()
            .map(|error| error.kind()),
        Some(io::ErrorKind::UnexpectedEof)
    );
    // Independently assemble the one strong span and its exact UTF-8 source map.
    let mut expected = b"BMD1".to_vec();
    expected.extend_from_slice(&3_u32.to_le_bytes());
    expected.extend_from_slice("界".as_bytes());
    for value in [1_u32, 0, 3] {
        expected.extend_from_slice(&value.to_le_bytes());
    }
    expected.extend_from_slice(&[4, 16]);
    for value in [1_u32, 2, 5, 0, 3] {
        expected.extend_from_slice(&value.to_le_bytes());
    }
    expected.push(1);
    assert_eq!(parse("**界**"), expected);
    let mut output = Cursor::new(output);
    for ticket in [0_u64, 7, 19] {
        let mut header = [0; 20];
        output.read_exact(&mut header)?;
        assert_eq!(&header[..8], b"BSMR0001");
        assert_eq!(&header[8..16], &ticket.to_le_bytes());
        let bytes = if ticket == 0 { &[][..] } else { &expected };
        assert_eq!(
            &header[16..],
            &u32::try_from(bytes.len())
                .map_err(io::Error::other)?
                .to_le_bytes()
        );
        let mut actual = vec![0; bytes.len()];
        output.read_exact(&mut actual)?;
        assert_eq!(actual, bytes);
    }
    assert_eq!(
        output.position(),
        u64::try_from(output.get_ref().len()).map_err(io::Error::other)?
    );
    Ok(())
}

#[test]
fn worker_rejects_invalid_frames_before_emitting_any_rendered_response() -> io::Result<()> {
    let mut bad_magic = frame(1, b"text")?;
    bad_magic[0] = b'X';
    let mut oversized = frame(1, &[])?;
    oversized[16..].copy_from_slice(
        &u32::try_from(MAX_MARKDOWN_BYTES + 1)
            .map_err(io::Error::other)?
            .to_le_bytes(),
    );
    let mut truncated = frame(1, b"text")?;
    truncated.pop();
    for request in [
        bad_magic,
        oversized,
        truncated,
        frame(0, b"text")?,
        frame(1, &[0xff])?,
    ] {
        let mut output = Vec::new();
        assert!(serve(&mut Cursor::new(request), &mut output).is_err());
        assert_eq!(output.len(), 20);
        assert_eq!(&output[..8], b"BSMR0001");
        assert_eq!(&output[8..], &[0; 12]);
    }
    assert!(parse(&"x".repeat(MAX_MARKDOWN_BYTES + 1)).is_empty());
    Ok(())
}
