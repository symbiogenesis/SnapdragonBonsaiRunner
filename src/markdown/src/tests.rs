use super::{MAX_MARKDOWN_BYTES, parse, parse_layout, serve};
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

#[test]
fn markdown_typography_preserves_six_heading_levels_and_inline_code_size() -> io::Result<()> {
    let source = "# One\n\n## Two\n\n### Three\n\n#### Four\n\n##### Five\n\n###### Six\n\n# `heading_code`\n\nBody `body_code`";
    let layout = parse_layout(source);
    for (needle, size) in [
        ("One", 28.0_f32),
        ("Two", 24.0),
        ("Three", 21.0),
        ("Four", 19.0),
        ("Five", 17.0),
        ("Six", 16.0),
        ("heading_code", 28.0),
        ("body_code", 16.0),
    ] {
        let offset = egui::text::ByteIndex(
            layout
                .job
                .text
                .find(needle)
                .ok_or_else(|| io::Error::other("Missing heading text"))?,
        );
        let format = layout.job.format_at_byte(offset);
        assert_eq!(format.font_id.size.to_bits(), size.to_bits(), "{needle}");
    }
    // All admitted heading sizes must survive the worker's closed wire format.
    assert!(!parse(source).is_empty());
    Ok(())
}

#[test]
fn markdown_typography_prefixes_quote_lines_and_indents_nested_lists() -> io::Result<()> {
    let source =
        "> First\n> second\n>\n> Third\n>\n>> Nested\n\n- Parent\n  - Child\n    - Grandchild";
    let layout = parse_layout(source);
    assert_eq!(
        layout.job.text,
        "│ First\n│ second\n\n│ Third\n\n│ │ Nested\n\n• Parent\n    • Child\n        • Grandchild"
    );
    for mapping in &layout.mappings {
        assert!(mapping.verbatim);
        assert_eq!(
            source.get(mapping.source.clone()),
            layout.job.text.get(mapping.display.clone())
        );
    }
    for pair in layout.mappings.windows(2) {
        assert!(pair[0].source.end <= pair[1].source.start);
        assert!(pair[0].display.end <= pair[1].display.start);
    }
    let first = egui::text::ByteIndex(
        layout
            .job
            .text
            .find("First")
            .ok_or_else(|| io::Error::other("Missing quote text"))?,
    );
    assert!(!layout.job.format_at_byte(first).italics);
    assert!(!parse(source).is_empty());
    Ok(())
}

#[test]
fn markdown_typography_marks_code_blocks_separately_from_inline_code() -> io::Result<()> {
    let source = "```rust\nlet first = 1;\nlet second = 2;\n```\n\nUse `inline`.";
    let layout = parse_layout(source);
    let packet = parse(source);
    let count_offset = 8 + layout.job.text.len();
    let count = usize::try_from(u32::from_le_bytes(
        packet
            .get(count_offset..count_offset + 4)
            .ok_or_else(|| io::Error::other("Missing section count"))?
            .try_into()
            .map_err(io::Error::other)?,
    ))
    .map_err(io::Error::other)?;
    let sections = packet
        .get(count_offset + 4..count_offset + 4 + count * 10)
        .ok_or_else(|| io::Error::other("Missing style sections"))?;
    let (chunks, _) = sections.as_chunks::<10>();
    let flags: Vec<_> = chunks.iter().map(|section| section[8]).collect();
    assert!(flags.contains(&17), "block code must carry CODE|CODE_BLOCK");
    assert!(flags.contains(&1), "inline code must carry only CODE");
    assert!(flags.iter().all(|flags| flags & 16 == 0 || flags & 1 != 0));
    for mapping in &layout.mappings {
        assert_eq!(
            source.get(mapping.source.clone()),
            layout.job.text.get(mapping.display.clone())
        );
    }
    Ok(())
}
