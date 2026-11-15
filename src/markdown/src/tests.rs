use super::{
    MAX_MARKDOWN_BYTES, MAX_RASTER_DIMENSION, MAX_RASTER_PIXELS, MAX_SVG_IMAGES, Segment, parse,
    parse_document, parse_layout, rasterize, serve, svg,
};
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

fn raster_frame(ticket: u64, max_w: u32, max_h: u32, svg: &[u8]) -> io::Result<Vec<u8>> {
    let mut frame = b"BSMG0001".to_vec();
    frame.extend_from_slice(&ticket.to_le_bytes());
    let len = svg
        .len()
        .checked_add(8)
        .ok_or_else(|| io::Error::other("len overflow"))?;
    frame.extend_from_slice(&u32::try_from(len).map_err(io::Error::other)?.to_le_bytes());
    frame.extend_from_slice(&max_w.to_le_bytes());
    frame.extend_from_slice(&max_h.to_le_bytes());
    frame.extend_from_slice(svg);
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
    let short_raster = frame(2, b"tiny")?;
    let mut bad_raster = short_raster.clone();
    bad_raster[..8].copy_from_slice(b"BSMG0001");
    for request in [
        bad_magic,
        oversized,
        truncated,
        bad_raster,
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

const SAMPLE_SVG: &str = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 200 100\"><rect width=\"200\" height=\"100\" fill=\"#204080\"/><circle cx=\"100\" cy=\"50\" r=\"35\" fill=\"#ffaa22\"/><text x=\"100\" y=\"55\" text-anchor=\"middle\" font-size=\"18\" fill=\"#ffffff\">Hi</text></svg>";

#[test]
fn segmented_markdown_extracts_top_level_svg_and_preserves_monotonic_source_maps() -> io::Result<()>
{
    let source = format!("Intro **text**:\n\n```svg\n{SAMPLE_SVG}\n```\n\nOutro note.\n");
    let doc = parse_document(&source);
    assert_eq!(doc.segments.len(), 3);
    let Segment::Text(before) = &doc.segments[0] else {
        return Err(io::Error::other("expected leading text segment"));
    };
    assert_eq!(before.job.text, "Intro text:");
    let Segment::Image(img) = &doc.segments[1] else {
        return Err(io::Error::other("expected image segment"));
    };
    assert_eq!(source[img.source.clone()].trim(), SAMPLE_SVG);
    assert!((img.intrinsic_width - 200.0).abs() < 0.01);
    assert!((img.intrinsic_height - 100.0).abs() < 0.01);
    let Segment::Text(after) = &doc.segments[2] else {
        return Err(io::Error::other("expected trailing text segment"));
    };
    assert_eq!(after.job.text, "Outro note.\n");
    let before_last = before
        .mappings
        .last()
        .ok_or_else(|| io::Error::other("missing leading source map"))?;
    let after_first = after
        .mappings
        .first()
        .ok_or_else(|| io::Error::other("missing trailing source map"))?;
    assert!(before_last.source.end <= img.source.start);
    assert!(img.source.end <= after_first.source.start);

    let packet = parse(&source);
    assert!(packet.starts_with(b"BMD2"));
    Ok(())
}

#[test]
fn svg_fallback_keeps_nested_unclosed_invalid_and_excess_blocks_as_code() {
    // Nested inside blockquote stays code (`BMD1`)
    let quoted = format!("> ```svg\n> {SAMPLE_SVG}\n> ```");
    assert!(parse(&quoted).starts_with(b"BMD1"));

    // Unclosed fence stays code (`BMD1`)
    let unclosed = format!("```svg\n{SAMPLE_SVG}");
    assert!(parse(&unclosed).starts_with(b"BMD1"));

    // Malformed XML stays code (`BMD1`)
    assert!(parse("```svg\n<svg><broken\n```").starts_with(b"BMD1"));

    // Tagged xml stays code (`BMD1`)
    let xml = format!("```xml\n{SAMPLE_SVG}\n```");
    assert!(parse(&xml).starts_with(b"BMD1"));

    // At most MAX_SVG_IMAGES are extracted; a 9th becomes a code block
    let many = format!("```svg\n{SAMPLE_SVG}\n```\n\n").repeat(MAX_SVG_IMAGES + 1);
    let doc = parse_document(&many);
    let image_count = doc
        .segments
        .iter()
        .filter(|seg| matches!(seg, Segment::Image(_)))
        .count();
    assert_eq!(image_count, MAX_SVG_IMAGES);
    assert!(matches!(doc.segments.last(), Some(Segment::Text(_))));
}

#[test]
fn offline_svg_rasterizer_produces_valid_premultiplied_rgba_and_blocks_hrefs() -> io::Result<()> {
    let packet = rasterize(SAMPLE_SVG, 400, 400);
    assert!(packet.starts_with(svg::RASTER_MAGIC));
    assert_eq!(packet[4], 1);
    let width = u32::from_le_bytes(packet[5..9].try_into().map_err(io::Error::other)?);
    let height = u32::from_le_bytes(packet[9..13].try_into().map_err(io::Error::other)?);
    assert_eq!((width, height), (400, 200));
    let rgba = &packet[svg::RASTER_HEADER_BYTES..];
    assert_eq!(
        rgba.len(),
        usize::try_from(width * height * 4).map_err(io::Error::other)?
    );
    let mut any_non_zero = false;
    for px in rgba.chunks_exact(4) {
        assert!(px[0] <= px[3] && px[1] <= px[3] && px[2] <= px[3]);
        if px != [0, 0, 0, 0] {
            any_non_zero = true;
        }
    }
    assert!(any_non_zero);

    // Pixel ceiling check (MAX_RASTER_DIMENSION x MAX_RASTER_DIMENSION exceeds MAX_RASTER_PIXELS)
    let square = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><rect width=\"100\" height=\"100\" fill=\"red\"/></svg>";
    let capped = rasterize(square, MAX_RASTER_DIMENSION, MAX_RASTER_DIMENSION);
    assert_eq!(capped[4], 1);
    let cw = u32::from_le_bytes(capped[5..9].try_into().map_err(io::Error::other)?);
    let ch = u32::from_le_bytes(capped[9..13].try_into().map_err(io::Error::other)?);
    assert!(usize::try_from(cw * ch).map_err(io::Error::other)? <= MAX_RASTER_PIXELS);

    // `<image>` elements (including `data:` and external URLs) are rejected outright.
    let with_data_img = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\"><image href=\"data:image/png;base64,iVBORw0KGgo=\" width=\"64\" height=\"64\"/><rect width=\"10\" height=\"10\" fill=\"#00ff00\"/></svg>";
    let data_packet = rasterize(with_data_img, 64, 64);
    assert_eq!(data_packet.len(), svg::RASTER_HEADER_BYTES);
    assert_eq!(data_packet[4], 0);
    assert!(svg::inspect(with_data_img).is_none());

    // Invalid SVG or zero dimension returns ok = 0 packet
    for bad in [rasterize("<not svg", 64, 64), rasterize(SAMPLE_SVG, 0, 64)] {
        assert_eq!(bad.len(), svg::RASTER_HEADER_BYTES);
        assert_eq!(bad[4], 0);
    }
    Ok(())
}

#[test]
fn svg_allowlists_are_strictly_sorted_and_exclude_active_constructs() {
    for pair in svg::ALLOWED_ELEMENTS.windows(2) {
        assert!(
            pair[0] < pair[1],
            "ALLOWED_ELEMENTS unsorted at {} >= {}",
            pair[0],
            pair[1]
        );
    }
    for pair in svg::ALLOWED_ATTRIBUTES.windows(2) {
        assert!(
            pair[0] < pair[1],
            "ALLOWED_ATTRIBUTES unsorted at {} >= {}",
            pair[0],
            pair[1]
        );
    }
    for forbidden_tag in [
        "a",
        "animate",
        "animateColor",
        "animateMotion",
        "animateTransform",
        "audio",
        "canvas",
        "cursor",
        "discard",
        "embed",
        "feImage",
        "foreignObject",
        "handler",
        "iframe",
        "image",
        "listener",
        "metadata",
        "mpath",
        "object",
        "script",
        "set",
        "tref",
        "video",
        "view",
    ] {
        assert!(
            svg::ALLOWED_ELEMENTS.binary_search(&forbidden_tag).is_err(),
            "forbidden tag {forbidden_tag} must not be in ALLOWED_ELEMENTS"
        );
    }
    for attr in svg::ALLOWED_ATTRIBUTES {
        assert!(
            !attr
                .as_bytes()
                .get(..2)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"on")),
            "event attribute {attr} must not be in ALLOWED_ATTRIBUTES"
        );
    }
}

#[test]
fn untrusted_svg_policy_rejects_scripts_events_dtds_urls_css_escapes_and_use_bombs() {
    let mut use_bomb = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><defs><rect id=\"l0\" width=\"2\" height=\"2\" fill=\"red\"/>",
    );
    for level in 1..=5 {
        let prev = level - 1;
        use_bomb.push_str(&format!("<g id=\"l{level}\">"));
        for _ in 0..10 {
            use_bomb.push_str(&format!("<use href=\"#l{prev}\"/>"));
        }
        use_bomb.push_str("</g>");
    }
    use_bomb.push_str("</defs><use href=\"#l5\"/></svg>");

    let mut too_deep =
        String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\">");
    for _ in 0..36 {
        too_deep.push_str("<g>");
    }
    too_deep.push_str("<rect width=\"5\" height=\"5\"/>");
    for _ in 0..36 {
        too_deep.push_str("</g>");
    }
    too_deep.push_str("</svg>");

    let oversized_kernel = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><defs><filter id=\"f\"><feConvolveMatrix order=\"9 9\" kernelMatrix=\"{}\"/></filter></defs><rect width=\"32\" height=\"32\" filter=\"url(#f)\"/></svg>",
        "1 ".repeat(81)
    );

    let hostile_cases: &[(&str, &str)] = &[
        (
            "script tag",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><script>alert(1)</script><rect width=\"32\" height=\"32\"/></svg>",
        ),
        (
            "event attribute",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\" onload=\"alert(1)\"><rect width=\"32\" height=\"32\"/></svg>",
        ),
        (
            "uppercase event attribute",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><rect width=\"32\" height=\"32\" ONCLICK=\"evil()\"/></svg>",
        ),
        (
            "foreignObject",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><foreignObject width=\"32\" height=\"32\">body</foreignObject></svg>",
        ),
        (
            "foreign namespace",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xhtml=\"http://www.w3.org/1999/xhtml\" viewBox=\"0 0 32 32\"><rect width=\"32\" height=\"32\"/></svg>",
        ),
        (
            "xml:base rewrite",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" xml:base=\"https://evil.test/\" viewBox=\"0 0 32 32\"><rect width=\"32\" height=\"32\"/></svg>",
        ),
        (
            "xlink:actuate",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" viewBox=\"0 0 32 32\"><defs><rect id=\"r\" width=\"10\" height=\"10\"/></defs><use xlink:href=\"#r\" xlink:actuate=\"onLoad\"/></svg>",
        ),
        (
            "anchor hyperlink",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><a href=\"javascript:alert(1)\"><rect width=\"32\" height=\"32\"/></a></svg>",
        ),
        (
            "SMIL animate",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><rect width=\"32\" height=\"32\"><animate attributeName=\"x\" from=\"0\" to=\"10\" dur=\"1s\"/></rect></svg>",
        ),
        (
            "SMIL set",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><rect width=\"32\" height=\"32\"><set attributeName=\"fill\" to=\"red\"/></rect></svg>",
        ),
        (
            "DTD billion laughs",
            "<!DOCTYPE svg [<!ENTITY lol \"lol\"><!ENTITY lol2 \"&lol;&lol;\">]><svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><text x=\"0\" y=\"16\">&lol2;</text></svg>",
        ),
        (
            "external DOCTYPE",
            "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\"><svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><rect width=\"32\" height=\"32\"/></svg>",
        ),
        (
            "processing instruction",
            "<?xml-stylesheet type=\"text/xsl\" href=\"https://evil.test/x.xsl\"?><svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><rect width=\"32\" height=\"32\"/></svg>",
        ),
        (
            "external image",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><image href=\"https://evil.test/p.png\" width=\"32\" height=\"32\"/></svg>",
        ),
        (
            "feImage",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><defs><filter id=\"f\"><feImage href=\"#r\"/></filter></defs><rect id=\"r\" width=\"32\" height=\"32\" filter=\"url(#f)\"/></svg>",
        ),
        (
            "external use href",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><use href=\"https://evil.test/icons.svg#a\"/></svg>",
        ),
        (
            "scheme-relative use href",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><use href=\"//evil.test/icons.svg#a\"/></svg>",
        ),
        (
            "CSS @import",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><style>@import url(\"https://evil.test/a.css\");</style><rect width=\"32\" height=\"32\"/></svg>",
        ),
        (
            "CSS @font-face",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><style>@font-face { font-family: x; src: url(https://evil.test/x.ttf); }</style><rect width=\"32\" height=\"32\"/></svg>",
        ),
        (
            "CSS backslash escape",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><style>\\40 import \"https://evil.test/a.css\";</style><rect width=\"32\" height=\"32\"/></svg>",
        ),
        (
            "external paint url",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><rect width=\"32\" height=\"32\" fill=\"url(https://evil.test/p.svg#g)\"/></svg>",
        ),
        (
            "inline style external url",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><rect width=\"32\" height=\"32\" style=\"fill: url(/*c*/ //evil.test/p.svg#g)\"/></svg>",
        ),
        (
            "cyclic use",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><defs><g id=\"a\"><use href=\"#b\"/></g><g id=\"b\"><use href=\"#a\"/></g></defs><use href=\"#a\"/></svg>",
        ),
        (
            "cyclic gradient",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 32 32\"><defs><linearGradient id=\"g1\" href=\"#g2\"/><linearGradient id=\"g2\" href=\"#g1\"/></defs><rect width=\"32\" height=\"32\" fill=\"url(#g1)\"/></svg>",
        ),
        ("use DAG amplification", &use_bomb),
        ("excessive nesting depth", &too_deep),
        ("oversized convolve kernel", &oversized_kernel),
    ];

    for (label, svg_source) in hostile_cases {
        assert!(
            svg::inspect(svg_source).is_none(),
            "{label} must fail svg::inspect"
        );
        let markdown = format!("```svg\n{svg_source}\n```");
        assert!(
            parse(&markdown).starts_with(b"BMD1"),
            "{label} must remain an inert BMD1 code block"
        );
        let raster = rasterize(svg_source, 64, 64);
        assert_eq!(raster.len(), svg::RASTER_HEADER_BYTES, "{label}");
        assert_eq!(raster[4], 0, "{label} must return ok = 0 raster header");
    }
}

#[test]
fn rich_static_svg_diagrams_with_defs_gradients_styles_use_and_filters_render() -> io::Result<()> {
    let diagram = r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" role="img" aria-label="Pipeline diagram" viewBox="0 0 240 120">
  <title>Static Pipeline</title>
  <desc>Bounded offline vector diagram</desc>
  <defs>
    <linearGradient id="base-grad">
      <stop offset="0%" stop-color="#1d3557"/>
      <stop offset="100%" stop-color="#457b9d"/>
    </linearGradient>
    <linearGradient id="derived-grad" href="#base-grad" x1="0" y1="0" x2="1" y2="1"/>
    <clipPath id="card-clip">
      <rect x="8" y="8" width="224" height="104" rx="12"/>
    </clipPath>
    <filter id="soft-shadow">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#000000" flood-opacity="0.35"/>
    </filter>
    <g id="node-pill">
      <rect width="72" height="36" rx="8" class="pill"/>
    </g>
  </defs>
  <style type="text/css">
    /* Safe internal styles */
    .bg { fill: url(#derived-grad); }
    .pill { fill: #a8dadc; stroke: #f1faee; stroke-width: 2; }
  </style>
  <g clip-path="url(#card-clip)">
    <rect width="240" height="120" class="bg"/>
    <use href="#node-pill" x="24" y="42" filter="url(#soft-shadow)"/>
    <use xlink:href="#node-pill" x="144" y="42"/>
    <line x1="96" y1="60" x2="144" y2="60" stroke="#f1faee" stroke-width="3"/>
    <text x="60" y="65" text-anchor="middle" font-size="13" fill="#1d3557">MXC</text>
    <text x="180" y="65" text-anchor="middle" font-size="13" fill="#1d3557">UI</text>
  </g>
</svg>"##;

    let (w, h) = svg::inspect(diagram).ok_or_else(|| io::Error::other("expected valid SVG"))?;
    assert!((w - 240.0).abs() < 0.01 && (h - 120.0).abs() < 0.01);
    let packet = rasterize(diagram, 240, 120);
    assert_eq!(packet[4], 1);
    assert!(parse(&format!("```svg\n{diagram}\n```")).starts_with(b"BMD2"));
    Ok(())
}

#[test]
fn persistent_worker_serves_interleaved_markdown_and_svg_raster_requests() -> io::Result<()> {
    let mut requests = frame(1, "**界**".as_bytes())?;
    requests.extend(raster_frame(2, 160, 160, SAMPLE_SVG.as_bytes())?);
    let mut output = Vec::new();
    assert_eq!(
        serve(&mut Cursor::new(requests), &mut output)
            .err()
            .map(|error| error.kind()),
        Some(io::ErrorKind::UnexpectedEof)
    );
    let expected_raster = rasterize(SAMPLE_SVG, 160, 160);
    let mut cursor = Cursor::new(output);
    let expected_first = parse("**界**");
    let expected_payloads: Vec<(u64, &[u8])> =
        vec![(0, &[]), (1, &expected_first), (2, &expected_raster)];
    for (ticket, payload) in expected_payloads {
        let mut header = [0; 20];
        cursor.read_exact(&mut header)?;
        assert_eq!(&header[..8], b"BSMR0001");
        assert_eq!(&header[8..16], &ticket.to_le_bytes());
        let len = u32::from_le_bytes(header[16..].try_into().map_err(io::Error::other)?);
        assert_eq!(
            usize::try_from(len).map_err(io::Error::other)?,
            payload.len()
        );
        let mut actual = vec![0; payload.len()];
        cursor.read_exact(&mut actual)?;
        assert_eq!(actual, payload);
    }
    Ok(())
}
