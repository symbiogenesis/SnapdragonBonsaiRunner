//! The parser can return only bounded, inert text and native style choices.
//! Every field is checked again in the trusted process before egui sees it.
use super::{
    Layout, MAX_MARKDOWN_BYTES, MAX_RENDERED_BYTES, MAX_STYLE_SPANS, Mapping, SourceMap, egui,
};
use std::{ops::Range, sync::Arc};

const MAGIC: &[u8; 4] = b"BMD1";
const MAGIC_SEGMENTED: &[u8; 4] = b"BMD2";
const MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;
const MAX_SVG_IMAGES: usize = 8;
const MAX_SEGMENTS: usize = 2 * MAX_SVG_IMAGES + 1;
const SECTION_BYTES: usize = 10;
const MAPPING_BYTES: usize = 17;
const SEGMENT_TEXT: u8 = 0;
const SEGMENT_IMAGE: u8 = 1;
const CODE: u8 = 1;
const ITALIC: u8 = 2;
const STRONG: u8 = 4;
const STRIKE: u8 = 8;
const CODE_BLOCK: u8 = 16;
const KNOWN_FLAGS: u8 = CODE | ITALIC | STRONG | STRIKE | CODE_BLOCK;

#[derive(Clone)]
pub(super) struct ImageBlock {
    pub(super) source_range: Range<usize>,
    pub(super) intrinsic_width: f32,
    pub(super) intrinsic_height: f32,
}

pub(super) enum Segment {
    Text(Layout),
    Image(ImageBlock),
}

pub(in crate::window::transcript) struct Document {
    pub(super) segments: Vec<Segment>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.position.checked_add(count)?;
        let bytes = self.bytes.get(self.position..end)?;
        self.position = end;
        Some(bytes)
    }

    fn byte(&mut self) -> Option<u8> {
        self.take(1)?.first().copied()
    }

    fn bits(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn count(&mut self) -> Option<usize> {
        usize::try_from(self.bits()?).ok()
    }

    fn range(&mut self) -> Option<Range<usize>> {
        Some(self.count()?..self.count()?)
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }
}

#[cfg(test)]
fn push_count(bytes: &mut Vec<u8>, value: usize) -> Option<()> {
    bytes.extend_from_slice(&u32::try_from(value).ok()?.to_le_bytes());
    Some(())
}

#[cfg(test)]
fn push_range(bytes: &mut Vec<u8>, range: &Range<usize>) -> Option<()> {
    push_count(bytes, range.start)?;
    push_count(bytes, range.end)
}

#[cfg(test)]
fn font_size(size: f32) -> Option<u8> {
    [16_u8, 17, 18, 19, 20, 21, 22, 24, 28]
        .into_iter()
        .find(|candidate| f32::from(*candidate).to_bits() == size.to_bits())
}

#[cfg(test)]
fn flags(format: &egui::TextFormat, visuals: &egui::Visuals) -> u8 {
    let mut flags = 0;
    if format.font_id.family == egui::FontFamily::Monospace {
        flags |= CODE;
    }
    if matches!(&format.font_id.family, egui::FontFamily::Name(name) if name.as_ref() == super::presentation::CODE_BLOCK)
    {
        flags |= CODE | CODE_BLOCK;
    }
    if format.italics {
        flags |= ITALIC;
    }
    if format.color == visuals.strong_text_color() {
        flags |= STRONG;
    }
    if format.strikethrough.width > 0.0 {
        flags |= STRIKE;
    }
    flags
}

#[cfg(test)]
fn encode_checked(layout: &Layout) -> Option<Vec<u8>> {
    let text = &layout.job.text;
    let sections = &layout.job.sections;
    let mappings = &layout.source_map.entries;
    if text.len() > MAX_RENDERED_BYTES
        || sections.len() > MAX_STYLE_SPANS
        || mappings.len() > MAX_STYLE_SPANS
    {
        return None;
    }
    let length = 16_usize
        .checked_add(text.len())?
        .checked_add(sections.len().checked_mul(SECTION_BYTES)?)?
        .checked_add(mappings.len().checked_mul(MAPPING_BYTES)?)?;
    if length > MAX_FRAME_BYTES {
        return None;
    }
    let visuals = egui::Visuals::dark();
    let mut bytes = Vec::with_capacity(length);
    bytes.extend_from_slice(MAGIC);
    push_count(&mut bytes, text.len())?;
    bytes.extend_from_slice(text.as_bytes());
    push_count(&mut bytes, sections.len())?;
    for section in sections {
        push_count(&mut bytes, section.byte_range.start.0)?;
        push_count(&mut bytes, section.byte_range.end.0)?;
        bytes.push(flags(&section.format, &visuals));
        bytes.push(font_size(section.format.font_id.size)?);
    }
    push_count(&mut bytes, mappings.len())?;
    for mapping in mappings {
        push_range(&mut bytes, &mapping.source)?;
        push_range(&mut bytes, &mapping.display)?;
        bytes.push(u8::from(mapping.verbatim));
    }
    Some(bytes)
}

/// Encoding runs only in the untrusted parser process. An invalid layout is
/// represented by a rejected empty payload rather than a partial message.
#[cfg(test)]
pub(super) fn encode(layout: &Layout) -> Vec<u8> {
    encode_checked(layout).unwrap_or_default()
}

fn text_format(flags: u8, size: u8, visuals: &egui::Visuals) -> Option<egui::TextFormat> {
    if flags & !KNOWN_FLAGS != 0
        || (flags & CODE_BLOCK != 0 && flags & CODE == 0)
        || ![16, 17, 18, 19, 20, 21, 22, 24, 28].contains(&size)
    {
        return None;
    }
    let color = if flags & STRONG != 0 {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    let font = if flags & CODE_BLOCK != 0 {
        egui::FontId::new(
            f32::from(size),
            egui::FontFamily::Name(super::presentation::CODE_BLOCK.into()),
        )
    } else if flags & CODE != 0 {
        egui::FontId::monospace(f32::from(size))
    } else {
        egui::FontId::proportional(f32::from(size))
    };
    let mut format = egui::TextFormat::simple(font, color);
    if flags & CODE != 0 {
        format.background = visuals.code_bg_color;
    }
    format.italics = flags & ITALIC != 0;
    if flags & STRIKE != 0 {
        format.strikethrough = egui::Stroke::new(1.0, color);
    }
    Some(format)
}

fn content_range(text: &str, range: &Range<usize>, previous_end: usize) -> bool {
    range.start >= previous_end
        && range.start < range.end
        && range.end <= text.len()
        && text.is_char_boundary(range.start)
        && text.is_char_boundary(range.end)
}

fn decode_text_segment(
    source: &str,
    reader: &mut Reader<'_>,
    visuals: &egui::Visuals,
    previous_source_end: &mut usize,
    rendered_bytes: &mut usize,
    span_count: &mut usize,
    text_budget: usize,
    aggregate_spans: bool,
) -> Option<Layout> {
    let text_len = reader.count()?;
    // An untrusted worker cannot inflate hundreds of tiny messages to a MiB
    // each. This also bounds aggregate parsed text by retained source bytes.
    *rendered_bytes = rendered_bytes.checked_add(text_len)?;
    if *rendered_bytes > text_budget {
        return None;
    }
    let text = std::str::from_utf8(reader.take(text_len)?).ok()?;
    if source.is_empty() && !text.is_empty() {
        return None;
    }
    let section_count = reader.count()?;
    if section_count > MAX_STYLE_SPANS
        || reader.remaining() < section_count.checked_mul(SECTION_BYTES)?.checked_add(4)?
    {
        return None;
    }
    let mut job = egui::text::LayoutJob {
        text: String::with_capacity(text_len),
        sections: Vec::with_capacity(section_count),
        ..Default::default()
    };
    let mut display_end = 0;
    for _ in 0..section_count {
        let range = reader.range()?;
        let flags = reader.byte()?;
        let size = reader.byte()?;
        if range.start != display_end || !content_range(text, &range, display_end) {
            return None;
        }
        let format = text_format(flags, size, visuals)?;
        job.append(text.get(range.clone())?, 0.0, format);
        display_end = range.end;
    }
    if display_end != text.len() {
        return None;
    }
    let mapping_count = reader.count()?;
    if mapping_count > MAX_STYLE_SPANS
        || reader.remaining() < mapping_count.checked_mul(MAPPING_BYTES)?
    {
        return None;
    }
    if aggregate_spans {
        *span_count = span_count
            .checked_add(section_count)?
            .checked_add(mapping_count)?;
        if *span_count > MAX_STYLE_SPANS {
            return None;
        }
    }
    let mut entries = Vec::with_capacity(mapping_count);
    let mut previous_display_end = 0;
    for _ in 0..mapping_count {
        let source_range = reader.range()?;
        let display_range = reader.range()?;
        let verbatim = match reader.byte()? {
            0 => false,
            1 => true,
            _ => return None,
        };
        if !content_range(source, &source_range, *previous_source_end)
            || !content_range(text, &display_range, previous_display_end)
            || (verbatim && source.get(source_range.clone())? != text.get(display_range.clone())?)
        {
            return None;
        }
        *previous_source_end = source_range.end;
        previous_display_end = display_range.end;
        entries.push(Mapping {
            source: source_range,
            display: display_range,
            verbatim,
        });
    }
    Some(Layout {
        job,
        source_map: Arc::new(SourceMap {
            entries,
            display_bytes: text_len,
        }),
    })
}

fn decode_image_segment(
    source: &str,
    reader: &mut Reader<'_>,
    previous_source_end: &mut usize,
    images: &mut usize,
) -> Option<ImageBlock> {
    if *images >= MAX_SVG_IMAGES {
        return None;
    }
    *images += 1;
    let source_range = reader.range()?;
    let intrinsic_width = f32::from_bits(reader.bits()?);
    let intrinsic_height = f32::from_bits(reader.bits()?);
    if !content_range(source, &source_range, *previous_source_end)
        || !intrinsic_width.is_finite()
        || !intrinsic_height.is_finite()
        || intrinsic_width <= 0.0
        || intrinsic_height <= 0.0
    {
        return None;
    }
    *previous_source_end = source_range.end;
    Some(ImageBlock {
        source_range,
        intrinsic_width,
        intrinsic_height,
    })
}

/// The trusted decoder accepts no native objects, coordinates, actionable
/// paths/URLs or arbitrary style parameters. All strings remain display text.
/// Source maps establish safe offsets, not truth of an untrusted parser's
/// Markdown interpretation or normalized content.
pub(super) fn decode(source: &str, bytes: &[u8]) -> Option<Document> {
    if source.len() > MAX_MARKDOWN_BYTES || bytes.len() > MAX_FRAME_BYTES {
        return None;
    }
    let mut reader = Reader { bytes, position: 0 };
    let magic = reader.take(MAGIC.len())?;
    let visuals = egui::Visuals::dark();
    let text_budget = MAX_RENDERED_BYTES.min(source.len().saturating_mul(4));
    let mut previous_source_end = 0;
    let mut rendered_bytes = 0;
    let mut span_count = 0;
    if magic == MAGIC {
        let layout = decode_text_segment(
            source,
            &mut reader,
            &visuals,
            &mut previous_source_end,
            &mut rendered_bytes,
            &mut span_count,
            text_budget,
            false,
        )?;
        if reader.remaining() != 0 {
            return None;
        }
        return Some(Document {
            segments: vec![Segment::Text(layout)],
        });
    }
    if magic != MAGIC_SEGMENTED {
        return None;
    }
    let segment_count = reader.count()?;
    if segment_count == 0 || segment_count > MAX_SEGMENTS {
        return None;
    }
    let mut images = 0;
    let mut segments = Vec::with_capacity(segment_count);
    for _ in 0..segment_count {
        match reader.byte()? {
            SEGMENT_TEXT => segments.push(Segment::Text(decode_text_segment(
                source,
                &mut reader,
                &visuals,
                &mut previous_source_end,
                &mut rendered_bytes,
                &mut span_count,
                text_budget,
                true,
            )?)),
            SEGMENT_IMAGE => segments.push(Segment::Image(decode_image_segment(
                source,
                &mut reader,
                &mut previous_source_end,
                &mut images,
            )?)),
            _ => return None,
        }
    }
    if reader.remaining() != 0 {
        return None;
    }
    Some(Document { segments })
}

#[cfg(test)]
mod tests {
    use super::*;

    type Section = (u32, u32, u8, u8);
    type Map = (u32, u32, u32, u32, u8);

    enum TestSegment<'a> {
        Text {
            text: &'a str,
            sections: &'a [Section],
            mappings: &'a [Map],
        },
        Image {
            source: Range<u32>,
            width: f32,
            height: f32,
        },
    }

    fn first_text(document: Document) -> Option<Layout> {
        match document.segments.into_iter().next()? {
            Segment::Text(layout) => Some(layout),
            Segment::Image(_) => None,
        }
    }

    fn push_text_body(bytes: &mut Vec<u8>, text: &str, sections: &[Section], mappings: &[Map]) {
        bytes.extend_from_slice(&u32::try_from(text.len()).unwrap_or_default().to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
        bytes.extend_from_slice(
            &u32::try_from(sections.len())
                .unwrap_or_default()
                .to_le_bytes(),
        );
        for (start, end, flags, size) in sections {
            bytes.extend_from_slice(&start.to_le_bytes());
            bytes.extend_from_slice(&end.to_le_bytes());
            bytes.extend_from_slice(&[*flags, *size]);
        }
        bytes.extend_from_slice(
            &u32::try_from(mappings.len())
                .unwrap_or_default()
                .to_le_bytes(),
        );
        for (source_start, source_end, display_start, display_end, verbatim) in mappings {
            for value in [source_start, source_end, display_start, display_end] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes.push(*verbatim);
        }
    }

    fn packet(text: &str, sections: &[Section], mappings: &[Map]) -> Vec<u8> {
        let mut bytes = Vec::from(*b"BMD1");
        push_text_body(&mut bytes, text, sections, mappings);
        bytes
    }

    fn bmd2_packet(segments: &[TestSegment<'_>]) -> Vec<u8> {
        let mut bytes = Vec::from(*b"BMD2");
        bytes.extend_from_slice(
            &u32::try_from(segments.len())
                .unwrap_or_default()
                .to_le_bytes(),
        );
        for segment in segments {
            match segment {
                TestSegment::Text {
                    text,
                    sections,
                    mappings,
                } => {
                    bytes.push(SEGMENT_TEXT);
                    push_text_body(&mut bytes, text, sections, mappings);
                }
                TestSegment::Image {
                    source,
                    width,
                    height,
                } => {
                    bytes.push(SEGMENT_IMAGE);
                    bytes.extend_from_slice(&source.start.to_le_bytes());
                    bytes.extend_from_slice(&source.end.to_le_bytes());
                    bytes.extend_from_slice(&width.to_bits().to_le_bytes());
                    bytes.extend_from_slice(&height.to_bits().to_le_bytes());
                }
            }
        }
        bytes
    }

    fn fixture() -> Vec<u8> {
        packet(
            "界 &",
            &[(0, 3, STRONG, 22), (3, 5, 0, 16)],
            &[(2, 5, 0, 3, 1), (7, 8, 3, 4, 1), (8, 13, 4, 5, 0)],
        )
    }

    #[test]
    fn markdown_wire_round_trip_preserves_unicode_styles_and_normalized_maps()
    -> Result<(), &'static str> {
        let source = "**界** &amp;";
        let first = decode(source, &fixture())
            .and_then(first_text)
            .ok_or("Valid independent fixture was rejected.")?;
        assert_eq!(first.job.text, "界 &");
        assert_eq!(first.source_map.range(2..5), 0..3);
        assert_eq!(first.source_map.range(8..13), 4..5);
        let second = decode(source, &encode(&first))
            .and_then(first_text)
            .ok_or("Encoded layout was rejected.")?;
        assert_eq!(second.job.text, first.job.text);
        assert_eq!(second.job.sections, first.job.sections);
        assert_eq!(second.source_map.entries.len(), 3);
        assert!(decode("", &packet("", &[], &[])).is_some());
        Ok(())
    }

    #[test]
    fn markdown_wire_rejects_truncation_lengths_counts_and_surplus() {
        let source = "**界** &amp;";
        let valid = fixture();
        for end in 0..valid.len() {
            assert!(
                decode(source, &valid[..end]).is_none(),
                "Accepted truncated frame at {end}."
            );
        }
        for offset in [4, 8 + "界 &".len()] {
            let mut forged = valid.clone();
            forged[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            assert!(decode(source, &forged).is_none());
        }
        let mut forged = valid.clone();
        forged.push(0);
        assert!(decode(source, &forged).is_none());
        let mut forged = valid.clone();
        forged[0] = b'X';
        assert!(decode(source, &forged).is_none());
        let mut forged = valid;
        let maps_at = 12 + "界 &".len() + 2 * SECTION_BYTES;
        forged[maps_at..maps_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(source, &forged).is_none());
        assert!(decode(&"x".repeat(MAX_MARKDOWN_BYTES + 1), &fixture()).is_none());
        assert!(decode(source, &vec![0; MAX_FRAME_BYTES + 1]).is_none());
        assert!(decode("x", &packet("yyyyy", &[(0, 5, 0, 16)], &[])).is_none());
    }

    #[test]
    fn markdown_wire_rejects_invalid_styles_gaps_and_unicode_boundaries() {
        let source = "界";
        for section in [
            (0, 3, 16, 16),
            (0, 3, 32, 16),
            (0, 3, 0, 255),
            (1, 3, 0, 16),
            (0, 1, 0, 16),
            (0, 4, 0, 16),
            (3, 0, 0, 16),
            (0, 0, 0, 16),
        ] {
            assert!(decode(source, &packet(source, &[section], &[(0, 3, 0, 3, 1)])).is_none());
        }
        assert!(decode("ab", &packet("ab", &[(0, 1, 0, 16), (0, 2, 0, 16)], &[])).is_none());
        assert!(decode(source, &packet(source, &[], &[])).is_none());
        let mut invalid_utf8 = packet(source, &[(0, 3, 0, 16)], &[]);
        invalid_utf8[8] = 255;
        assert!(decode(source, &invalid_utf8).is_none());
    }

    #[test]
    fn markdown_wire_admits_only_fixed_code_block_and_heading_styles() -> Result<(), &'static str> {
        for size in [16, 17, 19, 21, 24, 28] {
            let bytes = packet("heading", &[(0, 7, STRONG, size)], &[(0, 7, 0, 7, 1)]);
            assert!(decode("heading", &bytes).is_some());
        }
        let bytes = packet("code", &[(0, 4, CODE | CODE_BLOCK, 16)], &[(0, 4, 0, 4, 1)]);
        let layout = decode("code", &bytes)
            .and_then(first_text)
            .ok_or("Valid block style rejected")?;
        assert_eq!(
            layout.job.sections[0].format.font_id.family,
            egui::FontFamily::Name(super::super::presentation::CODE_BLOCK.into())
        );
        assert_eq!(encode(&layout), bytes);
        for flags in [CODE_BLOCK, CODE_BLOCK | STRONG, 32, 255] {
            assert!(decode("code", &packet("code", &[(0, 4, flags, 16)], &[])).is_none());
        }
        for size in [0, 15, 23, 29, 255] {
            assert!(
                decode(
                    "code",
                    &packet("code", &[(0, 4, CODE | CODE_BLOCK, size)], &[])
                )
                .is_none()
            );
        }
        Ok(())
    }

    #[test]
    fn markdown_wire_rejects_source_spoofing_and_unordered_maps() {
        for mapping in [
            (1, 3, 0, 3, 0),
            (0, 4, 0, 3, 0),
            (0, 3, 1, 3, 0),
            (0, 3, 0, 4, 0),
            (0, 3, 0, 0, 0),
            (0, 3, 0, 3, 2),
        ] {
            assert!(decode("界", &packet("界", &[(0, 3, 0, 16)], &[mapping])).is_none());
        }
        assert!(decode("界", &packet("海", &[(0, 3, 0, 16)], &[(0, 3, 0, 3, 1)])).is_none());
        for maps in [
            [(1, 2, 0, 1, 0), (0, 1, 1, 2, 0)],
            [(0, 1, 1, 2, 0), (1, 2, 0, 1, 0)],
            [(0, 2, 0, 1, 0), (1, 2, 1, 2, 0)],
            [(0, 1, 0, 2, 0), (1, 2, 1, 2, 0)],
        ] {
            assert!(decode("ab", &packet("ab", &[(0, 2, 0, 16)], &maps)).is_none());
        }
        assert!(decode("", &packet("injected", &[(0, 8, 0, 16)], &[])).is_none());
    }

    #[test]
    fn markdown_bmd2_decodes_segmented_images_and_rejects_invalid_segments()
    -> Result<(), &'static str> {
        let source = "Hi\n\n```svg\n<svg/>\n```\n\nBye";
        let image_start = u32::try_from("Hi\n\n".len()).map_err(|_| "offset overflow")?;
        let image_end = image_start
            + u32::try_from("```svg\n<svg/>\n```".len()).map_err(|_| "offset overflow")?;
        let bye_start = image_end + 2;
        let bye_end = u32::try_from(source.len()).map_err(|_| "offset overflow")?;
        let valid = bmd2_packet(&[
            TestSegment::Text {
                text: "Hi",
                sections: &[(0, 2, 0, 16)],
                mappings: &[(0, 2, 0, 2, 1)],
            },
            TestSegment::Image {
                source: image_start..image_end,
                width: 320.0,
                height: 240.0,
            },
            TestSegment::Text {
                text: "Bye",
                sections: &[(0, 3, 0, 16)],
                mappings: &[(bye_start, bye_end, 0, 3, 1)],
            },
        ]);
        let doc = decode(source, &valid).ok_or("Valid BMD2 document rejected")?;
        assert_eq!(doc.segments.len(), 3);
        assert!(matches!(
            &doc.segments[1],
            Segment::Image(image)
                if &source[image.source_range.clone()] == "```svg\n<svg/>\n```"
                    && (image.intrinsic_width - 320.0).abs() < f32::EPSILON
                    && (image.intrinsic_height - 240.0).abs() < f32::EPSILON
        ));

        // Non-monotonic text after image must fail.
        assert!(
            decode(
                source,
                &bmd2_packet(&[
                    TestSegment::Image {
                        source: image_start..image_end,
                        width: 100.0,
                        height: 100.0,
                    },
                    TestSegment::Text {
                        text: "Hi",
                        sections: &[(0, 2, 0, 16)],
                        mappings: &[(0, 2, 0, 2, 1)],
                    },
                ])
            )
            .is_none()
        );

        // Non-finite, zero, negative, and > MAX_SVG_IMAGES segments must fail.
        for bad_dims in [
            (0.0, 100.0),
            (100.0, -1.0),
            (f32::NAN, 100.0),
            (100.0, f32::INFINITY),
        ] {
            assert!(
                decode(
                    source,
                    &bmd2_packet(&[TestSegment::Image {
                        source: image_start..image_end,
                        width: bad_dims.0,
                        height: bad_dims.1,
                    }])
                )
                .is_none()
            );
        }
        assert!(decode(source, &bmd2_packet(&[])).is_none());
        let too_many_segments: Vec<_> = (0..18)
            .map(|_| TestSegment::Text {
                text: "",
                sections: &[],
                mappings: &[],
            })
            .collect();
        assert!(decode(source, &bmd2_packet(&too_many_segments)).is_none());
        Ok(())
    }
}
