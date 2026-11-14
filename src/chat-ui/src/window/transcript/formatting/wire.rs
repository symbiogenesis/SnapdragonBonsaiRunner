//! The parser can return only bounded, inert text and native style choices.
//! Every field is checked again in the trusted process before egui sees it.
use super::{
    Layout, MAX_MARKDOWN_BYTES, MAX_RENDERED_BYTES, MAX_STYLE_SPANS, Mapping, SourceMap, egui,
};
use std::{ops::Range, sync::Arc};

const MAGIC: &[u8; 4] = b"BMD1";
const MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;
const SECTION_BYTES: usize = 10;
const MAPPING_BYTES: usize = 17;
const CODE: u8 = 1;
const ITALIC: u8 = 2;
const STRONG: u8 = 4;
const STRIKE: u8 = 8;
const KNOWN_FLAGS: u8 = CODE | ITALIC | STRONG | STRIKE;

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

    fn count(&mut self) -> Option<usize> {
        usize::try_from(u32::from_le_bytes(self.take(4)?.try_into().ok()?)).ok()
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
    [16_u8, 18, 20, 22]
        .into_iter()
        .find(|candidate| f32::from(*candidate).to_bits() == size.to_bits())
}

#[cfg(test)]
fn flags(format: &egui::TextFormat, visuals: &egui::Visuals) -> u8 {
    let mut flags = 0;
    if format.font_id.family == egui::FontFamily::Monospace {
        flags |= CODE;
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
    if flags & !KNOWN_FLAGS != 0 || ![16, 18, 20, 22].contains(&size) {
        return None;
    }
    let color = if flags & STRONG != 0 {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    let font = if flags & CODE != 0 {
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

/// The trusted decoder accepts no native objects, coordinates, actionable
/// paths/URLs or arbitrary style parameters. All strings remain display text.
/// Source maps establish safe offsets, not truth of an untrusted parser's
/// Markdown interpretation or normalized content.
pub(super) fn decode(source: &str, bytes: &[u8]) -> Option<Layout> {
    if source.len() > MAX_MARKDOWN_BYTES || bytes.len() > MAX_FRAME_BYTES {
        return None;
    }
    let mut reader = Reader { bytes, position: 0 };
    if reader.take(MAGIC.len())? != MAGIC {
        return None;
    }
    let text_len = reader.count()?;
    // An untrusted worker cannot inflate hundreds of tiny messages to a MiB
    // each. This also bounds aggregate parsed text by retained source bytes.
    if text_len > MAX_RENDERED_BYTES || text_len > source.len().saturating_mul(4) {
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
    let visuals = egui::Visuals::dark();
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
        let format = text_format(flags, size, &visuals)?;
        job.append(text.get(range.clone())?, 0.0, format);
        display_end = range.end;
    }
    if display_end != text.len() {
        return None;
    }
    let mapping_count = reader.count()?;
    if mapping_count > MAX_STYLE_SPANS
        || reader.remaining() != mapping_count.checked_mul(MAPPING_BYTES)?
    {
        return None;
    }
    let mut entries = Vec::with_capacity(mapping_count);
    let mut previous_source_end = 0;
    let mut previous_display_end = 0;
    for _ in 0..mapping_count {
        let source_range = reader.range()?;
        let display_range = reader.range()?;
        let verbatim = match reader.byte()? {
            0 => false,
            1 => true,
            _ => return None,
        };
        if !content_range(source, &source_range, previous_source_end)
            || !content_range(text, &display_range, previous_display_end)
            || (verbatim && source.get(source_range.clone())? != text.get(display_range.clone())?)
        {
            return None;
        }
        previous_source_end = source_range.end;
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

#[cfg(test)]
mod tests {
    use super::*;

    type Section = (u32, u32, u8, u8);
    type Map = (u32, u32, u32, u32, u8);

    fn packet(text: &str, sections: &[Section], mappings: &[Map]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"BMD1");
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
        let first = decode(source, &fixture()).ok_or("Valid independent fixture was rejected.")?;
        assert_eq!(first.job.text, "界 &");
        assert_eq!(first.source_map.range(2..5), 0..3);
        assert_eq!(first.source_map.range(8..13), 4..5);
        let second = decode(source, &encode(&first)).ok_or("Encoded layout was rejected.")?;
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
}
