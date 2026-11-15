//! Only inert text, bounded native styles and source ranges leave the parser.
use super::{Layout, MAX_RENDERED_BYTES, MAX_STYLE_SPANS};
use std::ops::Range;

const MAGIC: &[u8; 4] = b"BMD1";
const MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;
const SECTION_BYTES: usize = 10;
const MAPPING_BYTES: usize = 17;
const CODE: u8 = 1;
const ITALIC: u8 = 2;
const STRONG: u8 = 4;
const STRIKE: u8 = 8;
const CODE_BLOCK: u8 = 16;

fn push_count(bytes: &mut Vec<u8>, value: usize) -> Option<()> {
    bytes.extend_from_slice(&u32::try_from(value).ok()?.to_le_bytes());
    Some(())
}

fn push_range(bytes: &mut Vec<u8>, range: &Range<usize>) -> Option<()> {
    push_count(bytes, range.start)?;
    push_count(bytes, range.end)
}

fn font_size(size: f32) -> Option<u8> {
    [16_u8, 17, 19, 21, 24, 28]
        .into_iter()
        .find(|candidate| f32::from(*candidate).to_bits() == size.to_bits())
}

fn flags(format: &egui::TextFormat, visuals: &egui::Visuals) -> u8 {
    let mut flags = 0;
    if format.font_id.family == egui::FontFamily::Monospace {
        flags |= CODE;
    } else if format.font_id.family == egui::FontFamily::Name("Bonsai Code Block".into()) {
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

fn encode_checked(layout: &Layout) -> Option<Vec<u8>> {
    let text = &layout.job.text;
    let sections = &layout.job.sections;
    let mappings = &layout.mappings;
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

pub(super) fn encode(layout: &Layout) -> Vec<u8> {
    encode_checked(layout).unwrap_or_default()
}
