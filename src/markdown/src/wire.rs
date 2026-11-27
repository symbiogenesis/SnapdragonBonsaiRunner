//! Only inert text, bounded native styles, source ranges and validated SVG bounds leave the parser.
use super::{
    DocumentLayout, ImageBlock, Layout, MAX_MARKDOWN_BYTES, MAX_RENDERED_BYTES, MAX_STYLE_SPANS,
    Segment, svg,
};
use std::ops::Range;

const MAGIC: &[u8; 4] = b"BMD1";
const MAGIC_SEGMENTED: &[u8; 4] = b"BMD2";
const SEGMENT_TEXT: u8 = 0;
const SEGMENT_IMAGE: u8 = 1;
const MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;
const MAX_SEGMENTS: usize = 2 * svg::MAX_SVG_IMAGES + 1;
const SECTION_BYTES: usize = 10;
const MAPPING_BYTES: usize = 17;
const IMAGE_SEGMENT_BYTES: usize = 17;

fn push_count(bytes: &mut Vec<u8>, value: usize) -> Option<()> {
    bytes.extend_from_slice(&u32::try_from(value).ok()?.to_le_bytes());
    Some(())
}

fn push_range(bytes: &mut Vec<u8>, range: &Range<usize>) -> Option<()> {
    push_count(bytes, range.start)?;
    push_count(bytes, range.end)
}

fn text_section_bytes(layout: &Layout) -> Option<usize> {
    12_usize
        .checked_add(layout.text.len())?
        .checked_add(layout.sections.len().checked_mul(SECTION_BYTES)?)?
        .checked_add(layout.mappings.len().checked_mul(MAPPING_BYTES)?)
}

fn encode_text_section(
    bytes: &mut Vec<u8>,
    layout: &Layout,
    previous_source_end: &mut usize,
) -> Option<()> {
    let text = &layout.text;
    let sections = &layout.sections;
    let mappings = &layout.mappings;
    push_count(bytes, text.len())?;
    bytes.extend_from_slice(text.as_bytes());
    push_count(bytes, sections.len())?;
    for section in sections {
        push_range(bytes, &section.range)?;
        bytes.push(section.style.flags);
        bytes.push(section.style.size);
    }
    push_count(bytes, mappings.len())?;
    for mapping in mappings {
        if mapping.source.start < *previous_source_end || mapping.source.start >= mapping.source.end
        {
            return None;
        }
        *previous_source_end = mapping.source.end;
        push_range(bytes, &mapping.source)?;
        push_range(bytes, &mapping.display)?;
        bytes.push(u8::from(mapping.verbatim));
    }
    Some(())
}

fn encode_image_segment(
    bytes: &mut Vec<u8>,
    image: &ImageBlock,
    previous_source_end: &mut usize,
) -> Option<()> {
    if image.source.start < *previous_source_end
        || image.source.start >= image.source.end
        || !svg::valid_intrinsic_dimensions(image.intrinsic_width, image.intrinsic_height)
    {
        return None;
    }
    *previous_source_end = image.source.end;
    bytes.push(SEGMENT_IMAGE);
    push_range(bytes, &image.source)?;
    bytes.extend_from_slice(&image.intrinsic_width.to_bits().to_le_bytes());
    bytes.extend_from_slice(&image.intrinsic_height.to_bits().to_le_bytes());
    Some(())
}

fn encode_single(layout: &Layout, max_rendered_bytes: usize) -> Option<Vec<u8>> {
    if layout.text.len() > max_rendered_bytes
        || layout.sections.len() > MAX_STYLE_SPANS
        || layout.mappings.len() > MAX_STYLE_SPANS
    {
        return None;
    }
    let length = 4_usize.checked_add(text_section_bytes(layout)?)?;
    if length > MAX_FRAME_BYTES {
        return None;
    }
    let mut bytes = Vec::with_capacity(length);
    bytes.extend_from_slice(MAGIC);
    let mut previous_source_end = 0;
    encode_text_section(&mut bytes, layout, &mut previous_source_end)?;
    Some(bytes)
}

fn encode_segmented(segments: &[Segment], max_rendered_bytes: usize) -> Option<Vec<u8>> {
    if segments.is_empty() || segments.len() > MAX_SEGMENTS {
        return None;
    }
    let mut rendered_bytes = 0_usize;
    let mut span_count = 0_usize;
    let mut length = 8_usize;
    for segment in segments {
        match segment {
            Segment::Text(layout) => {
                rendered_bytes = rendered_bytes.checked_add(layout.text.len())?;
                span_count = span_count
                    .checked_add(layout.sections.len())?
                    .checked_add(layout.mappings.len())?;
                length = length
                    .checked_add(1)?
                    .checked_add(text_section_bytes(layout)?)?;
            }
            Segment::Image(_) => {
                length = length.checked_add(IMAGE_SEGMENT_BYTES)?;
            }
        }
    }
    if rendered_bytes > max_rendered_bytes
        || span_count > MAX_STYLE_SPANS
        || length > MAX_FRAME_BYTES
    {
        return None;
    }
    let mut bytes = Vec::with_capacity(length);
    bytes.extend_from_slice(MAGIC_SEGMENTED);
    push_count(&mut bytes, segments.len())?;
    let mut previous_source_end = 0;
    for segment in segments {
        match segment {
            Segment::Text(layout) => {
                bytes.push(SEGMENT_TEXT);
                encode_text_section(&mut bytes, layout, &mut previous_source_end)?;
            }
            Segment::Image(image) => {
                encode_image_segment(&mut bytes, image, &mut previous_source_end)?;
            }
        }
    }
    Some(bytes)
}

pub(super) fn encode(source: &str, document: &DocumentLayout) -> Vec<u8> {
    if source.len() > MAX_MARKDOWN_BYTES {
        return Vec::new();
    }
    let max_rendered_bytes = MAX_RENDERED_BYTES.min(source.len().saturating_mul(4));
    let encoded = if document.segments.len() == 1
        && let Some(Segment::Text(only)) = document.segments.first()
    {
        encode_single(only, max_rendered_bytes)
    } else {
        encode_segmented(&document.segments, max_rendered_bytes)
    };
    encoded
        .or_else(|| encode_single(&super::plain(source), max_rendered_bytes))
        .unwrap_or_default()
}
