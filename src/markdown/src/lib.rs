//! Bounded inert Markdown packets and offline SVG rasterization without UI or OS APIs.
#![forbid(unsafe_code)]

use std::ops::Range;

mod markdown;
mod protocol;
mod svg;
mod wire;

#[cfg(test)]
mod tests;

pub use protocol::serve;
pub use svg::{MAX_RASTER_DIMENSION, MAX_RASTER_PIXELS, MAX_SVG_BYTES, MAX_SVG_IMAGES, rasterize};

const MAX_MARKDOWN_BYTES: usize = 256 * 1024;
const MAX_STYLE_SPANS: usize = 16_384;
const MAX_RENDERED_BYTES: usize = 4 * MAX_MARKDOWN_BYTES;

pub(crate) const STYLE_CODE: u8 = 1;
pub(crate) const STYLE_ITALIC: u8 = 2;
pub(crate) const STYLE_STRONG: u8 = 4;
pub(crate) const STYLE_STRIKE: u8 = 8;
pub(crate) const STYLE_CODE_BLOCK: u8 = 16;

const QUOTE_PREFIX: &str = "│ ";
const QUOTE_PREFIX_BLOCK: &str = "│ │ │ │ │ │ │ │ ";
const MAX_STATIC_REPEATS: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Style {
    flags: u8,
    size: u8,
}

impl Style {
    const NORMAL: Self = Self { flags: 0, size: 16 };

    const fn with(self, flag: u8) -> Self {
        Self {
            flags: self.flags | flag,
            size: self.size,
        }
    }

    const fn code(self, block: bool) -> Self {
        let extra = if block {
            STYLE_CODE | STYLE_CODE_BLOCK
        } else {
            STYLE_CODE
        };
        Self {
            flags: (self.flags & !STYLE_ITALIC) | extra,
            size: self.size,
        }
    }
}

struct Section {
    range: Range<usize>,
    style: Style,
}

struct Mapping {
    source: Range<usize>,
    display: Range<usize>,
    verbatim: bool,
}

struct Layout {
    text: String,
    sections: Vec<Section>,
    mappings: Vec<Mapping>,
}

#[cfg(test)]
impl Layout {
    fn style_at_byte(&self, byte_offset: usize) -> Option<Style> {
        self.sections
            .iter()
            .find(|section| section.range.contains(&byte_offset))
            .map(|section| section.style)
    }
}

struct ImageBlock {
    source: Range<usize>,
    intrinsic_width: f32,
    intrinsic_height: f32,
}

enum Segment {
    Text(Layout),
    Image(ImageBlock),
}

struct DocumentLayout {
    segments: Vec<Segment>,
}

struct BuilderOut {
    text: String,
    sections: Vec<Section>,
    mappings: Vec<Mapping>,
    segments: Vec<Segment>,
    quote_depth: usize,
    flushed_spans: usize,
    rendered_bytes: usize,
    max_rendered_bytes: usize,
    images_emitted: usize,
    limited: bool,
}

impl BuilderOut {
    fn append(
        &mut self,
        source_text: &str,
        text: &str,
        source: Option<Range<usize>>,
        style: Style,
    ) {
        if text.is_empty() || self.limited {
            return;
        }
        if self.quote_depth != 0
            && !text.starts_with('\n')
            && (self.text.is_empty() || self.text.ends_with('\n'))
        {
            // Container markers are presentation text, never part of a source
            // match. Prefix each authored line rather than only the first one.
            let mut overflow = String::new();
            let prefix = repeated_slice(
                QUOTE_PREFIX,
                QUOTE_PREFIX_BLOCK,
                self.quote_depth,
                self.max_rendered_bytes,
                &mut overflow,
            );
            self.append_raw(source_text, prefix, None, Style::NORMAL);
        }
        self.append_raw(source_text, text, source, style);
    }

    fn append_raw(
        &mut self,
        source_text: &str,
        text: &str,
        source: Option<Range<usize>>,
        style: Style,
    ) {
        if text.is_empty() || self.limited {
            return;
        }
        let next_bytes = self.rendered_bytes.saturating_add(text.len());
        let display_start = self.text.len();
        let merges_section = self
            .sections
            .last()
            .is_some_and(|section| section.range.end == display_start && section.style == style);
        let added_sections = usize::from(!merges_section);
        let added_mappings = usize::from(source.as_ref().is_some_and(|range| !range.is_empty()));
        let next_spans = self
            .flushed_spans
            .saturating_add(self.sections.len())
            .saturating_add(self.mappings.len())
            .saturating_add(added_sections)
            .saturating_add(added_mappings);
        if next_spans > MAX_STYLE_SPANS || next_bytes > self.max_rendered_bytes {
            self.limited = true;
            return;
        }
        self.rendered_bytes = next_bytes;
        self.text.push_str(text);
        if let Some(last) = self
            .sections
            .last_mut()
            .filter(|section| section.range.end == display_start && section.style == style)
        {
            last.range.end = self.text.len();
        } else {
            self.sections.push(Section {
                range: display_start..self.text.len(),
                style,
            });
        }
        if let Some(source) = source.filter(|range| !range.is_empty()) {
            self.mappings.push(Mapping {
                verbatim: source_text.get(source.clone()) == Some(text),
                source,
                display: display_start..display_start + text.len(),
            });
        }
    }

    fn flush_text(&mut self) {
        if self.text.is_empty() && self.mappings.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.text);
        let sections = std::mem::take(&mut self.sections);
        let mappings = std::mem::take(&mut self.mappings);
        self.flushed_spans = self
            .flushed_spans
            .saturating_add(sections.len())
            .saturating_add(mappings.len());
        self.segments.push(Segment::Text(Layout {
            text,
            sections,
            mappings,
        }));
    }

    fn push_image(&mut self, image: ImageBlock) {
        self.flush_text();
        self.images_emitted += 1;
        self.segments.push(Segment::Image(image));
    }
}

fn repeated_slice<'a>(
    unit: &str,
    static_block: &'a str,
    count: usize,
    max_rendered_bytes: usize,
    scratch: &'a mut String,
) -> &'a str {
    if count == 0 {
        return "";
    }
    if count <= MAX_STATIC_REPEATS
        && let Some(slice) = static_block.get(..count * unit.len())
    {
        return slice;
    }
    scratch.clear();
    let max_repeats = max_rendered_bytes / unit.len().max(1);
    let capped = count.min(max_repeats.saturating_add(1));
    for _ in 0..capped {
        scratch.push_str(unit);
    }
    scratch.as_str()
}

struct Builder<'a> {
    source: &'a str,
    scratch: String,
    out: BuilderOut,
}

impl Builder<'_> {
    fn append(&mut self, text: &str, source: Option<Range<usize>>, style: Style) {
        self.out.append(self.source, text, source, style);
    }

    fn push_image(&mut self, image: ImageBlock) {
        self.out.push_image(image);
    }

    fn generated(&mut self, text: &str, style: Style) {
        self.append(text, None, style);
    }

    fn generated_repeated(&mut self, unit: &str, static_block: &str, count: usize, style: Style) {
        let slice = repeated_slice(
            unit,
            static_block,
            count,
            self.out.max_rendered_bytes,
            &mut self.scratch,
        );
        self.out.append(self.source, slice, None, style);
    }

    fn newlines(&mut self, count: usize) {
        if self.out.text.is_empty() {
            return;
        }
        let existing = self
            .out
            .text
            .bytes()
            .rev()
            .take(count)
            .take_while(|byte| *byte == b'\n')
            .count();
        let needed = count.saturating_sub(existing);
        if needed == 0 {
            return;
        }
        const TWO_NEWLINES: &str = "\n\n";
        if let Some(slice) = TWO_NEWLINES.get(..needed) {
            self.out.append(self.source, slice, None, Style::NORMAL);
        } else {
            for _ in 0..needed {
                self.out.append(self.source, "\n", None, Style::NORMAL);
            }
        }
    }

    fn literal(&mut self, span: Option<markdown_syntax::Span>, style: Style) {
        if let Some(span) = span
            && let Some(text) = self.source.get(span.start..span.end)
        {
            self.append(text, Some(span.start..span.end), style);
        }
    }

    fn value(&mut self, value: &str, span: Option<markdown_syntax::Span>, style: Style) {
        let range = span.and_then(|span| {
            let source = self.source.get(span.start..span.end)?;
            if source.len() == value.len() && source == value {
                return Some(span.start..span.end);
            }
            if source.len() < value.len() {
                return Some(span.start..span.end);
            }
            Some(source.find(value).map_or(span.start..span.end, |offset| {
                span.start + offset..span.start + offset + value.len()
            }))
        });
        self.append(value, range, style);
    }
}

fn plain(body: &str) -> Layout {
    if body.is_empty() {
        return Layout {
            text: String::new(),
            sections: Vec::new(),
            mappings: Vec::new(),
        };
    }
    Layout {
        text: body.to_owned(),
        sections: vec![Section {
            range: 0..body.len(),
            style: Style::NORMAL,
        }],
        mappings: vec![Mapping {
            source: 0..body.len(),
            display: 0..body.len(),
            verbatim: true,
        }],
    }
}

#[cfg(test)]
fn parse_layout(body: &str) -> Layout {
    let mut doc = parse_document(body);
    if doc.segments.len() == 1
        && let Some(Segment::Text(layout)) = doc.segments.pop()
    {
        return layout;
    }
    plain(body)
}

fn parse_document(body: &str) -> DocumentLayout {
    let estimated_spans = if body.is_empty() {
        0
    } else {
        (body.len() / 16).clamp(4, MAX_STYLE_SPANS)
    };
    let max_rendered_bytes = MAX_RENDERED_BYTES.min(body.len().saturating_mul(4));
    let mut builder = Builder {
        source: body,
        scratch: String::new(),
        out: BuilderOut {
            text: String::with_capacity(body.len()),
            sections: Vec::with_capacity(estimated_spans),
            mappings: Vec::with_capacity(estimated_spans),
            segments: Vec::new(),
            quote_depth: 0,
            flushed_spans: 0,
            rendered_bytes: 0,
            max_rendered_bytes,
            images_emitted: 0,
            limited: false,
        },
    };
    let parsed = markdown_syntax::parse(body);
    builder.blocks(&parsed.document.children, Style::NORMAL, 0, false);
    if body.ends_with('\n') && (builder.out.segments.is_empty() || !builder.out.text.is_empty()) {
        builder.newlines(1);
    }
    if builder.out.limited {
        return DocumentLayout {
            segments: vec![Segment::Text(plain(body))],
        };
    }
    if builder.out.segments.is_empty()
        || !builder.out.text.is_empty()
        || !builder.out.mappings.is_empty()
    {
        builder.out.segments.push(Segment::Text(Layout {
            text: builder.out.text,
            sections: builder.out.sections,
            mappings: builder.out.mappings,
        }));
    }
    DocumentLayout {
        segments: builder.out.segments,
    }
}

/// Parses bounded Markdown into an inert `BMD1` or `BMD2` render-description packet.
///
/// Production callers run this only in the owned offline parser process. The
/// trusted parent must independently validate every returned field. Inputs over
/// 256 KiB or an unencodable result return an empty packet for rejection; styling
/// limits retain the complete literal source instead of a partial rendering.
#[must_use]
pub fn parse(source: &str) -> Vec<u8> {
    if source.len() > MAX_MARKDOWN_BYTES {
        return Vec::new();
    }
    wire::encode(source, &parse_document(source))
}
