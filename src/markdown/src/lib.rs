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

struct Mapping {
    source: Range<usize>,
    display: Range<usize>,
    verbatim: bool,
}

struct Layout {
    job: egui::text::LayoutJob,
    mappings: Vec<Mapping>,
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

struct Builder<'a> {
    source: &'a str,
    job: egui::text::LayoutJob,
    mappings: Vec<Mapping>,
    segments: Vec<Segment>,
    visuals: &'a egui::Visuals,
    normal: egui::TextFormat,
    quote_depth: usize,
    spans: usize,
    rendered_bytes: usize,
    images_emitted: usize,
    limited: bool,
}

impl Builder<'_> {
    fn append(&mut self, text: &str, source: Option<Range<usize>>, format: &egui::TextFormat) {
        if text.is_empty() || self.limited {
            return;
        }
        if self.quote_depth != 0
            && !text.starts_with('\n')
            && (self.job.text.is_empty() || self.job.text.ends_with('\n'))
        {
            // Container markers are presentation text, never part of a source
            // match. Prefix each authored line rather than only the first one.
            let prefix = "│ ".repeat(self.quote_depth);
            self.append_raw(&prefix, None, &self.normal.clone());
        }
        self.append_raw(text, source, format);
    }

    fn append_raw(&mut self, text: &str, source: Option<Range<usize>>, format: &egui::TextFormat) {
        if text.is_empty() || self.limited {
            return;
        }
        self.spans += 1;
        let next_bytes = self.rendered_bytes.saturating_add(text.len());
        if self.spans > MAX_STYLE_SPANS || next_bytes > MAX_RENDERED_BYTES {
            self.limited = true;
            return;
        }
        self.rendered_bytes = next_bytes;
        let display_start = self.job.text.len();
        self.job.append(text, 0.0, format.clone());
        if let Some(source) = source.filter(|range| !range.is_empty()) {
            self.mappings.push(Mapping {
                verbatim: self.source.get(source.clone()) == Some(text),
                source,
                display: display_start..display_start + text.len(),
            });
        }
    }

    fn flush_text(&mut self) {
        if self.job.text.is_empty() && self.mappings.is_empty() {
            return;
        }
        let job = std::mem::take(&mut self.job);
        let mappings = std::mem::take(&mut self.mappings);
        self.segments.push(Segment::Text(Layout { job, mappings }));
    }

    fn push_image(&mut self, image: ImageBlock) {
        self.flush_text();
        self.images_emitted += 1;
        self.segments.push(Segment::Image(image));
    }

    fn generated(&mut self, text: &str, format: &egui::TextFormat) {
        self.append(text, None, format);
    }

    fn newlines(&mut self, count: usize) {
        if self.job.text.is_empty() {
            return;
        }
        let existing = self
            .job
            .text
            .bytes()
            .rev()
            .take(count)
            .take_while(|byte| *byte == b'\n')
            .count();
        let normal = self.normal.clone();
        for _ in existing..count {
            self.generated("\n", &normal);
        }
    }

    fn literal(&mut self, span: Option<markdown_syntax::Span>, format: &egui::TextFormat) {
        if let Some(span) = span
            && let Some(text) = self.source.get(span.start..span.end)
        {
            self.append(text, Some(span.start..span.end), format);
        }
    }

    fn value(
        &mut self,
        value: &str,
        span: Option<markdown_syntax::Span>,
        format: &egui::TextFormat,
    ) {
        let range = span.and_then(|span| {
            let source = self.source.get(span.start..span.end)?;
            Some(source.find(value).map_or(span.start..span.end, |offset| {
                span.start + offset..span.start + offset + value.len()
            }))
        });
        self.append(value, range, format);
    }
}

fn plain(body: &str, normal: egui::TextFormat) -> Layout {
    let mut job = egui::text::LayoutJob::default();
    job.append(body, 0.0, normal);
    Layout {
        job,
        mappings: vec![Mapping {
            source: 0..body.len(),
            display: 0..body.len(),
            verbatim: true,
        }],
    }
}

#[cfg(test)]
fn parse_layout(body: &str) -> Layout {
    let visuals = egui::Visuals::dark();
    let normal = egui::TextFormat::simple(egui::FontId::proportional(16.0), visuals.text_color());
    let mut doc = parse_document(body);
    if doc.segments.len() == 1
        && let Some(Segment::Text(layout)) = doc.segments.pop()
    {
        return layout;
    }
    plain(body, normal)
}

fn parse_document(body: &str) -> DocumentLayout {
    let visuals = egui::Visuals::dark();
    let normal = egui::TextFormat::simple(egui::FontId::proportional(16.0), visuals.text_color());
    let mut builder = Builder {
        source: body,
        job: egui::text::LayoutJob {
            text: String::with_capacity(body.len()),
            ..Default::default()
        },
        mappings: Vec::new(),
        segments: Vec::new(),
        visuals: &visuals,
        normal: normal.clone(),
        quote_depth: 0,
        spans: 0,
        rendered_bytes: 0,
        images_emitted: 0,
        limited: false,
    };
    let parsed = markdown_syntax::parse(body);
    builder.blocks(&parsed.document.children, &normal, 0, false);
    if body.ends_with('\n') && (builder.segments.is_empty() || !builder.job.text.is_empty()) {
        builder.newlines(1);
    }
    if builder.limited {
        return DocumentLayout {
            segments: vec![Segment::Text(plain(body, normal))],
        };
    }
    if builder.segments.is_empty() || !builder.job.text.is_empty() || !builder.mappings.is_empty() {
        builder.segments.push(Segment::Text(Layout {
            job: builder.job,
            mappings: builder.mappings,
        }));
    }
    DocumentLayout {
        segments: builder.segments,
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
    wire::encode(&parse_document(source))
}
