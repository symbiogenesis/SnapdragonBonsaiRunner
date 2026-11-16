//! Bounded inert Markdown packets and offline SVG rasterization without UI or OS APIs.
#![forbid(unsafe_code)]

use std::{ops::Range, sync::OnceLock};

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
pub(crate) const CODE_BLOCK_FONT_NAME: &str = "Bonsai Code Block";

const QUOTE_PREFIX: &str = "│ ";
const QUOTE_PREFIX_BLOCK: &str = "│ │ │ │ │ │ │ │ ";
const MAX_STATIC_REPEATS: usize = 8;

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

pub(crate) struct RenderTheme {
    pub(crate) normal: egui::TextFormat,
    pub(crate) strong_text_color: egui::Color32,
    pub(crate) code_bg_color: egui::Color32,
}

pub(crate) fn theme() -> &'static RenderTheme {
    static THEME: OnceLock<RenderTheme> = OnceLock::new();
    THEME.get_or_init(|| {
        let visuals = egui::Visuals::dark();
        RenderTheme {
            normal: egui::TextFormat::simple(
                egui::FontId::proportional(16.0),
                visuals.text_color(),
            ),
            strong_text_color: visuals.strong_text_color(),
            code_bg_color: visuals.code_bg_color,
        }
    })
}

struct BuilderOut {
    job: egui::text::LayoutJob,
    mappings: Vec<Mapping>,
    segments: Vec<Segment>,
    quote_depth: usize,
    spans: usize,
    rendered_bytes: usize,
    images_emitted: usize,
    limited: bool,
}

impl BuilderOut {
    fn append(
        &mut self,
        source_text: &str,
        normal: &egui::TextFormat,
        scratch: &mut String,
        text: &str,
        source: Option<Range<usize>>,
        format: &egui::TextFormat,
    ) {
        if text.is_empty() || self.limited {
            return;
        }
        if self.quote_depth != 0
            && !text.starts_with('\n')
            && (self.job.text.is_empty() || self.job.text.ends_with('\n'))
        {
            // Container markers are presentation text, never part of a source
            // match. Prefix each authored line rather than only the first one.
            let prefix =
                repeated_slice(QUOTE_PREFIX, QUOTE_PREFIX_BLOCK, self.quote_depth, scratch);
            self.append_raw(source_text, prefix, None, normal);
        }
        self.append_raw(source_text, text, source, format);
    }

    fn append_raw(
        &mut self,
        source_text: &str,
        text: &str,
        source: Option<Range<usize>>,
        format: &egui::TextFormat,
    ) {
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
        if let Some(last) = self.job.sections.last_mut().filter(|section| {
            section.byte_range.end.0 == display_start && section.format == *format
        }) {
            self.job.text.push_str(text);
            last.byte_range.end.0 = self.job.text.len();
        } else {
            self.job.append(text, 0.0, format.clone());
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
}

fn repeated_slice<'a>(
    unit: &str,
    static_block: &'a str,
    count: usize,
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
    let max_repeats = MAX_RENDERED_BYTES / unit.len().max(1);
    let capped = count.min(max_repeats.saturating_add(1));
    for _ in 0..capped {
        scratch.push_str(unit);
    }
    scratch.as_str()
}

struct Builder<'a> {
    source: &'a str,
    theme: &'a RenderTheme,
    scratch: String,
    out: BuilderOut,
}

impl Builder<'_> {
    fn append(&mut self, text: &str, source: Option<Range<usize>>, format: &egui::TextFormat) {
        self.out.append(
            self.source,
            &self.theme.normal,
            &mut self.scratch,
            text,
            source,
            format,
        );
    }

    fn push_image(&mut self, image: ImageBlock) {
        self.out.push_image(image);
    }

    fn generated(&mut self, text: &str, format: &egui::TextFormat) {
        self.append(text, None, format);
    }

    fn generated_repeated(
        &mut self,
        unit: &str,
        static_block: &str,
        count: usize,
        format: &egui::TextFormat,
    ) {
        if count == 0 || self.out.limited {
            return;
        }
        if count <= MAX_STATIC_REPEATS
            && let Some(slice) = static_block.get(..count * unit.len())
        {
            self.append(slice, None, format);
            return;
        }
        self.scratch.clear();
        let max_repeats = MAX_RENDERED_BYTES / unit.len().max(1);
        let capped = count.min(max_repeats.saturating_add(1));
        for _ in 0..capped {
            self.scratch.push_str(unit);
        }
        // Only one small heap allocation if depth exceeds MAX_STATIC_REPEATS (> 8) inside an active quote line.
        if self.out.quote_depth != 0
            && (self.out.job.text.is_empty() || self.out.job.text.ends_with('\n'))
        {
            let mut prefix_buf = String::new();
            let prefix = repeated_slice(
                QUOTE_PREFIX,
                QUOTE_PREFIX_BLOCK,
                self.out.quote_depth,
                &mut prefix_buf,
            );
            self.out
                .append_raw(self.source, prefix, None, &self.theme.normal);
        }
        self.out
            .append_raw(self.source, &self.scratch, None, format);
    }

    fn newlines(&mut self, count: usize) {
        if self.out.job.text.is_empty() {
            return;
        }
        let existing = self
            .out
            .job
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
            self.out.append(
                self.source,
                &self.theme.normal,
                &mut self.scratch,
                slice,
                None,
                &self.theme.normal,
            );
        } else {
            for _ in 0..needed {
                self.out.append(
                    self.source,
                    &self.theme.normal,
                    &mut self.scratch,
                    "\n",
                    None,
                    &self.theme.normal,
                );
            }
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

fn plain(body: &str, normal: &egui::TextFormat) -> Layout {
    let mut job = egui::text::LayoutJob::default();
    job.append(body, 0.0, normal.clone());
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
    let mut doc = parse_document(body);
    if doc.segments.len() == 1
        && let Some(Segment::Text(layout)) = doc.segments.pop()
    {
        return layout;
    }
    plain(body, &theme().normal)
}

fn parse_document(body: &str) -> DocumentLayout {
    let theme = theme();
    let mut builder = Builder {
        source: body,
        theme,
        scratch: String::new(),
        out: BuilderOut {
            job: egui::text::LayoutJob {
                text: String::with_capacity(body.len()),
                ..Default::default()
            },
            mappings: Vec::new(),
            segments: Vec::new(),
            quote_depth: 0,
            spans: 0,
            rendered_bytes: 0,
            images_emitted: 0,
            limited: false,
        },
    };
    let parsed = markdown_syntax::parse(body);
    builder.blocks(&parsed.document.children, &theme.normal, 0, false);
    if body.ends_with('\n') && (builder.out.segments.is_empty() || !builder.out.job.text.is_empty())
    {
        builder.newlines(1);
    }
    if builder.out.limited {
        return DocumentLayout {
            segments: vec![Segment::Text(plain(body, &theme.normal))],
        };
    }
    if builder.out.segments.is_empty()
        || !builder.out.job.text.is_empty()
        || !builder.out.mappings.is_empty()
    {
        builder.out.segments.push(Segment::Text(Layout {
            job: builder.out.job,
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
    wire::encode(&parse_document(source))
}
