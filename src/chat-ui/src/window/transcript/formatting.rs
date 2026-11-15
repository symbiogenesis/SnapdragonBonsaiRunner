//! Bounded native Markdown presentation; original message bytes stay in history.
#![forbid(unsafe_code)]
use super::{egui, search};
use std::{ops::Range, sync::Arc};

mod pending;
mod presentation;
pub(super) mod wire;
pub(super) use pending::Pending;
#[cfg(test)]
mod tests;

const MAX_MARKDOWN_BYTES: usize = 256 * 1024;
const MAX_STYLE_SPANS: usize = 16_384;
const MAX_RENDERED_BYTES: usize = 4 * MAX_MARKDOWN_BYTES;

struct Mapping {
    source: Range<usize>,
    display: Range<usize>,
    verbatim: bool,
}

/// Ordered content runs, avoiding a per-byte map for long responses.
#[derive(Default)]
pub(super) struct SourceMap {
    entries: Vec<Mapping>,
    display_bytes: usize,
}

impl SourceMap {
    // Callers supply UTF-8 boundaries from the source Find scanner. Direct
    // content runs preserve those boundaries; normalized runs use an endpoint.
    pub(super) fn offset(&self, source_byte: usize) -> usize {
        self.entries
            .get(
                self.entries
                    .partition_point(|entry| entry.source.end <= source_byte),
            )
            .map_or(self.display_bytes, |entry| {
                entry.display.start
                    + if entry.verbatim {
                        source_byte.saturating_sub(entry.source.start)
                    } else {
                        0
                    }
            })
    }

    pub(super) fn range(&self, source: Range<usize>) -> Range<usize> {
        let first = self
            .entries
            .partition_point(|entry| entry.source.end <= source.start);
        let last = self
            .entries
            .partition_point(|entry| entry.source.start < source.end);
        if first >= last {
            let offset = self.offset(source.start);
            return offset..offset;
        }
        let start = self.offset(source.start);
        let entry = &self.entries[last - 1];
        let end = if entry.verbatim {
            entry.display.start + source.end.min(entry.source.end) - entry.source.start
        } else {
            entry.display.end
        };
        start..end
    }

    fn exact_match(&self, source: &Range<usize>) -> bool {
        let first = self
            .entries
            .partition_point(|entry| entry.source.end <= source.start);
        let last = self
            .entries
            .partition_point(|entry| entry.source.start < source.end);
        first < last
            && self.entries[first..last].iter().all(|entry| {
                entry.verbatim
                    || (source.start <= entry.source.start && source.end >= entry.source.end)
            })
    }

    fn covers_exact(&self, source: &Range<usize>) -> bool {
        let (Some(first), Some(last)) = (self.entries.first(), self.entries.last()) else {
            return false;
        };
        source.start >= first.source.start
            && source.end <= last.source.end
            && self.exact_match(source)
    }

    pub(super) fn first_source_start(&self) -> Option<usize> {
        self.entries.first().map(|entry| entry.source.start)
    }
}

pub(super) struct Layout {
    pub(super) job: egui::text::LayoutJob,
    pub(super) source_map: Arc<SourceMap>,
}

pub(super) enum BodySegment {
    Text(Layout),
    Image {
        svg: Arc<str>,
        intrinsic_size: egui::Vec2,
    },
}

fn plain(
    body: &str,
    width: f32,
    format: egui::TextFormat,
    visuals: &egui::Visuals,
    matches: &[search::Match],
) -> Layout {
    let mut job = egui::text::LayoutJob::default();
    // Plain fallback must retain the complete bounded transcript, even above
    // the smaller parser/rendering allowance.
    let mut cursor = 0;
    for found in matches {
        if cursor < found.range.start {
            job.append(&body[cursor..found.range.start], 0.0, format.clone());
        }
        let mut highlighted = format.clone();
        highlighted.background = visuals.selection.bg_fill;
        highlighted.color = visuals.selection.stroke.color;
        job.append(&body[found.range.clone()], 0.0, highlighted);
        cursor = found.range.end;
    }
    job.append(&body[cursor..], 0.0, format);
    job.wrap.max_width = width;
    let source_map = SourceMap {
        entries: vec![Mapping {
            source: 0..body.len(),
            display: 0..body.len(),
            verbatim: true,
        }],
        display_bytes: body.len(),
    };
    Layout {
        job,
        source_map: Arc::new(source_map),
    }
}

#[cfg(test)]
fn parse_layout(body: &str) -> Layout {
    let fallback = || {
        let visuals = egui::Visuals::dark();
        plain(
            body,
            600.0,
            egui::TextFormat::simple(egui::FontId::proportional(16.0), visuals.text_color()),
            &visuals,
            &[],
        )
    };
    let Some(doc) = wire::decode(body, &parse_wire(body)) else {
        return fallback();
    };
    match doc.segments.into_iter().next() {
        Some(wire::Segment::Text(layout)) => layout,
        _ => fallback(),
    }
}

#[cfg(test)]
pub(super) fn parse_wire(body: &str) -> Vec<u8> {
    bonsai_markdown::parse(body)
}

fn append_tail_slice(
    tail: &str,
    base_offset: usize,
    normal: egui::TextFormat,
    text_size: f32,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    job: &mut egui::text::LayoutJob,
) {
    let mut cursor = 0;
    for found in matches
        .iter()
        .filter(|found| found.range.start >= base_offset)
    {
        let start = (found.range.start - base_offset).min(tail.len());
        let end = (found.range.end - base_offset).min(tail.len());
        if cursor < start {
            presentation::append(job, &tail[cursor..start], normal.clone(), text_size);
        }
        let mut highlighted = normal.clone();
        highlighted.background = visuals.selection.bg_fill;
        highlighted.color = visuals.selection.stroke.color;
        presentation::append(job, &tail[start..end], highlighted, text_size);
        cursor = end;
    }
    if cursor < tail.len() {
        presentation::append(job, &tail[cursor..], normal, text_size);
    }
}

fn compute_tail_start(body: &str, prefix_len: usize, parsed: Option<&Layout>) -> usize {
    if prefix_len >= body.len() {
        return prefix_len;
    }
    let Some(parsed) = parsed else {
        return prefix_len;
    };
    let mapped_end = parsed
        .source_map
        .entries
        .last()
        .map_or(0, |entry| entry.source.end);
    let trailing_ws = body[mapped_end.min(prefix_len)..prefix_len]
        .bytes()
        .rev()
        .take_while(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
        .count();
    let emitted_ws = parsed
        .job
        .text
        .bytes()
        .rev()
        .take(trailing_ws)
        .take_while(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
        .count();
    prefix_len - (trailing_ws - emitted_ws)
}

fn render_parsed_text_segment(
    body_prefix: &str,
    width: f32,
    text_size: f32,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    parsed: &Layout,
    tail_start: usize,
    append_tail: bool,
) -> Layout {
    let normal = presentation::body_format(text_size, visuals);
    let highlights: Vec<_> = matches
        .iter()
        .filter(|found| found.range.end <= tail_start)
        .map(|found| parsed.source_map.range(found.range.clone()))
        .collect();
    let template = egui::Visuals::dark();
    let tail_len = if append_tail && tail_start < body_prefix.len() {
        body_prefix.len() - tail_start
    } else {
        0
    };
    let mut job = egui::text::LayoutJob {
        text: String::with_capacity(parsed.job.text.len() + tail_len),
        ..Default::default()
    };
    job.wrap.max_width = width;
    let mut next_match = 0;
    for section in &parsed.job.sections {
        let format = presentation::text_format(&section.format, text_size, visuals, &template);
        let mut cursor = section.byte_range.start.0;
        let end = section.byte_range.end.0;
        while cursor < end {
            while highlights
                .get(next_match)
                .is_some_and(|found| found.end <= cursor)
            {
                next_match += 1;
            }
            let found = highlights.get(next_match).filter(|found| found.start < end);
            let (until, highlighted) = match found {
                Some(found) if found.start <= cursor => (found.end.min(end), true),
                Some(found) => (found.start.min(end), false),
                None => (end, false),
            };
            let mut effective = format.clone();
            if highlighted {
                effective.background = visuals.selection.bg_fill;
                effective.color = visuals.selection.stroke.color;
            }
            presentation::append(
                &mut job,
                &parsed.job.text[cursor..until],
                effective,
                text_size,
            );
            cursor = until;
        }
    }
    let source_map = if append_tail && tail_start < body_prefix.len() {
        let tail_display_start = job.text.len();
        append_tail_slice(
            &body_prefix[tail_start..],
            tail_start,
            normal,
            text_size,
            visuals,
            matches,
            &mut job,
        );
        let mut entries: Vec<Mapping> = parsed
            .source_map
            .entries
            .iter()
            .map(|entry| Mapping {
                source: entry.source.clone(),
                display: entry.display.clone(),
                verbatim: entry.verbatim,
            })
            .collect();
        entries.push(Mapping {
            source: tail_start..body_prefix.len(),
            display: tail_display_start..job.text.len(),
            verbatim: true,
        });
        Arc::new(SourceMap {
            entries,
            display_bytes: job.text.len(),
        })
    } else {
        Arc::clone(&parsed.source_map)
    };
    Layout { job, source_map }
}

#[cfg(test)]
pub(super) fn layout_job(
    body: &str,
    structured: bool,
    width: f32,
    text_size: f32,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    parsed: Option<(&Layout, usize)>,
) -> Layout {
    let normal = presentation::body_format(text_size, visuals);
    let Some((parsed, prefix_len)) = parsed.filter(|_| structured) else {
        return plain(body, width, normal, visuals, matches);
    };
    if prefix_len > body.len() || !body.is_char_boundary(prefix_len) {
        return plain(body, width, normal, visuals, matches);
    }
    let tail_start = compute_tail_start(body, prefix_len, Some(parsed));
    if matches.iter().any(|found| {
        if found.range.end <= tail_start {
            !parsed.source_map.exact_match(&found.range)
        } else {
            found.range.start < tail_start
        }
    }) {
        return plain(body, width, normal, visuals, matches);
    }
    render_parsed_text_segment(
        body, width, text_size, visuals, matches, parsed, tail_start, true,
    )
}

pub(super) fn layout_document(
    body: &str,
    structured: bool,
    width: f32,
    text_size: f32,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    parsed: Option<(&wire::Document, usize)>,
    streaming_visible_prefix: Option<usize>,
) -> (Vec<BodySegment>, bool) {
    let normal = presentation::body_format(text_size, visuals);
    if !structured {
        return (
            vec![BodySegment::Text(plain(
                body, width, normal, visuals, matches,
            ))],
            false,
        );
    }
    if let Some(prefix_end) = streaming_visible_prefix
        && matches.iter().any(|found| found.range.end > prefix_end)
    {
        // A Find match hits inside the incomplete streaming SVG block: reveal raw source.
        return (
            vec![BodySegment::Text(plain(
                body, width, normal, visuals, matches,
            ))],
            false,
        );
    }
    let effective_end = streaming_visible_prefix.unwrap_or(body.len());
    let generating = streaming_visible_prefix.is_some();
    let effective_body = &body[..effective_end];

    let Some((doc, prefix_len)) = parsed else {
        if effective_body.is_empty() && generating {
            return (Vec::new(), true);
        }
        return (
            vec![BodySegment::Text(plain(
                effective_body,
                width,
                normal,
                visuals,
                matches,
            ))],
            generating,
        );
    };
    if prefix_len > effective_body.len() || !effective_body.is_char_boundary(prefix_len) {
        return (
            vec![BodySegment::Text(plain(
                body, width, normal, visuals, matches,
            ))],
            false,
        );
    }
    let last_text = doc.segments.iter().rev().find_map(|segment| match segment {
        wire::Segment::Text(layout) => Some(layout),
        wire::Segment::Image(_) => None,
    });
    let tail_start = compute_tail_start(effective_body, prefix_len, last_text);
    let match_unmapped = matches.iter().any(|found| {
        if found.range.end <= tail_start {
            !doc.segments.iter().any(|segment| {
                matches!(segment, wire::Segment::Text(layout) if layout.source_map.covers_exact(&found.range))
            })
        } else {
            found.range.start < tail_start
        }
    });
    if match_unmapped {
        return (
            vec![BodySegment::Text(plain(
                body, width, normal, visuals, matches,
            ))],
            false,
        );
    }

    let has_images = doc
        .segments
        .iter()
        .any(|segment| matches!(segment, wire::Segment::Image(_)));
    let ends_with_text = matches!(doc.segments.last(), Some(wire::Segment::Text(_)));
    let mut rendered = Vec::with_capacity(doc.segments.len() + 1);
    for (index, segment) in doc.segments.iter().enumerate() {
        match segment {
            wire::Segment::Text(layout) => {
                let is_last = index + 1 == doc.segments.len();
                let seg_matches: Vec<_> = matches
                    .iter()
                    .filter(|found| {
                        (found.range.end <= tail_start
                            && layout.source_map.covers_exact(&found.range))
                            || (is_last && found.range.start >= tail_start)
                    })
                    .cloned()
                    .collect();
                let out = render_parsed_text_segment(
                    effective_body,
                    width,
                    text_size,
                    visuals,
                    &seg_matches,
                    layout,
                    tail_start,
                    is_last && ends_with_text,
                );
                if (!out.job.text.is_empty() || (!has_images && !generating))
                    || (doc.segments.len() == 1 && !has_images && !generating)
                {
                    rendered.push(BodySegment::Text(out));
                }
            }
            wire::Segment::Image(image) => {
                let svg: Arc<str> = Arc::from(&effective_body[image.source_range.clone()]);
                rendered.push(BodySegment::Image {
                    svg,
                    intrinsic_size: egui::vec2(image.intrinsic_width, image.intrinsic_height),
                });
            }
        }
    }
    if !ends_with_text && tail_start < effective_body.len() {
        let tail_text = &effective_body[tail_start..];
        let mut job = egui::text::LayoutJob::default();
        job.wrap.max_width = width;
        append_tail_slice(
            tail_text, tail_start, normal, text_size, visuals, matches, &mut job,
        );
        let source_map = Arc::new(SourceMap {
            entries: vec![Mapping {
                source: tail_start..effective_body.len(),
                display: 0..job.text.len(),
                verbatim: true,
            }],
            display_bytes: job.text.len(),
        });
        rendered.push(BodySegment::Text(Layout { job, source_map }));
    }
    (rendered, generating)
}

pub(super) fn code_panels(galley: &egui::Galley) -> Vec<egui::Rect> {
    presentation::code_panels(galley)
}
