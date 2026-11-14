//! Bounded native Markdown presentation; original message bytes stay in history.
#![forbid(unsafe_code)]
use super::{egui, search};
use std::{ops::Range, sync::Arc};

mod pending;
mod wire;
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
}

pub(super) struct Layout {
    pub(super) job: egui::text::LayoutJob,
    pub(super) source_map: Arc<SourceMap>,
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
    wire::decode(body, &parse_wire(body)).unwrap_or_else(|| {
        let visuals = egui::Visuals::dark();
        plain(
            body,
            600.0,
            egui::TextFormat::simple(egui::FontId::proportional(16.0), visuals.text_color()),
            &visuals,
            &[],
        )
    })
}

#[cfg(test)]
pub(super) fn parse_wire(body: &str) -> Vec<u8> {
    bonsai_markdown::parse(body)
}
pub(super) fn layout_job(
    body: &str,
    structured: bool,
    width: f32,
    text_size: f32,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    parsed: Option<(&Layout, usize)>,
) -> Layout {
    let normal =
        egui::TextFormat::simple(egui::FontId::proportional(text_size), visuals.text_color());
    let Some((parsed, prefix_len)) = parsed.filter(|_| structured) else {
        return plain(body, width, normal, visuals, matches);
    };
    if prefix_len > body.len() || !body.is_char_boundary(prefix_len) {
        return plain(body, width, normal, visuals, matches);
    }
    let tail_start = if prefix_len < body.len() {
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
    } else {
        prefix_len
    };
    if matches.iter().any(|found| {
        if found.range.end <= tail_start {
            !parsed.source_map.exact_match(&found.range)
        } else {
            found.range.start < tail_start
        }
    }) {
        return plain(body, width, normal, visuals, matches);
    }
    let highlights: Vec<_> = matches
        .iter()
        .filter(|found| found.range.end <= tail_start)
        .map(|found| parsed.source_map.range(found.range.clone()))
        .collect();
    let template = egui::Visuals::dark();
    let mut job = egui::text::LayoutJob {
        text: String::with_capacity(parsed.job.text.len() + (body.len() - tail_start)),
        ..Default::default()
    };
    job.wrap.max_width = width;
    let mut next_match = 0;
    for section in &parsed.job.sections {
        let mut format = section.format.clone();
        format.font_id.size *= text_size / 16.0;
        format.color = if format.color == template.strong_text_color() {
            visuals.strong_text_color()
        } else {
            visuals.text_color()
        };
        format.background = if format.background == template.code_bg_color {
            visuals.code_bg_color
        } else {
            egui::Color32::TRANSPARENT
        };
        if format.strikethrough.width > 0.0 {
            format.strikethrough.color = format.color;
        }
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
            job.append(&parsed.job.text[cursor..until], 0.0, effective);
            cursor = until;
        }
    }
    let source_map = if tail_start < body.len() {
        let tail_display_start = job.text.len();
        let mut cursor = tail_start;
        for found in matches
            .iter()
            .filter(|found| found.range.start >= tail_start)
        {
            if cursor < found.range.start {
                job.append(&body[cursor..found.range.start], 0.0, normal.clone());
            }
            let mut highlighted = normal.clone();
            highlighted.background = visuals.selection.bg_fill;
            highlighted.color = visuals.selection.stroke.color;
            job.append(&body[found.range.clone()], 0.0, highlighted);
            cursor = found.range.end;
        }
        if cursor < body.len() {
            job.append(&body[cursor..], 0.0, normal);
        }
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
            source: tail_start..body.len(),
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
