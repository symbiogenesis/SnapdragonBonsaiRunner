//! Bounded native Markdown presentation; original message bytes stay in history.
#![forbid(unsafe_code)]
use super::{egui, search};
use crate::window::image_cache::PrehashedSvg;
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
        svg: PrehashedSvg,
        intrinsic_size: egui::Vec2,
    },
}

fn plain_slice(
    body: &str,
    base_offset: usize,
    source_start: usize,
    source_end: usize,
    width: f32,
    format: egui::TextFormat,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    use_presentation: bool,
) -> Layout {
    let slice = &body[source_start..source_end];
    let abs_start = base_offset + source_start;
    let abs_end = base_offset + source_end;
    let mut job = egui::text::LayoutJob::default();
    let mut cursor = 0;
    for found in matches {
        let start = found.range.start.saturating_sub(abs_start).min(slice.len());
        let end = found.range.end.saturating_sub(abs_start).min(slice.len());
        if end <= start {
            continue;
        }
        if cursor < start {
            if use_presentation {
                presentation::append(
                    &mut job,
                    &slice[cursor..start],
                    format.clone(),
                    format.font_id.size,
                );
            } else {
                job.append(&slice[cursor..start], 0.0, format.clone());
            }
        }
        let mut highlighted = format.clone();
        highlighted.background = visuals.selection.bg_fill;
        highlighted.color = visuals.selection.stroke.color;
        if use_presentation {
            presentation::append(
                &mut job,
                &slice[start..end],
                highlighted,
                format.font_id.size,
            );
        } else {
            job.append(&slice[start..end], 0.0, highlighted);
        }
        cursor = end;
    }
    if cursor < slice.len() {
        if use_presentation {
            presentation::append(
                &mut job,
                &slice[cursor..],
                format.clone(),
                format.font_id.size,
            );
        } else {
            job.append(&slice[cursor..], 0.0, format);
        }
    }
    job.wrap.max_width = width;
    let source_map = SourceMap {
        entries: vec![Mapping {
            source: abs_start..abs_end,
            display: 0..slice.len(),
            verbatim: true,
        }],
        display_bytes: slice.len(),
    };
    Layout {
        job,
        source_map: Arc::new(source_map),
    }
}

fn plain(
    body: &str,
    width: f32,
    format: egui::TextFormat,
    visuals: &egui::Visuals,
    matches: &[search::Match],
) -> Layout {
    plain_slice(
        body,
        0,
        0,
        body.len(),
        width,
        format,
        visuals,
        matches,
        false,
    )
}

fn fence_marker(line: &str) -> Option<(u8, usize, &str)> {
    let bytes = line.as_bytes();
    let mut spaces = 0_usize;
    while spaces < 3 && bytes.get(spaces) == Some(&b' ') {
        spaces += 1;
    }
    let rest = &line[spaces..];
    let first = *rest.as_bytes().first()?;
    if first != b'`' && first != b'~' {
        return None;
    }
    let count = rest.bytes().take_while(|&byte| byte == first).count();
    (count >= 3).then_some((first, count, &rest[count..]))
}

fn top_level_block_ranges(text: &str) -> Vec<(usize, usize)> {
    if text.is_empty() {
        return vec![(0, 0)];
    }
    let mut blocks = Vec::new();
    let mut block_start = 0_usize;
    let mut offset = 0_usize;
    let mut fence: Option<(u8, usize)> = None;
    let mut last_non_blank_end = 0_usize;
    let mut in_blank_gap = false;

    for line in text.split_inclusive('\n') {
        let line_start = offset;
        let line_end = offset + line.len();
        offset = line_end;

        if let Some((fence_ch, fence_len)) = fence {
            if let Some((ch, count, after)) = fence_marker(line)
                && ch == fence_ch
                && count >= fence_len
                && after
                    .bytes()
                    .all(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
            {
                fence = None;
            }
            last_non_blank_end = line_end;
            in_blank_gap = false;
            continue;
        }

        if let Some((ch, count, info)) = fence_marker(line)
            && (ch == b'~' || !info.contains('`'))
        {
            if in_blank_gap && last_non_blank_end > block_start {
                blocks.push((block_start, last_non_blank_end));
                block_start = line_start;
            }
            fence = Some((ch, count));
            last_non_blank_end = line_end;
            in_blank_gap = false;
            continue;
        }

        if line == "\n" || line == "\r\n" {
            if !in_blank_gap && line_start > block_start {
                let mut trimmed = line_start;
                if text[block_start..trimmed].ends_with('\n') {
                    trimmed -= 1;
                    if text[block_start..trimmed].ends_with('\r') {
                        trimmed -= 1;
                    }
                }
                last_non_blank_end = trimmed;
                in_blank_gap = true;
            }
            continue;
        }

        if in_blank_gap && last_non_blank_end > block_start {
            blocks.push((block_start, last_non_blank_end));
            block_start = line_start;
        }
        last_non_blank_end = line_end;
        in_blank_gap = false;
    }

    if block_start < text.len() || blocks.is_empty() {
        blocks.push((block_start, text.len()));
    }
    blocks
}

fn plain_chunked(
    body: &str,
    base_offset: usize,
    width: f32,
    text_size: f32,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    use_presentation: bool,
) -> Vec<BodySegment> {
    let normal = presentation::body_format(text_size, visuals);
    let ranges = top_level_block_ranges(body);
    let mut out = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        let abs_start = base_offset + start;
        let abs_end = base_offset + end;
        let seg_matches: Vec<_> = matches
            .iter()
            .filter(|found| found.range.start < abs_end && found.range.end > abs_start)
            .cloned()
            .collect();
        let layout = plain_slice(
            body,
            base_offset,
            start,
            end,
            width,
            normal.clone(),
            visuals,
            &seg_matches,
            use_presentation,
        );
        out.push(BodySegment::Text(layout));
    }
    out
}

#[cfg(test)]
fn parse_layout(body: &str) -> wire::TextSegment {
    let fallback = || wire::TextSegment {
        text: body.to_owned(),
        spans: (!body.is_empty())
            .then_some(wire::StyleSpan::plain(0, body.len()))
            .into_iter()
            .collect(),
        source_map: Arc::new(SourceMap {
            entries: vec![Mapping {
                source: 0..body.len(),
                display: 0..body.len(),
                verbatim: true,
            }],
            display_bytes: body.len(),
        }),
    };
    let Some(doc) = wire::decode(body, &parse_wire(body)) else {
        return fallback();
    };
    match doc.segments.into_iter().next() {
        Some(wire::Segment::Text(segment)) => segment,
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

fn compute_tail_start(body: &str, prefix_len: usize, parsed: Option<&wire::TextSegment>) -> usize {
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
    parsed: &wire::TextSegment,
    tail_start: usize,
    append_tail: bool,
) -> Layout {
    let normal = presentation::body_format(text_size, visuals);
    let highlights: Vec<_> = matches
        .iter()
        .filter(|found| found.range.end <= tail_start)
        .map(|found| parsed.source_map.range(found.range.clone()))
        .collect();
    let tail_len = if append_tail && tail_start < body_prefix.len() {
        body_prefix.len() - tail_start
    } else {
        0
    };
    let mut job = egui::text::LayoutJob {
        text: String::with_capacity(parsed.text.len() + tail_len),
        ..Default::default()
    };
    job.wrap.max_width = width;
    let mut next_match = 0;
    for &span in &parsed.spans {
        let format = presentation::text_format(span, text_size, visuals);
        let mut cursor = span.start;
        let end = span.end;
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
            presentation::append(&mut job, &parsed.text[cursor..until], effective, text_size);
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
fn layout_job(
    body: &str,
    structured: bool,
    width: f32,
    text_size: f32,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    parsed: Option<(&wire::TextSegment, usize)>,
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

fn parsed_block_ranges(parsed: &wire::TextSegment) -> Vec<(usize, usize)> {
    if parsed.text.is_empty() {
        return vec![(0, 0)];
    }
    let bytes = parsed.text.as_bytes();
    let mut blocks = Vec::new();
    let mut chunk_start = 0_usize;
    let mut span_idx = 0_usize;
    let mut pos = 0_usize;
    while pos + 1 < bytes.len() {
        if bytes[pos] == b'\n' && bytes[pos + 1] == b'\n' {
            while parsed
                .spans
                .get(span_idx)
                .is_some_and(|span| span.end <= pos + 1)
            {
                span_idx += 1;
            }
            let second_in_code_block = parsed
                .spans
                .get(span_idx)
                .is_some_and(|span| span.start <= pos + 1 && span.is_code_block());
            let mut gap_end = pos + 2;
            while bytes.get(gap_end) == Some(&b'\n') {
                gap_end += 1;
            }
            if !second_in_code_block && pos > chunk_start && gap_end < bytes.len() {
                blocks.push((chunk_start, pos));
                chunk_start = gap_end;
            }
            pos = gap_end;
        } else {
            pos += 1;
        }
    }
    if chunk_start < parsed.text.len() || blocks.is_empty() {
        blocks.push((chunk_start, parsed.text.len()));
    }
    blocks
}

fn parsed_ends_in_code_block(parsed: &wire::TextSegment) -> bool {
    parsed.spans.last().is_some_and(|span| span.is_code_block())
}

fn build_parsed_chunk_source_map(
    entries: &[Mapping],
    display_start: usize,
    display_end: usize,
    tail: Option<(Range<usize>, Range<usize>)>,
    total_display_bytes: usize,
) -> Arc<SourceMap> {
    let mut out = Vec::new();
    for entry in entries {
        if entry.display.end <= display_start || entry.display.start >= display_end {
            continue;
        }
        let d_start = entry.display.start.max(display_start);
        let d_end = entry.display.end.min(display_end);
        let source = if entry.verbatim {
            let s_start = entry.source.start + (d_start - entry.display.start);
            let s_end = entry.source.start + (d_end - entry.display.start);
            s_start..s_end
        } else {
            entry.source.clone()
        };
        out.push(Mapping {
            source,
            display: (d_start - display_start)..(d_end - display_start),
            verbatim: entry.verbatim,
        });
    }
    if let Some((source, display)) = tail {
        out.push(Mapping {
            source,
            display,
            verbatim: true,
        });
    }
    Arc::new(SourceMap {
        entries: out,
        display_bytes: total_display_bytes,
    })
}

fn append_parsed_display_range(
    parsed: &wire::TextSegment,
    range: Range<usize>,
    text_size: f32,
    visuals: &egui::Visuals,
    highlights: &[Range<usize>],
    job: &mut egui::text::LayoutJob,
) {
    let mut next_match = 0;
    for &span in &parsed.spans {
        if span.end <= range.start {
            continue;
        }
        if span.start >= range.end {
            break;
        }
        let format = presentation::text_format(span, text_size, visuals);
        let mut cursor = span.start.max(range.start);
        let end = span.end.min(range.end);
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
            presentation::append(job, &parsed.text[cursor..until], effective, text_size);
            cursor = until;
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Chunked streaming layout aligns parsed display ranges with source tail blocks."
)]
fn render_parsed_text_segment_chunked(
    body_prefix: &str,
    width: f32,
    text_size: f32,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    parsed: &wire::TextSegment,
    tail_start: usize,
    append_tail: bool,
) -> Vec<Layout> {
    let normal = presentation::body_format(text_size, visuals);
    let highlights: Vec<_> = matches
        .iter()
        .filter(|found| found.range.end <= tail_start)
        .map(|found| parsed.source_map.range(found.range.clone()))
        .collect();
    let blocks = parsed_block_ranges(parsed);
    let has_tail = append_tail && tail_start < body_prefix.len();
    let tail_ranges = if has_tail {
        top_level_block_ranges(&body_prefix[tail_start..])
    } else {
        Vec::new()
    };
    let tail_breaks_before = has_tail
        && !parsed_ends_in_code_block(parsed)
        && (parsed.text.ends_with("\n\n") || body_prefix[..tail_start].ends_with("\n\n"));

    let mut layouts = Vec::with_capacity(blocks.len() + tail_ranges.len());
    for (idx, &(chunk_start, chunk_end)) in blocks.iter().enumerate() {
        let is_last_parsed = idx + 1 == blocks.len();
        let mut effective_end = chunk_end;
        let mut combined_tail: Option<(usize, usize)> = None;
        if is_last_parsed && has_tail {
            if tail_breaks_before {
                let trimmed = parsed.text[chunk_start..effective_end].trim_end_matches('\n');
                if !trimmed.is_empty() {
                    effective_end = chunk_start + trimmed.len();
                }
            } else if let Some(&(t_start, t_end)) = tail_ranges.first() {
                combined_tail = Some((t_start, t_end));
            }
        }
        let tail_len = combined_tail.map_or(0, |(s, e)| e - s);
        let mut job = egui::text::LayoutJob {
            text: String::with_capacity((effective_end - chunk_start) + tail_len),
            ..Default::default()
        };
        job.wrap.max_width = width;
        append_parsed_display_range(
            parsed,
            chunk_start..effective_end,
            text_size,
            visuals,
            &highlights,
            &mut job,
        );
        let tail_mapping = if let Some((t_start, t_end)) = combined_tail {
            let tail_display_start = job.text.len();
            let abs_start = tail_start + t_start;
            let abs_end = tail_start + t_end;
            append_tail_slice(
                &body_prefix[abs_start..abs_end],
                abs_start,
                normal.clone(),
                text_size,
                visuals,
                matches,
                &mut job,
            );
            Some((abs_start..abs_end, tail_display_start..job.text.len()))
        } else {
            None
        };
        let source_map = build_parsed_chunk_source_map(
            &parsed.source_map.entries,
            chunk_start,
            effective_end,
            tail_mapping,
            job.text.len(),
        );
        layouts.push(Layout { job, source_map });
    }
    let tail_skip = usize::from(has_tail && !tail_breaks_before);
    for &(t_start, t_end) in tail_ranges.iter().skip(tail_skip) {
        let abs_start = tail_start + t_start;
        let abs_end = tail_start + t_end;
        let seg_matches: Vec<_> = matches
            .iter()
            .filter(|found| found.range.start < abs_end && found.range.end > abs_start)
            .cloned()
            .collect();
        let mut job = egui::text::LayoutJob {
            text: String::with_capacity(t_end - t_start),
            ..Default::default()
        };
        job.wrap.max_width = width;
        append_tail_slice(
            &body_prefix[abs_start..abs_end],
            abs_start,
            normal.clone(),
            text_size,
            visuals,
            &seg_matches,
            &mut job,
        );
        let source_map = Arc::new(SourceMap {
            entries: vec![Mapping {
                source: abs_start..abs_end,
                display: 0..job.text.len(),
                verbatim: true,
            }],
            display_bytes: job.text.len(),
        });
        layouts.push(Layout { job, source_map });
    }
    layouts
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
    streaming_chunks: bool,
) -> (Vec<BodySegment>, bool) {
    let plain_doc = |text: &str, generating: bool| {
        if streaming_chunks {
            (
                plain_chunked(text, 0, width, text_size, visuals, matches, false),
                generating,
            )
        } else {
            let normal = presentation::body_format(text_size, visuals);
            (
                vec![BodySegment::Text(plain(
                    text, width, normal, visuals, matches,
                ))],
                generating,
            )
        }
    };
    if !structured {
        return plain_doc(body, false);
    }
    if let Some(prefix_end) = streaming_visible_prefix
        && matches.iter().any(|found| found.range.end > prefix_end)
    {
        // A Find match hits inside the incomplete streaming SVG block: reveal raw source.
        return plain_doc(body, false);
    }
    let effective_end = streaming_visible_prefix.unwrap_or(body.len());
    let generating = streaming_visible_prefix.is_some();
    let effective_body = &body[..effective_end];

    let Some((doc, prefix_len)) = parsed else {
        if effective_body.is_empty() && generating {
            return (Vec::new(), true);
        }
        return plain_doc(effective_body, generating);
    };
    if prefix_len > effective_body.len() || !effective_body.is_char_boundary(prefix_len) {
        return plain_doc(body, false);
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
        return plain_doc(body, false);
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
                if streaming_chunks {
                    let chunks = render_parsed_text_segment_chunked(
                        effective_body,
                        width,
                        text_size,
                        visuals,
                        &seg_matches,
                        layout,
                        tail_start,
                        is_last && ends_with_text,
                    );
                    for out in chunks {
                        if !out.job.text.is_empty() || (!has_images && !generating) {
                            rendered.push(BodySegment::Text(out));
                        }
                    }
                } else {
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
                    if !out.job.text.is_empty() || (!has_images && !generating) {
                        rendered.push(BodySegment::Text(out));
                    }
                }
            }
            wire::Segment::Image(image) => {
                rendered.push(BodySegment::Image {
                    svg: image.svg.clone(),
                    intrinsic_size: egui::vec2(image.intrinsic_width, image.intrinsic_height),
                });
            }
        }
    }
    if !ends_with_text && tail_start < effective_body.len() {
        let tail_text = &effective_body[tail_start..];
        if streaming_chunks {
            rendered.extend(plain_chunked(
                tail_text, tail_start, width, text_size, visuals, matches, true,
            ));
        } else {
            let mut job = egui::text::LayoutJob::default();
            job.wrap.max_width = width;
            let normal = presentation::body_format(text_size, visuals);
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
    }
    (rendered, generating)
}

pub(super) fn code_panels(galley: &egui::Galley) -> Vec<egui::Rect> {
    presentation::code_panels(galley)
}
