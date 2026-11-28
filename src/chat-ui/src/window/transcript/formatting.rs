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

    pub(super) fn verbatim(source: Range<usize>, display_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            entries: vec![Mapping {
                source,
                display: 0..display_bytes,
                verbatim: true,
            }],
            display_bytes,
        })
    }

    pub(super) fn first_source_start(&self) -> Option<usize> {
        self.entries.first().map(|entry| entry.source.start)
    }

    fn matches_chunk(
        &self,
        entries: &[Mapping],
        mut mapping_cursor: usize,
        display_start: usize,
        display_end: usize,
    ) -> bool {
        if self.display_bytes != display_end - display_start {
            return false;
        }
        let mut out_idx = 0_usize;
        let matched = visit_chunk_mappings(
            entries,
            &mut mapping_cursor,
            display_start..display_end,
            |expected| {
                let Some(prev) = self.entries.get(out_idx) else {
                    return false;
                };
                if prev.source != expected.source
                    || prev.display != expected.display
                    || prev.verbatim != expected.verbatim
                {
                    return false;
                }
                out_idx += 1;
                true
            },
        );
        matched && out_idx == self.entries.len()
    }
}

#[derive(Clone, Copy)]
struct RenderStyle<'a> {
    width: f32,
    text_size: f32,
    visuals: &'a egui::Visuals,
}

impl<'a> RenderStyle<'a> {
    fn body_format(self) -> egui::TextFormat {
        presentation::body_format(self.text_size, self.visuals)
    }

    fn highlighted(self, mut format: egui::TextFormat) -> egui::TextFormat {
        format.background = self.visuals.selection.bg_fill;
        format.color = self.visuals.selection.stroke.color;
        format
    }

    fn new_job(self, capacity: usize) -> egui::text::LayoutJob {
        let mut job = egui::text::LayoutJob {
            text: String::with_capacity(capacity),
            ..Default::default()
        };
        job.wrap.max_width = self.width;
        job
    }
}

pub(super) struct Layout {
    pub(super) job: egui::text::LayoutJob,
    pub(super) source_map: Arc<SourceMap>,
}

pub(super) enum BodySegment {
    Text(Layout),
    ReusedText(super::TextSegmentCache),
    Image {
        svg: PrehashedSvg,
        intrinsic_size: egui::Vec2,
    },
}

fn append_highlighted_slice(
    text: &str,
    base_offset: usize,
    normal: egui::TextFormat,
    style: RenderStyle<'_>,
    matches: &[search::Match],
    use_presentation: bool,
    job: &mut egui::text::LayoutJob,
) {
    let emit = |job: &mut egui::text::LayoutJob, slice: &str, format: egui::TextFormat| {
        if use_presentation {
            presentation::append(job, slice, format, style.text_size);
        } else {
            job.append(slice, 0.0, format);
        }
    };
    let mut cursor = 0;
    let abs_end = base_offset + text.len();
    for found in matches {
        if found.range.end <= base_offset {
            continue;
        }
        if found.range.start >= abs_end {
            break;
        }
        let start = found
            .range
            .start
            .saturating_sub(base_offset)
            .min(text.len());
        let end = found.range.end.saturating_sub(base_offset).min(text.len());
        if end <= start {
            continue;
        }
        if cursor < start {
            emit(job, &text[cursor..start], normal.clone());
        }
        emit(job, &text[start..end], style.highlighted(normal.clone()));
        cursor = end;
    }
    if cursor < text.len() {
        emit(job, &text[cursor..], normal);
    }
}

fn next_match_slice<'a>(
    matches: &'a [search::Match],
    cursor: &mut usize,
    abs_start: usize,
    abs_end: usize,
) -> &'a [search::Match] {
    while matches
        .get(*cursor)
        .is_some_and(|found| found.range.end <= abs_start)
    {
        *cursor += 1;
    }
    let mut slice_end = *cursor;
    while matches
        .get(slice_end)
        .is_some_and(|found| found.range.start < abs_end)
    {
        slice_end += 1;
    }
    let slice = &matches[*cursor..slice_end];
    while matches
        .get(*cursor)
        .is_some_and(|found| found.range.end <= abs_end)
    {
        *cursor += 1;
    }
    slice
}

fn plain_slice(
    body: &str,
    base_offset: usize,
    source_range: Range<usize>,
    style: RenderStyle<'_>,
    format: egui::TextFormat,
    matches: &[search::Match],
    use_presentation: bool,
) -> Layout {
    let slice = &body[source_range.clone()];
    let abs_start = base_offset + source_range.start;
    let abs_end = base_offset + source_range.end;
    let mut job = style.new_job(slice.len());
    append_highlighted_slice(
        slice,
        abs_start,
        format,
        style,
        matches,
        use_presentation,
        &mut job,
    );
    Layout {
        job,
        source_map: SourceMap::verbatim(abs_start..abs_end, slice.len()),
    }
}

fn plain(
    body: &str,
    width: f32,
    format: egui::TextFormat,
    visuals: &egui::Visuals,
    matches: &[search::Match],
) -> Layout {
    let style = RenderStyle {
        width,
        text_size: format.font_id.size,
        visuals,
    };
    plain_slice(body, 0, 0..body.len(), style, format, matches, false)
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
    style: RenderStyle<'_>,
    matches: &[search::Match],
    use_presentation: bool,
) -> Vec<BodySegment> {
    let normal = style.body_format();
    let ranges = top_level_block_ranges(body);
    let mut out = Vec::with_capacity(ranges.len());
    let mut match_cursor = 0_usize;
    for (start, end) in ranges {
        let block_matches = next_match_slice(
            matches,
            &mut match_cursor,
            base_offset + start,
            base_offset + end,
        );
        let layout = plain_slice(
            body,
            base_offset,
            start..end,
            style,
            normal.clone(),
            block_matches,
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
        source_map: SourceMap::verbatim(0..body.len(), body.len()),
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
    style: RenderStyle<'_>,
    matches: &[search::Match],
    parsed: &wire::TextSegment,
    tail_start: usize,
    append_tail: bool,
) -> Layout {
    let normal = style.body_format();
    let highlights: Vec<_> = matches
        .iter()
        .filter(|found| found.range.end <= tail_start)
        .map(|found| parsed.source_map.range(found.range.clone()))
        .collect();
    let has_tail = append_tail && tail_start < body_prefix.len();
    let tail_len = if has_tail {
        body_prefix.len() - tail_start
    } else {
        0
    };
    let mut job = style.new_job(parsed.text.len() + tail_len);
    let mut span_cursor = 0_usize;
    let mut match_cursor = 0_usize;
    visit_styled_runs(
        parsed,
        0..parsed.text.len(),
        style,
        &highlights,
        &mut span_cursor,
        &mut match_cursor,
        |run, effective| {
            presentation::append(&mut job, &parsed.text[run], effective, style.text_size);
            true
        },
    );
    let source_map = if has_tail {
        let tail_display_start = job.text.len();
        append_highlighted_slice(
            &body_prefix[tail_start..],
            tail_start,
            normal,
            style,
            matches,
            true,
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
        body,
        RenderStyle {
            width,
            text_size,
            visuals,
        },
        matches,
        parsed,
        tail_start,
        true,
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

pub(super) struct PrefixAnchor {
    pub(super) doc: Arc<wire::Document>,
    pub(super) prefix_segments: usize,
    pub(super) prefix_images: usize,
    last_block_display_start: usize,
    last_block_span_cursor: usize,
    last_block_mapping_cursor: usize,
    tail_start: usize,
}

#[derive(Clone, Copy, Default)]
struct ParsedCursors {
    span: usize,
    highlight: usize,
    mapping: usize,
}

impl ParsedCursors {
    fn advance_parsed_to(&mut self, parsed: &wire::TextSegment, display_start: usize) {
        while parsed
            .spans
            .get(self.span)
            .is_some_and(|span| span.end <= display_start)
        {
            self.span += 1;
        }
        while parsed
            .source_map
            .entries
            .get(self.mapping)
            .is_some_and(|entry| entry.display.end <= display_start)
        {
            self.mapping += 1;
        }
    }

    fn advance_past(
        &mut self,
        parsed: &wire::TextSegment,
        highlights: &[Range<usize>],
        display_end: usize,
    ) {
        self.advance_parsed_to(parsed, display_end);
        while highlights
            .get(self.highlight)
            .is_some_and(|found| found.end <= display_end)
        {
            self.highlight += 1;
        }
    }
}

struct LastBlockAnchor {
    display_start: usize,
    span_cursor: usize,
    mapping_cursor: usize,
}

fn visit_chunk_mappings(
    entries: &[Mapping],
    cursor: &mut usize,
    display: Range<usize>,
    mut visit: impl FnMut(Mapping) -> bool,
) -> bool {
    if display.start >= display.end {
        return true;
    }
    while let Some(entry) = entries.get(*cursor) {
        if entry.display.end <= display.start {
            *cursor += 1;
            continue;
        }
        if entry.display.start >= display.end {
            break;
        }
        let d_start = entry.display.start.max(display.start);
        let d_end = entry.display.end.min(display.end);
        let source = if entry.verbatim {
            let s_start = entry.source.start + (d_start - entry.display.start);
            let s_end = entry.source.start + (d_end - entry.display.start);
            s_start..s_end
        } else {
            entry.source.clone()
        };
        let mapped = Mapping {
            source,
            display: (d_start - display.start)..(d_end - display.start),
            verbatim: entry.verbatim,
        };
        if !visit(mapped) {
            return false;
        }
        if entry.display.end <= display.end {
            *cursor += 1;
        } else {
            break;
        }
    }
    true
}

fn build_parsed_chunk_source_map(
    entries: &[Mapping],
    mapping_cursor: &mut usize,
    display: Range<usize>,
    tail: Option<(Range<usize>, Range<usize>)>,
    total_display_bytes: usize,
) -> Arc<SourceMap> {
    let mut out = Vec::new();
    visit_chunk_mappings(entries, mapping_cursor, display, |mapped| {
        out.push(mapped);
        true
    });
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

fn visit_styled_runs(
    parsed: &wire::TextSegment,
    range: Range<usize>,
    style: RenderStyle<'_>,
    highlights: &[Range<usize>],
    span_cursor: &mut usize,
    match_cursor: &mut usize,
    mut visit: impl FnMut(Range<usize>, egui::TextFormat) -> bool,
) -> bool {
    if range.start >= range.end {
        return true;
    }
    while let Some(&span) = parsed.spans.get(*span_cursor) {
        if span.end <= range.start {
            *span_cursor += 1;
            continue;
        }
        if span.start >= range.end {
            break;
        }
        let format = presentation::text_format(span, style.text_size, style.visuals);
        let mut cursor = span.start.max(range.start);
        let end = span.end.min(range.end);
        while cursor < end {
            while highlights
                .get(*match_cursor)
                .is_some_and(|found| found.end <= cursor)
            {
                *match_cursor += 1;
            }
            let found = highlights
                .get(*match_cursor)
                .filter(|found| found.start < end);
            let (until, highlighted) = match found {
                Some(found) if found.start <= cursor => (found.end.min(end), true),
                Some(found) => (found.start.min(end), false),
                None => (end, false),
            };
            let effective = if highlighted {
                style.highlighted(format.clone())
            } else {
                format.clone()
            };
            if !visit(cursor..until, effective) {
                return false;
            }
            cursor = until;
        }
        if span.end <= range.end {
            *span_cursor += 1;
        } else {
            break;
        }
    }
    true
}

struct TailPlan<'a> {
    body: &'a str,
    start: usize,
    ranges: Vec<(usize, usize)>,
    breaks_before: bool,
    matches: &'a [search::Match],
}

impl<'a> TailPlan<'a> {
    fn new(
        body: &'a str,
        start: usize,
        parsed: &wire::TextSegment,
        matches: &'a [search::Match],
    ) -> Self {
        let has_tail = start < body.len();
        let ranges = if has_tail {
            top_level_block_ranges(&body[start..])
        } else {
            Vec::new()
        };
        let breaks_before = has_tail
            && !parsed_ends_in_code_block(parsed)
            && (parsed.text.ends_with("\n\n") || body[..start].ends_with("\n\n"));
        Self {
            body,
            start,
            ranges,
            breaks_before,
            matches,
        }
    }

    fn is_present(&self) -> bool {
        !self.ranges.is_empty()
    }

    fn combined_first(&self) -> Option<(usize, usize)> {
        (self.is_present() && !self.breaks_before)
            .then(|| self.ranges.first().copied())
            .flatten()
    }

    fn tail_skip(&self) -> usize {
        usize::from(self.is_present() && !self.breaks_before)
    }

    fn append_remaining(
        &self,
        style: RenderStyle<'_>,
        normal: &egui::TextFormat,
        layouts: &mut Vec<Layout>,
    ) {
        let mut tail_match_cursor = self
            .matches
            .partition_point(|found| found.range.end <= self.start);
        for &(t_start, t_end) in self.ranges.iter().skip(self.tail_skip()) {
            let abs_start = self.start + t_start;
            let abs_end = self.start + t_end;
            let slice_matches =
                next_match_slice(self.matches, &mut tail_match_cursor, abs_start, abs_end);
            let mut job = style.new_job(t_end - t_start);
            append_highlighted_slice(
                &self.body[abs_start..abs_end],
                abs_start,
                normal.clone(),
                style,
                slice_matches,
                true,
                &mut job,
            );
            let display_bytes = job.text.len();
            layouts.push(Layout {
                job,
                source_map: SourceMap::verbatim(abs_start..abs_end, display_bytes),
            });
        }
    }
}

fn section_matches_expected(
    sections: &[egui::text::LayoutSection],
    sec_idx: &mut usize,
    start: usize,
    end: usize,
    format: &egui::TextFormat,
) -> bool {
    let Some(sec) = sections.get(*sec_idx) else {
        return false;
    };
    if sec.byte_range.start.0 != start
        || sec.byte_range.end.0 != end
        || sec.leading_space.to_bits() != 0.0_f32.to_bits()
        || sec.format != *format
    {
        return false;
    }
    *sec_idx += 1;
    true
}

fn check_presentation_sections(
    parsed_text: &str,
    chunk_start: usize,
    run: Range<usize>,
    format: &egui::TextFormat,
    size: f32,
    sections: &[egui::text::LayoutSection],
    sec_idx: &mut usize,
) -> bool {
    let prev_ended_with_newline =
        run.start > chunk_start && parsed_text.as_bytes()[run.start - 1] == b'\n';
    presentation::visit_runs(
        &parsed_text[run.clone()],
        prev_ended_with_newline,
        format,
        |sub, is_gap| {
            let start = run.start + sub.start - chunk_start;
            let end = run.start + sub.end - chunk_start;
            if is_gap {
                let gap = presentation::gap_format(format, size);
                section_matches_expected(sections, sec_idx, start, end, &gap)
            } else {
                section_matches_expected(sections, sec_idx, start, end, format)
            }
        },
    )
}

struct ChunkRenderer<'a> {
    parsed: &'a wire::TextSegment,
    style: RenderStyle<'a>,
    normal: egui::TextFormat,
    highlights: &'a [Range<usize>],
}

impl<'a> ChunkRenderer<'a> {
    fn matches_cached(
        &self,
        prev: &super::TextSegmentCache,
        chunk: Range<usize>,
        mut cursors: ParsedCursors,
    ) -> bool {
        if prev.body.job.text != self.parsed.text[chunk.clone()] {
            return false;
        }
        if !prev.source_map.matches_chunk(
            &self.parsed.source_map.entries,
            cursors.mapping,
            chunk.start,
            chunk.end,
        ) {
            return false;
        }
        let prev_sections = &prev.body.job.sections;
        let mut sec_idx = 0_usize;
        let matched = visit_styled_runs(
            self.parsed,
            chunk.clone(),
            self.style,
            self.highlights,
            &mut cursors.span,
            &mut cursors.highlight,
            |run, effective| {
                check_presentation_sections(
                    &self.parsed.text,
                    chunk.start,
                    run,
                    &effective,
                    self.style.text_size,
                    prev_sections,
                    &mut sec_idx,
                )
            },
        );
        matched && sec_idx == prev_sections.len()
    }

    fn render_block(
        &self,
        chunk: Range<usize>,
        tail: Option<&TailPlan<'_>>,
        cursors: &mut ParsedCursors,
    ) -> Layout {
        let mut effective_end = chunk.end;
        let mut combined_tail = None;
        if let Some(tail) = tail.filter(|tail| tail.is_present()) {
            if tail.breaks_before {
                let trimmed = self.parsed.text[chunk.start..effective_end].trim_end_matches('\n');
                if !trimmed.is_empty() {
                    effective_end = chunk.start + trimmed.len();
                }
            } else {
                combined_tail = tail.combined_first();
            }
        }
        let tail_len = combined_tail.map_or(0, |(s, e)| e - s);
        let mut job = self.style.new_job((effective_end - chunk.start) + tail_len);
        visit_styled_runs(
            self.parsed,
            chunk.start..effective_end,
            self.style,
            self.highlights,
            &mut cursors.span,
            &mut cursors.highlight,
            |run, effective| {
                presentation::append(
                    &mut job,
                    &self.parsed.text[run],
                    effective,
                    self.style.text_size,
                );
                true
            },
        );
        let tail_mapping = combined_tail.zip(tail).map(|((t_start, t_end), tail)| {
            let tail_display_start = job.text.len();
            let abs_start = tail.start + t_start;
            let abs_end = tail.start + t_end;
            let tail_idx = tail
                .matches
                .partition_point(|found| found.range.end <= abs_start);
            let tail_end_idx = tail_idx
                + tail.matches[tail_idx..].partition_point(|found| found.range.start < abs_end);
            append_highlighted_slice(
                &tail.body[abs_start..abs_end],
                abs_start,
                self.normal.clone(),
                self.style,
                &tail.matches[tail_idx..tail_end_idx],
                true,
                &mut job,
            );
            (abs_start..abs_end, tail_display_start..job.text.len())
        });
        let source_map = build_parsed_chunk_source_map(
            &self.parsed.source_map.entries,
            &mut cursors.mapping,
            chunk.start..effective_end,
            tail_mapping,
            job.text.len(),
        );
        Layout { job, source_map }
    }

    fn render_chunked(
        &self,
        tail: Option<&TailPlan<'_>>,
        previous_segments: Option<&[super::CachedSegment]>,
        prev_cursor: &mut usize,
    ) -> (Vec<BodySegment>, LastBlockAnchor) {
        let blocks = parsed_block_ranges(self.parsed);
        let tail_count = tail.map_or(0, |plan| plan.ranges.len());
        let mut segments = Vec::with_capacity(blocks.len() + tail_count);
        let mut cursors = ParsedCursors::default();
        let mut anchor = LastBlockAnchor {
            display_start: 0,
            span_cursor: 0,
            mapping_cursor: 0,
        };
        let tail_present = tail.is_some_and(TailPlan::is_present);

        for (idx, &(chunk_start, chunk_end)) in blocks.iter().enumerate() {
            let is_last_parsed = idx + 1 == blocks.len();
            if is_last_parsed {
                cursors.advance_parsed_to(self.parsed, chunk_start);
                anchor = LastBlockAnchor {
                    display_start: chunk_start,
                    span_cursor: cursors.span,
                    mapping_cursor: cursors.mapping,
                };
            }
            if (!is_last_parsed || !tail_present)
                && let Some(super::CachedSegment::Text(prev)) =
                    previous_segments.and_then(|prevs| prevs.get(*prev_cursor))
                && self.matches_cached(prev, chunk_start..chunk_end, cursors)
            {
                cursors.advance_past(self.parsed, self.highlights, chunk_end);
                *prev_cursor += 1;
                segments.push(BodySegment::ReusedText(prev.clone_settled()));
                continue;
            }
            let block_tail = is_last_parsed.then_some(tail).flatten();
            segments.push(BodySegment::Text(self.render_block(
                chunk_start..chunk_end,
                block_tail,
                &mut cursors,
            )));
        }

        if let Some(tail) = tail {
            let mut tail_layouts =
                Vec::with_capacity(tail.ranges.len().saturating_sub(tail.tail_skip()));
            tail.append_remaining(self.style, &self.normal, &mut tail_layouts);
            segments.extend(tail_layouts.into_iter().map(BodySegment::Text));
        }

        (segments, anchor)
    }
}

pub(super) fn layout_document_tail(
    body: &str,
    width: f32,
    text_size: f32,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    anchor: &PrefixAnchor,
    streaming_visible_prefix: Option<usize>,
) -> Option<(Vec<BodySegment>, bool)> {
    if let Some(prefix_end) = streaming_visible_prefix
        && matches
            .last()
            .is_some_and(|found| found.range.end > prefix_end)
    {
        return None;
    }
    let effective_end = streaming_visible_prefix.unwrap_or(body.len());
    let generating = streaming_visible_prefix.is_some();
    let effective_body = &body[..effective_end];
    let tail_start = anchor.tail_start;
    if tail_start > effective_body.len() || !effective_body.is_char_boundary(tail_start) {
        return None;
    }
    let tail_match_idx = matches.partition_point(|found| found.range.end <= tail_start);
    if matches[tail_match_idx..]
        .iter()
        .any(|found| found.range.start < tail_start)
    {
        return None;
    }
    let has_images = anchor
        .doc
        .segments
        .iter()
        .any(|segment| matches!(segment, wire::Segment::Image(_)));
    let style = RenderStyle {
        width,
        text_size,
        visuals,
    };
    match anchor.doc.segments.last()? {
        wire::Segment::Text(last_text) => {
            let min_source = last_text
                .source_map
                .entries
                .get(anchor.last_block_mapping_cursor)
                .map_or(tail_start, |entry| entry.source.start);
            let prefix_match_start =
                matches[..tail_match_idx].partition_point(|found| found.range.end <= min_source);
            let highlights: Vec<_> = matches[prefix_match_start..tail_match_idx]
                .iter()
                .map(|found| last_text.source_map.range(found.range.clone()))
                .collect();
            let tail = TailPlan::new(
                effective_body,
                tail_start,
                last_text,
                &matches[tail_match_idx..],
            );
            let renderer = ChunkRenderer {
                parsed: last_text,
                style,
                normal: style.body_format(),
                highlights: &highlights,
            };
            let mut cursors = ParsedCursors {
                span: anchor.last_block_span_cursor,
                highlight: 0,
                mapping: anchor.last_block_mapping_cursor,
            };
            let mut layouts = Vec::with_capacity(1 + tail.ranges.len());
            layouts.push(renderer.render_block(
                anchor.last_block_display_start..last_text.text.len(),
                Some(&tail),
                &mut cursors,
            ));
            tail.append_remaining(style, &renderer.normal, &mut layouts);
            let mut rendered = Vec::with_capacity(layouts.len());
            for out in layouts {
                if !out.job.text.is_empty() || (!has_images && !generating) {
                    rendered.push(BodySegment::Text(out));
                }
            }
            Some((rendered, generating))
        }
        wire::Segment::Image(_) => {
            let rendered = if tail_start < effective_body.len() {
                plain_chunked(
                    &effective_body[tail_start..],
                    tail_start,
                    style,
                    &matches[tail_match_idx..],
                    true,
                )
            } else {
                Vec::new()
            };
            Some((rendered, generating))
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Document layout tracks parsed prefix chunks, streaming SVG tails, Find matches and reusable segments."
)]
pub(super) fn layout_document(
    body: &str,
    structured: bool,
    width: f32,
    text_size: f32,
    visuals: &egui::Visuals,
    matches: &[search::Match],
    parsed: Option<(&Arc<wire::Document>, usize)>,
    streaming_visible_prefix: Option<usize>,
    streaming_chunks: bool,
    previous_segments: Option<&[super::CachedSegment]>,
) -> (Vec<BodySegment>, bool, Option<PrefixAnchor>) {
    let style = RenderStyle {
        width,
        text_size,
        visuals,
    };
    let plain_doc = |text: &str, generating: bool| {
        if streaming_chunks {
            (
                plain_chunked(text, 0, style, matches, false),
                generating,
                None,
            )
        } else {
            (
                vec![BodySegment::Text(plain(
                    text,
                    width,
                    style.body_format(),
                    visuals,
                    matches,
                ))],
                generating,
                None,
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
            return (Vec::new(), true, None);
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
    let mut prefix_segments = 0_usize;
    let mut prefix_images = 0_usize;
    let mut last_anchor = LastBlockAnchor {
        display_start: 0,
        span_cursor: 0,
        mapping_cursor: 0,
    };
    let mut match_cursor = 0_usize;
    let mut prev_cursor = 0_usize;

    for (index, segment) in doc.segments.iter().enumerate() {
        match segment {
            wire::Segment::Text(layout) => {
                let is_last = index + 1 == doc.segments.len();
                while matches
                    .get(match_cursor)
                    .is_some_and(|found| found.range.end <= tail_start)
                    && !layout.source_map.covers_exact(&matches[match_cursor].range)
                    && layout
                        .source_map
                        .entries
                        .last()
                        .is_some_and(|entry| matches[match_cursor].range.end <= entry.source.start)
                {
                    match_cursor += 1;
                }
                let mut slice_end = match_cursor;
                while let Some(found) = matches.get(slice_end) {
                    if found.range.end <= tail_start {
                        if layout.source_map.covers_exact(&found.range) {
                            slice_end += 1;
                        } else {
                            break;
                        }
                    } else if is_last && found.range.start >= tail_start {
                        slice_end += 1;
                    } else {
                        break;
                    }
                }
                let seg_matches = &matches[match_cursor..slice_end];
                if streaming_chunks {
                    let highlights: Vec<_> = seg_matches
                        .iter()
                        .filter(|found| found.range.end <= tail_start)
                        .map(|found| layout.source_map.range(found.range.clone()))
                        .collect();
                    let append_tail = is_last && ends_with_text;
                    let tail = append_tail
                        .then(|| TailPlan::new(effective_body, tail_start, layout, seg_matches));
                    let renderer = ChunkRenderer {
                        parsed: layout,
                        style,
                        normal: style.body_format(),
                        highlights: &highlights,
                    };
                    let (chunks, blk_anchor) =
                        renderer.render_chunked(tail.as_ref(), previous_segments, &mut prev_cursor);
                    let chunk_non_empty = |out: &BodySegment| match out {
                        BodySegment::Text(layout) => {
                            !layout.job.text.is_empty() || (!has_images && !generating)
                        }
                        BodySegment::ReusedText(reused) => {
                            !reused.body.job.text.is_empty() || (!has_images && !generating)
                        }
                        BodySegment::Image { .. } => true,
                    };
                    if append_tail {
                        let non_empty_chunks =
                            chunks.iter().filter(|out| chunk_non_empty(out)).count();
                        prefix_segments = rendered.len() + non_empty_chunks.saturating_sub(1);
                        last_anchor = blk_anchor;
                    }
                    for out in chunks {
                        if chunk_non_empty(&out) {
                            rendered.push(out);
                        }
                    }
                } else {
                    let out = render_parsed_text_segment(
                        effective_body,
                        style,
                        seg_matches,
                        layout,
                        tail_start,
                        is_last && ends_with_text,
                    );
                    if !out.job.text.is_empty() || (!has_images && !generating) {
                        rendered.push(BodySegment::Text(out));
                    }
                }
                match_cursor = slice_end;
            }
            wire::Segment::Image(image) => {
                if matches!(
                    previous_segments.and_then(|prevs| prevs.get(prev_cursor)),
                    Some(super::CachedSegment::Image { .. })
                ) {
                    prev_cursor += 1;
                }
                rendered.push(BodySegment::Image {
                    svg: image.svg.clone(),
                    intrinsic_size: egui::vec2(image.intrinsic_width, image.intrinsic_height),
                });
                prefix_images += 1;
            }
        }
    }
    if !ends_with_text {
        prefix_segments = rendered.len();
    }
    if !ends_with_text && tail_start < effective_body.len() {
        let tail_text = &effective_body[tail_start..];
        let tail_match_start = matches.partition_point(|found| found.range.end <= tail_start);
        let tail_matches = &matches[tail_match_start..];
        if streaming_chunks {
            rendered.extend(plain_chunked(
                tail_text,
                tail_start,
                style,
                tail_matches,
                true,
            ));
        } else {
            let mut job = style.new_job(tail_text.len());
            append_highlighted_slice(
                tail_text,
                tail_start,
                style.body_format(),
                style,
                tail_matches,
                true,
                &mut job,
            );
            let display_bytes = job.text.len();
            let source_map = SourceMap::verbatim(tail_start..effective_body.len(), display_bytes);
            rendered.push(BodySegment::Text(Layout { job, source_map }));
        }
    }
    let anchor = streaming_chunks.then(|| PrefixAnchor {
        doc: Arc::clone(doc),
        prefix_segments,
        prefix_images,
        last_block_display_start: last_anchor.display_start,
        last_block_span_cursor: last_anchor.span_cursor,
        last_block_mapping_cursor: last_anchor.mapping_cursor,
        tail_start,
    });
    (rendered, generating, anchor)
}

pub(super) fn code_panels(galley: &egui::Galley) -> Arc<[egui::Rect]> {
    presentation::code_panels(galley)
}
