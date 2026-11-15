//! Helpers for identifying and stripping top-level ```` ```svg ```` / ```` ~~~svg ````
//! fenced code blocks from assistant messages for speech synthesis and streaming
//! image-placeholder detection.
#![forbid(unsafe_code)]

/// Maximum buffered prefix length while waiting to classify a potential fence line.
const MAX_POTENTIAL_FENCE_PREFIX: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fence {
    ch: u8,
    len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FenceKind {
    Svg,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FenceState {
    Outside,
    InSvg { open: Fence, start: usize },
    InOther(Fence),
}

/// Output of pushing a streamed answer delta through [`StreamingSvgFilter`].
#[derive(Debug, PartialEq, Eq)]
pub(super) enum FilteredDelta {
    /// Text safe to forward to streaming speech output.
    Forward(String),
    /// The delta was suppressed (inside an SVG fence) or buffered awaiting classification.
    Suppress,
}

/// Filters top-level ```` ```svg ```` / ```` ~~~svg ```` fenced blocks out of
/// incremental streaming answer deltas without allocating on plain text deltas.
#[derive(Debug)]
pub(super) struct StreamingSvgFilter {
    state: StreamingState,
    line_buf: String,
    at_line_start: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum StreamingState {
    #[default]
    Outside,
    InSvg(Fence),
    InOther(Fence),
}

impl Default for StreamingSvgFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingSvgFilter {
    /// Creates a fresh filter at the start of an assistant response.
    pub(super) const fn new() -> Self {
        Self {
            state: StreamingState::Outside,
            line_buf: String::new(),
            at_line_start: true,
        }
    }

    /// Resets the filter between turns or after cancelling a stream.
    pub(super) fn reset(&mut self) {
        self.state = StreamingState::Outside;
        self.line_buf.clear();
        self.at_line_start = true;
    }

    /// Processes a newly received answer delta chunk.
    ///
    /// Returns [`FilteredDelta::Forward`] with the untouched input `String` when
    /// no line boundary inside `delta` can open or close a fence, preserving
    /// zero-copy forwarding for ordinary text streams.
    pub(super) fn push(&mut self, delta: String) -> FilteredDelta {
        if delta.is_empty() {
            return FilteredDelta::Suppress;
        }
        if self.can_fast_forward(&delta) {
            self.at_line_start = delta.ends_with('\n');
            return FilteredDelta::Forward(delta);
        }

        let mut out = String::new();
        self.process_chunk(&delta, &mut out);
        if out.is_empty() {
            FilteredDelta::Suppress
        } else {
            FilteredDelta::Forward(out)
        }
    }

    /// Flushes any non-SVG buffered partial line when the stream ends without a trailing newline.
    pub(super) fn finish(&mut self) -> Option<String> {
        if self.line_buf.is_empty() {
            self.reset();
            return None;
        }
        let pending = std::mem::take(&mut self.line_buf);
        let suppress = matches!(self.state, StreamingState::InSvg(_))
            || (matches!(self.state, StreamingState::Outside)
                && matches!(parse_opening_fence(&pending), Some((_, FenceKind::Svg))));
        self.reset();
        (!suppress).then_some(pending)
    }

    fn can_fast_forward(&self, delta: &str) -> bool {
        if !matches!(self.state, StreamingState::Outside) || !self.line_buf.is_empty() {
            return false;
        }
        let mut at_start = self.at_line_start;
        for line in delta.split_inclusive('\n') {
            if at_start && line_might_be_fence(line) {
                return false;
            }
            at_start = line.ends_with('\n');
        }
        true
    }

    fn process_chunk(&mut self, mut chunk: &str, out: &mut String) {
        while !chunk.is_empty() {
            if !self.line_buf.is_empty() {
                if let Some(nl_pos) = chunk.find('\n') {
                    let (rest_of_line, remaining) = chunk.split_at(nl_pos + 1);
                    self.line_buf.push_str(rest_of_line);
                    let complete_line = std::mem::take(&mut self.line_buf);
                    self.commit_complete_line(&complete_line, out);
                    self.at_line_start = true;
                    chunk = remaining;
                } else {
                    self.line_buf.push_str(chunk);
                    chunk = "";
                    let keep_buffering = match self.state {
                        StreamingState::Outside => {
                            self.line_buf.len() < MAX_POTENTIAL_FENCE_PREFIX
                                && line_might_be_fence(&self.line_buf)
                        }
                        StreamingState::InSvg(open) | StreamingState::InOther(open) => {
                            self.line_buf.len() < MAX_POTENTIAL_FENCE_PREFIX
                                && line_might_be_closer(&self.line_buf, open)
                        }
                    };
                    if !keep_buffering {
                        let flushed = std::mem::take(&mut self.line_buf);
                        if !matches!(self.state, StreamingState::InSvg(_)) {
                            out.push_str(&flushed);
                        }
                        self.at_line_start = false;
                    }
                }
                continue;
            }

            if !self.at_line_start {
                if let Some(nl_pos) = chunk.find('\n') {
                    let (rest_of_line, remaining) = chunk.split_at(nl_pos + 1);
                    if !matches!(self.state, StreamingState::InSvg(_)) {
                        out.push_str(rest_of_line);
                    }
                    self.at_line_start = true;
                    chunk = remaining;
                } else {
                    if !matches!(self.state, StreamingState::InSvg(_)) {
                        out.push_str(chunk);
                    }
                    chunk = "";
                }
                continue;
            }

            if let Some(nl_pos) = chunk.find('\n') {
                let (complete_line, remaining) = chunk.split_at(nl_pos + 1);
                self.commit_complete_line(complete_line, out);
                self.at_line_start = true;
                chunk = remaining;
            } else {
                let maybe_fence = match self.state {
                    StreamingState::Outside => line_might_be_fence(chunk),
                    StreamingState::InSvg(open) | StreamingState::InOther(open) => {
                        line_might_be_closer(chunk, open)
                    }
                };
                if maybe_fence && chunk.len() < MAX_POTENTIAL_FENCE_PREFIX {
                    self.line_buf.push_str(chunk);
                } else {
                    if !matches!(self.state, StreamingState::InSvg(_)) {
                        out.push_str(chunk);
                    }
                    self.at_line_start = false;
                }
                chunk = "";
            }
        }
    }

    fn commit_complete_line(&mut self, line: &str, out: &mut String) {
        match self.state {
            StreamingState::Outside => match parse_opening_fence(line) {
                Some((open, FenceKind::Svg)) => {
                    self.state = StreamingState::InSvg(open);
                }
                Some((open, FenceKind::Other)) => {
                    self.state = StreamingState::InOther(open);
                    out.push_str(line);
                }
                None => {
                    out.push_str(line);
                }
            },
            StreamingState::InSvg(open) => {
                if is_closing_fence(line, open) {
                    self.state = StreamingState::Outside;
                }
            }
            StreamingState::InOther(open) => {
                if is_closing_fence(line, open) {
                    self.state = StreamingState::Outside;
                }
                out.push_str(line);
            }
        }
    }
}

/// Removes top-level ```` ```svg ```` and ```` ~~~svg ```` fenced code blocks
/// (both closed and trailing unclosed) while preserving surrounding prose.
pub(super) fn strip_svg_fences(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut state = FenceState::Outside;
    let mut stripped_any = false;
    let mut just_stripped_svg = false;

    for line in text.split_inclusive('\n') {
        match state {
            FenceState::Outside => match parse_opening_fence(line) {
                Some((open, FenceKind::Svg)) => {
                    state = FenceState::InSvg { open, start: 0 };
                    stripped_any = true;
                    just_stripped_svg = true;
                }
                Some((open, FenceKind::Other)) => {
                    state = FenceState::InOther(open);
                    just_stripped_svg = false;
                    out.push_str(line);
                }
                None => {
                    if just_stripped_svg && is_blank_line(line) {
                        // Collapse extra blank lines at the seam left by a removed SVG block.
                        if out.is_empty() || ends_with_blank_line(&out) {
                            continue;
                        }
                    }
                    just_stripped_svg = false;
                    out.push_str(line);
                }
            },
            FenceState::InSvg { open, .. } => {
                if is_closing_fence(line, open) {
                    state = FenceState::Outside;
                }
            }
            FenceState::InOther(open) => {
                if is_closing_fence(line, open) {
                    state = FenceState::Outside;
                }
                out.push_str(line);
            }
        }
    }

    if !stripped_any {
        return text.to_owned();
    }

    if just_stripped_svg {
        trim_trailing_blank_lines(&mut out);
    }
    out
}

/// If `text` currently ends inside an unclosed top-level ```` ```svg ```` / ```` ~~~svg ````
/// block, returns `(visible_prefix, true)` where `visible_prefix` is the slice of `text`
/// preceding the unclosed SVG fence line. Otherwise returns `(text, false)`.
pub(super) fn split_streaming_svg(text: &str) -> (&str, bool) {
    let mut state = FenceState::Outside;
    let mut offset = 0usize;

    for line in text.split_inclusive('\n') {
        match state {
            FenceState::Outside => match parse_opening_fence(line) {
                Some((open, FenceKind::Svg)) => {
                    state = FenceState::InSvg {
                        open,
                        start: offset,
                    };
                }
                Some((open, FenceKind::Other)) => {
                    state = FenceState::InOther(open);
                }
                None => {}
            },
            FenceState::InSvg { open, .. } => {
                if is_closing_fence(line, open) {
                    state = FenceState::Outside;
                }
            }
            FenceState::InOther(open) => {
                if is_closing_fence(line, open) {
                    state = FenceState::Outside;
                }
            }
        }
        offset += line.len();
    }

    match state {
        FenceState::InSvg { start, .. } => (
            text[..start].trim_end_matches([' ', '\t', '\r', '\n']),
            true,
        ),
        FenceState::Outside | FenceState::InOther(_) => (text, false),
    }
}

fn trim_line_end(line: &str) -> &str {
    let without_nl = line.strip_suffix('\n').unwrap_or(line);
    without_nl.strip_suffix('\r').unwrap_or(without_nl)
}

fn is_blank_line(line: &str) -> bool {
    trim_line_end(line)
        .bytes()
        .all(|byte| matches!(byte, b' ' | b'\t'))
}

fn ends_with_blank_line(text: &str) -> bool {
    text.ends_with("\n\n") || text.ends_with("\r\n\r\n")
}

fn trim_trailing_blank_lines(text: &mut String) {
    while let Some(stripped) = text
        .strip_suffix("\r\n")
        .or_else(|| text.strip_suffix('\n'))
    {
        if text.len() == stripped.len() {
            break;
        }
        let check_line = stripped
            .rsplit_once('\n')
            .map_or(stripped, |(_, tail)| tail);
        if is_blank_line(check_line) || check_line.is_empty() {
            text.truncate(stripped.len());
        } else {
            break;
        }
    }
}

fn count_leading_indent_spaces(bytes: &[u8]) -> Option<usize> {
    let mut spaces = 0usize;
    for &byte in bytes {
        if byte == b' ' {
            spaces += 1;
            if spaces > 3 {
                return None;
            }
        } else {
            break;
        }
    }
    Some(spaces)
}

fn parse_opening_fence(line: &str) -> Option<(Fence, FenceKind)> {
    let trimmed_end = trim_line_end(line);
    let bytes = trimmed_end.as_bytes();
    let indent = count_leading_indent_spaces(bytes)?;
    let rest = trimmed_end.get(indent..)?;
    let first = *rest.as_bytes().first()?;
    if first != b'`' && first != b'~' {
        return None;
    }
    let len = rest.bytes().take_while(|&byte| byte == first).count();
    if len < 3 {
        return None;
    }
    let info = rest.get(len..)?.trim_matches([' ', '\t']);
    if first == b'`' && info.as_bytes().contains(&b'`') {
        return None;
    }
    let lang = info.split_whitespace().next().unwrap_or("");
    let kind = if lang.eq_ignore_ascii_case("svg") {
        FenceKind::Svg
    } else {
        FenceKind::Other
    };
    Some((Fence { ch: first, len }, kind))
}

fn is_closing_fence(line: &str, open: Fence) -> bool {
    let trimmed_end = trim_line_end(line);
    let bytes = trimmed_end.as_bytes();
    let Some(indent) = count_leading_indent_spaces(bytes) else {
        return false;
    };
    let Some(rest) = trimmed_end.get(indent..) else {
        return false;
    };
    let count = rest.bytes().take_while(|&byte| byte == open.ch).count();
    if count < open.len {
        return false;
    }
    rest.as_bytes()
        .get(count..)
        .is_some_and(|tail| tail.iter().all(|&byte| matches!(byte, b' ' | b'\t')))
}

fn line_might_be_fence(prefix: &str) -> bool {
    let trimmed = trim_line_end(prefix);
    let bytes = trimmed.as_bytes();
    let Some(indent) = count_leading_indent_spaces(bytes) else {
        return false;
    };
    let Some(rest) = bytes.get(indent..) else {
        return false;
    };
    let Some(&first) = rest.first() else {
        // Only `0..=3` spaces seen so far; a fence marker may follow.
        return true;
    };
    if first != b'`' && first != b'~' {
        return false;
    }
    let run = rest.iter().take_while(|&&byte| byte == first).count();
    if run < rest.len() {
        if run < 3 {
            return false;
        }
        if first == b'`' && rest[run..].contains(&b'`') {
            return false;
        }
    }
    true
}

fn line_might_be_closer(prefix: &str, open: Fence) -> bool {
    let trimmed = trim_line_end(prefix);
    let bytes = trimmed.as_bytes();
    let Some(indent) = count_leading_indent_spaces(bytes) else {
        return false;
    };
    let Some(rest) = bytes.get(indent..) else {
        return false;
    };
    let count = rest.iter().take_while(|&&byte| byte == open.ch).count();
    if count < rest.len() {
        count >= open.len
            && rest[count..]
                .iter()
                .all(|&byte| matches!(byte, b' ' | b'\t'))
    } else {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{FilteredDelta, StreamingSvgFilter, split_streaming_svg, strip_svg_fences};

    #[test]
    fn strip_svg_fences_removes_closed_and_unclosed_top_level_blocks() {
        let msg = "Here is a badge:\n\n```svg\n<svg viewBox=\"0 0 10 10\"></svg>\n```\n\nEnjoy!";
        assert_eq!(strip_svg_fences(msg), "Here is a badge:\n\nEnjoy!");

        let trailing_unclosed = "Drafting:\n```svg\n<svg><circle/></svg>";
        assert_eq!(strip_svg_fences(trailing_unclosed), "Drafting:\n");

        let only_svg = "~~~svg title=\"x\"\n<svg/>\n~~~\n";
        assert_eq!(strip_svg_fences(only_svg), "");
    }

    #[test]
    fn strip_svg_fences_preserves_nested_example_inside_other_fence_and_inline_code() {
        let nested = "Example:\n\n```markdown\n```svg\n<svg/>\n```\n```\n\nAlso ```inline``` text.";
        assert_eq!(strip_svg_fences(nested), nested);
    }

    #[test]
    fn split_streaming_svg_detects_only_unclosed_top_level_svg_tail() {
        assert_eq!(
            split_streaming_svg("Here:\n\n```svg\n<svg><rect"),
            ("Here:", true)
        );
        assert_eq!(split_streaming_svg("```svg\n<svg"), ("", true));

        let closed = "Here:\n\n```svg\n<svg/>\n```\nDone.";
        assert_eq!(split_streaming_svg(closed), (closed, false));

        let other_open = "Here:\n\n```markdown\n```svg\n<svg";
        assert_eq!(split_streaming_svg(other_open), (other_open, false));
    }

    #[test]
    fn streaming_filter_preserves_plain_delta_pointer_and_strips_split_svg_fences() {
        let mut filter = StreamingSvgFilter::new();
        let plain = String::from("First ");
        let plain_ptr = plain.as_ptr();
        assert!(matches!(
            filter.push(plain),
            FilteredDelta::Forward(out) if out == "First " && out.as_ptr() == plain_ptr
        ));

        let chunks = [
            "sentence.\n",
            "  ",
            "`",
            "``",
            "svg",
            "\n<svg viewBox='0 0 1 1'>",
            "\n</svg>\n",
            "   ",
            "```",
            "\nAfter.",
        ];
        let mut collected = String::new();
        for chunk in chunks {
            if let FilteredDelta::Forward(piece) = filter.push(chunk.to_owned()) {
                collected.push_str(&piece);
            }
        }
        if let Some(tail) = filter.finish() {
            collected.push_str(&tail);
        }
        assert_eq!(collected, "sentence.\nAfter.");
    }

    #[test]
    fn streaming_filter_does_not_suppress_non_svg_or_nested_fences() {
        let mut filter = StreamingSvgFilter::new();
        let chunks = [
            "Look:\n",
            "````markdown\n",
            "```svg\n<svg/>\n```\n",
            "````\nDone",
        ];
        let mut collected = String::new();
        for chunk in chunks {
            if let FilteredDelta::Forward(piece) = filter.push(chunk.to_owned()) {
                collected.push_str(&piece);
            }
        }
        if let Some(tail) = filter.finish() {
            collected.push_str(&tail);
        }
        assert_eq!(
            collected,
            "Look:\n````markdown\n```svg\n<svg/>\n```\n````\nDone"
        );
    }
}
