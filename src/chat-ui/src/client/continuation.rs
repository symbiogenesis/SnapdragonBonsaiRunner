//! Seamless, prefix-cache-preserving multi-segment response continuation.
//!
//! When a standard single-turn reply stops at `max_completion_tokens`
//! (`Completion::TokenLimit`), this worker-side runner stitches up to
//! `MAX_TOTAL_SEGMENTS` segments into one continuous assistant message while
//! keeping `wire_history`'s raw prefix intact for `PromptReuse`.

use super::{
    CONTEXT_LIMIT_MESSAGE, Command, Completion, Event, EventSender, Failure, Message, PromptReuse,
    RATE_LIMIT_MESSAGE, ResponseOptions, Result, check_commands, combined_turn_text,
    generate_with_stream_hooks, notify, serialize_request_with_options,
};
use bonsai_windows_process::OwnedProcess;
use std::{
    borrow::Cow,
    sync::{Arc, mpsc::Receiver},
    thread,
    time::Duration,
};

const MAX_TOTAL_SEGMENTS: u32 = 32;
const MIN_CONTINUATION_TOKENS: u32 = 64;
const MAX_STITCHED_BYTES: usize = 768 * 1024;
const MAX_WIRE_HISTORY_MESSAGES: usize = 127;
const MAX_CONTEXT_STEPDOWNS: usize = 6;
const MAX_RATE_LIMIT_RETRIES: usize = 3;
const ESTIMATOR_HEADROOM_TOKENS: usize = 128;
const SEAM_TAIL_BYTES: usize = 256;
const SEAM_HOLD_MAX_BYTES: usize = 256;
const SEAM_EARLY_FLUSH_BYTES: usize = 24;
const ANCHOR_QUOTE_BYTES: usize = 120;
pub(super) const RATE_LIMIT_WAIT_MS: u64 = 1_050;
const RATE_LIMIT_SLICE_MS: u64 = 25;
const REPETITION_WINDOW_BYTES: usize = 16 * 1024;
const REPETITION_NEEDLE_BYTES: usize = 120;
const REPETITION_LONG_NEEDLE_BYTES: usize = 256;
const REPETITION_INTRA_NEEDLE_BYTES: usize = 64;

const THINKING_HANDOFF_PROMPT: &str = "You have finished your reasoning above. \
Now write your complete final response directly, without repeating your reasoning.";

const META_PREAMBLES: [&str; 17] = [
    "continuing from where i left off",
    "continuing from where we left off",
    "continuing from the previous response",
    "continuing seamlessly",
    "continuing",
    "here is the continuation",
    "here's the continuation",
    "here is the rest",
    "here's the rest",
    "here is the rest of the response",
    "here's the rest of the response",
    "continuation of the response",
    "continuation",
    "continued",
    "(continued)",
    "[continued]",
    "the rest",
];

const POLITE_PREAMBLE_PREFIXES: [&str; 8] = [
    "sure, ",
    "sure! ",
    "of course, ",
    "of course! ",
    "certainly, ",
    "certainly! ",
    "okay, ",
    "ok, ",
];

#[derive(Clone, Debug, PartialEq, Eq)]
struct OpenFence {
    ch: char,
    len: usize,
    info: String,
}

impl OpenFence {
    fn display_tag(&self) -> String {
        let mut out = String::with_capacity(self.len + self.info.len());
        for _ in 0..self.len {
            out.push(self.ch);
        }
        if !self.info.is_empty() {
            out.push_str(&self.info);
        }
        out
    }

    fn language_token(&self) -> &str {
        self.info.split_whitespace().next().unwrap_or("")
    }

    fn is_closing_line(&self, line: &str) -> bool {
        let Some((ch, len, after)) = parse_fence_run(line) else {
            return false;
        };
        ch == self.ch && len >= self.len && after.bytes().all(|b| b == b' ' || b == b'\t')
    }
}

fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut pos = index.min(text.len());
    while pos > 0 && !text.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

fn ceil_char_boundary(text: &str, index: usize) -> usize {
    let mut pos = index.min(text.len());
    while pos < text.len() && !text.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

fn tail_slice(text: &str, max_bytes: usize) -> &str {
    let start = ceil_char_boundary(text, text.len().saturating_sub(max_bytes));
    &text[start..]
}

fn parse_fence_run(line: &str) -> Option<(char, usize, &str)> {
    let stripped = line.trim_end_matches('\r');
    let spaces = stripped.bytes().take_while(|&b| b == b' ').count();
    if spaces > 3 {
        return None;
    }
    let rest = &stripped[spaces..];
    let ch = rest.chars().next()?;
    if ch != '`' && ch != '~' {
        return None;
    }
    let fence_byte = if ch == '`' { b'`' } else { b'~' };
    let len = rest.bytes().take_while(|&b| b == fence_byte).count();
    if len < 3 {
        return None;
    }
    Some((ch, len, &rest[len..]))
}

fn open_fence_state(text: &str) -> Option<OpenFence> {
    let mut open: Option<OpenFence> = None;
    for line in text.split('\n') {
        let Some((ch, len, after)) = parse_fence_run(line) else {
            continue;
        };
        if let Some(current) = &open {
            if ch == current.ch
                && len >= current.len
                && after.bytes().all(|b| b == b' ' || b == b'\t')
            {
                open = None;
            }
        } else if ch != '`' || !after.contains('`') {
            open = Some(OpenFence {
                ch,
                len,
                info: after.trim().to_owned(),
            });
        }
    }
    open
}

fn contains_control_marker(text: &str) -> bool {
    const MARKERS: [&str; 6] = ["<|", "<think", "</think", "<tool_", "</tool_", "<tts_"];
    MARKERS.iter().any(|marker| text.contains(marker))
        || text
            .bytes()
            .any(|b| b < b' ' && !matches!(b, b'\n' | b'\r' | b'\t'))
}

fn message_has_control_marker(message: &Message) -> bool {
    contains_control_marker(&message.content)
        || message
            .reasoning_content
            .as_deref()
            .is_some_and(contains_control_marker)
}

fn build_continuation_prompt(stitched_text: &str, open_fence: Option<&OpenFence>) -> String {
    if stitched_text.trim().is_empty() {
        return THINKING_HANDOFF_PROMPT.to_owned();
    }
    let fence_note = open_fence.map_or_else(String::new, |fence| {
        if fence.language_token().eq_ignore_ascii_case("svg") {
            " You are currently inside an open ```svg block; emit only the remaining SVG \
             elements and the closing ``` without restarting <svg> or emitting an opening fence."
                .to_owned()
        } else {
            format!(
                " You are currently inside an open `{}` fenced block; \
                 do not emit an opening fence or repeat earlier lines.",
                fence.display_tag()
            )
        }
    });
    let tail = tail_slice(stitched_text, ANCHOR_QUOTE_BYTES);
    if tail.is_empty() || contains_control_marker(tail) {
        format!(
            "Continue your previous response seamlessly from the exact point where it stopped. \
             Do not repeat already-written text and do not add an introductory preamble.{fence_note}"
        )
    } else {
        format!(
            "Continue your previous response seamlessly from the exact point where it stopped. \
             Do not repeat already-written text and do not add an introductory preamble.{fence_note}\n\
             Exact ending of your previous text:\n{tail}"
        )
    }
}

fn normalize_preamble_line(line: &str) -> String {
    let lower = line.trim().to_ascii_lowercase();
    let without_polite = POLITE_PREAMBLE_PREFIXES
        .iter()
        .find_map(|&prefix| lower.strip_prefix(prefix))
        .unwrap_or(lower.as_str());
    without_polite
        .trim()
        .trim_end_matches([':', '-', '—', '.', '…', '!'])
        .trim()
        .to_owned()
}

fn is_meta_preamble_line(line: &str) -> bool {
    if line.len() > 80 {
        return false;
    }
    let normalized = normalize_preamble_line(line);
    if META_PREAMBLES.iter().any(|&p| normalized == p) {
        return true;
    }
    if let Some(digits) = normalized.strip_prefix("part ") {
        return (1..=2).contains(&digits.len()) && digits.bytes().all(|b| b.is_ascii_digit());
    }
    false
}

fn could_be_preamble_prefix(trimmed: &str) -> bool {
    if trimmed.len() > 80 {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    let without_polite = POLITE_PREAMBLE_PREFIXES
        .iter()
        .find_map(|&prefix| lower.strip_prefix(prefix))
        .unwrap_or(lower.as_str());
    if POLITE_PREAMBLE_PREFIXES
        .iter()
        .any(|&prefix| prefix.starts_with(&lower))
    {
        return true;
    }
    if META_PREAMBLES
        .iter()
        .any(|&p| p.starts_with(without_polite) || without_polite.starts_with(p))
    {
        return true;
    }
    if "part ".starts_with(without_polite) {
        return true;
    }
    if let Some(rest) = without_polite.strip_prefix("part ") {
        let trimmed_rest = rest.trim_end_matches([':', '-', '—', '.', '…', '!']);
        return trimmed_rest.len() <= 2 && trimmed_rest.bytes().all(|b| b.is_ascii_digit());
    }
    false
}

fn strip_preamble_line(text: &str) -> Option<(&str, bool)> {
    let newline_pos = text.find('\n')?;
    let line = text[..newline_pos].trim_end_matches('\r');
    if !is_meta_preamble_line(line) {
        return None;
    }
    let after_first = &text[newline_pos + 1..];
    let ws_len = after_first
        .bytes()
        .take_while(|&b| b == b'\r' || b == b'\n')
        .count();
    let extra_newlines = after_first[..ws_len]
        .bytes()
        .filter(|&b| b == b'\n')
        .count();
    Some((&after_first[ws_len..], extra_newlines >= 1))
}

fn strip_redundant_fence<'a>(prev_tail: &str, text: &'a str, open: &OpenFence) -> (&'a str, bool) {
    let leading_newlines = text
        .bytes()
        .take_while(|&b| b == b'\r' || b == b'\n')
        .count();
    let after_newlines = &text[leading_newlines..];
    let Some(newline_pos) = after_newlines.find('\n') else {
        return (text, false);
    };
    let first_line = &after_newlines[..newline_pos];
    let Some((ch, len, after)) = parse_fence_run(first_line) else {
        return (text, false);
    };
    if ch != open.ch || len < 3 {
        return (text, false);
    }
    let open_lang = open.language_token();
    let cand_lang = after.trim().split_whitespace().next().unwrap_or("");
    let after_fence = &after_newlines[newline_pos + 1..];
    if !open_lang.is_empty() {
        if !cand_lang.is_empty() && cand_lang.eq_ignore_ascii_case(open_lang) {
            return (after_fence, true);
        }
        return (text, false);
    }
    if cand_lang.is_empty()
        && !prev_tail.is_empty()
        && !after_fence.is_empty()
        && (find_overlap(prev_tail, after_fence) > 0
            || find_overlap(prev_tail.trim_end(), after_fence.trim_start()) > 0)
    {
        return (after_fence, true);
    }
    (text, false)
}

const fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn find_overlap(prev_tail: &str, candidate: &str) -> usize {
    let max_len = prev_tail.len().min(candidate.len());
    let mut len = floor_char_boundary(candidate, max_len);
    while len >= 4 {
        let Some(prev_start) = prev_tail.len().checked_sub(len) else {
            break;
        };
        if prev_tail.is_char_boundary(prev_start) && prev_tail.ends_with(&candidate[..len]) {
            let prev_before = prev_start
                .checked_sub(1)
                .and_then(|idx| prev_tail.as_bytes().get(idx).copied());
            let prev_last = prev_tail.as_bytes().last().copied();
            let cand_after = candidate.as_bytes().get(len).copied();
            let at_word_boundary = prev_before.is_none_or(|b| !is_word_byte(b));
            let split_word_join = len >= 4
                && at_word_boundary
                && prev_last.is_some_and(is_word_byte)
                && cand_after.is_some_and(is_word_byte);
            let boundary_phrase = len >= 6
                && at_word_boundary
                && (candidate[..len].bytes().any(|b| b.is_ascii_whitespace())
                    || cand_after.is_none_or(|b| !is_word_byte(b)));
            if len >= 12 || split_word_join || boundary_phrase {
                return len;
            }
        }
        len = floor_char_boundary(candidate, len.saturating_sub(1));
    }
    0
}

fn strip_leading_ellipsis<'a>(prev_tail: &str, rest: &'a str) -> &'a str {
    if prev_tail.ends_with('.') || prev_tail.ends_with('…') {
        return rest;
    }
    let after = if let Some(after) = rest.strip_prefix("...") {
        after
    } else if let Some(after) = rest.strip_prefix('…') {
        after
    } else {
        return rest;
    };
    if prev_tail
        .bytes()
        .last()
        .is_some_and(|b| b.is_ascii_whitespace())
    {
        after.strip_prefix(' ').unwrap_or(after)
    } else {
        after
    }
}

fn strip_trimmed_prev_overlap<'a>(
    prev_tail: &str,
    prev_trimmed: &str,
    candidate: &'a str,
) -> Option<&'a str> {
    let overlap = find_overlap(prev_trimmed, candidate);
    if overlap == 0 {
        return None;
    }
    let after = &candidate[overlap..];
    Some(if prev_tail.ends_with([' ', '\t']) {
        after.strip_prefix(' ').unwrap_or(after)
    } else {
        after
    })
}

fn strip_tail_overlap<'a>(prev_tail: &str, rest: &'a str) -> (&'a str, bool) {
    if prev_tail.is_empty() || rest.is_empty() {
        return (rest, false);
    }
    let exact = find_overlap(prev_tail, rest);
    if exact > 0 {
        return (&rest[exact..], true);
    }
    let trimmed = rest.trim_start();
    let leading_ws = rest.len().saturating_sub(trimmed.len());
    let has_short_ws = leading_ws > 0 && leading_ws <= 8;
    if has_short_ws {
        let trimmed_overlap = find_overlap(prev_tail, trimmed);
        if trimmed_overlap > 0 {
            return (&trimmed[trimmed_overlap..], true);
        }
    }
    if prev_tail
        .bytes()
        .last()
        .is_some_and(|b| b.is_ascii_whitespace())
    {
        let prev_trimmed = prev_tail.trim_end();
        if !prev_trimmed.is_empty() {
            if let Some(after) = strip_trimmed_prev_overlap(prev_tail, prev_trimmed, rest) {
                return (after, true);
            }
            if has_short_ws
                && let Some(after) = strip_trimmed_prev_overlap(prev_tail, prev_trimmed, trimmed)
            {
                return (after, true);
            }
        }
    }
    (rest, false)
}

fn starts_with_horizontal_rule(text: &str) -> bool {
    let first_line = text
        .split('\n')
        .next()
        .unwrap_or(text)
        .trim_end_matches('\r')
        .trim();
    let Some(ch) = first_line.bytes().next() else {
        return false;
    };
    if !matches!(ch, b'-' | b'*' | b'_') {
        return false;
    }
    let count = first_line.bytes().filter(|&b| b == ch).count();
    count >= 3
        && first_line
            .bytes()
            .all(|b| b == ch || b == b' ' || b == b'\t')
}

fn starts_with_block_heading_or_fence(rest: &str) -> bool {
    let first_line = rest.split('\n').next().unwrap_or(rest);
    if parse_fence_run(first_line).is_some() {
        return true;
    }
    let hashes = rest.bytes().take_while(|&b| b == b'#').count();
    if (1..=6).contains(&hashes)
        && rest
            .as_bytes()
            .get(hashes)
            .is_some_and(|&b| b == b' ' || b == b'\t')
    {
        return true;
    }
    rest.contains('\n') && starts_with_horizontal_rule(rest)
}

fn starts_with_list_or_quote_or_table(rest: &str) -> bool {
    if rest.starts_with("> ")
        || rest.starts_with("- ")
        || rest.starts_with("* ")
        || rest.starts_with("+ ")
        || rest.starts_with("| ")
    {
        return true;
    }
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    (1..=3).contains(&digits)
        && (rest[digits..].starts_with(". ") || rest[digits..].starts_with(") "))
}

struct SeamResolution<'a> {
    prefix: &'static str,
    text: &'a str,
}

fn resolve_seam<'a>(
    prev_tail: &str,
    open_fence: Option<&OpenFence>,
    incoming: &'a str,
) -> SeamResolution<'a> {
    let mut rest = incoming;
    let mut had_preamble_blank_line = false;
    if let Some((stripped, blank_line)) = strip_preamble_line(rest) {
        rest = stripped;
        had_preamble_blank_line = blank_line;
    }
    let mut stripped_redundant_fence = false;
    if let Some(fence) = open_fence {
        let (after_fence, stripped) = strip_redundant_fence(prev_tail, rest, fence);
        rest = after_fence;
        stripped_redundant_fence = stripped;
    } else {
        rest = strip_leading_ellipsis(prev_tail, rest);
    }
    let (rest, had_overlap) = strip_tail_overlap(prev_tail, rest);

    let mut prefix = "";
    if !prev_tail.is_empty() && !rest.is_empty() && !prev_tail.ends_with('\n') {
        let starts_nl = rest.starts_with('\n') || rest.starts_with('\r');
        if let Some(fence) = open_fence {
            let first_line = rest.split('\n').next().unwrap_or(rest);
            if !starts_nl
                && (fence.is_closing_line(first_line) || (stripped_redundant_fence && !had_overlap))
            {
                prefix = "\n";
            }
        } else if !starts_nl {
            if starts_with_block_heading_or_fence(rest) {
                prefix = "\n\n";
            } else if starts_with_list_or_quote_or_table(rest) {
                prefix = "\n";
            } else if had_preamble_blank_line
                && prev_tail.ends_with(['.', '!', '?', ':', ')', '"', '\''])
            {
                prefix = "\n\n";
            }
        }
    }

    SeamResolution { prefix, text: rest }
}

struct SeamFilter {
    prev_tail: String,
    open_fence: Option<OpenFence>,
    buffer: String,
    resolved: bool,
}

fn starts_with_fence_char(text: &str) -> bool {
    text.starts_with('`') || text.starts_with('~')
}

impl SeamFilter {
    const fn new(prev_tail: String, open_fence: Option<OpenFence>) -> Self {
        Self {
            prev_tail,
            open_fence,
            buffer: String::new(),
            resolved: false,
        }
    }

    fn contained_in_tail(&self, text: &str) -> bool {
        self.prev_tail.contains(text) || self.prev_tail.trim_end().contains(text)
    }

    fn potential_preamble_or_fence(&self) -> bool {
        if let Some(nl) = self.buffer.find('\n') {
            if let Some((after_preamble, _)) = strip_preamble_line(&self.buffer) {
                if self.open_fence.is_some()
                    && starts_with_fence_char(after_preamble.trim_start())
                    && !after_preamble.contains('\n')
                {
                    return true;
                }
                if after_preamble.len() < SEAM_EARLY_FLUSH_BYTES
                    && (after_preamble.is_empty()
                        || after_preamble.starts_with("...")
                        || after_preamble.starts_with('…')
                        || self.prev_tail.contains(after_preamble)
                        || self
                            .prev_tail
                            .trim_end()
                            .contains(after_preamble.trim_start()))
                {
                    return true;
                }
            }
            let first_line = self.buffer[..nl].trim_end_matches('\r');
            let rest = &self.buffer[nl + 1..];
            return self.open_fence.is_some()
                && first_line.is_empty()
                && !rest.contains('\n')
                && starts_with_fence_char(rest.trim_start());
        }
        let trimmed = self.buffer.trim_start();
        if self.open_fence.is_some() && starts_with_fence_char(trimmed) {
            return true;
        }
        if self.open_fence.is_none() && (trimmed.starts_with("...") || trimmed.starts_with('…')) {
            return true;
        }
        could_be_preamble_prefix(trimmed)
    }

    fn can_flush_early(&self) -> bool {
        if self.buffer.len() < SEAM_EARLY_FLUSH_BYTES || self.potential_preamble_or_fence() {
            return false;
        }
        let trimmed = self.buffer.trim_start();
        if !trimmed.is_empty()
            && (self.prev_tail.contains(&self.buffer) || self.contained_in_tail(trimmed))
        {
            return false;
        }
        true
    }

    fn flush_buffer(&mut self, emit: &mut impl FnMut(&str) -> Result<()>) -> Result<()> {
        if self.resolved {
            return Ok(());
        }
        self.resolved = true;
        let out = resolve_seam(&self.prev_tail, self.open_fence.as_ref(), &self.buffer);
        if !out.prefix.is_empty() {
            emit(out.prefix)?;
        }
        if !out.text.is_empty() {
            emit(out.text)?;
        }
        self.buffer.clear();
        Ok(())
    }

    fn push(&mut self, delta: &str, emit: &mut impl FnMut(&str) -> Result<()>) -> Result<()> {
        if delta.is_empty() {
            return Ok(());
        }
        if self.resolved {
            return emit(delta);
        }
        self.buffer.push_str(delta);
        if self.buffer.len() >= SEAM_HOLD_MAX_BYTES || self.can_flush_early() {
            self.flush_buffer(emit)?;
        }
        Ok(())
    }

    fn finish(&mut self, emit: &mut impl FnMut(&str) -> Result<()>) -> Result<()> {
        if !self.resolved {
            self.flush_buffer(emit)?;
        }
        Ok(())
    }
}

fn count_non_overlapping(haystack: &str, needle: &str, max_count: usize) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut search_from = 0;
    let mut matches = 0_usize;
    while let Some(offset) = haystack[search_from..].find(needle) {
        matches += 1;
        if matches >= max_count {
            return matches;
        }
        search_from += offset + needle.len();
    }
    matches
}

fn is_repetition_loop(prior_text: &str, prev_chunk: &str, new_chunk: &str) -> bool {
    let prev_trimmed = prev_chunk.trim();
    let new_trimmed = new_chunk.trim();
    if prev_trimmed.len() >= 32 && new_trimmed == prev_trimmed {
        return true;
    }
    if new_chunk.len() >= 512 {
        let intra_needle = tail_slice(new_chunk, REPETITION_INTRA_NEEDLE_BYTES);
        if intra_needle.trim().len() >= 32 && count_non_overlapping(new_chunk, intra_needle, 4) >= 4
        {
            return true;
        }
    }
    let haystack = tail_slice(prior_text, REPETITION_WINDOW_BYTES);
    if new_chunk.len() >= REPETITION_LONG_NEEDLE_BYTES
        && prior_text.len() >= REPETITION_LONG_NEEDLE_BYTES
    {
        let long_needle = tail_slice(new_chunk, REPETITION_LONG_NEEDLE_BYTES);
        if long_needle.trim().len() >= REPETITION_NEEDLE_BYTES && haystack.contains(long_needle) {
            return true;
        }
    }
    if new_chunk.len() < REPETITION_NEEDLE_BYTES || prior_text.len() < REPETITION_NEEDLE_BYTES * 2 {
        return false;
    }
    let needle = tail_slice(new_chunk, REPETITION_NEEDLE_BYTES);
    if needle.trim().len() < 40 {
        return false;
    }
    count_non_overlapping(haystack, needle, 2) >= 2
}

fn next_chunk_budget(
    wire_history: &[Message],
    system_appendix: Option<&str>,
    context_tokens: usize,
    base_cap: u32,
) -> Option<u32> {
    let sys_tokens = crate::context::system_prompt_tokens()
        .saturating_add(system_appendix.map_or(0, crate::context::text_tokens));
    let hist_tokens = crate::context::history_tokens(wire_history);
    let prompt_est = sys_tokens.saturating_add(hist_tokens);
    let available = context_tokens
        .saturating_sub(prompt_est)
        .saturating_sub(crate::context::RESERVE_TOKENS)
        .saturating_sub(ESTIMATOR_HEADROOM_TOKENS);
    let available_u32 = u32::try_from(available).unwrap_or(u32::MAX);
    let budget = base_cap.min(available_u32);
    (budget >= MIN_CONTINUATION_TOKENS).then_some(budget)
}

fn fold_turn_appendix_for_wire(wire_history: &mut [Message], turn_appendix: Option<&str>) {
    let Some(appendix) = turn_appendix.filter(|text| !text.is_empty()) else {
        return;
    };
    if let Some(user_msg) = wire_history.last_mut().filter(|msg| msg.role == "user") {
        let combined = combined_turn_text(user_msg.content.as_ref(), Some(appendix));
        user_msg.content = Arc::from(combined.as_ref());
    }
}

pub(super) fn sleep_interruptible(commands: &Receiver<Command>, total_ms: u64) -> Result<()> {
    let mut waited = 0_u64;
    while waited < total_ms {
        check_commands(commands)?;
        let step = RATE_LIMIT_SLICE_MS.min(total_ms - waited);
        thread::sleep(Duration::from_millis(step));
        waited += step;
    }
    check_commands(commands)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Continuation worker runs against explicit broker connection and prompt-reuse policy."
)]
pub(super) fn run(
    port: u16,
    key: &str,
    pending: &[Message],
    options: &ResponseOptions,
    context_tokens: usize,
    prompt_reuse: PromptReuse,
    commands: &Receiver<Command>,
    events: &EventSender,
    broker: Option<&OwnedProcess>,
) -> Result<(Message, Completion)> {
    run_with_wait(
        port,
        key,
        pending,
        options,
        context_tokens,
        prompt_reuse,
        commands,
        events,
        broker,
        sleep_interruptible,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Continuation worker runs against explicit broker connection and prompt-reuse policy."
)]
pub(super) fn run_with_wait(
    port: u16,
    key: &str,
    pending: &[Message],
    options: &ResponseOptions,
    context_tokens: usize,
    prompt_reuse: PromptReuse,
    commands: &Receiver<Command>,
    events: &EventSender,
    broker: Option<&OwnedProcess>,
    wait_rate_limit: impl FnMut(&Receiver<Command>, u64) -> Result<()>,
) -> Result<(Message, Completion)> {
    run_with(
        pending,
        options,
        context_tokens,
        prompt_reuse,
        commands,
        events,
        |request, on_text, on_reasoning| {
            let (msg, completion, _) = generate_with_stream_hooks(
                port,
                key,
                request,
                false,
                commands,
                broker,
                on_text,
                on_reasoning,
            )?;
            Ok((msg, completion))
        },
        wait_rate_limit,
    )
}

fn prepare_continuation_options(
    base: &ResponseOptions,
    chunk_budget: u32,
    prompt_reuse: PromptReuse,
) -> ResponseOptions {
    let mut cont = base.clone();
    cont.max_completion_tokens = chunk_budget;
    cont.turn_appendix = None;
    if cont.enable_thinking {
        if cont.cognitive_controls && prompt_reuse.enabled() {
            // Preserve `enable_thinking: true` and `reasoning_effort` so the
            // system prompt stays byte-identical for KV+GDN prefix reuse.
            cont.reasoning_budget_override = Some(1);
        } else {
            cont.enable_thinking = false;
        }
    }
    cont
}

fn build_stitched_message(stitched_text: String, stitched_reasoning: String) -> Message {
    Message {
        role: "assistant",
        content: stitched_text.into(),
        reasoning_content: (!stitched_reasoning.is_empty()).then(|| stitched_reasoning.into()),
        images: Arc::default(),
    }
}

fn compacted_wire_history(
    base_wire_history: &[Message],
    stitched_text: &str,
    stitched_reasoning: &str,
    cont_prompt: &str,
) -> Option<Vec<Message>> {
    if base_wire_history.len().saturating_add(2) > MAX_WIRE_HISTORY_MESSAGES {
        return None;
    }
    let mut compacted = Vec::with_capacity(base_wire_history.len() + 2);
    compacted.extend_from_slice(base_wire_history);
    compacted.push(build_stitched_message(
        stitched_text.to_owned(),
        stitched_reasoning.to_owned(),
    ));
    compacted.push(Message {
        role: "user",
        content: cont_prompt.into(),
        reasoning_content: None,
        images: Arc::default(),
    });
    Some(compacted)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Test-injectable worker loop separates generation and rate-limit wait hooks."
)]
fn run_with<G, S>(
    pending: &[Message],
    options: &ResponseOptions,
    context_tokens: usize,
    prompt_reuse: PromptReuse,
    commands: &Receiver<Command>,
    events: &EventSender,
    mut generate: G,
    wait_rate_limit: S,
) -> Result<(Message, Completion)>
where
    G: FnMut(
        Vec<u8>,
        &mut dyn FnMut(Cow<'_, str>) -> Result<()>,
        &mut dyn FnMut(Cow<'_, str>) -> Result<()>,
    ) -> Result<(Message, Completion)>,
    S: FnMut(&Receiver<Command>, u64) -> Result<()>,
{
    let initial_request = serialize_request_with_options(pending, options)?;
    let mut stitched_text = String::new();
    let mut stitched_reasoning = String::new();

    let (first_raw, first_completion) = {
        let mut on_text = |delta: Cow<'_, str>| -> Result<()> {
            stitched_text.push_str(&delta);
            events.stream_text(delta, false, commands)
        };
        let mut on_reasoning = |delta: Cow<'_, str>| -> Result<()> {
            stitched_reasoning.push_str(&delta);
            events.stream_text(delta, true, commands)
        };
        generate(initial_request, &mut on_text, &mut on_reasoning)?
    };

    if first_completion == Completion::Finished {
        return Ok((
            build_stitched_message(stitched_text, stitched_reasoning),
            Completion::Finished,
        ));
    }

    continue_after_token_limit(
        pending,
        first_raw,
        options,
        context_tokens,
        prompt_reuse,
        commands,
        events,
        generate,
        wait_rate_limit,
    )
}

struct ContinuationState<'a> {
    base_wire_history: Vec<Message>,
    wire_history: Vec<Message>,
    stitched_text: String,
    stitched_reasoning: String,
    prev_emitted_chunk: String,
    observed_budget_ceiling: Option<u32>,
    options: &'a ResponseOptions,
    context_tokens: usize,
    prompt_reuse: PromptReuse,
}

impl<'a> ContinuationState<'a> {
    fn clamped_budget(&self) -> Option<u32> {
        let initial = next_chunk_budget(
            &self.wire_history,
            self.options.system_appendix.as_deref(),
            self.context_tokens,
            self.options.max_completion_tokens,
        )?;
        let budget = self
            .observed_budget_ceiling
            .map_or(initial, |ceil| initial.min(ceil));
        (budget >= MIN_CONTINUATION_TOKENS).then_some(budget)
    }

    fn prepare_segment_budget(&mut self, cont_prompt: &str) -> Option<u32> {
        if self.prompt_reuse.enabled() {
            if self.wire_history.len().saturating_add(1) > MAX_WIRE_HISTORY_MESSAGES {
                if self.wire_history.len() <= self.base_wire_history.len().saturating_add(1) {
                    return None;
                }
                self.wire_history = compacted_wire_history(
                    &self.base_wire_history,
                    &self.stitched_text,
                    &self.stitched_reasoning,
                    cont_prompt,
                )?;
            } else {
                self.wire_history.push(Message {
                    role: "user",
                    content: cont_prompt.into(),
                    reasoning_content: None,
                    images: Arc::default(),
                });
            }
        } else {
            self.wire_history = compacted_wire_history(
                &self.base_wire_history,
                &self.stitched_text,
                &self.stitched_reasoning,
                cont_prompt,
            )?;
        }
        self.clamped_budget()
    }

    fn compact_for_serialize_retry(&mut self, cont_prompt: &str) -> Option<u32> {
        self.wire_history = compacted_wire_history(
            &self.base_wire_history,
            &self.stitched_text,
            &self.stitched_reasoning,
            cont_prompt,
        )?;
        self.clamped_budget()
    }

    fn stream_attempt<G>(
        &mut self,
        request: Vec<u8>,
        open_fence: Option<&OpenFence>,
        commands: &Receiver<Command>,
        events: &EventSender,
        generate: &mut G,
    ) -> Result<(Message, Completion, String)>
    where
        G: FnMut(
            Vec<u8>,
            &mut dyn FnMut(Cow<'_, str>) -> Result<()>,
            &mut dyn FnMut(Cow<'_, str>) -> Result<()>,
        ) -> Result<(Message, Completion)>,
    {
        let prev_tail = tail_slice(&self.stitched_text, SEAM_TAIL_BYTES).to_owned();
        let mut filter = SeamFilter::new(prev_tail, open_fence.cloned());
        let mut emitted_chunk = String::new();
        let mut stream_err: Option<Failure> = None;

        let outcome = {
            let stitched_text = &mut self.stitched_text;
            let mut on_text = |delta: Cow<'_, str>| -> Result<()> {
                let result = filter.push(&delta, &mut |clean| {
                    emitted_chunk.push_str(clean);
                    stitched_text.push_str(clean);
                    events.stream_text(Cow::Borrowed(clean), false, commands)
                });
                if let Err(err) = result {
                    stream_err = Some(err);
                    return Err(err);
                }
                Ok(())
            };
            let mut on_reasoning = |_delta: Cow<'_, str>| -> Result<()> { Ok(()) };
            generate(request, &mut on_text, &mut on_reasoning)
        };

        if let Some(err) = stream_err {
            return Err(err);
        }

        let (raw_msg, completion) = outcome?;
        filter.finish(&mut |clean| {
            emitted_chunk.push_str(clean);
            self.stitched_text.push_str(clean);
            events.stream_text(Cow::Borrowed(clean), false, commands)
        })?;
        Ok((raw_msg, completion, emitted_chunk))
    }

    fn run_segment<G, S>(
        &mut self,
        cont_prompt: &str,
        open_fence: Option<&OpenFence>,
        initial_budget: u32,
        commands: &Receiver<Command>,
        events: &EventSender,
        generate: &mut G,
        wait_rate_limit: &mut S,
    ) -> Result<Option<(Message, Completion, String)>>
    where
        G: FnMut(
            Vec<u8>,
            &mut dyn FnMut(Cow<'_, str>) -> Result<()>,
            &mut dyn FnMut(Cow<'_, str>) -> Result<()>,
        ) -> Result<(Message, Completion)>,
        S: FnMut(&Receiver<Command>, u64) -> Result<()>,
    {
        let mut chunk_budget = initial_budget;
        let mut stepdowns = 0_usize;
        let mut rate_retries = 0_usize;
        let mut compacted_on_serialize = !self.prompt_reuse.enabled();

        loop {
            let cont_options =
                prepare_continuation_options(self.options, chunk_budget, self.prompt_reuse);
            let request = match serialize_request_with_options(&self.wire_history, &cont_options) {
                Ok(bytes) => bytes,
                Err(_)
                    if !compacted_on_serialize
                        && self.wire_history.len()
                            > self.base_wire_history.len().saturating_add(2) =>
                {
                    compacted_on_serialize = true;
                    let Some(recomputed) = self.compact_for_serialize_retry(cont_prompt) else {
                        return Ok(None);
                    };
                    chunk_budget = recomputed;
                    continue;
                }
                Err(_) => return Ok(None),
            };

            match self.stream_attempt(request, open_fence, commands, events, generate) {
                Ok(chunk) => return Ok(Some(chunk)),
                Err(Failure::Rejected(CONTEXT_LIMIT_MESSAGE))
                    if stepdowns < MAX_CONTEXT_STEPDOWNS
                        && chunk_budget / 2 >= MIN_CONTINUATION_TOKENS =>
                {
                    stepdowns += 1;
                    chunk_budget /= 2;
                    self.observed_budget_ceiling = Some(
                        self.observed_budget_ceiling
                            .map_or(chunk_budget, |prev| prev.min(chunk_budget)),
                    );
                }
                Err(Failure::Rejected(RATE_LIMIT_MESSAGE))
                    if rate_retries < MAX_RATE_LIMIT_RETRIES =>
                {
                    rate_retries += 1;
                    wait_rate_limit(commands, RATE_LIMIT_WAIT_MS)?;
                }
                Err(Failure::Rejected(_)) => return Ok(None),
                Err(err) => return Err(err),
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shared continuation loop preserves exact wire prefix, context budget, and stream hooks."
)]
pub(super) fn continue_after_token_limit<G, S>(
    pending: &[Message],
    first_raw: Message,
    options: &ResponseOptions,
    context_tokens: usize,
    prompt_reuse: PromptReuse,
    commands: &Receiver<Command>,
    events: &EventSender,
    mut generate: G,
    mut wait_rate_limit: S,
) -> Result<(Message, Completion)>
where
    G: FnMut(
        Vec<u8>,
        &mut dyn FnMut(Cow<'_, str>) -> Result<()>,
        &mut dyn FnMut(Cow<'_, str>) -> Result<()>,
    ) -> Result<(Message, Completion)>,
    S: FnMut(&Receiver<Command>, u64) -> Result<()>,
{
    let stitched_text = first_raw.content.to_string();
    let stitched_reasoning = first_raw
        .reasoning_content
        .as_deref()
        .unwrap_or_default()
        .to_owned();

    if (stitched_text.trim().is_empty() && stitched_reasoning.trim().is_empty())
        || message_has_control_marker(&first_raw)
    {
        return Ok((
            build_stitched_message(stitched_text, stitched_reasoning),
            Completion::TokenLimit,
        ));
    }

    let mut base_wire_history = pending.to_vec();
    fold_turn_appendix_for_wire(&mut base_wire_history, options.turn_appendix.as_deref());

    let wire_history = if prompt_reuse.enabled() {
        let mut history = base_wire_history.clone();
        history.push(first_raw);
        history
    } else {
        Vec::new()
    };

    let mut state = ContinuationState {
        base_wire_history,
        wire_history,
        prev_emitted_chunk: stitched_text.clone(),
        stitched_text,
        stitched_reasoning,
        observed_budget_ceiling: None,
        options,
        context_tokens,
        prompt_reuse,
    };
    let mut final_completion = Completion::TokenLimit;

    for segment in 2..=MAX_TOTAL_SEGMENTS {
        check_commands(commands)?;
        if state
            .stitched_text
            .len()
            .saturating_add(state.stitched_reasoning.len())
            >= MAX_STITCHED_BYTES
        {
            break;
        }

        let open_fence = open_fence_state(&state.stitched_text);
        let cont_prompt = build_continuation_prompt(&state.stitched_text, open_fence.as_ref());
        let Some(chunk_budget) = state.prepare_segment_budget(&cont_prompt) else {
            break;
        };

        notify(events, Event::Continuing(segment))?;

        let Some((raw_msg, completion, emitted_chunk)) = state.run_segment(
            &cont_prompt,
            open_fence.as_ref(),
            chunk_budget,
            commands,
            events,
            &mut generate,
            &mut wait_rate_limit,
        )?
        else {
            break;
        };

        if emitted_chunk.trim().is_empty() {
            if completion == Completion::Finished && !state.stitched_text.trim().is_empty() {
                final_completion = Completion::Finished;
            }
            break;
        }
        let prior_len = state
            .stitched_text
            .len()
            .saturating_sub(emitted_chunk.len());
        if is_repetition_loop(
            &state.stitched_text[..prior_len],
            &state.prev_emitted_chunk,
            &emitted_chunk,
        ) {
            break;
        }
        if completion == Completion::Finished {
            final_completion = Completion::Finished;
            break;
        }
        if message_has_control_marker(&raw_msg) {
            break;
        }

        if prompt_reuse.enabled() {
            state.wire_history.push(raw_msg);
        }
        state.prev_emitted_chunk = emitted_chunk;
    }

    Ok((
        build_stitched_message(state.stitched_text, state.stitched_reasoning),
        final_completion,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{CognitiveOptions, ReasoningEffort, event_channel};
    use std::{collections::VecDeque, sync::mpsc};

    fn user_msg(content: &str) -> Message {
        Message {
            role: "user",
            content: content.into(),
            reasoning_content: None,
            images: Arc::default(),
        }
    }

    fn assistant_msg(content: &str, reasoning: Option<&str>) -> Message {
        Message {
            role: "assistant",
            content: content.into(),
            reasoning_content: reasoning.map(Into::into),
            images: Arc::default(),
        }
    }

    fn default_options(max_completion_tokens: u32) -> ResponseOptions {
        ResponseOptions {
            max_completion_tokens,
            enable_thinking: false,
            daisyrose_scale: None,
            cognitive: CognitiveOptions::default(),
            cognitive_controls: true,
            reset_on_cancel: true,
            system_appendix: None,
            turn_appendix: None,
            request_logprobs: false,
            temperature_nudge: 0.0,
            reasoning_budget_override: None,
        }
    }

    fn no_wait(_commands: &Receiver<Command>, _ms: u64) -> Result<()> {
        Ok(())
    }

    fn collect_streamed(events: &super::super::EventReceiver) -> (String, String, Vec<u32>) {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut parts = Vec::new();
        for event in events.try_iter() {
            match event {
                Event::Text(chunk) => text.push_str(&chunk),
                Event::Reasoning(chunk) => reasoning.push_str(&chunk),
                Event::Continuing(part) => parts.push(part),
                _ => {}
            }
        }
        (text, reasoning, parts)
    }

    #[test]
    fn single_shot_finished_makes_one_request() -> Result<()> {
        let (_commands, incoming) = mpsc::channel();
        let (events, outgoing) = event_channel();
        let pending = vec![user_msg("Hello")];
        let mut calls = 0_usize;
        let (msg, completion) = run_with(
            &pending,
            &default_options(4096),
            32_768,
            PromptReuse::Latest,
            &incoming,
            &events,
            |_req, on_text, _on_reasoning| {
                calls += 1;
                on_text(Cow::Borrowed("Complete response."))?;
                Ok((
                    assistant_msg("Complete response.", None),
                    Completion::Finished,
                ))
            },
            no_wait,
        )?;
        assert_eq!(calls, 1);
        assert_eq!(completion, Completion::Finished);
        assert_eq!(msg.content.as_ref(), "Complete response.");
        let (streamed, _, parts) = collect_streamed(&outgoing);
        assert_eq!(streamed, "Complete response.");
        assert!(parts.is_empty());
        Ok(())
    }

    #[test]
    fn multi_segment_stitches_into_one_message_and_preserves_wire_prefix() -> Result<()> {
        let (_commands, incoming) = mpsc::channel();
        let (events, outgoing) = event_channel();
        let pending = vec![user_msg("Write a long chapter")];
        let mut seen_requests = Vec::new();
        let mut responses = VecDeque::from([
            (
                vec![
                    "First section with details, ending cleanly at ",
                    "the boundary.",
                ],
                None,
                Completion::TokenLimit,
            ),
            (
                vec![
                    "the boundary. ",
                    "Second section continues seamlessly across tokens.",
                ],
                None,
                Completion::TokenLimit,
            ),
            (
                vec![" And the final conclusion wraps up everything."],
                None,
                Completion::Finished,
            ),
        ]);

        let (msg, completion) = run_with(
            &pending,
            &default_options(4096),
            65_536,
            PromptReuse::Latest,
            &incoming,
            &events,
            |req, on_text, _on_reasoning| {
                let json: serde_json::Value =
                    serde_json::from_slice(&req).map_err(|_| Failure::Message("invalid JSON"))?;
                seen_requests.push(json);
                let (chunks, reasoning, comp) = responses
                    .pop_front()
                    .ok_or(Failure::Message("unexpected request"))?;
                let mut raw = String::new();
                for chunk in chunks {
                    raw.push_str(chunk);
                    on_text(Cow::Borrowed(chunk))?;
                }
                Ok((assistant_msg(&raw, reasoning), comp))
            },
            no_wait,
        )?;

        assert_eq!(completion, Completion::Finished);
        assert_eq!(seen_requests.len(), 3);
        // Wire prefix from request N must be an exact prefix of request N+1 messages.
        let m0 = seen_requests[0]["messages"]
            .as_array()
            .ok_or_else(|| Failure::Message("missing m0 messages"))?;
        let m1 = seen_requests[1]["messages"]
            .as_array()
            .ok_or_else(|| Failure::Message("missing m1 messages"))?;
        let m2 = seen_requests[2]["messages"]
            .as_array()
            .ok_or_else(|| Failure::Message("missing m2 messages"))?;
        assert_eq!(m0.len(), 2);
        assert_eq!(m1.len(), 4);
        assert_eq!(m2.len(), 6);
        assert_eq!(&m1[..2], &m0[..]);
        assert_eq!(&m2[..4], &m1[..]);
        // Raw overlap is preserved in wire_history for KV+GDN cache parity,
        // while emitted/committed text strips the seam overlap seamlessly.
        let expected = "First section with details, ending cleanly at the boundary. \
                        Second section continues seamlessly across tokens. \
                        And the final conclusion wraps up everything.";
        assert_eq!(msg.content.as_ref(), expected);
        let (streamed, _, parts) = collect_streamed(&outgoing);
        assert_eq!(streamed, expected);
        // Event::Continuing coalesces in place when undrained so it never exhausts CONTROL_RESERVE.
        assert_eq!(parts, vec![3]);
        Ok(())
    }

    #[test]
    fn thinking_continuation_preserves_prefix_and_suppresses_part2_reasoning() -> Result<()> {
        let (_commands, incoming) = mpsc::channel();
        let (events, outgoing) = event_channel();
        let mut opts = default_options(4096);
        opts.enable_thinking = true;
        opts.cognitive.reasoning_effort = ReasoningEffort::Low;
        opts.cognitive_controls = true;
        opts.turn_appendix = Some(Arc::from("[Memory card: ocean]"));
        let pending = vec![user_msg("Explain tidal resonance")];

        let mut seen_requests = Vec::new();
        let mut step = 0_usize;
        let (msg, completion) = run_with(
            &pending,
            &opts,
            65_536,
            PromptReuse::Latest,
            &incoming,
            &events,
            |req, on_text, on_reasoning| {
                step += 1;
                let json: serde_json::Value =
                    serde_json::from_slice(&req).map_err(|_| Failure::Message("invalid JSON"))?;
                seen_requests.push(json);
                match step {
                    1 => {
                        on_reasoning(Cow::Borrowed("Deep initial thought"))?;
                        on_text(Cow::Borrowed("Part one prose"))?;
                        Ok((
                            assistant_msg("Part one prose", Some("Deep initial thought")),
                            Completion::TokenLimit,
                        ))
                    }
                    _ => {
                        on_reasoning(Cow::Borrowed("\n"))?;
                        on_text(Cow::Borrowed(" and part two prose."))?;
                        Ok((
                            assistant_msg(" and part two prose.", Some("\n")),
                            Completion::Finished,
                        ))
                    }
                }
            },
            no_wait,
        )?;

        assert_eq!(completion, Completion::Finished);
        assert_eq!(seen_requests.len(), 2);
        // Request 2 keeps enable_thinking=true & reasoning_effort="low" so system prompt matches!
        assert_eq!(seen_requests[1]["enable_thinking"], true);
        assert_eq!(seen_requests[1]["reasoning_effort"], "low");
        assert_eq!(seen_requests[1]["reasoning_budget"], 1);
        // turn_appendix was materialized on message[1] (original user) and NOT moved to message[3]!
        let m0 = seen_requests[0]["messages"]
            .as_array()
            .ok_or_else(|| Failure::Message("missing m0 messages"))?;
        let m1 = seen_requests[1]["messages"]
            .as_array()
            .ok_or_else(|| Failure::Message("missing m1 messages"))?;
        assert_eq!(m0[1]["content"], m1[1]["content"]);
        assert!(
            !m1[3]["content"]
                .as_str()
                .is_some_and(|s| s.contains("[Memory card: ocean]"))
        );

        assert_eq!(msg.content.as_ref(), "Part one prose and part two prose.");
        assert_eq!(
            msg.reasoning_content.as_deref(),
            Some("Deep initial thought")
        );
        let (streamed_text, streamed_reasoning, _) = collect_streamed(&outgoing);
        assert_eq!(streamed_text, "Part one prose and part two prose.");
        assert_eq!(streamed_reasoning, "Deep initial thought");
        Ok(())
    }

    #[test]
    fn publisher_thinking_continuation_disables_thinking_without_cognitive_controls() -> Result<()>
    {
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = event_channel();
        let mut opts = default_options(4096);
        opts.enable_thinking = true;
        opts.cognitive_controls = false;
        let pending = vec![user_msg("Explain")];
        let mut seen_requests = Vec::new();
        let mut step = 0_usize;
        let (msg, _) = run_with(
            &pending,
            &opts,
            65_536,
            PromptReuse::Latest,
            &incoming,
            &events,
            |req, on_text, on_reasoning| {
                step += 1;
                let json: serde_json::Value =
                    serde_json::from_slice(&req).map_err(|_| Failure::Message("invalid JSON"))?;
                seen_requests.push(json);
                if step == 1 {
                    on_reasoning(Cow::Borrowed("Thought"))?;
                    Ok((assistant_msg("", Some("Thought")), Completion::TokenLimit))
                } else {
                    on_text(Cow::Borrowed("Final rescued answer."))?;
                    Ok((
                        assistant_msg("Final rescued answer.", None),
                        Completion::Finished,
                    ))
                }
            },
            no_wait,
        )?;
        assert_eq!(seen_requests[1]["enable_thinking"], false);
        assert!(seen_requests[1].get("reasoning_budget").is_none());
        assert!(
            seen_requests[1]["messages"][3]["content"]
                .as_str()
                .is_some_and(|s| s.contains("finished your reasoning"))
        );
        assert_eq!(msg.content.as_ref(), "Final rescued answer.");
        assert_eq!(msg.reasoning_content.as_deref(), Some("Thought"));
        Ok(())
    }

    #[test]
    fn seam_filter_strips_preambles_redundant_fences_overlaps_and_handles_utf8() -> Result<()> {
        let mut out = String::new();
        let mut f1 = SeamFilter::new("the quick brown fox".into(), None);
        for d in ["brown ", "fox ", "jumps over the lazy dog and runs."] {
            f1.push(d, &mut |s| {
                out.push_str(s);
                Ok(())
            })?;
        }
        f1.finish(&mut |s| {
            out.push_str(s);
            Ok(())
        })?;
        assert_eq!(out, " jumps over the lazy dog and runs.");

        // Split-word completion
        let mut out2 = String::new();
        let mut f2 = SeamFilter::new("compute implem".into(), None);
        f2.push("implementation of the kernel now.", &mut |s| {
            out2.push_str(s);
            Ok(())
        })?;
        f2.finish(&mut |s| {
            out2.push_str(s);
            Ok(())
        })?;
        assert_eq!(out2, "entation of the kernel now.");

        // Short common word must NOT be falsely stripped
        let mut out3 = String::new();
        let mut f3 = SeamFilter::new("she had ".into(), None);
        f3.push("had a wonderful time in Paris.", &mut |s| {
            out3.push_str(s);
            Ok(())
        })?;
        f3.finish(&mut |s| {
            out3.push_str(s);
            Ok(())
        })?;
        assert_eq!(out3, "had a wonderful time in Paris.");

        // Redundant Rust fence stripped
        let fence = open_fence_state("Intro\n```rust\nfn a() {\n");
        assert_eq!(
            fence.as_ref().map(|f| (f.ch, f.len, f.info.as_str())),
            Some(('`', 3, "rust"))
        );
        let mut out4 = String::new();
        let mut f4 = SeamFilter::new("fn a() {\n".into(), fence);
        f4.push(
            "```rust\nfn a() {\n    println!(\"ok\");\n}\n```",
            &mut |s| {
                out4.push_str(s);
                Ok(())
            },
        )?;
        f4.finish(&mut |s| {
            out4.push_str(s);
            Ok(())
        })?;
        assert_eq!(out4, "    println!(\"ok\");\n}\n```");

        // Meta-preamble stripped; non-newline tail gets clean paragraph break
        let mut out5 = String::new();
        let mut f5 = SeamFilter::new("End of paragraph one.".into(), None);
        f5.push(
            "Continuing from where I left off:\n\nParagraph two begins cleanly.",
            &mut |s| {
                out5.push_str(s);
                Ok(())
            },
        )?;
        f5.finish(&mut |s| {
            out5.push_str(s);
            Ok(())
        })?;
        assert_eq!(out5, "\n\nParagraph two begins cleanly.");

        // If prev_tail already ends with newline, no duplicate blank lines are inserted
        let mut out5b = String::new();
        let mut f5b = SeamFilter::new("End of paragraph one.\n\n".into(), None);
        f5b.push(
            "Sure, here is the continuation:\n\nParagraph two begins cleanly.",
            &mut |s| {
                out5b.push_str(s);
                Ok(())
            },
        )?;
        f5b.finish(&mut |s| {
            out5b.push_str(s);
            Ok(())
        })?;
        assert_eq!(out5b, "Paragraph two begins cleanly.");

        // Multibyte UTF-8 safety
        let mut out6 = String::new();
        let mut f6 = SeamFilter::new("日本語のテスト🌱".into(), None);
        f6.push("テスト🌱が続く。", &mut |s| {
            out6.push_str(s);
            Ok(())
        })?;
        f6.finish(&mut |s| {
            out6.push_str(s);
            Ok(())
        })?;
        assert_eq!(out6, "が続く。");
        Ok(())
    }

    #[test]
    fn closing_fence_is_preserved_inside_open_rust_and_svg_blocks() -> Result<()> {
        fn run_filter(prev: &str, open_doc: &str, incoming: &str) -> Result<String> {
            let fence = open_fence_state(open_doc);
            let mut out = String::new();
            let mut filter = SeamFilter::new(prev.into(), fence);
            filter.push(incoming, &mut |s| {
                out.push_str(s);
                Ok(())
            })?;
            filter.finish(&mut |s| {
                out.push_str(s);
                Ok(())
            })?;
            Ok(out)
        }

        // Open ```rust block closing directly on segment 2 preserves ``` and adds \n when missing
        let rust_doc = "Example:\n```rust\nfn main() {\n    println!(\"hi\");\n}";
        let out_rust = run_filter(
            "    println!(\"hi\");\n}",
            rust_doc,
            "```\n\nNext paragraph explains main.",
        )?;
        assert_eq!(out_rust, "\n```\n\nNext paragraph explains main.");

        // Open ```svg block closing directly preserves ``` and adds \n when missing
        let svg_doc = "```svg\n<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\"><circle cx=\"5\" cy=\"5\" r=\"4\"/></svg>";
        let out_svg = run_filter(
            "<circle cx=\"5\" cy=\"5\" r=\"4\"/></svg>",
            svg_doc,
            "```\n\nHere is the icon.",
        )?;
        assert_eq!(out_svg, "\n```\n\nHere is the icon.");

        // Bare open ``` block without overlap preserves ``` as closing fence
        let bare_doc = "```\nsome code\nmore code";
        let out_bare = run_filter("more code", bare_doc, "```\n\nAfter bare block.")?;
        assert_eq!(out_bare, "\n```\n\nAfter bare block.");

        // Preamble followed by redundant opening fence inside open ```rust block strips both
        let out_both = run_filter(
            "fn main() {\n",
            "```rust\nfn main() {\n",
            "Part 2:\n```rust\n    println!(\"ok\");\n}\n```",
        )?;
        assert_eq!(out_both, "    println!(\"ok\");\n}\n```");

        // Standalone "Part 2: Architecture" heading is NOT treated as a meta-preamble
        let out_heading = run_filter(
            "Section 1 done.",
            "Section 1 done.",
            "Part 2: Architecture\nDetails follow.",
        )?;
        assert_eq!(out_heading, "Part 2: Architecture\nDetails follow.");
        Ok(())
    }

    #[test]
    fn ellipsis_trailing_space_overlap_and_block_separators() -> Result<()> {
        fn run_filter_chunks(prev: &str, chunks: &[&str]) -> Result<String> {
            let mut out = String::new();
            let mut filter = SeamFilter::new(prev.into(), None);
            for chunk in chunks {
                filter.push(chunk, &mut |s| {
                    out.push_str(s);
                    Ok(())
                })?;
            }
            filter.finish(&mut |s| {
                out.push_str(s);
                Ok(())
            })?;
            Ok(out)
        }

        // Leading continuation ellipsis stripped cleanly
        assert_eq!(
            run_filter_chunks("The quick brown ", &["...fox jumps over."])?,
            "fox jumps over."
        );
        assert_eq!(
            run_filter_chunks("The quick brown", &["… fox jumps."])?,
            " fox jumps."
        );

        // Trailing-space prev_tail overlap avoids double space
        assert_eq!(
            run_filter_chunks("the quick brown fox ", &["brown fox jumps high."])?,
            "jumps high."
        );

        // Markdown heading, list, and table after non-newline-terminated prev_tail
        assert_eq!(
            run_filter_chunks("Intro done.", &["## Next Section\nBody here."])?,
            "\n\n## Next Section\nBody here."
        );
        assert_eq!(
            run_filter_chunks("Items:", &["- First item\n- Second item"])?,
            "\n- First item\n- Second item"
        );
        assert_eq!(
            run_filter_chunks("Results:", &["| Col A | Col B |\n| --- | --- |"])?,
            "\n| Col A | Col B |\n| --- | --- |"
        );

        // Code identifier across boundary is NOT corrupted by space insertion
        assert_eq!(
            run_filter_chunks("Use React.", &["Component here."])?,
            "Component here."
        );
        Ok(())
    }

    #[test]
    fn control_marker_and_overlap_only_eos_guards() -> Result<()> {
        // <tts_pad> is excluded from continuation prompt anchor
        let prompt = build_continuation_prompt("Prose before <tts_pad> marker", None);
        assert!(!prompt.contains("<tts_pad>"));

        // SVG open fence uses targeted SVG instruction
        let svg_fence = open_fence_state("```svg\n<svg viewBox=\"0 0 10 10\">");
        let svg_prompt =
            build_continuation_prompt("<svg viewBox=\"0 0 10 10\">", svg_fence.as_ref());
        assert!(svg_prompt.contains("remaining SVG elements"));
        assert!(svg_prompt.contains("without restarting <svg>"));

        // Model-emitted control marker in raw_msg stops continuation cleanly without failing
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = event_channel();
        let pending = vec![user_msg("Write")];
        let mut calls = 0_usize;
        let (msg, completion) = run_with(
            &pending,
            &default_options(4096),
            65_536,
            PromptReuse::Disabled,
            &incoming,
            &events,
            |_req, on_text, _| {
                calls += 1;
                if calls == 1 {
                    on_text(Cow::Borrowed("Safe start."))?;
                    Ok((assistant_msg("Safe start.", None), Completion::TokenLimit))
                } else {
                    on_text(Cow::Borrowed(" Trailing <tts_pad> text."))?;
                    Ok((
                        assistant_msg(" Trailing <tts_pad> text.", None),
                        Completion::TokenLimit,
                    ))
                }
            },
            no_wait,
        )?;
        assert_eq!(calls, 2);
        assert_eq!(completion, Completion::TokenLimit);
        assert_eq!(msg.content.as_ref(), "Safe start. Trailing <tts_pad> text.");

        // Overlap-only final segment with Completion::Finished settles as Finished
        let (_commands2, incoming2) = mpsc::channel();
        let (events2, _outgoing2) = event_channel();
        let mut calls2 = 0_usize;
        let (msg2, completion2) = run_with(
            &pending,
            &default_options(4096),
            65_536,
            PromptReuse::Disabled,
            &incoming2,
            &events2,
            |_req, on_text, _| {
                calls2 += 1;
                if calls2 == 1 {
                    on_text(Cow::Borrowed("Complete thought ending cleanly."))?;
                    Ok((
                        assistant_msg("Complete thought ending cleanly.", None),
                        Completion::TokenLimit,
                    ))
                } else {
                    on_text(Cow::Borrowed("ending cleanly."))?;
                    Ok((assistant_msg("ending cleanly.", None), Completion::Finished))
                }
            },
            no_wait,
        )?;
        assert_eq!(calls2, 2);
        assert_eq!(completion2, Completion::Finished);
        assert_eq!(msg2.content.as_ref(), "Complete thought ending cleanly.");
        Ok(())
    }

    #[test]
    fn context_headroom_stepdown_and_rate_limit_retry_succeed() -> Result<()> {
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = event_channel();
        let pending = vec![user_msg("Long task")];
        let mut budgets = Vec::new();
        let mut call = 0_usize;
        let mut waited = 0_u64;

        let (msg, completion) = run_with(
            &pending,
            &default_options(4096),
            8_192,
            PromptReuse::Latest,
            &incoming,
            &events,
            |req, on_text, _| {
                call += 1;
                let json: serde_json::Value =
                    serde_json::from_slice(&req).map_err(|_| Failure::Message("invalid JSON"))?;
                let budget = json["max_completion_tokens"]
                    .as_u64()
                    .ok_or_else(|| Failure::Message("missing max_completion_tokens"))?;
                budgets.push(budget);
                match call {
                    1 => {
                        on_text(Cow::Borrowed("First block."))?;
                        Ok((assistant_msg("First block.", None), Completion::TokenLimit))
                    }
                    2 => Err(Failure::Rejected(CONTEXT_LIMIT_MESSAGE)),
                    3 => Err(Failure::Rejected(RATE_LIMIT_MESSAGE)),
                    _ => {
                        on_text(Cow::Borrowed(" Final block."))?;
                        Ok((assistant_msg(" Final block.", None), Completion::Finished))
                    }
                }
            },
            |_commands, ms| {
                waited += ms;
                Ok(())
            },
        )?;

        assert_eq!(completion, Completion::Finished);
        assert_eq!(budgets, vec![4096, 4096, 2048, 2048]);
        assert_eq!(waited, RATE_LIMIT_WAIT_MS);
        assert_eq!(msg.content.as_ref(), "First block. Final block.");
        Ok(())
    }

    #[test]
    fn recoverable_rejection_on_continuation_commits_stitched_partial() -> Result<()> {
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = event_channel();
        let pending = vec![user_msg("Task")];
        let mut call = 0_usize;
        let (msg, completion) = run_with(
            &pending,
            &default_options(4096),
            65_536,
            PromptReuse::Disabled,
            &incoming,
            &events,
            |_req, on_text, _| {
                call += 1;
                if call == 1 {
                    on_text(Cow::Borrowed("Useful part one."))?;
                    Ok((
                        assistant_msg("Useful part one.", None),
                        Completion::TokenLimit,
                    ))
                } else {
                    Err(Failure::Rejected("Control token rejected."))
                }
            },
            no_wait,
        )?;
        assert_eq!(completion, Completion::TokenLimit);
        assert_eq!(msg.content.as_ref(), "Useful part one.");
        Ok(())
    }

    #[test]
    fn repetition_loop_and_segment_cap_stop_cleanly() -> Result<()> {
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = event_channel();
        let pending = vec![user_msg("Repeat test")];
        let repeated =
            "This exact sentence keeps repeating over and over again without stopping. ".repeat(2);
        let mut calls = 0_usize;
        let (msg, completion) = run_with(
            &pending,
            &default_options(4096),
            131_072,
            PromptReuse::Latest,
            &incoming,
            &events,
            |_req, on_text, _| {
                calls += 1;
                on_text(Cow::Borrowed(&repeated))?;
                Ok((assistant_msg(&repeated, None), Completion::TokenLimit))
            },
            no_wait,
        )?;
        assert_eq!(completion, Completion::TokenLimit);
        assert_eq!(calls, 2);
        // SeamFilter strips the verbatim duplicate and zero-progress guard stops immediately.
        assert_eq!(msg.content.as_ref(), repeated);

        // Divergent prefixes carrying a repeated 120-byte tail trigger is_repetition_loop.
        let (_commands_loop, incoming_loop) = mpsc::channel();
        let (events_loop, _outgoing_loop) = event_channel();
        let tail_phrase = "The same concluding observation repeats verbatim across multiple generated segments without advancing the topic at all.";
        let mut loop_calls = 0_usize;
        let (_msg_loop, completion_loop) = run_with(
            &pending,
            &default_options(4096),
            131_072,
            PromptReuse::Latest,
            &incoming_loop,
            &events_loop,
            |_req, on_text, _| {
                loop_calls += 1;
                let text = format!("Section {loop_calls}: {tail_phrase}");
                on_text(Cow::Borrowed(&text))?;
                Ok((assistant_msg(&text, None), Completion::TokenLimit))
            },
            no_wait,
        )?;
        assert_eq!(completion_loop, Completion::TokenLimit);
        assert_eq!(loop_calls, 3);

        // Segment cap when chunks are distinct
        let (_commands2, incoming2) = mpsc::channel();
        let (events2, outgoing2) = event_channel();
        let mut calls2 = 0_u32;
        let (_msg2, completion2) = run_with(
            &pending,
            &default_options(4096),
            262_144,
            PromptReuse::Latest,
            &incoming2,
            &events2,
            |_req, on_text, _| {
                calls2 += 1;
                let text = format!("Distinct segment number {calls2} with plenty of new content.");
                on_text(Cow::Borrowed(&text))?;
                Ok((assistant_msg(&text, None), Completion::TokenLimit))
            },
            no_wait,
        )?;
        assert_eq!(completion2, Completion::TokenLimit);
        assert_eq!(calls2, MAX_TOTAL_SEGMENTS);
        let (_, _, parts) = collect_streamed(&outgoing2);
        assert_eq!(parts, vec![MAX_TOTAL_SEGMENTS]);
        Ok(())
    }

    #[test]
    fn stop_mid_continuation_propagates_stopped_after_streaming_prior_deltas() {
        let (tx, incoming) = mpsc::channel();
        let (events, outgoing) = event_channel();
        let pending = vec![user_msg("Stop midway")];
        let mut call = 0_usize;
        let result = run_with(
            &pending,
            &default_options(4096),
            65_536,
            PromptReuse::Latest,
            &incoming,
            &events,
            |_req, on_text, _| {
                call += 1;
                if call == 1 {
                    on_text(Cow::Borrowed("First segment streamed."))?;
                    Ok((
                        assistant_msg("First segment streamed.", None),
                        Completion::TokenLimit,
                    ))
                } else {
                    on_text(Cow::Borrowed(" Partial second segment exceeding buffer."))?;
                    let _ = tx.send(Command::Stop);
                    Err(Failure::Stopped)
                }
            },
            no_wait,
        );
        assert!(matches!(result, Err(Failure::Stopped)));
        assert_eq!(call, 2);
        let (streamed, _, parts) = collect_streamed(&outgoing);
        assert_eq!(
            streamed,
            "First segment streamed. Partial second segment exceeding buffer."
        );
        assert_eq!(parts, vec![2]);
    }

    #[test]
    fn prompt_reuse_disabled_compacts_wire_history_and_disables_continuation_thinking() -> Result<()>
    {
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = event_channel();
        let pending = vec![user_msg("Write a long essay")];
        let mut opts = default_options(4096);
        opts.enable_thinking = true;
        opts.cognitive_controls = true;

        let mut seen_requests = Vec::new();
        let mut step = 0_usize;
        let (msg, completion) = run_with(
            &pending,
            &opts,
            65_536,
            PromptReuse::Disabled,
            &incoming,
            &events,
            |req, on_text, on_reasoning| {
                step += 1;
                let json: serde_json::Value =
                    serde_json::from_slice(&req).map_err(|_| Failure::Message("invalid JSON"))?;
                seen_requests.push(json);
                match step {
                    1 => {
                        on_reasoning(Cow::Borrowed("Thought process"))?;
                        on_text(Cow::Borrowed("Alpha ends at boundary."))?;
                        Ok((
                            assistant_msg("Alpha ends at boundary.", Some("Thought process")),
                            Completion::TokenLimit,
                        ))
                    }
                    2 => {
                        on_text(Cow::Borrowed("at boundary. Beta continues cleanly."))?;
                        Ok((
                            assistant_msg("at boundary. Beta continues cleanly.", None),
                            Completion::TokenLimit,
                        ))
                    }
                    3 => {
                        on_text(Cow::Borrowed(" Gamma follows."))?;
                        Ok((
                            assistant_msg(" Gamma follows.", None),
                            Completion::TokenLimit,
                        ))
                    }
                    _ => {
                        on_text(Cow::Borrowed(" Delta finishes."))?;
                        Ok((
                            assistant_msg(" Delta finishes.", None),
                            Completion::Finished,
                        ))
                    }
                }
            },
            no_wait,
        )?;

        assert_eq!(completion, Completion::Finished);
        assert_eq!(seen_requests.len(), 4);
        assert_eq!(
            msg.content.as_ref(),
            "Alpha ends at boundary. Beta continues cleanly. Gamma follows. Delta finishes."
        );
        for req in &seen_requests[1..] {
            // Compaction keeps wire_history at base_len (1) + 2 = 3, plus 1 system message = 4
            assert_eq!(req["messages"].as_array().map(Vec::len), Some(4));
            // Without PromptReuse, thinking is disabled cleanly instead of 1-token handoff
            assert_eq!(req["enable_thinking"], false);
            assert!(req.get("reasoning_budget").is_none());
        }
        // Subsequent compacted requests carry cleaned stitched text without the raw overlap
        assert_eq!(
            seen_requests[2]["messages"][2]["content"],
            "Alpha ends at boundary. Beta continues cleanly."
        );
        assert_eq!(
            seen_requests[2]["messages"][2]["reasoning_content"],
            "Thought process"
        );
        Ok(())
    }

    #[test]
    fn prompt_reuse_latest_compacts_when_approaching_wire_message_limit() -> Result<()> {
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = event_channel();
        // 61 prior turns (122) + 1 current user = 123 messages (+ 1 system = 124 on wire)
        let mut pending = Vec::with_capacity(123);
        for i in 0..61 {
            pending.push(user_msg(&format!("U{i}")));
            pending.push(assistant_msg(&format!("A{i}"), None));
        }
        pending.push(user_msg("Long final prompt"));

        let mut msg_counts = Vec::new();
        let mut step = 0_usize;
        let (msg, completion) = run_with(
            &pending,
            &default_options(4096),
            65_536,
            PromptReuse::Latest,
            &incoming,
            &events,
            |req, on_text, _| {
                step += 1;
                let json: serde_json::Value =
                    serde_json::from_slice(&req).map_err(|_| Failure::Message("invalid JSON"))?;
                msg_counts.push(json["messages"].as_array().map_or(0, Vec::len));
                if step < 4 {
                    let chunk = format!(" Seg{step} unique text.");
                    on_text(Cow::Borrowed(&chunk))?;
                    Ok((assistant_msg(&chunk, None), Completion::TokenLimit))
                } else {
                    on_text(Cow::Borrowed(" Final Seg4."))?;
                    Ok((assistant_msg(" Final Seg4.", None), Completion::Finished))
                }
            },
            no_wait,
        )?;

        assert_eq!(completion, Completion::Finished);
        // Step 1: 124 (123+sys); Step 2: 126; Step 3: 128; Step 4 compacts to 126 (125+sys)!
        assert_eq!(msg_counts, vec![124, 126, 128, 126]);
        assert_eq!(
            msg.content.as_ref(),
            " Seg1 unique text. Seg2 unique text. Seg3 unique text. Final Seg4."
        );
        Ok(())
    }

    #[test]
    fn sticky_budget_ceiling_and_multi_rate_limit_retries() -> Result<()> {
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = event_channel();
        let pending = vec![user_msg("Long task")];
        let mut budgets = Vec::new();
        let mut call = 0_usize;
        let mut waited = 0_u64;

        let (msg, completion) = run_with(
            &pending,
            &default_options(4096),
            131_072,
            PromptReuse::Disabled,
            &incoming,
            &events,
            |req, on_text, _| {
                call += 1;
                let json: serde_json::Value =
                    serde_json::from_slice(&req).map_err(|_| Failure::Message("invalid JSON"))?;
                let budget = json["max_completion_tokens"]
                    .as_u64()
                    .ok_or_else(|| Failure::Message("missing max_completion_tokens"))?;
                budgets.push(budget);
                match call {
                    1 => {
                        on_text(Cow::Borrowed("Part one."))?;
                        Ok((assistant_msg("Part one.", None), Completion::TokenLimit))
                    }
                    // Segment 2 hits 413 on 4096, steps down to 2048
                    2 => Err(Failure::Rejected(CONTEXT_LIMIT_MESSAGE)),
                    3 => {
                        on_text(Cow::Borrowed(" Part two."))?;
                        Ok((assistant_msg(" Part two.", None), Completion::TokenLimit))
                    }
                    // Segment 3 starts at sticky ceiling 2048 (no 4096 retry!) and survives 2x 429
                    4 | 5 => Err(Failure::Rejected(RATE_LIMIT_MESSAGE)),
                    _ => {
                        on_text(Cow::Borrowed(" Part three."))?;
                        Ok((assistant_msg(" Part three.", None), Completion::Finished))
                    }
                }
            },
            |_commands, ms| {
                waited += ms;
                Ok(())
            },
        )?;

        assert_eq!(completion, Completion::Finished);
        assert_eq!(budgets, vec![4096, 4096, 2048, 2048, 2048, 2048]);
        assert_eq!(waited, RATE_LIMIT_WAIT_MS * 2);
        assert_eq!(msg.content.as_ref(), "Part one. Part two. Part three.");
        Ok(())
    }

    #[test]
    fn long_tail_and_intra_chunk_degeneration_stop_cleanly() -> Result<()> {
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = event_channel();
        let pending = vec![user_msg("Long repeat")];
        let shared_long = "A".repeat(140) + &"B".repeat(140);
        let mut calls = 0_usize;
        let (_msg, comp) = run_with(
            &pending,
            &default_options(4096),
            131_072,
            PromptReuse::Disabled,
            &incoming,
            &events,
            |_req, on_text, _| {
                calls += 1;
                let text = if calls == 1 {
                    format!("Intro prose. {shared_long} Middle transition.")
                } else {
                    format!("New bridge. {shared_long}")
                };
                on_text(Cow::Borrowed(&text))?;
                Ok((assistant_msg(&text, None), Completion::TokenLimit))
            },
            no_wait,
        )?;
        assert_eq!(comp, Completion::TokenLimit);
        // Stopped immediately after segment 2 due to 256-byte long-tail single repeat!
        assert_eq!(calls, 2);

        // Intra-chunk degeneration loop stops immediately on segment 2 without issuing segment 3
        let (_commands2, incoming2) = mpsc::channel();
        let (events2, _outgoing2) = event_channel();
        let mut calls2 = 0_usize;
        let degen_unit = "The system loops endlessly on the exact same clause without progress. ";
        let degen_chunk = degen_unit.repeat(10);
        let (_msg2, comp2) = run_with(
            &pending,
            &default_options(4096),
            131_072,
            PromptReuse::Disabled,
            &incoming2,
            &events2,
            |_req, on_text, _| {
                calls2 += 1;
                if calls2 == 1 {
                    on_text(Cow::Borrowed("Normal opening."))?;
                    Ok((
                        assistant_msg("Normal opening.", None),
                        Completion::TokenLimit,
                    ))
                } else {
                    on_text(Cow::Borrowed(&degen_chunk))?;
                    Ok((assistant_msg(&degen_chunk, None), Completion::TokenLimit))
                }
            },
            no_wait,
        )?;
        assert_eq!(comp2, Completion::TokenLimit);
        assert_eq!(calls2, 2);
        Ok(())
    }

    #[test]
    fn fence_parsing_prompt_and_control_marker_edge_cases() -> Result<()> {
        // Tilde fences, 4-backtick outer fences, info-string backticks, 4-space indent, CRLF, trailing tokens
        let tilde = open_fence_state("Intro\r\n~~~~python extra\r\ncode\r\n");
        assert_eq!(
            tilde.as_ref().map(|f| (f.ch, f.len, f.info.as_str())),
            Some(('~', 4, "python extra"))
        );
        // 3 tildes cannot close a 4-tilde fence
        assert!(open_fence_state("~~~~python\ncode\n~~~\nmore").is_some());
        // 4-tilde fence closes it
        assert!(open_fence_state("~~~~python\ncode\n~~~~  \t\r\n").is_none());
        // Nested ``` inside ````markdown does not close the outer fence
        let outer = open_fence_state("````markdown\n```rust\nfn x() {}\n```\nstill inside");
        assert_eq!(
            outer.as_ref().map(|f| (f.ch, f.len, f.info.as_str())),
            Some(('`', 4, "markdown"))
        );
        // Closing fence with non-whitespace trailing text does not close
        assert!(open_fence_state("```rust\ncode\n```rust\nmore").is_some());
        // Inline backtick in info string never opens a fence; 4-space indent never opens a fence
        assert!(open_fence_state("```code``` inline").is_none());
        assert!(open_fence_state("    ```rust\n    code").is_none());

        // Non-SVG open fence prompt includes display_tag; whitespace-only text gives THINKING_HANDOFF_PROMPT
        let non_svg_prompt = build_continuation_prompt("fn x() {", outer.as_ref());
        assert!(non_svg_prompt.contains("````markdown"));
        assert_eq!(
            build_continuation_prompt("   \n\t", outer.as_ref()),
            THINKING_HANDOFF_PROMPT
        );

        // Control marker checks (markers + C0 bytes vs allowed whitespace)
        for bad in [
            "<|im_end|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tts_pad>",
            "nul\0byte",
            "bell\x07",
            "esc\x1b",
        ] {
            assert!(contains_control_marker(bad), "expected marker in {bad:?}");
        }
        assert!(!contains_control_marker("normal\ttext\r\nnext line"));
        assert!(message_has_control_marker(&assistant_msg(
            "clean",
            Some("hidden </think> leak")
        )));

        // Control marker or empty/whitespace on segment 1 stops immediately in 1 call
        for seg1 in [
            assistant_msg("clean", Some("<|endoftext|>")),
            assistant_msg("prefix <tool_call>x", None),
            assistant_msg("", None),
            assistant_msg("   \n\t", Some("  \n")),
        ] {
            let (_commands, incoming) = mpsc::channel();
            let (events, _outgoing) = event_channel();
            let pending = vec![user_msg("Hi")];
            let mut calls = 0_usize;
            let (_msg, comp) = run_with(
                &pending,
                &default_options(4096),
                65_536,
                PromptReuse::Latest,
                &incoming,
                &events,
                |_req, _on_text, _| {
                    calls += 1;
                    Ok((seg1.clone(), Completion::TokenLimit))
                },
                no_wait,
            )?;
            assert_eq!(calls, 1);
            assert_eq!(comp, Completion::TokenLimit);
        }
        Ok(())
    }

    #[test]
    fn seam_filter_preamble_fence_overlap_and_block_separator_branches() -> Result<()> {
        fn run_chunks(prev: &str, open_doc: Option<&str>, chunks: &[&str]) -> Result<String> {
            let fence = open_doc.and_then(open_fence_state);
            let mut out = String::new();
            let mut filter = SeamFilter::new(prev.into(), fence);
            for c in chunks {
                filter.push(c, &mut |s| {
                    out.push_str(s);
                    Ok(())
                })?;
            }
            filter.finish(&mut |s| {
                out.push_str(s);
                Ok(())
            })?;
            Ok(out)
        }

        // Preamble boundaries: >80 byte line preserved, Part 12 stripped, Part 123 / Part 2a preserved
        assert!(!is_meta_preamble_line(&"a".repeat(81)));
        assert!(is_meta_preamble_line("Part 12:"));
        assert!(!is_meta_preamble_line("Part 123:"));
        assert!(!is_meta_preamble_line("Part 2a:"));

        // Token-by-token streaming of polite preamble vs genuine polite prose
        assert_eq!(
            run_chunks(
                "Sentence one.",
                None,
                &[
                    "Sure, ",
                    "here's ",
                    "the continuation:\r\n\r\n",
                    "Next para."
                ]
            )?,
            "\n\nNext para."
        );
        assert_eq!(
            run_chunks(
                "Sentence one.",
                None,
                &["Sure, ", "I think ", "this works well in practice."]
            )?,
            "Sure, I think this works well in practice."
        );
        // Single-newline preamble or non-sentence-ending prev_tail does NOT insert \n\n
        assert_eq!(
            run_chunks("Sentence one.", None, &["Continuing:\nNext line."])?,
            "Next line."
        );
        assert_eq!(
            run_chunks("working on", None, &["Continuing:\n\nnext step."])?,
            "next step."
        );

        // Redundant fence branches: leading newlines, bare fence with overlap, mismatched lang/char,
        // stripped fence without overlap adds \n, incoming without newline
        assert_eq!(
            run_chunks(
                "fn a() {\n",
                Some("```rust\nfn a() {\n"),
                &["\r\n```rust\n    x();"]
            )?,
            "    x();"
        );
        assert_eq!(
            run_chunks(
                "let v = 1;",
                Some("```\nlet v = 1;"),
                &["```\nlet v = 1;\nlet w = 2;"]
            )?,
            "\nlet w = 2;"
        );
        assert_eq!(
            run_chunks(
                "fn a() {",
                Some("```rust\nfn a() {"),
                &["```python\nprint(1)"]
            )?,
            "```python\nprint(1)"
        );
        assert_eq!(
            run_chunks(
                "fn a() {",
                Some("```rust\nfn a() {"),
                &["~~~rust\nfn b() {}"]
            )?,
            "~~~rust\nfn b() {}"
        );
        assert_eq!(
            run_chunks(
                "fn a() {",
                Some("```rust\nfn a() {"),
                &["```rust\n    y();"]
            )?,
            "\n    y();"
        );
        assert_eq!(
            run_chunks("fn a() {", Some("```rust\nfn a() {"), &["no_newline_yet"])?,
            "no_newline_yet"
        );

        // Overlap & ellipsis rules
        // >= 12 bytes mid-word overlap matches
        assert_eq!(
            run_chunks("prefix_abcdefghijklm", None, &["abcdefghijklm_next"])?,
            "_next"
        );
        // 4-5 char standalone word NOT stripped; 6-11 char non-boundary NOT stripped
        assert_eq!(run_chunks("that ", None, &["that stays"])?, "that stays");
        assert_eq!(run_chunks("foobar", None, &["bar baz"])?, "bar baz");
        // 1..=8 leading spaces before overlap stripped; 9+ preserved
        assert_eq!(
            run_chunks("alpha beta gamma", None, &["        beta gamma delta"])?,
            " delta"
        );
        assert_eq!(
            run_chunks("alpha beta gamma", None, &["         beta gamma delta"])?,
            "         beta gamma delta"
        );
        // prev_tail ending in \n or \t + combined trailing/leading spaces
        assert_eq!(
            run_chunks("alpha beta\n", None, &["alpha beta gamma"])?,
            " gamma"
        );
        assert_eq!(
            run_chunks("alpha beta\t", None, &["   alpha beta gamma"])?,
            "gamma"
        );
        // Ellipsis preserved after '.' or '…'
        assert_eq!(run_chunks("Wait.", None, &["...more"])?, "...more");
        assert_eq!(run_chunks("Wait…", None, &["…more"])?, "…more");
        assert_eq!(run_chunks("Wait ", None, &["... next"])?, "next");

        // Block separators
        assert_eq!(
            run_chunks("Intro.", None, &["```rust\nfn x() {}"])?,
            "\n\n```rust\nfn x() {}"
        );
        assert_eq!(
            run_chunks("Intro.", None, &["---\nSection"])?,
            "\n\n---\nSection"
        );
        assert_eq!(run_chunks("Intro.", None, &["---"])?, "---");
        assert_eq!(run_chunks("Intro.", None, &["> Quote"])?, "\n> Quote");
        assert_eq!(run_chunks("Intro.", None, &["* Bullet"])?, "\n* Bullet");
        assert_eq!(run_chunks("Intro.", None, &["+ Plus"])?, "\n+ Plus");
        assert_eq!(run_chunks("Intro.", None, &["1. First"])?, "\n1. First");
        assert_eq!(
            run_chunks("Intro.", None, &["12) Twelfth"])?,
            "\n12) Twelfth"
        );
        assert_eq!(
            run_chunks("Intro.", None, &["####### TooDeep"])?,
            "####### TooDeep"
        );
        assert_eq!(run_chunks("Intro.", None, &["#NoSpace"])?, "#NoSpace");
        assert_eq!(
            run_chunks("Intro.", None, &["\nAlready newline"])?,
            "\nAlready newline"
        );
        Ok(())
    }

    #[test]
    fn seam_filter_hold_timing_max_hold_and_stream_error_propagation() -> Result<()> {
        // Early-flush timing: <24 bytes held, >=24 non-overlap flushed before finish,
        // >=24 matching prev_tail held until divergence, post-resolution streams immediately.
        let mut emitted = Vec::new();
        let mut filter = SeamFilter::new("alpha beta gamma delta epsilon".into(), None);
        filter.push("short", &mut |s| {
            emitted.push(s.to_owned());
            Ok(())
        })?;
        assert!(emitted.is_empty());

        let mut f_overlap = SeamFilter::new("alpha beta gamma delta epsilon".into(), None);
        let mut overlap_emitted = Vec::new();
        f_overlap.push("alpha beta gamma delta eps", &mut |s| {
            overlap_emitted.push(s.to_owned());
            Ok(())
        })?;
        assert!(overlap_emitted.is_empty());
        f_overlap.push("ilon zeta eta theta iota kappa!", &mut |s| {
            overlap_emitted.push(s.to_owned());
            Ok(())
        })?;
        assert_eq!(overlap_emitted.concat(), " zeta eta theta iota kappa!");
        // Post-resolution push emits immediately without buffering
        f_overlap.push("!", &mut |s| {
            overlap_emitted.push(s.to_owned());
            Ok(())
        })?;
        assert_eq!(overlap_emitted.last().map(String::as_str), Some("!"));

        // Chunked preamble + open fence hold and blank line + fence hold
        let open_rust = open_fence_state("```rust\nfn main() {\n");
        let mut f_pre = SeamFilter::new("fn main() {\n".into(), open_rust.clone());
        let mut pre_out = String::new();
        for c in ["Part 2:\n", "```", "rust\n", "    let x = 1;"] {
            f_pre.push(c, &mut |s| {
                pre_out.push_str(s);
                Ok(())
            })?;
        }
        f_pre.finish(&mut |s| {
            pre_out.push_str(s);
            Ok(())
        })?;
        assert_eq!(pre_out, "    let x = 1;");

        let mut f_blank = SeamFilter::new("fn main() {\n".into(), open_rust);
        let mut blank_out = String::new();
        for c in ["\n", "```rust", "\n    let y = 2;"] {
            f_blank.push(c, &mut |s| {
                blank_out.push_str(s);
                Ok(())
            })?;
        }
        f_blank.finish(&mut |s| {
            blank_out.push_str(s);
            Ok(())
        })?;
        assert_eq!(blank_out, "    let y = 2;");

        // SEAM_HOLD_MAX_BYTES forces flush even when buffer stays a substring of prev_tail
        let tail = "x".repeat(SEAM_HOLD_MAX_BYTES);
        let mut f_max = SeamFilter::new(tail, None);
        let mut max_out = Vec::new();
        f_max.push(&"x".repeat(SEAM_HOLD_MAX_BYTES - 1), &mut |s| {
            max_out.push(s.to_owned());
            Ok(())
        })?;
        assert!(!f_max.resolved);
        f_max.push("Z", &mut |s| {
            max_out.push(s.to_owned());
            Ok(())
        })?;
        assert!(f_max.resolved);
        assert_eq!(max_out, vec!["Z"]);

        // Emit error during SeamFilter::push and SeamFilter::finish is captured in stream_err and propagated
        let pending = vec![user_msg("Trigger receiver close")];
        for flush_in_finish in [false, true] {
            let (_cmds, inc) = mpsc::channel();
            let (ev, out) = event_channel();
            let mut out_opt = Some(out);
            let mut call = 0_usize;
            let err_res = run_with(
                &pending,
                &default_options(4096),
                65_536,
                PromptReuse::Disabled,
                &inc,
                &ev,
                |_req, on_text, _| {
                    call += 1;
                    if call == 1 {
                        on_text(Cow::Borrowed("Initial segment."))?;
                        Ok((
                            assistant_msg("Initial segment.", None),
                            Completion::TokenLimit,
                        ))
                    } else {
                        drop(out_opt.take());
                        let delta = if flush_in_finish {
                            "tiny"
                        } else {
                            "Enough non-overlapping text to flush immediately!"
                        };
                        on_text(Cow::Borrowed(delta))?;
                        Ok((assistant_msg(delta, None), Completion::Finished))
                    }
                },
                no_wait,
            );
            assert_eq!(err_res.err(), Some(Failure::Shutdown));
        }
        Ok(())
    }

    #[test]
    fn repetition_loop_direct_window_and_long_chunk_verbatim_repeat() -> Result<()> {
        // Direct is_repetition_loop branches
        assert!(is_repetition_loop(
            "prior",
            "  Exact chunk >= 32 bytes repeating.  ",
            "Exact chunk >= 32 bytes repeating."
        ));
        assert!(!is_repetition_loop(
            &" ".repeat(600),
            "short",
            &" ".repeat(600)
        ));
        // Legitimate repeat outside the 16 KiB window is NOT flagged
        let needle =
            "A distinct factual summary sentence that appeared much earlier in the document. "
                .repeat(2);
        let old_prior = format!(
            "{needle}{}{needle}{}",
            "x".repeat(REPETITION_WINDOW_BYTES + 512),
            "y".repeat(256)
        );
        assert!(!is_repetition_loop(&old_prior, "prev", &needle));

        // >256-byte repeated chunk bypasses 256-byte SeamFilter and hits prev_trimmed == new_trimmed
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = event_channel();
        let pending = vec![user_msg("Long repeat")];
        let long_chunk = format!(
            "{} END_MARKER_UNIQUE_1234567890",
            "Alpha beta gamma delta epsilon zeta. ".repeat(8)
        );
        assert!(long_chunk.len() > SEAM_TAIL_BYTES);
        let mut calls = 0_usize;
        let (_msg, comp) = run_with(
            &pending,
            &default_options(4096),
            131_072,
            PromptReuse::Latest,
            &incoming,
            &events,
            |_req, on_text, _| {
                calls += 1;
                on_text(Cow::Borrowed(&long_chunk))?;
                Ok((assistant_msg(&long_chunk, None), Completion::TokenLimit))
            },
            no_wait,
        )?;
        assert_eq!(comp, Completion::TokenLimit);
        assert_eq!(calls, 2);
        Ok(())
    }

    #[test]
    fn continuation_budgets_caps_compaction_retries_and_cancellations() -> Result<()> {
        // 1. next_chunk_budget returns None when headroom < 64 or base_cap < 64 or system_appendix fills context
        assert_eq!(next_chunk_budget(&[user_msg("hi")], None, 1024, 32), None);
        let big_appendix = "word ".repeat(2000);
        assert_eq!(
            next_chunk_budget(&[user_msg("hi")], Some(&big_appendix), 1024, 512),
            None
        );

        // 2. MAX_STITCHED_BYTES cap stops cleanly without calling segment 2
        let (_cmds_cap, inc_cap) = mpsc::channel();
        let (ev_cap, _out_cap) = event_channel();
        let huge_reasoning = "r".repeat(MAX_STITCHED_BYTES - 5);
        let mut cap_calls = 0_usize;
        let (_msg_cap, comp_cap) = continue_after_token_limit(
            &[user_msg("Hi")],
            assistant_msg("short", Some(&huge_reasoning)),
            &default_options(4096),
            262_144,
            PromptReuse::Disabled,
            &inc_cap,
            &ev_cap,
            |_req, on_text, _| {
                cap_calls += 1;
                on_text(Cow::Borrowed("more"))?;
                Ok((assistant_msg("more", None), Completion::TokenLimit))
            },
            no_wait,
        )?;
        assert_eq!(comp_cap, Completion::TokenLimit);
        assert_eq!(cap_calls, 0);

        // 3. base_wire_history >= MAX_WIRE_HISTORY_MESSAGES stops cleanly in both Latest & Disabled
        let mut deep_pending = Vec::with_capacity(MAX_WIRE_HISTORY_MESSAGES);
        for i in 0..MAX_WIRE_HISTORY_MESSAGES {
            deep_pending.push(user_msg(&format!("U{i}")));
        }
        for reuse in [PromptReuse::Latest, PromptReuse::Disabled] {
            let (_c, inc) = mpsc::channel();
            let (ev, out) = event_channel();
            let mut calls = 0_usize;
            let (_m, comp) = continue_after_token_limit(
                &deep_pending,
                assistant_msg("Seg1.", None),
                &default_options(4096),
                131_072,
                reuse,
                &inc,
                &ev,
                |_req, _, _| {
                    calls += 1;
                    Ok((assistant_msg("unused", None), Completion::Finished))
                },
                no_wait,
            )?;
            assert_eq!(comp, Completion::TokenLimit);
            assert_eq!(calls, 0);
            assert_eq!(collect_streamed(&out).2, Vec::<u32>::new());
        }

        // 4. Serialize-failure fallback compaction under PromptReuse::Latest when wire_history > MAX_REQUEST (1 MiB)
        //    seg1 + seg2 fit individually (520 KiB each), but uncompacted wire_history (1040 KiB) > 1 MiB;
        //    compaction drops duplicate raw reasoning so Step 3 succeeds with 4 messages!
        let (_c_ser, inc_ser) = mpsc::channel();
        let (ev_ser, _out_ser) = event_channel();
        let big_reasoning = "t".repeat(520 * 1024);
        let mut ser_calls = 0_usize;
        let mut msg_counts = Vec::new();
        let (msg_ser, comp_ser) = run_with(
            &[user_msg("Compact me")],
            &default_options(4096),
            1_048_576,
            PromptReuse::Latest,
            &inc_ser,
            &ev_ser,
            |req, on_text, on_reasoning| {
                ser_calls += 1;
                let json: serde_json::Value =
                    serde_json::from_slice(&req).map_err(|_| Failure::Message("invalid JSON"))?;
                msg_counts.push(json["messages"].as_array().map_or(0, Vec::len));
                match ser_calls {
                    1 => {
                        on_reasoning(Cow::Borrowed(&big_reasoning))?;
                        on_text(Cow::Borrowed("Part one."))?;
                        Ok((
                            assistant_msg("Part one.", Some(&big_reasoning)),
                            Completion::TokenLimit,
                        ))
                    }
                    2 => {
                        on_text(Cow::Borrowed(" Part two."))?;
                        Ok((
                            assistant_msg(" Part two.", Some(&big_reasoning)),
                            Completion::TokenLimit,
                        ))
                    }
                    _ => {
                        on_text(Cow::Borrowed(" Part three."))?;
                        Ok((assistant_msg(" Part three.", None), Completion::Finished))
                    }
                }
            },
            no_wait,
        )?;
        assert_eq!(comp_ser, Completion::Finished);
        assert_eq!(ser_calls, 3);
        // Step 1: 2 msgs, Step 2: 4 msgs, Step 3 exceeded 512 KiB with 6 msgs -> compacted to 4 msgs!
        assert_eq!(msg_counts, vec![2, 4, 4]);
        assert_eq!(msg_ser.content.as_ref(), "Part one. Part two. Part three.");

        // 5. Exhausting MAX_CONTEXT_STEPDOWNS (6) and budget / 2 < MIN_CONTINUATION_TOKENS (64)
        let (_c_exh, inc_exh) = mpsc::channel();
        let (ev_exh, _out_exh) = event_channel();
        let mut c413 = 0_usize;
        let (_m413, comp413) = continue_after_token_limit(
            &[user_msg("413")],
            assistant_msg("Part 1.", None),
            &default_options(4096),
            131_072,
            PromptReuse::Disabled,
            &inc_exh,
            &ev_exh,
            |_req, _, _| {
                c413 += 1;
                Err(Failure::Rejected(CONTEXT_LIMIT_MESSAGE))
            },
            no_wait,
        )?;
        assert_eq!(comp413, Completion::TokenLimit);
        assert_eq!(c413, MAX_CONTEXT_STEPDOWNS + 1);

        let (_c_min, inc_min) = mpsc::channel();
        let (ev_min, _out_min) = event_channel();
        let mut c96 = 0_usize;
        let (_m96, comp96) = continue_after_token_limit(
            &[user_msg("413-min")],
            assistant_msg("Part 1.", None),
            &default_options(96),
            131_072,
            PromptReuse::Disabled,
            &inc_min,
            &ev_min,
            |_req, _, _| {
                c96 += 1;
                Err(Failure::Rejected(CONTEXT_LIMIT_MESSAGE))
            },
            no_wait,
        )?;
        assert_eq!(comp96, Completion::TokenLimit);
        assert_eq!(c96, 1);

        let (_c_429, inc_429) = mpsc::channel();
        let (ev_429, _out_429) = event_channel();
        let mut c429 = 0_usize;
        let mut waits429 = 0_usize;
        let (_m429, comp429) = continue_after_token_limit(
            &[user_msg("429")],
            assistant_msg("Part 1.", None),
            &default_options(4096),
            131_072,
            PromptReuse::Disabled,
            &inc_429,
            &ev_429,
            |_req, _, _| {
                c429 += 1;
                Err(Failure::Rejected(RATE_LIMIT_MESSAGE))
            },
            |_cmds, _ms| {
                waits429 += 1;
                Ok(())
            },
        )?;
        assert_eq!(comp429, Completion::TokenLimit);
        assert_eq!(c429, MAX_RATE_LIMIT_RETRIES + 1);
        assert_eq!(waits429, MAX_RATE_LIMIT_RETRIES);

        // 6. sleep_interruptible wakes promptly on Stop / StopModel / Shutdown
        for (cmd, expected) in [
            (Command::Stop, Failure::Stopped),
            (Command::StopModel, Failure::ModelStopped),
            (Command::Shutdown, Failure::Shutdown),
        ] {
            let (tx, rx) = mpsc::channel();
            tx.send(cmd)
                .map_err(|_| Failure::Message("channel send failed"))?;
            let started = std::time::Instant::now();
            assert_eq!(sleep_interruptible(&rx, 5_000), Err(expected));
            assert!(started.elapsed() < Duration::from_millis(500));
        }

        // 7. Fatal errors on segment 2 propagate Err instead of committing partial
        for fatal in [
            Failure::Message("boom"),
            Failure::ModelStopped,
            Failure::Shutdown,
        ] {
            let (_c, inc) = mpsc::channel();
            let (ev, _out) = event_channel();
            let err = continue_after_token_limit(
                &[user_msg("fatal")],
                assistant_msg("Part 1.", None),
                &default_options(4096),
                65_536,
                PromptReuse::Disabled,
                &inc,
                &ev,
                |_req, _, _| Err(fatal),
                no_wait,
            );
            assert_eq!(err.err(), Some(fatal));
        }

        // 8. Pre-segment Command::Stop aborts before Event::Continuing(2) or request 2
        let (tx_pre, rx_pre) = mpsc::channel();
        let (ev_pre, out_pre) = event_channel();
        tx_pre
            .send(Command::Stop)
            .map_err(|_| Failure::Message("send failed"))?;
        let mut pre_calls = 0_usize;
        let pre_res = continue_after_token_limit(
            &[user_msg("pre-stop")],
            assistant_msg("Part 1.", None),
            &default_options(4096),
            65_536,
            PromptReuse::Disabled,
            &rx_pre,
            &ev_pre,
            |_req, _, _| {
                pre_calls += 1;
                Ok((assistant_msg("unused", None), Completion::Finished))
            },
            no_wait,
        );
        assert_eq!(pre_res.err(), Some(Failure::Stopped));
        assert_eq!(pre_calls, 0);
        assert!(collect_streamed(&out_pre).2.is_empty());

        // 9. Thinking-only seg 1 + empty seg 2 Finished stays Completion::TokenLimit
        let (_c_th, inc_th) = mpsc::channel();
        let (ev_th, _out_th) = event_channel();
        let (msg_th, comp_th) = continue_after_token_limit(
            &[user_msg("think")],
            assistant_msg("", Some("reasoning only")),
            &default_options(4096),
            65_536,
            PromptReuse::Disabled,
            &inc_th,
            &ev_th,
            |_req, _, _| Ok((assistant_msg("", None), Completion::Finished)),
            no_wait,
        )?;
        assert_eq!(comp_th, Completion::TokenLimit);
        assert_eq!(msg_th.reasoning_content.as_deref(), Some("reasoning only"));

        // 10. fold_turn_appendix_for_wire preserves images & avoids duplicating turn_appendix under Disabled
        let mut user_with_img = user_msg("Look at this");
        user_with_img.images = Arc::from([Arc::from("data:image/png;base64,AA==")]);
        let mut opts_app = default_options(4096);
        opts_app.turn_appendix = Some(Arc::from("[Attached note]"));
        let (_c_app, inc_app) = mpsc::channel();
        let (ev_app, _out_app) = event_channel();
        let mut seen_app = Vec::new();
        let (_m_app, comp_app) = continue_after_token_limit(
            &[user_with_img],
            assistant_msg("First.", None),
            &opts_app,
            65_536,
            PromptReuse::Disabled,
            &inc_app,
            &ev_app,
            |req, on_text, _| {
                let json: serde_json::Value =
                    serde_json::from_slice(&req).map_err(|_| Failure::Message("invalid JSON"))?;
                seen_app.push(json);
                on_text(Cow::Borrowed(" Second."))?;
                Ok((assistant_msg(" Second.", None), Completion::Finished))
            },
            no_wait,
        )?;
        assert_eq!(comp_app, Completion::Finished);
        let u_msg = &seen_app[0]["messages"][1];
        assert!(
            u_msg["content"]
                .as_array()
                .is_some_and(|parts| parts.iter().any(|p| p["type"] == "image_url"))
        );
        assert!(
            !seen_app[0]["messages"][3]["content"]
                .as_str()
                .is_some_and(|s| s.contains("[Attached note]"))
        );
        Ok(())
    }
}
