//! Local conversation filtering. Cached matches never retain a second transcript.
use super::{Transcript, egui, workspace::Conversation};
use std::sync::Arc;

pub(super) fn field_id() -> egui::Id {
    egui::Id::unique("conversation-history-query")
}

#[derive(Default)]
pub(super) struct Search {
    pub(super) text: String,
    previous: String,
    query: Arc<Query>,
    pub(super) rows: Vec<usize>,
    unfiltered_count: Option<usize>,
    reset_scroll: bool,
}

impl Search {
    pub(super) fn clear(&mut self) {
        self.text.clear();
    }

    pub(super) fn clear_on_escape(&mut self, ctx: &egui::Context) -> bool {
        // egui's input prepass can release focus before the application sees
        // Escape. Only reclaim that release, never another control's focus.
        let owns_escape = ctx.memory(|memory| {
            memory.has_focus(field_id())
                || (memory.focused().is_none() && memory.had_focus_last_frame(field_id()))
        });
        if !self.text.is_empty()
            && owns_escape
            && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
        {
            self.clear();
            ctx.memory_mut(|memory| memory.request_focus(field_id()));
            return true;
        }
        false
    }

    pub(super) fn filtered(&self) -> bool {
        !self.query.text.is_empty()
    }

    pub(super) fn take_scroll_reset(&mut self) -> bool {
        std::mem::take(&mut self.reset_scroll)
    }

    pub(super) fn update(
        &mut self,
        conversations: &mut [Conversation],
        active: u64,
        transcript: &Transcript,
    ) -> bool {
        let changed = self.previous != self.text;
        if changed {
            self.text.clone_into(&mut self.previous);
            // Case or surrounding whitespace can change Find's displayed query
            // without changing which conversations match. Keep the shared query
            // identity so each conversation can reuse its cached body scan.
            if !self
                .query
                .text
                .chars()
                .eq(self.text.trim().chars().flat_map(char::to_lowercase))
            {
                self.query = Arc::new(Query::new(lowercase(self.text.trim())));
            }
            self.reset_scroll = true;
        }
        if self.query.text.is_empty() {
            if !changed && self.unfiltered_count == Some(conversations.len()) {
                return false;
            }
            self.rows.clear();
            self.rows.extend(0..conversations.len());
            self.unfiltered_count = Some(conversations.len());
            return changed;
        }
        self.unfiltered_count = None;
        self.rows.clear();
        for (index, conversation) in conversations.iter_mut().enumerate() {
            let transcript = if conversation.id == active {
                Some(transcript)
            } else {
                conversation.saved.as_ref().map(|saved| &saved.transcript)
            };
            if conversation
                .search
                .matches(&self.query, &conversation.title, transcript)
            {
                self.rows.push(index);
            }
        }
        changed
    }
}

#[derive(Default)]
pub(super) struct MatchCache {
    query: Option<Arc<Query>>,
    title: String,
    revision: Option<u64>,
    structural_revision: Option<u64>,
    scanned_last_body_len: usize,
    result: Option<bool>,
    #[cfg(test)]
    pub(super) scans: usize,
}

impl MatchCache {
    fn matches(
        &mut self,
        query: &Arc<Query>,
        title: &str,
        transcript: Option<&Transcript>,
    ) -> bool {
        let revision = transcript.map(Transcript::history_revision);
        let structural_revision = transcript.map(Transcript::structural_revision);
        let query_same = self
            .query
            .as_ref()
            .is_some_and(|cached| Arc::ptr_eq(cached, query));
        let title_same = self.title == title;
        if query_same
            && title_same
            && self.revision == revision
            && let Some(result) = self.result
        {
            return result;
        }
        if query_same
            && title_same
            && let Some(transcript) = transcript
            && self.structural_revision == structural_revision
            && let Some(prior_result) = self.result
            && let Some(last) = transcript.messages.last()
            && last.body.len() >= self.scanned_last_body_len
            && last.body.is_char_boundary(self.scanned_last_body_len)
        {
            if prior_result {
                self.revision = revision;
                self.scanned_last_body_len = last.body.len();
                return true;
            }
            #[cfg(test)]
            {
                self.scans += 1;
            }
            let tail_start = tail_scan_start(&last.body, self.scanned_last_body_len, &query.text);
            let matched = query.contains(&last.body[tail_start..]);
            self.revision = revision;
            self.scanned_last_body_len = last.body.len();
            self.result = Some(matched);
            return matched;
        }
        #[cfg(test)]
        {
            self.scans += 1;
        }
        let result = query.contains(title)
            || transcript.is_some_and(|transcript| {
                transcript
                    .messages
                    .iter()
                    .any(|message| query.contains(&message.body))
            });
        self.query = Some(Arc::clone(query));
        title.clone_into(&mut self.title);
        self.revision = revision;
        self.structural_revision = structural_revision;
        self.scanned_last_body_len = transcript
            .and_then(|transcript| transcript.messages.last())
            .map_or(0, |last| last.body.len());
        self.result = Some(result);
        result
    }
}

fn tail_scan_start(body: &str, scanned_len: usize, folded_query: &str) -> usize {
    let overlap_chars = folded_query.chars().count().saturating_sub(1);
    if overlap_chars == 0 || scanned_len == 0 {
        return scanned_len;
    }
    body[..scanned_len]
        .char_indices()
        .rev()
        .nth(overlap_chars.saturating_sub(1))
        .map_or(0, |(index, _)| index)
}

// Query-sized KMP state avoids making a lowercase copy of every retained body.
// Compile the fallback table once and reuse it across all conversations. Unlike
// Find, a boolean match needs neither source-offset mapping nor retained ranges.
#[derive(Default)]
struct Query {
    text: String,
    fallback: Vec<usize>,
}

impl Query {
    fn new(text: String) -> Self {
        let bytes = text.as_bytes();
        let mut fallback = vec![0; bytes.len()];
        let mut matched = 0;
        for index in 1..bytes.len() {
            while matched > 0 && bytes[index] != bytes[matched] {
                matched = fallback[matched - 1];
            }
            if bytes[index] == bytes[matched] {
                matched += 1;
            }
            fallback[index] = matched;
        }
        Self { text, fallback }
    }

    fn contains(&self, text: &str) -> bool {
        let query = self.text.as_bytes();
        if query.is_empty() {
            return true;
        }
        if text.is_ascii() {
            if !self.text.is_ascii() {
                return false;
            }
            let bytes = text.as_bytes();
            if query.len() > bytes.len() {
                return false;
            }
            let first_lower = query[0];
            let first_upper = if first_lower.is_ascii_alphabetic() {
                first_lower.to_ascii_uppercase()
            } else {
                first_lower
            };
            let mut matched = 0;
            let mut index = 0;
            while index < bytes.len() {
                if matched == 0 {
                    let Some(found) = memchr::memchr2(first_lower, first_upper, &bytes[index..])
                    else {
                        return false;
                    };
                    index += found;
                    if bytes.len() - index < query.len() {
                        return false;
                    }
                }
                let byte = bytes[index].to_ascii_lowercase();
                while matched > 0 && byte != query[matched] {
                    matched = self.fallback[matched - 1];
                }
                if byte == query[matched] {
                    matched += 1;
                    if matched == query.len() {
                        return true;
                    }
                }
                index += 1;
            }
            return false;
        }
        self.contains_chars(text.chars())
    }

    fn contains_chars(&self, text: impl Iterator<Item = char>) -> bool {
        let query = self.text.as_bytes();
        if query.is_empty() {
            return true;
        }
        let mut matched = 0;
        for character in text.flat_map(char::to_lowercase) {
            let mut encoded = [0; 4];
            for &byte in character.encode_utf8(&mut encoded).as_bytes() {
                while matched > 0 && byte != query[matched] {
                    matched = self.fallback[matched - 1];
                }
                if byte == query[matched] {
                    matched += 1;
                }
                if matched == query.len() {
                    return true;
                }
            }
        }
        false
    }
}

// Use the same scalar folding as transcript Find, including Greek sigma and
// lowercase expansions, so sidebar body results have identical highlights.
fn lowercase(text: &str) -> String {
    text.chars().flat_map(char::to_lowercase).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn history_query_repeated_prefix_fallback_matches_long_unicode_reference() {
        for block in ["ab", "İΣ", "界🌱"] {
            for repeats in [7, 31, 127] {
                let prefix = block.repeat(repeats);
                let raw_query = format!("{prefix}{block}終");
                let query = Query::new(lowercase(&raw_query));
                let bodies = [
                    (format!("{}終", block.repeat(repeats * 3)), true),
                    (format!("{}違", block.repeat(repeats * 3)), false),
                    (format!("{prefix}違{prefix}{block}終"), true),
                    (format!("{prefix}終"), false),
                    (String::new(), false),
                    (raw_query, true),
                ];
                for (body, expected) in bodies {
                    let folded: String = body.chars().flat_map(char::to_lowercase).collect();
                    assert_eq!(folded.contains(&query.text), expected);
                    assert_eq!(
                        query.contains(&body),
                        expected,
                        "block {block:?}, repeats {repeats}, body {body:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn history_query_matches_scalar_folding_and_stops_at_first_match() {
        let alphabet = ['A', 'a', 'Σ', 'σ', 'ς', 'İ', 'i', '\u{307}', '界', '🌱'];
        let samples: Vec<_> = std::iter::once(String::new())
            .chain(alphabet.iter().map(char::to_string))
            .chain(alphabet.iter().flat_map(|first| {
                alphabet
                    .iter()
                    .map(move |second| format!("{first}{second}"))
            }))
            .chain([
                "aaaaabaaaaac".into(),
                "aaaaac".into(),
                "ababababac".into(),
                "ababac".into(),
            ])
            .collect();
        for text in &samples {
            let query = Query::new(lowercase(text));
            for body in &samples {
                assert_eq!(
                    query.contains(body),
                    lowercase(body).contains(&query.text),
                    "query {text:?}, body {body:?}"
                );
            }
        }

        let query = Query::new("needle".into());
        let consumed = Cell::new(0);
        let body = "NEEDLE"
            .chars()
            .chain(std::iter::repeat_n('界', 1_000_000))
            .inspect(|_| consumed.set(consumed.get() + 1));
        assert!(query.contains_chars(body));
        assert_eq!(consumed.get(), 6);
    }
}
