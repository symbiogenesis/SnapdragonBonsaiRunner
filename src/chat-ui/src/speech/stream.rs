//! Unicode-safe incremental speech segmentation with a bounded text lifetime.

use super::segmentation::{
    self, horizontal_whitespace, is_clause_punctuation, is_closing_delimiter,
    is_sentence_punctuation, line_break,
};
use super::{Duration, Instant, MAX_TEXT, VecDeque};

const LATENCY: Duration = Duration::from_millis(700);
const FIRST_CLAUSE_LATENCY: Duration = Duration::from_millis(250);
const FIRST_CLAUSE_MIN_CHARS: usize = 24;

#[derive(Default)]
pub(super) struct Text {
    pending: String,
    units: usize,
    since: Option<Instant>,
    pub(super) finished: bool,
    first_emitted: bool,
    scanned_ws: usize,
    scanned_bytes: usize,
    scanned_chars: usize,
    #[cfg(test)]
    compacted_bytes: usize,
    #[cfg(test)]
    scanned_checks: usize,
}

fn safe_scan_end(pending: &str, min_offset: usize, watch_clauses: bool) -> usize {
    let mut end = pending.len();
    while end > min_offset {
        let Some(ch) = pending[min_offset..end].chars().next_back() else {
            break;
        };
        if is_sentence_punctuation(ch)
            || is_closing_delimiter(ch)
            || ch == '\r'
            || (watch_clauses && is_clause_punctuation(ch))
        {
            end -= ch.len_utf8();
        } else {
            break;
        }
    }
    end
}

impl Text {
    pub(super) fn append(&mut self, text: &str, now: Instant) -> Result<(), &'static str> {
        let units = text.encode_utf16().take(MAX_TEXT + 1).count();
        if self.finished || text.contains('\0') || units > MAX_TEXT.saturating_sub(self.units) {
            return Err(
                "Streaming read aloud exceeded its 32,768 UTF-16 character limit or received invalid text.",
            );
        }
        self.units += units;
        self.pending.push_str(text);
        // The fallback is an idle timeout, not a periodic word-fragment timer.
        // Keep collecting a continuous clause until punctuation or MAX_SEGMENT
        // releases it; otherwise each brief fragment restarts model synthesis.
        if !text.is_empty() && (self.since.is_some() || !text.trim().is_empty()) {
            self.since = Some(now);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn ready(&mut self, now: Instant) -> VecDeque<String> {
        let mut result = VecDeque::new();
        self.drain_ready(now, &mut result);
        result
    }

    pub(super) fn drain_ready(&mut self, now: Instant, result: &mut VecDeque<String>) {
        let latency = if self.first_emitted {
            LATENCY
        } else {
            FIRST_CLAUSE_LATENCY
        };
        let expired = self
            .since
            .is_some_and(|since| now.duration_since(since) >= latency);
        if !self.finished && !expired && self.since.is_some() {
            let ws_start = if self.scanned_bytes > 0 {
                self.scanned_ws
            } else {
                self.pending.len() - self.pending.trim_start_matches(horizontal_whitespace).len()
            };
            let resume = self.scanned_bytes.max(ws_start);
            let tail = &self.pending[resume..];
            #[cfg(test)]
            {
                self.scanned_checks += tail.len();
            }
            let tail_chars = tail.chars().count();
            let total_chars = self.scanned_chars + tail_chars;
            let clause_candidate = !self.first_emitted
                && total_chars >= FIRST_CLAUSE_MIN_CHARS
                && tail.chars().any(is_clause_punctuation);
            if !clause_candidate
                && !tail
                    .chars()
                    .any(|ch| line_break(ch) || is_sentence_punctuation(ch))
                && total_chars <= super::MAX_SEGMENT
            {
                let safe_end = safe_scan_end(&self.pending, ws_start, !self.first_emitted);
                if safe_end > resume {
                    self.scanned_chars += self.pending[resume..safe_end].chars().count();
                    self.scanned_bytes = safe_end;
                } else if self.scanned_bytes == 0 {
                    self.scanned_bytes = ws_start;
                    self.scanned_chars = 0;
                }
                self.scanned_ws = ws_start;
                return;
            }
        }
        let mut consumed = 0;
        let mut emitted = false;
        loop {
            // Keep line breaks as prosody cues, including when they arrive in
            // the append after a sentence has already been released.
            let pending = self.pending[consumed..].trim_start_matches(horizontal_whitespace);
            consumed = self.pending.len() - pending.len();
            if !self.finished && pending.trim().is_empty() {
                break;
            }
            #[cfg(test)]
            {
                self.scanned_checks += pending.len();
            }
            let maximum = segmentation::maximum(pending);
            let limit = maximum.unwrap_or(pending.len());
            let sentence = if self.finished {
                Some(segmentation::clip_boundary(pending))
            } else {
                segmentation::sentence_boundary(pending, limit, expired).or_else(|| {
                    (!self.first_emitted)
                        .then(|| {
                            segmentation::first_clause_boundary(
                                pending,
                                limit,
                                expired,
                                FIRST_CLAUSE_MIN_CHARS,
                            )
                        })
                        .flatten()
                })
            };
            let boundary = sentence.or_else(|| {
                if maximum.is_some() {
                    Some(segmentation::phrase_boundary(pending, limit))
                } else if self.finished {
                    Some(pending.len())
                } else if expired {
                    pending
                        .char_indices()
                        .rev()
                        .find(|(_, character)| character.is_whitespace())
                        .map(|(index, _)| index)
                } else {
                    None
                }
            });
            let Some(mut boundary) = boundary.filter(|boundary| *boundary > 0) else {
                break;
            };
            // Carry immediately available paragraph separators with their
            // sentence. Charge them to the same character cap as spoken text.
            let available = super::MAX_SEGMENT - pending[..boundary].chars().count();
            for character in pending[boundary..].chars().take(available) {
                if !character.is_whitespace() {
                    break;
                }
                boundary += character.len_utf8();
            }
            let piece = pending[..boundary].trim_matches(horizontal_whitespace);
            if !piece.trim().is_empty() {
                result.push_back(piece.to_owned());
                emitted = true;
                self.first_emitted = true;
            }
            consumed += boundary;
        }
        if consumed == self.pending.len() {
            self.pending.clear();
            self.since = None;
            self.scanned_ws = 0;
            self.scanned_bytes = 0;
            self.scanned_chars = 0;
        } else {
            if consumed > 0 {
                #[cfg(test)]
                {
                    self.compacted_bytes += self.pending.len() - consumed;
                }
                self.pending.drain(..consumed);
                self.scanned_ws = 0;
                self.scanned_bytes = 0;
                self.scanned_chars = 0;
            }
            if emitted {
                self.since = Some(now);
            }
            if !self.finished && !expired && self.since.is_some() && !self.pending.is_empty() {
                let ws_start = self.pending.len()
                    - self.pending.trim_start_matches(horizontal_whitespace).len();
                let safe_end = safe_scan_end(&self.pending, ws_start, !self.first_emitted);
                self.scanned_ws = ws_start;
                self.scanned_bytes = safe_end;
                self.scanned_chars = self.pending[ws_start..safe_end].chars().count();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FIRST_CLAUSE_LATENCY, Instant, LATENCY, MAX_TEXT, Text};
    use crate::speech::{MAX_SEGMENT, segments};

    #[test]
    fn unicode_line_breaks_survive_every_stream_fragment_boundary() -> Result<(), &'static str> {
        let now = Instant::now();
        for separator in [
            "\r", "\n", "\r\n", "\u{b}", "\u{c}", "\u{85}", "\u{2028}", "\u{2029}",
        ] {
            let input = format!("First{separator}{separator}Second{separator}tail");
            for split in input
                .char_indices()
                .map(|(offset, _)| offset)
                .chain([input.len()])
            {
                let mut text = Text::default();
                text.append(&input[..split], now)?;
                let mut ready = text.ready(now);
                text.append(&input[split..], now)?;
                ready.extend(text.ready(now));
                text.finished = true;
                ready.extend(text.ready(now));
                assert_eq!(ready.into_iter().collect::<String>(), input);
                assert!(text.ready(now).is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn unicode_line_breaks_are_retained_without_exceeding_the_scalar_clip_limit()
    -> Result<(), &'static str> {
        let now = Instant::now();
        for separator in [
            "\r", "\n", "\r\n", "\u{b}", "\u{c}", "\u{85}", "\u{2028}", "\u{2029}",
        ] {
            let input = format!("{}{separator}{separator}tail", "界".repeat(MAX_SEGMENT - 1));
            let mut text = Text::default();
            text.append(&input, now)?;
            let mut ready = text.ready(now);
            text.finished = true;
            ready.extend(text.ready(now));
            let complete = segments(&input).ok_or("Expected complete speech segments")?;
            assert_eq!(ready, complete);
            assert!(
                ready
                    .iter()
                    .all(|piece| piece.chars().count() <= MAX_SEGMENT)
            );
            assert_eq!(ready.into_iter().collect::<String>(), input);

            let mut text = Text::default();
            text.append(separator, now)?;
            assert!(text.ready(now).is_empty());
            text.finished = true;
            assert!(text.ready(now).is_empty());
        }
        Ok(())
    }

    #[test]
    fn complete_and_streaming_speech_share_hard_clip_boundaries() -> Result<(), &'static str> {
        let now = Instant::now();
        for input in [
            "a".repeat(MAX_SEGMENT),
            "🙂".repeat(MAX_SEGMENT + 5),
            "A bounded Unicode phrase 🙂 ".repeat(30),
            "First sentence. \"Second!\" (Third?) 🙂".to_owned(),
            format!(
                "{} {}",
                "é".repeat(MAX_SEGMENT / 2),
                "x".repeat(MAX_SEGMENT)
            ),
        ] {
            let expected = segments(&input).ok_or("Expected complete speech segments")?;
            let mut streamed = Text::default();
            streamed.append(&input, now)?;
            streamed.finished = true;
            assert_eq!(streamed.ready(now), expected);
        }
        Ok(())
    }

    #[test]
    fn streaming_sentences_start_before_finish_and_final_tail_is_not_repeated()
    -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        text.append("Hello 🙂. The next", now)?;
        assert_eq!(text.ready(now), ["Hello 🙂."]);
        text.append(" sentence! tail", now)?;
        assert_eq!(text.ready(now), ["The next sentence!"]);
        text.finished = true;
        assert_eq!(text.ready(now), ["tail"]);
        assert!(text.ready(now).is_empty());
        assert!(text.append("late", now).is_err());
        Ok(())
    }

    #[test]
    fn streaming_punctuation_handles_delimiters_numbered_lists_dividers_and_idle_endings()
    -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        text.append(
            "\"First sentence!\" (Second?) **Third.** 1. Numbered item. ---\nNext",
            now,
        )?;
        assert_eq!(
            text.ready(now),
            [
                "\"First sentence!\"",
                "(Second?)",
                "**Third.**",
                "1. Numbered item."
            ]
        );
        text.append(" sentence.", now)?;
        assert!(text.ready(now).is_empty());
        assert_eq!(text.ready(now + LATENCY), ["---\nNext sentence."]);
        assert!(text.ready(now + LATENCY).is_empty());
        Ok(())
    }

    #[test]
    fn streaming_latency_flush_preserves_partial_word_and_unicode_limit() -> Result<(), &'static str>
    {
        let now = Instant::now();
        let mut text = Text::default();
        text.append("A clause without punctuation part", now)?;
        assert!(text.ready(now).is_empty());
        assert_eq!(text.ready(now + LATENCY), ["A clause without punctuation"]);
        text.append("ial 🙂", now + LATENCY)?;
        text.finished = true;
        assert_eq!(text.ready(now + LATENCY), ["partial 🙂"]);
        let mut text = Text::default();
        text.append(&"🙂".repeat(MAX_SEGMENT + 5), now)?;
        let ready = text.ready(now);
        assert_eq!(
            ready.front().map(|part| part.chars().count()),
            Some(MAX_SEGMENT)
        );
        text.finished = true;
        assert_eq!(
            text.ready(now).front().map(|part| part.chars().count()),
            Some(5)
        );
        Ok(())
    }

    #[test]
    fn streaming_latency_waits_for_idle_while_words_keep_arriving() -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        // First segment uses the shorter 250 ms idle clause timeout, resetting while tokens arrive.
        text.append("A clause", now)?;
        let almost_first = FIRST_CLAUSE_LATENCY
            .checked_sub(std::time::Duration::from_millis(1))
            .ok_or("first-clause latency must exceed one millisecond")?;
        let later = now + almost_first;
        text.append(" continuing", later)?;
        assert!(text.ready(now + FIRST_CLAUSE_LATENCY).is_empty());
        assert!(text.ready(later + almost_first).is_empty());
        assert_eq!(text.ready(later + FIRST_CLAUSE_LATENCY), ["A clause"]);

        // Subsequent segments keep the full 700 ms latency window.
        let next_start = later + FIRST_CLAUSE_LATENCY;
        text.append(" softly", next_start)?;
        let almost_latency = LATENCY
            .checked_sub(std::time::Duration::from_millis(1))
            .ok_or("latency must exceed one millisecond")?;
        let second_later = next_start + almost_latency;
        text.append(" onward", second_later)?;
        assert!(text.ready(next_start + LATENCY).is_empty());
        assert!(text.ready(second_later + almost_latency).is_empty());
        assert_eq!(text.ready(second_later + LATENCY), ["continuing softly"]);

        text.append(".", second_later + LATENCY)?;
        text.finished = true;
        assert_eq!(text.ready(second_later + LATENCY), ["onward."]);
        assert!(text.ready(second_later + LATENCY).is_empty());
        Ok(())
    }

    #[test]
    fn streaming_empty_append_does_not_defer_an_idle_clause() -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        text.append("A paused clause tail", now)?;
        text.append("", now + LATENCY)?;
        assert_eq!(text.ready(now + LATENCY), ["A paused clause"]);
        text.finished = true;
        assert_eq!(text.ready(now + LATENCY), ["tail"]);
        Ok(())
    }

    #[test]
    fn streaming_text_limit_counts_all_consumed_utf16_and_rejects_nul() -> Result<(), &'static str>
    {
        let now = Instant::now();
        let mut text = Text::default();
        text.append(&"🙂".repeat(MAX_TEXT / 2), now)?;
        assert!(!text.ready(now).is_empty());
        assert!(text.append("x", now).is_err());
        assert!(Text::default().append("bad\0text", now).is_err());
        Ok(())
    }

    #[test]
    fn streaming_many_sentences_compact_only_the_final_unconsumed_tail() -> Result<(), &'static str>
    {
        let now = Instant::now();
        let mut text = Text::default();
        text.append(&format!("{}tail", "Sentence 🙂. ".repeat(1000)), now)?;
        let ready = text.ready(now);
        assert_eq!(ready.len(), 1000);
        assert!(ready.iter().all(|part| part == "Sentence 🙂."));
        assert_eq!(text.pending, "tail");
        assert_eq!(text.compacted_bytes, "tail".len());
        text.finished = true;
        assert_eq!(text.ready(now), ["tail"]);
        assert_eq!(text.compacted_bytes, "tail".len());
        Ok(())
    }

    #[test]
    fn streaming_titles_initials_and_decimals_stay_with_their_phrases() -> Result<(), &'static str>
    {
        let now = Instant::now();
        let mut text = Text::default();
        text.append("Dr. ", now)?;
        assert!(text.ready(now).is_empty());
        text.append(
            "Smith met J. R. Tolkien in the U.S. at 3.14 p.m., e.g. today, i.e. Tuesday. Next",
            now,
        )?;
        assert_eq!(
            text.ready(now),
            ["Dr. Smith met J. R. Tolkien in the U.S. at 3.14 p.m., e.g. today, i.e. Tuesday."]
        );
        text.finished = true;
        assert_eq!(text.ready(now), ["Next"]);
        Ok(())
    }

    #[test]
    fn streaming_cjk_punctuation_releases_without_whitespace_and_keeps_closing_quotes()
    -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        text.append("「你好。」世界！下一句", now)?;
        assert_eq!(text.ready(now), ["「你好。」", "世界！"]);
        text.finished = true;
        assert_eq!(text.ready(now), ["下一句"]);
        Ok(())
    }

    #[test]
    fn completed_speech_groups_sentences_without_changing_internal_whitespace()
    -> Result<(), &'static str> {
        let now = Instant::now();
        let input = "First.  Second!\n\"Third?\"\tFourth. Last unfinished phrase";
        assert_eq!(segments(input).ok_or("Expected speech segments")?, [input]);
        let mut text = Text::default();
        text.append(input, now)?;
        text.finished = true;
        assert_eq!(text.ready(now), [input]);
        assert!(text.ready(now).is_empty());
        Ok(())
    }

    #[test]
    fn completed_speech_groups_many_sentences_within_the_unicode_clip_limit()
    -> Result<(), &'static str> {
        let now = Instant::now();
        let sentence = "A short 🙂 sentence.  ";
        let input = sentence.repeat(40);
        let pieces = segments(&input).ok_or("Expected speech segments")?;
        assert!(pieces.len() > 1);
        assert!(pieces.len() < 40);
        assert!(
            pieces
                .iter()
                .all(|piece| piece.chars().count() <= MAX_SEGMENT)
        );
        let original: String = input
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        let rebuilt: String = pieces
            .iter()
            .flat_map(|piece| piece.chars())
            .filter(|character| !character.is_whitespace())
            .collect();
        assert_eq!(rebuilt, original);
        let mut text = Text::default();
        text.append(&input, now)?;
        text.finished = true;
        assert_eq!(text.ready(now), pieces);
        assert!(text.ready(now).is_empty());
        Ok(())
    }

    #[test]
    fn streaming_hard_clip_uses_the_same_clause_cut_and_preserves_text() -> Result<(), &'static str>
    {
        let now = Instant::now();
        let head = format!("{};", "A measured introduction ".repeat(6));
        let input = format!("{head} {}", "another phrase without an ending ".repeat(12));
        let expected = segments(&input).ok_or("Expected speech segments")?;
        assert_eq!(expected.front(), Some(&head));
        let mut text = Text::default();
        text.append(&input, now)?;
        let mut ready = text.ready(now);
        assert_eq!(ready.front(), Some(&head));
        text.finished = true;
        ready.extend(text.ready(now));
        let rebuilt: String = ready
            .iter()
            .flat_map(|piece| piece.chars())
            .filter(|character| !character.is_whitespace())
            .collect();
        let original: String = input
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        assert_eq!(rebuilt, original);
        assert!(
            ready
                .iter()
                .all(|piece| piece.chars().count() <= MAX_SEGMENT)
        );
        Ok(())
    }

    #[test]
    fn streaming_paragraph_breaks_survive_sentence_and_transport_boundaries()
    -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        text.append("First.\r\n\r\nSecond. tail", now)?;
        assert_eq!(text.ready(now), ["First.\r\n\r\n", "Second."]);
        text.finished = true;
        assert_eq!(text.ready(now), ["tail"]);

        let mut text = Text::default();
        text.append("First. ", now)?;
        assert_eq!(text.ready(now), ["First."]);
        text.append("\n\nSecond. tail", now)?;
        assert_eq!(text.ready(now), ["\n\nSecond."]);
        text.finished = true;
        assert_eq!(text.ready(now), ["tail"]);

        let mut text = Text::default();
        text.append("First. ", now)?;
        assert_eq!(text.ready(now), ["First."]);
        text.append("\n\n", now)?;
        assert!(text.ready(now + LATENCY).is_empty());
        text.append("Second. tail", now + LATENCY)?;
        assert_eq!(text.ready(now + LATENCY), ["\n\nSecond."]);

        let mut text = Text::default();
        text.append(" \n \n ", now)?;
        assert!(text.ready(now).is_empty());
        text.finished = true;
        assert!(text.ready(now).is_empty());
        Ok(())
    }

    #[test]
    fn streaming_paragraph_separators_share_the_clip_character_budget() -> Result<(), &'static str>
    {
        let now = Instant::now();
        let head = format!("{}.", "a".repeat(MAX_SEGMENT - 2));
        let mut text = Text::default();
        text.append(&format!("{head}\n\nNext. tail"), now)?;
        let ready = text.ready(now);
        assert_eq!(ready, [format!("{head}\n"), "\nNext.".to_owned()]);
        assert!(
            ready
                .iter()
                .all(|piece| piece.chars().count() <= MAX_SEGMENT)
        );
        Ok(())
    }

    #[test]
    fn prosody_streaming_clauses_wait_for_sentence_context() -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        // Short early comma (< 24 chars) does not split; once >= 24 chars, the first clause flushes.
        text.append("A quiet opening,", now)?;
        assert!(text.ready(now).is_empty());
        text.append(" followed by a pause;", now)?;
        assert!(text.ready(now).is_empty());
        text.append(" then a detail:", now)?;
        assert_eq!(
            text.ready(now),
            ["A quiet opening, followed by a pause;"]
        );
        text.append(" and a complete ending.", now)?;
        assert!(text.ready(now).is_empty());
        text.append(" ", now)?;
        assert_eq!(
            text.ready(now),
            ["then a detail: and a complete ending."]
        );
        text.finished = true;
        assert!(text.ready(now).is_empty());
        Ok(())
    }

    #[test]
    fn first_segment_flushes_early_on_clause_punctuation_only_above_threshold()
    -> Result<(), &'static str> {
        let now = Instant::now();
        for separator in [',', ';', ':'] {
            let mut text = Text::default();
            // Below 24 chars, clause punctuation waits; numeric 12,345 / 08:30 also waits.
            text.append(&format!("Brief{separator} 12,345 at 08:30"), now)?;
            assert!(text.ready(now).is_empty());

            // Crossing >= 24 chars and ending on clause punctuation + whitespace flushes first segment.
            text.append(&format!(" and enough text{separator}"), now)?;
            assert!(text.ready(now).is_empty());
            text.append(" following clause, kept for sentence.", now)?;
            assert_eq!(
                text.ready(now),
                [format!(
                    "Brief{separator} 12,345 at 08:30 and enough text{separator}"
                )]
            );
            // Subsequent segments must NOT flush early on clause punctuation even >= 24 chars.
            text.append(" ", now)?;
            assert_eq!(
                text.ready(now),
                ["following clause, kept for sentence."]
            );
        }
        Ok(())
    }

    #[test]
    fn prosody_fragmented_terminators_keep_their_closing_delimiters() -> Result<(), &'static str> {
        let now = Instant::now();
        for (parts, expected, tail) in [
            (
                &["\"Wait.", ".", ".", "\"", " Next"][..],
                "\"Wait...\"",
                "Next",
            ),
            (&["Really?", "!", "”", " Next"][..], "Really?!”", "Next"),
            (&["「待って。", "」", "次"][..], "「待って。」", "次"),
        ] {
            let mut text = Text::default();
            for (index, part) in parts.iter().enumerate() {
                text.append(part, now)?;
                if index + 1 == parts.len() {
                    assert_eq!(text.ready(now), [expected]);
                } else {
                    assert!(text.ready(now).is_empty());
                }
            }
            text.finished = true;
            assert_eq!(text.ready(now), [tail]);
            assert!(text.ready(now).is_empty());
        }
        Ok(())
    }

    #[test]
    fn prosody_fragmented_crlf_preserves_every_paragraph_break() -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        text.append("First.\r", now)?;
        let mut ready = text.ready(now);
        assert_eq!(ready, ["First.\r"]);
        text.append("\n\r", now)?;
        assert!(text.ready(now + LATENCY).is_empty());
        text.append("\nSecond. tail", now + LATENCY)?;
        ready.extend(text.ready(now + LATENCY));
        assert_eq!(
            ready.into_iter().collect::<String>(),
            "First.\r\n\r\nSecond."
        );
        text.finished = true;
        assert_eq!(text.ready(now + LATENCY), ["tail"]);
        assert!(text.ready(now + LATENCY).is_empty());
        Ok(())
    }

    #[test]
    fn prosody_fragmented_numbers_and_initials_keep_sentence_context() -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        for part in [
            "Dr.",
            " Smith read",
            " 3.",
            "14 at",
            " 08:",
            "30 with J.",
            " R.",
            " Tolkien.",
        ] {
            text.append(part, now)?;
            assert!(text.ready(now).is_empty());
        }
        text.append(" Next", now)?;
        assert_eq!(
            text.ready(now),
            ["Dr. Smith read 3.14 at 08:30 with J. R. Tolkien."]
        );
        text.finished = true;
        assert_eq!(text.ready(now), ["Next"]);
        assert!(text.ready(now).is_empty());
        Ok(())
    }

    #[test]
    fn prosody_leading_punctuation_waits_for_its_spoken_phrase() -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        for part in ["… ", "Well,"] {
            text.append(part, now)?;
            assert!(text.ready(now).is_empty());
        }
        text.append(" let's continue. ", now)?;
        assert_eq!(text.ready(now), ["… Well, let's continue."]);
        text.finished = true;
        assert!(text.ready(now).is_empty());
        Ok(())
    }

    #[test]
    fn incremental_clause_buffering_avoids_quadratic_rescan() -> Result<(), &'static str> {
        let now = Instant::now();
        let mut text = Text::default();
        let mut appended_bytes = 0_usize;
        // Stream many small fragments without sentence breaks below MAX_SEGMENT.
        for _ in 0..45 {
            let chunk = "ab ";
            appended_bytes += chunk.len();
            text.append(chunk, now)?;
            assert!(text.ready(now).is_empty());
        }
        assert!(
            text.scanned_checks <= appended_bytes * 2,
            "expected linear incremental scans ({} <= {})",
            text.scanned_checks,
            appended_bytes * 2
        );
        text.finished = true;
        assert_eq!(text.ready(now).len(), 1);
        Ok(())
    }
}
