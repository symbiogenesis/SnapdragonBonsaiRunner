//! Local, advisory context estimates. Only the broker's tokenizer can admit a request.
//!
//! These intentionally inexpensive heuristics are neither exact token counts nor
//! upper bounds. They must never gate Send, trim history or change response settings.
#![expect(
    clippy::redundant_pub_crate,
    reason = "The private context estimator exposes its policy and estimates to sibling UI modules."
)]

use crate::client::Message;

/// Largest context declared by the pinned Swift model metadata.
pub(crate) const MAX_CONTEXT_TOKENS: usize = 262_144;
/// Exact broker policy reserve, separate from the estimated rendered prompt.
pub(crate) const RESERVE_TOKENS: usize = 64;
/// Heuristic per-image vision token budget aligned with `cpu_policy::DEFAULT_IMAGE_MAX_TOKENS`.
pub(crate) const IMAGE_TOKENS: usize = 1_024;

const MESSAGE_FRAMING_TOKENS: usize = 8;
const NEXT_TURN_FRAMING_TOKENS: usize = 16;
// Approximate the planning instructions, workflow wrapper and null working JSON
// in client/effort.rs. Later passes also include generated notes/drafts and are
// deliberately outside this first-pass estimate.
const EFFORT_PLANNING_TOKENS: usize = 192;
const OUTPUT_CAPS: [u32; 4] = [512, 1_024, 2_048, 4_096];

static SYSTEM_PROMPT_TOKENS: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    text_tokens(crate::client::SYSTEM_PROMPT).saturating_add(MESSAGE_FRAMING_TOKENS)
});

/// Approximate token cost of the fixed system instruction turn sent with every request.
pub(crate) fn system_prompt_tokens() -> usize {
    *SYSTEM_PROMPT_TOKENS
}

/// Advisory presentation state, never an admission decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Pressure {
    Comfortable,
    Near,
    Over,
}

/// Estimated first-request occupancy with an exact selected response reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Estimate {
    /// Selected broker capacity; unknown until this model attempt is ready.
    pub(crate) capacity: Option<usize>,
    /// Committed content, retained reasoning and approximate message framing.
    pub(crate) history: usize,
    /// Trimmed draft plus approximate next-turn template framing.
    pub(crate) draft: usize,
    /// Estimated token cost of active persistent instructions & context cards.
    pub(crate) memory: usize,
    /// Approximate Extra effort planning wrapper; zero for Standard.
    pub(crate) workflow: usize,
    /// Selected response cap, including generated thinking.
    pub(crate) output: usize,
    /// Estimated prompt plus exact response cap and policy reserve.
    pub(crate) total: usize,
}

impl Estimate {
    /// Combine a cached committed-history estimate with the current draft/options.
    pub(crate) fn new(
        history: usize,
        draft: &str,
        output_cap: u32,
        extra_effort: bool,
        capacity: Option<usize>,
    ) -> Self {
        Self::with_images(history, draft, 0, output_cap, extra_effort, capacity)
    }

    /// Combine a cached committed-history estimate with the current draft, images and options.
    pub(crate) fn with_images(
        history: usize,
        draft: &str,
        draft_images: usize,
        output_cap: u32,
        extra_effort: bool,
        capacity: Option<usize>,
    ) -> Self {
        Self::from_draft_tokens(
            history,
            text_tokens(draft.trim()),
            draft_images,
            output_cap,
            extra_effort,
            capacity,
        )
    }

    /// Combine cached history, precomputed trimmed-draft tokens, images and options.
    pub(crate) fn from_draft_tokens(
        history: usize,
        trimmed_draft_tokens: usize,
        draft_images: usize,
        output_cap: u32,
        extra_effort: bool,
        capacity: Option<usize>,
    ) -> Self {
        let draft = trimmed_draft_tokens
            .saturating_add(draft_images.saturating_mul(IMAGE_TOKENS))
            .saturating_add(NEXT_TURN_FRAMING_TOKENS)
            .saturating_add(system_prompt_tokens());
        let workflow = if extra_effort {
            EFFORT_PLANNING_TOKENS
        } else {
            0
        };
        let output = usize::try_from(output_cap).unwrap_or(usize::MAX);
        let total = history
            .saturating_add(draft)
            .saturating_add(workflow)
            .saturating_add(output)
            .saturating_add(RESERVE_TOKENS);
        Self {
            capacity,
            history,
            draft,
            memory: 0,
            workflow,
            output,
            total,
        }
    }

    /// Add active persistent instructions & context-card token cost to this estimate.
    pub(crate) const fn with_memory(mut self, memory_tokens: usize) -> Self {
        let delta = memory_tokens.saturating_sub(self.memory);
        self.memory = memory_tokens;
        self.total = self.total.saturating_add(delta);
        self
    }

    /// Rendered input estimate before the response cap and policy reserve.
    pub(crate) const fn prompt_tokens(self) -> usize {
        self.history
            .saturating_add(self.draft)
            .saturating_add(self.memory)
            .saturating_add(self.workflow)
    }

    /// Estimated spare capacity after reserving the complete selected response.
    pub(crate) const fn remaining(self) -> usize {
        match self.capacity {
            Some(capacity) => capacity.saturating_sub(self.total),
            None => 0,
        }
    }

    /// Estimated excess; zero also includes exact equality with the policy limit.
    pub(crate) const fn overflow(self) -> usize {
        match self.capacity {
            Some(capacity) => self.total.saturating_sub(capacity),
            None => 0,
        }
    }

    /// Distinguish comfortable, near-capacity and estimated over-capacity states.
    pub(crate) const fn pressure(self) -> Pressure {
        let Some(capacity) = self.capacity else {
            return Pressure::Comfortable;
        };
        if self.total > capacity {
            Pressure::Over
        } else if self.total >= capacity.saturating_mul(85) / 100 {
            Pressure::Near
        } else {
            Pressure::Comfortable
        }
    }

    /// Largest supported smaller response cap that fits this input estimate.
    ///
    /// This is an explicit user action suggestion, never a promise of admission.
    pub(crate) fn suggested_cap(self) -> Option<u32> {
        let available = self
            .capacity?
            .saturating_sub(self.prompt_tokens().saturating_add(RESERVE_TOKENS));
        OUTPUT_CAPS.into_iter().rev().find(|&cap| {
            usize::try_from(cap).is_ok_and(|cap| cap < self.output && cap <= available)
        })
    }
}

/// Estimate committed model history, including reasoning even with Thinking off.
///
/// Call only when conversation history changes; adopted partial replies are
/// included and unadopted effort drafts are excluded. No history text is retained here.
pub(crate) fn history_tokens(history: &[Message]) -> usize {
    history.iter().fold(0_usize, |total, message| {
        total
            .saturating_add(MESSAGE_FRAMING_TOKENS)
            .saturating_add(text_tokens(&message.content))
            .saturating_add(message.images.len().saturating_mul(IMAGE_TOKENS))
            .saturating_add(message.reasoning_content.as_deref().map_or(0, text_tokens))
    })
}

// Eighth-token weights avoid per-character floating point and rounding drift.
// Prose has cheaper spaces; punctuation/digits make code and numeric material
// denser. Non-ASCII scripts and emoji must not inherit English chars/4 behavior.
pub(crate) fn text_units(text: &str) -> usize {
    text.chars().fold(0_usize, |total, character| {
        let weight = if character.is_ascii_alphabetic() {
            2
        } else if character.is_ascii_digit() {
            3
        } else if character.is_whitespace() {
            1
        } else if character.is_ascii() {
            8
        } else if matches!(
            character,
            '\u{2e80}'..='\u{9fff}' | '\u{ac00}'..='\u{d7af}' | '\u{f900}'..='\u{faff}'
        ) {
            12
        } else if matches!(character, '\u{2600}'..='\u{27bf}' | '\u{1f000}'..='\u{1faff}') {
            24
        } else {
            character.len_utf8() * 4
        };
        total.saturating_add(weight)
    })
}

pub(crate) const fn units_to_tokens(units: usize) -> usize {
    units.div_ceil(8)
}

pub(crate) fn text_tokens(text: &str) -> usize {
    units_to_tokens(text_units(text))
}

/// Incrementally maintains eighth-token units for trimmed draft text by scanning
/// only the changed interior span between shared UTF-8 prefix and suffix slices.
#[derive(Default)]
pub(crate) struct IncrementalDraftUnits {
    prev_trimmed: String,
    trimmed_units: usize,
}

impl IncrementalDraftUnits {
    #[cfg(test)]
    pub(crate) fn cached_ptr(&self) -> *const u8 {
        self.prev_trimmed.as_ptr()
    }

    #[cfg(test)]
    pub(crate) fn matches(&self, draft: &str) -> bool {
        self.prev_trimmed == draft.trim()
    }

    pub(crate) fn units(&mut self, draft: &str) -> usize {
        let trimmed = draft.trim();
        if self.prev_trimmed == trimmed {
            return self.trimmed_units;
        }
        let (prefix, old_mid_end, new_mid_end) = utf8_diff_bounds(&self.prev_trimmed, trimmed);
        let old_units = text_units(&self.prev_trimmed[prefix..old_mid_end]);
        let new_units = text_units(&trimmed[prefix..new_mid_end]);
        self.trimmed_units = self
            .trimmed_units
            .saturating_sub(old_units)
            .saturating_add(new_units);
        self.prev_trimmed
            .replace_range(prefix..old_mid_end, &trimmed[prefix..new_mid_end]);
        self.trimmed_units
    }

    pub(crate) fn tokens(&mut self, draft: &str) -> usize {
        units_to_tokens(self.units(draft))
    }
}

pub(crate) fn utf8_diff_bounds(old: &str, new: &str) -> (usize, usize, usize) {
    let old_bytes = old.as_bytes();
    let new_bytes = new.as_bytes();
    let mut prefix = old_bytes
        .iter()
        .zip(new_bytes.iter())
        .take_while(|(a, b)| a == b)
        .count();
    while prefix > 0 && (!old.is_char_boundary(prefix) || !new.is_char_boundary(prefix)) {
        prefix -= 1;
    }
    let max_suffix = old_bytes
        .len()
        .saturating_sub(prefix)
        .min(new_bytes.len().saturating_sub(prefix));
    let mut suffix = old_bytes[old_bytes.len() - max_suffix..]
        .iter()
        .rev()
        .zip(new_bytes[new_bytes.len() - max_suffix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    while suffix > 0
        && (!old.is_char_boundary(old_bytes.len() - suffix)
            || !new.is_char_boundary(new_bytes.len() - suffix))
    {
        suffix -= 1;
    }
    (
        prefix,
        old_bytes.len().saturating_sub(suffix),
        new_bytes.len().saturating_sub(suffix),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTEXT_TOKENS: usize = 8_192;

    #[test]
    fn selected_capacity_changes_guidance_without_changing_occupancy() {
        let small = Estimate::new(8_000, "hello", 4_096, false, Some(8_192));
        let large = Estimate::new(8_000, "hello", 4_096, false, Some(131_072));
        let unknown = Estimate::new(8_000, "hello", 4_096, false, None);
        assert_eq!(small.total, large.total);
        assert_eq!(small.pressure(), Pressure::Over);
        assert_eq!(large.pressure(), Pressure::Comfortable);
        assert_eq!(large.remaining(), 131_072 - large.total);
        assert_eq!(unknown.pressure(), Pressure::Comfortable);
        assert_eq!(unknown.suggested_cap(), None);
    }

    #[test]
    fn response_reservation_and_exact_boundary_remain_distinct_from_estimation() {
        let history = CONTEXT_TOKENS
            - RESERVE_TOKENS
            - NEXT_TURN_FRAMING_TOKENS
            - system_prompt_tokens()
            - 512;
        let fits = Estimate::new(history, "", 512, false, Some(CONTEXT_TOKENS));
        assert_eq!(fits.total, CONTEXT_TOKENS);
        assert_eq!(fits.pressure(), Pressure::Near);
        assert_eq!(fits.remaining(), 0);
        assert_eq!(fits.overflow(), 0);
        let over = Estimate::new(history + 1, "", 512, false, Some(CONTEXT_TOKENS));
        assert_eq!(over.pressure(), Pressure::Over);
        assert_eq!(over.overflow(), 1);
        assert_eq!(over.suggested_cap(), None);
        assert!(system_prompt_tokens() > MESSAGE_FRAMING_TOKENS);
    }

    #[test]
    fn lower_cap_suggestion_preserves_input_and_reserve() {
        let large = Estimate::new(5_000, "hello", 4_096, false, Some(CONTEXT_TOKENS));
        assert_eq!(large.pressure(), Pressure::Over);
        assert_eq!(large.suggested_cap(), Some(2_048));
        let small = Estimate::new(5_000, "hello", 2_048, false, Some(CONTEXT_TOKENS));
        assert_eq!(large.prompt_tokens(), small.prompt_tokens());
        assert_eq!(large.total - small.total, 2_048);
        assert_eq!(small.overflow(), 0);
        assert_eq!(
            Estimate::new(8_000, "hello", 4_096, false, Some(CONTEXT_TOKENS)).suggested_cap(),
            None
        );
    }

    #[test]
    fn committed_reasoning_counts_independently_of_visible_answer() {
        let mut messages = vec![Message {
            role: "assistant",
            content: "Visible answer".into(),
            reasoning_content: None,
            images: std::sync::Arc::default(),
        }];
        let visible = history_tokens(&messages);
        messages[0].reasoning_content = Some("Retained reasoning 雨".into());
        assert_eq!(
            history_tokens(&messages),
            visible + text_tokens("Retained reasoning 雨")
        );
        assert_eq!(history_tokens(&[]), 0);
    }

    #[test]
    fn unicode_and_code_are_not_english_character_counts() {
        assert!(text_tokens("雨雨雨雨") > text_tokens("aaaa"));
        assert!(text_tokens("🌱🌱🌱🌱") > text_tokens("雨雨雨雨"));
        assert!(text_tokens("{}[]();=") > text_tokens("abcdefgh"));
        assert!(text_tokens("مرحبا") > 0);
        assert!(text_tokens("नमस्ते") > 0);
        assert_eq!(text_tokens(""), 0);
    }

    #[test]
    fn draft_trimming_and_extra_effort_match_their_estimated_scope() {
        let standard = Estimate::new(0, "question", 512, false, Some(CONTEXT_TOKENS));
        assert_eq!(
            standard,
            Estimate::new(0, " \nquestion\t ", 512, false, Some(CONTEXT_TOKENS))
        );
        let effort = Estimate::new(0, "question", 512, true, Some(CONTEXT_TOKENS));
        assert_eq!(effort.workflow, EFFORT_PLANNING_TOKENS);
        assert_eq!(effort.total - standard.total, EFFORT_PLANNING_TOKENS);
        assert_eq!(standard.pressure(), Pressure::Comfortable);
    }

    #[test]
    fn saturated_totals_cannot_wrap_into_available_capacity() {
        let estimate = Estimate::new(usize::MAX, "question", u32::MAX, true, Some(CONTEXT_TOKENS));
        assert_eq!(estimate.total, usize::MAX);
        assert_eq!(estimate.pressure(), Pressure::Over);
        assert_eq!(estimate.remaining(), 0);
        assert_eq!(estimate.suggested_cap(), None);
    }

    #[test]
    fn image_tokens_count_for_draft_and_committed_history() {
        let text_only = Estimate::with_images(0, "Describe", 0, 512, false, Some(CONTEXT_TOKENS));
        let two_images = Estimate::with_images(0, "Describe", 2, 512, false, Some(CONTEXT_TOKENS));
        assert_eq!(two_images.draft - text_only.draft, 2 * IMAGE_TOKENS);
        assert_eq!(two_images.total - text_only.total, 2 * IMAGE_TOKENS);

        let messages = [Message {
            role: "user",
            content: "".into(),
            reasoning_content: None,
            images: vec![
                std::sync::Arc::from("data:image/png;base64,AAAA"),
                std::sync::Arc::from("data:image/png;base64,BBBB"),
                std::sync::Arc::from("data:image/png;base64,CCCC"),
            ]
            .into(),
        }];
        assert_eq!(
            history_tokens(&messages),
            MESSAGE_FRAMING_TOKENS + 3 * IMAGE_TOKENS
        );
    }

    #[test]
    fn incremental_draft_units_exactly_matches_full_trim_scan() {
        let mut tracker = IncrementalDraftUnits::default();
        let samples = [
            "",
            "   \n\t ",
            "hello world",
            "  hello world!  ",
            "hello 👋 world! 界",
            "hello 👋 updated 🌍 mid 界",
            "prefix hello 👋 updated 🌍 mid 界",
            "prefix hello 🌍 mid",
            "\t\n中文テスト 🌱🚀 café 12345 []{}",
            "  中文🌱🚀 café  ",
            "",
        ];
        for sample in samples {
            assert_eq!(tracker.tokens(sample), text_tokens(sample.trim()));
            assert!(tracker.matches(sample));
        }
    }
}
