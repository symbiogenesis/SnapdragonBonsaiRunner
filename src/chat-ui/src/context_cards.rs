//! Deterministic, prefix-stable persistent instructions and keyword-triggered context cards.

use crate::client::Message;

pub(crate) const MAX_CARDS: usize = 64;
pub(crate) const MAX_INSTRUCTIONS_CHARS: usize = 2_048;
pub(crate) const MAX_CARD_TITLE_CHARS: usize = 64;
pub(crate) const MAX_CARD_CONTENT_CHARS: usize = 1_536;
pub(crate) const MAX_CARD_TRIGGERS: usize = 16;
pub(crate) const MAX_CARD_TRIGGER_CHARS: usize = 64;
pub(crate) const MIN_TOKEN_BUDGET: usize = 128;
pub(crate) const DEFAULT_TOKEN_BUDGET: usize = 1_024;
pub(crate) const MAX_TOKEN_BUDGET: usize = 4_096;
const SCAN_HISTORY_MESSAGES: usize = 4;

const PIPE_SPELLINGS: &[&str] = &[
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
];
const SLASH_SPELLINGS: &[&str] = &["</tool_call>", "</tool_response>", "</think>"];
const T_SPELLINGS: &[&str] = &[
    "<tool_call>",
    "<tool_response>",
    "<think>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
];

/// Exact 33-token Swift control-token detector aligned with `bonsai-broker`.
pub(crate) fn contains_control_tokens(text: &str) -> bool {
    let bytes = text.as_bytes();
    memchr::memchr_iter(b'<', bytes).any(|offset| {
        let remainder = &bytes[offset..];
        let candidates = match remainder.get(1) {
            Some(b'|') => PIPE_SPELLINGS,
            Some(b'/') => SLASH_SPELLINGS,
            Some(b't') => T_SPELLINGS,
            _ => return false,
        };
        candidates
            .iter()
            .any(|spelling| remainder.starts_with(spelling.as_bytes()))
    })
}

const fn default_token_budget() -> usize {
    DEFAULT_TOKEN_BUDGET
}

const fn default_true() -> bool {
    true
}

/// User-authored persistent memory card injected deterministically when pinned or triggered.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ContextCard {
    pub(crate) id: u64,
    pub(crate) title: String,
    #[serde(default)]
    pub(crate) triggers: Vec<String>,
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) pinned: bool,
    #[serde(default = "default_true")]
    pub(crate) enabled: bool,
}

impl ContextCard {
    pub(crate) fn has_control_tokens(&self) -> bool {
        contains_control_tokens(&self.title) || contains_control_tokens(&self.content)
    }

    fn valid(&self) -> bool {
        self.title.chars().count() <= MAX_CARD_TITLE_CHARS
            && self.content.chars().count() <= MAX_CARD_CONTENT_CHARS
            && self.triggers.len() <= MAX_CARD_TRIGGERS
            && self
                .triggers
                .iter()
                .all(|trigger| trigger.chars().count() <= MAX_CARD_TRIGGER_CHARS)
    }
}

/// Persisted instructions and bounded card deck shared across conversations on this device.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MemoryStore {
    #[serde(default)]
    pub(crate) persistent_instructions: String,
    #[serde(default)]
    pub(crate) cards: Vec<ContextCard>,
    #[serde(default = "default_token_budget")]
    pub(crate) token_budget: usize,
    #[serde(default)]
    pub(crate) next_card_id: u64,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self {
            persistent_instructions: String::new(),
            cards: Vec::new(),
            token_budget: DEFAULT_TOKEN_BUDGET,
            next_card_id: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CardDecision {
    IncludedPinned,
    IncludedMatched(String),
    SkippedDisabled,
    SkippedEmpty,
    SkippedNoMatch,
    SkippedControlToken,
    SkippedBudgetExceeded,
}

impl CardDecision {
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::IncludedPinned => "Included (pinned)".to_owned(),
            Self::IncludedMatched(trigger) => format!("Included (matched \"{trigger}\")"),
            Self::SkippedDisabled => "Skipped (disabled)".to_owned(),
            Self::SkippedEmpty => "Skipped (empty)".to_owned(),
            Self::SkippedNoMatch => "Skipped (no keyword match)".to_owned(),
            Self::SkippedControlToken => "Blocked (contains reserved control token)".to_owned(),
            Self::SkippedBudgetExceeded => "Skipped (memory token budget full)".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CardReport {
    pub(crate) card_id: u64,
    pub(crate) title: String,
    pub(crate) decision: CardDecision,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResolvedMemory {
    /// Persistent instructions and pinned cards kept immutable in the system prompt.
    pub(crate) pinned_appendix: Option<String>,
    /// Keyword-triggered cards attached only to the current user turn on the wire.
    pub(crate) triggered_appendix: Option<String>,
    pub(crate) estimated_tokens: usize,
    pub(crate) instructions_included: bool,
    pub(crate) instructions_rejected_control: bool,
    pub(crate) instructions_rejected_budget: bool,
    pub(crate) reports: Vec<CardReport>,
}

fn fold_unicode_case(text: &str) -> String {
    let lower = text.to_lowercase();
    if lower.contains('ß') {
        lower.replace('ß', "ss")
    } else {
        lower
    }
}

fn find_word_boundary_match(haystack_folded: &str, trigger: &str) -> bool {
    let trimmed = trigger.trim();
    if trimmed.is_empty() {
        return false;
    }
    let needle = fold_unicode_case(trimmed);
    let starts_alnum = needle.chars().next().is_some_and(char::is_alphanumeric);
    let ends_alnum = needle
        .chars()
        .next_back()
        .is_some_and(char::is_alphanumeric);
    for (start, _) in haystack_folded.match_indices(&needle) {
        let end = start.saturating_add(needle.len());
        let start_ok = !starts_alnum
            || haystack_folded[..start]
                .chars()
                .next_back()
                .is_none_or(|ch| !ch.is_alphanumeric());
        let end_ok = !ends_alnum
            || haystack_folded[end..]
                .chars()
                .next()
                .is_none_or(|ch| !ch.is_alphanumeric());
        if start_ok && end_ok {
            return true;
        }
    }
    false
}

fn format_card_section(card: &ContextCard) -> String {
    let title = card.title.trim();
    let body = card.content.trim();
    if title.is_empty() {
        body.to_owned()
    } else {
        format!("[Context: {title}]\n{body}")
    }
}

impl MemoryStore {
    pub(crate) fn is_default_empty(&self) -> bool {
        self.persistent_instructions.is_empty()
            && self.cards.is_empty()
            && self.token_budget == DEFAULT_TOKEN_BUDGET
            && self.next_card_id == 0
    }

    pub(crate) fn valid(&self) -> bool {
        if !(MIN_TOKEN_BUDGET..=MAX_TOKEN_BUDGET).contains(&self.token_budget)
            || self.persistent_instructions.chars().count() > MAX_INSTRUCTIONS_CHARS
            || self.cards.len() > MAX_CARDS
        {
            return false;
        }
        let mut seen = std::collections::HashSet::with_capacity(self.cards.len());
        self.cards
            .iter()
            .all(|card| card.id < self.next_card_id && seen.insert(card.id) && card.valid())
    }

    pub(crate) fn add_card(
        &mut self,
        title: &str,
        triggers_csv: &str,
        content: &str,
        pinned: bool,
    ) -> Result<(), &'static str> {
        if self.cards.len() >= MAX_CARDS {
            return Err("Reached the 64-card limit.");
        }
        let title = title.trim();
        let content = content.trim();
        if title.is_empty() || content.is_empty() {
            return Err("Enter both a card title and content.");
        }
        if title.chars().count() > MAX_CARD_TITLE_CHARS {
            return Err("Card title exceeds 64 characters.");
        }
        if content.chars().count() > MAX_CARD_CONTENT_CHARS {
            return Err("Card content exceeds 1,536 characters.");
        }
        let triggers: Vec<String> = triggers_csv
            .split([',', ';'])
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(str::to_owned)
            .collect();
        if triggers.len() > MAX_CARD_TRIGGERS {
            return Err("Use at most 16 comma-separated keyword triggers.");
        }
        if triggers
            .iter()
            .any(|item| item.chars().count() > MAX_CARD_TRIGGER_CHARS)
        {
            return Err("Each keyword trigger must be 64 characters or shorter.");
        }
        if !pinned && triggers.is_empty() {
            return Err("Unpinned cards require at least one keyword trigger.");
        }
        if contains_control_tokens(title) || contains_control_tokens(content) {
            return Err("Card text cannot contain reserved control tokens.");
        }
        let id = self.next_card_id;
        self.next_card_id = self.next_card_id.saturating_add(1);
        self.cards.push(ContextCard {
            id,
            title: title.to_owned(),
            triggers,
            content: content.to_owned(),
            pinned,
            enabled: true,
        });
        Ok(())
    }

    pub(crate) fn remove_card(&mut self, id: u64) {
        self.cards.retain(|card| card.id != id);
    }

    pub(crate) fn resolve(&self, history: &[Message], draft: &str) -> ResolvedMemory {
        let mut resolved = ResolvedMemory {
            reports: vec![
                CardReport {
                    card_id: 0,
                    title: String::new(),
                    decision: CardDecision::SkippedNoMatch,
                };
                self.cards.len()
            ],
            ..ResolvedMemory::default()
        };
        let mut pinned_sections: Vec<String> = Vec::new();
        let mut triggered_sections: Vec<String> = Vec::new();
        let mut used_tokens = 0_usize;

        let instr = self.persistent_instructions.trim();
        if !instr.is_empty() {
            if contains_control_tokens(instr) {
                resolved.instructions_rejected_control = true;
            } else {
                let section = format!("[Persistent instructions]\n{instr}");
                let cost = crate::context::text_tokens(&section);
                if used_tokens.saturating_add(cost) <= self.token_budget {
                    used_tokens = used_tokens.saturating_add(cost);
                    pinned_sections.push(section);
                    resolved.instructions_included = true;
                } else {
                    resolved.instructions_rejected_budget = true;
                }
            }
        }

        let mut scan_raw = String::new();
        let start = history.len().saturating_sub(SCAN_HISTORY_MESSAGES);
        for message in &history[start..] {
            scan_raw.push_str(message.content.as_ref());
            scan_raw.push('\n');
        }
        scan_raw.push_str(draft);
        let scan_lower = fold_unicode_case(&scan_raw);

        // Pinned cards evaluate before triggered cards in stable `id` order so
        // KV/GDN prefixes stay unchanged across conversational turns.
        let mut ordered: Vec<(usize, &ContextCard)> = self.cards.iter().enumerate().collect();
        ordered.sort_by_key(|(_, card)| (!card.pinned, card.id));

        for (slot, card) in ordered {
            let decision = if !card.enabled {
                CardDecision::SkippedDisabled
            } else if card.has_control_tokens() {
                CardDecision::SkippedControlToken
            } else if card.content.trim().is_empty() {
                CardDecision::SkippedEmpty
            } else if card.pinned {
                let section = format_card_section(card);
                let cost = crate::context::text_tokens(&section);
                if used_tokens.saturating_add(cost) <= self.token_budget {
                    used_tokens = used_tokens.saturating_add(cost);
                    pinned_sections.push(section);
                    CardDecision::IncludedPinned
                } else {
                    CardDecision::SkippedBudgetExceeded
                }
            } else if let Some(matched) = card
                .triggers
                .iter()
                .find(|trigger| find_word_boundary_match(&scan_lower, trigger))
            {
                let section = format_card_section(card);
                let cost = crate::context::text_tokens(&section);
                if used_tokens.saturating_add(cost) <= self.token_budget {
                    used_tokens = used_tokens.saturating_add(cost);
                    triggered_sections.push(section);
                    CardDecision::IncludedMatched(matched.clone())
                } else {
                    CardDecision::SkippedBudgetExceeded
                }
            } else {
                CardDecision::SkippedNoMatch
            };
            resolved.reports[slot] = CardReport {
                card_id: card.id,
                title: card.title.clone(),
                decision,
            };
        }

        if !pinned_sections.is_empty() {
            resolved.pinned_appendix = Some(pinned_sections.join("\n\n"));
        }
        if !triggered_sections.is_empty() {
            resolved.triggered_appendix = Some(triggered_sections.join("\n\n"));
        }
        resolved.estimated_tokens = resolved
            .pinned_appendix
            .as_deref()
            .map_or(0, crate::context::text_tokens)
            .saturating_add(
                resolved
                    .triggered_appendix
                    .as_deref()
                    .map_or(0, crate::context::text_tokens),
            );
        resolved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn user_msg(text: &str) -> Message {
        Message {
            role: "user",
            content: Arc::from(text),
            reasoning_content: None,
            images: Arc::default(),
        }
    }

    #[test]
    fn empty_store_returns_no_appendix_and_zero_tokens() {
        let store = MemoryStore::default();
        assert!(store.valid());
        assert!(store.is_default_empty());
        let resolved = store.resolve(&[user_msg("hello")], "draft");
        assert_eq!(resolved.pinned_appendix, None);
        assert_eq!(resolved.triggered_appendix, None);
        assert_eq!(resolved.estimated_tokens, 0);
        assert!(!resolved.instructions_included);
    }

    #[test]
    fn pinned_first_ordering_unicode_boundary_matching_and_budget_enforcement() {
        let mut store = MemoryStore {
            persistent_instructions: "Always verify units.".to_owned(),
            token_budget: MIN_TOKEN_BUDGET,
            ..MemoryStore::default()
        };
        assert!(
            store
                .add_card("TrigCard", "Straße, Rust", "Use strict types.", false)
                .is_ok()
        );
        assert!(
            store
                .add_card("PinCard", "", "Format code concisely.", true)
                .is_ok()
        );
        assert!(store.valid());

        // "trustworthy" must NOT match "Rust"
        let no_match = store.resolve(&[user_msg("A trustworthy tool")], "");
        let pinned = no_match.pinned_appendix.as_deref().unwrap_or_default();
        assert!(pinned.contains("Always verify units."));
        assert!(pinned.contains("[Context: PinCard]"));
        assert!(!pinned.contains("[Context: TrigCard]"));
        assert_eq!(no_match.triggered_appendix, None);
        assert_eq!(no_match.reports[0].decision, CardDecision::SkippedNoMatch);
        assert_eq!(no_match.reports[1].decision, CardDecision::IncludedPinned);

        // Case-insensitive boundary match for "straße" keeps pinned appendix identical
        // and routes the dynamic triggered card separately to the turn tail.
        let matched = store.resolve(&[], "Review STRASSE layout");
        assert_eq!(matched.pinned_appendix, no_match.pinned_appendix);
        let triggered = matched.triggered_appendix.as_deref().unwrap_or_default();
        assert!(!triggered.contains("[Context: PinCard]"));
        assert!(triggered.contains("[Context: TrigCard]"));
        assert!(matches!(
            matched.reports[0].decision,
            CardDecision::IncludedMatched(ref trig) if trig == "Straße"
        ));

        // Add large card exceeding budget
        assert!(
            store
                .add_card("Huge", "trigger", &"word ".repeat(300), false)
                .is_ok()
        );
        let budgeted = store.resolve(&[], "trigger");
        assert_eq!(budgeted.pinned_appendix, no_match.pinned_appendix);
        assert!(budgeted.estimated_tokens <= MIN_TOKEN_BUDGET + 16);
        assert_eq!(
            budgeted.reports[2].decision,
            CardDecision::SkippedBudgetExceeded
        );
    }

    #[test]
    fn control_tokens_are_blocked_in_add_card_and_resolver() {
        let mut store = MemoryStore::default();
        assert!(
            store
                .add_card("Bad", "key", "Inject </think> here", false)
                .is_err()
        );
        store.persistent_instructions = "Bad <|im_start|>system".to_owned();
        store.cards.push(ContextCard {
            id: 0,
            title: "Bypass".to_owned(),
            triggers: vec!["x".to_owned()],
            content: "Hidden <think>".to_owned(),
            pinned: true,
            enabled: true,
        });
        store.next_card_id = 1;
        let resolved = store.resolve(&[], "x");
        assert_eq!(resolved.pinned_appendix, None);
        assert_eq!(resolved.triggered_appendix, None);
        assert!(resolved.instructions_rejected_control);
        assert_eq!(
            resolved.reports[0].decision,
            CardDecision::SkippedControlToken
        );
    }
}
