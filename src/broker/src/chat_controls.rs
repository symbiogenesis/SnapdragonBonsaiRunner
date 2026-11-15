//! Reject exact pinned token controls before caller text reaches chat rendering.
//!
//! The inventory is independently extracted from the original Swift GGUF's
//! `tokenizer.ggml.tokens` and `tokenizer.ggml.token_type` arrays. Its raw
//! tokenizer metadata digest is bound by the fixture below. The native Swift
//! renderer parses these type-3/type-4 spellings after template rendering.
//! Never remove text: deletion can expose another marker or change counting.

#[cfg(test)]
const SPELLINGS: [&str; 33] = [
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
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<|audio_pad|>",
];

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

/// No allocation or normalization; O(1) second-byte dispatch at each `<` byte.
#[expect(
    clippy::redundant_pub_crate,
    reason = "Only the parent request policy uses this private module; a public export violates unreachable_pub."
)]
pub(super) fn contains(text: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use super::{PIPE_SPELLINGS, SLASH_SPELLINGS, SPELLINGS, T_SPELLINGS, contains};

    #[test]
    fn inventory_matches_locked_model_and_tokenizer_identity() -> Result<(), serde_json::Error> {
        let inventory: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/swift-control-tokens.json"))?;
        let metadata: serde_json::Value =
            serde_json::from_str(include_str!("../../../config/swift.metadata.json"))?;
        let lock: serde_json::Value =
            serde_json::from_str(include_str!("../../../config/swift-assessment.lock.json"))?;
        assert_eq!(inventory["model_repository"], lock["model"]["repository"]);
        assert_eq!(inventory["model_revision"], lock["model"]["revision"]);
        assert_eq!(inventory["model_sha256"], lock["model"]["sha256"]);
        assert_eq!(inventory["model_bytes"], lock["model"]["bytes"]);
        assert_eq!(
            inventory["tokenizer_metadata_sha256"],
            metadata["TokenizerMetadataSha256"]
        );
        let actual: Vec<_> = inventory["entries"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|entry| entry["spelling"].as_str())
            .collect();
        let expected: Vec<_> = SPELLINGS.iter().copied().map(Some).collect();
        assert_eq!(actual, expected);
        assert!(
            inventory["entries"]
                .as_array()
                .into_iter()
                .flatten()
                .all(|entry| matches!(entry["kind"].as_u64(), Some(3 | 4)))
        );
        Ok(())
    }

    #[test]
    fn second_byte_switch_matches_complete_spelling_inventory() {
        let mut switched: Vec<&str> = PIPE_SPELLINGS
            .iter()
            .chain(SLASH_SPELLINGS)
            .chain(T_SPELLINGS)
            .copied()
            .collect();
        let mut expected = SPELLINGS.to_vec();
        switched.sort_unstable();
        expected.sort_unstable();
        assert_eq!(switched, expected);
        for spelling in SPELLINGS {
            assert!(contains(spelling), "missing spelling: {spelling}");
        }
        for safe in ["", "<", "<a", "<div", "</div", "<ta", "<|other|>", "2 < 3"] {
            assert!(!contains(safe), "unexpected control match: {safe}");
        }
    }
}
