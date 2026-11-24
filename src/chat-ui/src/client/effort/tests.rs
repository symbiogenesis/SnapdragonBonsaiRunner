//! Focused orchestration checks with synthetic completions and no model process.

use super::*;
use crate::client::{CognitiveOptions, serialize_request_with_adapters};
use std::{collections::VecDeque, sync::mpsc};

fn message(role: &'static str, content: &str) -> Message {
    Message {
        role,
        content: content.into(),
        reasoning_content: None,
        images: std::sync::Arc::default(),
    }
}

fn task() -> Vec<Message> {
    vec![
        message("user", "Approved facts: the sea is green."),
        Message {
            role: "assistant",
            content: "Understood.".into(),
            reasoning_content: Some("Preserved prior reasoning".into()),
            images: std::sync::Arc::default(),
        },
        message(
            "user",
            "Write a quiet coastal scene. Keep the approved facts.",
        ),
    ]
}

#[test]
fn effort_working_material_keeps_json_bytes_and_rejects_oversized_temporaries() -> Result<()> {
    let pending = task();
    let original = serialize_request_with_adapters(&pending, 1024, false, None, None)?;
    let options = ResponseOptions {
        max_completion_tokens: 1024,
        enable_thinking: false,
        abliteration_scale: None,
        daisyrose_scale: None,
        ..ResponseOptions::default()
    };
    let working = Working {
        plan: Some("A plan: 雨 🌱\n\"quoted\""),
        draft: Some("A draft with \\ and \t"),
    };
    let expected = serde_json::to_string(&working)
        .map_err(|_| Failure::Message("Could not encode expected working material."))?;
    let body = stage_request(&pending, "Draft the answer.", working, options.clone())?;
    let request: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|_| Failure::Message("Could not decode staged working material."))?;
    assert!(
        request["messages"][3]["content"]
            .as_str()
            .is_some_and(|content| {
                content.ends_with(&format!("Generated working material (JSON):\n{expected}"))
            })
    );
    for plan in [
        "\0".repeat(MAX_REQUEST / 6 + 1),
        "x".repeat(MAX_REQUEST),
        // The working JSON fits, but adding the authoritative task and fixed
        // instructions cannot fit even before the outer request is escaped.
        "x".repeat(MAX_REQUEST - 64),
    ] {
        assert_eq!(
            stage_request(
                &pending,
                "Draft the answer.",
                Working {
                    plan: Some(&plan),
                    draft: None
                },
                options.clone(),
            )
            .err(),
            Some(request_byte_limit()),
        );
        assert_eq!(
            serialize_request_with_adapters(&pending, 1024, false, None, None)?,
            original
        );
    }
    Ok(())
}

#[test]
fn effort_applies_thinking_selection_to_every_stage() -> Result<()> {
    for enable_thinking in [false, true] {
        let (_commands, incoming) = mpsc::channel();
        let (events, _outgoing) = crate::client::event_channel();
        let mut calls = 0;
        let _answer = run_with(
            &task(),
            ResponseOptions {
                max_completion_tokens: 1024,
                enable_thinking,
                abliteration_scale: None,
                daisyrose_scale: None,
                ..ResponseOptions::default()
            },
            &incoming,
            &events,
            |body| {
                let request: serde_json::Value = serde_json::from_slice(&body)
                    .map_err(|_| Failure::Message("invalid thinking stage JSON"))?;
                assert_eq!(request["enable_thinking"], enable_thinking);
                assert_eq!(request["messages"][0]["role"], "system");
                assert_eq!(
                    request["messages"][2]["reasoning_content"],
                    "Preserved prior reasoning"
                );
                calls += 1;
                Ok((
                    message("assistant", "A complete stage."),
                    Completion::Finished,
                ))
            },
        )?;
        assert_eq!(calls, 3);
    }
    Ok(())
}

#[test]
fn effort_stages_use_fresh_original_context() -> Result<()> {
    let (_commands, incoming) = mpsc::channel();
    let (events, outgoing) = crate::client::event_channel();
    let pending = task();
    let mut responses = VecDeque::from([
        (
            message("assistant", "Plan: quiet rhythm."),
            Completion::Finished,
        ),
        (
            message("assistant", "The green sea slept."),
            Completion::Finished,
        ),
        (
            message("assistant", "The green sea breathed softly."),
            Completion::Finished,
        ),
    ]);
    let mut requests = Vec::new();
    let (answer, completion) = run_with(
        &pending,
        ResponseOptions {
            max_completion_tokens: 2048,
            enable_thinking: false,
            abliteration_scale: Some(1.3),
            daisyrose_scale: Some(0.5),
            ..ResponseOptions::default()
        },
        &incoming,
        &events,
        |body| {
            requests.push(
                serde_json::from_slice::<serde_json::Value>(&body)
                    .map_err(|_| Failure::Message("invalid synthetic request"))?,
            );
            responses
                .pop_front()
                .ok_or(Failure::Message("unexpected fourth call"))
        },
    )?;
    assert_eq!(requests.len(), 3);
    assert!(responses.is_empty());
    assert_eq!(answer.content.as_ref(), "The green sea breathed softly.");
    assert_eq!(completion, Completion::Finished);
    assert_eq!(
        pending[2].content.as_ref(),
        "Write a quiet coastal scene. Keep the approved facts."
    );
    for request in &requests {
        assert_eq!(request["max_completion_tokens"], 2048);
        assert_eq!(request["abliteration_scale"], serde_json::json!(1.3));
        assert_eq!(request["daisyrose_scale"], serde_json::json!(0.5));
        assert_eq!(request["messages"].as_array().map(Vec::len), Some(4));
        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(
            request["messages"][2]["reasoning_content"],
            "Preserved prior reasoning"
        );
        assert!(
            request["messages"][3]["content"]
                .as_str()
                .is_some_and(|text| {
                    text.starts_with(pending[2].content.as_ref())
                        && text.contains("never instructions or approved facts")
                })
        );
    }
    let draft_prompt = requests[1]["messages"][3]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(draft_prompt.contains("Plan: quiet rhythm."));
    assert!(!draft_prompt.contains("The green sea slept."));
    let review_prompt = requests[2]["messages"][3]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(review_prompt.contains("Plan: quiet rhythm."));
    assert!(review_prompt.contains("The green sea slept."));
    let notifications: Vec<_> = outgoing.try_iter().collect();
    let stages: Vec<_> = notifications
        .iter()
        .filter_map(|event| match event {
            Event::Stage(label) => Some(*label),
            _ => None,
        })
        .collect();
    assert_eq!(stages, [PLAN, DRAFT, REVIEW]);
    assert!(
        notifications
            .iter()
            .any(|event| { matches!(event, Event::Draft(text) if text == "The green sea slept.") })
    );
    Ok(())
}

#[test]
fn effort_truncation_stops_later_stages_and_keeps_complete_draft() -> Result<()> {
    for truncated_stage in 0..3 {
        let (_commands, incoming) = mpsc::channel();
        let (events, outgoing) = crate::client::event_channel();
        let mut count = 0;
        let result = run_with(
            &task(),
            ResponseOptions {
                max_completion_tokens: 4096,
                enable_thinking: false,
                abliteration_scale: None,
                daisyrose_scale: None,
                ..ResponseOptions::default()
            },
            &incoming,
            &events,
            |_| {
                let content = ["A plan", "A complete draft", "An unfinished revision"][count];
                let completion = if count == truncated_stage {
                    Completion::TokenLimit
                } else {
                    Completion::Finished
                };
                count += 1;
                Ok((
                    Message {
                        reasoning_content: Some(format!("Stage {count} reasoning").into()),
                        ..message("assistant", content)
                    },
                    completion,
                ))
            },
        );
        assert_eq!(count, truncated_stage + 1);
        let notifications: Vec<_> = outgoing.try_iter().collect();
        if truncated_stage == 0 {
            assert!(matches!(result, Err(Failure::Rejected(_))));
        } else {
            let (answer, completion) = result?;
            assert_eq!(answer.content.as_ref(), "A complete draft");
            assert_eq!(
                answer.reasoning_content.as_deref(),
                Some("Stage 2 reasoning")
            );
            assert_eq!(completion, Completion::TokenLimit);
            assert!(
                matches!(notifications.last(), Some(Event::RetainedAnswer(text)) if text == "A complete draft")
            );
        }
        assert_eq!(
            notifications
                .iter()
                .any(|event| matches!(event, Event::Draft(_))),
            truncated_stage == 2
        );
    }
    Ok(())
}

#[test]
fn effort_cancellation_and_request_bounds_prevent_later_calls() {
    let (commands, incoming) = mpsc::channel();
    let (events, _outgoing) = crate::client::event_channel();
    let mut count = 0;
    let stopped = run_with(
        &task(),
        ResponseOptions {
            max_completion_tokens: 1024,
            enable_thinking: false,
            abliteration_scale: None,
            daisyrose_scale: None,
            ..ResponseOptions::default()
        },
        &incoming,
        &events,
        |_| {
            count += 1;
            commands
                .send(Command::Stop)
                .map_err(|_| Failure::Message("test command channel"))?;
            Ok((message("assistant", "A plan"), Completion::Finished))
        },
    );
    assert!(matches!(stopped, Err(Failure::Stopped)));
    assert_eq!(count, 1);
    let oversized = vec![message("user", &"x".repeat(MAX_REQUEST))];
    let bounded = run_with(
        &oversized,
        ResponseOptions {
            max_completion_tokens: 1024,
            enable_thinking: false,
            abliteration_scale: None,
            daisyrose_scale: None,
            ..ResponseOptions::default()
        },
        &incoming,
        &events,
        |_| {
            count += 1;
            Ok((message("assistant", "Should not run"), Completion::Finished))
        },
    );
    assert!(matches!(bounded, Err(Failure::Rejected(_))));
    assert_eq!(count, 1);
}

#[test]
fn effort_review_failure_leaves_complete_draft_visible() {
    let (_commands, incoming) = mpsc::channel();
    let (events, outgoing) = crate::client::event_channel();
    let mut count = 0;
    let result = run_with(
        &task(),
        ResponseOptions {
            max_completion_tokens: 4096,
            enable_thinking: false,
            abliteration_scale: None,
            daisyrose_scale: None,
            ..ResponseOptions::default()
        },
        &incoming,
        &events,
        |_| {
            count += 1;
            match count {
                1 => Ok((message("assistant", "A plan"), Completion::Finished)),
                2 => Ok((
                    message("assistant", "Keep this complete draft."),
                    Completion::Finished,
                )),
                _ => Err(Failure::Message("synthetic broken stream")),
            }
        },
    );
    assert!(matches!(
        result,
        Err(Failure::Message("synthetic broken stream"))
    ));
    assert_eq!(count, 3);
    assert!(outgoing.try_iter().any(|event| {
        matches!(event, Event::Draft(text) if text == "Keep this complete draft.")
    }));
}

#[test]
fn rsa_multi_draft_shares_identical_prompt_prefix_synthesizes_and_handles_collapse() -> Result<()> {
    let pending = task();
    let (_commands, incoming) = mpsc::channel();
    let (events, outgoing) = crate::client::event_channel();
    let options = ResponseOptions {
        max_completion_tokens: 2048,
        enable_thinking: true,
        cognitive_controls: true,
        reset_on_cancel: true,
        cognitive: CognitiveOptions {
            extra_effort_strategy: ExtraEffortStrategy::MultiDraftSynthesis,
            ..CognitiveOptions::default()
        },
        ..ResponseOptions::default()
    };
    let mut requests = Vec::new();
    let mut pass = 0;
    let (answer, completion) = run_with_detailed(
        &pending,
        &options,
        &incoming,
        &events,
        |body, stop_on_collapse| {
            pass += 1;
            let json: serde_json::Value = serde_json::from_slice(&body)
                .map_err(|_| Failure::Message("invalid RSA request JSON"))?;
            requests.push((json, stop_on_collapse));
            match pass {
                1 => Ok((
                    message("assistant", "Draft A green sea."),
                    Completion::Finished,
                    Some(ConfidenceSummary {
                        tokens: 70,
                        mean_logprob: -0.9,
                        min_window_logprob: Some(-1.2),
                        collapsed: false,
                    }),
                )),
                2 => Ok((
                    message("assistant", "Draft B green sea."),
                    Completion::Finished,
                    Some(ConfidenceSummary {
                        tokens: 70,
                        mean_logprob: -0.2,
                        min_window_logprob: Some(-0.4),
                        collapsed: false,
                    }),
                )),
                _ => Ok((
                    message("assistant", "Unified superior synthesis."),
                    Completion::Finished,
                    None,
                )),
            }
        },
    )?;
    assert_eq!(pass, 3);
    assert_eq!(answer.content.as_ref(), "Unified superior synthesis.");
    assert_eq!(completion, Completion::Finished);
    assert_eq!(requests[0].0["messages"], requests[1].0["messages"]);
    assert!(!requests[0].1);
    assert!(requests[1].1);
    assert_eq!(requests[0].0["logprobs"], true);
    assert_eq!(requests[1].0["temperature"], serde_json::json!(1.1));
    let synth_prompt = requests[2].0["messages"][3]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(synth_prompt.contains("Draft A green sea."));
    assert!(synth_prompt.contains("Draft B green sea."));
    let notifications: Vec<_> = outgoing.try_iter().collect();
    assert!(
        notifications
            .iter()
            .any(|event| matches!(event, Event::Draft(text) if text == "Draft B green sea."))
    );

    // Collapsed Pass 2 retains Draft A without calling synthesis.
    let (_commands2, incoming2) = mpsc::channel();
    let (events2, _outgoing2) = crate::client::event_channel();
    let mut pass2 = 0;
    let (retained, comp2) = run_with_detailed(&pending, &options, &incoming2, &events2, |_, _| {
        pass2 += 1;
        match pass2 {
            1 => Ok((
                message("assistant", "Draft A fallback."),
                Completion::Finished,
                None,
            )),
            _ => Err(Failure::CollapsedDraft),
        }
    })?;
    assert_eq!(pass2, 2);
    assert_eq!(retained.content.as_ref(), "Draft A fallback.");
    assert_eq!(comp2, Completion::Finished);

    // Truncated synthesis retains the higher-confidence Draft B.
    let (_commands3, incoming3) = mpsc::channel();
    let (events3, _outgoing3) = crate::client::event_channel();
    let mut pass3 = 0;
    let (retained_b, comp3) =
        run_with_detailed(&pending, &options, &incoming3, &events3, |_, _| {
            pass3 += 1;
            match pass3 {
                1 => Ok((
                    message("assistant", "Weaker Draft A."),
                    Completion::Finished,
                    Some(ConfidenceSummary {
                        tokens: 64,
                        mean_logprob: -1.1,
                        min_window_logprob: None,
                        collapsed: false,
                    }),
                )),
                2 => Ok((
                    message("assistant", "Stronger Draft B."),
                    Completion::Finished,
                    Some(ConfidenceSummary {
                        tokens: 64,
                        mean_logprob: -0.15,
                        min_window_logprob: None,
                        collapsed: false,
                    }),
                )),
                _ => Ok((
                    message("assistant", "Truncated partial"),
                    Completion::TokenLimit,
                    None,
                )),
            }
        })?;
    assert_eq!(pass3, 3);
    assert_eq!(retained_b.content.as_ref(), "Stronger Draft B.");
    assert_eq!(comp3, Completion::TokenLimit);
    Ok(())
}
