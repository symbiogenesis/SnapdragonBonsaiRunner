//! Focused orchestration checks with synthetic completions and no model process.

use super::*;
use crate::client::{CognitiveOptions, ReasoningEffort, serialize_request_with_adapters};
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
    let original = serialize_request_with_adapters(&pending, 1024, false, None)?;
    let options = ResponseOptions {
        max_completion_tokens: 1024,
        enable_thinking: false,
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
            serialize_request_with_adapters(&pending, 1024, false, None)?,
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
        assert!(request.get("abliteration_scale").is_none());
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

#[test]
fn plan_draft_review_auto_continues_truncated_draft_and_review() -> Result<()> {
    let (_commands, incoming) = mpsc::channel();
    let (events, outgoing) = crate::client::event_channel();
    let pending = task();
    let options = ResponseOptions {
        max_completion_tokens: 4096,
        enable_thinking: false,
        ..ResponseOptions::default()
    };
    let mut step = 0_usize;
    let (answer, completion) = run_with_stream_hooks(
        &pending,
        &options,
        true,
        65_536,
        PromptReuse::Disabled,
        &incoming,
        &events,
        |_req, _stop, on_text, _| {
            step += 1;
            match step {
                1 => {
                    on_text(Cow::Borrowed("Plan: outline."))?;
                    Ok((
                        message("assistant", "Plan: outline."),
                        Completion::Finished,
                        None,
                    ))
                }
                2 => {
                    on_text(Cow::Borrowed("Draft part one."))?;
                    Ok((
                        message("assistant", "Draft part one."),
                        Completion::TokenLimit,
                        None,
                    ))
                }
                3 => {
                    on_text(Cow::Borrowed(" Draft part two."))?;
                    Ok((
                        message("assistant", " Draft part two."),
                        Completion::Finished,
                        None,
                    ))
                }
                4 => {
                    on_text(Cow::Borrowed("Revised part one."))?;
                    Ok((
                        message("assistant", "Revised part one."),
                        Completion::TokenLimit,
                        None,
                    ))
                }
                _ => {
                    on_text(Cow::Borrowed(" Revised part two."))?;
                    Ok((
                        message("assistant", " Revised part two."),
                        Completion::Finished,
                        None,
                    ))
                }
            }
        },
        |_commands, _ms| Ok(()),
    )?;
    assert_eq!(step, 5);
    assert_eq!(completion, Completion::Finished);
    assert_eq!(
        answer.content.as_ref(),
        "Revised part one. Revised part two."
    );
    let notifications: Vec<_> = outgoing.try_iter().collect();
    assert!(notifications.iter().any(|event| {
        matches!(event, Event::Draft(text) if text == "Draft part one. Draft part two.")
    }));

    // Plan truncation still stops cleanly without continuing.
    let (_commands2, incoming2) = mpsc::channel();
    let (events2, _outgoing2) = crate::client::event_channel();
    let mut step2 = 0_usize;
    let plan_truncated = run_with_stream_hooks(
        &pending,
        &options,
        true,
        65_536,
        PromptReuse::Disabled,
        &incoming2,
        &events2,
        |_req, _stop, on_text, _| {
            step2 += 1;
            on_text(Cow::Borrowed("Partial plan"))?;
            Ok((
                message("assistant", "Partial plan"),
                Completion::TokenLimit,
                None,
            ))
        },
        |_commands, _ms| Ok(()),
    );
    assert_eq!(step2, 1);
    assert!(matches!(plan_truncated, Err(Failure::Rejected(_))));
    Ok(())
}

#[test]
fn rsa_multi_draft_auto_continues_draft_a_and_synthesis() -> Result<()> {
    let (_commands, incoming) = mpsc::channel();
    let (events, outgoing) = crate::client::event_channel();
    let pending = task();
    let options = ResponseOptions {
        max_completion_tokens: 2048,
        enable_thinking: false,
        cognitive_controls: true,
        cognitive: CognitiveOptions {
            extra_effort_strategy: ExtraEffortStrategy::MultiDraftSynthesis,
            ..CognitiveOptions::default()
        },
        ..ResponseOptions::default()
    };
    let mut step = 0_usize;
    let (answer, completion) = run_with_stream_hooks(
        &pending,
        &options,
        true,
        65_536,
        PromptReuse::Latest,
        &incoming,
        &events,
        |_req, _stop, on_text, _| {
            step += 1;
            match step {
                1 => {
                    on_text(Cow::Borrowed("Draft A opening."))?;
                    Ok((
                        message("assistant", "Draft A opening."),
                        Completion::TokenLimit,
                        Some(ConfidenceSummary {
                            tokens: 64,
                            mean_logprob: -0.2,
                            min_window_logprob: Some(-0.4),
                            collapsed: false,
                        }),
                    ))
                }
                2 => {
                    on_text(Cow::Borrowed(" Draft A ending."))?;
                    Ok((
                        message("assistant", " Draft A ending."),
                        Completion::Finished,
                        None,
                    ))
                }
                3 => {
                    on_text(Cow::Borrowed("Draft B complete."))?;
                    Ok((
                        message("assistant", "Draft B complete."),
                        Completion::Finished,
                        Some(ConfidenceSummary {
                            tokens: 64,
                            mean_logprob: -0.9,
                            min_window_logprob: Some(-1.1),
                            collapsed: false,
                        }),
                    ))
                }
                4 => {
                    on_text(Cow::Borrowed("Synthesis part one."))?;
                    Ok((
                        message("assistant", "Synthesis part one."),
                        Completion::TokenLimit,
                        None,
                    ))
                }
                _ => {
                    on_text(Cow::Borrowed(" Synthesis part two."))?;
                    Ok((
                        message("assistant", " Synthesis part two."),
                        Completion::Finished,
                        None,
                    ))
                }
            }
        },
        |_commands, _ms| Ok(()),
    )?;
    assert_eq!(step, 5);
    assert_eq!(completion, Completion::Finished);
    assert_eq!(
        answer.content.as_ref(),
        "Synthesis part one. Synthesis part two."
    );
    let notifications: Vec<_> = outgoing.try_iter().collect();
    // Draft A's initial confidence (-0.2 > -0.9) is preserved across its continuation.
    assert!(notifications.iter().any(|event| {
        matches!(event, Event::Draft(text) if text == "Draft A opening. Draft A ending.")
    }));
    Ok(())
}

#[test]
fn effort_auto_continue_false_and_rsa_draft_b_never_continue() -> Result<()> {
    let pending = task();
    // 1. PlanDraftReview with auto_continue = false stops Draft at 1 call on TokenLimit
    let (_c1, inc1) = mpsc::channel();
    let (ev1, out1) = crate::client::event_channel();
    let mut s1 = 0_usize;
    let (m1, comp1) = run_with_stream_hooks(
        &pending,
        &ResponseOptions {
            max_completion_tokens: 1024,
            enable_thinking: false,
            ..ResponseOptions::default()
        },
        false,
        65_536,
        PromptReuse::Latest,
        &inc1,
        &ev1,
        |_req, _stop, on_text, _| {
            s1 += 1;
            if s1 == 1 {
                on_text(Cow::Borrowed("Plan."))?;
                Ok((message("assistant", "Plan."), Completion::Finished, None))
            } else {
                on_text(Cow::Borrowed("Draft truncated."))?;
                Ok((
                    message("assistant", "Draft truncated."),
                    Completion::TokenLimit,
                    None,
                ))
            }
        },
        |_cmds, _ms| Ok(()),
    )?;
    assert_eq!(s1, 2);
    assert_eq!(comp1, Completion::TokenLimit);
    assert_eq!(m1.content.as_ref(), "Draft truncated.");
    assert!(!out1.try_iter().any(|e| matches!(e, Event::Continuing(_))));

    // 2. MultiDraftSynthesis with auto_continue = false stops Draft A at 1 call on TokenLimit
    let rsa_opts = ResponseOptions {
        max_completion_tokens: 1024,
        enable_thinking: false,
        cognitive_controls: true,
        cognitive: CognitiveOptions {
            extra_effort_strategy: ExtraEffortStrategy::MultiDraftSynthesis,
            ..CognitiveOptions::default()
        },
        ..ResponseOptions::default()
    };
    let (_c2, inc2) = mpsc::channel();
    let (ev2, out2) = crate::client::event_channel();
    let mut s2 = 0_usize;
    let (m2, comp2) = run_with_stream_hooks(
        &pending,
        &rsa_opts,
        false,
        65_536,
        PromptReuse::Latest,
        &inc2,
        &ev2,
        |_req, _stop, on_text, _| {
            s2 += 1;
            on_text(Cow::Borrowed("Draft A partial."))?;
            Ok((
                message("assistant", "Draft A partial."),
                Completion::TokenLimit,
                None,
            ))
        },
        |_cmds, _ms| Ok(()),
    )?;
    assert_eq!(s2, 1);
    assert_eq!(comp2, Completion::TokenLimit);
    assert_eq!(m2.content.as_ref(), "Draft A partial.");
    assert!(!out2.try_iter().any(|e| matches!(e, Event::Continuing(_))));

    // 3. MultiDraftSynthesis with auto_continue = true STILL never auto-continues Draft B
    let (_c3, inc3) = mpsc::channel();
    let (ev3, out3) = crate::client::event_channel();
    let mut s3 = 0_usize;
    let (m3, comp3) = run_with_stream_hooks(
        &pending,
        &rsa_opts,
        true,
        65_536,
        PromptReuse::Latest,
        &inc3,
        &ev3,
        |_req, _stop, on_text, _| {
            s3 += 1;
            match s3 {
                1 => {
                    on_text(Cow::Borrowed("Draft A1."))?;
                    Ok((
                        message("assistant", "Draft A1."),
                        Completion::TokenLimit,
                        None,
                    ))
                }
                2 => {
                    on_text(Cow::Borrowed(" Draft A2."))?;
                    Ok((
                        message("assistant", " Draft A2."),
                        Completion::Finished,
                        None,
                    ))
                }
                _ => {
                    on_text(Cow::Borrowed("Draft B truncated"))?;
                    Ok((
                        message("assistant", "Draft B truncated"),
                        Completion::TokenLimit,
                        None,
                    ))
                }
            }
        },
        |_cmds, _ms| Ok(()),
    )?;
    assert_eq!(s3, 3);
    assert_eq!(comp3, Completion::Finished);
    assert_eq!(m3.content.as_ref(), "Draft A1. Draft A2.");
    let events3: Vec<_> = out3.try_iter().collect();
    assert!(
        events3
            .iter()
            .any(|e| matches!(e, Event::Stage("Second draft skipped; Draft A retained")))
    );
    assert!(
        events3
            .iter()
            .any(|e| matches!(e, Event::RetainedAnswer(t) if t == "Draft A1. Draft A2."))
    );
    Ok(())
}

#[test]
fn effort_stage_fallbacks_when_continuation_exhausts_at_token_limit() -> Result<()> {
    let pending = task();
    let opts = ResponseOptions {
        max_completion_tokens: 1024,
        enable_thinking: false,
        ..ResponseOptions::default()
    };

    // 1. Plan-Draft-Review: Draft continuation stops at TokenLimit (via repetition) -> review skipped
    let (_c1, inc1) = mpsc::channel();
    let (ev1, out1) = crate::client::event_channel();
    let mut s1 = 0_usize;
    let repeated =
        "Repeated draft clause that triggers loop detection cleanly without finishing. ".repeat(2);
    let (m1, comp1) = run_with_stream_hooks(
        &pending,
        &opts,
        true,
        65_536,
        PromptReuse::Disabled,
        &inc1,
        &ev1,
        |_req, _stop, on_text, _| {
            s1 += 1;
            if s1 == 1 {
                on_text(Cow::Borrowed("Plan."))?;
                Ok((message("assistant", "Plan."), Completion::Finished, None))
            } else {
                on_text(Cow::Borrowed(&repeated))?;
                Ok((
                    message("assistant", &repeated),
                    Completion::TokenLimit,
                    None,
                ))
            }
        },
        |_cmds, _ms| Ok(()),
    )?;
    assert_eq!(comp1, Completion::TokenLimit);
    assert_eq!(m1.content.as_ref(), repeated);
    let e1: Vec<_> = out1.try_iter().collect();
    assert!(
        e1.iter()
            .any(|e| matches!(e, Event::Stage("Draft incomplete; review skipped")))
    );
    assert!(
        e1.iter()
            .any(|e| matches!(e, Event::RetainedAnswer(t) if t == &repeated))
    );

    // 2. Plan-Draft-Review: Review continuation stops at TokenLimit -> stitched Draft retained
    let (_c2, inc2) = mpsc::channel();
    let (ev2, out2) = crate::client::event_channel();
    let mut s2 = 0_usize;
    let (m2, comp2) = run_with_stream_hooks(
        &pending,
        &opts,
        true,
        65_536,
        PromptReuse::Disabled,
        &inc2,
        &ev2,
        |_req, _stop, on_text, _| {
            s2 += 1;
            match s2 {
                1 => {
                    on_text(Cow::Borrowed("Plan."))?;
                    Ok((message("assistant", "Plan."), Completion::Finished, None))
                }
                2 => {
                    on_text(Cow::Borrowed("Draft A."))?;
                    Ok((
                        message("assistant", "Draft A."),
                        Completion::TokenLimit,
                        None,
                    ))
                }
                3 => {
                    on_text(Cow::Borrowed(" Draft B."))?;
                    Ok((
                        message("assistant", " Draft B."),
                        Completion::Finished,
                        None,
                    ))
                }
                _ => {
                    on_text(Cow::Borrowed(&repeated))?;
                    Ok((
                        message("assistant", &repeated),
                        Completion::TokenLimit,
                        None,
                    ))
                }
            }
        },
        |_cmds, _ms| Ok(()),
    )?;
    assert_eq!(comp2, Completion::TokenLimit);
    assert_eq!(m2.content.as_ref(), "Draft A. Draft B.");
    let e2: Vec<_> = out2.try_iter().collect();
    assert!(e2.iter().any(|e| matches!(
        e,
        Event::Stage("Revision incomplete; complete draft retained")
    )));
    assert!(
        e2.iter()
            .any(|e| matches!(e, Event::RetainedAnswer(t) if t == "Draft A. Draft B."))
    );

    // 3. RSA Draft A continuation stops at TokenLimit -> Draft B & Synthesis skipped
    let rsa_opts = ResponseOptions {
        cognitive_controls: true,
        cognitive: CognitiveOptions {
            extra_effort_strategy: ExtraEffortStrategy::MultiDraftSynthesis,
            ..CognitiveOptions::default()
        },
        ..opts.clone()
    };
    let (_c3, inc3) = mpsc::channel();
    let (ev3, out3) = crate::client::event_channel();
    let (m3, comp3) = run_with_stream_hooks(
        &pending,
        &rsa_opts,
        true,
        65_536,
        PromptReuse::Disabled,
        &inc3,
        &ev3,
        |_req, _stop, on_text, _| {
            on_text(Cow::Borrowed(&repeated))?;
            Ok((
                message("assistant", &repeated),
                Completion::TokenLimit,
                None,
            ))
        },
        |_cmds, _ms| Ok(()),
    )?;
    assert_eq!(comp3, Completion::TokenLimit);
    assert_eq!(m3.content.as_ref(), repeated);
    let e3: Vec<_> = out3.try_iter().collect();
    assert!(e3.iter().any(|e| {
        matches!(
            e,
            Event::Stage("First draft reached output limit; complete answer retained")
        )
    }));

    // 4. RSA Synthesis continuation stops at TokenLimit -> best complete draft retained
    let (_c4, inc4) = mpsc::channel();
    let (ev4, out4) = crate::client::event_channel();
    let mut s4 = 0_usize;
    let (m4, comp4) = run_with_stream_hooks(
        &pending,
        &rsa_opts,
        true,
        65_536,
        PromptReuse::Disabled,
        &inc4,
        &ev4,
        |_req, _stop, on_text, _| {
            s4 += 1;
            match s4 {
                1 => {
                    on_text(Cow::Borrowed("Weaker A."))?;
                    Ok((
                        message("assistant", "Weaker A."),
                        Completion::Finished,
                        Some(ConfidenceSummary {
                            tokens: 64,
                            mean_logprob: -1.2,
                            min_window_logprob: None,
                            collapsed: false,
                        }),
                    ))
                }
                2 => {
                    on_text(Cow::Borrowed("Stronger B."))?;
                    Ok((
                        message("assistant", "Stronger B."),
                        Completion::Finished,
                        Some(ConfidenceSummary {
                            tokens: 64,
                            mean_logprob: -0.1,
                            min_window_logprob: None,
                            collapsed: false,
                        }),
                    ))
                }
                _ => {
                    on_text(Cow::Borrowed(&repeated))?;
                    Ok((
                        message("assistant", &repeated),
                        Completion::TokenLimit,
                        None,
                    ))
                }
            }
        },
        |_cmds, _ms| Ok(()),
    )?;
    assert_eq!(comp4, Completion::TokenLimit);
    assert_eq!(m4.content.as_ref(), "Stronger B.");
    let e4: Vec<_> = out4.try_iter().collect();
    assert!(e4.iter().any(|e| matches!(
        e,
        Event::Stage("Synthesis incomplete; best complete draft retained")
    )));
    assert!(
        e4.iter()
            .any(|e| matches!(e, Event::RetainedAnswer(t) if t == "Stronger B."))
    );
    Ok(())
}

#[test]
fn effort_continuation_wire_prefix_seam_thinking_stop_and_size_bounds() -> Result<()> {
    let pending = task();
    // 1. Wire-request parity under PromptReuse::Latest, seam overlap, Continuing ordering,
    //    thinking handoff (reasoning_budget=1), stop_on_collapse=false on continuation,
    //    and full stitched Draft/Draft A forwarded into Review/Synthesis.
    let mut opts = ResponseOptions {
        max_completion_tokens: 2048,
        enable_thinking: true,
        cognitive_controls: true,
        cognitive: CognitiveOptions {
            reasoning_effort: ReasoningEffort::Low,
            extra_effort_strategy: ExtraEffortStrategy::PlanDraftReview,
            ..CognitiveOptions::default()
        },
        ..ResponseOptions::default()
    };
    let (_c1, inc1) = mpsc::channel();
    let (ev1, out1) = crate::client::event_channel();
    let mut requests = Vec::new();
    let mut stop_flags = Vec::new();
    let mut step = 0_usize;
    let (answer, comp) = run_with_stream_hooks(
        &pending,
        &opts,
        true,
        65_536,
        PromptReuse::Latest,
        &inc1,
        &ev1,
        |req, stop_on_collapse, on_text, on_reasoning| {
            step += 1;
            let json: serde_json::Value =
                serde_json::from_slice(&req).map_err(|_| Failure::Message("invalid JSON"))?;
            requests.push(json);
            stop_flags.push(stop_on_collapse);
            match step {
                1 => {
                    on_text(Cow::Borrowed("Plan: concise outline."))?;
                    Ok((
                        message("assistant", "Plan: concise outline."),
                        Completion::Finished,
                        None,
                    ))
                }
                2 => {
                    on_reasoning(Cow::Borrowed("Draft initial reasoning"))?;
                    on_text(Cow::Borrowed("Draft ends at boundary."))?;
                    Ok((
                        Message {
                            role: "assistant",
                            content: "Draft ends at boundary.".into(),
                            reasoning_content: Some("Draft initial reasoning".into()),
                            images: std::sync::Arc::default(),
                        },
                        Completion::TokenLimit,
                        None,
                    ))
                }
                3 => {
                    on_reasoning(Cow::Borrowed("Suppressed part2 thought"))?;
                    on_text(Cow::Borrowed("at boundary. Draft finishes."))?;
                    Ok((
                        message("assistant", "at boundary. Draft finishes."),
                        Completion::Finished,
                        None,
                    ))
                }
                _ => {
                    on_text(Cow::Borrowed("Reviewed final."))?;
                    Ok((
                        message("assistant", "Reviewed final."),
                        Completion::Finished,
                        None,
                    ))
                }
            }
        },
        |_cmds, _ms| Ok(()),
    )?;
    assert_eq!(comp, Completion::Finished);
    assert_eq!(answer.content.as_ref(), "Reviewed final.");
    assert!(stop_flags.iter().all(|&f| !f));
    // Draft continuation (requests[2]) preserves requests[1] prefix and sets reasoning_budget=1
    let d0 = requests[1]["messages"]
        .as_array()
        .ok_or_else(|| Failure::Message("missing draft req0"))?;
    let d1 = requests[2]["messages"]
        .as_array()
        .ok_or_else(|| Failure::Message("missing draft req1"))?;
    assert_eq!(d1.len(), d0.len() + 2);
    assert_eq!(&d1[..d0.len()], &d0[..]);
    assert_eq!(requests[2]["enable_thinking"], true);
    assert_eq!(requests[2]["reasoning_budget"], 1);
    // Review (requests[3]) receives the seam-cleaned stitched Draft
    assert!(
        requests[3]["messages"][3]["content"]
            .as_str()
            .is_some_and(|s| s.contains("Draft ends at boundary. Draft finishes."))
    );
    // Verify Event::Continuing(2) occurs after Event::Stage(DRAFT) and before Draft part-2 Event::Text
    let evs1: Vec<_> = out1.try_iter().collect();
    let draft_stage_idx = evs1
        .iter()
        .position(|e| matches!(e, Event::Stage(DRAFT)))
        .ok_or_else(|| Failure::Message("missing DRAFT stage"))?;
    let cont_idx = evs1
        .iter()
        .position(|e| matches!(e, Event::Continuing(2)))
        .ok_or_else(|| Failure::Message("missing Continuing(2)"))?;
    let part2_idx = evs1
        .iter()
        .position(|e| matches!(e, Event::Text(t) if t.contains("Draft finishes.")))
        .ok_or_else(|| Failure::Message("missing part2 text"))?;
    assert!(draft_stage_idx < cont_idx && cont_idx < part2_idx);
    assert!(
        !evs1
            .iter()
            .any(|e| matches!(e, Event::Reasoning(r) if r.contains("Suppressed part2 thought")))
    );

    // 2. Publisher rollback / PromptReuse::Disabled disables thinking on stage continuation
    opts.cognitive_controls = false;
    let (_c_pub, inc_pub) = mpsc::channel();
    let (ev_pub, _out_pub) = crate::client::event_channel();
    let mut pub_reqs = Vec::new();
    let mut pub_step = 0_usize;
    let (_m_pub, comp_pub) = run_with_stream_hooks(
        &pending,
        &opts,
        true,
        65_536,
        PromptReuse::Disabled,
        &inc_pub,
        &ev_pub,
        |req, _stop, on_text, _| {
            pub_step += 1;
            pub_reqs.push(
                serde_json::from_slice::<serde_json::Value>(&req)
                    .map_err(|_| Failure::Message("invalid JSON"))?,
            );
            match pub_step {
                1 => {
                    on_text(Cow::Borrowed("Plan."))?;
                    Ok((message("assistant", "Plan."), Completion::Finished, None))
                }
                2 => {
                    on_text(Cow::Borrowed("Draft 1."))?;
                    Ok((
                        message("assistant", "Draft 1."),
                        Completion::TokenLimit,
                        None,
                    ))
                }
                3 => {
                    on_text(Cow::Borrowed(" Draft 2."))?;
                    Ok((
                        message("assistant", " Draft 2."),
                        Completion::Finished,
                        None,
                    ))
                }
                _ => {
                    on_text(Cow::Borrowed("Review."))?;
                    Ok((message("assistant", "Review."), Completion::Finished, None))
                }
            }
        },
        |_cmds, _ms| Ok(()),
    )?;
    assert_eq!(comp_pub, Completion::Finished);
    assert_eq!(pub_reqs[2]["enable_thinking"], false);
    assert!(pub_reqs[2].get("reasoning_budget").is_none());

    // 3. RSA Draft B stop_on_collapse=true vs continuation stop_on_collapse=false & Draft A payload
    let rsa_opts = ResponseOptions {
        cognitive_controls: true,
        reset_on_cancel: true,
        cognitive: CognitiveOptions {
            extra_effort_strategy: ExtraEffortStrategy::MultiDraftSynthesis,
            ..CognitiveOptions::default()
        },
        ..ResponseOptions {
            max_completion_tokens: 2048,
            enable_thinking: false,
            ..ResponseOptions::default()
        }
    };
    let (_c_rsa, inc_rsa) = mpsc::channel();
    let (ev_rsa, _out_rsa) = crate::client::event_channel();
    let mut rsa_reqs = Vec::new();
    let mut rsa_stops = Vec::new();
    let mut rsa_step = 0_usize;
    let (_m_rsa, comp_rsa) = run_with_stream_hooks(
        &pending,
        &rsa_opts,
        true,
        65_536,
        PromptReuse::Latest,
        &inc_rsa,
        &ev_rsa,
        |req, stop_on_collapse, on_text, _| {
            rsa_step += 1;
            rsa_reqs.push(
                serde_json::from_slice::<serde_json::Value>(&req)
                    .map_err(|_| Failure::Message("invalid JSON"))?,
            );
            rsa_stops.push(stop_on_collapse);
            match rsa_step {
                1 => {
                    on_text(Cow::Borrowed("A1."))?;
                    Ok((message("assistant", "A1."), Completion::TokenLimit, None))
                }
                2 => {
                    on_text(Cow::Borrowed(" A2."))?;
                    Ok((message("assistant", " A2."), Completion::Finished, None))
                }
                3 => {
                    on_text(Cow::Borrowed("B1."))?;
                    Ok((message("assistant", "B1."), Completion::Finished, None))
                }
                _ => {
                    on_text(Cow::Borrowed("Synth."))?;
                    Ok((message("assistant", "Synth."), Completion::Finished, None))
                }
            }
        },
        |_cmds, _ms| Ok(()),
    )?;
    assert_eq!(comp_rsa, Completion::Finished);
    assert_eq!(rsa_stops, vec![false, false, true, false]);
    assert!(
        rsa_reqs[3]["messages"][3]["content"]
            .as_str()
            .is_some_and(|s| s.contains("A1. A2."))
    );

    // 4. Command::Stop mid-continuation inside Draft, Review, and RSA Synthesis aborts cleanly
    for stop_at in [2_usize, 3_usize] {
        let (tx, rx) = mpsc::channel();
        let (ev, _out) = crate::client::event_channel();
        let mut s = 0_usize;
        let res = run_with_stream_hooks(
            &pending,
            &ResponseOptions {
                max_completion_tokens: 1024,
                enable_thinking: false,
                ..ResponseOptions::default()
            },
            true,
            65_536,
            PromptReuse::Disabled,
            &rx,
            &ev,
            |_req, _stop, on_text, _| {
                s += 1;
                if s == stop_at {
                    let _ = tx.send(Command::Stop);
                    on_text(Cow::Borrowed("Part."))?;
                    Ok((message("assistant", "Part."), Completion::TokenLimit, None))
                } else {
                    on_text(Cow::Borrowed("Ok."))?;
                    Ok((message("assistant", "Ok."), Completion::Finished, None))
                }
            },
            |_cmds, _ms| Ok(()),
        );
        assert_eq!(res.err(), Some(Failure::Stopped));
        assert_eq!(s, stop_at);
    }

    let (tx_s, rx_s) = mpsc::channel();
    let (ev_s, _out_s) = crate::client::event_channel();
    let mut ss = 0_usize;
    let res_s = run_with_stream_hooks(
        &pending,
        &rsa_opts,
        true,
        65_536,
        PromptReuse::Disabled,
        &rx_s,
        &ev_s,
        |_req, _stop, on_text, _| {
            ss += 1;
            if ss == 3 {
                let _ = tx_s.send(Command::Stop);
                on_text(Cow::Borrowed("Synth part."))?;
                Ok((
                    message("assistant", "Synth part."),
                    Completion::TokenLimit,
                    None,
                ))
            } else {
                on_text(Cow::Borrowed("Done."))?;
                Ok((message("assistant", "Done."), Completion::Finished, None))
            }
        },
        |_cmds, _ms| Ok(()),
    );
    assert_eq!(res_s.err(), Some(Failure::Stopped));
    assert_eq!(ss, 3);

    // 5. Stitched Draft exceeding MAX_REQUEST when escaped in Review's stage request
    //    fails cleanly with request_byte_limit() after Event::Draft was emitted.
    let (_c_lim, inc_lim) = mpsc::channel();
    let (ev_lim, out_lim) = crate::client::event_channel();
    let big_escaped_draft = "\\".repeat(300 * 1024);
    let mut lim_step = 0_usize;
    let res_lim = run_with_stream_hooks(
        &pending,
        &ResponseOptions {
            max_completion_tokens: 1024,
            enable_thinking: false,
            ..ResponseOptions::default()
        },
        true,
        2_097_152,
        PromptReuse::Disabled,
        &inc_lim,
        &ev_lim,
        |_req, _stop, on_text, _| {
            lim_step += 1;
            match lim_step {
                1 => {
                    on_text(Cow::Borrowed("Plan."))?;
                    Ok((message("assistant", "Plan."), Completion::Finished, None))
                }
                2 => {
                    on_text(Cow::Borrowed(&big_escaped_draft))?;
                    Ok((
                        message("assistant", &big_escaped_draft),
                        Completion::TokenLimit,
                        None,
                    ))
                }
                _ => {
                    on_text(Cow::Borrowed("Tail"))?;
                    Ok((message("assistant", "Tail"), Completion::Finished, None))
                }
            }
        },
        |_cmds, _ms| Ok(()),
    );
    assert_eq!(res_lim.err(), Some(request_byte_limit()));
    assert_eq!(lim_step, 3);
    assert!(
        out_lim
            .try_iter()
            .any(|e| matches!(e, Event::Draft(d) if d.len() == big_escaped_draft.len() + 4))
    );
    Ok(())
}
