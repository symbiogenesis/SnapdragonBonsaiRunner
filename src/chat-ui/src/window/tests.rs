//! Headless presentation regressions. No model, permission, capture or playback.
use super::*;

fn state() -> (State, Receiver<Command>, client::EventSender) {
    let (commands, incoming) = mpsc::channel();
    let (events, incoming_events) = client::event_channel();
    let mut state = State::create(commands, incoming_events, None, None);
    state.preview = false;
    state.model = ModelLifecycle::Preparing;
    (state, incoming, events)
}

#[test]
fn recording_preparation_requires_the_current_successful_capture_start() -> io::Result<()> {
    let (mut state, requests, _events) = state();
    let (session, commands) = speech::Session::fixture()?;
    session.set_input_generation(Some(7));
    state.speech = Some(session);
    state.microphone_generation = Some(7);
    state.dictation_requested = true;
    for message in [
        "Requesting permission",
        "Initializing microphone",
        "Starting recording",
    ] {
        state.microphone_event(microphone::Event::Status(message));
        assert!(commands.try_recv().is_err());
    }
    state.microphone_event(microphone::Event::Recording { generation: 6 });
    assert!(commands.try_recv().is_err());
    state.microphone_event(microphone::Event::Recording { generation: 7 });
    assert!(matches!(
        commands.recv_timeout(Duration::from_secs(2)),
        Ok(speech::Command::PrepareInput { generation: 7 })
    ));
    assert!(commands.try_recv().is_err());
    assert!(requests.try_recv().is_err());
    assert!(!state.dictation_finishing);
    assert!(state.pending_voice_transcript.is_none());
    state.microphone_event(microphone::Event::Failed(
        "Microphone permission was revoked",
    ));
    assert!(matches!(
        commands.recv_timeout(Duration::from_secs(2)),
        Ok(speech::Command::StopListening)
    ));
    state.microphone_event(microphone::Event::Recording { generation: 7 });
    assert!(commands.try_recv().is_err());
    assert!(!state.dictation_requested);
    assert!(requests.try_recv().is_err());
    Ok(())
}

#[test]
fn updater_install_admission_requires_idle_or_confirmed_stopped_model() {
    let (mut state, _commands, _events) = state();
    for phase in [
        ModelLifecycle::Unavailable,
        ModelLifecycle::Preparing,
        ModelLifecycle::Restarting,
        ModelLifecycle::Generating,
        ModelLifecycle::AwaitingTeardown,
        ModelLifecycle::Stopping,
    ] {
        state.model = phase;
        assert!(!state.can_install_update());
    }
    for phase in [ModelLifecycle::Ready, ModelLifecycle::Retryable] {
        state.model = phase;
        assert!(state.can_install_update());
    }
    assert!(state.updates.is_none());
}

#[test]
fn model_notifications_have_a_finite_per_frame_work_budget() {
    let (mut state, _commands, events) = state();
    for _ in 0..80 {
        assert!(events.send(Event::Stage("Queued stage")).is_ok());
    }
    state.drain_events();
    assert_eq!(state.events.try_iter().count(), 16);
}

#[test]
fn application_messages_never_enter_the_conversation() {
    let (mut state, _commands, _events) = state();
    state.model_event(Event::Starting);
    state.model_event(Event::Failed(
        "Startup failed. Protected credentials and technical evidence: C:\\private".to_owned(),
    ));
    state.model = ModelLifecycle::Generating;
    state.model_event(Event::Rejected {
        message: "Prompt exceeds the context limit.".to_owned(),
        text: "Keep this message".to_owned(),
        retry_required: false,
    });
    state.speech_event(speech::Event::Capabilities {
        input: false,
        output: false,
    });
    assert!(state.transcript.messages.is_empty());
    assert!(!state.diagnostics.is_empty());
    assert!(!state.status.contains("C:\\private"));
    assert_eq!(state.input, "Keep this message");
    assert!(state.model.ready());
    assert!(!state.model.waiting_model());
}

#[test]
fn stopped_empty_response_leaves_only_conversation_content() {
    let (mut state, commands, _events) = state();
    state.context_tokens = Some(131_072);
    state.model = ModelLifecycle::Ready;
    state.input = "Keep my question".to_owned();
    state.submit();
    assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
    state.stop_generation();
    assert!(matches!(commands.try_recv(), Ok(Command::Stop)));
    assert!(state.model.waiting_model());
    assert_eq!(state.transcript.messages.len(), 1);
    assert_eq!(state.transcript.plain_text(), "You\nKeep my question");
    state.model_event(Event::Stopping("Keep my question".to_owned()));
    assert_eq!(state.input, "Keep my question");
    assert!(!state.warning);
    assert!(!state.model.retryable());
    state.model_event(Event::Stopped);
    assert!(!state.model.waiting_model());
    assert_eq!(state.model, ModelLifecycle::Ready);
    assert!(!state.model.retryable());
    assert_eq!(state.context_tokens, Some(131_072));
    assert!(!state.warning);
    assert!(state.active_alerts().is_empty());
    assert_eq!(state.transcript.messages.len(), 1);
}

#[test]
fn stop_freezes_queued_output_and_preserves_a_new_composer_draft() {
    let (mut state, commands, events) = state();
    state.context_tokens = Some(131_072);
    state.model = ModelLifecycle::Ready;
    state.input = "Original question".to_owned();
    state.submit();
    assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
    state.model_event(Event::Text("Partial answer".to_owned()));
    state.stop_generation();
    state.stop_generation();
    assert!(matches!(commands.try_recv(), Ok(Command::Stop)));
    assert!(commands.try_recv().is_err());
    let transcript = state.transcript.plain_text();
    let status = state.status.clone();
    for event in [
        Event::Text(" late text".to_owned()),
        Event::Reasoning("late reasoning".to_owned()),
        Event::Stage("Late stage"),
        Event::Draft("Late draft".to_owned()),
        Event::RetainedAnswer("Late restored draft".to_owned()),
        Event::Complete(Completion::Finished),
    ] {
        assert!(events.send(event).is_ok());
    }
    state.drain_events();
    assert_eq!(state.transcript.plain_text(), transcript);
    assert_eq!(state.status, status);
    assert!(state.model.stopping() && state.model.waiting_model());
    assert!(!state.model.ready() && !state.model.retryable() && !state.warning);
    state.input = "My next question".to_owned();
    state.model_event(Event::Stopping("Original question".to_owned()));
    assert_eq!(state.input, "My next question");
    state.model_event(Event::Stopped);
    assert_eq!(state.model, ModelLifecycle::Ready);
    assert_eq!(state.context_tokens, Some(131_072));
    assert!(!state.warning);
    assert!(state.active_alerts().is_empty());
    assert!(matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.len() == 2));
    state.submit();
    assert!(
        matches!(commands.try_recv(), Ok(Command::Send { text, .. }) if text == "My next question")
    );
    // Publisher rollback automatically restarts a fresh warm attempt on Stop
    // without surfacing a model alert or requiring manual Retry.
    state.stop_generation();
    assert!(matches!(commands.try_recv(), Ok(Command::Stop)));
    state.model_event(Event::Stopping("My next question".to_owned()));
    state.model_event(Event::Restarting);
    assert_eq!(state.model, ModelLifecycle::Restarting);
    assert!(!state.warning);
    assert!(state.active_alerts().is_empty());
    state.model_event(Event::Starting);
    state.model_event(Event::ContextCapacity(131_072));
    state.model_event(Event::Ready);
    assert_eq!(state.model, ModelLifecycle::Ready);
    assert!(!state.warning);
}

#[test]
fn queued_rejections_after_stop_preserve_teardown_and_composer_drafts() {
    for retry_required in [false, true] {
        for newer_draft in ["", "My next question"] {
            let (mut state, commands, events) = state();
            state.model = ModelLifecycle::Ready;
            state.input = "Original question".to_owned();
            state.submit();
            assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
            state.model_event(Event::Text("Partial answer".to_owned()));
            state.stop_generation();
            assert!(matches!(commands.try_recv(), Ok(Command::Stop)));
            newer_draft.clone_into(&mut state.input);
            let transcript = state.transcript.plain_text();
            let status = state.status.clone();
            assert!(
                events
                    .send(Event::Rejected {
                        message: "Queued rejection".to_owned(),
                        text: "Original question".to_owned(),
                        retry_required,
                    })
                    .is_ok()
            );

            state.drain_events();
            assert!(state.model.waiting_model());
            assert_eq!(
                state.model,
                if retry_required {
                    ModelLifecycle::AwaitingTeardown
                } else {
                    ModelLifecycle::Stopping
                }
            );
            assert!(
                !state.model.ready()
                    && !state.model.busy()
                    && !state.model.retryable()
                    && !state.warning
            );
            assert_eq!(state.status, status);
            assert_eq!(state.transcript.plain_text(), transcript);
            assert_eq!(
                state.input,
                if newer_draft.is_empty() && !retry_required {
                    "Original question"
                } else {
                    newer_draft
                }
            );
            state.submit();
            assert!(commands.try_recv().is_err());

            state.model_event(Event::Stopped);
            if retry_required {
                assert_eq!(state.model, ModelLifecycle::AwaitingTeardown);
                state.model_event(Event::Retryable("Cleanup complete".to_owned()));
                assert!(
                    state.model.retryable()
                        && !state.model.ready()
                        && !state.model.waiting_model()
                        && state.warning
                );
                state.retry();
                assert!(matches!(commands.try_recv(), Ok(Command::Retry)));
                assert!(commands.try_recv().is_err());
            } else {
                assert!(
                    state.model.ready()
                        && !state.model.retryable()
                        && !state.model.waiting_model()
                        && !state.warning
                );
                assert!(state.active_alerts().is_empty());
            }
        }
    }
}

#[test]
fn queued_rejection_after_stop_retains_completed_extra_effort_draft() {
    for retry_required in [false, true] {
        let (mut state, commands, events) = state();
        state.model = ModelLifecycle::Ready;
        state.input = "Original question".to_owned();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        state.model_event(Event::Text("A complete draft.".to_owned()));
        state.model_event(Event::Draft("A complete draft.".to_owned()));
        state.model_event(Event::Stage("Review"));
        state.model_event(Event::Text("An unfinished revision".to_owned()));
        state.stop_generation();
        assert!(matches!(commands.try_recv(), Ok(Command::Stop)));
        let transcript = state.transcript.plain_text();
        let status = state.status.clone();
        assert!(
            events
                .send(Event::Rejected {
                    message: "Review failed".to_owned(),
                    text: "Original question".to_owned(),
                    retry_required,
                })
                .is_ok()
        );

        state.drain_events();
        assert_eq!(state.status, status);
        state.model_event(Event::Stopped);
        if retry_required {
            assert_eq!(state.model, ModelLifecycle::AwaitingTeardown);
            state.model_event(Event::Retryable("Cleanup complete".to_owned()));
        }
        assert_eq!(
            state.last_answer,
            if retry_required {
                "An unfinished revision"
            } else {
                "A complete draft."
            }
        );
        assert_eq!(state.answer_complete, !retry_required);
        assert_eq!(state.transcript.plain_text(), transcript);
        assert_eq!(
            state.input,
            if retry_required {
                ""
            } else {
                "Original question"
            }
        );
        assert_eq!(state.committed.len(), if retry_required { 2 } else { 0 });
        if retry_required {
            assert!(state.model.retryable() && !state.model.ready() && state.warning);
        } else {
            assert!(state.model.ready() && !state.model.retryable() && !state.warning);
            assert_eq!(state.status, client::STOPPED_MESSAGE);
        }
    }
}

#[test]
fn completed_turn_winning_stop_race_keeps_visible_and_committed_answers_consistent() {
    let (mut state, _commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.input = "A question".to_owned();
    state.submit();
    state.model_event(Event::Text("Partial".to_owned()));
    state.stop_generation();
    let status = state.status.clone();
    state.model_event(Event::Committed(vec![
        client::Message {
            role: "user",
            content: "A question".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Complete answer".into(),
            images: std::sync::Arc::default(),
            reasoning_content: Some("Complete reasoning".into()),
        },
    ]));
    state.model_event(Event::Complete(Completion::Finished));
    assert_eq!(state.transcript.messages.len(), 2);
    assert!(state.transcript.plain_text().ends_with("Complete answer"));
    assert_eq!(state.committed[1].content.as_ref(), state.last_answer);
    assert!(state.answer_complete);
    assert_eq!(state.status, status);
    assert!(!state.model.ready() && !state.model.retryable() && !state.warning);
    state.model_event(Event::Stopped);
    assert_eq!(state.model, ModelLifecycle::Ready);
    assert!(!state.warning);
    assert!(state.active_alerts().is_empty());
}

#[test]
fn response_controls_preserve_limits_recover_composer_and_retain_complete_draft() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.input = "A story about a garden".to_owned();
    state.extra_effort = true;
    state.output_budget = 1;
    state.submit();
    let command = commands.try_recv();
    assert!(matches!(
        command,
        Ok(Command::Send {
            max_completion_tokens: 1024,
            extra_effort: true,
            ..
        })
    ));
    assert!(state.model.busy());
    assert_eq!(state.input, "");
    assert_eq!(state.transcript.messages.len(), 2);
    state.model_event(Event::Stage("Draft"));
    state.model_event(Event::Text("A complete draft.".to_owned()));
    state.model_event(Event::Draft("A complete draft.".to_owned()));
    state.model_event(Event::Stage("Review"));
    state.model_event(Event::Text("Partial revision".to_owned()));
    let count = state.transcript.messages.len();
    state.model_event(Event::Rejected {
        message: "Review could not finish.".to_owned(),
        text: "A story about a garden".to_owned(),
        retry_required: true,
    });
    assert!(!state.model.ready());
    assert!(!state.model.busy());
    assert!(!state.answer_complete);
    assert_eq!(state.last_answer, "Partial revision");
    assert_eq!(state.transcript.messages.len(), count);
    assert_eq!(state.input, "");
    state.model_event(Event::Retryable("Retry the local model.".to_owned()));
    state.retry();
    assert!(matches!(commands.try_recv(), Ok(Command::Retry)));
    assert!(!state.model.retryable());
    assert_eq!(state.transcript.messages.len(), count);
}

#[test]
fn speech_controls_stay_visible_and_explain_unconfigured_services_without_listening() {
    let (mut state, _commands, _events) = state();
    assert!(!state.dictation_requested);
    assert!(state.microphone.is_none());
    assert!(state.microphone_help().contains("automatically"));
    assert!(state.read_aloud_help().contains("automatically"));
    for configured in [false, true] {
        state.tts_configured = configured;
        for checked in [false, true] {
            state.speech_checked = checked;
            let help = state.read_aloud_help();
            for retired in [
                "Start its service",
                "key file",
                "configured port",
                "bonsai-tools",
                "README.md",
                "--no-tts",
            ] {
                assert!(!help.contains(retired));
            }
        }
    }
    state.tts_configured = false;
    state.toggle_streaming_read_aloud();
    assert!(state.show_settings);
    assert!(state.help.is_none());
    assert!(!state.streaming_read_aloud);
    state.help = None;
    state.toggle_dictation();
    assert!(state.help.is_some());
    assert!(!state.dictation_requested);
    assert!(state.microphone_generation.is_none());
    state.help = None;
    state
        .transcript
        .push(Role::Assistant, "A completed answer".to_owned());
    state.transcript.remove_empty_response();
    let message_id = state.transcript.messages[0].id;
    state.toggle_read_aloud(message_id);
    assert!(state.show_settings);
    assert!(state.help.is_none());
    assert!(!state.speaking);
    assert_eq!(state.transcript.messages.len(), 1);
}

#[test]
fn speech_service_diagnostics_survive_later_status_updates() {
    let (mut state, _commands, _events) = state();
    let detail = "Speech stopped; exit code 1; retained evidence C:/Speech/service".to_owned();
    state.speech_event(speech::Event::Diagnostic(detail.clone()));
    state.speech_event(speech::Event::Capabilities {
        input: false,
        output: false,
    });
    assert!(state.diagnostics.iter().any(|message| message == &detail));
    assert!(!state.speaking);
    assert!(!state.dictation_requested);
}

#[test]
fn failed_owned_speech_capabilities_offer_setup_retry_and_preserve_the_draft() -> io::Result<()> {
    use egui_kittest::{
        Harness,
        kittest::{NodeT as _, Queryable as _},
    };
    use std::ffi::OsString;
    let scratch = std::env::temp_dir().join(format!(
        "Bonsai-SpeechCapabilityRetry-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos()
    ));
    std::fs::create_dir(&scratch)?;
    let tool = scratch.join("fixture-tool.exe");
    let payload = scratch.join("request.json");
    std::fs::write(&tool, [])?;
    std::fs::write(&payload, r#"{"PreparationStage":"all"}"#)?;
    let request = crate::startup::Request::parse(&[
        OsString::from("--startup-tool"),
        tool.into_os_string(),
        OsString::from("--startup-request"),
        payload.into_os_string(),
        OsString::from("--startup-data-directory"),
        scratch.clone().into_os_string(),
    ])?;
    let (mut state, _commands, _events) = state();
    state.input = "Keep this draft".into();
    state.model = ModelLifecycle::Ready;
    state.tts_configured = true;
    state.setup.request = Some(request.clone());
    state.setup.broker = Some(scratch.join("broker.exe"));
    state.speech_event(speech::Event::Capabilities {
        input: false,
        output: false,
    });
    assert!(
        state.setup.speech_error.is_none(),
        "External services do not invoke app-owned setup"
    );
    state.tts_managed = true;
    state.startup_request = Some(request);
    state.speech_event(speech::Event::Capabilities {
        input: false,
        output: false,
    });
    assert!(
        state.setup.speech_error.is_none(),
        "Preparation in progress is not a failure"
    );
    state.startup_request.take();
    state.speech_event(speech::Event::Capabilities {
        input: false,
        output: false,
    });
    assert!(
        state
            .setup
            .speech_error
            .as_deref()
            .is_some_and(|message| message.contains("Retry speech setup"))
    );
    assert_eq!(state.input, "Keep this draft");
    assert!(state.model.ready());
    state.open_voice_setup();
    let mut harness = Harness::builder()
        .with_size([1000.0, 800.0])
        .with_context_setup(configure)
        .build_ui_state(|ui, app: &mut State| app.settings(ui.ctx()), state);
    harness.run();
    assert!(
        !harness
            .get_by_role_and_label(egui::Role::Button, "Retry speech setup")
            .accesskit_node()
            .is_disabled()
    );
    harness.state_mut().setup.speech_error =
        Some("Setup process cleanup could not be confirmed.".into());
    harness.run();
    assert!(
        harness
            .get_by_role_and_label(egui::Role::Button, "Retry speech setup")
            .accesskit_node()
            .is_disabled()
    );
    assert_eq!(harness.state().input, "Keep this draft");
    std::fs::remove_file(scratch.join("fixture-tool.exe"))?;
    std::fs::remove_file(scratch.join("request.json"))?;
    std::fs::remove_dir(scratch)?;
    Ok(())
}

#[test]
fn response_speakers_select_individual_answers_and_ignore_replaced_playback_events()
-> io::Result<()> {
    let (mut state, _commands, _events) = state();
    let (session, speech_commands) = speech::Session::fixture()?;
    state.speech = Some(session);
    state.speech_output = true;
    state.model = ModelLifecycle::Ready;
    state
        .transcript
        .push(Role::User, "Do not speak my prompt".to_owned());
    state
        .transcript
        .push(Role::Assistant, "The older answer".to_owned());
    state.transcript.append("Private reasoning", true);
    state.transcript.remove_empty_response();
    state
        .transcript
        .push(Role::Assistant, "The latest answer".to_owned());
    state.transcript.remove_empty_response();
    let user = state.transcript.messages[0].id;
    let older = state.transcript.messages[1].id;
    let latest = state.transcript.messages[2].id;
    state.last_answer = "Unrelated last-answer state".to_owned();
    state.answer_complete = false;
    state.toggle_read_aloud(user);
    assert!(!state.speaking);
    state.toggle_read_aloud(older);
    assert_eq!(state.speaking_message, Some(older));
    assert!(matches!(
        speech_commands.recv_timeout(Duration::from_secs(2)),
        Ok(speech::Command::StopListening)
    ));
    assert!(matches!(
        speech_commands.recv_timeout(Duration::from_secs(2)),
        Ok(speech::Command::StopSpeaking)
    ));
    assert!(
        matches!(speech_commands.recv_timeout(Duration::from_secs(2)), Ok(speech::Command::Speak { text, voice_description })
        if text == "The older answer" && voice_description == DEFAULT_VOICE)
    );
    state.toggle_read_aloud(latest);
    assert_eq!(state.speaking_message, Some(latest));
    assert!(matches!(
        speech_commands.recv_timeout(Duration::from_secs(2)),
        Ok(speech::Command::StopListening)
    ));
    assert!(matches!(
        speech_commands.recv_timeout(Duration::from_secs(2)),
        Ok(speech::Command::StopSpeaking)
    ));
    assert!(
        matches!(speech_commands.recv_timeout(Duration::from_secs(2)), Ok(speech::Command::Speak { text, .. }) if text == "The latest answer")
    );
    state.speech_event(speech::Event::Speaking {
        generation: 2,
        speaking: false,
    });
    assert!(state.speaking);
    assert_eq!(state.speaking_message, Some(latest));
    state.speech_event(speech::Event::Failed {
        generation: 4,
        message: "Speech failed",
    });
    assert!(!state.speaking);
    assert_eq!(state.speaking_message, None);
    state.toggle_read_aloud(older);
    state.toggle_read_aloud(older);
    assert!(!state.speaking);
    assert_eq!(state.speaking_message, None);
    state
        .transcript
        .push(Role::Assistant, "Still streaming".to_owned());
    state.toggle_read_aloud(state.transcript.messages[3].id);
    assert!(!state.speaking);
    state.model = ModelLifecycle::Generating;
    state.toggle_read_aloud(older);
    assert!(state.speaking);
    assert_eq!(state.speaking_message, Some(older));
    state.model = ModelLifecycle::Ready;
    state.toggle_read_aloud(latest);
    state.new_chat();
    assert!(!state.speaking);
    assert_eq!(state.speaking_message, None);
    Ok(())
}

#[test]
fn speech_memory_failure_keeps_text_chat_and_completed_answer_available() -> io::Result<()> {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.input = "A short answer".to_owned();
    state.submit();
    assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
    state.model_event(Event::Text("The completed answer.".to_owned()));
    state.model_event(Event::Complete(Completion::Finished));
    let transcript = state.transcript.plain_text();
    state.speech = Some(speech::start(None, None)?);
    state.speech_output = true;
    state.speaking = true;
    state.speech_event(speech::Event::Failed {
        generation: 0,
        message: "Read aloud stopped to preserve text chat memory.",
    });
    assert!(state.model.ready());
    assert!(state.answer_complete);
    assert!(!state.speaking);
    assert!(!state.model.retryable());
    assert_eq!(state.last_answer, "The completed answer.");
    assert_eq!(state.transcript.plain_text(), transcript);
    assert!(state.speech_status.contains("preserve text chat"));
    assert!(matches!(
        commands.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    Ok(())
}

#[test]
fn dictation_updates_loading_drafts_and_switching_cancels_late_transcripts() {
    for phase in [
        ModelLifecycle::Preparing,
        ModelLifecycle::Unavailable,
        ModelLifecycle::Retryable,
    ] {
        let (mut state, commands, _events) = state();
        state.model = phase;
        state.dictation_requested = true;
        state.dictation_finishing = true;
        state.microphone_generation = Some(7);
        state.accept_transcript(7, "Dictated draft");
        assert_eq!(state.input, "Dictated draft");
        state.submit();
        assert!(commands.try_recv().is_err());
        state.new_chat();
        state.accept_transcript(7, "Late transcript");
        assert_eq!(state.input, "");
        assert!(!state.dictation_requested);
    }
}

#[test]
fn cancelled_dictation_cannot_insert_a_late_transcript() {
    let (mut state, _commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.input = "Existing text".to_owned();
    state.dictation_requested = true;
    state.microphone_generation = Some(7);
    state.cancel_microphone();
    state.accept_transcript(7, "late recognition");
    assert_eq!(state.input, "Existing text");
    assert!(!state.dictation_requested);
    assert!(state.microphone_generation.is_none());
}

#[test]
fn dictation_replaces_the_unicode_selection_and_advances_the_caret() {
    let (mut state, _commands, _events) = state();
    state.input = "café 日本語 🌿 end".to_owned();
    state.composer_cursor = Some(egui::text::CCursorRange::two(
        egui::text::CCursor::new(5),
        egui::text::CCursor::new(9),
    ));
    assert!(state.insert_dictation("こんにちは"));
    assert_eq!(state.input, "café こんにちは 🌿 end");
    assert!(
        state
            .pending_cursor
            .is_some_and(|cursor| usize::from(cursor.primary.index) == 11 && cursor.is_empty())
    );
    assert!(state.insert_dictation("again"));
    assert_eq!(state.input, "café こんにちは again 🌿 end");
    state.input = "界".repeat(INPUT_LIMIT);
    state.place_composer_cursor(INPUT_LIMIT);
    assert!(!state.insert_dictation("overflow"));
    assert_eq!(state.input.chars().count(), INPUT_LIMIT);
    assert!(state.input.chars().all(|character| character == '界'));
}

#[test]
fn diagnostics_are_bounded_and_separate_from_copyable_chat() {
    let (mut state, _commands, _events) = state();
    state.transcript.push(Role::User, "Hello".to_owned());
    state.transcript.push(Role::Assistant, "Welcome".to_owned());
    for index in 0..200 {
        state.record(&format!("Technical event {index}"));
    }
    assert_eq!(state.diagnostics.len(), MAX_DIAGNOSTICS);
    assert_eq!(state.diagnostics_revision, 200);
    state.record("Technical event 199");
    assert_eq!(state.diagnostics_revision, 200);
    assert_eq!(
        state.transcript.plain_text(),
        "You\nHello\n\nSwift Bonsai\nWelcome"
    );
}

#[test]
fn chat_renders_at_minimum_window_size_without_starting_services() {
    let (mut state, _commands, _events) = state();
    state.preview = true;
    state.model = ModelLifecycle::Ready;
    state.sidebar_open = false;
    state.transcript.push(
        Role::User,
        "Unicode stays intact: café 日本語 🌿".to_owned(),
    );
    state.transcript.push(
        Role::Assistant,
        "A readable answer.\nWith a second paragraph.".to_owned(),
    );
    let context = egui::Context::default();
    configure(&context);
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(620.0, 640.0),
        )),
        ..egui::RawInput::default()
    };
    let mut output = context.run_ui(input, |ui| state.draw(ui));
    assert_ne!(output.shapes.len(), 0);
    output.textures_delta.clear();
    assert!(state.worker.is_none());
    assert!(state.microphone.is_none());
    assert!(state.speech.is_none());
}

#[test]
fn conversation_workspace_loading_restores_only_the_selected_history_at_readiness() {
    let (mut state, commands, _events) = state();
    let original = state.active_conversation;
    state.input = "First unsent draft".into();
    state.enable_thinking = true;
    state.output_budget = 1;
    state.committed = vec![
        client::Message {
            role: "user",
            content: "Completed question".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Completed answer".into(),
            images: std::sync::Arc::default(),
            reasoning_content: Some("Retained reasoning".into()),
        },
    ];
    state
        .transcript
        .push(Role::Assistant, "Visible answer".into());
    state.transcript.remove_empty_response();
    let status = state.status.clone();
    state.new_chat();
    let second = state.active_conversation;
    state.input = "Second unsent draft".into();
    state.switch_conversation(original);
    assert_eq!(state.input, "First unsent draft");
    assert!(state.enable_thinking);
    assert_eq!(state.output_budget, 1);
    assert_eq!(
        state.transcript.plain_text(),
        "Swift Bonsai\nVisible answer"
    );
    assert_eq!(state.status, status);
    state.switch_conversation(second);
    assert_eq!(state.input, "Second unsent draft");
    state.switch_conversation(original);
    state.remove_conversation(second);
    state.submit();
    assert!(commands.try_recv().is_err());
    assert!(state.restore_history);
    state.model_event(Event::Ready);
    assert!(
        matches!(commands.try_recv(), Ok(Command::RestoreChat(history))
        if history.len() == 2
            && history[1].content.as_ref() == "Completed answer"
            && history[1].reasoning_content.as_deref() == Some("Retained reasoning"))
    );
    assert!(!state.restore_history);
    assert!(commands.try_recv().is_err());
    state.submit();
    assert!(matches!(commands.try_recv(), Ok(Command::Send { text, .. })
        if text == "First unsent draft"));
}

#[test]
fn conversation_workspace_loading_deletion_clears_recovered_history_before_ready() {
    let (mut state, commands, _events) = state();
    state.restore_history = true;
    state.committed.push(client::Message {
        role: "user",
        content: "Deleted history".into(),
        images: std::sync::Arc::default(),
        reasoning_content: None,
    });
    state.input = "Deleted draft".into();
    let original = state.active_conversation;
    let status = state.status.clone();
    state.remove_conversation(original);
    assert_ne!(state.active_conversation, original);
    assert_eq!(state.conversations.len(), 1);
    assert!(state.input.is_empty() && state.committed.is_empty());
    assert_eq!(state.status, status);
    assert!(commands.try_recv().is_err());
    state.model_event(Event::Ready);
    assert!(matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.is_empty()));
    assert!(commands.try_recv().is_err());
}

#[test]
fn conversation_workspace_remains_local_when_the_model_worker_is_unavailable() {
    for phase in [
        ModelLifecycle::Unavailable,
        ModelLifecycle::Retryable,
        ModelLifecycle::Ready,
    ] {
        let (mut state, commands, _events) = state();
        drop(commands);
        state.model = phase;
        state.input = "Keep this draft".into();
        let original = state.active_conversation;
        state.new_chat();
        let second = state.active_conversation;
        assert_ne!(second, original);
        state.switch_conversation(original);
        assert_eq!(state.input, "Keep this draft");
        state.remove_conversation(second);
        state.remove_conversation(original);
        assert_eq!(state.conversations.len(), 1);
        assert_ne!(state.active_conversation, original);
        assert!(!state.model.ready());
        assert!(state.restore_history);
    }
}

#[test]
fn conversation_workspace_late_rejection_restores_only_its_original_conversation() {
    for delete_original in [false, true] {
        let (mut state, commands, _events) = state();
        state.model_event(Event::Ready);
        let original = state.active_conversation;
        state.input = "Original prompt".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        state.model_event(Event::Retryable("Model stopped".into()));
        state.new_chat();
        let second = state.active_conversation;
        if delete_original {
            state.remove_conversation(original);
        }
        state.model_event(Event::Rejected {
            message: "Queued prompt could not run".into(),
            text: "Original prompt".into(),
            retry_required: true,
        });
        assert_eq!(state.active_conversation, second);
        assert_eq!(state.input, "");
        assert!(state.transcript.messages.is_empty());
        if !delete_original {
            state.switch_conversation(original);
            assert_eq!(state.input, "Original prompt");
        }
        assert!(commands.try_recv().is_err());
    }
}

#[test]
fn conversation_workspace_preserves_drafts_settings_and_exact_committed_turns() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.input = "A garden story".to_owned();
    state.submit();
    assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
    state.model_event(Event::Text("Complete story".to_owned()));
    state.model_event(Event::Committed(vec![
        client::Message {
            role: "user",
            content: "A garden story".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Complete story".into(),
            images: std::sync::Arc::default(),
            reasoning_content: Some("Private reasoning".into()),
        },
    ]));
    state.model_event(Event::Complete(Completion::Finished));
    state.input = "An unsent revision 🌱".to_owned();
    state.extra_effort = true;
    state.enable_thinking = true;
    state.output_budget = 1;
    let original = state.active_conversation;
    let visible = state.transcript.plain_text();
    state.new_chat();
    assert!(matches!(commands.try_recv(), Ok(Command::NewChat)));
    assert_eq!(state.conversations.len(), 2);
    assert_eq!(state.conversations[0].id, state.active_conversation);
    assert_eq!(state.conversations[1].title, "A garden story");
    assert!(state.transcript.messages.is_empty());
    assert!(state.committed.is_empty());
    assert_eq!(state.input, "");
    assert!(!state.extra_effort);
    assert!(!state.enable_thinking);
    state.input = "Second conversation draft".to_owned();
    let second = state.active_conversation;
    state.switch_conversation(original);
    let restored = commands.try_recv();
    assert!(
        matches!(&restored, Ok(Command::RestoreChat(history)) if history.len() == 2 && history[1].reasoning_content.as_deref() == Some("Private reasoning"))
    );
    assert_eq!(state.transcript.plain_text(), visible);
    assert_eq!(state.input, "An unsent revision 🌱");
    assert!(state.extra_effort);
    assert!(state.enable_thinking);
    assert_eq!(state.output_budget, 1);
    state.switch_conversation(second);
    assert!(matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.is_empty()));
    assert_eq!(state.input, "Second conversation draft");
    assert!(!state.enable_thinking);
}

#[test]
fn conversation_workspace_restores_interrupted_output_as_followup_context() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.input = "Unfinished request".to_owned();
    state.submit();
    assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
    state.model_event(Event::Text("Partial answer".to_owned()));
    state.model_event(Event::Rejected {
        message: "Interrupted".to_owned(),
        text: "Unfinished request".to_owned(),
        retry_required: true,
    });
    state.model_event(Event::Retryable("Retry required".to_owned()));
    state.retry();
    assert!(matches!(commands.try_recv(), Ok(Command::Retry)));
    state.model_event(Event::Starting);
    state.model_event(Event::Ready);
    assert!(
        matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.len() == 2 && history[1].content.as_ref() == "Partial answer")
    );
    let original = state.active_conversation;
    state.new_chat();
    assert!(matches!(commands.try_recv(), Ok(Command::NewChat)));
    state.switch_conversation(original);
    assert!(
        matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.len() == 2 && history[1].content.as_ref() == "Partial answer")
    );
    assert!(state.transcript.plain_text().contains("Partial answer"));
    assert_eq!(state.input, "");
    assert_eq!(state.committed.len(), 2);
}

#[test]
fn conversation_history_grows_without_a_count_cap_and_preserves_teardown_guards() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    let original = state.active_conversation;
    state.new_chat();
    assert!(matches!(commands.try_recv(), Ok(Command::NewChat)));
    let second = state.active_conversation;
    for phase in [
        ModelLifecycle::Stopping,
        ModelLifecycle::AwaitingTeardown,
        ModelLifecycle::Restarting,
    ] {
        state.model = phase;
        state.switch_conversation(original);
        state.remove_conversation(second);
        state.new_chat();
        assert_eq!(state.active_conversation, second);
        assert_eq!(state.conversations.len(), 2);
        assert!(commands.try_recv().is_err());
    }
    state.model = ModelLifecycle::Ready;
    state.remove_conversation(second);
    assert_eq!(state.active_conversation, original);
    assert_eq!(state.conversations.len(), 1);
    assert!(matches!(commands.try_recv(), Ok(Command::RestoreChat(_))));
    state.remove_conversation(original);
    assert_ne!(state.active_conversation, original);
    assert_eq!(state.conversations.len(), 1);
    assert!(matches!(commands.try_recv(), Ok(Command::NewChat)));
    for _ in 1..128 {
        state.new_chat();
        assert!(matches!(commands.try_recv(), Ok(Command::NewChat)));
    }
    state.new_chat();
    assert_eq!(state.conversations.len(), 129);
    assert!(!state.warning);
    assert!(matches!(commands.try_recv(), Ok(Command::NewChat)));
    let identities: std::collections::HashSet<_> =
        state.conversations.iter().map(|item| item.id).collect();
    assert_eq!(identities.len(), state.conversations.len());
}

#[test]
fn model_lifecycle_requires_teardown_and_explicit_retry_before_verified_readiness() {
    let (mut state, commands, _events) = state();
    state.model_event(Event::Ready);
    state.input = "Original question".into();
    state.submit();
    assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
    assert_eq!(state.model, ModelLifecycle::Generating);
    for event in [Event::Ready, Event::Starting, Event::Stopped] {
        state.model_event(event);
        assert_eq!(state.model, ModelLifecycle::Generating);
    }
    state.input = "A newer draft".into();
    state.model_event(Event::Rejected {
        message: "Connection failed".into(),
        text: "Original question".into(),
        retry_required: true,
    });
    assert_eq!(state.model, ModelLifecycle::AwaitingTeardown);
    assert_eq!(state.input, "A newer draft");
    let status = state.status.clone();
    for event in [
        Event::Ready,
        Event::Starting,
        Event::Complete(Completion::Finished),
    ] {
        state.model_event(event);
        assert_eq!(state.model, ModelLifecycle::AwaitingTeardown);
        assert_eq!(state.status, status);
    }
    state.submit();
    state.retry();
    assert!(commands.try_recv().is_err());
    state.model_event(Event::Retryable("Teardown finished".into()));
    assert_eq!(state.model, ModelLifecycle::Retryable);
    state.model_event(Event::Ready);
    assert_eq!(state.model, ModelLifecycle::Retryable);
    state.retry();
    assert!(matches!(commands.try_recv(), Ok(Command::Retry)));
    assert_eq!(state.model, ModelLifecycle::Restarting);
    for event in [
        Event::Retryable("Old teardown notice".into()),
        Event::Ready,
        Event::Stopped,
        Event::Stopping("Old prompt".into()),
        Event::Complete(Completion::Finished),
    ] {
        state.model_event(event);
        assert_eq!(state.model, ModelLifecycle::Restarting);
    }
    state.submit();
    assert!(commands.try_recv().is_err());
    state.model_event(Event::Starting);
    assert_eq!(state.model, ModelLifecycle::Preparing);
    state.model_event(Event::Ready);
    assert_eq!(state.model, ModelLifecycle::Ready);
    assert_eq!(state.input, "A newer draft");
}

#[test]
fn model_lifecycle_stop_and_terminal_failure_never_reopen_send_from_late_events() {
    let (mut state, commands, _events) = state();
    state.model_event(Event::Ready);
    state.input = "Original question".into();
    state.submit();
    assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
    state.stop_generation();
    assert!(matches!(commands.try_recv(), Ok(Command::Stop)));
    for event in [
        Event::Ready,
        Event::Starting,
        Event::Complete(Completion::Finished),
    ] {
        state.model_event(event);
        assert_eq!(state.model, ModelLifecycle::Stopping);
    }
    state.input = "A newer draft".into();
    state.model_event(Event::Failed("Owned teardown could not be verified".into()));
    assert_eq!(state.model, ModelLifecycle::Unavailable);
    let status = state.status.clone();
    for event in [
        Event::Ready,
        Event::Starting,
        Event::Stopped,
        Event::Retryable("Old notice".into()),
        Event::Rejected {
            message: "Old rejection".into(),
            text: "Original question".into(),
            retry_required: false,
        },
    ] {
        state.model_event(event);
        assert_eq!(state.model, ModelLifecycle::Unavailable);
        assert_eq!(state.status, status);
        assert_eq!(state.input, "A newer draft");
    }
    state.submit();
    state.retry();
    assert!(commands.try_recv().is_err());
}

#[test]
fn model_lifecycle_recoverable_rejection_preserves_a_newer_composer_draft() {
    let (mut state, commands, _events) = state();
    state.model_event(Event::Ready);
    state.input = "Original question".into();
    state.submit();
    assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
    state.input = "A newer draft".into();
    state.model_event(Event::Rejected {
        message: "Prompt rejected".into(),
        text: "Original question".into(),
        retry_required: false,
    });
    assert_eq!(state.model, ModelLifecycle::Ready);
    assert_eq!(state.input, "A newer draft");
    state.model_event(Event::Complete(Completion::Finished));
    assert_eq!(state.status, "Prompt rejected");
    assert!(!state.answer_complete);
}

#[test]
fn model_lifecycle_queued_rejection_cannot_cancel_an_explicit_restart() {
    for newer_draft in ["", "A newer draft"] {
        let (mut state, commands, _events) = state();
        state.model_event(Event::Retryable("Teardown finished".into()));
        newer_draft.clone_into(&mut state.input);
        state.retry();
        assert!(matches!(commands.try_recv(), Ok(Command::Retry)));
        let status = state.status.clone();
        state.model_event(Event::Rejected {
            message: "Queued behind teardown".into(),
            text: "Original question".into(),
            retry_required: true,
        });
        assert_eq!(state.model, ModelLifecycle::Restarting);
        assert_eq!(state.status, status);
        assert_eq!(
            state.input,
            if newer_draft.is_empty() {
                "Original question"
            } else {
                newer_draft
            }
        );
        for event in [
            Event::Retryable("Duplicate teardown notice".into()),
            Event::Stopped,
            Event::Ready,
        ] {
            state.model_event(event);
            assert_eq!(state.model, ModelLifecycle::Restarting);
            assert_eq!(state.status, status);
            state.retry();
            state.submit();
            assert!(commands.try_recv().is_err());
        }
        state.model_event(Event::Starting);
        assert_eq!(state.model, ModelLifecycle::Preparing);
        state.model_event(Event::Ready);
        assert_eq!(state.model, ModelLifecycle::Ready);
    }
}

#[test]
fn finished_voice_message_submits_once_through_the_normal_prompt_flow() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.dictation_requested = true;
    state.dictation_finishing = true;
    state.microphone_generation = Some(7);
    state.accept_transcript(7, "  Explain bonsai pruning  ");
    assert!(commands.try_recv().is_err());
    assert_eq!(state.input, "Explain bonsai pruning");
    assert_eq!(
        state.speech_status,
        "Voice message transcribed. Review the draft, then select Send."
    );
    assert!(state.model.ready());
    assert!(!state.dictation_requested && !state.dictation_finishing);
    assert!(state.microphone_generation.is_none());
    state.accept_transcript(7, "Duplicate recognition");
    assert_eq!(state.input, "Explain bonsai pruning");
    assert!(commands.try_recv().is_err());
    assert!(state.transcript.messages.is_empty());
    state.submit();
    assert!(matches!(
        commands.try_recv(),
        Ok(Command::Send { text, .. }) if text == "Explain bonsai pruning"
    ));
    assert_eq!(state.input, "");
    assert!(state.model.busy());
    assert_eq!(state.transcript.messages[0].body, "Explain bonsai pruning");
}

#[test]
fn unfinished_or_stale_voice_message_cannot_submit() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.input = "Existing draft".to_owned();
    state.dictation_requested = true;
    state.microphone_generation = Some(8);
    state.accept_transcript(8, "Unfinished recognition");
    state.dictation_finishing = true;
    state.accept_transcript(7, "Old recognition");
    assert_eq!(state.input, "Existing draft");
    assert!(commands.try_recv().is_err());
    state.stop_speech();
    state.accept_transcript(8, "Cancelled recognition");
    assert_eq!(state.input, "Existing draft");
    assert!(commands.try_recv().is_err());
}

#[test]
fn empty_or_failed_voice_message_preserves_existing_draft_without_sending() {
    for failed in [false, true] {
        let (mut state, commands, _events) = state();
        state.model = ModelLifecycle::Ready;
        state.input = "Existing draft".to_owned();
        state.dictation_requested = true;
        state.dictation_finishing = true;
        state.microphone_generation = Some(7);
        if failed {
            state.speech_event(speech::Event::InputFailed {
                generation: 7,
                message: "Recognition failed.",
            });
        } else {
            state.accept_transcript(7, "  ");
        }
        assert_eq!(state.input, "Existing draft");
        assert!(!state.dictation_requested && !state.dictation_finishing);
        state.accept_transcript(7, "Late recognition");
        assert!(commands.try_recv().is_err());
    }
}

#[test]
fn voice_message_kept_during_model_preparation_never_sends_on_readiness() {
    let (mut state, commands, _events) = state();
    state.dictation_requested = true;
    state.dictation_finishing = true;
    state.microphone_generation = Some(7);
    state.accept_transcript(7, "Keep this voice draft");
    assert_eq!(state.input, "Keep this voice draft");
    assert!(commands.try_recv().is_err());
    state.model_event(Event::Ready);
    state.accept_transcript(7, "Late recognition");
    assert_eq!(state.input, "Keep this voice draft");
    assert!(commands.try_recv().is_err());
}

#[test]
fn voice_message_exceeding_prompt_limit_keeps_the_complete_existing_draft() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.input = "x".repeat(INPUT_LIMIT - 2);
    state.place_composer_cursor(INPUT_LIMIT - 2);
    state.dictation_requested = true;
    state.dictation_finishing = true;
    state.microphone_generation = Some(7);
    state.accept_transcript(7, "too long");
    assert_eq!(state.input, "x".repeat(INPUT_LIMIT - 2));
    assert!(state.speech_status.contains("prompt limit"));
    assert!(commands.try_recv().is_err());
}

#[test]
fn long_request_rejection_and_stop_restore_the_complete_unicode_draft() {
    for stopping in [false, true] {
        let (mut state, commands, _events) = state();
        state.model = ModelLifecycle::Ready;
        let draft = "界🌱\\\"\n".repeat(20_000);
        state.input.clone_from(&draft);
        state.submit();
        assert!(
            matches!(commands.try_recv(), Ok(Command::Send { text, .. }) if text == draft.trim())
        );
        if stopping {
            state.stopping_response(draft.trim());
        } else {
            state.reject(
                "Serialized request exceeds the byte limit.",
                draft.trim(),
                false,
            );
        }
        assert_eq!(state.input, draft.trim());
    }
}

#[test]
fn switching_conversations_cancels_an_inflight_voice_message() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.input = "Original draft".to_owned();
    state.dictation_requested = true;
    state.dictation_finishing = true;
    state.microphone_generation = Some(7);
    state.new_chat();
    assert!(matches!(commands.try_recv(), Ok(Command::NewChat)));
    state.accept_transcript(7, "Late recognition from previous chat");
    assert_eq!(state.input, "");
    assert!(state.transcript.messages.is_empty());
    assert!(commands.try_recv().is_err());
    assert!(!state.dictation_requested && !state.dictation_finishing);
}

#[test]
fn busy_model_keeps_voice_message_as_a_draft_without_a_second_request() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Generating;
    state.dictation_requested = true;
    state.dictation_finishing = true;
    state.microphone_generation = Some(7);
    state.accept_transcript(7, "Next question");
    assert_eq!(state.input, "Next question");
    assert!(state.model.busy());
    assert!(commands.try_recv().is_err());
}

#[test]
fn voice_message_separates_spoken_words_from_an_existing_draft() {
    for existing in ["Existing draft", "Existing draft "] {
        let (mut state, commands, _events) = state();
        state.model = ModelLifecycle::Ready;
        state.input = existing.to_owned();
        state.place_composer_cursor(state.input.chars().count());
        state.dictation_requested = true;
        state.dictation_finishing = true;
        state.microphone_generation = Some(7);
        state.accept_transcript(7, "Explain bonsai");
        assert_eq!(state.input, "Existing draft Explain bonsai");
        assert!(commands.try_recv().is_err());
    }
}

fn queue_voice_message(state: &mut State) {
    state.dictation_requested = true;
    state.dictation_finishing = true;
    state.microphone_generation = Some(7);
    state.speech_event(speech::Event::Recognized {
        generation: 7,
        text: "Queued voice message".to_owned(),
    });
}

#[test]
fn escape_cancels_a_queued_voice_message_before_overlay_dismissal() {
    for modal in [None, Some(false), Some(true)] {
        let (mut state, commands, _events) = state();
        state.model = ModelLifecycle::Ready;
        state.input = "Keep this draft".to_owned();
        if let Some(rename) = modal {
            if rename {
                state.rename_conversation = Some((1, "Changed name".to_owned()));
            } else {
                state.delete_conversation = Some(1);
            }
        }
        queue_voice_message(&mut state);
        let ctx = egui::Context::default();
        let output = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..egui::RawInput::default()
            },
            |ui| {
                state.handle_input_shortcuts(ui.ctx());
                state.speech_event(speech::Event::Recognized {
                    generation: 7,
                    text: "Late voice message".to_owned(),
                });
                state.flush_voice_transcript(ui.ctx());
            },
        );
        output.drop_without_applying_deltas();
        assert_eq!(state.input, "Keep this draft");
        assert!(state.pending_voice_transcript.is_none());
        assert!(!state.dictation_requested);
        assert!(commands.try_recv().is_err());
    }
}

#[test]
fn new_chat_shortcut_cancels_a_queued_voice_message_before_submission() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    queue_voice_message(&mut state);
    let ctx = egui::Context::default();
    let output = ctx.run_ui(
        egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::N,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::CTRL,
            }],
            ..egui::RawInput::default()
        },
        |ui| {
            state.handle_input_shortcuts(ui.ctx());
            state.flush_voice_transcript(ui.ctx());
        },
    );
    output.drop_without_applying_deltas();
    assert!(matches!(commands.try_recv(), Ok(Command::NewChat)));
    assert!(commands.try_recv().is_err());
    assert_eq!(state.active_conversation, 2);
    assert_eq!(state.input, "");
}

#[test]
fn pointer_press_defers_voice_message_until_cancel_click_settles() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    state.input = "Keep this draft".to_owned();
    queue_voice_message(&mut state);
    let ctx = egui::Context::default();
    let output = ctx.run_ui(
        egui::RawInput {
            events: vec![egui::Event::PointerButton {
                pos: egui::pos2(20.0, 20.0),
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            }],
            ..egui::RawInput::default()
        },
        |ui| state.flush_voice_transcript(ui.ctx()),
    );
    output.drop_without_applying_deltas();
    assert!(state.pending_voice_transcript.is_some());
    assert!(commands.try_recv().is_err());
    let output = ctx.run_ui(
        egui::RawInput {
            events: vec![egui::Event::PointerButton {
                pos: egui::pos2(20.0, 20.0),
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..egui::RawInput::default()
        },
        |ui| {
            state.finish_dictation();
            state.flush_voice_transcript(ui.ctx());
        },
    );
    output.drop_without_applying_deltas();
    assert_eq!(state.input, "Keep this draft");
    assert!(state.pending_voice_transcript.is_none());
    assert!(commands.try_recv().is_err());
}

#[test]
fn voice_message_waits_for_native_input_dispatch_before_sending_once() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    queue_voice_message(&mut state);
    assert_eq!(state.input, "");
    assert!(commands.try_recv().is_err());
    let ctx = egui::Context::default();
    let output = ctx.run_ui(egui::RawInput::default(), |ui| {
        state.flush_voice_transcript(ui.ctx());
        state.flush_voice_transcript(ui.ctx());
    });
    output.drop_without_applying_deltas();
    assert_eq!(state.input, "Queued voice message");
    assert!(commands.try_recv().is_err());
}

#[test]
fn ime_preedit_keeps_a_queued_voice_message_as_an_unsent_draft() {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    queue_voice_message(&mut state);
    let ctx = egui::Context::default();
    let output = ctx.run_ui(
        egui::RawInput {
            events: vec![egui::Event::Ime(egui::ImeEvent::Preedit {
                text: "composing".to_owned(),
                active_range_chars: None,
            })],
            ..egui::RawInput::default()
        },
        |ui| {
            state.handle_input_shortcuts(ui.ctx());
            state.flush_voice_transcript(ui.ctx());
        },
    );
    output.drop_without_applying_deltas();
    assert!(state.composing);
    assert_eq!(state.input, "Queued voice message");
    assert!(state.pending_voice_transcript.is_none());
    assert!(commands.try_recv().is_err());
}

#[test]
fn speech_progress_and_cancellation_do_not_raise_error_alerts() -> io::Result<()> {
    let (mut state, _commands, _events) = state();
    let (session, _speech_commands) = speech::Session::fixture()?;
    state.speech = Some(session);
    state.speech_output = true;
    state.model = ModelLifecycle::Ready;
    state
        .transcript
        .push(Role::Assistant, "Read this answer.".into());
    state.transcript.remove_empty_response();
    let answer = state.transcript.messages[0].id;
    state.toggle_read_aloud(answer);
    assert!(state.speaking && state.speech_loading);
    assert_eq!(state.active_alerts(), []);
    state.speech_event(speech::Event::Speaking {
        generation: 2,
        speaking: true,
    });
    assert_eq!(state.active_alerts(), []);
    state.speech_event(speech::Event::OutputBuffering {
        generation: 2,
        buffering: false,
    });
    assert!(state.speaking && !state.speech_loading);
    state.speech_event(speech::Event::OutputBuffering {
        generation: 0,
        buffering: true,
    });
    assert!(
        !state.speech_loading,
        "Stale progress cannot restart the spinner."
    );
    state.toggle_read_aloud(answer);
    assert!(!state.speaking && !state.speech_loading);
    assert_eq!(
        state.active_alerts(),
        [],
        "An explicit stop is not an error."
    );
    state.speech_event(speech::Event::Failed {
        generation: 2,
        message: "Stale failure",
    });
    assert_eq!(state.active_alerts(), []);
    state.toggle_read_aloud(answer);
    state.speech_event(speech::Event::Failed {
        generation: 5,
        message: "Synthesis failed.",
    });
    assert_eq!(state.active_alerts(), vec![error_ui::AlertKind::Speech]);
    assert!(!state.speaking && !state.speech_loading);
    state.toggle_read_aloud(answer);
    assert_eq!(
        state.active_alerts(),
        [],
        "A new attempt replaces the old failure."
    );
    Ok(())
}

#[test]
fn conversation_history_orders_by_most_recent_update_and_ignores_viewing() -> io::Result<()> {
    let (mut state, commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    let first = state.active_conversation;
    state.conversations[0].title = "First".into();
    state.new_chat();
    let second = state.active_conversation;
    state.conversations[0].title = "Second".into();
    state.new_chat();
    let third = state.active_conversation;
    state.conversations[0].title = "Third".into();
    let order = |state: &State| {
        state
            .conversations
            .iter()
            .map(|conversation| conversation.id)
            .collect::<Vec<_>>()
    };
    assert_eq!(order(&state), vec![third, second, first]);

    // Viewing/switching alone must never reorder conversation history.
    state.switch_conversation(first);
    assert_eq!(order(&state), vec![third, second, first]);
    state.switch_conversation(second);
    assert_eq!(order(&state), vec![third, second, first]);

    // Dictation/draft insertion updates recency of the edited conversation.
    assert!(state.insert_dictation("Updated draft"));
    assert_eq!(order(&state), vec![second, third, first]);

    // Sending in `first` promotes `first`, and background streaming keeps `first`
    // at index 0 while preserving its saved snapshot even after switching away.
    state.switch_conversation(first);
    assert_eq!(order(&state), vec![second, third, first]);
    state.input = "First prompt".into();
    state.submit();
    assert_eq!(order(&state), vec![first, second, third]);
    state.switch_conversation(third);
    assert_eq!(order(&state), vec![first, second, third]);

    // Draft edit in `third` promotes `third` above `first`, then a background
    // stream token in `first` promotes `first` back to index 0 without losing
    // either conversation's saved state.
    assert!(state.insert_dictation("Third draft"));
    assert_eq!(order(&state), vec![third, first, second]);
    state.model_event(Event::Text("Background token".into()));
    assert_eq!(order(&state), vec![first, third, second]);
    state.model_event(Event::Committed(vec![
        client::Message {
            role: "user",
            content: "First prompt".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Background token".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
    ]));
    state.model_event(Event::Complete(Completion::Finished));
    assert_eq!(order(&state), vec![first, third, second]);
    assert_eq!(state.active_conversation, third);
    assert_eq!(state.input, "Third draft");
    let first_saved = state
        .conversations
        .iter()
        .find(|conversation| conversation.id == first)
        .and_then(|conversation| conversation.saved.as_ref());
    assert!(
        first_saved.is_some_and(
            |saved| saved.last_answer == "Background token" && saved.committed.len() == 2
        )
    );

    // Recovery serialization/deserialization preserves exact recency order.
    let bytes = state.workspace_bytes()?;
    let (mut restored, _restored_commands, _restored_events) = self::state();
    restored.restore_workspace(&bytes)?;
    assert_eq!(order(&restored), vec![first, third, second]);
    assert_eq!(restored.active_conversation, third);

    // Deleting the selected conversation switches to the most recently updated
    // remaining conversation (`first`).
    while commands.try_recv().is_ok() {}
    state.remove_conversation(third);
    assert_eq!(state.active_conversation, first);
    assert_eq!(order(&state), vec![first, second]);
    Ok(())
}

#[test]
fn prefill_repaint_cadence_throttles_until_first_token_arrives() {
    let (mut state, _commands, _events) = state();
    state.model = ModelLifecycle::Ready;
    assert_eq!(state.next_repaint_delay(), Duration::from_secs(1));

    state.input = "Wait for prefill".into();
    state.submit();
    assert!(state.model.busy());
    assert!(state.awaiting_first_token);
    assert_eq!(state.next_repaint_delay(), Duration::from_millis(125));

    // Empty deltas keep prefill throttling; first non-empty reasoning or text token switches to 150 ms fallback.
    state.model_event(Event::Text(String::new()));
    assert_eq!(state.next_repaint_delay(), Duration::from_millis(125));

    state.model_event(Event::Reasoning("Thinking".into()));
    assert!(!state.awaiting_first_token);
    assert_eq!(state.next_repaint_delay(), Duration::from_millis(150));

    // Extra effort stage transitions return to 125 ms until that stage emits its first token.
    state.model_event(Event::Stage("Extra effort: Drafting (2/3)"));
    assert_eq!(state.next_repaint_delay(), Duration::from_millis(125));

    state.model_event(Event::Text("First answer token".into()));
    assert_eq!(state.next_repaint_delay(), Duration::from_millis(150));

    state.model_event(Event::Complete(Completion::Finished));
    assert_eq!(state.next_repaint_delay(), Duration::from_secs(1));
}

#[test]
fn close_requested_with_active_dialog_defers_close_until_worker_finishes_or_times_out()
-> io::Result<()> {
    let (mut state, _commands, _events) = state();
    let ctx = egui::Context::default();
    let session = crate::audio_import::Session::fixture(vec![Ok(None)])?;
    state.audio_import = Some(session);

    assert!(state.defer_close_for_dialogs(&ctx, true));
    assert!(matches!(
        state.dialog_close,
        DialogCloseState::Waiting { .. }
    ));

    state.cancel_active_dialogs_for_close();
    state.audio_import = None;
    assert!(state.defer_close_for_dialogs(&ctx, false));
    assert_eq!(state.dialog_close, DialogCloseState::Idle);
    assert!(!state.defer_close_for_dialogs(&ctx, true));

    // Bounded safety timeout stops deferring once expired.
    let session = crate::audio_import::Session::fixture(vec![Ok(None)])?;
    state.audio_import = Some(session);
    state.dialog_close = DialogCloseState::Waiting {
        deadline: Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now),
    };
    assert!(state.defer_close_for_dialogs(&ctx, false));
    assert_eq!(state.dialog_close, DialogCloseState::TimedOut);
    assert!(!state.defer_close_for_dialogs(&ctx, true));

    // Drain any commands emitted during the non-pass checks above before inspecting per-pass output.
    ctx.begin_pass(egui::RawInput::default());
    let mut drained = ctx.end_pass();
    drained.textures_delta.clear();

    // Additional dialog session types, repaint delay, Escape cancellation, second close_requested,
    // cancel_attachment_import retention, and poll_updates close suppression while Waiting.
    state.audio_import = None;
    state.dialog_close = DialogCloseState::Idle;
    let (att, att_shared) = crate::attachment_import::Session::fixture_pending()?;
    state.attachment_import = Some(att);
    assert!(state.has_active_dialog_session());
    ctx.begin_pass(egui::RawInput::default());
    assert!(state.defer_close_for_dialogs(&ctx, true));
    let mut first_wait = ctx.end_pass();
    first_wait.textures_delta.clear();
    assert_eq!(state.next_repaint_delay(), Duration::from_millis(10));

    // Second close_requested before deadline stays Waiting and emits CancelClose,
    // and poll_updates suppresses ViewportCommand::Close while Waiting.
    state.updates = Some(updates::Updates::fixture_ready_to_close());
    ctx.begin_pass(egui::RawInput::default());
    assert!(state.defer_close_for_dialogs(&ctx, true));
    state.poll_updates(&ctx);
    let mut pass_during_wait = ctx.end_pass();
    pass_during_wait.textures_delta.clear();
    let cmds_during_wait = &pass_during_wait.viewport_output[&egui::ViewportId::ROOT].commands;
    assert!(cmds_during_wait.contains(&egui::ViewportCommand::CancelClose));
    assert!(!cmds_during_wait.contains(&egui::ViewportCommand::Close));
    assert!(matches!(
        state.dialog_close,
        DialogCloseState::Waiting { .. }
    ));

    // cancel_attachment_import keeps session until poll_attachment_import reaps the worker.
    state.cancel_attachment_import();
    assert!(state.attachment_import.is_some());
    assert!(
        att_shared
            .cancelled
            .load(std::sync::atomic::Ordering::Acquire)
    );
    let reap_deadline = Instant::now() + Duration::from_secs(2);
    while state.attachment_import.is_some() && Instant::now() < reap_deadline {
        state.poll_attachment_import(&ctx);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(state.attachment_import.is_none());

    // Once active dialog ends, poll_updates emits ViewportCommand::Close when Waiting transitions.
    ctx.begin_pass(egui::RawInput::default());
    assert!(state.defer_close_for_dialogs(&ctx, false));
    assert_eq!(state.dialog_close, DialogCloseState::Idle);
    state.poll_updates(&ctx);
    let mut pass_after_wait = ctx.end_pass();
    pass_after_wait.textures_delta.clear();
    let cmds_after_wait = &pass_after_wait.viewport_output[&egui::ViewportId::ROOT].commands;
    assert!(cmds_after_wait.contains(&egui::ViewportCommand::Close));

    // image_export has_active_dialog_session and non-blocking Escape cancellation.
    let img_export = crate::image_export::Export::fixture_pending()?;
    state.image_export = Some(img_export);
    assert!(state.has_active_dialog_session());
    ctx.begin_pass(egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..egui::RawInput::default()
    });
    state.handle_escape(&ctx);
    let mut escape_pass = ctx.end_pass();
    escape_pass.textures_delta.clear();
    assert!(state.image_export.is_some());
    let export_reap = Instant::now() + Duration::from_secs(2);
    while state.image_export.is_some() && Instant::now() < export_reap {
        state.poll_image_export(&ctx);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(state.image_export.is_none());
    Ok(())
}

#[test]
fn speech_disconnect_stops_per_frame_alerts_and_oversized_read_aloud_preserves_session()
-> io::Result<()> {
    let (mut state, _commands, _events) = state();
    state.speech = Some(speech::start(None, None)?);
    state.speech_output = true;
    state.speech_input = true;

    // Oversized message, whitespace-only, and NUL-containing messages fail toggle_read_aloud
    // pre-check without disabling speech.
    state.model = ModelLifecycle::Generating;
    state
        .transcript
        .push(Role::Assistant, "a".repeat(speech::MAX_TEXT + 1));
    state.model_event(Event::Complete(Completion::Finished));
    let id = state.transcript.messages[0].id;
    state.toggle_read_aloud(id);
    assert!(state.speech.is_some());
    assert!(state.speech_output);
    assert!(state.speech_input);
    assert!(state.speech_notice);

    // Whitespace-only is filtered before speech_command without raising an error notice,
    // while NUL-containing text fails toggle_read_aloud pre-check and raises speech_notice.
    for (invalid_body, expect_notice) in [("   \n\t", false), ("hello\0world", true)] {
        state.speech_notice = false;
        state.model = ModelLifecycle::Generating;
        state.transcript.push(Role::Assistant, invalid_body.to_owned());
        state.model_event(Event::Complete(Completion::Finished));
        let invalid_id = state.transcript.messages.last().map_or(0, |m| m.id);
        state.toggle_read_aloud(invalid_id);
        assert!(state.speech.is_some());
        assert!(state.speech_output);
        assert!(state.speech_input);
        assert_eq!(state.speech_notice, expect_notice);
    }

    // PayloadTooLarge from speech_command keeps speech enabled.
    state.speech_notice = false;
    assert!(!state.speech_command(speech::Command::Transcribe {
        generation: 1,
        wav: vec![0; 2 * 1024 * 1024],
    }));
    assert!(state.speech.is_some());
    assert!(state.speech_output);
    assert!(state.speech_input);
    assert!(state.speech_notice);

    // Disconnecting the worker clears self.speech and never re-raises the alert once dismissed.
    if let Some(session) = &mut state.speech {
        session.stop_for_test();
    }
    state.drain_speech_events();
    assert!(state.speech.is_none());
    assert!(!state.speech_output);
    assert!(!state.speech_input);
    assert!(state.speech_notice);

    state.dismiss_alert(error_ui::AlertKind::Speech);
    assert!(!state.speech_notice);
    state.drain_speech_events();
    assert!(!state.speech_notice);

    // SendError::QueueFull keeps self.speech.is_some(), preserves flags, and sets busy notice.
    let (bounded, hold) = speech::Session::fixture_bounded();
    bounded.set_input_generation(Some(1));
    for _ in 0..64 {
        bounded.send_overflow_for_test().map_err(io::Error::other)?;
    }
    state.speech = Some(bounded);
    state.speech_output = true;
    state.speech_input = true;
    state.speech_notice = false;
    assert!(!state.speech_command(speech::Command::PrepareInput { generation: 1 }));
    assert!(state.speech.is_some());
    assert!(state.speech_output);
    assert!(state.speech_input);
    assert!(state.speech_notice);
    assert!(state.speech_status.contains("busy"));

    // SendError::Disconnected directly from speech_command clears self.speech immediately,
    // and calling speech_command with self.speech == None returns false.
    hold.close();
    assert!(!state.speech_command(speech::Command::StopSpeaking));
    assert!(state.speech.is_none());
    assert!(!state.speech_output);
    assert!(!state.speech_input);
    assert!(!state.speech_command(speech::Command::StopSpeaking));
    Ok(())
}
