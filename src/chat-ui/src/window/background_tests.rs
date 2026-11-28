//! Background response ownership with inert model and speech channels.
use super::*;

fn background() -> (State, Receiver<Command>, client::EventSender, u64) {
    let (commands, incoming) = mpsc::channel();
    let (events, incoming_events) = client::event_channel();
    let mut state = State::create(commands, incoming_events, None, None);
    state.preview = false;
    state.model = ModelLifecycle::Ready;
    state.input = "Original question".into();
    state.submit();
    assert!(matches!(incoming.try_recv(), Ok(Command::Send { .. })));
    let owner = state.active_conversation;
    state.input = "Owner's next draft".into();
    state.new_chat();
    assert_ne!(state.active_conversation, owner);
    state.input = "Browsing draft 🌱".into();
    assert!(incoming.try_recv().is_err());
    (state, incoming, events, owner)
}

fn completed_history() -> Vec<client::Message> {
    vec![
        client::Message {
            role: "user",
            content: "Original question".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Finished answer".into(),
            images: std::sync::Arc::default(),
            reasoning_content: Some("Reasoning".into()),
        },
    ]
}

#[test]
fn background_completion_preserves_browsing_drafts_and_restores_selected_history() {
    let (mut state, commands, _events, owner) = background();
    let selected = state.active_conversation;
    state.extra_effort = true;
    state.output_budget = 1;
    state.focus_composer = false;
    state.place_composer_cursor(3);
    let cursor = (state.composer_cursor, state.pending_cursor);
    assert!(commands.try_recv().is_err());

    state.model_event(Event::Stage("Draft"));
    state.model_event(Event::Reasoning("Reasoning".into()));
    state.model_event(Event::Text("Finished answer".into()));
    state.model_event(Event::Draft("Finished answer".into()));
    state.model_event(Event::Committed(completed_history()));
    assert!(state.model.busy());
    assert!(state.transcript.messages.is_empty() && state.committed.is_empty());
    assert_eq!(state.active_conversation, selected);
    assert_eq!(state.input, "Browsing draft 🌱");
    assert_eq!((state.composer_cursor, state.pending_cursor), cursor);
    assert!(!state.focus_composer);
    assert!(commands.try_recv().is_err());
    state.remove_conversation(owner);
    assert_eq!(state.conversations.len(), 2);

    state.switch_conversation(owner);
    assert_eq!(state.stage, Some("Draft"));
    assert_eq!(state.input, "Owner's next draft");
    state.switch_conversation(selected);
    state.model_event(Event::Complete(Completion::Finished));
    assert_eq!(state.model, ModelLifecycle::Ready);
    assert_eq!(state.input, "Browsing draft 🌱");
    assert!(state.extra_effort && state.output_budget == 1);
    assert!(matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.is_empty()));
    state.submit();
    assert!(
        matches!(commands.try_recv(), Ok(Command::Send { text, .. }) if text == "Browsing draft 🌱")
    );
    assert!(commands.try_recv().is_err());
    state.switch_conversation(owner);
    assert_eq!(state.last_answer, "Finished answer");
    assert_eq!(state.committed.len(), 2);
    assert_eq!(
        state.committed[1].reasoning_content.as_deref(),
        Some("Reasoning")
    );
    assert!(state.answer_complete);
}

#[test]
fn background_stop_completion_race_preserves_both_drafts_and_exact_owner_history() {
    let (mut state, commands, _events, owner) = background();
    state.model_event(Event::Text("Partial".into()));
    state.stop_generation();
    assert!(matches!(commands.try_recv(), Ok(Command::Stop)));
    state.model_event(Event::Text("Ignored after Stop".into()));
    state.model_event(Event::Committed(completed_history()));
    state.model_event(Event::Stopping("Original question".into()));
    state.model_event(Event::Stopped);
    assert_eq!(state.model, ModelLifecycle::Ready);
    assert!(!state.warning);
    assert!(matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.is_empty()));
    assert_eq!(state.input, "Browsing draft 🌱");
    assert!(state.transcript.messages.is_empty());
    state.switch_conversation(owner);
    assert_eq!(state.input, "Owner's next draft");
    assert_eq!(state.last_answer, "Finished answer");
    assert_eq!(state.committed.len(), 2);
    assert!(state.transcript.plain_text().contains("Finished answer"));
    assert!(!state.transcript.plain_text().contains("Ignored after Stop"));
    assert!(matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.len() == 2));
    assert!(commands.try_recv().is_err());
}

#[test]
fn background_rejection_restores_only_empty_owner_drafts_and_preserves_lifecycle() {
    for retry_required in [false, true] {
        for owner_draft in ["", "Keep newer draft"] {
            let (mut state, commands, _events, owner) = background();
            let selected = state.active_conversation;
            state.switch_conversation(owner);
            owner_draft.clone_into(&mut state.input);
            state.switch_conversation(selected);
            state.model_event(Event::Rejected {
                message: "Rejected response".into(),
                text: "Original question".into(),
                retry_required,
            });
            assert_eq!(state.input, "Browsing draft 🌱");
            assert_eq!(
                state.model,
                if retry_required {
                    ModelLifecycle::AwaitingTeardown
                } else {
                    ModelLifecycle::Ready
                }
            );
            if retry_required {
                assert!(commands.try_recv().is_err());
                state.model_event(Event::Retryable("Cleanup complete".into()));
            } else {
                assert!(
                    matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.is_empty())
                );
            }
            state.switch_conversation(owner);
            assert_eq!(
                state.input,
                if owner_draft.is_empty() {
                    "Original question"
                } else {
                    owner_draft
                }
            );
            assert!(state.committed.is_empty());
        }
    }
}

#[test]
fn background_restart_late_rejection_preserves_global_status_and_visible_draft() {
    let (mut state, commands, _events, owner) = background();
    state.stop_generation();
    assert!(matches!(commands.try_recv(), Ok(Command::Stop)));
    state.model_event(Event::Stopping("Original question".into()));
    state.model_event(Event::Restarting);
    let status = state.status.clone();
    state.model_event(Event::Rejected {
        message: "Old request could not run".into(),
        text: "Original question".into(),
        retry_required: true,
    });
    assert_eq!(state.model, ModelLifecycle::Restarting);
    assert_eq!(state.status, status);
    assert_eq!(state.input, "Browsing draft 🌱");
    state.model_event(Event::Starting);
    state.model_event(Event::Ready);
    assert!(matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.is_empty()));
    state.switch_conversation(owner);
    assert_eq!(state.input, "Owner's next draft");
}

#[test]
fn background_disconnect_cleans_up_the_owner_and_preserves_browsed_content() {
    let (mut state, commands, events, owner) = background();
    state
        .transcript
        .push(Role::Assistant, "Existing browsed answer".into());
    drop(events);
    state.drain_events();
    assert_eq!(state.model, ModelLifecycle::Unavailable);
    assert_eq!(state.input, "Browsing draft 🌱");
    assert!(state.warning);
    assert_eq!(state.transcript.messages[0].body, "Existing browsed answer");
    state.switch_conversation(owner);
    assert_eq!(state.transcript.messages.len(), 1);
    assert_eq!(state.input, "Owner's next draft");
    assert!(commands.try_recv().is_err());
}

#[test]
fn background_recovery_keeps_partial_owner_text_separate_from_committed_history() -> io::Result<()>
{
    let (mut state, _commands, _events, owner) = background();
    state.model_event(Event::Text("Partial owner answer".into()));
    let saved: serde_json::Value = serde_json::from_slice(&state.workspace_bytes()?)?;
    assert_eq!(saved["active"]["input"], "Browsing draft 🌱");
    assert_eq!(
        saved["active"]["transcript"]["messages"],
        serde_json::json!([])
    );
    let owner = saved["conversations"]
        .as_array()
        .and_then(|conversations| conversations.iter().find(|value| value["id"] == owner))
        .ok_or_else(|| io::Error::other("Owner conversation was not saved"))?;
    assert_eq!(owner["saved"]["input"], "Owner's next draft");
    assert_eq!(
        owner["saved"]["transcript"]["messages"][1]["body"],
        "Partial owner answer"
    );
    assert_eq!(owner["saved"]["committed"], serde_json::json!([]));
    assert!(owner["saved"].get("stage").is_none());
    Ok(())
}

#[test]
fn background_speech_never_starts_hidden_playback_or_finishes_selected_playback() -> io::Result<()>
{
    let (mut state, _commands, _events, owner) = background();
    let selected = state.active_conversation;
    let (session, speech_commands) = speech::Session::fixture()?;
    state.speech = Some(session);
    state.speech_output = true;
    state
        .transcript
        .push(Role::Assistant, "Selected answer".into());
    state.transcript.remove_empty_response();
    state.last_answer = "Selected answer".into();
    state.toggle_streaming_read_aloud();
    assert!(speech_commands.try_recv().is_err());
    state.switch_conversation(owner);
    state.streaming_read_aloud = true;
    state.switch_conversation(selected);
    while speech_commands.try_recv().is_ok() {}
    state.streaming_suppressed = false;
    let message = state.transcript.messages[0].id;
    state.toggle_read_aloud(message);
    while speech_commands.try_recv().is_ok() {}
    assert!(state.speaking);
    state.model_event(Event::Stage("Draft"));
    state.model_event(Event::Text("Hidden answer".into()));
    state.model_event(Event::Complete(Completion::Finished));
    assert!(state.speaking);
    assert!(speech_commands.try_recv().is_err());
    Ok(())
}

#[test]
fn queued_messages_dispatch_fifo_while_preserving_active_draft_and_cursor() {
    let (commands, incoming) = mpsc::channel();
    let (_events, incoming_events) = client::event_channel();
    let mut state = State::create(commands, incoming_events, None, None);
    state.preview = false;
    state.model = ModelLifecycle::Ready;
    state.input = "First turn".into();
    state.submit();
    assert!(matches!(
        incoming.try_recv(),
        Ok(Command::Send { text, .. }) if text == "First turn"
    ));

    state.input = "Queued one".into();
    state.submit();
    assert_eq!(state.input, "");
    assert_eq!(state.queued_messages.len(), 1);
    assert!(incoming.try_recv().is_err());

    state.input = "Queued two".into();
    state.queue_draft();
    assert_eq!(state.input, "");
    assert_eq!(state.queued_messages.len(), 2);

    state.input = "Still typing".into();
    state.place_composer_cursor(5);
    state.focus_composer = false;
    let cursor = (state.composer_cursor, state.pending_cursor);

    state.model_event(Event::Text("First answer".into()));
    state.model_event(Event::Committed(completed_history()));
    state.model_event(Event::Complete(Completion::Finished));

    assert_eq!(state.model, ModelLifecycle::Generating);
    assert_eq!(state.queued_messages.len(), 1);
    assert_eq!(state.input, "Still typing");
    assert_eq!((state.composer_cursor, state.pending_cursor), cursor);
    assert!(!state.focus_composer);
    assert!(matches!(
        incoming.try_recv(),
        Ok(Command::Send { text, .. }) if text == "Queued one"
    ));
    assert!(incoming.try_recv().is_err());

    let second_history = vec![
        client::Message {
            role: "user",
            content: "Original question".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Finished answer".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "user",
            content: "Queued one".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Second answer".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
    ];
    state.model_event(Event::Text("Second answer".into()));
    state.model_event(Event::Committed(second_history));
    state.model_event(Event::Complete(Completion::Finished));

    assert_eq!(state.model, ModelLifecycle::Generating);
    assert!(state.queued_messages.is_empty());
    assert_eq!(state.input, "Still typing");
    assert!(matches!(
        incoming.try_recv(),
        Ok(Command::Send { text, .. }) if text == "Queued two"
    ));
    assert!(incoming.try_recv().is_err());
}

#[test]
fn background_owner_queue_dispatches_without_disturbing_active_draft_or_speech() -> io::Result<()> {
    let (commands, incoming) = mpsc::channel();
    let (_events, incoming_events) = client::event_channel();
    let mut state = State::create(commands, incoming_events, None, None);
    state.preview = false;
    state.model = ModelLifecycle::Ready;
    let (session, speech_commands) = speech::Session::fixture()?;
    state.speech = Some(session);
    state.speech_output = true;

    state.input = "Owner prompt 1".into();
    state.submit();
    assert!(matches!(incoming.try_recv(), Ok(Command::Send { .. })));
    while speech_commands.try_recv().is_ok() {}
    let owner = state.active_conversation;

    state.input = "Owner queued prompt 2".into();
    state.queue_draft();
    assert_eq!(state.queued_messages.len(), 1);

    state.new_chat();
    let selected = state.active_conversation;
    assert_ne!(selected, owner);
    state
        .transcript
        .push(Role::Assistant, "Selected existing answer".into());
    state.transcript.remove_empty_response();
    let message_id = state.transcript.messages[0].id;
    state.toggle_read_aloud(message_id);
    while speech_commands.try_recv().is_ok() {}
    assert!(state.speaking);

    state.input = "Active browsing draft".into();
    state.place_composer_cursor(6);
    state.focus_composer = false;
    let cursor = (state.composer_cursor, state.pending_cursor);

    state.model_event(Event::Text("Owner answer 1".into()));
    state.model_event(Event::Committed(completed_history()));
    state.model_event(Event::Complete(Completion::Finished));

    assert_eq!(state.active_conversation, selected);
    assert_eq!(state.request_conversation, Some(owner));
    assert_eq!(state.model, ModelLifecycle::Generating);
    assert!(state.queued_messages.is_empty());
    assert_eq!(state.input, "Active browsing draft");
    assert_eq!((state.composer_cursor, state.pending_cursor), cursor);
    assert!(!state.focus_composer);
    assert!(state.speaking);
    assert!(speech_commands.try_recv().is_err());
    assert!(matches!(
        incoming.try_recv(),
        Ok(Command::Send { text, .. }) if text == "Owner queued prompt 2"
    ));
    assert!(incoming.try_recv().is_err());

    state.model_event(Event::Text("Owner answer 2".into()));
    state.model_event(Event::Complete(Completion::Finished));
    assert_eq!(state.model, ModelLifecycle::Ready);
    assert!(matches!(
        incoming.try_recv(),
        Ok(Command::RestoreChat(history)) if history.is_empty()
    ));
    Ok(())
}

#[test]
fn cross_conversation_queue_restores_target_history_before_sending() {
    let (mut state, commands, _events, owner) = background();
    let selected = state.active_conversation;
    state.committed = vec![
        client::Message {
            role: "user",
            content: "Prior selected turn".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Prior selected reply".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
    ];
    state.input = "Queued in selected".into();
    state.submit();
    assert_eq!(state.queued_messages.len(), 1);
    assert_eq!(state.input, "");
    assert!(commands.try_recv().is_err());

    state.model_event(Event::Text("Owner answer".into()));
    state.model_event(Event::Committed(completed_history()));
    state.model_event(Event::Complete(Completion::Finished));

    assert_eq!(state.model, ModelLifecycle::Generating);
    assert_eq!(state.request_conversation, Some(selected));
    assert!(state.queued_messages.is_empty());
    assert!(matches!(
        commands.try_recv(),
        Ok(Command::RestoreChat(history)) if history.len() == 2 && history[0].content.as_ref() == "Prior selected turn"
    ));
    assert!(matches!(
        commands.try_recv(),
        Ok(Command::Send { text, .. }) if text == "Queued in selected"
    ));
    assert!(commands.try_recv().is_err());

    state.switch_conversation(owner);
    assert_eq!(state.committed.len(), 2);
}

#[test]
fn non_finished_outcomes_pause_queue_without_replaying_items() {
    // 1. User Stop pauses queue and preserves items.
    {
        let (commands, incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::create(commands, incoming_events, None, None);
        state.preview = false;
        state.model = ModelLifecycle::Ready;
        state.input = "Turn 1".into();
        state.submit();
        assert!(matches!(incoming.try_recv(), Ok(Command::Send { .. })));
        state.input = "Queued follow-up".into();
        state.queue_draft();
        state.model_event(Event::Text("Partial".into()));
        state.stop_generation();
        assert!(matches!(incoming.try_recv(), Ok(Command::Stop)));
        assert!(state.queue_paused);
        state.model_event(Event::Stopping("Turn 1".into()));
        state.model_event(Event::Stopped);
        assert_eq!(state.model, ModelLifecycle::Ready);
        assert!(state.queue_paused);
        assert_eq!(state.queued_messages.len(), 1);
        assert!(matches!(
            incoming.try_recv(),
            Ok(Command::RestoreChat(history)) if history.len() == 2
        ));
        assert!(incoming.try_recv().is_err());

        state.send_next_queued_for_active();
        assert!(!state.queue_paused);
        assert!(state.queued_messages.is_empty());
        assert!(matches!(
            incoming.try_recv(),
            Ok(Command::Send { text, .. }) if text == "Queued follow-up"
        ));
    }

    // 2. TokenLimit pauses queue without auto-dispatching.
    {
        let (commands, incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::create(commands, incoming_events, None, None);
        state.preview = false;
        state.model = ModelLifecycle::Ready;
        state.input = "Turn 1".into();
        state.submit();
        assert!(matches!(incoming.try_recv(), Ok(Command::Send { .. })));
        state.input = "Queued follow-up".into();
        state.queue_draft();
        state.model_event(Event::Text("Capped".into()));
        state.model_event(Event::Committed(completed_history()));
        state.model_event(Event::Complete(Completion::TokenLimit));
        assert_eq!(state.model, ModelLifecycle::Ready);
        assert!(state.queue_paused);
        assert_eq!(state.queued_messages.len(), 1);
        assert!(incoming.try_recv().is_err());
    }

    // 3. Rejection, Retryable, and Failed pause queue and keep items across Ready.
    for outcome in [0, 1, 2] {
        let (commands, incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::create(commands, incoming_events, None, None);
        state.preview = false;
        state.model = ModelLifecycle::Ready;
        state.input = "Turn 1".into();
        state.submit();
        assert!(matches!(incoming.try_recv(), Ok(Command::Send { .. })));
        state.input = "Queued follow-up".into();
        state.queue_draft();
        match outcome {
            0 => state.model_event(Event::Rejected {
                message: "Too large".into(),
                text: "Turn 1".into(),
                retry_required: false,
            }),
            1 => state.model_event(Event::Retryable("Temporary teardown".into())),
            _ => state.model_event(Event::Failed("Fatal failure".into())),
        }
        assert!(state.queue_paused);
        assert_eq!(state.queued_messages.len(), 1);
        assert!(incoming.try_recv().is_err());
    }
}

#[test]
fn queue_limit_vision_and_edit_preserve_drafts_and_attachments() {
    let (commands, incoming) = mpsc::channel();
    let (_events, incoming_events) = client::event_channel();
    let mut state = State::create(commands, incoming_events, None, None);
    state.preview = false;
    state.model = ModelLifecycle::Ready;
    state.vision_available = false;
    state.input = "Initial".into();
    state.submit();
    assert!(matches!(incoming.try_recv(), Ok(Command::Send { .. })));

    let png = crate::attachment_import::PrehashedPng::new(std::sync::Arc::from(vec![1, 2, 3, 4]));
    let media = crate::attachment_import::Attachment::Image {
        name: "photo.png".into(),
        width: 100,
        height: 100,
        data_url: "data:image/png;base64,AAAA".into(),
        thumbnail_png: png,
    };
    state.set_draft_attachments(vec![media]);
    state.queue_draft();
    assert!(state.queued_messages.is_empty());
    assert_eq!(state.draft_attachments.len(), 1);
    assert!(state.warning);

    state.set_draft_attachments(Vec::new());
    state.warning = false;
    for index in 1..=MAX_QUEUED_MESSAGES {
        state.input = format!("Message {index}");
        state.queue_draft();
        assert_eq!(state.input, "");
        assert_eq!(state.queued_messages.len(), index);
    }

    state.input = "Ninth message".into();
    state.queue_draft();
    assert_eq!(state.queued_messages.len(), MAX_QUEUED_MESSAGES);
    assert_eq!(state.input, "Ninth message");
    assert!(state.conversation_notice.is_some());

    let first_id = state.queued_messages[0].id;
    state.edit_queued_message(first_id);
    assert_eq!(state.queued_messages.len(), MAX_QUEUED_MESSAGES);
    assert_eq!(state.input, "Ninth message");

    state.input.clear();
    state.edit_queued_message(first_id);
    assert_eq!(state.input, "Message 1");
    assert_eq!(state.queued_messages.len(), MAX_QUEUED_MESSAGES - 1);

    let second_id = state.queued_messages[0].id;
    state.remove_queued_message(second_id);
    assert_eq!(state.queued_messages.len(), MAX_QUEUED_MESSAGES - 2);

    state.clear_queued_messages_for_active();
    assert!(state.queued_messages.is_empty());
}

#[test]
fn deleting_conversation_prunes_its_queue_and_recovery_never_persists_queue() -> io::Result<()> {
    let (mut state, _commands, _events, owner) = background();
    let selected = state.active_conversation;
    state.input = "Queued in selected".into();
    state.queue_draft();
    state.switch_conversation(owner);
    state.input = "Queued in owner".into();
    state.queue_draft();
    assert_eq!(state.queued_messages.len(), 2);

    let bytes = state.workspace_bytes()?;
    let (commands2, incoming2) = mpsc::channel();
    let (_events2, incoming_events2) = client::event_channel();
    let mut restored = State::create(commands2, incoming_events2, None, None);
    restored.preview = false;
    restored.restore_workspace(&bytes)?;
    restored.model = ModelLifecycle::Ready;
    restored.restore_selected_history();
    assert!(restored.queued_messages.is_empty());
    assert!(matches!(incoming2.try_recv(), Ok(Command::RestoreChat(_))));
    assert!(incoming2.try_recv().is_err());

    state.remove_conversation(selected);
    assert_eq!(state.queued_messages.len(), 1);
    assert_eq!(state.queued_messages[0].conversation_id, owner);
    Ok(())
}
