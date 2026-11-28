//! Message menus dispatch explicit actions through the real native widgets.
use super::*;
use egui_kittest::{Harness, kittest::Queryable as _};

struct Fixture {
    app: State,
    commands: Receiver<Command>,
    _events: client::EventSender,
}

fn harness(size: [f32; 2]) -> Harness<'static, Fixture> {
    let (commands, incoming) = mpsc::channel();
    let (events, incoming_events) = client::event_channel();
    let mut app = State::create(commands, incoming_events, None, None);
    app.preview = false;
    app.model = ModelLifecycle::Ready;
    app.input = "Keep my next draft 🌱".into();
    app.transcript.push(Role::User, "Original question".into());
    app.transcript
        .push(Role::Assistant, "Original answer".into());
    app.transcript.remove_empty_response();
    app.committed = vec![
        client::Message {
            role: "user",
            content: "Original question".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Original answer".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
    ];
    app.transcript.bind_committed(&[], &app.committed);
    Harness::builder()
        .with_size(size)
        .with_context_setup(configure_test_context)
        .build_ui_state(
            |ui, fixture: &mut Fixture| {
                fixture.app.handle_input_shortcuts(ui.ctx());
                fixture.app.draw(ui);
            },
            Fixture {
                app,
                commands: incoming,
                _events: events,
            },
        )
}

fn open_actions(chat: &mut Harness<'_, Fixture>, id: u64) {
    chat.get_by_role_and_label(egui::Role::Button, &format!("Actions for message {id}"))
        .click();
    chat.run();
}

#[test]
fn retry_response_button_reloads_then_retries_existing_turn_at_minimum_width() {
    for size in [[1180.0, 860.0], [480.0, 640.0]] {
        let mut chat = harness(size);
        chat.state_mut().app.input = "Interrupted question".into();
        chat.state_mut().app.submit();
        assert!(matches!(
            chat.state().commands.try_recv(),
            Ok(Command::Send { .. })
        ));
        chat.state_mut().app.model_event(Event::Rejected {
            message: "The response did not complete.".into(),
            text: "Interrupted question".into(),
            retry_required: true,
        });
        chat.state_mut()
            .app
            .model_event(Event::Retryable("Select Retry model.".into()));
        chat.state_mut().app.input = "Keep my newer draft".into();
        chat.run();
        let reload = chat.get_by_role_and_label(egui::Role::Button, "Reload model");
        let rect = reload.rect();
        assert!(rect.left() >= 0.0 && rect.right() <= size[0]);
        assert!(rect.height() >= 32.0);
        reload.click();
        chat.run();
        assert!(matches!(
            chat.state().commands.try_recv(),
            Ok(Command::Retry)
        ));
        chat.state_mut().app.model_event(Event::Starting);
        chat.state_mut().app.model_event(Event::Ready);
        chat.run();
        assert!(chat.state().commands.try_recv().is_err());
        chat.get_by_role_and_label(egui::Role::Button, "Retry response")
            .click();
        chat.run();
        assert!(
            matches!(chat.state().commands.try_recv(), Ok(Command::Send { text, .. }) if text == "Interrupted question")
        );
        assert!(chat.state().commands.try_recv().is_err());
        assert_eq!(chat.state().app.input, "Keep my newer draft");
        assert_eq!(chat.state().app.transcript.messages.len(), 4);
        assert_eq!(chat.state().app.conversations.len(), 1);
    }
}

#[test]
fn intentional_stop_offers_immediate_retry_response_without_reloading_model() {
    for size in [[1180.0, 860.0], [480.0, 640.0]] {
        let mut chat = harness(size);
        chat.state_mut().app.input = "Interrupted question".into();
        chat.state_mut().app.submit();
        assert!(matches!(
            chat.state().commands.try_recv(),
            Ok(Command::Send { .. })
        ));
        chat.state_mut()
            .app
            .model_event(Event::Stopping("Interrupted question".into()));
        chat.state_mut().app.model_event(Event::Stopped);
        chat.state_mut().app.input = "Keep my newer draft".into();
        chat.run();
        assert!(chat.query_by_label("Model needs attention").is_none());
        assert!(
            chat.query_by_role_and_label(egui::Role::Button, "Reload model")
                .is_none()
        );
        chat.get_by_role_and_label(egui::Role::Button, "Retry response")
            .click();
        chat.run();
        assert!(
            matches!(chat.state().commands.try_recv(), Ok(Command::Send { text, .. }) if text == "Interrupted question")
        );
        assert!(chat.state().commands.try_recv().is_err());
        assert_eq!(chat.state().app.input, "Keep my newer draft");
        assert_eq!(chat.state().app.transcript.messages.len(), 4);
    }
}

#[test]
fn iteration_menu_edit_escape_preserves_original_draft_and_blocks_global_shortcuts() {
    for size in [[1180.0, 860.0], [620.0, 800.0]] {
        let mut chat = harness(size);
        let original = chat.state().app.active_conversation;
        open_actions(&mut chat, 1);
        chat.get_by_role_and_label(egui::Role::Button, "Edit and resend")
            .click();
        chat.run();
        assert!(chat.state().app.edit_message.is_some());
        chat.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::N);
        chat.key_press(egui::Key::Enter);
        chat.run();
        assert_eq!(chat.state().app.conversations.len(), 1);
        assert!(chat.state().commands.try_recv().is_err());
        chat.key_press(egui::Key::Escape);
        chat.run();
        assert!(chat.state().app.edit_message.is_none());
        assert_eq!(chat.state().app.active_conversation, original);
        assert_eq!(chat.state().app.input, "Keep my next draft 🌱");
        assert_eq!(chat.state().app.committed.len(), 2);
        assert!(chat.state().commands.try_recv().is_err());
    }
}

#[test]
fn iteration_menu_branches_without_sending_and_continue_keeps_a_draft() {
    let mut chat = harness([1180.0, 860.0]);
    let original = chat.state().app.active_conversation;
    open_actions(&mut chat, 2);
    chat.get_by_role_and_label(egui::Role::Button, "Branch from here")
        .click();
    chat.run();
    assert_ne!(chat.state().app.active_conversation, original);
    assert_eq!(chat.state().app.conversations.len(), 2);
    assert_eq!(chat.state().app.committed.len(), 2);
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::RestoreChat(history)) if history.len() == 2
    ));
    assert!(chat.state().commands.try_recv().is_err());
    chat.state_mut().app.input = "Branch draft".into();
    open_actions(&mut chat, 2);
    chat.get_by_role_and_label(egui::Role::Button, "Continue response")
        .click();
    chat.run();
    assert!(chat.state().app.model.busy());
    assert_eq!(chat.state().app.input, "Branch draft");
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::Send { text, .. }) if text.contains("Continue")
    ));
    assert!(chat.state().commands.try_recv().is_err());
    let saved = chat
        .state()
        .app
        .conversations
        .iter()
        .find(|conversation| conversation.id == original)
        .and_then(|conversation| conversation.saved.as_ref());
    assert!(saved.is_some_and(|snapshot| snapshot.input == "Keep my next draft 🌱"));
}

#[test]
fn unanswered_stopped_prompt_supports_menu_edit_and_branch_while_preserving_source_draft() {
    let mut chat = harness([1180.0, 860.0]);
    let original = chat.state().app.active_conversation;
    chat.state_mut().app.input = "Unanswered stopped prompt".into();
    chat.state_mut().app.submit();
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::Send { .. })
    ));
    let prompt_id = chat
        .state()
        .app
        .transcript
        .messages
        .iter()
        .rfind(|message| message.role == Role::User)
        .map(|message| message.id)
        .unwrap_or(0);
    chat.state_mut()
        .app
        .model_event(Event::Stopping("Unanswered stopped prompt".into()));
    chat.state_mut().app.model_event(Event::Stopped);
    chat.state_mut().app.input = "Separate composer draft 🌱".into();
    chat.run();

    // Branch from the unanswered stopped prompt places its text into the new branch composer.
    open_actions(&mut chat, prompt_id);
    chat.get_by_role_and_label(egui::Role::Button, "Branch from here")
        .click();
    chat.run();
    assert_ne!(chat.state().app.active_conversation, original);
    assert_eq!(chat.state().app.committed.len(), 2);
    assert_eq!(chat.state().app.input, "Unanswered stopped prompt");
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::RestoreChat(history)) if history.len() == 2
    ));

    // Switch back to the original conversation and use Edit and resend on the unanswered card.
    chat.state_mut().app.switch_conversation(original);
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::RestoreChat(history)) if history.len() == 2
    ));
    chat.run();
    open_actions(&mut chat, prompt_id);
    chat.get_by_role_and_label(egui::Role::Button, "Edit and resend")
        .click();
    chat.run();
    if let Some(edit) = chat.state_mut().app.edit_message.as_mut() {
        assert_eq!(edit.text, "Unanswered stopped prompt");
        edit.text = "Edited unanswered prompt".into();
    }
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Create branch and send")
        .click();
    chat.run();
    assert_ne!(chat.state().app.active_conversation, original);
    assert_eq!(chat.state().app.conversations.len(), 3);
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::RestoreChat(history)) if history.len() == 2
    ));
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::Send { text, .. }) if text == "Edited unanswered prompt"
    ));
    let saved = chat
        .state()
        .app
        .conversations
        .iter()
        .find(|conversation| conversation.id == original)
        .and_then(|conversation| conversation.saved.as_ref());
    assert!(saved.is_some_and(|snapshot| snapshot.input == "Separate composer draft 🌱"));
}

#[test]
fn token_limit_response_offers_continue_and_retry_buttons_at_minimum_width() {
    for size in [[1180.0, 860.0], [480.0, 640.0]] {
        let mut chat = harness(size);
        let original = chat.state().app.active_conversation;
        chat.state_mut().app.input = "Write a long walkthrough".into();
        chat.state_mut().app.submit();
        assert!(matches!(
            chat.state().commands.try_recv(),
            Ok(Command::Send { .. })
        ));
        chat.state_mut()
            .app
            .model_event(Event::Text("Part one stopped at the reply cap.".into()));
        let mut committed = chat.state().app.committed.clone();
        committed.push(client::Message {
            role: "user",
            content: "Write a long walkthrough".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        });
        committed.push(client::Message {
            role: "assistant",
            content: "Part one stopped at the reply cap.".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        });
        chat.state_mut()
            .app
            .model_event(Event::Committed(committed));
        chat.state_mut()
            .app
            .model_event(Event::Complete(Completion::TokenLimit));
        chat.state_mut().app.input = "Keep my newer draft 🌱".into();
        chat.run();

        assert!(chat.query_by_label("Max length").is_some());
        assert!(chat.query_by_label("Model needs attention").is_none());
        assert!(chat.query_by_label("1 alert").is_none());
        let continue_rect = chat
            .get_by_role_and_label(egui::Role::Button, "Continue response")
            .rect();
        let retry_rect = chat
            .get_by_role_and_label(egui::Role::Button, "Retry response")
            .rect();
        assert!(continue_rect.left() >= 0.0 && continue_rect.right() <= retry_rect.left());
        assert!(retry_rect.right() <= size[0]);
        assert!(continue_rect.height() >= 32.0 && retry_rect.height() >= 32.0);

        chat.get_by_role_and_label(egui::Role::Button, "Continue response")
            .click();
        chat.run();
        assert!(matches!(
            chat.state().commands.try_recv(),
            Ok(Command::Send { text, .. }) if text.contains("Continue your previous response")
        ));
        assert!(chat.state().commands.try_recv().is_err());
        assert_eq!(chat.state().app.active_conversation, original);
        assert_eq!(chat.state().app.input, "Keep my newer draft 🌱");

        // Chaining another capped continuation moves Continue response to the newest
        // message while keeping Retry response on the earlier capped message.
        let continue_prompt = chat
            .state()
            .app
            .transcript
            .messages
            .iter()
            .rfind(|message| message.role == Role::User)
            .map(|message| message.body.clone())
            .unwrap_or_default();
        chat.state_mut()
            .app
            .model_event(Event::Text("Part two also hit the cap.".into()));
        let mut second_committed = chat.state().app.committed.clone();
        second_committed.push(client::Message {
            role: "user",
            content: continue_prompt.into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        });
        second_committed.push(client::Message {
            role: "assistant",
            content: "Part two also hit the cap.".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        });
        chat.state_mut()
            .app
            .model_event(Event::Committed(second_committed));
        chat.state_mut()
            .app
            .model_event(Event::Complete(Completion::TokenLimit));
        chat.run();

        // Exactly one visible Continue response button stays active on the newest turn.
        chat.get_by_role_and_label(egui::Role::Button, "Continue response")
            .click();
        chat.run();
        assert!(matches!(
            chat.state().commands.try_recv(),
            Ok(Command::Send { text, .. }) if text.contains("Continue your previous response")
        ));
        assert_eq!(chat.state().app.committed.len(), 6);
        assert_eq!(chat.state().app.input, "Keep my newer draft 🌱");
    }
}

#[test]
fn interrupted_partial_response_offers_visible_continue_and_retry_buttons() {
    for size in [[1180.0, 860.0], [480.0, 640.0]] {
        let mut chat = harness(size);
        chat.state_mut().app.input = "Explain step by step".into();
        chat.state_mut().app.submit();
        assert!(matches!(
            chat.state().commands.try_recv(),
            Ok(Command::Send { .. })
        ));
        chat.state_mut()
            .app
            .model_event(Event::Text("Step 1 of 3".into()));
        chat.state_mut()
            .app
            .model_event(Event::Stopping("Explain step by step".into()));
        chat.state_mut().app.model_event(Event::Stopped);
        chat.state_mut().app.input = "Unsent followup".into();
        chat.run();

        let continue_rect = chat
            .get_by_role_and_label(egui::Role::Button, "Continue response")
            .rect();
        let retry_rect = chat
            .get_by_role_and_label(egui::Role::Button, "Retry response")
            .rect();
        assert!(continue_rect.left() >= 0.0 && continue_rect.right() <= retry_rect.left());
        assert!(retry_rect.right() <= size[0]);
        chat.get_by_role_and_label(egui::Role::Button, "Continue response")
            .click();
        chat.run();
        assert!(matches!(
            chat.state().commands.try_recv(),
            Ok(Command::RestoreChat(history)) if history.len() == 4
        ));
        assert!(matches!(
            chat.state().commands.try_recv(),
            Ok(Command::Send { text, .. }) if text.contains("Continue your previous response")
        ));
        assert_eq!(chat.state().app.input, "Unsent followup");
    }
}

#[test]
fn worker_continuation_streams_seamlessly_without_alerting_and_token_limit_stays_quiet() {
    let mut chat = harness([1180.0, 860.0]);
    chat.state_mut().app.context_tokens = Some(131_072);
    chat.state_mut().app.input = "Write a comprehensive guide".into();
    chat.state_mut().app.submit();
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::Send {
            auto_continue: true,
            ..
        })
    ));
    chat.state_mut().app.input = "Draft for after completion 🌱".into();
    chat.state_mut()
        .app
        .model_event(Event::Text("Section 1 hit the reply cap. ".into()));
    chat.state_mut().app.model_event(Event::Continuing(2));
    chat.state_mut()
        .app
        .model_event(Event::Text("Section 2 completed the guide.".into()));
    chat.run();

    assert!(chat.query_by_label("Model needs attention").is_none());
    assert!(chat.query_by_label("1 alert").is_none());
    assert!(chat.state().app.model.busy());
    assert_eq!(chat.state().app.input, "Draft for after completion 🌱");
    assert!(chat.state().commands.try_recv().is_err());

    let mut final_committed = chat.state().app.committed.clone();
    final_committed.push(client::Message {
        role: "user",
        content: "Write a comprehensive guide".into(),
        images: std::sync::Arc::default(),
        reasoning_content: None,
    });
    final_committed.push(client::Message {
        role: "assistant",
        content: "Section 1 hit the reply cap. Section 2 completed the guide.".into(),
        images: std::sync::Arc::default(),
        reasoning_content: None,
    });
    chat.state_mut()
        .app
        .model_event(Event::Committed(final_committed));
    chat.state_mut()
        .app
        .model_event(Event::Complete(Completion::TokenLimit));
    chat.run();

    assert_eq!(chat.state().app.model, ModelLifecycle::Ready);
    assert!(chat.query_by_label("Model needs attention").is_none());
    assert!(chat.query_by_label("1 alert").is_none());
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Continue response")
            .is_some()
    );
    assert_eq!(chat.state().app.input, "Draft for after completion 🌱");
    assert!(chat.state().commands.try_recv().is_err());
}

fn wait_for_branch_diff(chat: &mut Harness<'_, Fixture>) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        chat.step();
        if chat
            .state()
            .app
            .branch_comparison
            .as_ref()
            .is_none_or(|comparison| !comparison.is_running())
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "branch diff comparison timed out"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    chat.run();
}

#[test]
fn branch_comparison_modal_completes_off_thread_and_preserves_source_on_dismiss() {
    for size in [[1180.0, 860.0], [480.0, 640.0]] {
        let mut chat = harness(size);
        let original = chat.state().app.active_conversation;
        open_actions(&mut chat, 1);
        chat.get_by_role_and_label(egui::Role::Button, "Edit and resend")
            .click();
        chat.run();
        if let Some(edit) = chat.state_mut().app.edit_message.as_mut() {
            edit.text = "Original question\r\nUpdated with 日本語 🌱".into();
        }
        chat.run();
        chat.get_by_role_and_label(egui::Role::Button, "Compare changes")
            .click();
        wait_for_branch_diff(&mut chat);

        assert!(chat.state().app.branch_comparison.is_some());
        assert!(chat.query_by_label("Compare branch and draft").is_some());
        // Unaccepted comparison never mutates draft, history, or branches.
        assert_eq!(chat.state().app.active_conversation, original);
        assert_eq!(chat.state().app.conversations.len(), 1);
        assert_eq!(chat.state().app.input, "Keep my next draft 🌱");
        assert_eq!(chat.state().app.committed.len(), 2);

        chat.get_by_role_and_label(egui::Role::Button, "Dismiss comparison")
            .click();
        chat.run();
        assert!(chat.state().app.branch_comparison.is_none());
        // Dismissing comparison returns safely to the edit dialog without changing drafts/history.
        assert!(chat.state().app.edit_message.is_some());
        assert_eq!(chat.state().app.active_conversation, original);
        assert_eq!(chat.state().app.conversations.len(), 1);
        assert_eq!(chat.state().app.input, "Keep my next draft 🌱");
        assert_eq!(chat.state().app.committed.len(), 2);
        assert!(chat.state().commands.try_recv().is_err());

        chat.key_press(egui::Key::Escape);
        chat.run();
        assert!(chat.state().app.edit_message.is_none());
        assert_eq!(chat.state().app.input, "Keep my next draft 🌱");
    }
}

#[test]
fn branch_comparison_requires_explicit_acceptance_for_draft_and_new_branch() {
    // ActiveDraft acceptance replaces only the draft after explicit button click.
    let mut chat = harness([1180.0, 860.0]);
    let original = chat.state().app.active_conversation;
    let answer_id = chat.state().app.transcript.messages[1].id;
    let ctx = chat.ctx.clone();
    assert!(
        chat.state_mut()
            .app
            .compare_message_with_draft(answer_id, Some(&ctx))
    );
    wait_for_branch_diff(&mut chat);
    assert_eq!(chat.state().app.input, "Keep my next draft 🌱");
    assert_eq!(chat.state().app.committed.len(), 2);

    // Reject stale generation acceptance cleanly.
    let generation = chat
        .state()
        .app
        .branch_comparison
        .as_ref()
        .map_or(0, iteration::BranchComparison::generation);
    assert!(!chat.state_mut().app.accept_comparison_replacement(
        iteration::AcceptComparisonReplacement {
            generation: generation.wrapping_add(99),
            side: iteration::ComparisonSide::Left,
            target: iteration::ComparisonAcceptanceTarget::ActiveDraft {
                conversation_id: original,
            },
        }
    ));
    assert_eq!(chat.state().app.input, "Keep my next draft 🌱");

    chat.get_by_role_and_label(egui::Role::Button, "Use left in draft")
        .click();
    chat.run();
    assert!(chat.state().app.branch_comparison.is_none());
    assert_eq!(chat.state().app.input, "Original answer");
    assert_eq!(chat.state().app.active_conversation, original);
    assert_eq!(chat.state().app.committed.len(), 2);
    assert!(chat.state().commands.try_recv().is_err());

    // NewBranchDraft acceptance creates a new branch with the accepted snapshot draft while
    // preserving the original conversation's history and unsent draft untouched.
    chat.state_mut().app.input = "Source draft preserved 🚀".into();
    let left = branch_diff::BranchDiffSource {
        origin: branch_diff::BranchDiffOrigin::Message {
            conversation_id: original,
            message_id: answer_id,
            history_index: Some(1),
        },
        title: "Original branch".to_owned(),
        text: "Original answer\r\n日本語".to_owned(),
    };
    let right = branch_diff::BranchDiffSource {
        origin: branch_diff::BranchDiffOrigin::Snapshot {
            conversation_id: original,
            label: "Alternate draft".to_owned(),
        },
        title: "Alternate branch".to_owned(),
        text: "Alternate answer\r\n🚀".to_owned(),
    };
    assert!(chat.state_mut().app.start_branch_comparison(
        left,
        right,
        iteration::ComparisonAcceptanceTarget::NewBranchDraft {
            source_conversation: original,
            history_prefix: 2,
        },
        Some(&ctx),
    ));
    wait_for_branch_diff(&mut chat);
    assert_eq!(chat.state().app.conversations.len(), 1);
    assert_eq!(chat.state().app.input, "Source draft preserved 🚀");

    chat.get_by_role_and_label(egui::Role::Button, "Branch with right draft")
        .click();
    chat.run();
    assert!(chat.state().app.branch_comparison.is_none());
    assert_ne!(chat.state().app.active_conversation, original);
    assert_eq!(chat.state().app.conversations.len(), 2);
    assert_eq!(chat.state().app.input, "Alternate answer\r\n🚀");
    assert_eq!(chat.state().app.committed.len(), 2);
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::RestoreChat(history)) if history.len() == 2
    ));
    assert!(chat.state().commands.try_recv().is_err());

    let saved_original = chat
        .state()
        .app
        .conversations
        .iter()
        .find(|conversation| conversation.id == original)
        .and_then(|conversation| conversation.saved.as_ref());
    assert!(saved_original.is_some_and(|snapshot| {
        snapshot.input == "Source draft preserved 🚀"
            && snapshot.committed.len() == 2
            && snapshot.committed[1].content.as_ref() == "Original answer"
    }));
}

#[test]
fn branch_comparison_oversized_error_offers_no_acceptance_and_keeps_draft_intact() {
    let mut chat = harness([1180.0, 860.0]);
    let original = chat.state().app.active_conversation;
    let ctx = chat.ctx.clone();
    let left = branch_diff::BranchDiffSource {
        origin: branch_diff::BranchDiffOrigin::Draft {
            conversation_id: original,
        },
        title: "Oversized".to_owned(),
        text: "x".repeat(INPUT_LIMIT * 4 + 1),
    };
    let right = branch_diff::BranchDiffSource {
        origin: branch_diff::BranchDiffOrigin::Draft {
            conversation_id: original,
        },
        title: "Current draft".to_owned(),
        text: "Keep my next draft 🌱".to_owned(),
    };
    assert!(chat.state_mut().app.start_branch_comparison(
        left,
        right,
        iteration::ComparisonAcceptanceTarget::ActiveDraft {
            conversation_id: original,
        },
        Some(&ctx),
    ));
    wait_for_branch_diff(&mut chat);

    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Use left in draft")
            .is_none()
    );
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Use right in draft")
            .is_none()
    );
    let generation = chat
        .state()
        .app
        .branch_comparison
        .as_ref()
        .map_or(0, iteration::BranchComparison::generation);
    assert!(!chat.state_mut().app.accept_comparison_replacement(
        iteration::AcceptComparisonReplacement {
            generation,
            side: iteration::ComparisonSide::Right,
            target: iteration::ComparisonAcceptanceTarget::ActiveDraft {
                conversation_id: original,
            },
        }
    ));

    chat.get_by_role_and_label(egui::Role::Button, "Close")
        .click();
    chat.run();
    assert!(chat.state().app.branch_comparison.is_none());
    assert_eq!(chat.state().app.input, "Keep my next draft 🌱");
    assert_eq!(chat.state().app.committed.len(), 2);
}

#[test]
fn branch_comparison_escape_and_previous_branch_comparison_preserve_sources() {
    let mut chat = harness([1180.0, 860.0]);
    let original = chat.state().app.active_conversation;
    // Create a branch so two distinct conversations exist.
    open_actions(&mut chat, 2);
    chat.get_by_role_and_label(egui::Role::Button, "Branch from here")
        .click();
    chat.run();
    let branch = chat.state().app.active_conversation;
    assert_ne!(branch, original);
    while chat.state().commands.try_recv().is_ok() {}

    chat.state_mut().app.input = "Branch draft before compare".into();
    open_actions(&mut chat, 1);
    chat.get_by_role_and_label(egui::Role::Button, "Edit and resend")
        .click();
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Compare previous branch")
        .click();
    wait_for_branch_diff(&mut chat);
    assert!(chat.query_by_label("Compare branch and draft").is_some());

    // Pressing Ctrl+N or Escape while comparison modal is open must not open a new chat
    // and Escape must only dismiss the comparison modal.
    chat.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::N);
    chat.run();
    assert_eq!(chat.state().app.conversations.len(), 2);
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert!(chat.state().app.branch_comparison.is_none());
    assert!(chat.state().app.edit_message.is_some());
    assert_eq!(chat.state().app.active_conversation, branch);
    assert_eq!(chat.state().app.input, "Branch draft before compare");
    assert!(chat.state().commands.try_recv().is_err());

    // Close edit dialog and verify "Compare with draft" path too.
    chat.get_by_role_and_label(egui::Role::Button, "Compare with draft")
        .click();
    wait_for_branch_diff(&mut chat);
    assert!(chat.query_by_label("Compare branch and draft").is_some());
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss comparison")
        .click();
    chat.run();
    assert!(chat.state().app.branch_comparison.is_none());
    assert_eq!(chat.state().app.input, "Branch draft before compare");
}
