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
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Original answer".into(),
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
