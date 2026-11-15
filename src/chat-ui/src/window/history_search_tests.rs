//! Model-free history filtering, keyboard routing and large-list interaction.
use super::*;
use egui_kittest::{Harness, kittest::Queryable as _};

const DRAFT: &str = "Keep my next thought 🌱";

struct Fixture {
    app: State,
    commands: Receiver<Command>,
    _events: client::EventSender,
}

fn fixture() -> Fixture {
    let (commands, requests) = mpsc::channel();
    let (events, incoming) = client::event_channel();
    let mut app = State::create(commands, incoming, None, None);
    app.preview = false;
    app.model = ModelLifecycle::Ready;
    app.conversations[0].title = "Garden notes".into();
    app.transcript
        .push(Role::User, "An ÉLAN beneath the cedar trees.".into());
    app.transcript
        .push(Role::Assistant, "A quiet corner of the garden.".into());
    app.transcript.remove_empty_response();
    app.input = "An unfinished gardening draft".into();
    app.new_chat();
    app.conversations[0].title = "Voyage à Montréal".into();
    app.transcript
        .push(Role::User, "Walk beside the river.".into());
    app.transcript.push(Role::Assistant, String::new());
    app.transcript.append("PrivateCompass", true);
    app.transcript
        .append("A route through the old city.", false);
    app.transcript.remove_empty_response();
    app.input = "Unsent Cobalt itinerary".into();
    app.new_chat();
    app.conversations[0].title = "Notebook".into();
    app.transcript.push(Role::User, "Watch a seedling.".into());
    app.transcript.push(Role::Assistant, String::new());
    app.input = DRAFT.into();
    while requests.try_recv().is_ok() {}
    Fixture {
        app,
        commands: requests,
        _events: events,
    }
}

fn harness(size: [f32; 2]) -> Harness<'static, Fixture> {
    Harness::builder()
        .with_size(size)
        .with_context_setup(configure_test_context)
        .build_ui_state(
            |ui, fixture: &mut Fixture| {
                fixture.app.handle_input_shortcuts(ui.ctx());
                fixture.app.draw(ui);
            },
            fixture(),
        )
}

fn type_query(chat: &mut Harness<'_, Fixture>, query: &str) {
    chat.get_by_role_and_label(egui::Role::TextInput, "Search conversations")
        .focus();
    chat.run();
    chat.get_by_role_and_label(egui::Role::TextInput, "Search conversations")
        .type_text(query);
    chat.run();
}

fn clear_query(chat: &mut Harness<'_, Fixture>) {
    chat.get_by_role_and_label(egui::Role::Button, "Clear conversation search")
        .click();
    chat.run();
}

fn open_sidebar(chat: &mut Harness<'_, Fixture>, width: f32) {
    if width < navigation::DOCK_THRESHOLD {
        chat.get_by_role_and_label(egui::Role::Button, "Conversation menu")
            .click();
        chat.run();
    }
}

#[test]
fn find_sync_sidebar_edits_update_open_result_without_reselecting() {
    for size in [[1180.0, 860.0], [480.0, 640.0]] {
        let mut chat = harness(size);
        open_sidebar(&mut chat, size[0]);
        type_query(&mut chat, "cedar");
        chat.get_by_role_and_label(egui::Role::Button, "Garden notes")
            .click();
        chat.run();
        let selected = chat.state().app.active_conversation;
        open_sidebar(&mut chat, size[0]);
        chat.get_by_role_and_label(egui::Role::TextInput, "Search conversations")
            .focus();
        chat.run();
        chat.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        chat.get_by_role_and_label(egui::Role::TextInput, "Search conversations")
            .type_text("garden");
        chat.run();
        assert!(
            chat.get_by_role_and_label(egui::Role::TextInput, "Search conversations")
                .is_focused()
        );
        if size[0] < navigation::DOCK_THRESHOLD {
            chat.get_by_role_and_label(egui::Role::Button, "Close conversations")
                .click();
            chat.run();
        }
        assert_eq!(chat.state().app.active_conversation, selected);
        assert_eq!(
            chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
                .value()
                .as_deref(),
            Some("garden")
        );
        assert!(chat.get_by_label("1 of 1").rect().is_positive());
        assert_eq!(chat.state().app.input, "An unfinished gardening draft");
        while let Ok(command) = chat.state().commands.try_recv() {
            assert!(!matches!(command, Command::Send { .. }));
        }
    }
}

#[test]
fn find_sync_sidebar_updates_selected_chat_and_clears_without_taking_focus() {
    let mut chat = harness([1180.0, 860.0]);
    let selected = chat.state().app.active_conversation;
    type_query(&mut chat, "seed");
    assert_eq!(
        chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
            .value()
            .as_deref(),
        Some("seed")
    );
    assert!(chat.get_by_label("1 of 1").rect().is_positive());
    type_query(&mut chat, " does not exist");
    assert_eq!(
        chat.state().app.conversation_search.rows,
        Vec::<usize>::new()
    );
    assert_eq!(
        chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
            .value()
            .as_deref(),
        Some("seed does not exist")
    );
    assert!(chat.get_by_label("No matches").rect().is_positive());
    clear_query(&mut chat);
    assert_eq!(
        chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
            .value()
            .as_deref(),
        Some("")
    );
    type_query(&mut chat, "seed");
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert_eq!(
        chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
            .value()
            .as_deref(),
        Some("")
    );
    assert!(
        chat.get_by_role_and_label(egui::Role::TextInput, "Search conversations")
            .is_focused()
    );
    assert_eq!(chat.state().app.active_conversation, selected);
    assert_eq!(chat.state().app.input, DRAFT);
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn find_sync_sidebar_selection_and_reselection_open_the_same_find() {
    for size in [[1180.0, 860.0], [480.0, 640.0]] {
        let mut chat = harness(size);
        let original = chat.state().app.active_conversation;
        open_sidebar(&mut chat, size[0]);
        type_query(&mut chat, "  ÉLAN  ");
        chat.get_by_role_and_label(egui::Role::Button, "Garden notes")
            .click();
        chat.run();
        assert!(!chat.state().app.sidebar_open);
        let field = chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation");
        assert!(field.is_focused());
        assert_eq!(field.value().as_deref(), Some("ÉLAN"));
        assert!(chat.get_by_label("1 of 1").rect().is_positive());
        assert_eq!(chat.state().app.input, "An unfinished gardening draft");
        let saved = chat
            .state()
            .app
            .conversations
            .iter()
            .find(|conversation| conversation.id == original)
            .and_then(|conversation| conversation.saved.as_ref());
        assert_eq!(saved.map(|snapshot| snapshot.input.as_str()), Some(DRAFT));

        // Local edits and dismissal must not rewrite the cross-chat filter.
        field.type_text("!");
        chat.run();
        assert!(chat.get_by_label("No matches").rect().is_positive());
        assert_eq!(chat.state().app.conversation_search.text, "  ÉLAN  ");
        chat.key_press(egui::Key::Escape);
        chat.run();
        chat.run_steps(3);
        assert!(chat.query_by_label("Find in conversation").is_none());
        open_sidebar(&mut chat, size[0]);
        chat.get_by_role_and_label(egui::Role::Button, "Garden notes")
            .click();
        chat.run();
        assert!(chat.get_by_label("1 of 1").rect().is_positive());
        assert!(
            chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
                .is_focused()
        );
        while let Ok(command) = chat.state().commands.try_recv() {
            assert!(!matches!(command, Command::Send { .. }));
        }
    }
}

#[test]
fn find_sync_ctrl_f_leaves_the_drawer_and_preserves_its_filter() {
    let mut chat = harness([480.0, 640.0]);
    open_sidebar(&mut chat, 480.0);
    type_query(&mut chat, "garden");
    chat.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::F);
    chat.run();
    assert!(!chat.state().app.sidebar_open);
    assert!(
        chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
            .is_focused()
    );
    assert_eq!(chat.state().app.conversation_search.text, "garden");
    assert_eq!(chat.state().app.input, DRAFT);
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn find_sync_sidebar_and_find_share_unicode_and_long_phrase_matching() {
    for query in ["Σ".to_owned(), "cedar ".repeat(50)] {
        let mut chat = harness([1180.0, 860.0]);
        let query = query.trim();
        chat.state_mut()
            .app
            .transcript
            .replace_response(&format!("Ο{query} then Ο{query}"), None);
        chat.run();
        type_query(&mut chat, query);
        assert_eq!(chat.state().app.conversation_search.rows, [0]);
        chat.get_by_role_and_label(egui::Role::Button, "Notebook")
            .click();
        chat.run();
        assert_eq!(
            chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
                .value()
                .as_deref(),
            Some(query)
        );
        assert!(chat.get_by_label("1 of 2").rect().is_positive());
        chat.key_press(egui::Key::F3);
        chat.run();
        chat.run_steps(3);
        assert!(chat.get_by_label("2 of 2").rect().is_positive());
        chat.get_by_role_and_label(egui::Role::Button, "Notebook")
            .click();
        chat.run();
        assert!(chat.get_by_label("1 of 2").rect().is_positive());
        assert!(chat.state().commands.try_recv().is_err());
    }
}

#[test]
fn find_sync_title_only_result_shows_no_body_matches() {
    let mut chat = harness([1180.0, 860.0]);
    chat.state_mut().app.transcript.clear();
    chat.run();
    type_query(&mut chat, "Notebook");
    chat.get_by_role_and_label(egui::Role::Button, "Notebook")
        .click();
    chat.run();
    assert_eq!(
        chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
            .value()
            .as_deref(),
        Some("Notebook")
    );
    assert!(chat.get_by_label("No matches").rect().is_positive());
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn conversation_history_filters_unicode_titles_and_inactive_messages_as_you_type() {
    let mut chat = harness([1180.0, 860.0]);
    let active = chat.state().app.active_conversation;
    type_query(&mut chat, "éL");
    assert_eq!(chat.state().app.conversation_search.rows, [2]);
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Garden notes")
            .is_some()
    );
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Voyage à Montréal")
            .is_none()
    );
    type_query(&mut chat, "AN");
    assert_eq!(chat.state().app.conversation_search.rows, [2]);
    clear_query(&mut chat);
    type_query(&mut chat, "MONTRÉAL");
    assert_eq!(chat.state().app.conversation_search.rows, [1]);
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Voyage à Montréal")
            .is_some()
    );
    assert_eq!(chat.state().app.active_conversation, active);
    assert_eq!(chat.state().app.input, DRAFT);
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn conversation_history_equivalent_folded_queries_reuse_scans_and_keep_raw_changes() {
    let mut fixture = fixture();
    let app = &mut fixture.app;
    app.transcript.replace_response("ΟΣ İ then cedar", None);
    app.conversation_search.text = "  Σ İ  ".into();
    assert!(app.conversation_search.update(
        &mut app.conversations,
        app.active_conversation,
        &app.transcript,
    ));
    assert_eq!(app.conversation_search.rows, [0]);
    assert!(app.conversation_search.take_scroll_reset());
    let scans: Vec<_> = app
        .conversations
        .iter()
        .map(|conversation| conversation.search.scans)
        .collect();

    for query in ["σ i\u{307}", "\tΣ İ\n"] {
        query.clone_into(&mut app.conversation_search.text);
        // Raw changes still synchronize Find and reset the sidebar viewport.
        assert!(app.conversation_search.update(
            &mut app.conversations,
            app.active_conversation,
            &app.transcript,
        ));
        assert_eq!(app.conversation_search.rows, [0]);
        assert!(app.conversation_search.take_scroll_reset());
        assert_eq!(
            app.conversations
                .iter()
                .map(|conversation| conversation.search.scans)
                .collect::<Vec<_>>(),
            scans
        );
    }

    app.conversation_search.text = "cedar".into();
    assert!(app.conversation_search.update(
        &mut app.conversations,
        app.active_conversation,
        &app.transcript,
    ));
    assert_eq!(app.conversation_search.rows, [0, 2]);
    assert_eq!(
        app.conversations
            .iter()
            .map(|conversation| conversation.search.scans)
            .collect::<Vec<_>>(),
        scans.into_iter().map(|count| count + 1).collect::<Vec<_>>()
    );
}

#[test]
fn conversation_history_matches_never_cross_source_boundaries() -> io::Result<()> {
    let mut fixture = fixture();
    let app = &mut fixture.app;
    for (index, title, bodies) in [(2, "nee", ["dle", "nee"]), (1, "dle", ["dle", "unrelated"])] {
        let conversation = &mut app.conversations[index];
        title.clone_into(&mut conversation.title);
        let saved = conversation
            .saved
            .as_mut()
            .ok_or_else(|| io::Error::other("Saved conversation missing"))?;
        saved.transcript.clear();
        saved.transcript.push(Role::User, bodies[0].into());
        saved.transcript.push(Role::Assistant, bodies[1].into());
    }
    app.transcript.clear();
    app.transcript.push(Role::User, "nee".into());
    app.transcript.push(Role::Assistant, "dle".into());

    // No single title/body owns the whole query. Prefixes from a title, a
    // previous message or another conversation must not carry into the next.
    for (query, expected) in [
        ("NEEDLE", Vec::new()),
        ("dle", vec![0, 1, 2]),
        ("needle", Vec::new()),
    ] {
        query.clone_into(&mut app.conversation_search.text);
        app.conversation_search.update(
            &mut app.conversations,
            app.active_conversation,
            &app.transcript,
        );
        assert_eq!(app.conversation_search.rows, expected);
    }

    // Complete matches still work after those misses, first in the active
    // stream and then in a saved conversation and an independently renamed title.
    app.transcript.append(" NEEDLE", false);
    app.conversation_search.update(
        &mut app.conversations,
        app.active_conversation,
        &app.transcript,
    );
    assert_eq!(app.conversation_search.rows, [0]);
    app.conversations[2]
        .saved
        .as_mut()
        .ok_or_else(|| io::Error::other("Saved conversation missing"))?
        .transcript
        .replace_response("A whole Needle", None);
    app.conversations[1].title = "NEEDLE".into();
    app.conversation_search.update(
        &mut app.conversations,
        app.active_conversation,
        &app.transcript,
    );
    assert_eq!(app.conversation_search.rows, [0, 1, 2]);
    assert!(fixture.commands.try_recv().is_err());
    Ok(())
}

#[test]
fn conversation_history_reuses_matches_and_refreshes_streams_renames_and_replacements() {
    let mut chat = harness([1180.0, 860.0]);
    type_query(&mut chat, "new leaf");
    assert_eq!(
        chat.state().app.conversation_search.rows,
        Vec::<usize>::new()
    );
    let scans: Vec<_> = chat
        .state()
        .app
        .conversations
        .iter()
        .map(|conversation| conversation.search.scans)
        .collect();
    chat.run_steps(3);
    assert_eq!(
        chat.state()
            .app
            .conversations
            .iter()
            .map(|conversation| conversation.search.scans)
            .collect::<Vec<_>>(),
        scans
    );
    chat.state_mut().app.transcript.append("A NEW LEAF", false);
    chat.run();
    assert_eq!(chat.state().app.conversation_search.rows, [0]);
    assert_eq!(chat.state().app.conversations[0].search.scans, scans[0] + 1);
    assert_eq!(chat.state().app.conversations[1].search.scans, scans[1]);
    assert_eq!(chat.state().app.conversations[2].search.scans, scans[2]);
    chat.state_mut().app.conversations[2].title = "New leaf archive".into();
    chat.run();
    assert_eq!(chat.state().app.conversation_search.rows, [0, 2]);
    assert_eq!(chat.state().app.conversations[2].search.scans, scans[2] + 1);
    chat.state_mut()
        .app
        .transcript
        .replace_response("Other words", None);
    chat.run();
    assert_eq!(chat.state().app.conversation_search.rows, [2]);
    assert_eq!(chat.state().app.conversations[0].search.scans, scans[0] + 2);
}

#[test]
fn conversation_history_excludes_reasoning_and_drafts_and_clears_without_switching() {
    let mut chat = harness([1180.0, 860.0]);
    let active = chat.state().app.active_conversation;
    for query in ["privatecompass", "cobalt"] {
        type_query(&mut chat, query);
        assert_eq!(
            chat.state().app.conversation_search.rows,
            Vec::<usize>::new()
        );
        assert!(chat.query_by_label("No conversations found").is_some());
        clear_query(&mut chat);
        assert_eq!(chat.state().app.conversation_search.rows, [0, 1, 2]);
        assert_eq!(chat.state().app.active_conversation, active);
        assert_eq!(chat.state().app.input, DRAFT);
    }
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn conversation_history_enter_never_sends_the_composer() {
    let mut chat = harness([1180.0, 860.0]);
    type_query(&mut chat, "garden");
    chat.key_press(egui::Key::Enter);
    chat.run();
    assert_eq!(chat.state().app.model, ModelLifecycle::Ready);
    assert_eq!(chat.state().app.input, DRAFT);
    assert_eq!(chat.state().app.conversation_search.text, "garden");
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn conversation_history_escape_clears_before_drawer_dismissal_during_generation() {
    let mut chat = harness([480.0, 640.0]);
    chat.state_mut().app.model = ModelLifecycle::Generating;
    let active = chat.state().app.active_conversation;
    chat.state_mut().app.request_conversation = Some(active);
    chat.get_by_role_and_label(egui::Role::Button, "Conversation menu")
        .click();
    chat.run();
    type_query(&mut chat, "élan");
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert_eq!(chat.state().app.conversation_search.text, "");
    assert!(chat.state().app.sidebar_open);
    assert!(chat.state().app.model.busy());
    assert!(
        chat.get_by_role_and_label(egui::Role::TextInput, "Search conversations")
            .is_focused()
    );
    assert_eq!(chat.state().app.input, DRAFT);
    assert!(chat.state().commands.try_recv().is_err());
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert!(!chat.state().app.sidebar_open);
    assert!(chat.state().app.model.busy());
    assert_eq!(chat.state().app.input, DRAFT);
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn conversation_history_escape_from_another_control_preserves_the_query() {
    let mut chat = harness([480.0, 640.0]);
    chat.state_mut().app.model = ModelLifecycle::Generating;
    chat.get_by_role_and_label(egui::Role::Button, "Conversation menu")
        .click();
    chat.run();
    type_query(&mut chat, "garden");
    chat.get_by_role_and_label(egui::Role::Button, "Close conversations")
        .focus();
    chat.run();
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert!(!chat.state().app.sidebar_open);
    assert_eq!(chat.state().app.conversation_search.text, "garden");
    assert!(chat.state().app.model.busy());
    assert_eq!(chat.state().app.input, DRAFT);
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn conversation_history_large_history_selects_offscreen_results_by_stable_identity() {
    let mut chat = harness([1180.0, 860.0]);
    for index in 3..80 {
        let app = &mut chat.state_mut().app;
        app.new_chat();
        app.conversations[0].title = if index == 22 {
            "Needle far in history".into()
        } else {
            format!("Conversation {index}")
        };
        app.input = format!("Draft {index}");
    }
    let target = chat.state().app.conversations[57].id;
    let original = chat.state().app.conversations[1].id;
    let selected = chat.state().app.active_conversation;
    while chat.state().commands.try_recv().is_ok() {}
    chat.run();
    assert_eq!(chat.state().app.conversations.len(), 80);
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Needle far in history")
            .is_none()
    );
    type_query(&mut chat, "needle");
    assert_eq!(chat.state().app.conversation_search.rows, [57]);
    assert_eq!(chat.state().app.active_conversation, selected);
    chat.state_mut().app.remove_conversation(original);
    chat.run();
    assert_eq!(chat.state().app.conversation_search.rows, [56]);
    chat.get_by_role_and_label(egui::Role::Button, "Needle far in history")
        .click();
    chat.run();
    assert_eq!(chat.state().app.active_conversation, target);
    assert_eq!(chat.state().app.input, "Draft 22");
    while let Ok(command) = chat.state().commands.try_recv() {
        assert!(!matches!(command, Command::Send { .. }));
    }
}

#[test]
fn conversation_history_new_conversation_clears_the_filter_and_preserves_the_draft() {
    let mut chat = harness([1180.0, 860.0]);
    let original = chat.state().app.active_conversation;
    type_query(&mut chat, "garden");
    chat.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::N);
    chat.run();
    assert_eq!(chat.state().app.conversations.len(), 4);
    assert_ne!(chat.state().app.active_conversation, original);
    assert_eq!(chat.state().app.conversation_search.text, "");
    assert_eq!(chat.state().app.conversation_search.rows, [0, 1, 2, 3]);
    assert!(chat.query_by_label("Find in conversation").is_none());
    assert!(
        chat.get_by_role_and_label(egui::Role::MultilineTextInput, "Message")
            .is_focused()
    );
    let saved = chat
        .state()
        .app
        .conversations
        .iter()
        .find(|conversation| conversation.id == original)
        .and_then(|conversation| conversation.saved.as_ref());
    assert_eq!(saved.map(|snapshot| snapshot.input.as_str()), Some(DRAFT));
    while let Ok(command) = chat.state().commands.try_recv() {
        assert!(!matches!(command, Command::Send { .. }));
    }
}

#[test]
fn conversation_history_search_box_keeps_stable_bounds_when_typing_and_clearing() {
    for (size, sidebar_width) in [
        ([1180.0, 860.0], 256.0),
        ([1180.0, 860.0], 216.0),
        ([480.0, 640.0], 256.0),
    ] {
        let mut chat = harness(size);
        chat.state_mut().app.appearance.sidebar_width = sidebar_width;
        chat.run();
        open_sidebar(&mut chat, size[0]);

        let new_chat = chat
            .get_by_role_and_label(egui::Role::Button, "+ New conversation")
            .rect();
        let initial_input = chat
            .get_by_role_and_label(egui::Role::TextInput, "Search conversations")
            .rect();
        let initial_heading = chat.get_by_label("CONVERSATIONS").rect();

        chat.get_by_role_and_label(egui::Role::TextInput, "Search conversations")
            .focus();
        chat.run();

        for query in ["g", "arden", &" cedar".repeat(20)] {
            chat.get_by_role_and_label(egui::Role::TextInput, "Search conversations")
                .type_text(query);
            // Check both the immediate typing step and the settled frame.
            for _ in 0..2 {
                chat.step();
                let input = chat
                    .get_by_role_and_label(egui::Role::TextInput, "Search conversations")
                    .rect();
                let clear = chat
                    .get_by_role_and_label(egui::Role::Button, "Clear conversation search")
                    .rect();
                let heading = chat.get_by_label("CONVERSATIONS").rect();

                assert!(
                    (input.left() - initial_input.left()).abs() < 0.5
                        && (input.right() - initial_input.right()).abs() < 0.5
                        && (input.top() - initial_input.top()).abs() < 0.5
                        && (input.bottom() - initial_input.bottom()).abs() < 0.5,
                    "size={size:?} sidebar_width={sidebar_width} initial={initial_input:?} input={input:?}"
                );
                assert!(
                    (clear.width() - 24.0).abs() < 0.5 && (clear.height() - 24.0).abs() < 0.5,
                    "size={size:?} clear={clear:?}"
                );
                assert!(
                    (clear.left() - (input.right() + 6.0)).abs() < 0.5,
                    "size={size:?} input={input:?} clear={clear:?}"
                );
                // The search frame has 10pt horizontal inner margin and a 1pt stroke;
                // its right edge must align with the sidebar's inner right edge without clipping.
                assert!(
                    ((clear.right() + 11.0) - new_chat.right()).abs() < 0.5,
                    "size={size:?} clear_right={} new_chat_right={}",
                    clear.right(),
                    new_chat.right()
                );
                assert!(
                    (heading.top() - initial_heading.top()).abs() < 0.5
                        && (heading.bottom() - initial_heading.bottom()).abs() < 0.5,
                    "size={size:?} initial_heading={initial_heading:?} heading={heading:?}"
                );
            }
        }

        clear_query(&mut chat);
        let cleared_input = chat
            .get_by_role_and_label(egui::Role::TextInput, "Search conversations")
            .rect();
        let cleared_heading = chat.get_by_label("CONVERSATIONS").rect();
        assert!((cleared_input.width() - initial_input.width()).abs() < 0.5);
        assert!((cleared_input.height() - initial_input.height()).abs() < 0.5);
        assert!((cleared_heading.top() - initial_heading.top()).abs() < 0.5);
    }
}

#[test]
fn conversation_history_pins_responding_and_active_rows_above_non_active_scroll() {
    for size in [[1180.0, 860.0], [480.0, 640.0], [480.0, 480.0]] {
        let mut chat = harness(size);
        let garden = chat
            .state()
            .app
            .conversations
            .iter()
            .find(|conversation| conversation.title == "Garden notes")
            .map(|conversation| conversation.id)
            .unwrap_or_default();
        let voyage = chat
            .state()
            .app
            .conversations
            .iter()
            .find(|conversation| conversation.title == "Voyage à Montréal")
            .map(|conversation| conversation.id)
            .unwrap_or_default();
        for index in 3..80 {
            let app = &mut chat.state_mut().app;
            app.new_chat();
            app.conversations[0].title = format!("Conversation {index}");
            app.input = format!("Draft {index}");
        }
        chat.state_mut().app.switch_conversation(garden);
        chat.state_mut().app.input = "Respond in Garden notes".into();
        chat.state_mut().app.submit();
        assert_eq!(chat.state().app.request_conversation, Some(garden));
        chat.state_mut().app.switch_conversation(voyage);
        assert_eq!(chat.state().app.active_conversation, voyage);
        // Viewing `voyage` alone leaves it near the tail of `conversations`, yet
        // it must still be pinned immediately below the responding conversation.
        assert_eq!(
            chat.state().app.conversations.last().map(|c| c.id),
            Some(voyage)
        );
        chat.run();
        open_sidebar(&mut chat, size[0]);

        let initial_garden = chat
            .get_by_role_and_label(egui::Role::Button, "Garden notes")
            .rect();
        let initial_voyage = chat
            .get_by_role_and_label(egui::Role::Button, "Voyage à Montréal")
            .rect();
        let first_scrollable = chat
            .get_by_role_and_label(egui::Role::Button, "Conversation 79")
            .rect();
        assert!(
            chat.query_by_role_and_label(
                egui::Role::ProgressIndicator,
                "Receiving response in Garden notes"
            )
            .is_some()
        );
        assert!(initial_garden.bottom() <= initial_voyage.top());
        assert!(initial_voyage.bottom() <= first_scrollable.top());
        assert!(
            chat.query_by_role_and_label(egui::Role::Button, "Notebook")
                .is_none()
        );

        // Scroll the non-active conversation history all the way to the bottom.
        let scroll_target = egui::pos2(first_scrollable.center().x, first_scrollable.top() + 4.0);
        for _ in 0..12 {
            chat.hover_at(scroll_target);
            chat.input_mut().events.push(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, -400.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Unknown,
                modifiers: egui::Modifiers::NONE,
            });
            chat.run_steps(2);
        }

        let scrolled_garden = chat
            .get_by_role_and_label(egui::Role::Button, "Garden notes")
            .rect();
        let scrolled_voyage = chat
            .get_by_role_and_label(egui::Role::Button, "Voyage à Montréal")
            .rect();
        assert_eq!(scrolled_garden, initial_garden, "size={size:?}");
        assert_eq!(scrolled_voyage, initial_voyage, "size={size:?}");
        assert!(
            chat.query_by_role_and_label(egui::Role::Button, "Conversation 79")
                .is_none(),
            "size={size:?}"
        );
        let tail_row = chat
            .get_by_role_and_label(egui::Role::Button, "Notebook")
            .rect();
        assert!(
            tail_row.bottom() > scrolled_voyage.bottom(),
            "size={size:?} tail_row={tail_row:?} scrolled_voyage={scrolled_voyage:?}"
        );
        if size[1] >= 640.0 {
            assert!(
                tail_row.top() >= scrolled_voyage.bottom(),
                "size={size:?} tail_row={tail_row:?} scrolled_voyage={scrolled_voyage:?}"
            );
        }
        assert!(tail_row.bottom() <= size[1], "size={size:?}");
    }
}

#[test]
fn conversation_history_search_keeps_pinned_rows_visible_without_duplicating_matches() {
    let mut chat = harness([1180.0, 860.0]);
    let garden = chat
        .state()
        .app
        .conversations
        .iter()
        .find(|conversation| conversation.title == "Garden notes")
        .map(|conversation| conversation.id)
        .unwrap_or_default();
    let voyage = chat
        .state()
        .app
        .conversations
        .iter()
        .find(|conversation| conversation.title == "Voyage à Montréal")
        .map(|conversation| conversation.id)
        .unwrap_or_default();
    chat.state_mut().app.switch_conversation(garden);
    chat.state_mut().app.input = "Question in Garden".into();
    chat.state_mut().app.submit();
    chat.state_mut().app.switch_conversation(voyage);
    chat.run();

    // Query matching neither pinned row keeps both pinned rows visible above results.
    type_query(&mut chat, "notebook");
    let garden_rect = chat
        .get_by_role_and_label(egui::Role::Button, "Garden notes")
        .rect();
    let voyage_rect = chat
        .get_by_role_and_label(egui::Role::Button, "Voyage à Montréal")
        .rect();
    let notebook_rect = chat
        .get_by_role_and_label(egui::Role::Button, "Notebook")
        .rect();
    assert!(garden_rect.bottom() <= voyage_rect.top());
    assert!(voyage_rect.bottom() <= notebook_rect.top());

    // Query matching a pinned row never duplicates it in the scrollable list.
    clear_query(&mut chat);
    type_query(&mut chat, "montréal");
    assert_eq!(
        chat.query_all_by_role_and_label(egui::Role::Button, "Voyage à Montréal")
            .count(),
        1
    );
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Garden notes")
            .is_some()
    );
    assert!(chat.query_by_label("No conversations found").is_none());

    // Query matching zero conversations keeps pinned rows visible above the empty notice.
    clear_query(&mut chat);
    type_query(&mut chat, "unmatched_search_term");
    let garden_empty = chat
        .get_by_role_and_label(egui::Role::Button, "Garden notes")
        .rect();
    let voyage_empty = chat
        .get_by_role_and_label(egui::Role::Button, "Voyage à Montréal")
        .rect();
    let empty_notice = chat.get_by_label("No conversations found").rect();
    assert!(garden_empty.bottom() <= voyage_empty.top());
    assert!(voyage_empty.bottom() <= empty_notice.top());
}
