//! Model-free interaction coverage for navigation and reading ergonomics.
use super::*;
use egui_kittest::{Harness, kittest::Queryable as _};

struct Fixture {
    app: State,
    commands: Receiver<Command>,
    _events: client::EventSender,
}

fn harness(size: [f32; 2], render: bool) -> Harness<'static, Fixture> {
    harness_with_preferences(size, render, appearance::Preferences::default())
}

fn harness_with_preferences(
    size: [f32; 2],
    render: bool,
    preferences: appearance::Preferences,
) -> Harness<'static, Fixture> {
    let (commands, requests) = mpsc::channel();
    let (events, incoming) = client::event_channel();
    let mut app = State::create(commands, incoming, None, None);
    app.appearance = preferences;
    app.preview = false;
    app.model = ModelLifecycle::Ready;
    app.input = "Keep my next thought 🌱".into();
    app.transcript
        .push(Role::User, "Help me make a little room to think.".into());
    app.transcript.push(Role::Assistant, "# A fresh perspective\n\nLeave a little space between ideas.\n\n- Start with what matters.\n- Let the details follow.\n\n`Small steps add up.`".into());
    app.transcript.remove_empty_response();
    let builder = Harness::builder()
        .with_size(size)
        .with_context_setup(configure_test_context);
    let builder = if render {
        let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
        setup.instance_descriptor.backends = wgpu::Backends::DX12;
        builder.wgpu_setup(eframe::egui_wgpu::WgpuSetup::CreateNew(setup))
    } else {
        builder
    };
    builder.build_ui_state(
        |ui, fixture: &mut Fixture| {
            // `build_ui_state` adds an 8-point widget-test outer margin; match
            // `eframe::App::ui`, which draws across the full `content_rect`.
            let content_rect = ui.ctx().content_rect();
            let mut root = ui.new_child(egui::UiBuilder::new().max_rect(content_rect));
            root.set_clip_rect(content_rect);
            fixture.app.handle_input_shortcuts(root.ctx());
            fixture.app.draw(&mut root);
        },
        Fixture {
            app,
            commands: requests,
            _events: events,
        },
    )
}

#[test]
fn conversation_tools_leave_room_for_messages_and_keep_find_and_navigation_accessible() {
    for size in [[1180.0, 860.0], [480.0, 480.0]] {
        let mut chat = harness(size, false);
        chat.state_mut().app.transcript.clear();
        chat.state_mut()
            .app
            .transcript
            .push(Role::User, "Help me make a little room to think.".into());
        chat.state_mut()
            .app
            .transcript
            .push(Role::Assistant, "A little space for ideas.".into());
        chat.run();
        let bounds = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::from(size));
        let navigation = if size[0] < navigation::DOCK_THRESHOLD {
            "Conversation menu"
        } else {
            "Hide conversations"
        };
        let toggle = chat
            .get_by_role_and_label(egui::Role::Button, navigation)
            .rect();
        let composer = chat
            .get_by_role_and_label(egui::Role::MultilineTextInput, "Message")
            .rect();
        assert!(bounds.contains_rect(toggle) && bounds.contains_rect(composer));
        assert!(
            bounds.contains_rect(
                chat.get_by_role_and_label(egui::Role::Button, "Send  ↑")
                    .rect()
            )
        );
        assert!(toggle.right() < composer.left());
        assert!(
            chat.query_by_role_and_label(egui::Role::Button, "Find")
                .is_none()
        );
        let first_action = format!(
            "Actions for message {}",
            chat.state().app.transcript.messages[0].id
        );
        let second_action = format!(
            "Actions for message {}",
            chat.state().app.transcript.messages[1].id
        );
        let first_top = chat
            .get_by_role_and_label(egui::Role::Button, &first_action)
            .rect()
            .top()
            - transcript::MESSAGE_INSET.y;
        let first_bottom = first_top + chat.state().app.transcript.messages[0].height();
        let second_top = chat
            .get_by_role_and_label(egui::Role::Button, &second_action)
            .rect()
            .top()
            - transcript::MESSAGE_INSET.y;
        assert!((first_top - transcript::MESSAGE_GAP).abs() < 0.5);
        assert!((second_top - first_bottom - first_top).abs() < 0.5);
        assert!(
            chat.get_by_label("Help me make a little room to think.")
                .rect()
                .top()
                < 100.0
        );
        assert!(chat.query_by_label("Find in conversation").is_none());
        chat.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::F);
        chat.run();
        let field = chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation");
        assert!(field.is_focused());
        field.type_text("little");
        chat.run();
        assert!(chat.get_by_label("1 of 2").rect().is_positive());
        chat.key_press(egui::Key::Escape);
        chat.run();
        assert!(chat.query_by_label("Find in conversation").is_none());
        chat.get_by_role_and_label(egui::Role::Button, navigation)
            .click();
        chat.run();
        if size[0] < navigation::DOCK_THRESHOLD {
            assert!(chat.state().app.sidebar_open);
            chat.key_press(egui::Key::Escape);
        } else {
            assert!(!chat.state().app.appearance.sidebar_visible);
            chat.get_by_role_and_label(egui::Role::Button, "Show conversations")
                .click();
        }
        chat.run();
        chat.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::F);
        chat.run();
        assert!(
            chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
                .is_focused()
        );
        chat.get_by_role_and_label(egui::Role::Button, "Close find")
            .click();
        chat.run();
        assert!(chat.query_by_label("Find in conversation").is_none());
        assert!(
            chat.get_by_role_and_label(egui::Role::MultilineTextInput, "Message")
                .is_focused()
        );
        assert_eq!(chat.state().app.input, "Keep my next thought 🌱");
        assert!(chat.state().commands.try_recv().is_err());
    }
}

#[test]
fn ergonomics_sidebar_resizes_collapses_and_reopens_with_the_draft_intact() {
    let mut chat = harness([1180.0, 860.0], false);
    let start = egui::pos2(chat.state().app.appearance.sidebar_width, 260.0);
    let end = start + egui::vec2(48.0, 0.0);
    chat.hover_at(start);
    chat.drag_at(start);
    chat.run_steps(1);
    chat.hover_at(end);
    chat.run_steps(1);
    chat.drop_at(end);
    chat.run();
    assert!(chat.state().app.appearance.sidebar_width > 280.0);
    chat.get_by_role_and_label(egui::Role::Button, "Hide conversations")
        .click();
    chat.run();
    assert!(!chat.state().app.appearance.sidebar_visible);
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "+ New conversation")
            .is_none()
    );
    chat.get_by_role_and_label(egui::Role::Button, "Show conversations")
        .click();
    chat.run();
    assert!(chat.state().app.appearance.sidebar_visible);
    assert_eq!(chat.state().app.input, "Keep my next thought 🌱");
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn ergonomics_sidebar_keeps_manual_shrink_after_release_and_layout_changes() -> io::Result<()> {
    let mut chat = harness([1180.0, 860.0], false);
    chat.state_mut().app.conversations[0].title = "A long conversation title ".repeat(16);
    chat.run();
    let start = egui::pos2(chat.state().app.appearance.sidebar_width, 260.0);
    let end = egui::pos2(216.0, start.y);
    chat.hover_at(start);
    chat.drag_at(start);
    chat.run_steps(1);
    chat.hover_at(end);
    chat.run_steps(1);
    chat.drop_at(end);
    chat.run();
    chat.run_steps(8);
    assert!((chat.state().app.appearance.sidebar_width - 216.0).abs() < 1.0);
    assert!(
        chat.get_by_role_and_label(egui::Role::Button, "+ New conversation")
            .rect()
            .right()
            < 216.0
    );

    chat.state_mut().app.model = ModelLifecycle::Generating;
    chat.state_mut().app.save_conversations = false;
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Hide conversations")
        .click();
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Show conversations")
        .click();
    chat.run();
    chat.set_size(egui::vec2(480.0, 640.0));
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Conversation menu")
        .click();
    chat.run();
    chat.set_size(egui::vec2(1180.0, 860.0));
    chat.run();
    chat.run_steps(8);
    assert!((chat.state().app.appearance.sidebar_width - 216.0).abs() < 1.0);
    assert!(chat.state().app.model.busy());
    assert_eq!(chat.state().app.input, "Keep my next thought 🌱");
    assert!(chat.state().commands.try_recv().is_err());

    let saved = serde_json::to_vec(&chat.state().app.appearance).map_err(io::Error::other)?;
    let preferences = serde_json::from_slice(&saved).map_err(io::Error::other)?;
    let mut restored = harness_with_preferences([1180.0, 860.0], false, preferences);
    restored.run();
    assert!((restored.state().app.appearance.sidebar_width - 216.0).abs() < 1.0);
    Ok(())
}

#[test]
fn ergonomics_drawer_escape_preserves_generation_and_returns_to_the_draft() {
    let mut chat = harness([480.0, 480.0], false);
    chat.state_mut().app.model = ModelLifecycle::Generating;
    chat.get_by_role_and_label(egui::Role::Button, "Conversation menu")
        .click();
    chat.run();
    assert!(chat.state().app.sidebar_open);
    let close = chat
        .get_by_role_and_label(egui::Role::Button, "Close conversations")
        .rect();
    assert!(close.right() < 400.0 && close.top() < 50.0);
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert!(!chat.state().app.sidebar_open);
    assert!(chat.state().app.model.busy());
    assert_eq!(chat.state().app.input, "Keep my next thought 🌱");
    assert!(chat.state().commands.try_recv().is_err());
    let composer = chat.get_by_role_and_label(egui::Role::MultilineTextInput, "Message");
    assert!(composer.is_focused());
}

#[test]
fn ergonomics_hidden_dock_dismisses_the_drawer_after_resizing_wide() {
    let mut chat = harness([480.0, 640.0], false);
    chat.state_mut().app.appearance.sidebar_visible = false;
    chat.get_by_role_and_label(egui::Role::Button, "Conversation menu")
        .click();
    chat.run();
    assert!(chat.state().app.sidebar_open);
    chat.set_size(egui::vec2(1180.0, 860.0));
    chat.run();
    assert!(!chat.state().app.sidebar_open);
    assert!(!chat.state().app.appearance.sidebar_visible);
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Close conversations")
            .is_none()
    );
    chat.state_mut().app.model = ModelLifecycle::Generating;
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::Stop)
    ));
}

#[test]
fn ergonomics_largest_reader_and_appearance_settings_fit_the_minimum_window() {
    let mut chat = harness([480.0, 480.0], false);
    chat.state_mut().app.appearance.text_size = 24;
    chat.state().app.appearance.apply(&chat.ctx);
    chat.run();
    let bounds = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(480.0, 480.0));
    assert!(
        bounds.contains_rect(
            chat.get_by_role_and_label(egui::Role::Button, "Send  ↑")
                .rect()
        )
    );
    chat.get_by_role_and_label(egui::Role::Button, "Conversation menu")
        .click();
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Settings")
        .click();
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Appearance")
        .click();
    chat.run();
    assert!(
        bounds.contains_rect(
            chat.get_by_role_and_label(egui::Role::Button, "Done")
                .rect()
        )
    );
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert!(!chat.state().app.show_settings);
    assert_eq!(chat.state().app.input, "Keep my next thought 🌱");
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn workspace_halves_side_padding_and_keeps_conversation_scrollbar_at_window_edge() {
    for (size, reading_width, expected_padding) in [
        ([480.0, 480.0], 800_u16, 6.0_f32),
        ([960.0, 640.0], 800_u16, 12.0_f32),
        ([1180.0, 640.0], 1_040_u16, 12.0_f32),
        ([1180.0, 640.0], 800_u16, 12.0_f32),
    ] {
        let mut preferences = appearance::Preferences::default();
        preferences.reading_width = reading_width;
        let mut chat = harness_with_preferences(size, false, preferences);
        chat.state_mut().app.transcript.clear();
        for index in 0..15 {
            let role = if index % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            };
            chat.state_mut().app.transcript.push(
                role,
                format!("Turn {index} line one\nTurn {index} line two"),
            );
        }
        let anchor_body = "Bottom anchor message for padding check.";
        chat.state_mut()
            .app
            .transcript
            .push(Role::Assistant, anchor_body.into());
        chat.state_mut().app.transcript.remove_empty_response();
        chat.run();
        chat.run_steps(4);

        let nav_right = if size[0] < navigation::DOCK_THRESHOLD {
            44.0
        } else {
            chat.state().app.appearance.sidebar_width
        };
        let message_input = chat
            .get_by_role_and_label(egui::Role::MultilineTextInput, "Message")
            .rect();
        let send_button = chat
            .get_by_role_and_label(egui::Role::Button, "Send  ↑")
            .rect();
        let composer_left = message_input.left() - 14.0;
        let composer_right = send_button.right() + 14.0;
        let composer_top = message_input.top() - 14.0;
        if size[0] - nav_right - 2.0 * expected_padding <= f32::from(reading_width) {
            assert!(
                (composer_left - nav_right - expected_padding).abs() < 0.5,
                "size={size:?} composer_left={composer_left} nav_right={nav_right}"
            );
            assert!(
                (size[0] - composer_right - expected_padding).abs() < 0.5,
                "size={size:?} composer_right={composer_right}"
            );
        } else {
            let left_gap = composer_left - nav_right;
            let right_gap = size[0] - composer_right;
            assert!((left_gap - right_gap).abs() < 0.5);
            assert!(left_gap > expected_padding);
        }

        let visible_cards: Vec<(egui::Rect, egui::Rect)> = chat
            .state()
            .app
            .transcript
            .messages
            .iter()
            .filter_map(|message| {
                let actions_label = format!("Actions for message {}", message.id);
                let body = chat.query_by_label(&message.body)?;
                let actions = chat.query_by_role_and_label(egui::Role::Button, &actions_label)?;
                Some((body.rect(), actions.rect()))
            })
            .collect();
        let scrollbar = chat.get_by_role(egui::Role::ScrollBar).rect();
        assert!(
            !visible_cards.is_empty(),
            "size={size:?} expected at least one visible message card"
        );
        for (body_rect, actions_rect) in visible_cards {
            let card_left = body_rect.left() - transcript::MESSAGE_INSET.x;
            let card_right = actions_rect.right() + transcript::MESSAGE_INSET.x;
            assert!(
                (card_left - composer_left).abs() < 0.5,
                "size={size:?} card_left={card_left} composer_left={composer_left}"
            );
            assert!(
                (card_right - composer_right).abs() < 0.5,
                "size={size:?} card_right={card_right} composer_right={composer_right}"
            );
            assert!(scrollbar.right() > card_right);
        }

        if let Some(latest) = chat.query_by_role_and_label(egui::Role::Button, "Latest") {
            let latest_center = latest.rect().center().x;
            let composer_center = (composer_left + composer_right) * 0.5;
            assert!(
                (latest_center - composer_center).abs() < 0.5,
                "size={size:?} latest_center={latest_center} composer_center={composer_center}"
            );
        }

        assert!(
            (scrollbar.right() - size[0]).abs() < 0.5,
            "size={size:?} scrollbar={scrollbar:?}"
        );
        let context_button = chat
            .get_by_role_and_label(egui::Role::Button, "Context")
            .rect();
        assert!(scrollbar.bottom() < composer_top);
        assert!(scrollbar.bottom() <= context_button.top());

        chat.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::F);
        chat.run();
        let find_input = chat
            .get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
            .rect();
        assert!(
            (find_input.left() - nav_right - expected_padding).abs() < 1.5,
            "size={size:?} find_left={} nav_right={nav_right}",
            find_input.left()
        );
    }
}

#[test]
fn ergonomics_capture_when_requested() -> io::Result<()> {
    let Some(directory) = std::env::var_os("BONSAI_ERGONOMICS_CAPTURE_DIRECTORY") else {
        return Ok(());
    };
    let directory = std::path::PathBuf::from(directory);
    bonsai_windows_credentials::validate_external_path(&directory)?;
    std::fs::create_dir_all(&directory)?;
    for (name, size, theme) in [
        ("light-wide.png", [1080.0, 780.0], appearance::Theme::Light),
        (
            "conversation-minimum.png",
            [480.0, 480.0],
            appearance::Theme::Dark,
        ),
        ("hidden-find.png", [1080.0, 780.0], appearance::Theme::Dark),
        ("dark-drawer.png", [480.0, 640.0], appearance::Theme::Dark),
        (
            "appearance-minimum.png",
            [480.0, 480.0],
            appearance::Theme::Light,
        ),
    ] {
        let mut chat = harness(size, true);
        chat.state_mut().app.appearance.theme = theme;
        chat.state().app.appearance.apply(&chat.ctx);
        if name == "dark-drawer.png" {
            chat.state_mut().app.sidebar_open = true;
        } else if name == "hidden-find.png" {
            chat.state_mut().app.appearance.sidebar_visible = false;
            chat.state_mut().app.transcript.find_query("little");
        } else if name == "appearance-minimum.png" {
            chat.state_mut()
                .app
                .open_settings(settings::Page::Appearance);
        }
        chat.run();
        chat.render()
            .map_err(io::Error::other)?
            .save(directory.join(name))
            .map_err(io::Error::other)?;
    }
    Ok(())
}
