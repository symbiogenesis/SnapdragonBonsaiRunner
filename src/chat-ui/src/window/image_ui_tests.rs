//! End-to-end egui UI tests for inline generated SVG bubbles, streaming, and Find fallback.
#![forbid(unsafe_code)]

use super::*;
use egui_kittest::{Harness, kittest::Queryable as _};

const SVG_DIAGRAM: &str = r##"Before diagram.

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="240" height="160"><rect width="240" height="160" fill="#2a9d8f"/></svg>
```

After diagram."##;

struct Fixture {
    app: State,
    _commands: Receiver<Command>,
    _events: client::EventSender,
}

impl Fixture {
    fn new() -> Self {
        let (commands, incoming) = mpsc::channel();
        let (events, incoming_events) = client::event_channel();
        let mut app = State::create(commands, incoming_events, None, None);
        app.preview = false;
        app.model = ModelLifecycle::Ready;
        app.status = "Ready when you are.".to_owned();
        Self {
            app,
            _commands: incoming,
            _events: events,
        }
    }
}

fn harness(size: [f32; 2], seed: impl FnOnce(&mut State)) -> Harness<'static, Fixture> {
    let mut fixture = Fixture::new();
    seed(&mut fixture.app);
    Harness::builder()
        .with_size(size)
        .with_step_dt(1.0 / 60.0)
        .with_max_steps(8)
        .with_theme(egui::Theme::Dark)
        .with_context_setup(configure_test_context)
        .build_ui_state(
            |ui, fixture: &mut Fixture| {
                fixture.app.handle_input_shortcuts(ui.ctx());
                fixture.app.draw(ui);
            },
            fixture,
        )
}

#[test]
fn assistant_svg_renders_inline_card_and_preserves_raw_source() {
    let mut chat = harness([1080.0, 780.0], |app| {
        app.transcript
            .push(Role::User, "Draw a teal rectangle diagram.".to_owned());
        app.transcript.push(Role::Assistant, SVG_DIAGRAM.to_owned());
        app.transcript.remove_empty_response();
        app.last_answer = SVG_DIAGRAM.to_owned();
        app.answer_complete = true;
    });
    chat.run();

    assert!(
        chat.get_by_role_and_label(egui::Role::Image, "Generated image")
            .rect()
            .is_positive()
    );
    assert!(
        chat.get_by_role_and_label(egui::Role::Button, "Save generated image")
            .rect()
            .is_positive()
    );
    assert!(chat.query_by_label(SVG_DIAGRAM).is_none());
    assert!(chat.get_by_label("Before diagram.").rect().is_positive());
    assert!(chat.get_by_label("After diagram.").rect().is_positive());
    assert_eq!(chat.state().app.transcript.messages[1].body, SVG_DIAGRAM);
}

#[test]
fn streaming_open_svg_fence_shows_generating_card_until_closed() {
    let mut chat = harness([1080.0, 780.0], |app| {
        app.model = ModelLifecycle::Generating;
        app.request_conversation = Some(app.active_conversation);
        app.transcript
            .push(Role::User, "Generate an SVG badge.".to_owned());
        app.transcript.push(
            Role::Assistant,
            "Here is the badge:\n\n```svg\n<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"120\" height=\"80\"><circle cx=\"60\""
                .to_owned(),
        );
    });
    chat.run_steps(4);

    assert!(chat.get_by_label("Generating image…").rect().is_positive());
    assert!(
        chat.query_by_role_and_label(egui::Role::Image, "Generated image")
            .is_none()
    );

    chat.state_mut().app.transcript.append(
        " cy=\"40\" r=\"30\" fill=\"teal\"/></svg>\n```\n\nDone.",
        false,
    );
    chat.state_mut().app.transcript.remove_empty_response();
    chat.state_mut().app.model = ModelLifecycle::Ready;
    chat.run();

    assert!(chat.query_by_label("Generating image…").is_none());
    assert!(
        chat.get_by_role_and_label(egui::Role::Image, "Generated image")
            .rect()
            .is_positive()
    );
}

#[test]
fn find_match_inside_svg_falls_back_to_raw_source_and_restores_image_on_close() {
    let mut chat = harness([1080.0, 780.0], |app| {
        app.transcript
            .push(Role::User, "Render the teal rectangle.".to_owned());
        app.transcript.push(Role::Assistant, SVG_DIAGRAM.to_owned());
        app.transcript.remove_empty_response();
    });
    chat.run();
    assert!(
        chat.get_by_role_and_label(egui::Role::Image, "Generated image")
            .rect()
            .is_positive()
    );

    chat.state_mut().app.transcript.focus_find();
    chat.run();
    chat.get_by_role_and_label(egui::Role::TextInput, "Find in conversation")
        .type_text("#2a9d8f");
    chat.run();

    assert!(chat.get_by_label("1 of 1").rect().is_positive());
    assert!(
        chat.query_by_role_and_label(egui::Role::Image, "Generated image")
            .is_none()
    );

    chat.key_press(egui::Key::Escape);
    chat.run();

    assert!(chat.query_by_label("Find in conversation").is_none());
    assert!(
        chat.get_by_role_and_label(egui::Role::Image, "Generated image")
            .rect()
            .is_positive()
    );
}
