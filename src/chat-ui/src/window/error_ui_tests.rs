//! Exercise shared alert selection, exact copy payloads and modal isolation.
use super::*;
use egui_kittest::{Harness, kittest::Queryable as _};

struct AlertFixture {
    app: State,
    commands: Receiver<Command>,
    copied: Option<String>,
}

fn harness(size: [f32; 2], font_size: f32) -> Harness<'static, AlertFixture> {
    let (commands, incoming) = mpsc::channel();
    let (_events, incoming_events) = client::event_channel();
    let mut app = State::create(commands, incoming_events, None, None);
    app.preview = false;
    app.model = ModelLifecycle::Ready;
    app.input = "Keep this draft".to_owned();
    Harness::builder()
        .with_size(size)
        .with_context_setup(move |ctx| {
            configure_test_context(ctx);
            ctx.global_style_mut(|style| {
                style
                    .text_styles
                    .insert(egui::TextStyle::Body, egui::FontId::proportional(font_size));
            });
        })
        .build_ui_state(
            |ui, fixture: &mut AlertFixture| {
                fixture.app.handle_input_shortcuts(ui.ctx());
                fixture.app.draw(ui);
                ui.ctx().output(|output| {
                    for command in &output.commands {
                        if let egui::OutputCommand::CopyText(text) = command {
                            fixture.copied = Some(text.clone());
                        }
                    }
                });
            },
            AlertFixture {
                app,
                commands: incoming,
                copied: None,
            },
        )
}

fn long_error() -> String {
    format!(
        "The operation could not finish.\r\n{}\nFinal cause: permission denied. 日本語 🌱",
        "C:\\a-very-long-path-without-spaces\\日本語\\artifact: nested cause\n".repeat(1_000)
    )
}

fn collect_paint_shapes<'a>(shape: &'a egui::Shape, shapes: &mut Vec<&'a egui::Shape>) {
    if let egui::Shape::Vec(children) = shape {
        for child in children {
            collect_paint_shapes(child, shapes);
        }
    } else {
        shapes.push(shape);
    }
}

fn luminance(color: egui::Color32) -> f32 {
    let linear = egui::Rgba::from(color);
    linear
        .r()
        .mul_add(0.2126, linear.g().mul_add(0.7152, 0.0722 * linear.b()))
}

fn assert_popup_text_contrast(chat: &Harness<'_, AlertFixture>, title: &str) -> io::Result<()> {
    let row = chat.get_by_role_and_label(egui::Role::Button, title).rect();
    let mut shapes = Vec::new();
    for clipped in &chat.output().shapes {
        collect_paint_shapes(&clipped.shape, &mut shapes);
    }
    let mut background = None;
    let mut checked = 0;
    for shape in shapes {
        match shape {
            egui::Shape::Rect(rect)
                if rect.fill.is_opaque() && rect.rect.contains(row.center()) =>
            {
                background = Some(rect.fill);
            }
            egui::Shape::Text(text) if row.contains(text.pos) && text.galley.job.text == title => {
                let background = luminance(background.ok_or_else(|| {
                    io::Error::other("Popup row does not have a painted background")
                })?);
                for section in &text.galley.job.sections {
                    let foreground = text.override_text_color.unwrap_or_else(|| {
                        if section.format.color == egui::Color32::PLACEHOLDER {
                            text.fallback_color
                        } else {
                            section.format.color
                        }
                    });
                    let foreground = luminance(foreground);
                    let contrast =
                        (foreground.max(background) + 0.05) / (foreground.min(background) + 0.05);
                    assert!(contrast >= 4.5, "{title}: rendered contrast was {contrast}");
                    checked += 1;
                }
            }
            _ => {}
        }
    }
    assert!(checked > 0, "Popup row text was actually painted");
    Ok(())
}

#[test]
fn error_reporting_selected_popup_text_has_contrast_when_idle_and_hovered() -> io::Result<()> {
    for theme in [egui::ThemePreference::Light, egui::ThemePreference::Dark] {
        let mut chat = harness([480.0, 480.0], 24.0);
        chat.ctx.set_theme(theme);
        chat.state_mut().app.set_status("Model failed", true);
        chat.state_mut().app.speech_notice = true;
        chat.state_mut().app.speech_status = "Speech failed".into();
        chat.run();
        chat.get_by_role_and_label(egui::Role::ComboBox, "Current alert")
            .click();
        chat.run();
        assert_popup_text_contrast(&chat, "Model needs attention")?;
        chat.get_by_role_and_label(egui::Role::Button, "Model needs attention")
            .hover();
        chat.run();
        assert_popup_text_contrast(&chat, "Model needs attention")?;
    }
    Ok(())
}

#[test]
fn error_reporting_single_alert_omits_selector_and_prioritizes_recovery_in_tab_order() {
    let mut chat = harness([1080.0, 780.0], 16.0);
    chat.state_mut().app.set_status("Model failed", true);
    chat.state_mut().app.model = ModelLifecycle::Retryable;
    chat.run();
    assert!(chat.get_by_label("1 alert").rect().is_positive());
    assert!(
        chat.get_by_label("Model needs attention")
            .rect()
            .is_positive()
    );
    assert!(
        chat.query_by_role_and_label(egui::Role::ComboBox, "Current alert")
            .is_none()
    );
    chat.get_by_role_and_label(egui::Role::Button, "Retry model")
        .focus();
    chat.run();
    chat.key_press(egui::Key::Tab);
    chat.run();
    assert!(
        chat.get_by_role_and_label(egui::Role::Button, "View details")
            .is_focused()
    );
    chat.key_press(egui::Key::Tab);
    chat.run();
    assert!(
        chat.get_by_role_and_label(egui::Role::Button, "Copy error")
            .is_focused()
    );
    assert!(chat.state().commands.try_recv().is_err());
    assert_eq!(chat.state().app.input, "Keep this draft");
}

#[test]
fn error_reporting_offers_windows_update_action_only_for_sandbox_os_alerts() {
    let mut chat = harness([1080.0, 780.0], 16.0);
    chat.state_mut().app.set_status("Model failed", true);
    chat.state_mut().app.model = ModelLifecycle::Retryable;
    chat.run();
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Open Windows Update")
            .is_none()
    );

    chat.state_mut().app.fail_startup(&io::Error::other(
        "Required Windows sandbox (PSEC 1.1) is unavailable because a Windows Update restart is pending.",
    ));
    chat.run();
    assert!(
        chat.get_by_role_and_label(egui::Role::Button, "Open Windows Update")
            .rect()
            .is_positive()
    );
    assert_eq!(chat.state().app.input, "Keep this draft");
}

#[test]
fn error_reporting_popup_rows_wrap_and_fill_the_width_at_minimum_size() {
    let mut chat = harness([480.0, 480.0], 24.0);
    chat.state_mut().app.set_status("Model failed", true);
    chat.state_mut().app.recovery_notice = "Could not save conversations".into();
    chat.run();
    chat.get_by_role_and_label(egui::Role::ComboBox, "Current alert")
        .click();
    chat.run();
    let model = chat
        .get_by_role_and_label(egui::Role::Button, "Model needs attention")
        .rect();
    let recovery = chat
        .get_by_role_and_label(egui::Role::Button, "Conversation saving needs attention")
        .rect();
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(480.0, 480.0));
    assert!(viewport.contains_rect(model) && viewport.contains_rect(recovery));
    assert!((model.width() - recovery.width()).abs() < 1.0);
    assert!(recovery.height() > model.height());
}

#[test]
fn error_reporting_shared_panel_selects_errors_and_copies_every_character() {
    for (size, theme) in [
        ([480.0, 480.0], egui::Theme::Light),
        ([1080.0, 780.0], egui::Theme::Dark),
    ] {
        let model = long_error();
        let speech = format!("Speech failed: {}", long_error());
        let mut chat = harness(size, 24.0);
        chat.ctx.set_theme(match theme {
            egui::Theme::Light => egui::ThemePreference::Light,
            egui::Theme::Dark => egui::ThemePreference::Dark,
        });
        chat.state_mut().app.set_status(&model, true);
        chat.state_mut().app.speech_notice = true;
        chat.state_mut().app.speech_status = speech.clone();
        chat.run();
        assert!(chat.get_by_label("2 alerts").rect().is_positive());
        chat.get_by_role_and_label(egui::Role::Button, "Copy error")
            .click();
        chat.run();
        assert_eq!(chat.state().copied.as_deref(), Some(model.as_str()));
        chat.get_by_role_and_label(egui::Role::ComboBox, "Current alert")
            .click();
        chat.run();
        chat.get_by_label("Speech needs attention").click();
        chat.run();
        assert_eq!(
            chat.state().app.alerts.selection,
            Some(error_ui::AlertKind::Speech)
        );
        chat.get_by_role_and_label(egui::Role::Button, "Copy error")
            .click();
        chat.run();
        assert_eq!(chat.state().copied.as_deref(), Some(speech.as_str()));
        chat.get_by_role_and_label(egui::Role::Button, "View details")
            .click();
        chat.run();
        assert!(chat.get_by_label("Error details").rect().is_positive());
        chat.get_by_role_and_label(egui::Role::Button, "Copy full error")
            .click();
        chat.run();
        assert_eq!(chat.state().copied.as_deref(), Some(speech.as_str()));
        chat.get_by_role_and_label(egui::Role::Button, "Close")
            .click();
        chat.run();
        assert_eq!(chat.state().app.alerts.details, None);
        let composer = chat
            .get_by_role_and_label(egui::Role::MultilineTextInput, "Message")
            .rect();
        assert!(egui::Rect::from_min_size(egui::Pos2::ZERO, size.into()).contains_rect(composer));
        assert_eq!(chat.state().app.input, "Keep this draft");
    }
}

#[test]
fn error_reporting_current_status_keeps_details_without_breaking_recovery() -> io::Result<()> {
    let (commands, _incoming) = mpsc::channel();
    let (_events, incoming) = client::event_channel();
    let mut state = State::create(commands, incoming, None, None);
    let message = format!(
        "{}\nProtected credentials and technical evidence: final detail",
        long_error()
    );
    state.set_status(&message, true);
    assert_eq!(state.status_detail, message);
    assert!(state.status.chars().count() <= MAX_DIAGNOSTIC_CHARS);
    assert!(
        state
            .diagnostics
            .back()
            .is_some_and(|entry| entry.chars().count() <= MAX_DIAGNOSTIC_CHARS)
    );
    let saved: serde_json::Value =
        serde_json::from_slice(&state.workspace_bytes()?).map_err(io::Error::other)?;
    assert!(saved["active"].get("status_detail").is_none());
    Ok(())
}

#[test]
fn error_reporting_multiple_failures_preserve_selection_and_audio_actions() {
    let mut chat = harness([480.0, 480.0], 24.0);
    let app = &mut chat.state_mut().app;
    app.fail_startup(&io::Error::other(long_error()));
    app.setup.speech_error = Some(long_error());
    app.speech_notice = true;
    app.speech_status = long_error();
    app.audio_file_error = true;
    app.audio_file_notice = Some(format!("Audio import failed: {}", long_error()));
    app.audio_file_transcript = "Retained partial transcript 日本語 🌱".to_owned();
    app.recovery_notice = format!("Could not save conversations: {}", long_error());
    app.alerts.selection = Some(error_ui::AlertKind::AudioFile);
    chat.run();
    assert!(chat.get_by_label("4 alerts").rect().is_positive());
    assert!(
        chat.get_by_role_and_label(egui::Role::Button, "View transcript")
            .rect()
            .is_positive()
    );
    chat.get_by_role_and_label(egui::Role::Button, "Copy transcript")
        .click();
    chat.run();
    assert_eq!(
        chat.state().copied.as_deref(),
        Some("Retained partial transcript 日本語 🌱")
    );
    // Removing an earlier source must not select a different surviving alert.
    chat.state_mut().app.recovery_notice.clear();
    chat.run();
    assert_eq!(
        chat.state().app.alerts.selection,
        Some(error_ui::AlertKind::AudioFile)
    );
    assert!(chat.get_by_label("3 alerts").rect().is_positive());
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss alert")
        .click();
    chat.run();
    assert_eq!(
        chat.state().app.alerts.selection,
        Some(error_ui::AlertKind::Setup)
    );
    assert!(chat.state().app.audio_file_notice.is_none());
    assert_eq!(chat.state().app.audio_file_transcript, "");
    assert!(!chat.state().app.audio_file_error);
    assert!(chat.get_by_label("2 alerts").rect().is_positive());
    let composer = chat
        .get_by_role_and_label(egui::Role::MultilineTextInput, "Message")
        .rect();
    assert!(composer.min.y >= 0.0 && composer.max.y <= 480.0);
    assert!(composer.width() > 200.0 && composer.height() > 20.0);
    assert_eq!(chat.state().app.input, "Keep this draft");
}

#[test]
fn error_reporting_sources_suppress_duplicates_and_resolve_to_empty() {
    use error_ui::AlertKind;
    let mut chat = harness([480.0, 480.0], 16.0);
    let app = &mut chat.state_mut().app;
    app.set_status("Model failed", true);
    app.setup.error = Some("Setup failed".into());
    app.setup.speech_error = Some("Speech setup failed".into());
    app.speech_notice = true;
    app.speech_status = "Speech failed".into();
    assert_eq!(
        app.active_alerts(),
        vec![AlertKind::Setup, AlertKind::SpeechSetup]
    );
    app.setup.speech_error = None;
    app.audio_file_error = true;
    app.audio_file_notice = Some(app.speech_status.clone());
    assert_eq!(
        app.active_alerts(),
        vec![AlertKind::Setup, AlertKind::AudioFile]
    );
    app.audio_file_notice = None;
    app.audio_file_error = false;
    app.help = Some(("Speech help", "Check speech setup"));
    assert_eq!(app.active_alerts(), vec![AlertKind::Setup]);
    app.setup.error = None;
    assert_eq!(app.active_alerts(), vec![AlertKind::Model]);
    app.set_status("Ready", false);
    chat.run();
    assert_eq!(chat.state().app.alerts.selection, None);
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Copy error")
            .is_none()
    );
}

#[test]
fn error_reporting_details_own_shortcuts_without_stopping_generation() {
    let mut chat = harness([1080.0, 780.0], 16.0);
    chat.state_mut().app.model = ModelLifecycle::Generating;
    chat.state_mut().app.speech_notice = true;
    chat.state_mut().app.speech_status = long_error();
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "View details")
        .click();
    chat.run();
    let count = chat.state().app.conversations.len();
    chat.key_press(egui::Key::Enter);
    chat.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::N);
    chat.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::F);
    chat.run();
    assert_eq!(chat.state().app.conversations.len(), count);
    assert!(!chat.state().app.transcript.find_open());
    assert_eq!(chat.state().app.input, "Keep this draft");
    assert!(chat.state().commands.try_recv().is_err());
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert_eq!(chat.state().app.alerts.details, None);
    assert_eq!(chat.state().app.model, ModelLifecycle::Generating);
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn intentional_stop_never_raises_model_alert_and_keeps_composer_ready_to_send() {
    let mut chat = harness([1080.0, 780.0], 16.0);
    chat.state_mut().app.context_tokens = Some(131_072);
    chat.state_mut().app.input = "Draft prompt to revise".into();
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Send  ↑")
        .click();
    chat.run();
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::Send { .. })
    ));

    // Intentional Stop before any prose restores the prompt for immediate editing and resending.
    chat.get_by_role_and_label(egui::Role::Button, "Stop")
        .click();
    chat.run();
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::Stop)
    ));
    assert!(chat.query_by_label("Model needs attention").is_none());
    assert!(chat.query_by_label("1 alert").is_none());

    chat.state_mut()
        .app
        .model_event(Event::Stopping("Draft prompt to revise".into()));
    chat.state_mut().app.model_event(Event::Stopped);
    chat.run();
    assert_eq!(chat.state().app.model, ModelLifecycle::Ready);
    assert_eq!(chat.state().app.context_tokens, Some(131_072));
    assert!(!chat.state().app.warning);
    assert!(chat.state().app.active_alerts().is_empty());
    assert!(chat.query_by_label("Model needs attention").is_none());
    assert!(chat.query_by_label("1 alert").is_none());
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Retry model")
            .is_none()
    );
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Reload model")
            .is_none()
    );
    assert_eq!(chat.state().app.input, "Draft prompt to revise");

    chat.state_mut().app.input = "Revised prompt after stop".into();
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Send  ↑")
        .click();
    chat.run();
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::Send { text, .. }) if text == "Revised prompt after stop"
    ));

    // Intentional Escape stop mid-stream adopts partial prose without alerting or parking.
    chat.state_mut()
        .app
        .model_event(Event::Text("Partial answer".into()));
    chat.run();
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::Stop)
    ));
    chat.state_mut()
        .app
        .model_event(Event::Stopping("Revised prompt after stop".into()));
    chat.state_mut().app.model_event(Event::Stopped);
    chat.run();
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(Command::RestoreChat(history)) if history.len() == 2
    ));
    assert_eq!(chat.state().app.model, ModelLifecycle::Ready);
    assert!(!chat.state().app.warning);
    assert!(chat.query_by_label("Model needs attention").is_none());
    assert!(chat.query_by_label("1 alert").is_none());
    assert!(
        chat.query_by_role_and_label(egui::Role::Button, "Send  ↑")
            .is_some()
    );
}

#[test]
fn error_reporting_capture_when_requested() -> io::Result<()> {
    let Some(directory) = std::env::var_os("BONSAI_ERROR_CAPTURE_DIRECTORY") else {
        return Ok(());
    };
    let directory = std::path::PathBuf::from(directory);
    bonsai_windows_credentials::validate_external_path(&directory)?;
    std::fs::create_dir_all(&directory)?;
    for (name, size, theme) in [
        ("error-minimum.png", [480.0, 480.0], appearance::Theme::Dark),
        ("error-wide.png", [1080.0, 780.0], appearance::Theme::Light),
    ] {
        let (commands, _incoming) = mpsc::channel();
        let (_events, incoming) = client::event_channel();
        let mut app = State::create(commands, incoming, None, None);
        app.preview = false;
        app.fail_startup(&io::Error::other(long_error()));
        app.setup.speech_error = Some(long_error());
        app.input = "My draft stays available while I resolve this error.".to_owned();
        app.appearance.theme = theme;
        let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
        setup.instance_descriptor.backends = wgpu::Backends::DX12;
        let mut chat = Harness::builder()
            .with_size(size)
            .with_context_setup(configure_test_context)
            .wgpu_setup(eframe::egui_wgpu::WgpuSetup::CreateNew(setup))
            .build_ui_state(|ui, state: &mut State| state.draw(ui), app);
        chat.state().appearance.apply(&chat.ctx);
        chat.run();
        chat.render()
            .map_err(io::Error::other)?
            .save(directory.join(name))
            .map_err(io::Error::other)?;
    }
    Ok(())
}

#[test]
fn every_regular_alert_can_be_dismissed_with_x_preserving_state_and_reappearing_on_new_failure()
-> io::Result<()> {
    use error_ui::AlertKind;

    let mut chat = harness([1080.0, 780.0], 16.0);

    // 1. Setup alert: dismissing preserves setup.error (update safety) and does not surface Model.
    chat.state_mut()
        .app
        .fail_startup(&io::Error::other("Download failed"));
    chat.run();
    assert_eq!(chat.state().app.active_alerts(), vec![AlertKind::Setup]);
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss alert")
        .click();
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    assert!(chat.state().app.setup.error.is_some());
    assert!(chat.query_by_label("1 alert").is_none());

    chat.state_mut()
        .app
        .fail_startup(&io::Error::other("Disk full"));
    chat.state_mut().app.alerts.details = Some(AlertKind::Setup);
    chat.state_mut().app.dismiss_alert(AlertKind::Setup);
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    assert_eq!(chat.state().app.alerts.details, None);
    chat.state_mut().app.setup.error = None;
    chat.state_mut().app.set_status("Ready", false);

    // 2. Model alert (ready-state warning) and ModelLifecycle::Retryable.
    chat.state_mut().app.model = ModelLifecycle::Ready;
    chat.state_mut().app.set_status("Transient warning", true);
    chat.run();
    assert_eq!(chat.state().app.active_alerts(), vec![AlertKind::Model]);
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss alert")
        .click();
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    assert!(!chat.state().app.warning);

    chat.state_mut()
        .app
        .model_event(Event::Retryable("Worker exited".into()));
    chat.run();
    assert_eq!(chat.state().app.model, ModelLifecycle::Retryable);
    assert_eq!(chat.state().app.active_alerts(), vec![AlertKind::Model]);
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss alert")
        .click();
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    assert_eq!(chat.state().app.model, ModelLifecycle::Retryable);
    // Unrelated non-warning status updates must not resurrect a dismissed Retryable alert.
    chat.state_mut()
        .app
        .set_status("Image save cancelled.", false);
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    chat.state_mut()
        .app
        .model_event(Event::Retryable("Worker exited again".into()));
    chat.run();
    assert_eq!(chat.state().app.active_alerts(), vec![AlertKind::Model]);
    chat.state_mut().app.model = ModelLifecycle::Ready;
    chat.state_mut().app.set_status("Ready", false);

    // 3. Recovery alert: identical repeated failure stays dismissed; distinct failure reappears.
    chat.state_mut().app.recovery_notice = "Could not save conversations: disk full".into();
    chat.run();
    assert_eq!(chat.state().app.active_alerts(), vec![AlertKind::Recovery]);
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss alert")
        .click();
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    assert_eq!(
        chat.state().app.recovery_notice,
        "Could not save conversations: disk full"
    );
    chat.state_mut().app.recovery_notice = "Could not save conversations: disk full".into();
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    chat.state_mut().app.recovery_notice = "Could not save conversations: permission denied".into();
    chat.run();
    assert_eq!(chat.state().app.active_alerts(), vec![AlertKind::Recovery]);
    chat.state_mut().app.recovery_notice.clear();

    // 4. SpeechSetup alert: dismissing preserves speech_error and keeps Speech suppressed.
    chat.state_mut().app.setup.speech_error = Some("Speech setup failed".into());
    chat.state_mut().app.speech_notice = true;
    chat.state_mut().app.speech_status = "Speech failed".into();
    chat.run();
    assert_eq!(
        chat.state().app.active_alerts(),
        vec![AlertKind::SpeechSetup]
    );
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss alert")
        .click();
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    assert!(chat.state().app.setup.speech_error.is_some());
    chat.state_mut().app.setup.speech_error = None;
    chat.state_mut().app.speech_notice = false;

    // 5. AudioFile alert.
    chat.state_mut().app.audio_file_error = true;
    chat.state_mut().app.audio_file_notice = Some("Audio failed".into());
    chat.state_mut().app.audio_file_transcript = "Partial".into();
    chat.run();
    assert_eq!(chat.state().app.active_alerts(), vec![AlertKind::AudioFile]);
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss alert")
        .click();
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    assert!(chat.state().app.audio_file_notice.is_none());
    assert!(!chat.state().app.audio_file_error);
    assert!(chat.state().app.audio_file_transcript.is_empty());

    // 6. Speech alert.
    chat.state_mut().app.speech_notice = true;
    chat.state_mut().app.speech_status = "Microphone unavailable".into();
    chat.run();
    assert_eq!(chat.state().app.active_alerts(), vec![AlertKind::Speech]);
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss alert")
        .click();
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    assert!(!chat.state().app.speech_notice);

    // 7. Update alert.
    chat.state_mut().app.updates = Some(updates::Updates::test_fixture_with_error(
        "Update signature invalid",
    )?);
    chat.run();
    assert_eq!(chat.state().app.active_alerts(), vec![AlertKind::Update]);
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss alert")
        .click();
    chat.run();
    assert!(chat.state().app.active_alerts().is_empty());
    assert!(
        chat.state()
            .app
            .updates
            .as_ref()
            .and_then(updates::Updates::error_notice)
            .is_none()
    );
    Ok(())
}

#[test]
fn deleted_conversation_notice_auto_dismisses_and_supports_manual_dismiss() {
    let mut chat = harness([1080.0, 780.0], 16.0);

    // Manual dismissal still works immediately.
    chat.state_mut().app.new_chat();
    let second = chat.state().app.active_conversation;
    chat.state_mut().app.remove_conversation(second);
    chat.run();
    assert!(chat.query_by_label("Conversation deleted.").is_some());
    chat.get_by_role_and_label(egui::Role::Button, "Dismiss")
        .click();
    chat.run();
    assert!(chat.query_by_label("Conversation deleted.").is_none());
    assert!(chat.state().app.conversation_notice.is_none());

    // Auto-dismissal waits for the configured duration and resets when another deletion occurs.
    let duration = CONVERSATION_DELETED_NOTICE_DURATION.as_secs_f64();
    chat.state_mut().app.new_chat();
    let third = chat.state().app.active_conversation;
    chat.state_mut().app.remove_conversation(third);
    let start = 100.0;
    chat.input_mut().time = Some(start);
    chat.step();
    assert!(chat.query_by_label("Conversation deleted.").is_some());

    chat.input_mut().time = Some(start + duration - 0.5);
    chat.step();
    assert!(chat.query_by_label("Conversation deleted.").is_some());

    chat.state_mut().app.new_chat();
    let fourth = chat.state().app.active_conversation;
    chat.state_mut().app.remove_conversation(fourth);
    let reset_start = start + duration - 0.5;
    chat.input_mut().time = Some(reset_start);
    chat.step();
    assert!(chat.query_by_label("Conversation deleted.").is_some());

    // Past the first deadline, the reset timer keeps the notice visible.
    chat.input_mut().time = Some(start + duration + 0.5);
    chat.step();
    assert!(chat.query_by_label("Conversation deleted.").is_some());

    // Past the second deadline, the notice auto-dismisses.
    chat.input_mut().time = Some(reset_start + duration + 0.1);
    chat.step();
    assert!(chat.query_by_label("Conversation deleted.").is_none());
    assert!(chat.state().app.conversation_notice.is_none());

    // Persistent conversation notices do not auto-dismiss over time.
    chat.state_mut().app.conversation_notice = Some(ConversationNotice::persistent(
        "Created a branch. The original conversation is unchanged.",
    ));
    chat.input_mut().time = Some(200.0);
    chat.step();
    chat.input_mut().time = Some(260.0);
    chat.step();
    assert!(
        chat.query_by_label("Created a branch. The original conversation is unchanged.")
            .is_some()
    );
}

#[test]
fn deleted_conversation_notice_undo_restores_conversation_and_clears_notice() {
    let mut chat = harness([1080.0, 780.0], 16.0);
    let original_id = chat.state().app.active_conversation;
    chat.state_mut().app.input = "Recoverable draft".to_owned();
    chat.state_mut().app.new_chat();
    let second_id = chat.state().app.active_conversation;
    chat.state_mut().app.remove_conversation(original_id);
    chat.run();

    assert_eq!(chat.state().app.conversations.len(), 1);
    assert_eq!(chat.state().app.active_conversation, second_id);
    assert!(chat.query_by_label("Conversation deleted.").is_some());
    chat.get_by_role_and_label(egui::Role::Button, "Undo")
        .click();
    chat.run();

    assert!(chat.query_by_label("Conversation deleted.").is_none());
    assert_eq!(chat.state().app.conversations.len(), 2);
    assert_eq!(chat.state().app.active_conversation, original_id);
    assert_eq!(chat.state().app.input, "Recoverable draft");
}
