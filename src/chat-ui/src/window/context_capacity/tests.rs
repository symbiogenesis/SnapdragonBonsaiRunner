//! Capacity guidance follows the same histories and controls as actual requests.
use super::*;
use crate::{client, window::transcript::Role};
use egui_kittest::{
    Harness,
    kittest::{NodeT as _, Queryable as _},
};
use std::sync::mpsc;

fn state() -> (State, mpsc::Receiver<client::Command>) {
    let (commands, incoming) = mpsc::channel();
    let (_events, events) = client::event_channel();
    let mut state = State::create(commands, events, None, None);
    state.preview = false;
    state.model = ModelLifecycle::Ready;
    state.context_tokens = Some(8_192);
    (state, incoming)
}

#[test]
fn context_capacity_tracks_model_attempt_and_large_windows() {
    let (mut app, _commands) = state();
    app.input = "Retained draft".into();
    app.commit_history(history(&"a".repeat(40_000)));
    let old = app.context_estimate();
    assert_eq!(old.pressure(), Pressure::Over);
    app.model = ModelLifecycle::Preparing;
    app.model_event(client::Event::ContextCapacity(131_072));
    let large = app.context_estimate();
    assert_eq!(old.total, large.total);
    assert_eq!(large.capacity, Some(131_072));
    assert_eq!(large.pressure(), Pressure::Comfortable);
    assert!((fraction(65_536, 131_072) - 0.5).abs() < 0.001);
    assert!((fraction(262_144, 262_144) - 1.0).abs() < f32::EPSILON);
    app.model_event(client::Event::Starting);
    assert_eq!(app.context_estimate().capacity, None);
    assert!(app.context_pending());
    assert_eq!(app.input, "Retained draft");
}

#[test]
fn context_capacity_is_unknown_before_model_startup() {
    let mut chat = harness([620.0, 640.0], "Question");
    chat.state_mut().app.context_tokens = None;
    chat.run();
    let description = chat
        .get_by_role_and_label(egui::Role::Button, "Context")
        .accesskit_node()
        .description()
        .unwrap_or_default();
    assert!(description.contains("after the model starts"));
    assert!(!description.contains('%'));
}

fn history(text: &str) -> Vec<client::Message> {
    vec![
        client::Message {
            role: "user",
            content: text.into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        },
        client::Message {
            role: "assistant",
            content: "Answer".into(),
            images: std::sync::Arc::default(),
            reasoning_content: Some("Retained reasoning".into()),
        },
    ]
}

struct Fixture {
    app: State,
    commands: mpsc::Receiver<client::Command>,
}

fn harness(size: [f32; 2], text: &str) -> Harness<'static, Fixture> {
    harness_with_render(size, text, false)
}

fn harness_with_render(size: [f32; 2], text: &str, render: bool) -> Harness<'static, Fixture> {
    let (mut app, commands) = state();
    app.input = "Keep this draft".into();
    app.commit_history(history(text));
    let builder = Harness::builder()
        .with_size(size)
        .with_context_setup(crate::window::configure_test_context);
    let builder = if render {
        let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
        setup.instance_descriptor.backends = wgpu::Backends::DX12;
        builder.wgpu_setup(eframe::egui_wgpu::WgpuSetup::CreateNew(setup))
    } else {
        builder
    };
    builder.build_ui_state(
        |ui, fixture: &mut Fixture| {
            fixture.app.handle_input_shortcuts(ui.ctx());
            fixture.app.draw(ui);
        },
        Fixture { app, commands },
    )
}

fn context_details_rect(chat: &Harness<'_, Fixture>) -> Result<egui::Rect, &'static str> {
    assert!(chat.query_by_label("Context details").is_some());
    // Named accessibility containers need not carry a bounding box. The popup
    // area's actual rectangle is the geometry used for layout and hit testing.
    let areas = chat
        .ctx
        .memory(|memory| memory.layer_ids().map(|layer| layer.id).collect::<Vec<_>>());
    let popup = areas
        .into_iter()
        .find(|id| egui::Popup::is_id_open(&chat.ctx, *id))
        .ok_or("Context details must have an open popup area")?;
    chat.ctx
        .memory(|memory| memory.area_rect(popup))
        .ok_or("The open Context details area must retain its rectangle")
}

#[test]
fn context_capacity_capture_when_requested() -> std::io::Result<()> {
    let Some(value) = std::env::var_os("BONSAI_UI_CAPTURE_DIRECTORY") else {
        return Ok(());
    };
    let directory = std::path::PathBuf::from(value);
    bonsai_windows_credentials::validate_external_path(&directory)?;
    std::fs::create_dir_all(&directory)?;
    for (name, size, details, crowded) in [
        ("context-details-wide.png", [1180.0, 860.0], true, true),
        ("context-warning-narrow.png", [620.0, 640.0], false, true),
        ("composer-toolbar-minimum.png", [480.0, 480.0], false, false),
    ] {
        let mut chat = harness_with_render(size, "Help me refine this idea.", true);
        let mut messages = history("Help me refine this idea.");
        messages[1].content =
            "A clear question is a good place to begin. What would you like to explore?".into();
        if crowded {
            messages[1].reasoning_content = Some("a".repeat(20_000).into());
        }
        chat.state_mut().app.commit_history(messages);
        let transcript = crate::window::Transcript::from_committed(&chat.state().app.committed);
        chat.state_mut().app.transcript = transcript;
        chat.run();
        if details {
            chat.get_by_role_and_label(egui::Role::Button, "Context")
                .click();
            chat.run();
        }
        chat.render()
            .map_err(std::io::Error::other)?
            .save(directory.join(name))
            .map_err(std::io::Error::other)?;
        assert!(chat.state().app.worker.is_none());
        assert!(chat.state().app.speech.is_none());
    }
    Ok(())
}

#[test]
fn context_capacity_cache_tracks_equal_length_edits_and_committed_replacement() {
    let (mut app, _commands) = state();
    app.commit_history(history("aaaaaaaa"));
    app.input = "aaaaaaaa".into();
    let before = app.context_estimate();
    app.input = "{}[]();=".into();
    assert!(app.context_estimate().draft > before.draft);
    app.commit_history(history("{}[]();="));
    assert!(app.context_estimate().history > before.history);
    let committed = app.context_estimate();
    app.transcript
        .push(Role::Assistant, "uncommitted".repeat(1_000));
    assert_eq!(app.context_estimate(), committed);
    // No copy or fresh estimate is needed for an unchanged draft/history.
    let allocation = app.context_draft.units.cached_ptr();
    assert_eq!(app.context_estimate(), committed);
    assert_eq!(app.context_draft.units.cached_ptr(), allocation);
}

#[test]
fn context_capacity_background_commit_and_switch_preserve_selected_estimates() {
    let (mut app, _commands) = state();
    app.commit_history(history("short"));
    app.input = "Original draft".into();
    let original = app.active_conversation;
    app.context_estimate();
    app.new_chat();
    app.input = "Other draft".into();
    let selected = app.active_conversation;
    let visible = app.context_estimate();
    app.request_conversation = Some(original);
    app.model = ModelLifecycle::Generating;
    app.with_request_conversation(|owner| owner.commit_history(history(&"long".repeat(1_000))));
    assert_eq!(app.active_conversation, selected);
    assert_eq!(app.context_estimate(), visible);
    assert!(!app.context_pending());
    app.switch_conversation(original);
    assert!(app.context_pending());
    assert!(app.context_estimate().history > visible.history);
    assert_eq!(app.input, "Original draft");
}

#[test]
fn context_capacity_recovery_and_branch_rebuild_without_saving_estimates() -> std::io::Result<()> {
    let (mut app, _commands) = state();
    app.commit_history(history("Original question"));
    app.transcript = crate::window::Transcript::from_committed(&app.committed);
    app.input = "A saved draft".into();
    let expected = app.context_estimate();
    let bytes = app.workspace_bytes()?;
    assert!(!String::from_utf8_lossy(&bytes).contains("context_history"));
    // Exercise the production deserialization path without a DPAPI/disk workload.
    let mut snapshot: crate::window::workspace::ConversationSnapshot = serde_json::from_value(
        serde_json::from_slice::<serde_json::Value>(&bytes)?["active"].clone(),
    )?;
    assert!(snapshot.context_history.is_none());
    snapshot.warning = false;
    let (mut restored, _commands) = state();
    restored.load_conversation(snapshot);
    assert_eq!(restored.context_estimate(), expected);
    let message = app.transcript.messages[0].id;
    app.message_action(crate::window::iteration::MessageAction::Branch(message));
    assert_eq!(app.context_estimate().history, 0);
    assert_eq!(app.input, "Original question");
    Ok(())
}

#[test]
fn context_capacity_warning_keeps_send_available_without_changing_reply_limits() {
    for size in [[1180.0, 860.0], [620.0, 640.0]] {
        let mut chat = harness(size, &"a".repeat(20_000));
        assert_eq!(
            chat.state_mut().app.context_estimate().pressure(),
            Pressure::Over
        );
        assert!(
            !chat
                .get_by_role_and_label(egui::Role::Button, "Send  ↑")
                .accesskit_node()
                .is_disabled()
        );
        let send = chat
            .get_by_role_and_label(egui::Role::Button, "Send  ↑")
            .rect();
        assert!(egui::Rect::from_min_size(egui::Pos2::ZERO, size.into()).contains_rect(send));
        assert_eq!(chat.state().app.input, "Keep this draft");
        assert_eq!(chat.state().app.output_budget, 3);
        assert!(chat.query_by_label("Use 2,048-token reply").is_none());
        chat.get_by_role_and_label(egui::Role::Button, "Send  ↑")
            .click();
        chat.run();
        assert!(chat.state().commands.try_iter().any(|command| matches!(
            command,
            client::Command::Send { max_completion_tokens: 4096, text, .. } if text == "Keep this draft"
        )));
    }
}

#[test]
fn context_capacity_toolbar_keeps_plain_actions_visible_at_minimum_and_wide_sizes() {
    for size in [[480.0, 480.0], [620.0, 640.0], [1180.0, 860.0]] {
        for theme in [
            crate::window::appearance::Theme::Light,
            crate::window::appearance::Theme::Dark,
        ] {
            let mut chat = harness(size, "Question");
            chat.state_mut().app.appearance.theme = theme;
            chat.state_mut().app.appearance.text_size = 24;
            chat.state_mut().app.enable_thinking = true;
            chat.state_mut().app.extra_effort = true;
            chat.state().app.appearance.apply(&chat.ctx);
            chat.run();
            let bounds = egui::Rect::from_min_size(egui::Pos2::ZERO, size.into());
            let space = chat
                .get_by_role_and_label(egui::Role::Button, "Context")
                .rect();
            let options = chat
                .get_by_role_and_label(egui::Role::Button, "Thinking")
                .rect();
            assert!(bounds.contains_rect(space) && bounds.contains_rect(options));
            assert!(!space.intersects(options));
            assert!(space.height() >= 32.0 && options.height() >= 32.0);
            assert!(space.width() < 112.0 && options.width() < 104.0);
            for label in ["Context", "Thinking"] {
                assert!(
                    chat.get_by_role_and_label(egui::Role::Button, label)
                        .accesskit_node()
                        .toggled()
                        .is_none()
                );
            }
            assert!(chat.query_by_label("Reply token limit").is_none());
            assert!(
                chat.query_by_label("Thinking on · Extra effort · Adapter 2.0")
                    .is_none()
            );
            assert_eq!(chat.state().app.input, "Keep this draft");
            assert!(chat.state().commands.try_recv().is_err());
        }
    }
}

#[test]
fn context_capacity_ring_description_tracks_drafts_and_hides_unfinished_occupancy() {
    let mut chat = harness([480.0, 480.0], "Question");
    let description = chat
        .get_by_role_and_label(egui::Role::Button, "Context")
        .accesskit_node()
        .description()
        .unwrap_or_default();
    assert!(description.contains('%'));
    chat.state_mut().app.input = "Longer next message. ".repeat(100);
    chat.run();
    let longer = chat
        .get_by_role_and_label(egui::Role::Button, "Context")
        .accesskit_node()
        .description()
        .unwrap_or_default();
    assert!(longer.contains('%'));
    assert_ne!(description, longer);
    let owner = chat.state().app.active_conversation;
    chat.state_mut().app.model = ModelLifecycle::Generating;
    chat.state_mut().app.request_conversation = Some(owner);
    chat.run();
    let pending = chat
        .get_by_role_and_label(egui::Role::Button, "Context")
        .accesskit_node()
        .description()
        .unwrap_or_default();
    assert!(pending.contains("updates when this response finishes"));
    assert!(!pending.contains('%'));
    assert_eq!(chat.state().app.output_budget, 3);
    assert!(chat.state().commands.try_recv().is_err());
}

#[test]
fn context_capacity_popover_is_keyboard_accessible_and_scopes_extra_effort()
-> Result<(), &'static str> {
    let mut chat = harness([620.0, 640.0], "Question");
    chat.state_mut().app.extra_effort = true;
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Context")
        .focus();
    chat.key_press(egui::Key::Space);
    chat.run();
    assert!(chat.query_by_label("Space for this draft").is_some());
    let extra_effort = "First pass only. Extra effort's later passes add generated notes and drafts and need more room.";
    chat.get_by_label(extra_effort).scroll_to_me();
    chat.run();
    let details = context_details_rect(&chat)?;
    assert!(details.contains_rect(chat.get_by_label(extra_effort).rect()));
    chat.key_press(egui::Key::Enter);
    chat.run();
    assert!(chat.state().commands.try_recv().is_err());
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert!(chat.query_by_label("Space for this draft").is_none());
    assert_eq!(chat.state().app.input, "Keep this draft");
    assert!(chat.state().commands.try_recv().is_err());
    Ok(())
}

#[test]
fn context_capacity_details_leave_both_toolbar_actions_uncovered() -> Result<(), &'static str> {
    for size in [[480.0, 480.0], [620.0, 640.0], [1180.0, 860.0]] {
        let mut chat = harness(size, "Question");
        chat.get_by_role_and_label(egui::Role::Button, "Context")
            .click();
        chat.run();
        let details = context_details_rect(&chat)?;
        let bounds = egui::Rect::from_min_size(egui::Pos2::ZERO, size.into());
        assert!(bounds.contains_rect(details));
        for label in ["Context", "Thinking"] {
            let action = chat.get_by_role_and_label(egui::Role::Button, label).rect();
            assert!(!details.intersects(action));
        }
        chat.get_by_role_and_label(egui::Role::Button, "Thinking")
            .click();
        chat.run();
        assert!(chat.query_by_label("Context details").is_none());
        assert!(chat.query_by_label("Thinking options").is_some());
        assert_eq!(chat.state().app.input, "Keep this draft");
        assert!(chat.state().commands.try_recv().is_err());
    }
    Ok(())
}

#[test]
fn context_capacity_start_fresh_preserves_original_draft_and_full_history() {
    let mut chat = harness([620.0, 640.0], &"a".repeat(40_000));
    let original = chat.state().app.active_conversation;
    chat.get_by_role_and_label(egui::Role::Button, "Start fresh")
        .click();
    chat.run();
    assert_ne!(chat.state().app.active_conversation, original);
    assert_eq!(chat.state_mut().app.context_estimate().history, 0);
    chat.state_mut().app.switch_conversation(original);
    chat.run();
    assert_eq!(chat.state().app.input, "Keep this draft");
    assert_eq!(chat.state().app.committed.len(), 2);
    assert!(
        chat.state()
            .commands
            .try_iter()
            .all(|command| !matches!(command, client::Command::Send { .. }))
    );
}

#[test]
fn context_capacity_escape_closes_details_before_stopping_generation() {
    let mut chat = harness([620.0, 640.0], "Question");
    let owner = chat.state().app.active_conversation;
    chat.state_mut().app.model = ModelLifecycle::Generating;
    chat.state_mut().app.request_conversation = Some(owner);
    chat.run();
    chat.get_by_role_and_label(egui::Role::Button, "Context")
        .click();
    chat.run();
    assert!(chat.query_by_label("Space for this draft").is_some());
    assert!(
        chat.query_by_label(
            "Updates when this response finishes. The unfinished turn is not included below."
        )
        .is_some()
    );
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert!(chat.query_by_label("Space for this draft").is_none());
    assert!(chat.state().app.model.busy());
    assert!(chat.state().commands.try_recv().is_err());
    chat.key_press(egui::Key::Escape);
    chat.run();
    assert!(matches!(
        chat.state().commands.try_recv(),
        Ok(client::Command::Stop)
    ));
}

#[test]
fn footprint_diagnostics_surface_ptq1_cow_npu_dedup_and_mapped_commit_split()
-> Result<(), &'static str> {
    let pq2_divergent = estimate_runtime_footprint(PlannerRuntimeShape {
        context_tokens: 262_144,
        weight_packing: WeightPacking::Pq2_0,
        kv_layout: KvCacheLayout::Rsq4,
        kv_sharing: KvSharingMode::DivergentRollback,
        evictable_mapped_weight_bytes: 0,
        npu_resident_arena_bytes: 0,
        npu_released_host_pq2_bytes: 0,
        daisyrose_adapter_bytes: 0,
        branch_checkpoints: 1,
    })
    .ok_or("valid PQ2 divergent baseline")?;

    let ptq1_cow = estimate_runtime_footprint(PlannerRuntimeShape {
        context_tokens: 262_144,
        weight_packing: WeightPacking::Ptq1_0,
        kv_layout: KvCacheLayout::Rsq4,
        kv_sharing: KvSharingMode::CowRollback,
        evictable_mapped_weight_bytes: WeightPacking::Ptq1_0.weight_bytes(),
        npu_resident_arena_bytes: 0,
        npu_released_host_pq2_bytes: 0,
        daisyrose_adapter_bytes: 0,
        branch_checkpoints: 1,
    })
    .ok_or("valid PTQ1 CoW estimate")?;

    assert_eq!(WeightPacking::Pq2_0.weight_bytes(), 7_206_168_928);
    assert_eq!(WeightPacking::Ptq1_0.weight_bytes(), 5_946_648_928);
    assert_eq!(
        pq2_divergent.kv_bytes - ptq1_cow.kv_bytes,
        262_080 * KvCacheLayout::Rsq4.bytes_per_token()
    );
    assert_eq!(
        ptq1_cow.physical_savings_over(&pq2_divergent),
        1_259_520_000 + 262_080 * 16_896
    );
    assert_eq!(
        ptq1_cow.steady_state_physical_bytes - ptq1_cow.private_job_commit_bytes,
        WeightPacking::Ptq1_0.weight_bytes()
    );

    let ptq1_append_only = estimate_runtime_footprint(PlannerRuntimeShape {
        context_tokens: 262_144,
        weight_packing: WeightPacking::Ptq1_0,
        kv_layout: KvCacheLayout::Rsq4,
        kv_sharing: KvSharingMode::AppendOnlyCow,
        evictable_mapped_weight_bytes: 0,
        npu_resident_arena_bytes: 0,
        npu_released_host_pq2_bytes: 0,
        daisyrose_adapter_bytes: 0,
        branch_checkpoints: 1,
    })
    .ok_or("valid PTQ1 append-only CoW estimate")?;
    assert_eq!(
        ptq1_cow.kv_bytes - ptq1_append_only.kv_bytes,
        64 * KvCacheLayout::Rsq4.bytes_per_token()
    );

    let npu_dedup = estimate_runtime_footprint(PlannerRuntimeShape {
        context_tokens: 131_072,
        weight_packing: WeightPacking::Pq2_0,
        kv_layout: KvCacheLayout::Rsq4K3v,
        kv_sharing: KvSharingMode::CowRollback,
        evictable_mapped_weight_bytes: 0,
        npu_resident_arena_bytes: HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES + 32 * 1024 * 1024,
        npu_released_host_pq2_bytes: HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
        daisyrose_adapter_bytes: 0,
        branch_checkpoints: 1,
    })
    .ok_or("valid NPU dedup estimate")?;
    let cpu_same = estimate_runtime_footprint(PlannerRuntimeShape {
        context_tokens: 131_072,
        weight_packing: WeightPacking::Pq2_0,
        kv_layout: KvCacheLayout::Rsq4K3v,
        kv_sharing: KvSharingMode::CowRollback,
        evictable_mapped_weight_bytes: 0,
        npu_resident_arena_bytes: 0,
        npu_released_host_pq2_bytes: 0,
        daisyrose_adapter_bytes: 0,
        branch_checkpoints: 1,
    })
    .ok_or("valid CPU comparison estimate")?;
    assert_eq!(
        npu_dedup.job_commit_savings_over(&cpu_same),
        HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES
    );
    assert_eq!(
        npu_dedup.steady_state_physical_bytes - cpu_same.steady_state_physical_bytes,
        32 * 1024 * 1024
    );

    let ptq1_diag = format_footprint_diagnostic(&ptq1_cow);
    assert!(ptq1_diag.contains("PTQ1_0"));
    assert!(ptq1_diag.contains("K4/V4 CoW KV"));
    assert!(ptq1_diag.contains("Est. physical"));
    assert!(ptq1_diag.contains("Job commit"));
    assert!(ptq1_diag.contains("mapped"));
    assert!(ptq1_diag.contains("1 branch checkpoint"));

    let npu_diag = format_footprint_diagnostic(&npu_dedup);
    assert!(npu_diag.contains("PQ2_0"));
    assert!(npu_diag.contains("K4/V3 CoW KV"));
    assert!(npu_diag.contains("NPU dedup -2.69 GiB host / +2.72 GiB arena"));
    assert_eq!(format_gib_bytes(1024 * 1024 * 1024), "1.00 GiB");
    Ok(())
}

#[test]
fn footprint_diagnostics_enforce_branch_checkpoint_dedup_and_mapped_bounds()
-> Result<(), &'static str> {
    let mut prev_physical = 0_u64;
    for branches in 1..=MAX_REUSABLE_BRANCH_CHECKPOINTS {
        let estimate = estimate_runtime_footprint(PlannerRuntimeShape {
            context_tokens: 65_536,
            weight_packing: WeightPacking::Ptq1_0,
            kv_layout: KvCacheLayout::F16,
            kv_sharing: KvSharingMode::CowRollback,
            evictable_mapped_weight_bytes: 0,
            npu_resident_arena_bytes: 0,
            npu_released_host_pq2_bytes: 0,
            daisyrose_adapter_bytes: 0,
            branch_checkpoints: branches,
        })
        .ok_or("valid branch checkpoint count")?;
        if branches > 1 {
            assert_eq!(
                estimate.steady_state_physical_bytes - prev_physical,
                RECURRENT_BYTES
            );
            assert!(
                format_footprint_diagnostic(&estimate)
                    .contains(&format!("{branches} branch checkpoints"))
            );
        }
        prev_physical = estimate.steady_state_physical_bytes;
    }

    assert!(
        estimate_runtime_footprint(PlannerRuntimeShape {
            context_tokens: 65_536,
            weight_packing: WeightPacking::Ptq1_0,
            kv_layout: KvCacheLayout::Rsq4,
            kv_sharing: KvSharingMode::CowRollback,
            evictable_mapped_weight_bytes: 0,
            npu_resident_arena_bytes: 0,
            npu_released_host_pq2_bytes: 0,
            daisyrose_adapter_bytes: 0,
            branch_checkpoints: 9,
        })
        .is_none()
    );
    assert!(
        estimate_runtime_footprint(PlannerRuntimeShape {
            context_tokens: 65_536,
            weight_packing: WeightPacking::Ptq1_0,
            kv_layout: KvCacheLayout::Rsq4,
            kv_sharing: KvSharingMode::CowRollback,
            evictable_mapped_weight_bytes: 0,
            npu_resident_arena_bytes: HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
            npu_released_host_pq2_bytes: HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
            daisyrose_adapter_bytes: 0,
            branch_checkpoints: 1,
        })
        .is_none()
    );
    assert!(
        estimate_runtime_footprint(PlannerRuntimeShape {
            context_tokens: 65_536,
            weight_packing: WeightPacking::Pq2_0,
            kv_layout: KvCacheLayout::Rsq4,
            kv_sharing: KvSharingMode::CowRollback,
            evictable_mapped_weight_bytes: 0,
            npu_resident_arena_bytes: HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES + 1,
            npu_released_host_pq2_bytes: HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES + 1,
            daisyrose_adapter_bytes: 0,
            branch_checkpoints: 1,
        })
        .is_none()
    );
    assert!(
        estimate_runtime_footprint(PlannerRuntimeShape {
            context_tokens: 65_536,
            weight_packing: WeightPacking::Ptq1_0,
            kv_layout: KvCacheLayout::Rsq4,
            kv_sharing: KvSharingMode::CowRollback,
            evictable_mapped_weight_bytes: WeightPacking::Ptq1_0.fixed_runtime_bytes() + 1,
            npu_resident_arena_bytes: 0,
            npu_released_host_pq2_bytes: 0,
            daisyrose_adapter_bytes: 0,
            branch_checkpoints: 1,
        })
        .is_none()
    );
    Ok(())
}

#[test]
fn context_details_popover_renders_live_footprint_diagnostic() -> Result<(), &'static str> {
    let mut chat = harness([620.0, 640.0], "Question");
    chat.state_mut().app.context_tokens = Some(262_144);
    chat.get_by_role_and_label(egui::Role::Button, "Context")
        .click();
    chat.run();
    let expected = format_footprint_diagnostic(
        &estimate_runtime_footprint(PlannerRuntimeShape {
            context_tokens: 262_144,
            weight_packing: WeightPacking::Pq2_0,
            kv_layout: KvCacheLayout::Rsq4,
            kv_sharing: KvSharingMode::CowRollback,
            evictable_mapped_weight_bytes: WeightPacking::Pq2_0.weight_bytes(),
            npu_resident_arena_bytes: 0,
            npu_released_host_pq2_bytes: 0,
            daisyrose_adapter_bytes: 0,
            branch_checkpoints: 1,
        })
        .ok_or("valid default footprint")?,
    );
    assert!(chat.query_by_label(expected.as_str()).is_some());
    Ok(())
}

#[test]
fn daisyrose_adapter_footprint_scales_physical_commit_and_formats_diagnostic_cleanly()
-> Result<(), &'static str> {
    assert_eq!(npu_friendly_daisyrose_adapter_bytes(0), None);
    assert_eq!(npu_friendly_daisyrose_adapter_bytes(65), None);
    let rank16_bytes = npu_friendly_daisyrose_adapter_bytes(16).ok_or("rank 16 bytes")?;
    assert_eq!(rank16_bytes, 16 * 4_194_304 + 16 * 1024 * 1024);

    let baseline = estimate_runtime_footprint(PlannerRuntimeShape {
        context_tokens: 262_144,
        weight_packing: WeightPacking::Pq2_0,
        kv_layout: KvCacheLayout::Rsq4,
        kv_sharing: KvSharingMode::CowRollback,
        evictable_mapped_weight_bytes: WeightPacking::Pq2_0.weight_bytes(),
        npu_resident_arena_bytes: 0,
        npu_released_host_pq2_bytes: 0,
        daisyrose_adapter_bytes: 0,
        branch_checkpoints: 1,
    })
    .ok_or("baseline footprint")?;
    let stacked = estimate_runtime_footprint(PlannerRuntimeShape {
        context_tokens: 262_144,
        weight_packing: WeightPacking::Pq2_0,
        kv_layout: KvCacheLayout::Rsq4,
        kv_sharing: KvSharingMode::CowRollback,
        evictable_mapped_weight_bytes: WeightPacking::Pq2_0.weight_bytes(),
        npu_resident_arena_bytes: 0,
        npu_released_host_pq2_bytes: 0,
        daisyrose_adapter_bytes: rank16_bytes,
        branch_checkpoints: 1,
    })
    .ok_or("stacked DaisyRose footprint")?;

    assert_eq!(
        stacked.steady_state_physical_bytes - baseline.steady_state_physical_bytes,
        rank16_bytes
    );
    assert_eq!(
        stacked.private_job_commit_bytes - baseline.private_job_commit_bytes,
        rank16_bytes
    );
    assert!(!format_footprint_diagnostic(&baseline).contains("DaisyRose"));
    assert!(format_footprint_diagnostic(&stacked).ends_with(" + 80 MiB DaisyRose"));

    let mut chat = harness([680.0, 640.0], "Question");
    chat.state_mut().app.context_tokens = Some(262_144);
    chat.state_mut().app.daisyrose_available = true;
    chat.state_mut().app.daisyrose_max_rank = Some(16);
    chat.get_by_role_and_label(egui::Role::Button, "Context")
        .click();
    chat.run();
    let expected_live = format_footprint_diagnostic(&stacked);
    assert!(chat.query_by_label(expected_live.as_str()).is_some());
    Ok(())
}

#[test]
fn footprint_diagnostics_cover_bonsai25_pq2_and_ptq1_variants() -> Result<(), &'static str> {
    let pq2 = estimate_runtime_footprint(PlannerRuntimeShape {
        context_tokens: 262_144,
        weight_packing: WeightPacking::Bonsai25Pq2,
        kv_layout: KvCacheLayout::Rsq4,
        kv_sharing: KvSharingMode::CowRollback,
        evictable_mapped_weight_bytes: WeightPacking::Bonsai25Pq2.weight_bytes(),
        npu_resident_arena_bytes: 3_000_000_000,
        npu_released_host_pq2_bytes: HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
        daisyrose_adapter_bytes: 0,
        branch_checkpoints: 1,
    })
    .ok_or("Expected Bonsai 2.5 PQ2 footprint estimate")?;
    let ptq1 = estimate_runtime_footprint(PlannerRuntimeShape {
        context_tokens: 262_144,
        weight_packing: WeightPacking::Bonsai25Ptq1,
        kv_layout: KvCacheLayout::Rsq4,
        kv_sharing: KvSharingMode::CowRollback,
        evictable_mapped_weight_bytes: WeightPacking::Bonsai25Ptq1.weight_bytes(),
        npu_resident_arena_bytes: 0,
        npu_released_host_pq2_bytes: 0,
        daisyrose_adapter_bytes: 0,
        branch_checkpoints: 1,
    })
    .ok_or("Expected Bonsai 2.5 PTQ1 footprint estimate")?;
    assert_eq!(pq2.weight_packing, WeightPacking::Bonsai25Pq2);
    assert_eq!(ptq1.weight_packing, WeightPacking::Bonsai25Ptq1);
    assert_eq!(
        WeightPacking::Bonsai25Pq2.weight_bytes() - WeightPacking::Bonsai25Ptq1.weight_bytes(),
        1_259_520_000
    );
    assert!(
        estimate_runtime_footprint(PlannerRuntimeShape {
            context_tokens: 262_144,
            weight_packing: WeightPacking::Bonsai25Ptq1,
            kv_layout: KvCacheLayout::Rsq4,
            kv_sharing: KvSharingMode::CowRollback,
            evictable_mapped_weight_bytes: WeightPacking::Bonsai25Ptq1.weight_bytes(),
            npu_resident_arena_bytes: 3_000_000_000,
            npu_released_host_pq2_bytes: HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES,
            daisyrose_adapter_bytes: 0,
            branch_checkpoints: 1,
        })
        .is_none()
    );
    assert!(format_footprint_diagnostic(&pq2).contains("Bonsai 2.5 PQ2_0"));
    assert!(format_footprint_diagnostic(&ptq1).contains("Bonsai 2.5 PTQ1_0"));

    let mut chat = harness([680.0, 640.0], "Question");
    chat.state_mut().app.context_tokens = Some(262_144);
    chat.state_mut().app.weight_packing = WeightPacking::Bonsai25Ptq1;
    chat.get_by_role_and_label(egui::Role::Button, "Context")
        .click();
    chat.run();
    assert!(
        chat.query_by_label(format_footprint_diagnostic(&ptq1).as_str())
            .is_some()
    );
    Ok(())
}

#[test]
fn memory_resolve_cache_invalidates_on_draft_history_and_memory_changes() -> Result<(), &'static str>
{
    let (mut app, _commands) = state();
    app.memory
        .add_card("Rust", "rust", "Prefer zero-copy iterators.", false)?;
    app.memory_revision = app.memory_revision.wrapping_add(1);
    app.commit_history(history("hello world"));
    app.input = "plain draft".into();
    let baseline = app.context_estimate();
    assert_eq!(baseline.memory, 0);

    // Matching input/history/memory hits the cached memory estimate.
    assert_eq!(app.context_estimate().memory, 0);

    // Keyword in input triggers the card and updates the cached estimate.
    app.input = "rust draft".into();
    let triggered = app.context_estimate();
    assert!(triggered.memory > 0);
    assert_eq!(app.context_estimate().memory, triggered.memory);

    // Equal-length committed replacement invalidates the memory cache.
    app.input = "plain draft".into();
    assert_eq!(app.context_estimate().memory, 0);
    app.commit_history(history("rust  world"));
    assert_eq!(app.context_estimate().memory, triggered.memory);

    // Memory store edits invalidate the cache.
    app.memory.cards[0].enabled = false;
    app.memory_revision = app.memory_revision.wrapping_add(1);
    assert_eq!(app.context_estimate().memory, 0);
    Ok(())
}
