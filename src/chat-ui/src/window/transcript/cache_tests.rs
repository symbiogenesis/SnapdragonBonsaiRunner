//! Work-count regressions guard streaming and search responsiveness without timing a benchmark.
use super::*;
use egui_kittest::Harness;

fn harness(transcript: Transcript) -> Harness<'static, Transcript> {
    Harness::builder()
        .with_theme(egui::Theme::Dark)
        .with_context_setup(|context| {
            crate::window::appearance::configure_fonts(context);
            context.add_font_provider(crate::window::test_font_provider());
        })
        .build_ui_state(
            |ui, transcript: &mut Transcript| {
                egui::ScrollArea::vertical()
                    .animated(false)
                    .show_viewport(ui, |ui, viewport| {
                        transcript.show(ui, ui.available_width().min(600.0), viewport, None, true)
                    });
            },
            transcript,
        )
}

fn layout_counts(transcript: &Transcript) -> Vec<usize> {
    transcript
        .messages
        .iter()
        .map(|message| message.body_layouts)
        .collect()
}

#[test]
fn markdown_code_surfaces_reflow_with_unicode_and_preserve_find() -> std::io::Result<()> {
    use crate::window::appearance;
    appearance::prepare_fonts();
    let capture =
        std::env::var_os("BONSAI_MARKDOWN_CAPTURE_DIRECTORY").map(std::path::PathBuf::from);
    if let Some(directory) = &capture {
        bonsai_windows_credentials::validate_external_path(directory)?;
        std::fs::create_dir_all(directory)?;
    }
    let source = "# A clearer reading experience\n\nMarkdown should make a response **easy to scan**, with *emphasis* where it helps and **bold with *nested emphasis***.\n\n## A little structure\n\n- Comfortable spacing between lines\n- Clear hierarchy for lists\n  - Nested details stay together\n\n> A short quotation can span several lines.\n> Keep its visual rhythm consistent.\n\n```rust\nlet greeting = \"Hello, 界 🌱\";\n\nprintln!(\"{greeting}\");\n```\n\nUse `inline code` within a sentence.\n\n```text\nA second block\n```";
    for (name, size, theme) in [
        (
            "markdown-light.png",
            [760.0, 940.0],
            appearance::Theme::Light,
        ),
        (
            "markdown-dark-narrow.png",
            [480.0, 1_100.0],
            appearance::Theme::Dark,
        ),
    ] {
        let mut transcript = Transcript::default();
        transcript.push(Role::Assistant, source.to_owned());
        let mut chat = Harness::builder()
            .with_size(size)
            .with_theme(match theme {
                appearance::Theme::Light => egui::Theme::Light,
                appearance::Theme::System | appearance::Theme::Dark => egui::Theme::Dark,
            })
            .with_context_setup(move |context| {
                crate::window::configure_test_context(context);
                appearance::Preferences {
                    theme,
                    ..Default::default()
                }
                .apply(context);
            })
            .build_ui_state(
                |ui, transcript: &mut Transcript| {
                    egui::ScrollArea::vertical().animated(false).show_viewport(
                        ui,
                        |ui, viewport| {
                            transcript.show(
                                ui,
                                ui.available_width().min(700.0),
                                viewport,
                                None,
                                true,
                            )
                        },
                    );
                },
                transcript,
            );
        chat.run();
        assert_eq!(
            chat.ctx.style_of(chat.ctx.theme()).visuals.dark_mode,
            theme == appearance::Theme::Dark
        );
        let before = chat.state().messages[0]
            .cache
            .as_ref()
            .ok_or_else(|| std::io::Error::other("Missing formatted cache"))?;
        let before_text = before
            .first_text()
            .ok_or_else(|| std::io::Error::other("Missing formatted text segment"))?;
        assert_eq!(before_text.code_panels.len(), 2);
        assert!(before_text.code_panels.iter().all(|panel| {
            panel.is_finite()
                && panel.left() == 0.0
                && panel.width() == before.width
                && panel.height() >= 24.0
        }));
        assert!(before_text.code_panels[0].height() > before_text.code_panels[1].height());
        let panels = before_text.code_panels.clone();
        if let Some(directory) = &capture {
            chat.render()
                .map_err(std::io::Error::other)?
                .save(directory.join(name))
                .map_err(std::io::Error::other)?;
        }
        chat.state_mut().find_query("greeting");
        chat.run();
        let message = &chat.state().messages[0];
        assert_eq!(message.body, source);
        assert_eq!(
            message
                .cache
                .as_ref()
                .and_then(LayoutCache::first_text)
                .map(|text| &text.code_panels),
            Some(&panels)
        );
        assert_active_underline_matches_glyphs(message);
        chat.run();
        assert_eq!(chat.state().messages[0].body, source);
    }
    Ok(())
}

#[test]
fn unchanged_layouts_reuse_cache_and_reasoning_visibility_invalidates_it() {
    let mut transcript = Transcript::default();
    transcript.push(Role::Assistant, "A visible hit".to_owned());
    transcript.append("Private reasoning", true);
    transcript.search.query = "hit".to_owned();
    let mut chat = harness(transcript);
    chat.run();
    let initial = chat.state().messages[0].cache_updates;
    assert_eq!(initial, 1);
    for _ in 0..3 {
        chat.run();
    }
    assert_eq!(chat.state().messages[0].cache_updates, initial);

    for visible in [true, false, true] {
        chat.state_mut().messages[0].show_reasoning = visible;
        let previous = chat.state().messages[0].cache_updates;
        chat.run();
        let message = &chat.state().messages[0];
        assert_eq!(message.cache_updates, previous + 1);
        assert_eq!(message.body_layouts, 1);
        assert!(
            message
                .cache
                .as_ref()
                .is_some_and(|cache| !cache.reasoning.is_empty() == visible)
        );
        assert_active_underline_matches_glyphs(message);
        chat.run();
        assert_eq!(chat.state().messages[0].cache_updates, previous + 1);
    }
}

#[test]
fn hidden_reasoning_streams_without_cache_updates_and_expands_complete_text() {
    let mut transcript = Transcript::default();
    transcript.push(Role::Assistant, "A visible hit".to_owned());
    transcript.search.query = "hit".to_owned();
    let mut chat = harness(transcript);
    chat.run();
    let initial_height = chat.state().messages[0].height();
    let initial_updates = chat.state().messages[0].cache_updates;
    let initial_underlines = chat.state().messages[0].underline_layouts;

    for fragment in ["Private", " reasoning", " 界", " 🌱"] {
        chat.state_mut().append(fragment, true);
        chat.run();
        let message = &chat.state().messages[0];
        assert_eq!(message.cache_updates, initial_updates);
        assert_eq!(message.body_layouts, 1);
        assert_eq!(message.underline_layouts, initial_underlines);
        assert!(message.height() > initial_height);
        assert!(
            message
                .cache
                .as_ref()
                .is_some_and(|cache| cache.reasoning.is_empty())
        );
        assert_active_underline_matches_glyphs(message);
    }

    chat.state_mut().messages[0].show_reasoning = true;
    chat.run();
    let message = &chat.state().messages[0];
    assert_eq!(message.cache_updates, initial_updates + 1);
    assert_eq!(message.body_layouts, 1);
    assert!(message.cache.as_ref().is_some_and(|cache| {
        cache.reasoning.len() == 1 && cache.reasoning[0].job.text == "Private reasoning 界 🌱"
    }));

    chat.state_mut().messages[0].show_reasoning = false;
    chat.run();
    let hidden_updates = chat.state().messages[0].cache_updates;
    chat.state_mut().append(" later text", true);
    chat.run();
    assert_eq!(chat.state().messages[0].cache_updates, hidden_updates);
    chat.state_mut().messages[0].show_reasoning = true;
    chat.run();
    let message = &chat.state().messages[0];
    assert_eq!(message.cache_updates, hidden_updates + 1);
    assert!(message.cache.as_ref().is_some_and(|cache| {
        cache.reasoning.len() == 1
            && cache.reasoning[0].job.text == "Private reasoning 界 🌱 later text"
    }));
    assert_active_underline_matches_glyphs(message);
}

#[test]
fn reader_size_reflows_styled_text_reasoning_and_find_once() {
    let body = "# Heading 界\nUse `needle` here.\n```rust\nlet x = 1;\n```";
    let mut transcript = Transcript::default();
    transcript.push(Role::User, "A literal draft".to_owned());
    transcript.push(Role::Assistant, body.to_owned());
    transcript.append("Visible reasoning 🌱", true);
    transcript.messages[1].show_reasoning = true;
    transcript.search.query = "needle".to_owned();
    let mut chat = harness(transcript);
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![1, 1]);
    assert_active_underline_matches_glyphs(&chat.state().messages[1]);
    let initial_updates = chat.state().messages[1].cache_updates;
    let initial_underline = chat.state().messages[1].underline_layouts;
    let initial_height = chat.state().messages[1].height();

    // Width, DPI, colors and source bytes stay unchanged. Font choice alone
    // must invalidate both galleys and the geometry of the active Find result.
    chat.ctx.data_mut(|data| {
        data.insert_temp(egui::Id::unique("reading-text-size"), 24.0_f32);
    });
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![2, 2]);
    let message = &chat.state().messages[1];
    assert_eq!(message.body, body);
    assert_eq!(message.cache_updates, initial_updates + 1);
    assert_eq!(message.underline_layouts, initial_underline + 1);
    assert!(message.height() > initial_height);
    let Some(cache) = &message.cache else {
        assert!(message.cache.is_some(), "Expected a reader layout cache");
        return;
    };
    let Some(text) = cache.first_text() else {
        assert!(
            cache.first_text().is_some(),
            "Expected text segment in cache"
        );
        return;
    };
    assert!(text.body.job.text.contains("Heading 界"));
    assert!(text.body.job.text.contains("Use needle here."));
    assert!(!text.body.job.text.contains('`'));
    assert!(
        text.body
            .job
            .sections
            .iter()
            .any(|section| { section.format.font_id == FontId::monospace(24.0) })
    );
    assert!(
        text.body
            .job
            .sections
            .iter()
            .any(|section| { section.format.font_id.size.to_bits() == 42.0_f32.to_bits() })
    );
    assert!(
        cache.reasoning.len() == 1
            && cache.reasoning[0].job.text == "Visible reasoning 🌱"
            && cache.reasoning[0]
                .job
                .sections
                .iter()
                .all(|section| section.format.font_id == FontId::proportional(21.0))
    );
    assert_active_underline_matches_glyphs(message);
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![2, 2]);
    assert_eq!(chat.state().messages[1].cache_updates, initial_updates + 1);
    assert_eq!(
        chat.state().messages[1].underline_layouts,
        initial_underline + 1
    );
}

#[test]
fn find_discards_replaced_scroll_offsets_before_rendering_shorter_unicode_text() {
    let mut transcript = Transcript::default();
    transcript.push(Role::User, "retained hit".to_owned());
    transcript.push(Role::Assistant, "A long prefix before hit".to_owned());
    transcript.search.query = "hit".to_owned();
    let mut chat = harness(transcript);
    chat.run();
    chat.state_mut().search.active = 1;
    let target = chat.state().search.matches.get(1).cloned();
    chat.state_mut().search.target = target;

    // The old byte offset is beyond the replacement's length and is not a
    // character boundary. Rendering must never slice with that stale offset.
    chat.state_mut().replace_response("\u{754c} hit", None);
    chat.run();
    assert!(chat.state().search.target.is_none());
    assert_eq!(chat.state().search.active, 1);
    assert_eq!(chat.state().search.matches[1].range, 4..7);
    assert_active_underline_matches_glyphs(&chat.state().messages[1]);
    assert_eq!(chat.state().messages[0].body_layouts, 1);
}

#[test]
fn streamed_reasoning_and_search_only_relayout_affected_message_bodies() {
    let mut transcript = Transcript::default();
    transcript.push(Role::User, "Hit in the first message".to_owned());
    transcript.push(Role::Assistant, "An unrelated answer".to_owned());
    transcript.push(Role::User, "A second hit".to_owned());
    transcript.push(Role::Assistant, "A streamed hit".to_owned());
    transcript.search.query = "hit".to_owned();
    let mut chat = harness(transcript);
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![1, 1, 1, 1]);

    chat.state_mut().append(" Private reasoning", true);
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![1, 1, 1, 1]);
    chat.state_mut().messages[3].show_reasoning = true;
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![1, 1, 1, 1]);
    assert!(
        chat.state().messages[3]
            .cache
            .as_ref()
            .is_some_and(|cache| !cache.reasoning.is_empty())
    );

    chat.state_mut().append(" and another hit", false);
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![1, 1, 1, 2]);
    assert_eq!(chat.state().search.matches.len(), 4);

    chat.state_mut().search.active = 1;
    chat.state_mut().search.revision += 1;
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![1, 1, 1, 2]);
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![1, 1, 1, 2]);
}

#[test]
fn transcript_highlights_follow_selection_colors_without_reflowing_plain_messages() {
    let mut transcript = Transcript::default();
    transcript.push(Role::User, "highlight".to_owned());
    transcript.push(Role::Assistant, "plain".to_owned());
    transcript.search.query = "highlight".to_owned();
    let mut chat = harness(transcript);
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![1, 1]);
    let mut visuals = chat.ctx.style_of(egui::Theme::Dark).visuals.clone();
    visuals.selection.bg_fill = Color32::RED;
    chat.ctx.set_visuals(visuals);
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![2, 1]);
}

#[test]
fn highlighted_layout_cache_keeps_constant_size_keys_during_streaming_and_navigation() {
    let mut transcript = Transcript::default();
    let highlights = 10_000;
    transcript.push(Role::User, "hit ".repeat(highlights));
    transcript.push(Role::Assistant, "A streamed answer".to_owned());
    transcript.search.query = "hit".to_owned();
    let mut chat = harness(transcript);
    chat.run();
    assert_eq!(chat.state().search.matches.len(), highlights);
    let visits = chat.state().messages[0].highlight_range_visits.get();
    assert_eq!(visits, highlights);

    for _ in 0..8 {
        chat.state_mut().append(" plain continuation", false);
        chat.run();
    }
    assert_eq!(chat.state().messages[0].body_layouts, 1);
    assert_eq!(
        chat.state().messages[0].highlight_range_visits.get(),
        visits
    );
    assert!(
        chat.state().messages[0]
            .cache
            .as_ref()
            .is_some_and(|cache| {
                cache.highlights.count == highlights && cache.selected.as_ref() == Some(&(0..3))
            })
    );

    // Selection changes only rebuild the active underline. All ten thousand
    // highlight sections and the shared galley remain unchanged.
    let body = chat.state().messages[0]
        .cache
        .as_ref()
        .and_then(LayoutCache::first_text)
        .map(|text| Arc::clone(&text.body));
    let underline_layouts = chat.state().messages[0].underline_layouts;
    chat.state_mut().search.active = highlights - 1;
    chat.state_mut().search.revision += 1;
    chat.run();
    assert_eq!(
        chat.state().messages[0].highlight_range_visits.get(),
        visits
    );
    assert_eq!(chat.state().messages[0].body_layouts, 1);
    assert_eq!(
        chat.state().messages[0].underline_layouts,
        underline_layouts + 1
    );
    assert!(
        chat.state().messages[0]
            .cache
            .as_ref()
            .is_some_and(|cache| {
                body.as_ref().is_some_and(|body| {
                    cache.first_text().is_some_and(|text| {
                        Arc::ptr_eq(body, &text.body) && !text.underline.is_empty()
                    })
                }) && cache.selected.as_ref()
                    == Some(&((highlights - 1) * 4..(highlights - 1) * 4 + 3))
            })
    );
    chat.run();
    assert_eq!(
        chat.state().messages[0].highlight_range_visits.get(),
        visits
    );
    assert_eq!(
        chat.state().messages[0].underline_layouts,
        underline_layouts + 1
    );
}

fn assert_active_underline_matches_glyphs(message: &Message) {
    let Some(cache) = &message.cache else {
        assert!(message.cache.is_some(), "Expected a layout cache");
        return;
    };
    let Some(selected) = &cache.selected else {
        assert!(cache.selected.is_some(), "Expected an active Find result");
        return;
    };
    let Some(text) = cache.first_text() else {
        assert!(cache.first_text().is_some(), "Expected text segment");
        return;
    };
    let selected = text.source_map.range(selected.clone());
    let start = text.body.job.text[..selected.start].chars().count();
    let end = start + text.body.job.text[selected].chars().count();
    let stroke = egui::Stroke::new(2.0, cache.highlight_color);
    let mut offset = 0;
    let mut selected_glyphs = 0;
    for row in &text.body.rows {
        for (index, glyph) in row.glyphs.iter().enumerate() {
            let mut y = glyph.logical_rect().bottom();
            stroke.round_center_to_pixel(cache.pixels_per_point, &mut y);
            let left = egui::pos2(glyph.pos.x, y) + row.pos.to_vec2();
            let right = egui::pos2(glyph.max_x(), y) + row.pos.to_vec2();
            let underlined = text.underline.iter().any(|line| {
                line[0].y.to_bits() == left.y.to_bits()
                    && line[0].x <= left.x
                    && line[1].x >= right.x
            });
            let selected = (start..end).contains(&(offset + index));
            assert_eq!(
                underlined,
                selected,
                "Glyph {} at {}",
                glyph.chr,
                offset + index
            );
            selected_glyphs += usize::from(selected);
        }
        offset += row.char_count_including_newline().0;
    }
    assert_ne!(text.underline.len(), 0);
    assert!(text.underline.len() <= selected_glyphs);
}

#[test]
fn markdown_find_maps_unicode_source_and_reuses_display_layout_on_navigation()
-> Result<(), &'static str> {
    let source = "# Heading 界\n\nA **needle 🌱** and `needle 界`.";
    let mut transcript = Transcript::default();
    transcript.push(Role::Assistant, source.to_owned());
    transcript.find_query("needle");
    let mut chat = harness(transcript);
    chat.run();
    assert_eq!(chat.state().search.matches.len(), 2);
    let message = &chat.state().messages[0];
    let cache = message.cache.as_ref().ok_or("Expected rendered layout")?;
    let text = cache.first_text().ok_or("Expected text segment")?;
    assert_eq!(message.body, source);
    assert!(chat.state().plain_text().contains(source));
    assert!(text.body.job.text.contains("A needle 🌱 and needle 界."));
    assert!(!text.body.job.text.contains("**"));
    assert!(!text.body.job.text.contains('`'));
    for found in &chat.state().search.matches {
        assert_eq!(source.get(found.range.clone()), Some("needle"));
        assert_eq!(
            text.body
                .job
                .text
                .get(text.source_map.range(found.range.clone())),
            Some("needle")
        );
    }
    assert_active_underline_matches_glyphs(message);
    let galley = Arc::clone(&text.body);
    let initial_underlines = message.underline_layouts;
    chat.state_mut().search.active = 1;
    chat.run();
    let message = &chat.state().messages[0];
    let cache = message.cache.as_ref().ok_or("Expected retained layout")?;
    let text = cache.first_text().ok_or("Expected retained text segment")?;
    assert!(Arc::ptr_eq(&galley, &text.body));
    assert_eq!(message.body_layouts, 1);
    assert_eq!(message.underline_layouts, initial_underlines + 1);
    assert_active_underline_matches_glyphs(message);
    Ok(())
}

#[test]
fn active_find_underline_maps_unicode_wrapping_and_newlines_without_relayout() {
    let matched = format!("世界 {}\n🌱 continuation", "wrapped words ".repeat(12));
    let body = format!("界🌱 before\n{matched}\nAfter {matched}");
    let mut transcript = Transcript::default();
    transcript.push(Role::User, body);
    transcript.search.query = matched;
    let mut chat = harness(transcript);
    chat.run();
    assert_eq!(chat.state().search.matches.len(), 2);
    assert_active_underline_matches_glyphs(&chat.state().messages[0]);
    let initial = chat.state().messages[0].underline_layouts;
    assert!(
        chat.state().messages[0]
            .cache
            .as_ref()
            .and_then(LayoutCache::first_text)
            .is_some_and(|text| {
                text.underline.len() >= 3
                    && text
                        .body
                        .job
                        .sections
                        .iter()
                        .all(|section| section.format.underline.is_empty())
            })
    );

    chat.state_mut().search.active = 1;
    chat.run();
    assert_eq!(chat.state().messages[0].body_layouts, 1);
    assert_eq!(chat.state().messages[0].underline_layouts, initial + 1);
    assert_active_underline_matches_glyphs(&chat.state().messages[0]);
    chat.run();
    assert_eq!(chat.state().messages[0].underline_layouts, initial + 1);

    chat.set_size(egui::vec2(360.0, 800.0));
    chat.run();
    assert_eq!(chat.state().messages[0].body_layouts, 2);
    assert_eq!(chat.state().messages[0].underline_layouts, initial + 2);
    assert_active_underline_matches_glyphs(&chat.state().messages[0]);

    // A scale change invalidates both the galley and the active geometry.
    chat.ctx.set_pixels_per_point(1.5);
    chat.run();
    assert_eq!(chat.state().messages[0].body_layouts, 3);
    assert_eq!(chat.state().messages[0].underline_layouts, initial + 3);
    assert_active_underline_matches_glyphs(&chat.state().messages[0]);
}

#[test]
fn find_sync_live_query_refreshes_highlights_without_resetting_unchanged_matches() {
    let mut transcript = Transcript::default();
    transcript.push(Role::User, "hit cat hit".to_owned());
    transcript.sync_find_query("hit");
    let mut chat = harness(transcript);
    chat.run();
    chat.state_mut().search.active = 1;
    chat.state_mut().sync_find_query("hit");
    chat.run();
    assert_eq!(chat.state().search.active, 1);
    chat.state_mut().sync_find_query("cat");
    chat.run();
    assert!(
        chat.state().messages[0]
            .cache
            .as_ref()
            .is_some_and(|cache| {
                cache.highlights.count == 1 && cache.selected.as_ref() == Some(&(4..7))
            })
    );
    chat.state_mut().sync_find_query("");
    assert!(!chat.state().find_open());
    chat.run();
    assert!(chat.state().search.matches.is_empty());
    assert!(
        chat.state().messages[0]
            .cache
            .as_ref()
            .is_some_and(|cache| {
                cache.highlights.count == 0
                    && cache
                        .first_text()
                        .is_some_and(|text| text.underline.is_empty())
            })
    );
}

#[test]
fn find_sync_transferred_query_uses_native_highlights_and_first_match_scroll() {
    let mut transcript = Transcript::default();
    transcript.push(Role::User, "hit cat hit".to_owned());
    transcript.find_query("hit");
    assert_eq!(transcript.search.active, 0);
    assert!(
        transcript
            .search
            .target
            .as_ref()
            .is_some_and(|found| found.range == (0..3))
    );
    let mut chat = harness(transcript);
    chat.run();
    assert_eq!(chat.state().search.matches.len(), 2);
    assert!(chat.state().search.target.is_none());
    assert!(
        chat.state().messages[0]
            .cache
            .as_ref()
            .is_some_and(|cache| {
                cache.highlights.count == 2 && cache.selected.as_ref() == Some(&(0..3))
            })
    );
    assert_active_underline_matches_glyphs(&chat.state().messages[0]);
    chat.state_mut().search.latest = true;
    chat.state_mut().find_query("cat");
    assert!(!chat.state().search.latest);
    chat.run();
    assert!(
        chat.state().messages[0]
            .cache
            .as_ref()
            .is_some_and(|cache| {
                cache.highlights.count == 1 && cache.selected.as_ref() == Some(&(4..7))
            })
    );
    assert_active_underline_matches_glyphs(&chat.state().messages[0]);
}

#[test]
fn highlight_identity_tracks_same_length_queries_replacement_and_result_eviction() {
    let mut transcript = Transcript::default();
    transcript.push(Role::User, "hit cat".to_owned());
    transcript.push(Role::Assistant, "hit cat".to_owned());
    transcript.search.query = "hit".to_owned();
    let mut chat = harness(transcript);
    chat.run();
    let query_revision = chat.state().search.query_revision;
    assert_eq!(layout_counts(chat.state()), vec![1, 1]);

    // Counts and byte lengths stay unchanged while every highlight moves.
    chat.state_mut().search.query = "cat".to_owned();
    chat.state_mut().search.invalidate();
    chat.run();
    assert_ne!(chat.state().search.query_revision, query_revision);
    assert_eq!(layout_counts(chat.state()), vec![2, 2]);
    assert!(
        chat.state()
            .search
            .matches
            .iter()
            .all(|found| found.range == (4..7))
    );
    assert!(
        chat.state().messages[0]
            .cache
            .as_ref()
            .is_some_and(|cache| { cache.selected.as_ref() == Some(&(4..7)) })
    );
    assert_active_underline_matches_glyphs(&chat.state().messages[0]);

    chat.state_mut().search.active = 1;
    chat.state_mut().search.revision += 1;
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![2, 2]);
    assert!(
        chat.state().messages[0]
            .cache
            .as_ref()
            .is_some_and(|cache| {
                cache.selected.is_none()
                    && cache
                        .first_text()
                        .is_some_and(|text| text.underline.is_empty())
            })
    );
    assert!(
        chat.state().messages[1]
            .cache
            .as_ref()
            .is_some_and(|cache| { cache.selected.as_ref() == Some(&(4..7)) })
    );
    assert_active_underline_matches_glyphs(&chat.state().messages[1]);
    chat.state_mut().replace_response("cat hit", None);
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![2, 3]);
    assert_eq!(chat.state().search.matches[1].range, 0..3);
    assert!(
        chat.state().messages[1]
            .cache
            .as_ref()
            .is_some_and(|cache| {
                cache.body_revision == chat.state().messages[1].body_revision
                    && cache.selected.as_ref() == Some(&(0..3))
            })
    );
    assert_active_underline_matches_glyphs(&chat.state().messages[1]);

    // Closing Find clears its query. Reopening the same query must acquire a
    // new identity even though the message bodies never changed.
    chat.state_mut().search.query.clear();
    chat.state_mut().search.invalidate();
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![3, 4]);
    assert!(chat.state().messages.iter().all(|message| {
        message.cache.as_ref().is_some_and(|cache| {
            cache.highlights.count == 0
                && cache
                    .first_text()
                    .is_some_and(|text| text.underline.is_empty())
        })
    }));
    chat.state_mut().search.query = "cat".to_owned();
    chat.state_mut().search.invalidate();
    chat.run();
    assert_eq!(layout_counts(chat.state()), vec![4, 5]);
    assert!(chat.state().messages.iter().all(|message| {
        message
            .cache
            .as_ref()
            .is_some_and(|cache| cache.highlights.count == 1)
    }));

    // Removing a saturated prefix exposes a previously hidden match without
    // changing that later message's body or the effective query.
    let mut transcript = Transcript::default();
    transcript.push(Role::User, "hit ".repeat(10_001));
    transcript.push(Role::Assistant, "hit".to_owned());
    transcript.search.query = "hit".to_owned();
    let mut chat = harness(transcript);
    chat.run();
    assert_eq!(chat.state().messages[1].highlight_range_visits.get(), 0);
    chat.state_mut().messages.remove(0);
    chat.state_mut().search.invalidate();
    chat.run();
    assert_eq!(chat.state().search.matches.len(), 1);
    assert_eq!(chat.state().messages[0].body_layouts, 2);
    assert_eq!(chat.state().messages[0].highlight_range_visits.get(), 1);
}

fn assert_byte_accounting(transcript: &Transcript) {
    assert_eq!(
        transcript.text_bytes,
        transcript
            .messages
            .iter()
            .map(Message::text_bytes)
            .sum::<usize>()
    );
}

#[test]
fn transcript_accounting_streaming_visits_no_completed_messages() {
    let mut transcript = Transcript::default();
    for _ in 0..400 {
        transcript.push(Role::User, "Completed 界 🌱".to_owned());
    }
    transcript.push(Role::Assistant, String::new());
    for _ in 0..4096 {
        transcript.append("答🌱", false);
        transcript.append("考", true);
    }
    assert_byte_accounting(&transcript);
    assert_eq!(transcript.messages.len(), 401);
    assert_eq!(transcript.trim_message_visits, 0);
    assert_eq!(transcript.trim_batches, 0);
}

#[test]
fn transcript_accounting_caps_empty_messages_before_bytes_require_eviction() {
    let mut transcript = Transcript::default();
    for _ in 0..MAX_MESSAGES {
        transcript.push(Role::User, String::new());
    }
    let retained = transcript.messages[1].id;
    transcript.push(Role::Assistant, String::new());
    assert_eq!(transcript.messages.len(), MAX_MESSAGES);
    assert_eq!(transcript.messages[0].id, retained);
    assert_eq!(transcript.trim_message_visits, 1);
    assert_eq!(transcript.trim_batches, 1);
    assert!(transcript.valid_saved());
    assert!(
        transcript
            .messages
            .last()
            .is_some_and(|message| message.pending)
    );
    for _ in 0..4096 {
        transcript.append("x", false);
    }
    assert_byte_accounting(&transcript);
    assert_eq!(transcript.trim_message_visits, 1);
    assert_eq!(transcript.trim_batches, 1);
}

#[test]
fn transcript_accounting_count_eviction_preserves_find_and_recovery()
-> Result<(), serde_json::Error> {
    let mut transcript = Transcript::default();
    for _ in 0..MAX_MESSAGES {
        transcript.push(Role::User, "hit".to_owned());
    }
    transcript.search.query = "hit".to_owned();
    transcript.search.refresh(&transcript.messages);
    transcript.search.active = 10;
    let selected = transcript.search.matches[10].clone();
    transcript.push(Role::Assistant, "new hit".to_owned());
    transcript.search.refresh(&transcript.messages);
    assert_eq!(transcript.messages.len(), MAX_MESSAGES);
    assert_eq!(transcript.search.matches.len(), MAX_MESSAGES);
    assert_eq!(transcript.search.active, 9);
    assert!(transcript.search.matches[transcript.search.active] == selected);
    assert!(transcript.search.matches.iter().all(|found| {
        transcript
            .messages
            .iter()
            .any(|message| message.id == found.message)
    }));
    assert_byte_accounting(&transcript);
    let restored: Transcript = serde_json::from_slice(&serde_json::to_vec(&transcript)?)?;
    assert_eq!(restored.messages.len(), MAX_MESSAGES);
    assert_eq!(restored.messages[0].body, "hit");
    assert_eq!(restored.messages[MAX_MESSAGES - 1].body, "new hit");
    assert_byte_accounting(&restored);
    Ok(())
}

#[test]
fn transcript_accounting_count_and_byte_eviction_remove_one_combined_prefix() {
    let mut transcript = Transcript::default();
    transcript.push(Role::User, "x".repeat(MAX_BYTES / 2));
    transcript.push(Role::Assistant, "y".repeat(MAX_BYTES / 2));
    for _ in 2..MAX_MESSAGES {
        transcript.push(Role::User, String::new());
    }
    let retained = transcript.messages[2].id;
    // The count bound requires one eviction, but freeing that message alone
    // leaves the byte budget exceeded. Drain the combined prefix only once.
    transcript.push(Role::Assistant, "z".repeat(MAX_BYTES / 2 + 1));
    assert_eq!(transcript.messages.len(), MAX_MESSAGES - 1);
    assert_eq!(transcript.messages[0].id, retained);
    assert_eq!(transcript.trim_message_visits, 2);
    assert_eq!(transcript.trim_batches, 1);
    assert!(transcript.valid_saved());
    assert_byte_accounting(&transcript);
}

#[test]
fn transcript_accounting_bulk_eviction_drains_only_the_required_prefix() {
    let mut transcript = Transcript::default();
    for _ in 0..8 {
        transcript.push(Role::User, "x".repeat(MAX_BYTES / 8));
    }
    transcript.push(Role::Assistant, String::new());
    let retained_id = transcript.messages[4].id;
    transcript.replace_response(&"y".repeat(MAX_BYTES / 2), None);
    assert_byte_accounting(&transcript);
    assert_eq!(transcript.text_bytes, MAX_BYTES);
    assert_eq!(transcript.messages.len(), 5);
    assert_eq!(transcript.messages[0].id, retained_id);
    assert_eq!(transcript.trim_message_visits, 4);
    assert_eq!(transcript.trim_batches, 1);

    // A streamed continuation crosses the bound again without revisiting any
    // message already evicted or shifting the suffix more than once.
    transcript.append("界", true);
    assert_byte_accounting(&transcript);
    assert_eq!(transcript.messages.len(), 4);
    assert_eq!(transcript.trim_message_visits, 5);
    assert_eq!(transcript.trim_batches, 2);
}

#[test]
fn transcript_accounting_preserves_replacement_stage_clear_and_last_two() {
    let mut transcript = Transcript::default();
    transcript.append("ignored without an assistant", false);
    assert_byte_accounting(&transcript);
    transcript.push(Role::User, "界".to_owned());
    transcript.append("ignored after a user", true);
    assert_eq!(transcript.text_bytes, "界".len());
    transcript.replace_response("🌱 answer", Some("考"));
    assert_byte_accounting(&transcript);
    transcript.stage();
    assert_eq!(transcript.messages.len(), 3);
    assert_byte_accounting(&transcript);
    transcript.replace_response("短", None);
    assert_byte_accounting(&transcript);
    transcript.replace_response("", None);
    assert_eq!(transcript.messages.len(), 2);
    assert_byte_accounting(&transcript);
    transcript.replace_response("replaced", Some("🌱"));
    assert_byte_accounting(&transcript);

    transcript.append(&"x".repeat(MAX_BYTES), false);
    assert_eq!(transcript.messages.len(), 2);
    assert!(transcript.text_bytes > MAX_BYTES);
    assert!(!transcript.valid_saved());
    assert_byte_accounting(&transcript);
    transcript.clear();
    assert_eq!(transcript.text_bytes, 0);
    transcript.push(Role::Assistant, String::new());
    transcript.remove_empty_response();
    assert!(transcript.messages.is_empty());
    assert_byte_accounting(&transcript);
}

#[test]
fn transcript_accounting_restore_rebuilds_total_without_changing_saved_schema()
-> Result<(), serde_json::Error> {
    let mut transcript = Transcript::default();
    transcript.push(Role::User, "界 🌱".to_owned());
    transcript.replace_response("答", Some("考"));
    let bytes = serde_json::to_vec(&transcript)?;
    let document: serde_json::Value = serde_json::from_slice(&bytes)?;
    assert_eq!(document.as_object().map(serde_json::Map::len), Some(1));
    assert!(document.get("messages").is_some());
    let mut restored: Transcript = serde_json::from_slice(&bytes)?;
    assert_byte_accounting(&restored);
    assert_eq!(restored.text_bytes, transcript.text_bytes);
    restored.append(" 🌱", false);
    restored.append("思", true);
    restored.replace_response("shorter", None);
    assert_byte_accounting(&restored);
    assert_eq!(restored.trim_message_visits, 0);

    let mut message_limit = Transcript::default();
    for _ in 0..MAX_MESSAGES {
        message_limit.push(Role::User, String::new());
    }
    assert!(serde_json::from_slice::<Transcript>(&serde_json::to_vec(&message_limit)?).is_ok());
    let mut saved = serde_json::to_value(&message_limit)?;
    if let Some(messages) = saved["messages"].as_array_mut() {
        messages.push(serde_json::json!({"role": "User", "body": "", "reasoning": ""}));
    }
    assert!(serde_json::from_value::<Transcript>(saved).is_err());
    Ok(())
}

#[test]
fn incremental_streaming_layout_reuses_completed_chunks_and_preserves_find_offsets()
-> Result<(), &'static str> {
    let mut transcript = Transcript::default();
    transcript.start_turn("Prompt".to_owned(), 0);
    transcript.append(
        "First paragraph **bold**.\n\n```rust\nlet x = 1;\n\nlet y = 2;\n```\n\nActive needle",
        false,
    );
    let mut chat = harness(transcript);
    chat.run();

    let message = &chat.state().messages[1];
    let cache = message.cache.as_ref().ok_or("Expected streaming cache")?;
    assert_eq!(cache.segments.len(), 3);
    let CachedSegment::Text(first) = &cache.segments[0] else {
        return Err("Expected first paragraph segment");
    };
    let CachedSegment::Text(fence) = &cache.segments[1] else {
        return Err("Expected closed fence segment");
    };
    let first_galley = Arc::clone(&first.body);
    let first_source_map = Arc::clone(&first.source_map);
    let fence_galley = Arc::clone(&fence.body);
    let fence_source_map = Arc::clone(&fence.source_map);
    let shaped_before = message.shaped_galleys;
    assert_eq!(shaped_before, 3);

    // Streaming tokens into the active tail block only reshapes that tail block.
    for token in [" one", " two", " three"] {
        chat.state_mut().append(token, false);
        chat.run();
    }
    let message = &chat.state().messages[1];
    let cache = message.cache.as_ref().ok_or("Expected updated cache")?;
    assert_eq!(cache.segments.len(), 3);
    let CachedSegment::Text(first_after) = &cache.segments[0] else {
        return Err("Expected first paragraph segment");
    };
    let CachedSegment::Text(fence_after) = &cache.segments[1] else {
        return Err("Expected closed fence segment");
    };
    assert!(Arc::ptr_eq(&first_galley, &first_after.body));
    assert!(Arc::ptr_eq(&first_source_map, &first_after.source_map));
    assert!(Arc::ptr_eq(&fence_galley, &fence_after.body));
    assert!(Arc::ptr_eq(&fence_source_map, &fence_after.source_map));
    assert_eq!(message.shaped_galleys, shaped_before + 3);

    // Find maps into the active tail chunk cleanly.
    chat.state_mut().find_query("needle");
    chat.run();
    let message = &chat.state().messages[1];
    let cache = message.cache.as_ref().ok_or("Expected Find cache")?;
    let found_offset = message.body.find("needle").ok_or("Missing needle")?;
    assert!(cache.text_offset_y(found_offset).is_some());
    let CachedSegment::Text(tail_after) = &cache.segments[2] else {
        return Err("Expected tail segment");
    };
    assert!(!tail_after.underline.is_empty());
    Ok(())
}

#[test]
fn streaming_reasoning_reuses_settled_paragraph_galleys_and_reflows_on_geometry_change()
-> Result<(), &'static str> {
    let mut transcript = Transcript::default();
    transcript.push(Role::Assistant, "Answer".to_owned());
    transcript.append(
        "First thinking paragraph.\n\nSecond thinking paragraph.\n\nActive",
        true,
    );
    transcript.messages[0].show_reasoning = true;
    let mut chat = harness(transcript);
    chat.run();

    let message = &chat.state().messages[0];
    let cache = message.cache.as_ref().ok_or("Expected reasoning cache")?;
    assert_eq!(cache.reasoning.len(), 3);
    assert_eq!(cache.reasoning[0].job.text, "First thinking paragraph.");
    assert_eq!(cache.reasoning[1].job.text, "Second thinking paragraph.");
    assert_eq!(cache.reasoning[2].job.text, "Active");
    let first_galley = Arc::clone(&cache.reasoning[0]);
    let second_galley = Arc::clone(&cache.reasoning[1]);
    let shaped_before = message.reasoning_galleys_shaped;
    assert_eq!(shaped_before, 3);

    // Appending tokens to the trailing paragraph only reshapes that active paragraph.
    for token in [" token", " one", " two"] {
        chat.state_mut().append(token, true);
        chat.run();
    }
    let message = &chat.state().messages[0];
    let cache = message
        .cache
        .as_ref()
        .ok_or("Expected updated reasoning cache")?;
    assert_eq!(cache.reasoning.len(), 3);
    assert!(Arc::ptr_eq(&first_galley, &cache.reasoning[0]));
    assert!(Arc::ptr_eq(&second_galley, &cache.reasoning[1]));
    assert_eq!(cache.reasoning[2].job.text, "Active token one two");
    assert_eq!(message.reasoning_galleys_shaped, shaped_before + 3);

    // Completing a new paragraph boundary settles the prior active paragraph without touching earlier ones.
    chat.state_mut().append("\n\nFourth", true);
    chat.run();
    let message = &chat.state().messages[0];
    let cache = message
        .cache
        .as_ref()
        .ok_or("Expected fourth paragraph cache")?;
    assert_eq!(cache.reasoning.len(), 4);
    assert!(Arc::ptr_eq(&first_galley, &cache.reasoning[0]));
    assert!(Arc::ptr_eq(&second_galley, &cache.reasoning[1]));
    assert_eq!(cache.reasoning[2].job.text, "Active token one two");
    assert_eq!(cache.reasoning[3].job.text, "Fourth");
    assert_eq!(message.reasoning_galleys_shaped, shaped_before + 5);
    let third_galley = Arc::clone(&cache.reasoning[2]);

    // Changing reader text size reflows all reasoning paragraphs cleanly.
    chat.ctx.data_mut(|data| {
        data.insert_temp(egui::Id::unique("reading-text-size"), 24.0_f32);
    });
    chat.run();
    let message = &chat.state().messages[0];
    let cache = message
        .cache
        .as_ref()
        .ok_or("Expected text-size reflowed cache")?;
    assert_eq!(cache.reasoning.len(), 4);
    assert!(!Arc::ptr_eq(&first_galley, &cache.reasoning[0]));
    assert!(!Arc::ptr_eq(&second_galley, &cache.reasoning[1]));
    assert!(!Arc::ptr_eq(&third_galley, &cache.reasoning[2]));
    assert_eq!(message.reasoning_galleys_shaped, shaped_before + 9);
    let size_first = Arc::clone(&cache.reasoning[0]);

    // Changing width also reflows all reasoning paragraphs cleanly.
    chat.set_size(egui::vec2(360.0, 800.0));
    chat.run();
    let message = &chat.state().messages[0];
    let cache = message
        .cache
        .as_ref()
        .ok_or("Expected width reflowed cache")?;
    assert_eq!(cache.reasoning.len(), 4);
    assert!(!Arc::ptr_eq(&size_first, &cache.reasoning[0]));
    assert_eq!(message.reasoning_galleys_shaped, shaped_before + 13);
    Ok(())
}
