//! Native layout/source correspondence, without model or graphics execution.
use super::*;

fn render(source: &str, matches: &[search::Match]) -> Layout {
    let parsed = parse_layout(source);
    layout_job(
        source,
        true,
        600.0,
        16.0,
        &egui::Visuals::dark(),
        matches,
        Some(&parsed),
    )
}

fn format_at<'a>(job: &'a egui::text::LayoutJob, needle: &str) -> Option<&'a egui::TextFormat> {
    let offset = job.text.find(needle)?;
    job.sections
        .iter()
        .find(|section| section.byte_range.contains(&egui::text::ByteIndex(offset)))
        .map(|section| &section.format)
}

#[test]
fn markdown_renders_native_headings_emphasis_and_literal_code() {
    let source = "# Heading 界\n\nUse **bold** and *emphasis*, ~~old~~ and `let 界 = 1;`.\n\n```rust\nfn main() {}\n```\n";
    let layout = render(source, &[]);
    assert_eq!(
        layout.job.text,
        "Heading 界\n\nUse bold and emphasis, old and let 界 = 1;.\n\nfn main() {}\n"
    );
    assert!(
        format_at(&layout.job, "Heading")
            .is_some_and(|format| format.font_id.size.to_bits() == 22.0_f32.to_bits())
    );
    assert!(format_at(&layout.job, "emphasis").is_some_and(|format| format.italics));
    assert!(format_at(&layout.job, "old").is_some_and(|format| format.strikethrough.width > 0.0));
    for needle in ["let 界", "fn main"] {
        assert!(
            format_at(&layout.job, needle).is_some_and(|format| format.font_id.family
                == egui::FontFamily::Monospace
                && format.background == egui::Visuals::dark().code_bg_color)
        );
    }
}

#[test]
fn markdown_lists_tables_and_inert_content_preserve_readable_text() {
    let source = "- First\n- [x] **Done**\n  - Nested\n\n3. Third\n4. Fourth\n\n> A quote\n\n| Name | Value |\n| --- | --- |\n| 界 | `🌱` |\n\n[Label](https://example.com) ![alt](file:///secret)\n\n<script>alert(1)</script>\n\n$x$";
    let layout = render(source, &[]);
    for needle in [
        "• First",
        "☑ Done",
        "• Nested",
        "3. Third",
        "4. Fourth",
        "│ A quote",
        "Name  │  Value",
        "界  │  🌱",
        "Label alt",
        "<script>alert(1)</script>",
        "$x$",
    ] {
        assert!(
            layout.job.text.contains(needle),
            "Missing {needle:?} in {:?}",
            layout.job.text
        );
    }
    assert!(!layout.job.text.contains("https://example.com"));
    assert!(!layout.job.text.contains("file:///secret"));
    for pair in layout.source_map.entries.windows(2) {
        assert!(pair[0].source.end <= pair[1].source.start);
        assert!(pair[0].display.end <= pair[1].display.start);
    }
}

#[test]
fn markdown_find_maps_unicode_styles_entities_and_code_lines() {
    let source =
        "# Find 界\n\n**🌱 target** and &amp; \\*escaped\\*\n\n> ```rust\n> let 界 = 1;\n> ```";
    for needle in ["界", "🌱 target", "&amp;", "escaped", "let 界 = 1;"] {
        let start = source.find(needle).unwrap_or_default();
        let range = start..start + needle.len();
        let matches = [search::Match {
            message: 1,
            range: range.clone(),
        }];
        let layout = render(source, &matches);
        let display = layout.source_map.range(range);
        let expected = if needle == "&amp;" { "&" } else { needle };
        assert_eq!(layout.job.text.get(display.clone()), Some(expected));
        assert!(layout.job.sections.iter().filter(|section| section.byte_range.start.0 < display.end && section.byte_range.end.0 > display.start).all(|section| section.format.background == egui::Visuals::dark().selection.bg_fill));
    }
}

#[test]
fn markdown_find_hidden_syntax_reveals_complete_source() {
    let source = "**界** [label](https://example.com/hidden)\n```rust\n🌱\n```";
    for needle in ["**", "hidden", "```rust"] {
        let start = source.find(needle).unwrap_or_default();
        let range = start..start + needle.len();
        let layout = render(
            source,
            &[search::Match {
                message: 1,
                range: range.clone(),
            }],
        );
        assert_eq!(layout.job.text, source);
        assert_eq!(layout.source_map.range(range.clone()), range);
    }
    let source = "`first\nlast`";
    let start = source.find("last").unwrap_or_default();
    let layout = render(
        source,
        &[search::Match {
            message: 1,
            range: start..start + 4,
        }],
    );
    assert_eq!(layout.job.text, source);
}

#[test]
fn markdown_bounds_and_user_text_retain_complete_literal_source() {
    let source = "**界** `literal`\n# Heading\n";
    let plain = layout_job(
        source,
        false,
        400.0,
        16.0,
        &egui::Visuals::dark(),
        &[],
        None,
    );
    assert_eq!(plain.job.text, source);
    for source in [
        format!("**{}**", "x".repeat(MAX_MARKDOWN_BYTES)),
        "*x* ".repeat(MAX_STYLE_SPANS),
    ] {
        let layout = render(&source, &[]);
        assert_eq!(layout.job.text, source);
        assert_eq!(layout.source_map.entries.len(), 1);
        assert_eq!(layout.source_map.offset(source.len()), source.len());
    }
    let split_tab = " ```\n\tfoo\n\tfoo\n ```";
    assert_eq!(render(split_tab, &[]).job.text, split_tab);
}

#[test]
fn markdown_streaming_incomplete_and_normalized_spans_stay_readable() {
    for (source, expected) in [
        ("~~~rust\n界", "界\n"),
        ("````rust\n```\nstill code", "```\nstill code\n"),
        ("An unfinished `界", "An unfinished `界"),
        ("`界\n🌱`", "界 🌱"),
        ("\\*literal\\* &amp; &#x1f331;", "*literal* & 🌱"),
        ("<https://example.com>", "https://example.com"),
    ] {
        let layout = render(source, &[]);
        assert_eq!(layout.job.text, expected, "Source {source:?}");
        for entry in &layout.source_map.entries {
            assert!(
                source.is_char_boundary(entry.source.start)
                    && source.is_char_boundary(entry.source.end)
            );
            assert!(
                layout.job.text.is_char_boundary(entry.display.start)
                    && layout.job.text.is_char_boundary(entry.display.end)
            );
        }
    }
}

#[test]
fn markdown_theme_changes_invalidate_only_assistant_layouts() {
    use super::super::{HighlightKey, Role, Transcript};
    use egui_kittest::Harness;

    let mut transcript = Transcript::default();
    transcript.push(Role::User, "`literal`".to_owned());
    transcript.push(Role::Assistant, "**bold** `code`".to_owned());
    let mut chat = Harness::builder()
        .with_theme(egui::Theme::Dark)
        .build_ui_state(
            |ui, transcript: &mut Transcript| {
                for message in &mut transcript.messages {
                    message.layout(
                        ui,
                        600.0,
                        &[],
                        HighlightKey {
                            query_revision: 0,
                            count: 0,
                        },
                        None,
                    );
                }
            },
            transcript,
        );
    chat.run();
    let before: Vec<_> = chat
        .state()
        .messages
        .iter()
        .map(|message| message.body_layouts)
        .collect();
    let mut visuals = chat.ctx.style_of(egui::Theme::Dark).visuals.clone();
    visuals.code_bg_color = egui::Color32::RED;
    visuals.widgets.active.fg_stroke.color = egui::Color32::GREEN;
    chat.ctx.set_visuals(visuals);
    chat.run();
    assert_eq!(chat.state().messages[0].body_layouts, before[0]);
    assert_eq!(chat.state().messages[1].body_layouts, before[1] + 1);
}
