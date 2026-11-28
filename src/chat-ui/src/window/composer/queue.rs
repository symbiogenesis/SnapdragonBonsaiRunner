//! Stateless queued-message strip presentation; callers own queue state and dispatch.
use super::{RichText, controls, egui};

pub(super) struct ItemView<'a> {
    pub(super) id: u64,
    pub(super) text: &'a str,
    pub(super) attachment_count: usize,
    pub(super) first_attachment_name: Option<&'a str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Action {
    SendNext,
    ClearAll,
    Edit(u64),
    Remove(u64),
}

pub(super) fn panel(
    ui: &mut egui::Ui,
    conversation_id: u64,
    items: &[ItemView<'_>],
    paused: bool,
    can_send_next: bool,
    can_edit: bool,
) -> Option<Action> {
    if items.is_empty() {
        return None;
    }
    let mut action = None;
    let count = items.len();
    let heading = if count == 1 {
        "1 queued message".to_owned()
    } else {
        format!("{count} queued messages")
    };
    let title = if paused {
        format!("{heading} · Paused")
    } else {
        heading
    };
    controls::notice_frame(ui)
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(title).small().strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(controls::quiet_button("Clear"))
                        .accessible_name("Clear queued messages")
                        .on_hover_text("Remove all queued messages for this conversation")
                        .clicked()
                    {
                        action = Some(Action::ClearAll);
                    }
                    if can_send_next
                        && ui
                            .add(controls::quiet_button("Send next"))
                            .accessible_name("Send next queued message")
                            .on_hover_text("Send the first queued message now")
                            .clicked()
                    {
                        action = Some(Action::SendNext);
                    }
                });
            });
            ui.add_space(4.0);
            egui::ScrollArea::vertical()
                .id_salt(("queued-messages-scroll", conversation_id))
                .max_height(112.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for (index, item) in items.iter().enumerate() {
                        let position = index + 1;
                        ui.push_id(("queued-message", item.id), |ui| {
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add(controls::quiet_button("×"))
                                        .accessible_name(format!(
                                            "Remove queued message {position}"
                                        ))
                                        .on_hover_text("Remove from queue")
                                        .clicked()
                                    {
                                        action = Some(Action::Remove(item.id));
                                    }
                                    let edit = ui
                                        .add_enabled(can_edit, controls::quiet_button("Edit"))
                                        .accessible_name(format!("Edit queued message {position}"))
                                        .on_hover_text("Move back to the message box to edit")
                                        .on_disabled_hover_text(
                                            "Clear or queue your current draft before editing a queued message.",
                                        );
                                    if edit.clicked() {
                                        action = Some(Action::Edit(item.id));
                                    }
                                    let summary = item_summary(item);
                                    let preview = format!("{position}. {summary}");
                                    let label = ui.add_sized(
                                        [ui.available_width().max(40.0), 20.0],
                                        egui::Label::new(RichText::new(&preview).small()).truncate(),
                                    );
                                    if !summary.is_empty() {
                                        label.on_hover_ui(|ui| {
                                            ui.set_max_width(ui.spacing().tooltip_width);
                                            ui.label(truncate_tooltip(&summary, 280));
                                        });
                                    }
                                },
                            );
                        });
                    }
                });
        });
    ui.add_space(6.0);
    action
}

fn item_summary(item: &ItemView<'_>) -> String {
    let attachment_note = match (item.attachment_count, item.first_attachment_name) {
        (0, _) => String::new(),
        (1, Some(name)) => format!("[{name}]"),
        (count, Some(name)) => format!("[{name} +{more}]", more = count - 1),
        (count, None) => format!("[{count} attachments]"),
    };
    let text = collapse_whitespace(item.text, 160);
    match (text.is_empty(), attachment_note.is_empty()) {
        (false, false) => format!("{attachment_note} {text}"),
        (false, true) => text,
        (true, false) => attachment_note,
        (true, true) => String::new(),
    }
}

fn collapse_whitespace(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut count = 0_usize;
    let mut last_space = true;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_space {
                if count >= max_chars {
                    out.push('…');
                    break;
                }
                out.push(' ');
                count += 1;
                last_space = true;
            }
            continue;
        }
        if count >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
        count += 1;
        last_space = false;
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

fn truncate_tooltip(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let taken: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{taken}…")
    } else {
        taken
    }
}
