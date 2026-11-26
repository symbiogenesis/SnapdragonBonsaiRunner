//! Native Settings surface for persistent instructions and deterministic context cards.
use super::{RichText, State, egui, settings_card};
use crate::context_cards::{
    MAX_CARD_CONTENT_CHARS, MAX_CARD_TITLE_CHARS, MAX_INSTRUCTIONS_CHARS, MAX_TOKEN_BUDGET,
    MIN_TOKEN_BUDGET,
};

#[derive(Default)]
pub(in crate::window) struct MemoryEditorDraft {
    title: String,
    triggers: String,
    content: String,
    pinned: bool,
    notice: Option<&'static str>,
}

impl State {
    pub(super) fn memory_preferences(&mut self, ui: &mut egui::Ui) {
        let resolved = self.memory.resolve(&self.committed, &self.input);
        ui.label(
            RichText::new(
                "Persistent instructions and pinned cards stay in the system message \
                 for prefix cache stability. Triggered cards attach to the current user \
                 turn when their keywords appear.",
            )
            .small()
            .weak(),
        );
        ui.add_space(8.0);

        settings_card(ui, |ui| {
            ui.label(RichText::new("Persistent instructions").strong());
            ui.add_space(6.0);
            let instr_response = ui.add(
                egui::TextEdit::multiline(&mut self.memory.persistent_instructions)
                    .id_salt("memory-persistent-instructions")
                    .desired_rows(3)
                    .desired_width(f32::INFINITY)
                    .char_limit(MAX_INSTRUCTIONS_CHARS)
                    .hint_text("Optional standing instructions included in every conversation."),
            );
            if instr_response.changed() {
                self.recovery_dirty = true;
                self.memory_revision = self.memory_revision.wrapping_add(1);
            }
            if resolved.instructions_rejected_control {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(
                        "Persistent instructions contain a reserved model control token and were \
                         excluded from requests.",
                    )
                    .small()
                    .color(ui.visuals().warn_fg_color),
                );
            } else if resolved.instructions_rejected_budget {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(
                        "Persistent instructions exceed the selected memory token budget.",
                    )
                    .small()
                    .color(ui.visuals().warn_fg_color),
                );
            }
            ui.add_space(8.0);
            let budget_changed = ui
                .add(
                    egui::Slider::new(
                        &mut self.memory.token_budget,
                        MIN_TOKEN_BUDGET..=MAX_TOKEN_BUDGET,
                    )
                    .step_by(64.0)
                    .text("Memory token budget"),
                )
                .changed();
            if budget_changed {
                self.recovery_dirty = true;
                self.memory_revision = self.memory_revision.wrapping_add(1);
            }
        });

        ui.add_space(8.0);
        settings_card(ui, |ui| {
            ui.label(RichText::new("Context cards").strong());
            ui.add_space(6.0);
            let mut remove_id = None;
            let mut cards_changed = false;
            if self.memory.cards.is_empty() {
                ui.label(RichText::new("No context cards added yet.").small().weak());
            } else {
                for card in &mut self.memory.cards {
                    ui.group(|ui| {
                        ui.horizontal_wrapped(|ui| {
                            cards_changed |= ui.checkbox(&mut card.enabled, &card.title).changed();
                            cards_changed |= ui.checkbox(&mut card.pinned, "Pin").changed();
                            if ui.small_button("Remove").clicked() {
                                remove_id = Some(card.id);
                            }
                        });
                        if !card.pinned && !card.triggers.is_empty() {
                            ui.label(
                                RichText::new(format!("Triggers: {}", card.triggers.join(", ")))
                                    .small()
                                    .weak(),
                            );
                        }
                        ui.label(RichText::new(&card.content).small());
                        if card.has_control_tokens() {
                            ui.label(
                                RichText::new("Blocked: contains a reserved control token.")
                                    .small()
                                    .color(ui.visuals().warn_fg_color),
                            );
                        }
                    });
                    ui.add_space(4.0);
                }
            }
            if cards_changed {
                self.recovery_dirty = true;
                self.memory_revision = self.memory_revision.wrapping_add(1);
            }
            if let Some(id) = remove_id {
                self.memory.remove_card(id);
                self.recovery_dirty = true;
                self.memory_revision = self.memory_revision.wrapping_add(1);
            }

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(6.0);
            ui.label(RichText::new("Add context card").strong());
            ui.add_space(4.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.memory_draft.title)
                    .id_salt("memory-card-title")
                    .desired_width(f32::INFINITY)
                    .char_limit(MAX_CARD_TITLE_CHARS)
                    .hint_text("Card title (e.g. Project style rules)"),
            );
            ui.add_space(4.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.memory_draft.triggers)
                    .id_salt("memory-card-triggers")
                    .desired_width(f32::INFINITY)
                    .hint_text("Comma-separated keyword triggers (ignored when pinned)"),
            );
            ui.add_space(4.0);
            ui.add(
                egui::TextEdit::multiline(&mut self.memory_draft.content)
                    .id_salt("memory-card-content")
                    .desired_rows(2)
                    .desired_width(f32::INFINITY)
                    .char_limit(MAX_CARD_CONTENT_CHARS)
                    .hint_text("Card guidance to inject when active."),
            );
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.memory_draft.pinned, "Pin for every turn");
                if ui.button("Add card").clicked() {
                    match self.memory.add_card(
                        &self.memory_draft.title,
                        &self.memory_draft.triggers,
                        &self.memory_draft.content,
                        self.memory_draft.pinned,
                    ) {
                        Ok(()) => {
                            self.memory_draft = MemoryEditorDraft::default();
                            self.recovery_dirty = true;
                            self.memory_revision = self.memory_revision.wrapping_add(1);
                        }
                        Err(message) => self.memory_draft.notice = Some(message),
                    }
                }
            });
            if let Some(notice) = self.memory_draft.notice {
                ui.label(
                    RichText::new(notice)
                        .small()
                        .color(ui.visuals().warn_fg_color),
                );
            }
        });

        ui.add_space(8.0);
        settings_card(ui, |ui| {
            egui::CollapsingHeader::new("Active context inspector")
                .id_salt("memory-active-inspector")
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(format!(
                            "Estimated active memory: ~{} / {} tokens",
                            resolved.estimated_tokens, self.memory.token_budget
                        ))
                        .small(),
                    );
                    if !self.memory.persistent_instructions.trim().is_empty() {
                        let status = if resolved.instructions_included {
                            "Persistent instructions: Included"
                        } else if resolved.instructions_rejected_control {
                            "Persistent instructions: Blocked (control token)"
                        } else {
                            "Persistent instructions: Skipped (budget)"
                        };
                        ui.label(RichText::new(status).small().weak());
                    }
                    for report in &resolved.reports {
                        ui.label(
                            RichText::new(format!(
                                "{} — {}",
                                report.title,
                                report.decision.describe()
                            ))
                            .small()
                            .weak(),
                        );
                    }
                });
        });
    }
}
