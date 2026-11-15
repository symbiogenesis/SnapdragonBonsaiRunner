//! Conversation-list orchestration keeps workspace actions outside row rendering.
use super::{Conversation, State};
use crate::window::{Transcript, egui};

mod controls;
use controls::{Action, Availability, Row};

impl State {
    pub(in crate::window) fn conversation_list(&mut self, ui: &mut egui::Ui) {
        let field = controls::search_field(ui, &mut self.conversation_search);
        if field.clicked() || field.changed() {
            self.focus_composer = false;
        }
        if self.conversation_search.update(
            &mut self.conversations,
            self.active_conversation,
            &self.transcript,
        ) {
            self.transcript
                .sync_find_query(self.conversation_search.text.trim());
            // Drawer input is drawn after the disabled workspace. Refresh Find
            // next frame without taking focus from the sidebar.
            ui.ctx().request_repaint();
        }
        controls::list_heading(
            ui,
            self.conversation_search.rows.len(),
            self.conversations.len(),
            self.conversation_search.filtered(),
        );
        let (pinned_indices, pinned_len) = self.pinned_conversation_indices();
        let pinned = &pinned_indices[..pinned_len];
        self.conversation_search.update_scrollable(pinned);
        let mut action = None;
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.y = 4.0;
            let width = ui.available_width();
            for &index in pinned {
                if let Some(selected) = self.show_conversation_row(ui, index, width) {
                    action = Some(selected);
                }
            }
            if self.conversation_search.rows.is_empty() {
                let _ = self.conversation_search.take_scroll_reset();
                if pinned_len > 0 {
                    controls::pinned_divider(ui);
                }
                controls::empty_list(ui);
            } else if !self.conversation_search.scrollable_rows.is_empty() {
                if pinned_len > 0 {
                    controls::pinned_divider(ui);
                }
                self.conversation_scroll().show_rows(
                    ui,
                    controls::ROW_HEIGHT,
                    self.conversation_search.scrollable_rows.len(),
                    |ui, rows| {
                        let width = ui.available_width();
                        for row in rows {
                            let index = self.conversation_search.scrollable_rows[row];
                            if let Some(selected) = self.show_conversation_row(ui, index, width) {
                                action = Some(selected);
                            }
                        }
                    },
                );
            } else {
                let _ = self.conversation_search.take_scroll_reset();
            }
        });
        if let Some(action) = action {
            self.apply_conversation_action(ui.ctx(), action);
        }
    }

    fn pinned_conversation_indices(&self) -> ([usize; 2], usize) {
        let mut indices = [0; 2];
        let mut len = 0;
        let responding = self.request_conversation.filter(|_| self.model.busy());
        if let Some(owner) = responding
            && let Some(index) = self
                .conversations
                .iter()
                .position(|conversation| conversation.id == owner)
        {
            indices[len] = index;
            len += 1;
        }
        if responding != Some(self.active_conversation)
            && let Some(index) = self
                .conversations
                .iter()
                .position(|conversation| conversation.id == self.active_conversation)
        {
            indices[len] = index;
            len += 1;
        }
        (indices, len)
    }

    fn show_conversation_row(&self, ui: &mut egui::Ui, index: usize, width: f32) -> Option<Action> {
        let conversation = &self.conversations[index];
        let transcript = if conversation.id == self.active_conversation {
            Some(&self.transcript)
        } else {
            conversation
                .saved
                .as_ref()
                .map(|snapshot| &snapshot.transcript)
        };
        let view = Row {
            conversation,
            active: conversation.id == self.active_conversation,
            responding: self.model.busy() && self.request_conversation == Some(conversation.id),
            copy_available: transcript.is_some_and(|transcript| !transcript.messages.is_empty()),
            availability: Availability {
                select: self.model.workspace_available(),
                rename: !self.model.busy(),
                delete: self.can_remove_conversation(conversation.id),
            },
        };
        view.show(ui, width)
    }

    fn conversation_scroll(&mut self) -> egui::ScrollArea {
        let scroll = egui::ScrollArea::vertical()
            .id_salt("conversation-list")
            .auto_shrink([false, false])
            .min_scrolled_height(0.0);
        if self.conversation_search.take_scroll_reset() {
            return scroll.vertical_scroll_offset(0.0);
        }
        scroll
    }

    fn conversation_transcript(&self, id: u64) -> Option<&Transcript> {
        if id == self.active_conversation {
            Some(&self.transcript)
        } else {
            self.conversations
                .iter()
                .find(|conversation| conversation.id == id)
                .and_then(|conversation| conversation.saved.as_ref())
                .map(|snapshot| &snapshot.transcript)
        }
    }

    fn apply_conversation_action(&mut self, ctx: &egui::Context, action: Action) {
        match action {
            Action::Select(id) => {
                self.switch_conversation(id);
                self.sidebar_open = false;
                self.focus_composer = true;
                // Reselecting a result also resets Find to the sidebar filter.
                // Local Find edits never rewrite that filter.
                let query = self.conversation_search.text.trim();
                if self.active_conversation == id && !query.is_empty() {
                    self.transcript.find_query(query);
                    self.focus_composer = false;
                }
            }
            Action::Rename(id) => {
                if let Some(conversation) = self.conversations.iter().find(|value| value.id == id) {
                    self.rename_conversation = Some((id, conversation.title.clone()));
                    ctx.memory_mut(|memory| {
                        memory.request_focus(egui::Id::unique("conversation-name"));
                    });
                }
            }
            Action::Copy(id) => {
                if let Some(transcript) = self.conversation_transcript(id) {
                    ctx.copy_text(transcript.plain_text());
                }
            }
            Action::Delete(id) => self.delete_conversation = Some(id),
        }
    }
}
