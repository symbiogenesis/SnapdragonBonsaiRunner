//! Explicit message iteration preserves the source conversation and exact history.
use super::{Command, Conversation, INPUT_LIMIT, ModelLifecycle, State, Transcript, egui};

pub(super) const CONTINUE_PROMPT: &str = "Continue your previous response from where it stopped, without repeating what you already wrote.";

#[derive(Clone, Copy, Debug)]
pub(super) enum MessageAction {
    Retry(u64),
    Edit(u64),
    Regenerate(u64),
    Branch(u64),
    Continue(u64),
}

#[derive(Clone, Copy, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Each explicit message action has an independent eligibility rule."
)]
pub(super) struct MessageOptions {
    pub(super) edit: bool,
    pub(super) regenerate: bool,
    pub(super) branch: bool,
    pub(super) continue_response: bool,
}

pub(super) struct EditMessage {
    pub(super) source_conversation: u64,
    pub(super) message: u64,
    pub(super) text: String,
    pub(super) attachments: Vec<crate::attachment_import::Attachment>,
    focus: bool,
    limit_notice: bool,
}

impl State {
    pub(super) fn message_options(&self, id: u64) -> MessageOptions {
        let available = !self
            .updates
            .as_ref()
            .is_some_and(super::updates::Updates::installing);
        let can_send = available && !self.preview && self.model == ModelLifecycle::Ready;
        let can_branch = available && self.model.workspace_available();
        self.transcript
            .messages
            .iter()
            .find(|message| message.id == id)
            .map_or_else(MessageOptions::default, |message| {
                message.iteration_options(&self.committed, can_send, can_branch)
            })
    }

    pub(super) fn message_action(&mut self, action: MessageAction) {
        match action {
            MessageAction::Retry(id) => self.retry_response(id),
            MessageAction::Edit(id) if self.message_options(id).edit => {
                let Some(transcript_message) = self
                    .transcript
                    .messages
                    .iter()
                    .find(|message| message.id == id && message.role == super::Role::User)
                else {
                    return;
                };
                let (text, attachments) =
                    if let Some(index) = self.transcript.committed_index(id, &self.committed) {
                        if transcript_message.attachments.is_empty() {
                            (self.committed[index].content.to_string(), Vec::new())
                        } else {
                            (
                                transcript_message.body.clone(),
                                transcript_message.attachments.clone(),
                            )
                        }
                    } else if transcript_message.retry_prefix(&self.committed).is_some() {
                        (
                            transcript_message.body.clone(),
                            transcript_message.attachments.clone(),
                        )
                    } else {
                        return;
                    };
                self.stop_speech();
                self.help = None;
                self.edit_message = Some(EditMessage {
                    source_conversation: self.active_conversation,
                    message: id,
                    text,
                    attachments,
                    focus: true,
                    limit_notice: false,
                });
                self.focus_composer = false;
            }
            MessageAction::Regenerate(id) if self.message_options(id).regenerate => {
                let Some(index) = self.transcript.committed_index(id, &self.committed) else {
                    return;
                };
                let prompt_index = index - 1;
                let (prompt, attachments) = self
                    .transcript
                    .messages
                    .iter()
                    .find(|message| message.committed_index(&self.committed) == Some(prompt_index))
                    .filter(|message| !message.attachments.is_empty())
                    .map_or_else(
                        || (self.committed[prompt_index].content.to_string(), Vec::new()),
                        |message| (message.body.clone(), message.attachments.clone()),
                    );
                if self.branch_history(prompt_index, "Regenerated") {
                    self.input = prompt;
                    self.set_draft_attachments(attachments);
                    self.submit();
                }
            }
            MessageAction::Branch(id) if self.message_options(id).branch => {
                let (end, prompt, attachments) =
                    if let Some(index) = self.transcript.committed_index(id, &self.committed) {
                        // A committed history always contains complete user/assistant
                        // pairs. A user-message branch places that prompt in the draft.
                        if self.committed[index].role == "user" {
                            let (body, attachments) = self
                                .transcript
                                .messages
                                .iter()
                                .find(|message| message.id == id)
                                .filter(|message| !message.attachments.is_empty())
                                .map_or_else(
                                    || (self.committed[index].content.to_string(), Vec::new()),
                                    |message| (message.body.clone(), message.attachments.clone()),
                                );
                            (index, body, attachments)
                        } else {
                            (index + 1, String::new(), Vec::new())
                        }
                    } else if let Some(message) = self
                        .transcript
                        .messages
                        .iter()
                        .find(|message| message.id == id && message.role == super::Role::User)
                        && let Some(prefix) = message.retry_prefix(&self.committed)
                    {
                        (prefix, message.body.clone(), message.attachments.clone())
                    } else {
                        return;
                    };
                if self.branch_history(end, "Branch") {
                    self.input = prompt;
                    self.set_draft_attachments(attachments);
                    self.place_composer_cursor(self.input.chars().count());
                }
            }
            MessageAction::Continue(id) if self.message_options(id).continue_response => {
                // Continue is an explicit ordinary followup, not an automatic
                // replay or modification of an already committed answer.
                let draft = std::mem::replace(&mut self.input, CONTINUE_PROMPT.to_owned());
                let draft_attachments = self.take_draft_attachments();
                let cursor = (
                    self.composer_cursor,
                    self.pending_cursor,
                    self.focus_composer,
                );
                self.restore_selected_history();
                self.submit();
                self.input = draft;
                self.set_draft_attachments(draft_attachments);
                (
                    self.composer_cursor,
                    self.pending_cursor,
                    self.focus_composer,
                ) = cursor;
                self.recovery_dirty = true;
            }
            _ => {}
        }
    }

    pub(super) fn try_auto_continue(&mut self) -> bool {
        if self.preview
            || self.model != ModelLifecycle::Ready
            || self
                .updates
                .as_ref()
                .is_some_and(super::updates::Updates::installing)
            || self.last_answer.trim().is_empty()
            || self.committed.len() >= 127
            || !self
                .transcript
                .messages
                .last()
                .is_some_and(|message| message.can_continue(&self.committed))
        {
            return false;
        }
        let Some(capacity) = self.context_tokens else {
            return false;
        };
        let max_completion_tokens = super::OUTPUT_TOKENS
            .get(self.output_budget)
            .copied()
            .unwrap_or(4096);
        let history_tokens = *self
            .context_history
            .get_or_insert_with(|| crate::context::history_tokens(&self.committed));
        let estimate = crate::context::Estimate::new(
            history_tokens,
            CONTINUE_PROMPT,
            max_completion_tokens,
            self.extra_effort,
            Some(capacity),
        );
        if estimate.overflow() > 0 {
            return false;
        }
        if !self.routing_request {
            self.restore_selected_history();
            if self.model != ModelLifecycle::Ready {
                return true;
            }
        }
        if self
            .commands
            .send(Command::Send {
                text: CONTINUE_PROMPT.to_owned(),
                images: std::sync::Arc::default(),
                max_completion_tokens,
                extra_effort: self.extra_effort,
                enable_thinking: self.enable_thinking,
                abliteration_scale: self.abliteration_scale,
            })
            .is_err()
        {
            self.fail("The model worker stopped. Close and restart the app.");
            return true;
        }
        self.last_answer.clear();
        self.retained_draft.clear();
        self.stage = None;
        self.answer_complete = false;
        if !self.streaming_active {
            self.streaming_svg_filter.reset();
            self.streaming_suppressed =
                self.routing_request || !self.speech_output || self.speaking;
        }
        let owner = self.active_or_routed_conversation();
        self.transcript
            .start_turn(CONTINUE_PROMPT.to_owned(), self.committed.len());
        if self.streaming_active {
            self.speaking_message = self.transcript.messages.last().map(|message| message.id);
        }
        self.request_conversation = Some(owner);
        self.touch_conversation(owner);
        self.model = ModelLifecycle::Generating;
        self.auto_continuing = true;
        self.set_status("Continuing response…", false);
        true
    }

    fn retry_response(&mut self, id: u64) {
        if self.preview
            || self
                .updates
                .as_ref()
                .is_some_and(super::updates::Updates::installing)
        {
            return;
        }
        let Some(message) = self
            .transcript
            .messages
            .iter()
            .find(|message| message.id == id)
        else {
            return;
        };
        let Some(prefix) = message.retry_prefix(&self.committed) else {
            return;
        };
        if self.model.reloadable() {
            self.retry();
            return;
        }
        if self.model != ModelLifecycle::Ready {
            return;
        }
        if message.role == super::Role::Assistant {
            self.message_action(MessageAction::Regenerate(id));
            return;
        }
        let prompt = message.body.clone();
        let prompt_attachments = message.attachments.clone();
        let draft = std::mem::replace(&mut self.input, prompt);
        let draft_attachments = self.replace_draft_attachments(prompt_attachments);
        let cursor = (
            self.composer_cursor,
            self.pending_cursor,
            self.focus_composer,
        );
        self.restore_selected_history();
        if self.model == ModelLifecycle::Ready && prefix == self.committed.len() {
            self.submit_turn(Some(id));
        }
        self.input = draft;
        self.set_draft_attachments(draft_attachments);
        (
            self.composer_cursor,
            self.pending_cursor,
            self.focus_composer,
        ) = cursor;
        self.recovery_dirty = true;
    }

    fn branch_history(&mut self, end: usize, label: &str) -> bool {
        if !self.model.workspace_available() || end > self.committed.len() || !end.is_multiple_of(2)
        {
            return false;
        }
        let Some(next) = self.next_conversation.checked_add(1) else {
            self.conversation_notice =
                Some("Cannot create another conversation in this workspace.");
            return false;
        };
        let history = self.committed[..end].to_vec();
        let mut transcript = Transcript::from_committed(&history);
        transcript.copy_interrupted_marks(&self.transcript, &history);
        let title = self
            .conversations
            .iter()
            .find(|conversation| conversation.id == self.active_conversation)
            .map_or_else(
                || "Conversation".to_owned(),
                |conversation| conversation.title.chars().take(44).collect::<String>(),
            );
        let settings = (
            self.extra_effort,
            self.enable_thinking,
            self.streaming_read_aloud,
            self.output_budget,
            self.abliteration_scale,
        );
        self.stop_speech();
        self.save_conversation();
        self.clear_conversation();
        self.active_conversation = self.next_conversation;
        self.next_conversation = next;
        let mut branch = Conversation::new(self.active_conversation);
        branch.title = format!("{title} · {label}");
        branch.renamed = true;
        self.conversations.insert(0, branch);
        self.clear_conversation_search();
        self.transcript = transcript;
        self.committed = history;
        (
            self.extra_effort,
            self.enable_thinking,
            self.streaming_read_aloud,
            self.output_budget,
            self.abliteration_scale,
        ) = settings;
        if let Some(answer) = self.committed.last() {
            self.last_answer = answer.content.to_string();
            self.answer_complete = true;
        }
        self.update_worker_conversation(Command::RestoreChat(self.committed.clone()));
        if self.model == ModelLifecycle::Ready {
            self.set_status("Ready in your new branch.", false);
        }
        self.conversation_notice =
            Some("Created a branch. The original conversation is unchanged.");
        self.recovery_dirty = true;
        true
    }

    fn resend_edited_message(&mut self, edit: EditMessage) {
        if edit.source_conversation != self.active_conversation
            || !self.message_options(edit.message).edit
            || (edit.text.trim().is_empty() && edit.attachments.is_empty())
            || edit.text.chars().count() > INPUT_LIMIT
        {
            return;
        }
        let prefix = if let Some(index) = self
            .transcript
            .committed_index(edit.message, &self.committed)
        {
            index
        } else if let Some(message) = self
            .transcript
            .messages
            .iter()
            .find(|message| message.id == edit.message && message.role == super::Role::User)
            && let Some(prefix) = message.retry_prefix(&self.committed)
        {
            prefix
        } else {
            return;
        };
        if !self.vision_available
            && (edit
                .attachments
                .iter()
                .any(crate::attachment_import::Attachment::is_media)
                || self.history_has_media_up_to(prefix))
        {
            self.set_status(
                "Images and videos require the vision projector (mmproj). Remove media attachments or relaunch without --no-mmproj.",
                true,
            );
            return;
        }
        if self.branch_history(prefix, "Edited") {
            self.input = edit.text;
            self.set_draft_attachments(edit.attachments);
            self.submit();
        }
    }

    pub(super) fn iteration_dialog(&mut self, ctx: &egui::Context) {
        let Some(mut edit) = self.edit_message.take() else {
            return;
        };
        let mut confirmed = false;
        let mut cancelled = false;
        let enabled = edit.source_conversation == self.active_conversation
            && self.message_options(edit.message).edit;
        let modal = egui::Modal::new(egui::Id::unique("edit-and-resend")).show(ctx, |ui| {
            ui.set_width((ctx.content_rect().width() - 80.0).clamp(280.0, 560.0));
            ui.heading("Edit and resend");
            ui.label("Creates a new conversation from this prompt. The original conversation and its draft stay unchanged.");
            ui.add_space(8.0);
            let label = ui.label("Edited message");
            egui::ScrollArea::vertical()
                .max_height((ctx.content_rect().height() - 240.0).max(100.0))
                .show(ui, |ui| {
                    let id = ui.make_persistent_id("edited-message");
                    let mut buffer = super::draft_buffer::Buffer::new(
                        &mut edit.text, ui.ctx(), id, INPUT_LIMIT,
                    );
                    let mut output = egui::TextEdit::multiline(&mut buffer)
                            .id(id)
                            .desired_width(f32::INFINITY)
                            .desired_rows(6)
                            .show(ui);
                    if buffer.finish(&mut output) {
                        edit.limit_notice = true;
                    } else if output.response.response.changed() {
                        edit.limit_notice = false;
                    }
                    let response = output.response.response.labelled_by(label.id);
                    if edit.focus {
                        response.request_focus();
                        edit.focus = false;
                    }
                });
            if edit.limit_notice {
                ui.label(super::draft_buffer::LIMIT_MESSAGE);
            }
            if !edit.attachments.is_empty() {
                ui.label(
                    super::RichText::new(format!(
                        "{} attached context item(s) preserved.",
                        edit.attachments.len()
                    ))
                    .small()
                    .weak(),
                );
            }
            if !enabled {
                ui.label("Return to the source conversation and wait until the model is ready to resend.");
            }
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                confirmed = ui
                    .add_enabled(
                        enabled && (!edit.text.trim().is_empty() || !edit.attachments.is_empty()),
                        egui::Button::new("Create branch and send"),
                    )
                    .clicked();
                cancelled = ui.button("Cancel").clicked();
            });
        });
        if confirmed {
            self.resend_edited_message(edit);
        } else if !cancelled && !modal.should_close() {
            self.edit_message = Some(edit);
        } else {
            self.focus_composer = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{client, window::mpsc};

    fn fixture() -> (State, mpsc::Receiver<Command>) {
        let (commands, incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::create(commands, incoming_events, None, None);
        state.preview = false;
        state.model = ModelLifecycle::Ready;
        state.committed = [
            ("user", "Same question"),
            ("assistant", "First answer"),
            ("user", "Same question"),
            ("assistant", "Second answer"),
        ]
        .into_iter()
        .map(|(role, content)| client::Message {
            role,
            content: content.into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        })
        .collect();
        state.transcript = Transcript::from_committed(&state.committed);
        state.input = "Unsent source draft 🌱".into();
        state.enable_thinking = true;
        state.extra_effort = true;
        state.output_budget = 1;
        state.abliteration_scale = 1.4;
        (state, incoming)
    }

    #[test]
    fn retry_response_reuses_failed_prompt_preserves_draft_and_never_replays_on_ready() {
        let (mut state, commands) = fixture();
        let source = state.active_conversation;
        state.input = "Same question".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        let prompt = state.transcript.messages[4].id;
        state.input = "Same question".into();
        state.place_composer_cursor(4);
        state.model_event(client::Event::Stopping("Same question".into()));
        let cursor = (
            state.composer_cursor,
            state.pending_cursor,
            state.focus_composer,
        );
        state.message_action(MessageAction::Retry(prompt));
        assert!(commands.try_recv().is_err());
        state.model_event(client::Event::Stopped);
        assert_eq!(state.model, ModelLifecycle::Ready);
        assert!(!state.warning);
        state.message_action(MessageAction::Retry(prompt));
        assert!(
            matches!(commands.try_recv(), Ok(Command::Send { text, .. }) if text == "Same question")
        );
        state.message_action(MessageAction::Retry(prompt));
        assert!(commands.try_recv().is_err());
        assert_eq!(state.active_conversation, source);
        assert_eq!(state.transcript.messages.len(), 6);
        assert_eq!(state.transcript.messages[4].id, prompt);
        assert_eq!(state.committed.len(), 4);
        assert_eq!(state.input, "Same question");
        // Readiness and Stop may focus the composer; retry preserves the cursor.
        assert_eq!(
            (state.composer_cursor, state.pending_cursor),
            (cursor.0, cursor.1)
        );
        // Genuine backend teardown still requires Reload model before Retry response.
        state.model_event(client::Event::Rejected {
            message: "The response did not complete.".into(),
            text: "Same question".into(),
            retry_required: true,
        });
        state.model_event(client::Event::Retryable("Select Retry model.".into()));
        state.message_action(MessageAction::Retry(prompt));
        assert!(matches!(commands.try_recv(), Ok(Command::Retry)));
        assert!(commands.try_recv().is_err());
        state.model_event(client::Event::Starting);
        state.model_event(client::Event::Ready);
        assert!(commands.try_recv().is_err());
        state.input = "A newer draft 🌱".into();
        state.message_action(MessageAction::Retry(prompt));
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        assert_eq!(state.transcript.messages[4].id, prompt);
        assert_eq!(state.transcript.messages.len(), 6);
        assert_eq!(state.input, "A newer draft 🌱");
    }

    #[test]
    fn retry_response_branches_partial_answer_from_original_prefix_and_keeps_source() {
        let (mut state, commands) = fixture();
        let source = state.active_conversation;
        state.input = "Same question".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        state.model_event(client::Event::Text("Partial answer".into()));
        state.model_event(client::Event::Stopping("Same question".into()));
        state.model_event(client::Event::Stopped);
        assert_eq!(state.model, ModelLifecycle::Ready);
        assert!(
            matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.len() == 6)
        );
        let answer = state.transcript.messages[5].id;
        state.input = "Keep this draft".into();
        state.message_action(MessageAction::Retry(answer));
        assert_ne!(state.active_conversation, source);
        assert!(
            matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.len() == 4 && history[3].content.as_ref() == "Second answer")
        );
        assert!(
            matches!(commands.try_recv(), Ok(Command::Send { text, .. }) if text == "Same question")
        );
        assert!(commands.try_recv().is_err());
        assert_eq!(state.committed.len(), 4);
        assert_eq!(state.conversations[0].id, state.active_conversation);
        assert!(state.conversations[1].saved.as_ref().is_some_and(|saved| {
            saved.input == "Keep this draft"
                && saved.committed[5].content.as_ref() == "Partial answer"
        }));
    }

    #[test]
    fn iteration_regenerates_selected_duplicate_turn_in_a_branch_with_exact_prefix() {
        let (mut state, commands) = fixture();
        let source = state.active_conversation;
        let answer = state.transcript.messages[3].id;
        state.message_action(MessageAction::Regenerate(answer));
        assert_ne!(state.active_conversation, source);
        assert_eq!(state.committed.len(), 2);
        assert_eq!(state.committed[1].content.as_ref(), "First answer");
        assert!(
            matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.len() == 2)
        );
        assert!(
            matches!(commands.try_recv(), Ok(Command::Send { text, enable_thinking: true, extra_effort: true, max_completion_tokens: 1024, abliteration_scale, .. }) if text == "Same question" && (abliteration_scale - 1.4).abs() < f32::EPSILON)
        );
        assert_eq!(state.conversations[0].id, state.active_conversation);
        let original = state.conversations[1].saved.as_ref();
        assert!(original.is_some_and(
            |saved| saved.committed.len() == 4 && saved.input == "Unsent source draft 🌱"
        ));
        assert!(state.model.busy());
    }

    #[test]
    fn iteration_edit_cancellation_is_inert_and_confirmation_preserves_original()
    -> std::io::Result<()> {
        let (mut state, commands) = fixture();
        let prompt = state.transcript.messages[0].id;
        state.message_action(MessageAction::Edit(prompt));
        let mut edit = state
            .edit_message
            .take()
            .ok_or_else(|| std::io::Error::other("Edit must open a review dialog"))?;
        assert_eq!(edit.text, "Same question");
        assert!(commands.try_recv().is_err());
        assert_eq!(state.conversations.len(), 1);
        assert_eq!(state.input, "Unsent source draft 🌱");
        edit.text = "Revised question".into();
        state.resend_edited_message(edit);
        assert!(
            matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.is_empty())
        );
        assert!(
            matches!(commands.try_recv(), Ok(Command::Send { text, .. }) if text == "Revised question")
        );
        assert!(state.committed.is_empty());
        assert_eq!(state.conversations[0].id, state.active_conversation);
        assert!(
            state.conversations[1]
                .saved
                .as_ref()
                .is_some_and(|saved| saved.committed.len() == 4)
        );
        Ok(())
    }

    #[test]
    fn iteration_branches_a_user_prompt_without_committing_or_sending_it() {
        let (mut state, commands) = fixture();
        let prompt = state.transcript.messages[2].id;
        state.message_action(MessageAction::Branch(prompt));
        assert_eq!(state.committed.len(), 2);
        assert_eq!(state.input, "Same question");
        assert_eq!(state.transcript.messages.len(), 2);
        assert!(
            matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.len() == 2)
        );
        assert!(commands.try_recv().is_err());
        assert_eq!(state.model, ModelLifecycle::Ready);
    }

    #[test]
    fn iteration_continue_preserves_draft_and_only_targets_the_last_committed_answer() {
        let (mut state, commands) = fixture();
        state.place_composer_cursor(7);
        let cursor = (state.composer_cursor, state.pending_cursor);
        let earlier = state.transcript.messages[1].id;
        let last = state.transcript.messages[3].id;
        state.message_action(MessageAction::Continue(earlier));
        assert!(commands.try_recv().is_err());
        state.message_action(MessageAction::Continue(last));
        assert!(
            matches!(commands.try_recv(), Ok(Command::Send { text, .. }) if text == CONTINUE_PROMPT)
        );
        assert_eq!(state.input, "Unsent source draft 🌱");
        assert_eq!((state.composer_cursor, state.pending_cursor), cursor);
        assert_eq!(state.committed.len(), 4);
        assert_eq!(state.transcript.messages[4].body, CONTINUE_PROMPT);
    }

    #[test]
    fn iteration_respects_busy_request_ownership() {
        let (mut state, commands) = fixture();
        let answer = state.transcript.messages[3].id;
        state.model = ModelLifecycle::Generating;
        state.request_conversation = Some(state.active_conversation);
        let owner = state.active_conversation;
        state.message_action(MessageAction::Regenerate(answer));
        state.message_action(MessageAction::Continue(answer));
        assert!(commands.try_recv().is_err());
        state.message_action(MessageAction::Branch(answer));
        assert_eq!(state.request_conversation, Some(owner));
        assert_ne!(state.active_conversation, owner);
        assert!(state.restore_history);
        assert!(commands.try_recv().is_err());
    }

    #[test]
    fn conversation_history_iteration_has_no_count_cap_and_reveals_new_branches()
    -> std::io::Result<()> {
        for action in 0..3 {
            let (mut state, commands) = fixture();
            let source = state.active_conversation;
            for _ in 1..32 {
                state.new_chat();
            }
            state.switch_conversation(source);
            while commands.try_recv().is_ok() {}
            state.conversation_search.text = "An unrelated search".into();
            let answer = state.transcript.messages[3].id;
            let prompt = state.transcript.messages[2].id;
            match action {
                0 => state.message_action(MessageAction::Branch(answer)),
                1 => state.message_action(MessageAction::Regenerate(answer)),
                _ => {
                    state.message_action(MessageAction::Edit(prompt));
                    let mut edit = state.edit_message.take().ok_or_else(|| {
                        std::io::Error::other("Edit must remain available after the former limit.")
                    })?;
                    edit.text = "A revised question".into();
                    state.resend_edited_message(edit);
                }
            }
            assert_eq!(state.conversations.len(), 33);
            assert_ne!(state.active_conversation, source);
            assert_eq!(state.conversation_search.text, "");
            let original = state
                .conversations
                .iter()
                .find(|item| item.id == source)
                .and_then(|item| item.saved.as_ref());
            assert_eq!(
                original.map(|saved| saved.input.as_str()),
                Some("Unsent source draft 🌱")
            );
        }
        Ok(())
    }

    #[test]
    fn interruption_continue_and_branch_use_partial_reply_and_preserve_draft() -> std::io::Result<()>
    {
        let (mut state, commands) = fixture();
        state.input = "Interrupted prompt".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        state.model_event(client::Event::Text("Partial answer".into()));
        state.input = "New draft".into();
        state.model_event(client::Event::Stopping("Interrupted prompt".into()));
        let answer = state
            .transcript
            .messages
            .last()
            .ok_or_else(|| std::io::Error::other("partial answer missing"))?
            .id;
        assert!(
            state
                .committed
                .last()
                .is_some_and(|message| message.content.as_ref() == "Partial answer")
        );
        assert!(!state.message_options(answer).continue_response);
        state.model_event(client::Event::Stopped);
        assert_eq!(state.model, ModelLifecycle::Ready);
        assert!(state.message_options(answer).branch);
        assert!(
            matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.last().is_some_and(|message| message.content.as_ref() == "Partial answer"))
        );
        assert!(state.message_options(answer).continue_response);
        state.message_action(MessageAction::Branch(answer));
        assert!(
            matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.last().is_some_and(|message| message.content.as_ref() == "Partial answer"))
        );
        assert_eq!(state.conversations[0].id, state.active_conversation);
        assert!(
            state.conversations[1]
                .saved
                .as_ref()
                .is_some_and(|saved| saved.input == "New draft")
        );
        let branch_answer = state
            .transcript
            .messages
            .last()
            .ok_or_else(|| std::io::Error::other("branch answer missing"))?
            .id;
        state.message_action(MessageAction::Continue(branch_answer));
        assert!(
            matches!(commands.try_recv(), Ok(Command::Send { text, .. }) if text == CONTINUE_PROMPT)
        );
        assert!(
            state
                .committed
                .last()
                .is_some_and(|message| message.content.as_ref() == "Partial answer")
        );
        Ok(())
    }

    #[test]
    fn token_limit_marks_only_truncated_replies_and_preserves_marks_across_branches()
    -> std::io::Result<()> {
        let entry = |role, content: &str| client::Message {
            role,
            content: content.into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        };
        let (mut state, commands) = fixture();
        state.input = "Long explanation".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        state.model_event(client::Event::Text("Truncated part one".into()));
        let mut committed = state.committed.clone();
        committed.push(entry("user", "Long explanation"));
        committed.push(entry("assistant", "Truncated part one"));
        state.model_event(client::Event::Committed(committed));
        state.model_event(client::Event::Complete(client::Completion::TokenLimit));
        let capped = state
            .transcript
            .messages
            .last()
            .ok_or_else(|| std::io::Error::other("capped answer missing"))?;
        assert!(capped.token_limit);
        assert!(capped.can_continue(&state.committed));
        assert_eq!(capped.retry_prefix(&state.committed), Some(4));
        let capped_id = capped.id;

        state.message_action(MessageAction::Branch(capped_id));
        assert!(matches!(commands.try_recv(), Ok(Command::RestoreChat(_))));
        let branched = state
            .transcript
            .messages
            .last()
            .ok_or_else(|| std::io::Error::other("branched capped answer missing"))?;
        assert!(branched.token_limit);
        assert!(branched.can_continue(&state.committed));

        // Extra-effort Review hitting TokenLimit while retaining a complete draft
        // must not mark the retained draft as truncated.
        state.extra_effort = true;
        state.input = "Careful answer".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        state.model_event(client::Event::Draft("Complete draft".into()));
        state.model_event(client::Event::RetainedAnswer("Complete draft".into()));
        let mut retained = state.committed.clone();
        retained.push(entry("user", "Careful answer"));
        retained.push(entry("assistant", "Complete draft"));
        state.model_event(client::Event::Committed(retained));
        state.model_event(client::Event::Complete(client::Completion::TokenLimit));
        let final_answer = state
            .transcript
            .messages
            .last()
            .ok_or_else(|| std::io::Error::other("retained draft missing"))?;
        assert!(!final_answer.token_limit);
        assert!(!final_answer.can_continue(&state.committed));
        assert!(!state.warning);
        Ok(())
    }

    #[test]
    fn token_limit_auto_continues_within_context_capacity_without_alerting_and_stops_when_full()
    -> std::io::Result<()> {
        let entry = |role, content: &str| client::Message {
            role,
            content: content.into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        };
        let (mut state, commands) = fixture();
        state.context_tokens = Some(131_072);
        state.extra_effort = false;
        state.input = "Long walkthrough".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));

        state.input = "Keep my next draft 🌱".into();
        state.place_composer_cursor(6);
        let cursor = (state.composer_cursor, state.pending_cursor);

        state.model_event(client::Event::Text("Part one.".into()));
        let mut committed = state.committed.clone();
        committed.push(entry("user", "Long walkthrough"));
        committed.push(entry("assistant", "Part one."));
        state.model_event(client::Event::Committed(committed));
        state.model_event(client::Event::Complete(client::Completion::TokenLimit));

        assert_eq!(state.model, ModelLifecycle::Generating);
        assert!(state.auto_continuing);
        assert!(!state.warning);
        assert!(state.active_alerts().is_empty());
        assert_eq!(state.input, "Keep my next draft 🌱");
        assert_eq!((state.composer_cursor, state.pending_cursor), cursor);
        assert!(matches!(
            commands.try_recv(),
            Ok(Command::Send {
                text,
                enable_thinking: true,
                extra_effort: false,
                max_completion_tokens: 1024,
                ..
            }) if text == CONTINUE_PROMPT
        ));
        assert!(commands.try_recv().is_err());

        // Shrink reported capacity so the next output reservation no longer fits.
        state.context_tokens = Some(1_024);
        state.model_event(client::Event::Text("Part two.".into()));
        let mut second = state.committed.clone();
        second.push(entry("user", CONTINUE_PROMPT));
        second.push(entry("assistant", "Part two."));
        state.model_event(client::Event::Committed(second));
        state.model_event(client::Event::Complete(client::Completion::TokenLimit));

        assert_eq!(state.model, ModelLifecycle::Ready);
        assert!(!state.auto_continuing);
        assert!(!state.warning);
        assert!(state.active_alerts().is_empty());
        assert!(commands.try_recv().is_err());
        let capped = state
            .transcript
            .messages
            .last()
            .ok_or_else(|| std::io::Error::other("second capped answer missing"))?;
        assert!(capped.token_limit);
        assert!(capped.can_continue(&state.committed));
        assert_eq!(state.input, "Keep my next draft 🌱");
        Ok(())
    }

    #[test]
    fn auto_continue_rejection_and_early_stop_discard_synthetic_prompt_and_keep_empty_composer()
    -> std::io::Result<()> {
        let entry = |role, content: &str| client::Message {
            role,
            content: content.into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        };
        let (mut state, commands) = fixture();
        state.context_tokens = Some(32_768);
        state.extra_effort = false;
        state.input = "Write a guide".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        assert!(state.input.is_empty());

        state.model_event(client::Event::Text("Part one.".into()));
        let mut committed = state.committed.clone();
        committed.push(entry("user", "Write a guide"));
        committed.push(entry("assistant", "Part one."));
        state.model_event(client::Event::Committed(committed));
        state.model_event(client::Event::Complete(client::Completion::TokenLimit));
        assert!(matches!(
            commands.try_recv(),
            Ok(Command::Send { text, .. }) if text == CONTINUE_PROMPT
        ));
        assert!(state.auto_continuing);

        // Exact broker count rejection on the automatic continuation cleans up silently.
        state.model_event(client::Event::Rejected {
            message: "Context limit reached.".into(),
            text: CONTINUE_PROMPT.into(),
            retry_required: false,
        });
        assert_eq!(state.model, ModelLifecycle::Ready);
        assert!(!state.warning);
        assert!(state.active_alerts().is_empty());
        assert!(state.input.is_empty());
        assert_eq!(state.last_answer, "Part one.");
        let last = state
            .transcript
            .messages
            .last()
            .ok_or_else(|| std::io::Error::other("capped answer missing"))?;
        assert_eq!(last.body, "Part one.");
        assert!(last.can_continue(&state.committed));

        // Thinking-only TokenLimit (no visible text) does not loop automatic continuations.
        state.input = "Hard puzzle".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        state.model_event(client::Event::Reasoning("All budget spent thinking".into()));
        let mut thinking_only = state.committed.clone();
        thinking_only.push(entry("user", "Hard puzzle"));
        thinking_only.push(client::Message {
            role: "assistant",
            content: "".into(),
            images: std::sync::Arc::default(),
            reasoning_content: Some("All budget spent thinking".into()),
        });
        state.model_event(client::Event::Committed(thinking_only));
        state.model_event(client::Event::Complete(client::Completion::TokenLimit));
        assert_eq!(state.model, ModelLifecycle::Ready);
        assert!(!state.warning);
        assert!(commands.try_recv().is_err());

        // Stopping an automatic continuation before prose arrives also discards the synthetic prompt.
        state.input = "Another long guide".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        state.model_event(client::Event::Text("First section.".into()));
        let mut third = state.committed.clone();
        third.push(entry("user", "Another long guide"));
        third.push(entry("assistant", "First section."));
        state.model_event(client::Event::Committed(third));
        state.model_event(client::Event::Complete(client::Completion::TokenLimit));
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        state.stop_generation();
        assert!(matches!(commands.try_recv(), Ok(Command::Stop)));
        state.model_event(client::Event::Stopping(CONTINUE_PROMPT.into()));
        state.model_event(client::Event::Stopped);
        assert_eq!(state.model, ModelLifecycle::Ready);
        assert!(!state.warning);
        assert!(state.input.is_empty());
        let stopped_last = state
            .transcript
            .messages
            .last()
            .ok_or_else(|| std::io::Error::other("first section missing"))?;
        assert_eq!(stopped_last.body, "First section.");
        assert!(stopped_last.can_continue(&state.committed));
        Ok(())
    }

    #[test]
    fn background_token_limit_auto_continues_owner_and_defers_selected_history_restore() {
        let (mut state, commands) = fixture();
        state.context_tokens = Some(65_536);
        state.extra_effort = false;
        let owner = state.active_conversation;
        state.input = "Background walkthrough".into();
        state.submit();
        assert!(matches!(commands.try_recv(), Ok(Command::Send { .. })));
        state.input = "Owner draft".into();

        state.new_chat();
        let selected = state.active_conversation;
        state.input = "Selected draft 🌱".into();
        assert!(state.restore_history);

        state.model_event(client::Event::Text("Background part one.".into()));
        let mut owner_history = state
            .conversations
            .iter()
            .find(|conversation| conversation.id == owner)
            .and_then(|conversation| conversation.saved.as_ref())
            .map(|saved| saved.committed.clone())
            .unwrap_or_default();
        owner_history.push(client::Message {
            role: "user",
            content: "Background walkthrough".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        });
        owner_history.push(client::Message {
            role: "assistant",
            content: "Background part one.".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        });
        state.model_event(client::Event::Committed(owner_history.clone()));
        state.model_event(client::Event::Complete(client::Completion::TokenLimit));

        assert_eq!(state.active_conversation, selected);
        assert_eq!(state.request_conversation, Some(owner));
        assert_eq!(state.model, ModelLifecycle::Generating);
        assert!(state.restore_history);
        assert!(!state.warning);
        assert_eq!(state.input, "Selected draft 🌱");
        assert!(matches!(
            commands.try_recv(),
            Ok(Command::Send { text, .. }) if text == CONTINUE_PROMPT
        ));
        assert!(commands.try_recv().is_err());

        state.model_event(client::Event::Text("Background part two.".into()));
        owner_history.push(client::Message {
            role: "user",
            content: CONTINUE_PROMPT.into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        });
        owner_history.push(client::Message {
            role: "assistant",
            content: "Background part two.".into(),
            images: std::sync::Arc::default(),
            reasoning_content: None,
        });
        state.model_event(client::Event::Committed(owner_history));
        state.model_event(client::Event::Complete(client::Completion::Finished));

        assert_eq!(state.model, ModelLifecycle::Ready);
        assert!(!state.restore_history);
        assert!(
            matches!(commands.try_recv(), Ok(Command::RestoreChat(history)) if history.is_empty())
        );
        assert!(commands.try_recv().is_err());
        state.switch_conversation(owner);
        assert_eq!(state.input, "Owner draft");
        assert_eq!(state.committed.len(), 8);
        assert_eq!(state.last_answer, "Background part two.");
    }
}
