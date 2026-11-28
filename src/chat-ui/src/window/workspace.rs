//! Bounded conversation workspace. Locally identified interrupted replies are
//! retained as context without replaying their original requests.
use super::{Command, State, Transcript, client, history_search};

mod dialogs;
mod list;

const fn default_auto_continue() -> bool {
    true
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent response choices and transcript status are preserved for each conversation."
)]
pub(super) struct ConversationSnapshot {
    pub(super) transcript: Transcript,
    #[serde(deserialize_with = "super::persistence::deserialize_history")]
    pub(super) committed: Vec<client::Message>,
    #[serde(skip)]
    pub(super) context_history: Option<usize>,
    pub(super) input: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) draft_attachments: Vec<crate::attachment_import::Attachment>,
    pub(super) last_answer: String,
    pub(super) retained_draft: String,
    pub(super) answer_complete: bool,
    pub(super) extra_effort: bool,
    #[serde(default = "default_auto_continue")]
    pub(super) auto_continue: bool,
    #[serde(default)]
    pub(super) enable_thinking: bool,
    #[serde(default)]
    pub(super) streaming_read_aloud: bool,
    pub(super) output_budget: usize,
    #[serde(default, skip_serializing, rename = "abliteration_scale")]
    pub(super) _legacy_abliteration_scale: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    pub(super) daisyrose_enabled: bool,
    #[serde(default = "client::default_daisyrose_scale")]
    pub(super) daisyrose_scale: f32,
    #[serde(default)]
    pub(super) cognitive: client::CognitiveOptions,
    pub(super) status: String,
    #[serde(skip)]
    pub(super) status_detail: String,
    pub(super) warning: bool,
    #[serde(skip)]
    pub(super) stage: Option<&'static str>,
}

pub(super) struct CachedConversationJson {
    pub(super) title: String,
    pub(super) renamed: bool,
    pub(super) json: std::sync::Arc<[u8]>,
}

pub(in crate::window) struct DeletedConversation {
    conversation: Conversation,
    original_index: usize,
    created_empty_placeholder: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Conversation {
    pub(super) id: u64,
    pub(super) title: String,
    pub(super) renamed: bool,
    pub(super) saved: Option<ConversationSnapshot>,
    #[serde(skip)]
    pub(super) search: history_search::MatchCache,
    #[serde(skip)]
    pub(super) cached_json: std::sync::Mutex<Option<CachedConversationJson>>,
}

impl Conversation {
    pub(super) fn new(id: u64) -> Self {
        Self {
            id,
            title: "New conversation".to_owned(),
            renamed: false,
            saved: None,
            search: history_search::MatchCache::default(),
            cached_json: std::sync::Mutex::new(None),
        }
    }

    pub(super) fn invalidate_json_cache(&self) {
        if let Ok(mut slot) = self.cached_json.lock() {
            *slot = None;
        }
    }
}

impl State {
    pub(super) const fn daisyrose_default_enabled(&self) -> bool {
        self.daisyrose_available && self.daisyrose_default_scale_tenths > 0
    }

    pub(super) fn daisyrose_default_scale(&self) -> f32 {
        if self.daisyrose_default_enabled() {
            f32::from(self.daisyrose_default_scale_tenths) / 10.0
        } else {
            client::DEFAULT_DAISYROSE_SCALE
        }
    }

    pub(super) const fn effective_daisyrose_scale(&self) -> f32 {
        if self.daisyrose_available && self.daisyrose_enabled {
            self.daisyrose_scale
        } else {
            client::DEFAULT_DAISYROSE_SCALE
        }
    }

    pub(super) fn update_worker_conversation(&mut self, command: Command) {
        // Startup and generation consume history commands without applying them.
        // Coalesce navigation until the worker can accept the selected history.
        self.restore_history = true;
        if self.preview || self.model != super::ModelLifecycle::Ready {
            return;
        }
        if self.commands.send(command).is_ok() {
            self.restore_history = false;
        } else {
            // A lost worker must not prevent local conversation management.
            self.fail("The model worker stopped. Close and restart the app.");
        }
    }

    pub(super) fn sync_conversation_find(&mut self) {
        let query = self.conversation_search.text.trim();
        self.transcript.sync_find_query(query);
        if query.is_empty() {
            for conversation in &mut self.conversations {
                if let Some(saved) = conversation.saved.as_mut() {
                    saved.transcript.close_find();
                }
            }
        }
    }

    pub(super) fn clear_conversation_search(&mut self) {
        self.conversation_search.clear();
        self.sync_conversation_find();
    }

    pub(super) fn take_conversation(&mut self) -> ConversationSnapshot {
        self.input_revision = self.input_revision.wrapping_add(1);
        self.draft_attachments_revision = self.draft_attachments_revision.wrapping_add(1);
        ConversationSnapshot {
            transcript: std::mem::take(&mut self.transcript),
            committed: std::mem::take(&mut self.committed),
            context_history: self.context_history.take(),
            input: std::mem::take(&mut self.input),
            draft_attachments: std::mem::take(&mut self.draft_attachments),
            last_answer: std::mem::take(&mut self.last_answer),
            retained_draft: std::mem::take(&mut self.retained_draft),
            answer_complete: self.answer_complete,
            extra_effort: self.extra_effort,
            auto_continue: self.auto_continue,
            enable_thinking: self.enable_thinking,
            streaming_read_aloud: self.streaming_read_aloud,
            output_budget: self.output_budget,
            _legacy_abliteration_scale: None,
            daisyrose_enabled: self.daisyrose_enabled,
            daisyrose_scale: self.daisyrose_scale,
            cognitive: self.cognitive,
            status: if self.model.ready() {
                self.status.clone()
            } else {
                "Ready when you are.".to_owned()
            },
            warning: self.model.ready() && self.warning,
            status_detail: if self.model.ready() {
                self.status_detail.clone()
            } else {
                String::new()
            },
            stage: self.stage,
        }
    }

    pub(super) fn save_conversation(&mut self) {
        let snapshot = self.take_conversation();
        if let Some(conversation) = self
            .conversations
            .iter_mut()
            .find(|conversation| conversation.id == self.active_conversation)
        {
            conversation.saved = Some(snapshot);
            conversation.invalidate_json_cache();
        }
    }

    pub(super) fn clear_conversation(&mut self) {
        self.transcript.clear();
        self.committed.clear();
        self.context_history = None;
        self.input.clear();
        self.input_revision = self.input_revision.wrapping_add(1);
        self.set_draft_attachments(Vec::new());
        self.attachment_notice = None;
        super::clear_and_shrink_buffer(&mut self.last_answer);
        super::clear_and_shrink_buffer(&mut self.retained_draft);
        self.stage = None;
        self.answer_complete = false;
        self.streaming_svg_filter.reset();
        self.lightbox = None;
        self.extra_effort = false;
        self.auto_continue = true;
        self.enable_thinking = false;
        self.streaming_read_aloud = false;
        self.output_budget = 3;
        self.daisyrose_enabled = self.daisyrose_default_enabled();
        self.daisyrose_scale = self.daisyrose_default_scale();
        self.daisyrose_customized = false;
        self.cognitive = client::CognitiveOptions::default();
        self.place_composer_cursor(0);
        self.focus_composer = true;
    }

    pub(super) fn load_conversation(&mut self, snapshot: ConversationSnapshot) {
        self.transcript = snapshot.transcript;
        self.committed = snapshot.committed;
        self.context_history = snapshot.context_history;
        self.input = snapshot.input;
        self.input_revision = self.input_revision.wrapping_add(1);
        self.set_draft_attachments(snapshot.draft_attachments);
        self.attachment_notice = None;
        self.last_answer = snapshot.last_answer;
        self.retained_draft = snapshot.retained_draft;
        self.answer_complete = snapshot.answer_complete;
        self.extra_effort = snapshot.extra_effort;
        self.auto_continue = snapshot.auto_continue;
        self.enable_thinking = snapshot.enable_thinking;
        self.streaming_read_aloud = snapshot.streaming_read_aloud;
        self.output_budget = snapshot.output_budget;
        self.daisyrose_enabled = snapshot.daisyrose_enabled;
        self.daisyrose_scale = snapshot.daisyrose_scale;
        self.daisyrose_customized = true;
        self.cognitive = snapshot.cognitive;
        // Loading/failure status belongs to the model, not a saved transcript.
        if self.model.ready() {
            self.status = snapshot.status;
            self.status_detail = snapshot.status_detail;
            self.warning = snapshot.warning;
            if self.warning {
                self.alerts
                    .clear_dismissed(super::error_ui::AlertKind::Model);
            }
        }
        self.stage = snapshot.stage;
        self.place_composer_cursor(self.input.chars().count());
        self.focus_composer = true;
    }

    pub(super) fn active_or_routed_conversation(&self) -> u64 {
        self.routed_conversation.unwrap_or(self.active_conversation)
    }

    pub(super) fn touch_conversation(&mut self, id: u64) {
        if self
            .conversations
            .first()
            .is_some_and(|value| value.id == id)
        {
            self.recovery_dirty = true;
            return;
        }
        if let Some(index) = self
            .conversations
            .iter()
            .position(|conversation| conversation.id == id)
        {
            self.conversations[..=index].rotate_right(1);
            self.recovery_dirty = true;
        }
    }

    // Event routing and background queue dispatch swap only conversation content.
    // The actual selection, Find state in its transcript, and live composer
    // cursor/focus stay with the user.
    pub(super) fn with_conversation(&mut self, target: u64, action: impl FnOnce(&mut Self)) {
        if self.routed_conversation.is_some() || target == self.active_conversation {
            action(self);
            return;
        }
        let Some(index) = self
            .conversations
            .iter()
            .position(|value| value.id == target)
        else {
            return;
        };
        let Some(mut saved) = self.conversations[index].saved.take() else {
            return;
        };
        let cursor = (
            self.composer_cursor,
            self.pending_cursor,
            self.focus_composer,
        );
        self.swap_conversation(&mut saved);
        let previous = (
            self.model,
            self.status.clone(),
            self.warning,
            self.status_detail.clone(),
        );
        self.routed_conversation = Some(target);
        action(self);
        self.routed_conversation = None;
        // Late rejections, completion races and failed background dispatches can
        // update a hidden transcript without losing global stop/restart/error status.
        let changed = previous.0 != self.model
            || previous.1 != self.status
            || previous.2 != self.warning
            || previous.3 != self.status_detail;
        let global_status = (!self.model.ready() && changed).then(|| {
            (
                self.status.clone(),
                self.warning,
                self.status_detail.clone(),
            )
        });
        self.swap_conversation(&mut saved);
        if let Some(conversation) = self
            .conversations
            .iter_mut()
            .find(|value| value.id == target)
        {
            conversation.saved = Some(saved);
            conversation.invalidate_json_cache();
        }
        (
            self.composer_cursor,
            self.pending_cursor,
            self.focus_composer,
        ) = cursor;
        if let Some((status, warning, detail)) = global_status {
            self.status = status;
            self.status_detail = detail;
            self.warning = warning;
        }
    }

    pub(super) fn with_request_conversation(&mut self, action: impl FnOnce(&mut Self)) {
        if self.routed_conversation.is_some() {
            action(self);
            return;
        }
        let Some(owner) = self
            .request_conversation
            .filter(|id| *id != self.active_conversation)
        else {
            action(self);
            return;
        };
        if self
            .conversations
            .iter()
            .any(|value| value.id == owner && value.saved.is_some())
        {
            self.with_conversation(owner, action);
        } else if !self.conversations.iter().any(|value| value.id == owner) {
            action(self);
        }
    }

    const fn swap_conversation(&mut self, saved: &mut ConversationSnapshot) {
        std::mem::swap(&mut self.transcript, &mut saved.transcript);
        std::mem::swap(&mut self.committed, &mut saved.committed);
        std::mem::swap(&mut self.context_history, &mut saved.context_history);
        std::mem::swap(&mut self.input, &mut saved.input);
        self.input_revision = self.input_revision.wrapping_add(1);
        std::mem::swap(&mut self.draft_attachments, &mut saved.draft_attachments);
        self.draft_attachments_revision = self.draft_attachments_revision.wrapping_add(1);
        std::mem::swap(&mut self.last_answer, &mut saved.last_answer);
        std::mem::swap(&mut self.retained_draft, &mut saved.retained_draft);
        std::mem::swap(&mut self.answer_complete, &mut saved.answer_complete);
        std::mem::swap(&mut self.extra_effort, &mut saved.extra_effort);
        std::mem::swap(&mut self.auto_continue, &mut saved.auto_continue);
        std::mem::swap(&mut self.enable_thinking, &mut saved.enable_thinking);
        std::mem::swap(
            &mut self.streaming_read_aloud,
            &mut saved.streaming_read_aloud,
        );
        std::mem::swap(&mut self.output_budget, &mut saved.output_budget);
        std::mem::swap(&mut self.daisyrose_enabled, &mut saved.daisyrose_enabled);
        std::mem::swap(&mut self.daisyrose_scale, &mut saved.daisyrose_scale);
        std::mem::swap(&mut self.cognitive, &mut saved.cognitive);
        std::mem::swap(&mut self.status, &mut saved.status);
        std::mem::swap(&mut self.status_detail, &mut saved.status_detail);
        std::mem::swap(&mut self.warning, &mut saved.warning);
        std::mem::swap(&mut self.stage, &mut saved.stage);
    }

    pub(super) fn restore_selected_history(&mut self) {
        if self.restore_history && self.model == super::ModelLifecycle::Ready {
            self.update_worker_conversation(Command::RestoreChat(self.committed.clone()));
        }
    }

    pub(super) fn background_generation_title(&self) -> Option<&str> {
        let owner = self
            .request_conversation
            .filter(|_| self.model.busy())
            .filter(|&owner| owner != self.active_conversation)?;
        self.conversations
            .iter()
            .find(|value| value.id == owner)
            .map(|conversation| conversation.title.as_str())
    }

    pub(super) fn active_conversation_busy(&self) -> bool {
        self.model.busy() && self.background_generation_title().is_none()
    }

    pub(super) fn generation_status(&self) -> String {
        if let Some(title) = self.background_generation_title() {
            return format!("Responding in ‘{title}’. You can queue messages here while you wait.");
        }
        format!(
            "{} You can queue your next message while you wait.",
            self.status
        )
    }

    fn can_remove_conversation(&self, id: u64) -> bool {
        self.model.workspace_available()
            && !(self.model.busy() && self.request_conversation == Some(id))
    }

    pub(super) fn switch_conversation(&mut self, id: u64) {
        if !self.model.workspace_available() || id == self.active_conversation {
            return;
        }
        let Some(index) = self
            .conversations
            .iter()
            .position(|conversation| conversation.id == id)
        else {
            return;
        };
        let Some(saved) = self.conversations[index].saved.as_ref() else {
            return;
        };
        self.update_worker_conversation(Command::RestoreChat(saved.committed.clone()));
        self.stop_speech();
        self.save_conversation();
        self.conversation_notice = None;
        if let Some(snapshot) = self.conversations[index].saved.take() {
            self.conversations[index].invalidate_json_cache();
            self.active_conversation = id;
            self.load_conversation(snapshot);
            self.recovery_dirty = true;
            self.recovery_flush_now = true;
        }
    }

    pub(super) fn remove_conversation(&mut self, id: u64) {
        if !self.can_remove_conversation(id) {
            return;
        }
        let Some(original_index) = self
            .conversations
            .iter()
            .position(|conversation| conversation.id == id)
        else {
            return;
        };
        let mut created_empty_placeholder = false;
        if id == self.active_conversation {
            if let Some(next) = self
                .conversations
                .iter()
                .find(|conversation| conversation.id != id)
                .map(|conversation| conversation.id)
            {
                self.switch_conversation(next);
                if self.active_conversation == id {
                    return;
                }
            } else {
                self.update_worker_conversation(Command::NewChat);
                self.stop_speech();
                self.save_conversation();
                self.clear_conversation();
                self.active_conversation = self.next_conversation;
                self.next_conversation = self.next_conversation.saturating_add(1);
                self.conversations
                    .insert(0, Conversation::new(self.active_conversation));
                created_empty_placeholder = true;
                if self.model.ready() {
                    self.set_status("Ready for a new conversation.", false);
                }
            }
        }
        if let Some(current_index) = self
            .conversations
            .iter()
            .position(|conversation| conversation.id == id)
        {
            let removed = self.conversations.remove(current_index);
            self.recently_deleted = Some(DeletedConversation {
                conversation: removed,
                original_index,
                created_empty_placeholder,
            });
        }
        self.queued_messages
            .retain(|item| item.conversation_id != id);
        if self.queued_messages.is_empty() {
            self.queue_paused = false;
        }
        self.recovery_dirty = true;
        self.recovery_flush_now = true;
        self.conversation_notice = Some(super::ConversationNotice::auto_dismiss(
            "Conversation deleted.",
            super::CONVERSATION_DELETED_NOTICE_DURATION,
        ));
    }

    pub(in crate::window) fn undo_delete_conversation(&mut self) -> bool {
        if !self.model.workspace_available() {
            return false;
        }
        let Some(mut deleted) = self.recently_deleted.take() else {
            return false;
        };
        let id = deleted.conversation.id;
        if deleted.conversation.saved.is_none()
            || self
                .conversations
                .iter()
                .any(|conversation| conversation.id == id)
        {
            return false;
        }
        let sole_empty_placeholder = deleted.created_empty_placeholder
            && self.conversations.len() == 1
            && self.conversations[0].id == self.active_conversation
            && !self.conversations[0].renamed
            && self.conversations[0].title == "New conversation"
            && self.transcript.messages.is_empty()
            && self.committed.is_empty()
            && self.input.is_empty()
            && self.draft_attachments.is_empty()
            && self.last_answer.is_empty()
            && self.retained_draft.is_empty();
        if sole_empty_placeholder {
            let Some(snapshot) = deleted.conversation.saved.take() else {
                return false;
            };
            deleted.conversation.invalidate_json_cache();
            self.conversations.clear();
            self.conversations.push(deleted.conversation);
            self.active_conversation = id;
            self.next_conversation = self.next_conversation.max(id.saturating_add(1));
            self.update_worker_conversation(Command::RestoreChat(snapshot.committed.clone()));
            self.stop_speech();
            self.load_conversation(snapshot);
            self.conversation_notice = None;
            self.recovery_dirty = true;
            self.recovery_flush_now = true;
            return true;
        }
        let insert_at = deleted.original_index.min(self.conversations.len());
        self.conversations.insert(insert_at, deleted.conversation);
        self.next_conversation = self.next_conversation.max(id.saturating_add(1));
        self.switch_conversation(id);
        true
    }
}
