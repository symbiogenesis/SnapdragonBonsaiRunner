//! Versioned workspace recovery. Never restores an in-flight request.
use super::{
    Conversation, Duration, Instant, MAX_DIAGNOSTIC_CHARS, OUTPUT_TOKENS, RichText, State,
    Transcript, VOICE_LIMIT, client, egui, io, workspace::ConversationSnapshot,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SalvageReport {
    pub(super) recovered_conversations: usize,
    pub(super) dropped_conversations: usize,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedWorkspace {
    version: u32,
    conversations: Vec<Conversation>,
    active_conversation: u64,
    next_conversation: u64,
    active: ConversationSnapshot,
    voice_profile: String,
    #[serde(default)]
    memory: crate::context_cards::MemoryStore,
}

pub(super) fn deserialize_history<'de, D>(deserializer: D) -> Result<Vec<client::Message>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_bounded_sequence(deserializer, 128)
}

pub(super) fn deserialize_transcript_messages<'de, D>(
    deserializer: D,
) -> Result<Vec<super::transcript::Message>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_bounded_sequence(deserializer, super::transcript::MAX_MESSAGES)
}

// Reject an extra element before deserializing its payload. Post-parse checks
// alone let a small file containing many empty objects allocate a large graph
// of messages before discovering that it cannot be restored. Conversation count
// is unrestricted; the complete encoded workspace has a separate byte budget.
fn deserialize_bounded_sequence<'de, D, T>(
    deserializer: D,
    limit: usize,
) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    struct BoundedSequence<T> {
        limit: usize,
        marker: std::marker::PhantomData<T>,
    }

    struct ExcessItem;

    impl<'de> serde::Deserialize<'de> for ExcessItem {
        fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            Err(serde::de::Error::custom(
                "Saved collection exceeds its item limit.",
            ))
        }
    }

    impl<'de, T: serde::Deserialize<'de>> serde::de::Visitor<'de> for BoundedSequence<T> {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "a sequence of at most {} items", self.limit)
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut items = Vec::new();
            while items.len() < self.limit {
                let Some(item) = sequence.next_element()? else {
                    return Ok(items);
                };
                items.push(item);
            }
            let _: Option<ExcessItem> = sequence.next_element()?;
            Ok(items)
        }
    }

    deserializer.deserialize_seq(BoundedSequence {
        limit,
        marker: std::marker::PhantomData,
    })
}

fn attachments_empty(attachments: &[crate::attachment_import::Attachment]) -> bool {
    attachments.is_empty()
}

#[derive(serde::Serialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Recovery serializes independent response choices and transcript status."
)]
struct ActiveView<'a> {
    transcript: &'a Transcript,
    committed: &'a [client::Message],
    input: &'a str,
    #[serde(skip_serializing_if = "attachments_empty")]
    draft_attachments: &'a [crate::attachment_import::Attachment],
    last_answer: &'a str,
    retained_draft: &'a str,
    answer_complete: bool,
    extra_effort: bool,
    auto_continue: bool,
    enable_thinking: bool,
    streaming_read_aloud: bool,
    output_budget: usize,
    daisyrose_enabled: bool,
    daisyrose_scale: f32,
    cognitive: client::CognitiveOptions,
    status: &'a str,
    warning: bool,
}

impl State {
    pub(super) fn attach_recovery(&mut self, opened: io::Result<crate::recovery::Opened>) {
        self.alerts
            .clear_dismissed(super::error_ui::AlertKind::Recovery);
        match opened {
            Ok(opened) => {
                self.recovery = Some(opened.store);
                if let Some(bytes) = opened.saved {
                    if self.restore_workspace(&bytes).is_ok() {
                        self.save_conversations = true;
                        self.recovery_quarantine_primary = false;
                        "Conversations restored. Interrupted replies remain available as context."
                            .clone_into(&mut self.recovery_notice);
                    } else {
                        self.recovery_quarantine_primary = true;
                        self.recover_corrupt_primary(Some(&bytes), false);
                    }
                } else if opened.load_failed {
                    self.recovery_quarantine_primary = true;
                    self.recover_corrupt_primary(None, true);
                } else if opened.saving_disabled {
                    self.recovery_erase_pending = true;
                    "Saving is off. Enable it to keep conversations after closing this window."
                        .clone_into(&mut self.recovery_notice);
                } else {
                    self.save_conversations = true;
                    self.recovery_dirty = true;
                    "Conversations save automatically on this device with Windows user encryption."
                        .clone_into(&mut self.recovery_notice);
                }
            }
            Err(_) => {
                "Could not open conversation saving. Another window may be using this data folder, or the folder is not writable.".clone_into(&mut self.recovery_notice);
            }
        }
    }

    fn recover_corrupt_primary(&mut self, primary: Option<&[u8]>, load_failed: bool) {
        let primary_salvage = primary.and_then(|bytes| self.salvage_workspace_data(bytes).ok());
        if let Some((saved, report)) = primary_salvage
            && report.dropped_conversations == 0
        {
            self.apply_saved_workspace(saved);
            self.save_conversations = true;
            self.recovery_dirty = true;
            self.recovery_flush_now = true;
            "Recovered conversations from a modified or newer save format."
                .clone_into(&mut self.recovery_notice);
            return;
        }
        let backup_bytes = self
            .recovery
            .as_ref()
            .and_then(|recovery| recovery.read_backup().ok().flatten());
        if let Some(backup) = backup_bytes.as_deref()
            && self.restore_workspace(backup).is_ok()
        {
            self.save_conversations = true;
            self.recovery_dirty = true;
            self.recovery_flush_now = true;
            "Restored conversations from your previous encrypted backup because the latest file could not be read.".clone_into(&mut self.recovery_notice);
            return;
        }
        if let Some((saved, report)) =
            primary.and_then(|bytes| self.salvage_workspace_data(bytes).ok())
        {
            self.apply_saved_workspace(saved);
            self.save_conversations = true;
            self.recovery_dirty = true;
            self.recovery_flush_now = true;
            self.recovery_notice = format!(
                "Recovered {} conversation(s); {} damaged conversation(s) could not be restored.",
                report.recovered_conversations, report.dropped_conversations
            );
            return;
        }
        if let Some(backup) = backup_bytes.as_deref()
            && let Ok((saved, report)) = self.salvage_workspace_data(backup)
        {
            self.apply_saved_workspace(saved);
            self.save_conversations = true;
            self.recovery_dirty = true;
            self.recovery_flush_now = true;
            self.recovery_notice = if report.dropped_conversations == 0 {
                "Restored conversations from your previous encrypted backup.".to_owned()
            } else {
                format!(
                    "Restored {} conversation(s) from your previous backup; {} damaged conversation(s) could not be restored.",
                    report.recovered_conversations, report.dropped_conversations
                )
            };
            return;
        }
        if load_failed {
            "Could not unlock saved conversations. The existing file is preserved. Enable saving to replace it with this workspace.".clone_into(&mut self.recovery_notice);
        } else {
            "Could not restore saved conversations. The existing file is preserved. Enable saving to replace it with this workspace.".clone_into(&mut self.recovery_notice);
        }
    }

    pub(super) fn workspace_bytes(&self) -> io::Result<Vec<u8>> {
        if !client::valid_daisyrose_scale(self.daisyrose_scale)
            || !self.memory.valid()
            || !valid_parts(
                &self.transcript,
                &self.input,
                &self.draft_attachments,
                self.output_budget,
                &self.status,
                &self.last_answer,
                &self.retained_draft,
                &self.committed,
            )
            || self
                .conversations
                .iter()
                .filter_map(|conversation| conversation.saved.as_ref())
                .any(|saved| !valid_snapshot(saved))
        {
            return Err(io::Error::other(
                "Conversation recovery reached its saved-state limits. Delete older conversations or disable saving.",
            ));
        }
        self.serialize_workspace().map_err(|error| {
            if error.io_error_kind() == Some(io::ErrorKind::FileTooLarge) {
                io::Error::other(
                    "Conversation recovery reached its 8 MiB limit. Delete older conversations or disable saving.",
                )
            } else {
                io::Error::other(error)
            }
        })
    }

    fn serialize_workspace(&self) -> Result<Vec<u8>, serde_json::Error> {
        const LIMIT: usize = 8 * 1024 * 1024;
        let initial_capacity = self.last_workspace_bytes.get().max(4096);
        let mut writer = crate::bounded_json::BoundedWriter::with_capacity(LIMIT, initial_capacity)
            .map_err(serde_json::Error::io)?;
        writer.write_raw(br#"{"version":1,"conversations":["#)?;
        for (index, conversation) in self.conversations.iter().enumerate() {
            if index > 0 {
                writer.write_raw(b",")?;
            }
            if conversation.id != self.active_conversation && conversation.saved.is_some() {
                let cached = conversation.cached_json.lock().ok().and_then(|slot| {
                    slot.as_ref()
                        .filter(|entry| {
                            entry.title == conversation.title
                                && entry.renamed == conversation.renamed
                        })
                        .map(|entry| std::sync::Arc::clone(&entry.json))
                });
                if let Some(bytes) = cached {
                    writer.write_raw(&bytes)?;
                } else {
                    let encoded = crate::bounded_json::to_vec_with_capacity(
                        conversation,
                        writer.remaining(),
                        4096,
                    )?;
                    let arc: std::sync::Arc<[u8]> = std::sync::Arc::from(encoded);
                    if let Ok(mut slot) = conversation.cached_json.lock() {
                        *slot = Some(super::workspace::CachedConversationJson {
                            title: conversation.title.clone(),
                            renamed: conversation.renamed,
                            json: std::sync::Arc::clone(&arc),
                        });
                    }
                    writer.write_raw(&arc)?;
                }
            } else {
                writer.write_json(conversation)?;
            }
        }
        writer.write_raw(br#"],"active_conversation":"#)?;
        writer.write_json(&self.active_conversation)?;
        writer.write_raw(br#","next_conversation":"#)?;
        writer.write_json(&self.next_conversation)?;
        writer.write_raw(br#","active":"#)?;
        writer.write_json(&ActiveView {
            transcript: &self.transcript,
            committed: &self.committed,
            input: &self.input,
            draft_attachments: &self.draft_attachments,
            last_answer: &self.last_answer,
            retained_draft: &self.retained_draft,
            answer_complete: self.answer_complete,
            extra_effort: self.extra_effort,
            auto_continue: self.auto_continue,
            enable_thinking: self.enable_thinking,
            streaming_read_aloud: self.streaming_read_aloud,
            output_budget: self.output_budget,
            daisyrose_enabled: self.daisyrose_enabled,
            daisyrose_scale: self.daisyrose_scale,
            cognitive: self.cognitive,
            status: &self.status,
            warning: self.warning,
        })?;
        writer.write_raw(br#","voice_profile":"#)?;
        writer.write_json(&self.voice_profile)?;
        if !self.memory.is_default_empty() {
            writer.write_raw(br#","memory":"#)?;
            writer.write_json(&self.memory)?;
        }
        writer.write_raw(b"}")?;
        let bytes = writer.into_vec();
        self.last_workspace_bytes.set(bytes.len());
        Ok(bytes)
    }

    pub(super) fn restore_workspace(&mut self, bytes: &[u8]) -> io::Result<()> {
        if bytes.len() > 8 * 1024 * 1024 {
            return Err(io::Error::other("Saved workspace is too large."));
        }
        let saved: SavedWorkspace = serde_json::from_slice(bytes).map_err(io::Error::other)?;
        if !saved.valid() {
            return Err(io::Error::other("Invalid saved workspace."));
        }
        self.apply_saved_workspace(saved);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn salvage_workspace(&mut self, bytes: &[u8]) -> io::Result<SalvageReport> {
        let (saved, report) = self.salvage_workspace_data(bytes)?;
        self.apply_saved_workspace(saved);
        Ok(report)
    }

    fn apply_saved_workspace(&mut self, mut saved: SavedWorkspace) {
        // A locally identified in-flight turn can supply context after a crash.
        // Recovery never starts or replays its request.
        let mut adopted = saved.active.transcript.recover_interrupted(
            &mut saved.active.committed,
            &saved.active.last_answer,
            saved.active.answer_complete,
        );
        saved
            .active
            .transcript
            .interrupt_turn(saved.active.committed.len());
        for conversation in &mut saved.conversations {
            if let Some(snapshot) = &mut conversation.saved {
                adopted |= snapshot.transcript.recover_interrupted(
                    &mut snapshot.committed,
                    &snapshot.last_answer,
                    snapshot.answer_complete,
                );
                snapshot.transcript.interrupt_turn(snapshot.committed.len());
            }
        }
        // Readiness belongs to the fresh owned broker, never to disk.
        "Recovered conversation. Preparing your local model…".clone_into(&mut saved.active.status);
        saved.active.warning = false;
        self.conversations = saved.conversations;
        self.active_conversation = saved.active_conversation;
        self.next_conversation = saved.next_conversation;
        self.voice_profile = saved.voice_profile;
        self.memory = saved.memory;
        self.memory_revision = self.memory_revision.wrapping_add(1);
        self.load_conversation(saved.active);
        self.restore_history = true;
        self.recovery_dirty |= adopted;
    }

    fn salvage_workspace_data(&self, bytes: &[u8]) -> io::Result<(SavedWorkspace, SalvageReport)> {
        if bytes.len() > 8 * 1024 * 1024 {
            return Err(io::Error::other("Saved workspace is too large."));
        }
        let root: serde_json::Value = serde_json::from_slice(bytes).map_err(io::Error::other)?;
        if root.get("version").and_then(serde_json::Value::as_u64) != Some(1) {
            return Err(io::Error::other("Unsupported workspace version."));
        }
        let raw_conversations = root
            .get("conversations")
            .and_then(serde_json::Value::as_array)
            .filter(|items| !items.is_empty())
            .ok_or_else(|| io::Error::other("Missing saved conversations."))?;

        let requested_active = root
            .get("active_conversation")
            .and_then(serde_json::Value::as_u64);
        let default_scale = self.daisyrose_default_scale();
        let mut conversations = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        let mut dropped_conversations = 0_usize;
        let mut active_entry: Option<(usize, ConversationSnapshot)> = None;

        for item in raw_conversations {
            let Some(id) = item
                .get("id")
                .and_then(serde_json::Value::as_u64)
                .filter(|id| *id < u64::MAX && seen_ids.insert(*id))
            else {
                dropped_conversations += 1;
                continue;
            };
            let title = item
                .get("title")
                .and_then(serde_json::Value::as_str)
                .map_or_else(
                    || "Recovered conversation".to_owned(),
                    |text| truncate_chars(text, 64),
                );
            let renamed = item
                .get("renamed")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let is_active = Some(id) == requested_active
                && item.get("saved").is_none_or(serde_json::Value::is_null);
            let snapshot_value = if is_active {
                root.get("active")
            } else {
                item.get("saved").filter(|value| !value.is_null())
            };
            let Some(snapshot) =
                snapshot_value.and_then(|value| salvage_snapshot(value, default_scale))
            else {
                dropped_conversations += 1;
                continue;
            };

            let mut conversation = Conversation::new(id);
            conversation.title = title;
            conversation.renamed = renamed;
            if is_active {
                active_entry = Some((conversations.len(), snapshot));
            } else {
                conversation.saved = Some(snapshot);
            }
            conversations.push(conversation);
        }

        if conversations.is_empty() {
            return Err(io::Error::other("No recoverable conversations found."));
        }

        let (active_index, active_snapshot) = if let Some((index, snapshot)) = active_entry {
            (index, snapshot)
        } else {
            let mut found = None;
            for (index, conversation) in conversations.iter_mut().enumerate() {
                if let Some(snapshot) = conversation.saved.take() {
                    found = Some((index, snapshot));
                    break;
                }
            }
            found.ok_or_else(|| io::Error::other("Missing active snapshot."))?
        };
        let active_conversation = conversations[active_index].id;
        let max_id = conversations
            .iter()
            .map(|conversation| conversation.id)
            .max()
            .unwrap_or(active_conversation);
        let next_conversation = root
            .get("next_conversation")
            .and_then(serde_json::Value::as_u64)
            .filter(|next| *next > max_id && *next < u64::MAX)
            .unwrap_or_else(|| max_id.saturating_add(1).min(u64::MAX - 1));
        if max_id >= next_conversation {
            return Err(io::Error::other(
                "Conversation identifiers exceed capacity.",
            ));
        }

        let voice_profile = root
            .get("voice_profile")
            .and_then(serde_json::Value::as_str)
            .filter(|voice| voice.chars().count() <= VOICE_LIMIT)
            .unwrap_or_default()
            .to_owned();
        let memory = root.get("memory").map(salvage_memory).unwrap_or_default();

        let has_user_content = !memory.is_default_empty()
            || snapshot_has_user_content(&active_snapshot)
            || conversations.iter().any(|conversation| {
                conversation
                    .saved
                    .as_ref()
                    .is_some_and(snapshot_has_user_content)
            });
        if !has_user_content {
            return Err(io::Error::other(
                "No saved conversation content to salvage.",
            ));
        }

        let recovered_conversations = conversations.len();
        let saved = SavedWorkspace {
            version: 1,
            conversations,
            active_conversation,
            next_conversation,
            active: active_snapshot,
            voice_profile,
            memory,
        };
        if !saved.valid() {
            return Err(io::Error::other("Salvaged workspace failed validation."));
        }
        Ok((
            saved,
            SalvageReport {
                recovered_conversations,
                dropped_conversations,
            },
        ))
    }

    pub(super) fn recovery_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Conversations");
        if ui
            .add_enabled(
                self.recovery.is_some(),
                egui::Checkbox::new(
                    &mut self.save_conversations,
                    "Save conversations on this device",
                ),
            )
            .changed()
        {
            self.alerts
                .clear_dismissed(super::error_ui::AlertKind::Recovery);
            self.recovery_dirty = self.save_conversations;
            self.recovery_erase_pending = !self.save_conversations;
            if !self.save_conversations {
                self.recently_deleted = None;
            }
            self.last_recovery_save = Instant::now()
                .checked_sub(Duration::from_secs(3))
                .unwrap_or_else(Instant::now);
            if self.save_conversations {
                "Saving drafts and conversations with Windows user encryption…"
            } else {
                "Removing the saved workspace. Open conversations stay available in this window."
            }
            .clone_into(&mut self.recovery_notice);
        }
        ui.label(RichText::new("Saves automatically with Windows user encryption in your model data folder. Restores drafts and completed history on the next launch. Turning this off removes the saved copy and keeps saving off on future launches. Your window, appearance and voice preferences remain saved on this device.").small().weak());
        if self.recently_deleted.is_some() {
            ui.add_space(6.0);
            if ui
                .add_enabled(
                    self.model.workspace_available(),
                    egui::Button::new("Restore last deleted conversation"),
                )
                .clicked()
            {
                self.undo_delete_conversation();
            }
        }
        if !self.recovery_notice.is_empty() {
            ui.label(RichText::new(&self.recovery_notice).small());
        }
        ui.add_space(16.0);
    }

    fn disconnect_recovery(&mut self) {
        self.recovery = None;
        self.recovery_dirty = false;
        self.recovery_erase_pending = false;
        self.alerts
            .clear_dismissed(super::error_ui::AlertKind::Recovery);
        "Could not save conversations: the recovery worker stopped. Unsaved work remains in this window."
            .clone_into(&mut self.recovery_notice);
    }

    pub(super) fn poll_recovery(&mut self) {
        let mut disconnected = false;
        if let Some(recovery) = &self.recovery {
            loop {
                match recovery.events.try_recv() {
                    Ok(event) => {
                        if !recovery.is_current(&event) {
                            continue;
                        }
                        match event {
                            crate::recovery::Event::Saved(_) if self.save_conversations => {
                                self.alerts
                                    .clear_dismissed(super::error_ui::AlertKind::Recovery);
                                self.recovery_quarantine_primary = false;
                                "Saved on this device with Windows user encryption."
                                    .clone_into(&mut self.recovery_notice);
                            }
                            crate::recovery::Event::Erased(_) if !self.save_conversations => {
                                self.alerts
                                    .clear_dismissed(super::error_ui::AlertKind::Recovery);
                                self.recovery_erase_pending = false;
                                self.recovery_quarantine_primary = false;
                                "Saving is off. The saved workspace has been removed."
                                    .clone_into(&mut self.recovery_notice);
                            }
                            crate::recovery::Event::Failed(_, error) => {
                                self.recovery_notice = error.notice();
                                self.recovery_dirty = self.save_conversations;
                            }
                            crate::recovery::Event::Saved(_)
                            | crate::recovery::Event::Erased(_) => {}
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        if disconnected {
            self.disconnect_recovery();
            return;
        }
        let flush_now = std::mem::take(&mut self.recovery_flush_now);
        let interval = if !self.preview && self.model.busy() {
            Duration::from_secs(15)
        } else {
            Duration::from_secs(2)
        };
        if !flush_now && self.last_recovery_save.elapsed() < interval {
            return;
        }
        self.last_recovery_save = Instant::now();
        if self.recovery_erase_pending {
            if let Some(recovery) = &self.recovery
                && recovery.erase() == crate::recovery::DispatchStatus::Disconnected
            {
                self.disconnect_recovery();
            }
        } else if self.save_conversations
            && self.recovery_dirty
            && let Some(recovery) = &self.recovery
        {
            match recovery
                .save_with_quarantine(self.recovery_quarantine_primary, || self.workspace_bytes())
            {
                Ok(crate::recovery::DispatchStatus::Queued) => self.recovery_dirty = false,
                Ok(crate::recovery::DispatchStatus::Busy) => self.recovery_dirty = true,
                Ok(crate::recovery::DispatchStatus::Disconnected) => {
                    self.disconnect_recovery();
                }
                Err(error) => {
                    self.recovery_notice = format!("Could not save conversations: {error}");
                }
            }
        }
    }
}

impl SavedWorkspace {
    fn valid(&self) -> bool {
        let mut ids = std::collections::HashSet::new();
        self.version == 1
            && !self.conversations.is_empty()
            && self.voice_profile.chars().count() <= VOICE_LIMIT
            && self.next_conversation < u64::MAX
            && self.conversations.iter().all(|conversation| {
                conversation.id < self.next_conversation
                    && ids.insert(conversation.id)
                    && conversation.title.chars().count() <= 64
                    && if conversation.id == self.active_conversation {
                        conversation.saved.is_none()
                    } else {
                        conversation.saved.as_ref().is_some_and(valid_snapshot)
                    }
            })
            && ids.contains(&self.active_conversation)
            && self.memory.valid()
            && valid_snapshot(&self.active)
    }
}

fn valid_snapshot(saved: &ConversationSnapshot) -> bool {
    client::valid_daisyrose_scale(saved.daisyrose_scale)
        && valid_parts(
            &saved.transcript,
            &saved.input,
            &saved.draft_attachments,
            saved.output_budget,
            &saved.status,
            &saved.last_answer,
            &saved.retained_draft,
            &saved.committed,
        )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Validation checks every active/saved conversation field uniformly."
)]
fn valid_parts(
    transcript: &Transcript,
    input: &str,
    draft_attachments: &[crate::attachment_import::Attachment],
    output_budget: usize,
    status: &str,
    last_answer: &str,
    retained_draft: &str,
    committed: &[client::Message],
) -> bool {
    transcript.valid_saved()
        && super::transcript::valid_attachments(draft_attachments)
        && super::draft_buffer::fits(input)
        && output_budget < OUTPUT_TOKENS.len()
        && status.len() <= MAX_DIAGNOSTIC_CHARS * 4
        && last_answer.len() <= 4 * 1024 * 1024
        && retained_draft.len() <= 4 * 1024 * 1024
        && committed.len() <= 128
        && committed.len().is_multiple_of(2)
        && committed
            .iter()
            .map(|message| message.images.len())
            .fold(0_usize, usize::saturating_add)
            <= client::MAX_REQUEST_IMAGES
        && committed.iter().enumerate().all(|(index, message)| {
            let expected_role = if index.is_multiple_of(2) {
                "user"
            } else {
                "assistant"
            };
            message.role == expected_role
                && (message.role == "user" || message.images.is_empty())
                && message.content.len() <= 1024 * 1024
                && message
                    .reasoning_content
                    .as_ref()
                    .is_none_or(|text| text.len() <= 1024 * 1024)
        })
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn truncate_utf8(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn snapshot_has_user_content(snapshot: &ConversationSnapshot) -> bool {
    !snapshot.transcript.messages.is_empty()
        || !snapshot.committed.is_empty()
        || !snapshot.input.is_empty()
        || !snapshot.draft_attachments.is_empty()
        || !snapshot.last_answer.is_empty()
        || !snapshot.retained_draft.is_empty()
}

fn salvage_attachments(value: &serde_json::Value) -> Vec<crate::attachment_import::Attachment> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let candidate: Vec<crate::attachment_import::Attachment> = items
        .iter()
        .filter_map(|item| serde_json::from_value(item.clone()).ok())
        .take(crate::attachment_import::MAX_ATTACHMENTS_PER_MESSAGE)
        .collect();
    if super::transcript::valid_attachments(&candidate) {
        candidate
    } else {
        Vec::new()
    }
}

fn salvage_committed(value: &serde_json::Value) -> Vec<client::Message> {
    #[derive(serde::Deserialize)]
    struct LooseWireMessage {
        role: String,
        content: String,
        #[serde(default)]
        reasoning_content: Option<String>,
        #[serde(default)]
        images: Vec<String>,
    }

    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let mut committed = Vec::new();
    let mut total_images = 0_usize;
    for item in items.iter().take(128) {
        let expected_role = if committed.len().is_multiple_of(2) {
            "user"
        } else {
            "assistant"
        };
        let message = if let Ok(strict) = serde_json::from_value::<client::Message>(item.clone()) {
            strict
        } else if let Ok(loose) = serde_json::from_value::<LooseWireMessage>(item.clone()) {
            let role = match loose.role.as_str() {
                "user" => "user",
                "assistant" => "assistant",
                _ => break,
            };
            let images: Vec<std::sync::Arc<str>> = if role == "user" {
                loose
                    .images
                    .into_iter()
                    .take(client::MAX_REQUEST_IMAGES)
                    .map(std::sync::Arc::from)
                    .collect()
            } else {
                Vec::new()
            };
            client::Message {
                role,
                content: loose.content.into(),
                reasoning_content: loose.reasoning_content.map(Into::into),
                images: images.into(),
            }
        } else {
            break;
        };
        if message.role != expected_role
            || (message.role != "user" && !message.images.is_empty())
            || message.content.len() > 1024 * 1024
            || message
                .reasoning_content
                .as_ref()
                .is_some_and(|text| text.len() > 1024 * 1024)
            || total_images.saturating_add(message.images.len()) > client::MAX_REQUEST_IMAGES
        {
            break;
        }
        total_images = total_images.saturating_add(message.images.len());
        committed.push(message);
    }
    if !committed.len().is_multiple_of(2) {
        committed.pop();
    }
    committed
}

fn salvage_cognitive(value: &serde_json::Value) -> client::CognitiveOptions {
    if let Ok(strict) = serde_json::from_value::<client::CognitiveOptions>(value.clone()) {
        return strict;
    }
    client::CognitiveOptions {
        reasoning_effort: value
            .get("reasoning_effort")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default(),
        thinking_budget: value
            .get("thinking_budget")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default(),
        sampling_preset: value
            .get("sampling_preset")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default(),
        extra_effort_strategy: value
            .get("extra_effort_strategy")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default(),
    }
}

fn salvage_memory(value: &serde_json::Value) -> crate::context_cards::MemoryStore {
    #[derive(serde::Deserialize)]
    struct LooseCard {
        id: u64,
        title: String,
        #[serde(default)]
        triggers: Vec<String>,
        content: String,
        #[serde(default)]
        pinned: bool,
        #[serde(default = "default_true_flag")]
        enabled: bool,
    }
    const fn default_true_flag() -> bool {
        true
    }
    #[derive(serde::Deserialize)]
    struct LooseMemory {
        #[serde(default)]
        persistent_instructions: String,
        #[serde(default)]
        cards: Vec<serde_json::Value>,
        #[serde(default)]
        token_budget: Option<usize>,
        #[serde(default)]
        next_card_id: u64,
    }

    if let Ok(strict) = serde_json::from_value::<crate::context_cards::MemoryStore>(value.clone())
        && strict.valid()
    {
        return strict;
    }
    let Ok(loose) = serde_json::from_value::<LooseMemory>(value.clone()) else {
        return crate::context_cards::MemoryStore::default();
    };
    let persistent_instructions = truncate_chars(
        &loose.persistent_instructions,
        crate::context_cards::MAX_INSTRUCTIONS_CHARS,
    );
    let token_budget = loose
        .token_budget
        .filter(|budget| {
            (crate::context_cards::MIN_TOKEN_BUDGET..=crate::context_cards::MAX_TOKEN_BUDGET)
                .contains(budget)
        })
        .unwrap_or(crate::context_cards::DEFAULT_TOKEN_BUDGET);
    let mut cards = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();
    let mut max_id = 0_u64;
    for raw in loose
        .cards
        .into_iter()
        .take(crate::context_cards::MAX_CARDS)
    {
        let Ok(card) = serde_json::from_value::<LooseCard>(raw) else {
            continue;
        };
        if card.id == u64::MAX || !seen_ids.insert(card.id) {
            continue;
        }
        let title = truncate_chars(&card.title, crate::context_cards::MAX_CARD_TITLE_CHARS);
        let content = truncate_chars(&card.content, crate::context_cards::MAX_CARD_CONTENT_CHARS);
        let triggers = card
            .triggers
            .into_iter()
            .take(crate::context_cards::MAX_CARD_TRIGGERS)
            .map(|trigger| truncate_chars(&trigger, crate::context_cards::MAX_CARD_TRIGGER_CHARS))
            .collect();
        max_id = max_id.max(card.id);
        cards.push(crate::context_cards::ContextCard {
            id: card.id,
            title,
            triggers,
            content,
            pinned: card.pinned,
            enabled: card.enabled,
        });
    }
    let next_card_id = if cards.is_empty() {
        loose.next_card_id
    } else {
        loose.next_card_id.max(max_id.saturating_add(1))
    };
    let store = crate::context_cards::MemoryStore {
        persistent_instructions,
        cards,
        token_budget,
        next_card_id,
    };
    if store.valid() {
        store
    } else {
        crate::context_cards::MemoryStore::default()
    }
}

fn salvage_snapshot(
    value: &serde_json::Value,
    default_daisyrose_scale: f32,
) -> Option<ConversationSnapshot> {
    let object = value.as_object()?;
    if ![
        "transcript",
        "committed",
        "input",
        "last_answer",
        "retained_draft",
        "draft_attachments",
    ]
    .iter()
    .any(|key| object.contains_key(*key))
    {
        return None;
    }
    let transcript = object
        .get("transcript")
        .map(Transcript::salvage_from_json)
        .unwrap_or_default();
    let committed = object
        .get("committed")
        .map(salvage_committed)
        .unwrap_or_default();
    let input = object
        .get("input")
        .and_then(serde_json::Value::as_str)
        .filter(|text| super::draft_buffer::fits(text))
        .unwrap_or_default()
        .to_owned();
    let draft_attachments = object
        .get("draft_attachments")
        .map(salvage_attachments)
        .unwrap_or_default();
    let last_answer = object
        .get("last_answer")
        .and_then(serde_json::Value::as_str)
        .map(|text| truncate_utf8(text, 4 * 1024 * 1024).to_owned())
        .unwrap_or_default();
    let retained_draft = object
        .get("retained_draft")
        .and_then(serde_json::Value::as_str)
        .map(|text| truncate_utf8(text, 4 * 1024 * 1024).to_owned())
        .unwrap_or_default();
    let answer_complete = object
        .get("answer_complete")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let extra_effort = object
        .get("extra_effort")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let auto_continue = object
        .get("auto_continue")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let enable_thinking = object
        .get("enable_thinking")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let streaming_read_aloud = object
        .get("streaming_read_aloud")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let output_budget = object
        .get("output_budget")
        .and_then(serde_json::Value::as_u64)
        .and_then(|budget| usize::try_from(budget).ok())
        .filter(|budget| *budget < OUTPUT_TOKENS.len())
        .unwrap_or(3);
    let daisyrose_enabled = object
        .get("daisyrose_enabled")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let daisyrose_scale = object
        .get("daisyrose_scale")
        .and_then(|scale| serde_json::from_value::<f32>(scale.clone()).ok())
        .filter(|scale| client::valid_daisyrose_scale(*scale))
        .unwrap_or(default_daisyrose_scale);
    let cognitive = object
        .get("cognitive")
        .map(salvage_cognitive)
        .unwrap_or_default();
    let status = object
        .get("status")
        .and_then(serde_json::Value::as_str)
        .map_or_else(
            || "Ready when you are.".to_owned(),
            |text| truncate_chars(text, MAX_DIAGNOSTIC_CHARS),
        );
    let warning = object
        .get("warning")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let snapshot = ConversationSnapshot {
        transcript,
        committed,
        context_history: None,
        input,
        draft_attachments,
        last_answer,
        retained_draft,
        answer_complete,
        extra_effort,
        auto_continue,
        enable_thinking,
        streaming_read_aloud,
        output_budget,
        _legacy_abliteration_scale: None,
        daisyrose_enabled,
        daisyrose_scale,
        cognitive,
        status,
        status_detail: String::new(),
        warning,
        stage: None,
    };
    valid_snapshot(&snapshot).then_some(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::ModelLifecycle;
    use crate::window::{Role, mpsc, transcript::MAX_MESSAGES};

    #[test]
    fn retry_response_recovery_keeps_failed_turn_identity_and_requires_explicit_send()
    -> io::Result<()> {
        for settled in [false, true] {
            let mut original = state();
            original.transcript.start_turn("Original prompt".into(), 0);
            original.transcript.append("Unfinished thinking", true);
            if settled {
                original.transcript.interrupt_turn(0);
            }
            original.input = "Newer draft".into();
            let (commands, incoming) = mpsc::channel();
            let (_events, events) = client::event_channel();
            let mut restored = State::create(commands, events, None, None);
            restored.preview = false;
            restored.restore_workspace(&original.workspace_bytes()?)?;
            restored.model = ModelLifecycle::Ready;
            restored.restore_selected_history();
            assert!(
                matches!(incoming.try_recv(), Ok(super::super::Command::RestoreChat(history)) if history.is_empty())
            );
            assert!(incoming.try_recv().is_err());
            let prompt = restored.transcript.messages[0].id;
            restored.message_action(super::super::iteration::MessageAction::Retry(prompt));
            assert!(
                matches!(incoming.try_recv(), Ok(super::super::Command::Send { text, .. }) if text == "Original prompt")
            );
            assert!(incoming.try_recv().is_err());
            assert_eq!(restored.input, "Newer draft");
            assert_eq!(restored.transcript.messages[0].id, prompt);
            assert_eq!(restored.transcript.messages.len(), 2);
            assert!(
                !restored
                    .transcript
                    .plain_text()
                    .contains("Unfinished thinking")
            );
        }
        Ok(())
    }

    #[test]
    fn long_draft_recovery_retains_every_unicode_character_at_the_limit() -> io::Result<()> {
        let mut original = state();
        original.input = "🌱".repeat(super::super::INPUT_LIMIT);
        let bytes = original.workspace_bytes()?;
        let mut restored = state();
        restored.restore_workspace(&bytes)?;
        assert_eq!(restored.input, original.input);
        assert_eq!(restored.input.chars().count(), super::super::INPUT_LIMIT);
        assert!(!restored.model.busy());
        Ok(())
    }

    #[test]
    fn long_draft_recovery_rejects_an_extra_character_without_replacing_the_workspace()
    -> io::Result<()> {
        let mut original = state();
        original.input = "Keep this draft".into();
        let accepted = original.workspace_bytes()?;
        let mut saved: serde_json::Value =
            serde_json::from_slice(&accepted).map_err(io::Error::other)?;
        saved["active"]["input"] = serde_json::json!("x".repeat(super::super::INPUT_LIMIT + 1));
        assert!(
            original
                .restore_workspace(&serde_json::to_vec(&saved).map_err(io::Error::other)?)
                .is_err()
        );
        assert_eq!(original.workspace_bytes()?, accepted);
        Ok(())
    }

    fn state() -> State {
        let (commands, _incoming) = mpsc::channel();
        let (_events, incoming) = client::event_channel();
        State::create(commands, incoming, None, None)
    }

    #[test]
    fn interruption_recovery_adopts_identified_partial_without_replaying() -> io::Result<()> {
        let mut original = state();
        original.transcript.start_turn("Original prompt".into(), 0);
        original.transcript.append("Partial response", false);
        original.input = "New draft".into();
        let bytes = original.workspace_bytes()?;
        let (commands, incoming) = mpsc::channel();
        let (_events, events) = client::event_channel();
        let mut restored = State::create(commands, events, None, None);
        restored.restore_workspace(&bytes)?;
        assert_eq!(restored.committed.len(), 2);
        assert_eq!(restored.committed[0].content.as_ref(), "Original prompt");
        assert_eq!(restored.committed[1].content.as_ref(), "Partial response");
        assert_eq!(restored.input, "New draft");
        assert!(incoming.try_recv().is_err());
        let answer = restored
            .transcript
            .messages
            .last()
            .ok_or_else(|| io::Error::other("partial answer missing"))?;
        assert_eq!(answer.committed_index(&restored.committed), Some(1));
        assert!(answer.interrupted);
        Ok(())
    }

    #[test]
    fn token_limit_recovery_preserves_continue_and_retry_marks_without_replaying() -> io::Result<()>
    {
        let mut original = state();
        original.model = ModelLifecycle::Generating;
        original.transcript.start_turn("Long prompt".into(), 0);
        original.transcript.append("Capped response", false);
        let history = vec![
            client::Message {
                role: "user",
                content: "Long prompt".into(),
                reasoning_content: None,
                images: std::sync::Arc::default(),
            },
            client::Message {
                role: "assistant",
                content: "Capped response".into(),
                reasoning_content: None,
                images: std::sync::Arc::default(),
            },
        ];
        original.model_event(client::Event::Committed(history));
        original.model_event(client::Event::Complete(client::Completion::TokenLimit));
        original.input = "Saved draft".into();
        let bytes = original.workspace_bytes()?;

        let (commands, incoming) = mpsc::channel();
        let (_events, events) = client::event_channel();
        let mut restored = State::create(commands, events, None, None);
        restored.restore_workspace(&bytes)?;
        assert!(incoming.try_recv().is_err());
        let answer = restored
            .transcript
            .messages
            .last()
            .ok_or_else(|| io::Error::other("capped answer missing"))?;
        assert!(answer.token_limit);
        assert!(answer.can_continue(&restored.committed));
        assert_eq!(answer.retry_prefix(&restored.committed), Some(0));
        Ok(())
    }

    #[test]
    fn recovery_automatically_saves_and_restores_completed_history() -> io::Result<()> {
        let directory = recovery_directory("automatic")?;
        let mut original = state();
        original.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert!(original.save_conversations && original.recovery_dirty);
        original.input = "Draft for the next launch".to_owned();
        original
            .transcript
            .push(Role::User, "Remember this".to_owned());
        original
            .transcript
            .push(Role::Assistant, "Remembered".to_owned());
        original.committed = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "Remember this"},
            {"role": "assistant", "content": "Remembered", "reasoning_content": "Saved reasoning"},
        ]))
        .map_err(io::Error::other)?;
        // Closing before the autosave interval must still flush the final state.
        drop(original);

        let (commands, incoming) = mpsc::channel();
        let (_events, events) = client::event_channel();
        let mut restored = State::create(commands, events, None, None);
        restored.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert!(restored.save_conversations && restored.restore_history);
        assert_eq!(restored.input, "Draft for the next launch");
        assert!(restored.transcript.plain_text().contains("Remembered"));
        assert_eq!(restored.committed.len(), 2);
        assert_eq!(
            restored.committed[1].reasoning_content.as_deref(),
            Some("Saved reasoning")
        );
        assert!(incoming.try_recv().is_err());
        restored.preview = false;
        restored.model = ModelLifecycle::Preparing;
        restored.model_event(client::Event::Ready);
        assert!(
            matches!(incoming.try_recv(), Ok(client::Command::RestoreChat(history)) if history.len() == 2)
        );
        assert!(incoming.try_recv().is_err());
        assert!(!restored.speech_input && !restored.speaking);
        drop(restored);
        remove_recovery_directory(&directory)
    }

    #[test]
    fn recovery_remembers_opt_out_and_explicit_reenable() -> io::Result<()> {
        let directory = recovery_directory("preference")?;
        let mut original = state();
        original.attach_recovery(crate::recovery::Recovery::open(&directory));
        original.input = "Do not retain this draft".to_owned();
        original.save_conversations = false;
        original.recovery_erase_pending = true;
        drop(original);
        let path = directory.join("chat-recovery.dpapi");
        assert!(!path.exists());
        // Even ciphertext left after a failed/interrupted deletion cannot be
        // restored while saving is disabled. It is removed on the next close.
        std::fs::write(&path, b"Unwanted unreadable old ciphertext")?;
        let mut disabled = state();
        disabled.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert!(!disabled.save_conversations && disabled.recovery_erase_pending);
        assert!(disabled.input.is_empty() && !disabled.restore_history);
        drop(disabled);
        assert!(!path.exists());

        let mut enabled = state();
        enabled.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert!(!enabled.save_conversations);
        enabled.save_conversations = true;
        enabled.recovery_erase_pending = false;
        enabled.recovery_dirty = true;
        enabled.input = "Keep this newer draft".to_owned();
        drop(enabled);
        let mut restored = state();
        restored.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert!(restored.save_conversations);
        assert_eq!(restored.input, "Keep this newer draft");
        drop(restored);
        remove_recovery_directory(&directory)
    }

    #[test]
    fn recovery_automatic_saving_preserves_unreadable_and_invalid_workspaces() -> io::Result<()> {
        let directory = recovery_directory("invalid")?;
        let path = directory.join("chat-recovery.dpapi");
        for bytes in [
            b"Unreadable ciphertext".to_vec(),
            bonsai_windows_credentials::protect_private_data(b"{}")?,
        ] {
            std::fs::write(&path, &bytes)?;
            let mut original = state();
            original.attach_recovery(crate::recovery::Recovery::open(&directory));
            assert!(!original.save_conversations && !original.recovery_erase_pending);
            assert!(original.recovery_notice.starts_with("Could not"));
            original.input = "Must not replace the saved file".to_owned();
            original.recovery_dirty = true;
            drop(original);
            assert_eq!(std::fs::read(&path)?, bytes);
        }
        remove_recovery_directory(&directory)
    }

    fn recovery_directory(label: &str) -> io::Result<std::path::PathBuf> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("Bonsai-recovery-{label}-{stamp}"));
        std::fs::create_dir(&directory)?;
        Ok(directory)
    }

    fn remove_recovery_directory(directory: &std::path::Path) -> io::Result<()> {
        for name in [
            "chat-recovery.dpapi",
            "chat-recovery.backup.dpapi",
            "chat-recovery.preserved.dpapi",
            "chat-recovery.lock",
        ] {
            let path = directory.join(name);
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
        std::fs::remove_dir(directory)
    }

    #[test]
    fn conversation_history_recovery_round_trips_many_conversations() -> io::Result<()> {
        let mut original = state();
        original.input = "Active draft 雨 🌱".to_owned();
        let mut saved: serde_json::Value =
            serde_json::from_slice(&original.workspace_bytes()?).map_err(io::Error::other)?;
        let active = saved["active"].clone();
        let mut conversations = vec![saved["conversations"][0].clone()];
        // Exercise a workspace well beyond the former 16-conversation cap,
        // including independent inactive drafts, settings and model history.
        for id in 2..=128_u64 {
            let prompt = format!("Prompt {id}");
            let answer = format!("Answer {id}");
            let mut snapshot = active.clone();
            snapshot["input"] = serde_json::json!(format!("Draft {id} 🌱"));
            snapshot["last_answer"] = serde_json::json!(answer);
            snapshot["answer_complete"] = serde_json::json!(true);
            snapshot["enable_thinking"] = serde_json::json!(id.is_multiple_of(2));
            snapshot["committed"] = serde_json::json!([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]);
            snapshot["transcript"]["messages"] = serde_json::json!([
                {"role": "User", "body": prompt, "reasoning": ""},
                {"role": "Assistant", "body": answer, "reasoning": ""},
            ]);
            conversations.push(serde_json::json!({
                "id": id,
                "title": format!("Conversation {id}"),
                "renamed": true,
                "saved": snapshot,
            }));
        }
        saved["conversations"] = serde_json::json!(conversations);
        saved["next_conversation"] = serde_json::json!(129);
        original.restore_workspace(&serde_json::to_vec(&saved).map_err(io::Error::other)?)?;
        let mut restored = state();
        restored.restore_workspace(&original.workspace_bytes()?)?;
        assert_eq!(restored.conversations.len(), 128);
        assert_eq!(restored.active_conversation, 1);
        assert_eq!(restored.next_conversation, 129);
        assert_eq!(restored.input, "Active draft 雨 🌱");
        for conversation in restored.conversations.iter().skip(1) {
            let id = conversation.id;
            assert_eq!(conversation.title, format!("Conversation {id}"));
            let snapshot = conversation
                .saved
                .as_ref()
                .ok_or_else(|| io::Error::other("Inactive conversation snapshot missing."))?;
            assert_eq!(snapshot.input, format!("Draft {id} 🌱"));
            assert_eq!(snapshot.committed.len(), 2);
            assert_eq!(
                snapshot.committed[0].content.as_ref(),
                format!("Prompt {id}")
            );
            assert_eq!(
                snapshot.committed[1].content.as_ref(),
                format!("Answer {id}")
            );
            assert_eq!(snapshot.enable_thinking, id.is_multiple_of(2));
            assert!(
                snapshot
                    .transcript
                    .plain_text()
                    .contains(&format!("Answer {id}"))
            );
        }
        assert!(!restored.model.busy() && !restored.speaking);
        Ok(())
    }

    #[test]
    fn conversation_history_recovery_rejects_invalid_identities_without_replacing_state()
    -> io::Result<()> {
        let mut original = state();
        original.input = "Keep this draft after any rejected restore.".to_owned();
        let mut saved: serde_json::Value =
            serde_json::from_slice(&original.workspace_bytes()?).map_err(io::Error::other)?;
        let active = saved["active"].clone();
        saved["conversations"]
            .as_array_mut()
            .ok_or_else(|| io::Error::other("Expected conversation list."))?
            .push(serde_json::json!({
                "id": 2, "title": "Inactive", "renamed": false, "saved": active,
            }));
        saved["next_conversation"] = serde_json::json!(3);
        original.restore_workspace(&serde_json::to_vec(&saved).map_err(io::Error::other)?)?;
        let accepted = original.workspace_bytes()?;
        for invalid in 0..5 {
            let mut rejected = saved.clone();
            match invalid {
                0 => rejected["conversations"][1]["id"] = serde_json::json!(1),
                1 => rejected["conversations"][1]["id"] = serde_json::json!(3),
                2 => rejected["active_conversation"] = serde_json::json!(99),
                3 => rejected["next_conversation"] = serde_json::json!(u64::MAX),
                _ => rejected["conversations"][1]["saved"] = serde_json::Value::Null,
            }
            assert!(
                original
                    .restore_workspace(&serde_json::to_vec(&rejected).map_err(io::Error::other)?)
                    .is_err()
            );
            assert_eq!(original.workspace_bytes()?, accepted);
        }
        Ok(())
    }

    #[test]
    fn conversation_history_recovery_rejects_oversized_bytes_without_replacing_state()
    -> io::Result<()> {
        let mut original = state();
        original.input = "Keep this draft after an oversized restore.".to_owned();
        let accepted = original.workspace_bytes()?;
        let error = original
            .restore_workspace(&vec![b' '; 8 * 1024 * 1024 + 1])
            .err()
            .ok_or_else(|| io::Error::other("Expected recovery byte limit."))?;
        assert!(error.to_string().contains("too large"));
        assert_eq!(original.workspace_bytes()?, accepted);
        Ok(())
    }

    #[test]
    fn recovery_rejects_excess_messages_before_decoding_their_payloads() -> io::Result<()> {
        let mut original = state();
        // Empty strings still allocate message/container ownership. Stop at the
        // count limit, including inactive conversations, before decoding a new
        // object. A malformed extra object must not change the reported cause.
        for inactive in [false, true] {
            for transcript in [false, true] {
                let mut saved: serde_json::Value =
                    serde_json::from_slice(&state().workspace_bytes()?)
                        .map_err(io::Error::other)?;
                if inactive {
                    let active = saved["active"].clone();
                    saved["conversations"]
                        .as_array_mut()
                        .ok_or_else(|| io::Error::other("Expected conversation list."))?
                        .push(serde_json::json!({
                            "id": 2, "title": "Inactive", "renamed": false, "saved": active,
                        }));
                    saved["next_conversation"] = serde_json::json!(3);
                }
                let snapshot = if inactive {
                    &mut saved["conversations"][1]["saved"]
                } else {
                    &mut saved["active"]
                };
                let limit: usize = if transcript { 512 } else { 128 };
                let messages =
                    if transcript {
                        (0..limit)
                        .map(|_| serde_json::json!({"role": "User", "body": "", "reasoning": ""}))
                        .collect::<Vec<_>>()
                    } else {
                        (0..limit)
                        .map(|index| serde_json::json!({
                            "role": if index.is_multiple_of(2) { "user" } else { "assistant" },
                            "content": "",
                        }))
                        .collect::<Vec<_>>()
                    };
                let field = if transcript {
                    &mut snapshot["transcript"]["messages"]
                } else {
                    &mut snapshot["committed"]
                };
                *field = serde_json::json!(messages);
                let bytes = serde_json::to_vec(&saved).map_err(io::Error::other)?;
                original.restore_workspace(&bytes)?;
                let accepted = original.workspace_bytes()?;
                let snapshot = if inactive {
                    &mut saved["conversations"][1]["saved"]
                } else {
                    &mut saved["active"]
                };
                let field = if transcript {
                    &mut snapshot["transcript"]["messages"]
                } else {
                    &mut snapshot["committed"]
                };
                field
                    .as_array_mut()
                    .ok_or_else(|| io::Error::other("Expected message list."))?
                    .push(serde_json::json!({"invalid excess message": true}));
                let error = original
                    .restore_workspace(&serde_json::to_vec(&saved).map_err(io::Error::other)?)
                    .err()
                    .ok_or_else(|| io::Error::other("Expected an early message limit."))?;
                assert!(error.to_string().contains("item limit"));
                assert_eq!(original.workspace_bytes()?, accepted);
            }
        }
        Ok(())
    }

    #[test]
    fn recovery_backpressure_skips_encoding_until_the_snapshot_slot_is_free() -> io::Result<()> {
        let mut state = state();
        let (store, controlled) = crate::recovery::Recovery::controlled();
        state.recovery = Some(store);
        state.save_conversations = true;
        state.recovery_dirty = true;
        state.last_recovery_save = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .unwrap_or_else(Instant::now);
        state.poll_recovery();
        assert!(!state.recovery_dirty);

        // Encoding this workspace would reject its saved-state limits. While
        // the first snapshot occupies the budget, no new encoding is attempted.
        state.last_answer = "x".repeat(4 * 1024 * 1024 + 1);
        state.recovery_dirty = true;
        state.last_recovery_save = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .unwrap_or_else(Instant::now);
        state.poll_recovery();
        assert!(state.recovery_dirty);
        assert!(!state.recovery_notice.starts_with("Could not save"));
        let saved = controlled.next_completion()?;
        controlled.complete(saved)?;
        state.last_recovery_save = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .unwrap_or_else(Instant::now);
        state.poll_recovery();
        assert!(state.recovery_notice.starts_with("Could not save"));
        assert!(state.recovery_dirty);
        state.save_conversations = false;
        drop(controlled);
        Ok(())
    }

    #[test]
    fn recovery_bounds_escaped_json_before_retaining_an_oversized_workspace() -> io::Result<()> {
        let mut original = state();
        // This is valid retained text, but its JSON escapes alone exceed the
        // recovery limit. The encoder must stop rather than build that body.
        original.last_answer = "\0".repeat(2 * 1024 * 1024);
        let error = original
            .workspace_bytes()
            .err()
            .ok_or_else(|| io::Error::other("Expected recovery byte rejection."))?;
        assert!(error.to_string().contains("8 MiB limit"));
        original.last_answer = "A recovered answer: 雨 🌱\n\"quoted\"".to_owned();
        let bytes = original.workspace_bytes()?;
        assert!(bytes.capacity() <= 8 * 1024 * 1024);
        let mut restored = state();
        restored.restore_workspace(&bytes)?;
        assert_eq!(restored.last_answer, original.last_answer);
        assert!(restored.committed.is_empty());
        assert!(!restored.model.busy() && !restored.speaking);
        Ok(())
    }

    #[test]
    fn streaming_read_aloud_recovery_preserves_preference_without_resuming_playback()
    -> io::Result<()> {
        let mut original = state();
        original.streaming_read_aloud = true;
        original.save_conversation();
        original.conversations.push(Conversation::new(2));
        original.active_conversation = 2;
        original.next_conversation = 3;
        original.clear_conversation();
        assert!(!original.streaming_read_aloud);
        original.streaming_read_aloud = true;
        let bytes = original.workspace_bytes()?;
        let mut restored = state();
        restored.restore_workspace(&bytes)?;
        assert!(restored.streaming_read_aloud);
        assert!(!restored.streaming_active && !restored.speaking);
        assert!(
            restored.conversations[0]
                .saved
                .as_ref()
                .is_some_and(|saved| saved.streaming_read_aloud)
        );

        let mut old: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        old["active"]
            .as_object_mut()
            .ok_or_else(|| io::Error::other("active object"))?
            .remove("streaming_read_aloud");
        old["conversations"][0]["saved"]
            .as_object_mut()
            .ok_or_else(|| io::Error::other("saved object"))?
            .remove("streaming_read_aloud");
        restored.restore_workspace(&serde_json::to_vec(&old).map_err(io::Error::other)?)?;
        assert!(!restored.streaming_read_aloud);
        assert!(
            restored.conversations[0]
                .saved
                .as_ref()
                .is_some_and(|saved| !saved.streaming_read_aloud)
        );
        Ok(())
    }

    #[test]
    fn thinking_recovery_preserves_each_conversation_and_defaults_old_workspaces_off()
    -> io::Result<()> {
        let mut original = state();
        assert!(!original.enable_thinking);
        original.enable_thinking = true;
        original.save_conversation();
        original.conversations.push(Conversation::new(2));
        original.active_conversation = 2;
        original.next_conversation = 3;
        original.clear_conversation();
        assert!(!original.enable_thinking);
        let bytes = original.workspace_bytes()?;
        let mut restored = state();
        restored.restore_workspace(&bytes)?;
        assert!(!restored.enable_thinking);
        assert!(
            restored.conversations[0]
                .saved
                .as_ref()
                .is_some_and(|saved| saved.enable_thinking)
        );

        original.enable_thinking = true;
        let bytes = original.workspace_bytes()?;
        restored.restore_workspace(&bytes)?;
        assert!(restored.enable_thinking);
        let mut old: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        old["active"]
            .as_object_mut()
            .ok_or_else(|| io::Error::other("active object"))?
            .remove("enable_thinking");
        old["conversations"][0]["saved"]
            .as_object_mut()
            .ok_or_else(|| io::Error::other("saved object"))?
            .remove("enable_thinking");
        restored.restore_workspace(&serde_json::to_vec(&old).map_err(io::Error::other)?)?;
        assert!(!restored.enable_thinking);
        assert!(
            restored.conversations[0]
                .saved
                .as_ref()
                .is_some_and(|saved| !saved.enable_thinking)
        );
        for invalid in [
            serde_json::json!(null),
            serde_json::json!("false"),
            serde_json::json!(0),
        ] {
            old["active"]["enable_thinking"] = invalid;
            assert!(
                restored
                    .restore_workspace(&serde_json::to_vec(&old).map_err(io::Error::other)?)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn daisyrose_recovery_ignores_legacy_abliteration_and_validates_scales() -> io::Result<()> {
        let mut original = state();
        let bytes = original.workspace_bytes()?;
        let mut with_legacy: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        with_legacy["active"]["abliteration_scale"] = serde_json::json!(1.7);
        let mut restored = state();
        restored.restore_workspace(&serde_json::to_vec(&with_legacy).map_err(io::Error::other)?)?;
        let re_saved: serde_json::Value =
            serde_json::from_slice(&restored.workspace_bytes()?).map_err(io::Error::other)?;
        assert!(re_saved["active"].get("abliteration_scale").is_none());
        original.daisyrose_enabled = true;
        original.daisyrose_scale = 0.5;
        let daisy_bytes = original.workspace_bytes()?;
        restored.restore_workspace(&daisy_bytes)?;
        assert!(restored.daisyrose_enabled);
        assert!((restored.daisyrose_scale - 0.5).abs() < f32::EPSILON);
        let mut legacy: serde_json::Value =
            serde_json::from_slice(&daisy_bytes).map_err(io::Error::other)?;
        let active_obj = legacy["active"]
            .as_object_mut()
            .ok_or_else(|| io::Error::other("active object"))?;
        active_obj.remove("daisyrose_enabled");
        active_obj.remove("daisyrose_scale");
        restored.restore_workspace(&serde_json::to_vec(&legacy).map_err(io::Error::other)?)?;
        assert!(!restored.daisyrose_enabled);
        assert!((restored.daisyrose_scale - client::DEFAULT_DAISYROSE_SCALE).abs() < f32::EPSILON);
        for invalid in [
            serde_json::json!(-0.1),
            serde_json::json!(2.1),
            serde_json::json!(null),
        ] {
            legacy["active"]["daisyrose_scale"] = invalid;
            assert!(
                restored
                    .restore_workspace(&serde_json::to_vec(&legacy).map_err(io::Error::other)?)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn recovery_preserves_drafts_without_promoting_partial_turns() -> io::Result<()> {
        let mut original = state();
        original.input = "Unsent draft 🌱".to_owned();
        original
            .transcript
            .push(Role::User, "Interrupted request".to_owned());
        original.transcript.push(Role::Assistant, String::new());
        original.transcript.append("Partial answer", false);
        let bytes = original.workspace_bytes()?;
        let mut restored = state();
        restored.restore_workspace(&bytes)?;
        assert_eq!(restored.input, original.input);
        assert_eq!(
            restored.transcript.plain_text(),
            original.transcript.plain_text()
        );
        assert!(restored.committed.is_empty());
        assert!(restored.restore_history);
        assert!(!restored.model.busy());
        assert!(!restored.speech_input);
        assert!(!restored.speaking);
        let mut invalid: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        invalid["active"]["output_budget"] = serde_json::json!(99);
        assert!(
            restored
                .restore_workspace(&serde_json::to_vec(&invalid).map_err(io::Error::other)?)
                .is_err()
        );
        invalid["active"]["output_budget"] = serde_json::json!(3);
        invalid["version"] = serde_json::json!(2);
        assert!(
            restored
                .restore_workspace(&serde_json::to_vec(&invalid).map_err(io::Error::other)?)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn recovery_rapid_toggles_ignore_stale_erase_and_save_acknowledgments() -> io::Result<()> {
        let mut state = state();
        let (store, controlled) = crate::recovery::Recovery::controlled();
        state.recovery = Some(store);
        state.recovery_erase_pending = true;
        state.last_recovery_save = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .unwrap_or_else(Instant::now);
        state.poll_recovery();
        let first_erase = controlled.next_completion()?;
        assert!(matches!(first_erase, crate::recovery::Event::Erased(_)));

        // Enable saving while the first deletion's acknowledgment is delayed.
        state.save_conversations = true;
        state.recovery_erase_pending = false;
        state.recovery_dirty = true;
        state.last_recovery_save = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .unwrap_or_else(Instant::now);
        state.poll_recovery();
        let later_save = controlled.next_completion()?;
        assert!(matches!(later_save, crate::recovery::Event::Saved(_)));

        // Disable again before that queued save completes. The old erase must
        // not clear the new deletion intention, even during window shutdown.
        state.save_conversations = false;
        state.recovery_erase_pending = true;
        controlled.complete(first_erase)?;
        state.poll_recovery();
        assert!(state.recovery_erase_pending);
        state.last_recovery_save = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .unwrap_or_else(Instant::now);
        state.poll_recovery();
        let final_erase = controlled.next_completion()?;
        assert!(matches!(final_erase, crate::recovery::Event::Erased(_)));
        controlled.complete(later_save)?;
        state.poll_recovery();
        assert!(state.recovery_erase_pending);
        controlled.complete(final_erase)?;
        state.poll_recovery();
        assert!(!state.recovery_erase_pending);
        assert!(!state.save_conversations);
        assert!(state.recovery_notice.contains("removed"));
        drop(controlled);
        Ok(())
    }

    #[test]
    fn recovery_invalid_live_state_preserves_last_valid_encrypted_snapshot() -> io::Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("Bonsai-recovery-bounds-{stamp}"));
        std::fs::create_dir(&directory)?;
        let mut state = state();
        state.attach_recovery(crate::recovery::Recovery::open(&directory));
        state.save_conversations = true;
        state.recovery_dirty = true;
        state.input = "Last valid draft".to_owned();
        state.last_recovery_save = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .unwrap_or_else(Instant::now);
        state.poll_recovery();
        let recovery = state
            .recovery
            .as_ref()
            .ok_or_else(|| io::Error::other("Recovery did not open."))?;
        assert!(matches!(
            recovery.events.recv_timeout(Duration::from_secs(5)),
            Ok(crate::recovery::Event::Saved(_))
        ));
        let path = directory.join("chat-recovery.dpapi");
        let valid = std::fs::read(&path)?;

        // Normal retry-heavy transcripts evict their oldest messages and remain
        // recoverable at the live message limit.
        for _ in 0..=MAX_MESSAGES {
            state
                .transcript
                .push(Role::User, "Rejected attempt".to_owned());
        }
        assert_eq!(state.transcript.messages.len(), MAX_MESSAGES);
        let mut bounded = self::state();
        bounded.restore_workspace(&state.workspace_bytes()?)?;
        assert_eq!(
            bounded.transcript.plain_text(),
            state.transcript.plain_text()
        );
        assert!(bounded.committed.is_empty());

        // Deliberately bypass live eviction to exercise the invalid-state guard.
        // Never replace the last valid snapshot with a state restore rejects.
        let mut excess = Transcript::default();
        excess.push(Role::User, "Excess message".to_owned());
        state.transcript.messages.extend(excess.messages);
        assert!(state.workspace_bytes().is_err());
        state.recovery_dirty = true;
        state.last_recovery_save = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .unwrap_or_else(Instant::now);
        state.poll_recovery();
        assert!(state.recovery_notice.starts_with("Could not save"));
        state.model = ModelLifecycle::Ready;
        state.new_chat();
        assert!(state.transcript.messages.is_empty());
        assert!(state.workspace_bytes().is_err());
        drop(state);
        assert_eq!(std::fs::read(&path)?, valid);
        let plain = crate::scrub::ScrubbingBytes::new(
            bonsai_windows_credentials::unprotect_private_data(&valid)?,
        );
        let mut recovered = self::state();
        recovered.restore_workspace(&plain)?;
        assert_eq!(recovered.input, "Last valid draft");
        remove_recovery_directory(&directory)?;
        Ok(())
    }

    #[test]
    fn multimodal_attachments_and_legacy_v1_workspaces_roundtrip_cleanly() -> io::Result<()> {
        use crate::attachment_import::{Attachment, PrehashedPng, VideoFrame, format_user_turn};
        use std::sync::Arc;

        let mut original = state();
        let draft_text = Attachment::Text {
            name: "notes.txt".to_owned(),
            text: "draft notes".to_owned(),
        };
        let committed_image = Attachment::Image {
            name: "diagram.png".to_owned(),
            width: 64,
            height: 64,
            data_url: Arc::from("data:image/png;base64,AAAA"),
            thumbnail_png: PrehashedPng::from([1_u8, 2, 3]),
        };
        let committed_video = Attachment::Video {
            name: "clip.mp4".to_owned(),
            width: 32,
            height: 24,
            duration_ms: 1_500,
            frames: Arc::from([VideoFrame {
                timestamp_ms: 500,
                data_url: Arc::from("data:image/png;base64,BBBB"),
            }]),
            thumbnail_png: PrehashedPng::from([4_u8, 5]),
        };
        let attachments = vec![committed_image.clone(), committed_video.clone()];
        let (wire_text, wire_images) = format_user_turn("Analyze these", &attachments);
        original.transcript.start_turn_with_attachments(
            "Analyze these".to_owned(),
            attachments.clone(),
            0,
        );
        original.transcript.append("Both look clear.", false);
        let history = vec![
            client::Message {
                role: "user",
                content: wire_text.into(),
                reasoning_content: None,
                images: wire_images.clone(),
            },
            client::Message {
                role: "assistant",
                content: "Both look clear.".into(),
                reasoning_content: None,
                images: Arc::default(),
            },
        ];
        original.model_event(client::Event::Committed(history));
        original.model_event(client::Event::Complete(client::Completion::Finished));
        original.input = "Follow up".to_owned();
        original.set_draft_attachments(vec![draft_text.clone()]);

        let bytes = original.workspace_bytes()?;
        let mut restored = state();
        restored.restore_workspace(&bytes)?;
        assert_eq!(restored.draft_attachments, vec![draft_text]);
        assert_eq!(restored.committed.len(), 2);
        assert_eq!(restored.committed[0].images.as_ref(), wire_images.as_ref());
        assert_eq!(restored.transcript.messages[0].body, "Analyze these");
        assert_eq!(restored.transcript.messages[0].attachments, attachments);
        assert_eq!(
            restored.transcript.messages[0].committed_index(&restored.committed),
            Some(0)
        );

        // Legacy v1 JSON without `draft_attachments`, `attachments`, or `images` must load cleanly.
        let mut legacy_json: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        if let Some(active) = legacy_json["active"].as_object_mut() {
            active.remove("draft_attachments");
            active["committed"] = serde_json::json!([
                {"role": "user", "content": "Legacy prompt"},
                {"role": "assistant", "content": "Legacy answer", "reasoning_content": null}
            ]);
            active["transcript"] = serde_json::json!({
                "messages": [
                    {"role": "User", "body": "Legacy prompt", "reasoning": "", "history_index": 0},
                    {"role": "Assistant", "body": "Legacy answer", "reasoning": "", "history_index": 1}
                ]
            });
        }
        let legacy_bytes = serde_json::to_vec(&legacy_json).map_err(io::Error::other)?;
        let mut legacy_restored = state();
        legacy_restored.restore_workspace(&legacy_bytes)?;
        assert!(legacy_restored.draft_attachments.is_empty());
        assert!(legacy_restored.committed[0].images.is_empty());
        assert!(
            legacy_restored.transcript.messages[0]
                .attachments
                .is_empty()
        );
        assert_eq!(
            legacy_restored.cognitive,
            client::CognitiveOptions::default()
        );
        assert!(legacy_restored.memory.is_default_empty());

        let mut custom = state();
        custom.cognitive = client::CognitiveOptions {
            reasoning_effort: client::ReasoningEffort::Medium,
            thinking_budget: client::ThinkingBudgetMode::Balanced,
            sampling_preset: client::SamplingPreset::BalancedDry,
            extra_effort_strategy: client::ExtraEffortStrategy::MultiDraftSynthesis,
        };
        custom.memory.persistent_instructions = "Always cite sources.".to_owned();
        custom
            .memory
            .add_card("Style", "code, rust", "Prefer small pure helpers.", false)
            .map_err(io::Error::other)?;
        let saved_custom = custom.workspace_bytes()?;
        let mut restored_custom = state();
        restored_custom.restore_workspace(&saved_custom)?;
        assert_eq!(restored_custom.cognitive, custom.cognitive);
        assert_eq!(restored_custom.memory, custom.memory);

        let mut legacy_no_cog: serde_json::Value =
            serde_json::from_slice(&saved_custom).map_err(io::Error::other)?;
        if let Some(root) = legacy_no_cog.as_object_mut() {
            root.remove("memory");
            if let Some(active) = root["active"].as_object_mut() {
                active.remove("cognitive");
            }
        }
        let mut legacy_loaded = state();
        legacy_loaded
            .restore_workspace(&serde_json::to_vec(&legacy_no_cog).map_err(io::Error::other)?)?;
        assert!(legacy_loaded.memory.is_default_empty());
        assert_eq!(legacy_loaded.cognitive, client::CognitiveOptions::default());
        Ok(())
    }

    #[test]
    fn recovery_throttles_active_generation_and_flushes_immediately_on_completion() -> io::Result<()>
    {
        let mut state = state();
        let (store, controlled) = crate::recovery::Recovery::controlled();
        state.recovery = Some(store);
        state.preview = false;
        state.save_conversations = true;
        state.model = ModelLifecycle::Generating;
        state.recovery_dirty = true;
        state.last_recovery_save = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .unwrap_or_else(Instant::now);
        // Active generation throttles saves to 15s, so 3s elapsed skips serialization.
        state.poll_recovery();
        assert!(state.recovery_dirty);

        // 16s elapsed during generation still saves periodically.
        state.last_recovery_save = Instant::now()
            .checked_sub(Duration::from_secs(16))
            .unwrap_or_else(Instant::now);
        state.poll_recovery();
        assert!(!state.recovery_dirty);
        let saved = controlled.next_completion()?;
        controlled.complete(saved)?;

        // Completion triggers an immediate flush even within the idle 2s window.
        state.recovery_dirty = true;
        state.last_recovery_save = Instant::now();
        state.model_event(client::Event::Complete(client::Completion::Finished));
        state.poll_recovery();
        assert!(!state.recovery_dirty);
        let completed_save = controlled.next_completion()?;
        controlled.complete(completed_save)?;
        drop(controlled);
        Ok(())
    }

    #[test]
    fn recovery_worker_disconnect_surfaces_alert_and_stops_reserializing() {
        let mut state = state();
        let (store, controlled) = crate::recovery::Recovery::controlled();
        state.recovery = Some(store);
        state.save_conversations = true;
        state.recovery_dirty = true;
        drop(controlled);

        state.poll_recovery();
        assert!(state.recovery.is_none());
        assert!(!state.recovery_dirty);
        assert!(!state.recovery_erase_pending);
        assert!(state.recovery_notice.starts_with("Could not"));
        assert_eq!(
            state.active_alerts(),
            vec![crate::window::error_ui::AlertKind::Recovery]
        );
    }

    #[test]
    fn auto_continue_workspace_round_trip_legacy_default_and_mid_continuation_crash()
    -> io::Result<()> {
        let mut s = state();
        s.auto_continue = false;
        s.new_chat();
        s.auto_continue = true;
        s.new_chat();
        s.auto_continue = false;

        let bytes = s.workspace_bytes()?;
        let mut restored = state();
        restored.restore_workspace(&bytes)?;
        assert!(!restored.auto_continue);
        assert_eq!(restored.conversations.len(), 3);
        let middle_id = restored.conversations[1].id;
        restored.switch_conversation(middle_id);
        assert!(restored.auto_continue);

        // Legacy workspace omitting "auto_continue" in active and saved defaults to true
        let mut json: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        if let Some(active) = json["active"].as_object_mut() {
            active.remove("auto_continue");
        }
        if let Some(convs) = json["conversations"].as_array_mut() {
            for conv in convs {
                if let Some(saved) = conv["saved"].as_object_mut() {
                    saved.remove("auto_continue");
                }
            }
        }
        let mut legacy = state();
        legacy.auto_continue = false;
        legacy.restore_workspace(&serde_json::to_vec(&json).map_err(io::Error::other)?)?;
        assert!(legacy.auto_continue);
        for conv in &legacy.conversations {
            if let Some(saved) = &conv.saved {
                assert!(saved.auto_continue);
            }
        }

        // Invalid types for auto_continue fail cleanly without clobbering live state
        for bad in [
            serde_json::Value::Null,
            serde_json::json!("true"),
            serde_json::json!(1),
        ] {
            let mut malformed =
                serde_json::from_slice::<serde_json::Value>(&bytes).map_err(io::Error::other)?;
            malformed["active"]["auto_continue"] = bad;
            let live_before = restored.active_conversation;
            let err = restored
                .restore_workspace(&serde_json::to_vec(&malformed).map_err(io::Error::other)?);
            assert!(err.is_err());
            assert_eq!(restored.active_conversation, live_before);
        }

        // Crash recovery mid-continuation adopts a single stitched partial answer
        let (mid_commands, _mid_cmd_rx) = mpsc::channel();
        let (_mid_events, mid_incoming) = client::event_channel();
        let mut mid = State::create(mid_commands, mid_incoming, None, None);
        mid.preview = false;
        mid.model = ModelLifecycle::Ready;
        mid.context_tokens = Some(65_536);
        mid.input = "Mid-continuation prompt".to_owned();
        mid.submit();
        mid.model_event(client::Event::Text("Segment one.".to_owned()));
        mid.model_event(client::Event::Continuing(2));
        mid.model_event(client::Event::Text(" Segment two.".to_owned()));
        let mid_bytes = mid.workspace_bytes()?;

        let mut recovered = state();
        recovered.restore_workspace(&mid_bytes)?;
        let last = recovered
            .transcript
            .messages
            .last()
            .ok_or_else(|| io::Error::other("missing stitched partial answer"))?;
        assert_eq!(last.body, "Segment one. Segment two.");
        assert!(last.interrupted);
        assert!(last.can_continue(&recovered.committed));
        Ok(())
    }

    #[test]
    fn poll_recovery_handles_erase_disconnect_and_busy_save_status() -> io::Result<()> {
        // Busy save keeps recovery_dirty = true without disconnecting.
        let mut s = state();
        let (store, controlled) = crate::recovery::Recovery::controlled();
        assert_eq!(store.erase(), crate::recovery::DispatchStatus::Queued);
        s.recovery = Some(store);
        s.save_conversations = true;
        s.recovery_dirty = true;
        s.recovery_flush_now = true;
        // Occupy capacity-1 channel or fill via queued save? `save_with` uses TrySendError::Full -> Busy.
        s.poll_recovery();
        assert!(s.recovery.is_some());
        assert!(s.recovery_dirty);

        // Drain queued command and drop worker -> Disconnected on save_with triggers disconnect_recovery.
        let _ = controlled.next_completion()?;
        drop(controlled);
        s.recovery_flush_now = true;
        s.poll_recovery();
        assert!(s.recovery.is_none());
        assert!(!s.recovery_dirty);

        // Disconnected on erase() also calls disconnect_recovery.
        let mut s_erase = state();
        let (erase_store, erase_ctrl) = crate::recovery::Recovery::controlled();
        assert_eq!(erase_store.erase(), crate::recovery::DispatchStatus::Queued);
        s_erase.recovery = Some(erase_store);
        s_erase.recovery_erase_pending = true;
        let _ = erase_ctrl.next_completion()?;
        drop(erase_ctrl);
        s_erase.recovery_flush_now = true;
        s_erase.poll_recovery();
        assert!(s_erase.recovery.is_none());
        assert!(!s_erase.recovery_erase_pending);
        assert!(s_erase.recovery_notice.starts_with("Could not"));
        Ok(())
    }

    #[test]
    fn corrupt_primary_restores_from_backup_and_quarantines_unreadable_file() -> io::Result<()> {
        let directory = recovery_directory("backup-restore")?;
        let mut first = state();
        first.attach_recovery(crate::recovery::Recovery::open(&directory));
        first.input = "Saved backup draft".to_owned();
        first
            .transcript
            .push(Role::User, "Remembered prompt".to_owned());
        drop(first);

        let mut second = state();
        second.attach_recovery(crate::recovery::Recovery::open(&directory));
        second.input = "Newer primary draft".to_owned();
        drop(second);

        // Corrupt primary after two saves so backup holds "Saved backup draft".
        let primary = directory.join("chat-recovery.dpapi");
        let preserved = directory.join("chat-recovery.preserved.dpapi");
        std::fs::write(&primary, b"corrupt-primary")?;

        let mut recovered = state();
        recovered.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert!(recovered.save_conversations);
        assert!(recovered.recovery_quarantine_primary);
        assert_eq!(recovered.input, "Saved backup draft");
        assert!(
            recovered
                .transcript
                .plain_text()
                .contains("Remembered prompt")
        );
        assert!(recovered.recovery_notice.contains("backup"));
        drop(recovered);

        assert_eq!(std::fs::read(&preserved)?, b"corrupt-primary");
        let mut reopened = state();
        reopened.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert_eq!(reopened.input, "Saved backup draft");
        drop(reopened);

        remove_recovery_directory(&directory)
    }

    #[test]
    fn corrupt_primary_without_backup_preserves_failure_until_explicit_save_then_quarantines()
    -> io::Result<()> {
        let directory = recovery_directory("quarantine-no-backup")?;
        let primary = directory.join("chat-recovery.dpapi");
        let preserved = directory.join("chat-recovery.preserved.dpapi");
        std::fs::write(&primary, b"unreadable-without-backup")?;

        let mut failed = state();
        failed.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert!(!failed.save_conversations);
        assert!(failed.recovery_quarantine_primary);
        assert!(failed.recovery_notice.starts_with("Could not"));
        drop(failed);
        assert_eq!(std::fs::read(&primary)?, b"unreadable-without-backup");
        assert!(!preserved.exists());

        let mut replaced = state();
        replaced.attach_recovery(crate::recovery::Recovery::open(&directory));
        replaced.save_conversations = true;
        replaced.recovery_dirty = true;
        replaced.input = "Fresh workspace".to_owned();
        drop(replaced);

        assert_eq!(std::fs::read(&preserved)?, b"unreadable-without-backup");
        let mut reopened = state();
        reopened.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert_eq!(reopened.input, "Fresh workspace");
        drop(reopened);

        remove_recovery_directory(&directory)
    }

    #[test]
    fn salvage_recovers_newer_schema_fields_and_quarantines_original_file() -> io::Result<()> {
        let mut original = state();
        original.input = "Draft from future schema".to_owned();
        original.transcript.push(Role::User, "Hello".to_owned());
        original
            .transcript
            .push(Role::Assistant, "Hi there".to_owned());
        original.committed = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]))
        .map_err(io::Error::other)?;
        let bytes = original.workspace_bytes()?;
        let mut future: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        future["future_root_field"] = serde_json::json!("new");
        future["active"]["future_snapshot_field"] = serde_json::json!(42);
        future["active"]["cognitive"]["future_cog_field"] = serde_json::json!(true);
        future["active"]["transcript"]["messages"][0]["future_msg_field"] =
            serde_json::json!("meta");
        future["active"]["committed"][0]["future_wire_field"] = serde_json::json!(123);
        let future_bytes = serde_json::to_vec(&future).map_err(io::Error::other)?;

        // Strict restore still rejects unknown fields.
        let mut strict = state();
        assert!(strict.restore_workspace(&future_bytes).is_err());

        // Opening via attach_recovery salvages 100% of conversations and quarantines original bytes.
        let directory = recovery_directory("salvage-future")?;
        let primary = directory.join("chat-recovery.dpapi");
        let preserved = directory.join("chat-recovery.preserved.dpapi");
        let sealed = bonsai_windows_credentials::protect_private_data(&future_bytes)?;
        std::fs::write(&primary, &sealed)?;

        let mut recovered = state();
        recovered.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert!(recovered.save_conversations);
        assert!(recovered.recovery_quarantine_primary);
        assert_eq!(recovered.input, "Draft from future schema");
        assert_eq!(recovered.committed.len(), 2);
        assert!(recovered.transcript.plain_text().contains("Hi there"));
        drop(recovered);

        assert_eq!(std::fs::read(&preserved)?, sealed);
        remove_recovery_directory(&directory)
    }

    #[test]
    fn salvage_recovers_healthy_conversations_around_partial_corruption_and_rejects_empty()
    -> io::Result<()> {
        let mut original = state();
        original
            .transcript
            .push(Role::User, "First chat".to_owned());
        original.new_chat();
        original
            .transcript
            .push(Role::User, "Second chat".to_owned());
        let bytes = original.workspace_bytes()?;
        let mut damaged: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(io::Error::other)?;

        // Corrupt conversation[0] (active) and give conversation[1] an odd committed tail and bad budget.
        damaged["active"] = serde_json::json!({"corrupted": true});
        damaged["conversations"][1]["saved"]["output_budget"] = serde_json::json!(999);
        damaged["conversations"][1]["saved"]["committed"] = serde_json::json!([
            {"role": "user", "content": "First chat"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Unpaired trailing turn"}
        ]);

        let mut salvaged = state();
        let report =
            salvaged.salvage_workspace(&serde_json::to_vec(&damaged).map_err(io::Error::other)?)?;
        assert_eq!(report.recovered_conversations, 1);
        assert_eq!(report.dropped_conversations, 1);
        assert!(salvaged.transcript.plain_text().contains("First chat"));
        assert_eq!(salvaged.committed.len(), 2);
        assert_eq!(salvaged.output_budget, 3);

        // Empty objects and unsupported versions must still be rejected.
        let mut rejected = state();
        assert!(rejected.salvage_workspace(b"{}").is_err());
        let mut wrong_version: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        wrong_version["version"] = serde_json::json!(2);
        assert!(
            rejected
                .salvage_workspace(&serde_json::to_vec(&wrong_version).map_err(io::Error::other)?)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn undo_delete_restores_active_inactive_and_last_conversation_and_sets_immediate_flush()
    -> io::Result<()> {
        // 1. Deleting the sole (last) conversation and undoing replaces the empty placeholder.
        let mut s = state();
        s.input = "Keep this unsent draft".to_owned();
        s.transcript.push(Role::User, "Sole question".to_owned());
        s.transcript.push(Role::Assistant, "Sole answer".to_owned());
        s.committed = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "Sole question"},
            {"role": "assistant", "content": "Sole answer"},
        ]))
        .map_err(io::Error::other)?;
        let sole_id = s.active_conversation;
        s.conversations[0].title = "Sole title".to_owned();
        s.conversations[0].renamed = true;
        s.recovery_flush_now = false;
        s.remove_conversation(sole_id);
        assert!(s.recovery_flush_now);
        assert_ne!(s.active_conversation, sole_id);
        assert_eq!(s.conversations.len(), 1);
        assert!(s.transcript.messages.is_empty());

        s.recovery_flush_now = false;
        assert!(s.undo_delete_conversation());
        assert!(s.recovery_flush_now);
        assert_eq!(s.active_conversation, sole_id);
        assert_eq!(s.conversations.len(), 1);
        assert_eq!(s.conversations[0].title, "Sole title");
        assert_eq!(s.input, "Keep this unsent draft");
        assert_eq!(s.committed.len(), 2);
        assert!(s.transcript.plain_text().contains("Sole answer"));
        assert!(s.conversation_notice.is_none());

        // 2. Deleting an inactive conversation and undoing restores it at its original list index.
        s.recovery_flush_now = false;
        s.new_chat();
        assert!(s.recovery_flush_now);
        let second_id = s.active_conversation;
        s.transcript.push(Role::User, "Second".to_owned());
        s.recovery_flush_now = false;
        s.remove_conversation(sole_id);
        assert!(s.recovery_flush_now);
        assert_eq!(s.conversations.len(), 1);
        assert!(s.undo_delete_conversation());
        assert_eq!(s.conversations.len(), 2);
        assert_eq!(s.active_conversation, sole_id);
        assert_eq!(s.conversations[1].id, sole_id);
        assert_eq!(s.conversations[0].id, second_id);

        // 3. Undo is one-shot once consumed.
        assert!(!s.undo_delete_conversation());
        Ok(())
    }
}
