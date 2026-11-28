//! Atomic native editor transactions and byte-bounded inverse-operation undo history.
use super::egui;
use egui::{
    TextBuffer,
    text::{CCursorRange, CharIndex},
    text_edit::{TextEditOutput, TextEditState},
};
use std::{
    any::TypeId,
    collections::VecDeque,
    ops::Range,
    sync::{Arc, Mutex},
};

type TextEditUndoer = egui::util::undoer::Undoer<(CCursorRange, String)>;

/// Maximum combined heap payload retained across committed undo and redo groups (1 MiB).
const MAX_RETAINED_UNDO_BYTES: usize = 1_048_576;
/// Maximum number of retained inverse-operation undo groups.
const MAX_UNDO_GROUPS: usize = 64;
/// Independent per-frame transaction scratch cap protecting rollback memory.
const MAX_TX_SCRATCH_BYTES: usize = super::INPUT_LIMIT * 4 + 4_096;
/// Contiguous single-step edits within this window coalesce into one undo group.
const COALESCE_WINDOW_SECS: f64 = 1.0;
/// Maximum delta size permitted to coalesce into a prior typing group.
const COALESCE_MAX_GROUP_BYTES: usize = 4_096;

pub(super) const LIMIT_MESSAGE: &str = "This edit exceeds the 1,048,576-character draft limit. Your draft was kept. Paste a smaller passage.";

fn apply_insert(target: &mut String, byte_idx: usize, text: &str) -> bool {
    if byte_idx <= target.len() && target.is_char_boundary(byte_idx) {
        target.insert_str(byte_idx, text);
        true
    } else {
        false
    }
}

fn apply_delete(target: &mut String, byte_idx: usize, byte_len: usize) -> bool {
    let end = byte_idx.saturating_add(byte_len);
    if end <= target.len() && target.is_char_boundary(byte_idx) && target.is_char_boundary(end) {
        target.drain(byte_idx..end);
        true
    } else {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum EditOp {
    Insert { byte_idx: usize, text: String },
    Delete { byte_idx: usize, text: String },
}

impl EditOp {
    fn apply_forward(&self, target: &mut String) -> bool {
        match self {
            Self::Insert { byte_idx, text } => apply_insert(target, *byte_idx, text),
            Self::Delete { byte_idx, text } => apply_delete(target, *byte_idx, text.len()),
        }
    }

    fn apply_inverse(&self, target: &mut String) -> bool {
        match self {
            Self::Insert { byte_idx, text } => apply_delete(target, *byte_idx, text.len()),
            Self::Delete { byte_idx, text } => apply_insert(target, *byte_idx, text),
        }
    }

    const fn byte_cost(&self) -> usize {
        match self {
            Self::Insert { text, .. } | Self::Delete { text, .. } => text.len(),
        }
    }

    fn net_char_delta(&self) -> isize {
        match self {
            Self::Insert { text, .. } => {
                isize::try_from(text.chars().count()).unwrap_or(isize::MAX)
            }
            Self::Delete { text, .. } => {
                let count = isize::try_from(text.chars().count()).unwrap_or(isize::MAX);
                count.saturating_neg()
            }
        }
    }

    fn net_byte_delta(&self) -> isize {
        match self {
            Self::Insert { text, .. } => isize::try_from(text.len()).unwrap_or(isize::MAX),
            Self::Delete { text, .. } => {
                let count = isize::try_from(text.len()).unwrap_or(isize::MAX);
                count.saturating_neg()
            }
        }
    }
}

#[derive(Clone, Debug)]
struct UndoGroup {
    ops: Vec<EditOp>,
    before_cursor: Option<CCursorRange>,
    after_cursor: Option<CCursorRange>,
    timestamp: f64,
    byte_cost: usize,
    coalesceable: bool,
}

impl UndoGroup {
    fn preview_dims(
        &self,
        undo: bool,
        cur_chars: usize,
        cur_bytes: usize,
    ) -> Option<(usize, usize)> {
        let mut chars = isize::try_from(cur_chars).ok()?;
        let mut bytes = isize::try_from(cur_bytes).ok()?;
        for op in &self.ops {
            let dc = op.net_char_delta();
            let db = op.net_byte_delta();
            if undo {
                chars = chars.checked_sub(dc)?;
                bytes = bytes.checked_sub(db)?;
            } else {
                chars = chars.checked_add(dc)?;
                bytes = bytes.checked_add(db)?;
            }
        }
        Some((usize::try_from(chars).ok()?, usize::try_from(bytes).ok()?))
    }

    fn try_coalesce(&mut self, next: &Self) -> bool {
        if !self.coalesceable
            || !next.coalesceable
            || self.ops.len() != 1
            || next.ops.len() != 1
            || (next.timestamp - self.timestamp).abs() > COALESCE_WINDOW_SECS
            || self.byte_cost.saturating_add(next.byte_cost) > COALESCE_MAX_GROUP_BYTES
        {
            return false;
        }
        match (&mut self.ops[0], &next.ops[0]) {
            (
                EditOp::Insert {
                    byte_idx: prev_idx,
                    text: prev_text,
                },
                EditOp::Insert {
                    byte_idx: next_idx,
                    text: next_text,
                },
            ) if !next_text.contains('\n')
                && !prev_text.contains('\n')
                && prev_idx.saturating_add(prev_text.len()) == *next_idx =>
            {
                prev_text.push_str(next_text);
                self.after_cursor = next.after_cursor;
                self.timestamp = next.timestamp;
                self.byte_cost = self.byte_cost.saturating_add(next.byte_cost);
                true
            }
            (
                EditOp::Delete {
                    byte_idx: prev_idx,
                    text: prev_text,
                },
                EditOp::Delete {
                    byte_idx: next_idx,
                    text: next_text,
                },
            ) if !next_text.contains('\n')
                && !prev_text.contains('\n')
                && next_idx.saturating_add(next_text.len()) == *prev_idx =>
            {
                let mut combined =
                    String::with_capacity(next_text.len().saturating_add(prev_text.len()));
                combined.push_str(next_text);
                combined.push_str(prev_text);
                *prev_idx = *next_idx;
                *prev_text = combined;
                self.after_cursor = next.after_cursor;
                self.timestamp = next.timestamp;
                self.byte_cost = self.byte_cost.saturating_add(next.byte_cost);
                true
            }
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TextFingerprint {
    byte_len: usize,
    prefix: [u8; 32],
    prefix_len: usize,
    mid: [u8; 32],
    mid_len: usize,
    suffix: [u8; 32],
    suffix_len: usize,
    char_count: usize,
}

impl TextFingerprint {
    fn from_text(text: &str, char_count: usize) -> Self {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut prefix = [0_u8; 32];
        let prefix_len = len.min(32);
        prefix[..prefix_len].copy_from_slice(&bytes[..prefix_len]);

        let mut mid = [0_u8; 32];
        let mid_start = (len / 2).saturating_sub(16);
        let mid_len = len.saturating_sub(mid_start).min(32);
        mid[..mid_len].copy_from_slice(&bytes[mid_start..mid_start + mid_len]);

        let mut suffix = [0_u8; 32];
        let suffix_start = len.saturating_sub(32);
        let suffix_len = len.saturating_sub(suffix_start);
        suffix[..suffix_len].copy_from_slice(&bytes[suffix_start..]);

        Self {
            byte_len: len,
            prefix,
            prefix_len,
            mid,
            mid_len,
            suffix,
            suffix_len,
            char_count,
        }
    }

    fn matches_text(&self, text: &str) -> bool {
        let bytes = text.as_bytes();
        let len = bytes.len();
        if self.byte_len != len {
            return false;
        }
        let prefix_len = len.min(32);
        let mid_start = (len / 2).saturating_sub(16);
        let mid_len = len.saturating_sub(mid_start).min(32);
        let suffix_start = len.saturating_sub(32);
        self.prefix_len == prefix_len
            && self.prefix[..prefix_len] == bytes[..prefix_len]
            && self.mid_len == mid_len
            && self.mid[..mid_len] == bytes[mid_start..mid_start + mid_len]
            && self.suffix_len == len.saturating_sub(suffix_start)
            && self.suffix[..self.suffix_len] == bytes[suffix_start..]
    }
}

#[derive(Default)]
struct DraftHistory {
    undo_stack: VecDeque<UndoGroup>,
    redo_stack: Vec<UndoGroup>,
    retained_bytes: usize,
    #[cfg(test)]
    max_retained_bytes: Option<usize>,
    ime_active: bool,
    ime_pending: Option<UndoGroup>,
    fingerprint: Option<TextFingerprint>,
}

impl DraftHistory {
    #[cfg(test)]
    fn with_budget(max_retained_bytes: usize) -> Self {
        Self {
            max_retained_bytes: Some(max_retained_bytes),
            ..Self::default()
        }
    }

    fn max_retained_bytes(&self) -> usize {
        #[cfg(test)]
        {
            self.max_retained_bytes.unwrap_or(MAX_RETAINED_UNDO_BYTES)
        }
        #[cfg(not(test))]
        {
            MAX_RETAINED_UNDO_BYTES
        }
    }

    const fn is_empty(&self) -> bool {
        self.undo_stack.is_empty()
            && self.redo_stack.is_empty()
            && !self.ime_active
            && self.ime_pending.is_none()
    }

    fn clear_history(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.retained_bytes = 0;
        self.ime_active = false;
        self.ime_pending = None;
    }

    fn push_undo_group(&mut self, group: UndoGroup) {
        let budget = self.max_retained_bytes();
        for dropped in self.redo_stack.drain(..) {
            self.retained_bytes = self.retained_bytes.saturating_sub(dropped.byte_cost);
        }
        if group.byte_cost > budget {
            self.clear_history();
            return;
        }
        if let Some(last) = self.undo_stack.back_mut() {
            let prev_cost = last.byte_cost;
            if last.try_coalesce(&group) {
                let added = last.byte_cost.saturating_sub(prev_cost);
                self.retained_bytes = self.retained_bytes.saturating_add(added);
                self.enforce_budget();
                return;
            }
        }
        self.retained_bytes = self.retained_bytes.saturating_add(group.byte_cost);
        self.undo_stack.push_back(group);
        self.enforce_budget();
    }

    fn enforce_budget(&mut self) {
        let budget = self.max_retained_bytes();
        while self.undo_stack.len() > MAX_UNDO_GROUPS || self.retained_bytes > budget {
            if !self.redo_stack.is_empty() && self.retained_bytes > budget {
                let dropped = self.redo_stack.remove(0);
                self.retained_bytes = self.retained_bytes.saturating_sub(dropped.byte_cost);
                continue;
            }
            let Some(oldest) = self.undo_stack.pop_front() else {
                break;
            };
            self.retained_bytes = self.retained_bytes.saturating_sub(oldest.byte_cost);
        }
    }
}

enum TxOp {
    Insert {
        byte_idx: usize,
        byte_len: usize,
        copied: Option<String>,
    },
    Delete {
        byte_idx: usize,
        removed: String,
    },
}

impl TxOp {
    const fn scratch_bytes(&self) -> usize {
        match self {
            Self::Insert { copied, .. } => match copied {
                Some(text) => text.len(),
                None => 0,
            },
            Self::Delete { removed, .. } => removed.len(),
        }
    }

    fn rollback(&self, target: &mut String) {
        match self {
            Self::Insert {
                byte_idx, byte_len, ..
            } => {
                apply_delete(target, *byte_idx, *byte_len);
            }
            Self::Delete { byte_idx, removed } => {
                apply_insert(target, *byte_idx, removed);
            }
        }
    }

    fn into_edit_op(self) -> Option<EditOp> {
        match self {
            Self::Insert {
                byte_idx,
                copied: Some(text),
                ..
            } => Some(EditOp::Insert { byte_idx, text }),
            Self::Insert { copied: None, .. } => None,
            Self::Delete { byte_idx, removed } => Some(EditOp::Delete {
                byte_idx,
                text: removed,
            }),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum FrameInputMode {
    Idle,
    ImePreedit,
    ImeCommit,
    ImeCancel,
    NonCoalescingEdit,
    TypingOrDelete,
}

/// A `TextEdit` may delete the selection before inserting pasted or IME text.
/// Retain only per-frame inverse deltas so rejected oversized edits roll back
/// atomically without cloning the entire draft, while keeping committed undo
/// history within a strict 1 MiB byte budget across up to 64 groups.
pub(super) struct Buffer<'a> {
    text: &'a mut String,
    limit: usize,
    char_count: usize,
    previous: TextEditState,
    undoer: Option<TextEditUndoer>,
    history: Arc<Mutex<DraftHistory>>,
    tx_ops: Vec<TxOp>,
    tx_bytes: usize,
    frame_time: f64,
    input_mode: FrameInputMode,
    history_applied: bool,
    rejected: bool,
}

impl<'a> Buffer<'a> {
    pub(super) fn new(
        text: &'a mut String,
        ctx: &egui::Context,
        id: egui::Id,
        limit: usize,
    ) -> Self {
        let previous = TextEditState::load(ctx, id).unwrap_or_else(|| {
            let mut state = TextEditState::default();
            state.set_undoer(TextEditUndoer::with_settings(
                egui::util::undoer::Settings {
                    max_undos: 0,
                    ..Default::default()
                },
            ));
            state.clone().store(ctx, id);
            state
        });
        let history = ctx.data_mut(|data| {
            Arc::clone(
                data.get_temp_mut_or_insert_with(id.with("draft_history"), || {
                    Arc::new(Mutex::new(DraftHistory::default()))
                }),
            )
        });

        let (history_empty, char_count) = if let Ok(mut guard) = history.lock() {
            let chars = match &guard.fingerprint {
                Some(fp) if fp.matches_text(text) => fp.char_count,
                _ => {
                    guard.clear_history();
                    let counted = text.chars().count();
                    guard.fingerprint = Some(TextFingerprint::from_text(text, counted));
                    counted
                }
            };
            (guard.is_empty(), chars)
        } else {
            (true, text.chars().count())
        };

        let (has_mutating_event, frame_time, input_mode, history_action) =
            inspect_frame_events(ctx, id);
        let legacy_mode = history_empty && has_mutating_event && legacy_undoer_populated(&previous);

        let undoer = legacy_mode.then(|| previous.undoer());
        let mut buffer = Self {
            text,
            limit,
            char_count,
            previous,
            undoer,
            history,
            tx_ops: Vec::new(),
            tx_bytes: 0,
            frame_time,
            input_mode,
            // Preserve legacy TextEditUndoer ownership for external pre-populated
            // undoers so Undo/Redo steps do not split history into DraftHistory.
            history_applied: legacy_mode && history_action.is_some(),
            rejected: false,
        };

        if !legacy_mode
            && let Some(is_redo) = history_action
            && ctx.memory(|mem| mem.has_focus(id))
        {
            buffer.try_apply_history(ctx, id, is_redo);
        }

        buffer
    }

    pub(super) const fn was_mutated(&self) -> bool {
        !self.rejected && (!self.tx_ops.is_empty() || self.history_applied)
    }

    pub(super) fn finish(mut self, output: &mut TextEditOutput) -> bool {
        if self.rejected {
            self.rollback_tx();
            if let Some(undoer) = self.undoer.take() {
                self.previous.set_undoer(undoer);
            }
            output.cursor_range = self.previous.cursor.char_range();
            output.state = self.previous;
            output
                .state
                .clone()
                .store(&output.response.response.ctx, output.response.response.id);
            output.response.response.ctx.request_repaint();
            return true;
        }

        if !self.was_mutated()
            && !matches!(
                self.input_mode,
                FrameInputMode::ImeCommit | FrameInputMode::ImeCancel
            )
        {
            return false;
        }

        let history = Arc::clone(&self.history);
        if let Ok(mut guard) = history.lock() {
            if !self.tx_ops.is_empty() && !self.history_applied {
                self.commit_to_history(
                    &mut guard,
                    self.previous.cursor.char_range(),
                    output.cursor_range,
                );
            } else if self.input_mode == FrameInputMode::ImeCommit
                || self.input_mode == FrameInputMode::ImeCancel
            {
                guard.ime_active = false;
                if let Some(pending) = guard.ime_pending.take() {
                    guard.push_undo_group(pending);
                }
            }
            guard.fingerprint = Some(TextFingerprint::from_text(self.text, self.char_count));
        }
        false
    }

    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Used by draft buffer unit tests.")
    )]
    fn has_tx_changes(&self) -> bool {
        !self.tx_ops.is_empty()
    }

    fn rollback_tx(&mut self) {
        while let Some(op) = self.tx_ops.pop() {
            op.rollback(self.text);
        }
        self.tx_bytes = 0;
        self.char_count = self.text.chars().count();
    }

    fn try_record_op(&mut self, op: TxOp) -> bool {
        let added = op.scratch_bytes();
        if self.tx_bytes.saturating_add(added) > MAX_TX_SCRATCH_BYTES {
            self.rejected = true;
            return false;
        }
        self.tx_bytes = self.tx_bytes.saturating_add(added);
        self.tx_ops.push(op);
        true
    }

    fn commit_to_history(
        &mut self,
        guard: &mut DraftHistory,
        before_cursor: Option<CCursorRange>,
        after_cursor: Option<CCursorRange>,
    ) {
        let max_retained_bytes = guard.max_retained_bytes();
        let mut ops = Vec::with_capacity(self.tx_ops.len());
        let mut byte_cost = 0_usize;
        for tx in std::mem::take(&mut self.tx_ops) {
            let Some(edit) = tx.into_edit_op() else {
                guard.clear_history();
                return;
            };
            byte_cost = byte_cost.saturating_add(edit.byte_cost());
            if byte_cost > max_retained_bytes {
                guard.clear_history();
                return;
            }
            ops.push(edit);
        }
        if ops.is_empty() {
            return;
        }

        let coalesceable = self.input_mode == FrameInputMode::TypingOrDelete;
        let group = UndoGroup {
            ops,
            before_cursor,
            after_cursor,
            timestamp: self.frame_time,
            byte_cost,
            coalesceable,
        };

        match self.input_mode {
            FrameInputMode::ImePreedit => {
                guard.ime_active = true;
                if let Some(pending) = &mut guard.ime_pending {
                    let next_cost = pending.byte_cost.saturating_add(group.byte_cost);
                    if next_cost > max_retained_bytes {
                        guard.clear_history();
                    } else {
                        pending.ops.extend(group.ops);
                        pending.after_cursor = group.after_cursor;
                        pending.byte_cost = next_cost;
                        pending.timestamp = group.timestamp;
                    }
                } else {
                    guard.ime_pending = Some(UndoGroup {
                        coalesceable: false,
                        ..group
                    });
                }
            }
            FrameInputMode::ImeCommit | FrameInputMode::ImeCancel => {
                guard.ime_active = false;
                if let Some(mut pending) = guard.ime_pending.take() {
                    let next_cost = pending.byte_cost.saturating_add(group.byte_cost);
                    if next_cost > max_retained_bytes {
                        guard.clear_history();
                    } else {
                        pending.ops.extend(group.ops);
                        pending.after_cursor = group.after_cursor;
                        pending.byte_cost = next_cost;
                        pending.timestamp = group.timestamp;
                        guard.push_undo_group(pending);
                    }
                } else {
                    guard.push_undo_group(UndoGroup {
                        coalesceable: false,
                        ..group
                    });
                }
            }
            _ => {
                if guard.ime_active {
                    guard.ime_active = false;
                    if let Some(pending) = guard.ime_pending.take() {
                        guard.push_undo_group(pending);
                    }
                }
                guard.push_undo_group(group);
            }
        }
    }

    fn try_apply_history(&mut self, ctx: &egui::Context, id: egui::Id, is_redo: bool) {
        let Ok(mut guard) = self.history.lock() else {
            return;
        };
        if guard.ime_active
            && let Some(pending) = guard.ime_pending.take()
        {
            guard.ime_active = false;
            guard.push_undo_group(pending);
        }

        let Some(candidate) = (if is_redo {
            guard.redo_stack.last()
        } else {
            guard.undo_stack.back()
        }) else {
            return;
        };

        let Some((new_chars, new_bytes)) =
            candidate.preview_dims(!is_redo, self.char_count, self.text.len())
        else {
            return;
        };
        if new_bytes > self.limit.saturating_mul(4) || new_chars > self.limit {
            self.rejected = true;
            return;
        }

        let Some(group) = (if is_redo {
            guard.redo_stack.pop()
        } else {
            guard.undo_stack.pop_back()
        }) else {
            return;
        };

        let applied = if is_redo {
            group.ops.iter().all(|op| op.apply_forward(self.text))
        } else {
            group.ops.iter().rev().all(|op| op.apply_inverse(self.text))
        };
        if !applied {
            guard.clear_history();
            return;
        }

        let restored_cursor = if is_redo {
            group.after_cursor
        } else {
            group.before_cursor
        };
        if is_redo {
            guard.undo_stack.push_back(group);
        } else {
            guard.redo_stack.push(group);
        }

        self.char_count = new_chars;
        self.history_applied = true;
        guard.fingerprint = Some(TextFingerprint::from_text(self.text, self.char_count));
        if let Some(cursor) = restored_cursor {
            self.previous.cursor.set_char_range(Some(cursor));
            self.previous.clone().store(ctx, id);
        }
    }

    fn byte_index(&self, char_idx: usize) -> usize {
        if char_idx == 0 {
            return 0;
        }
        if char_idx >= self.char_count {
            return self.text.len();
        }
        if self.text.len() == self.char_count {
            return char_idx;
        }
        let mid = self.char_count / 2;
        if char_idx <= mid {
            self.text
                .char_indices()
                .nth(char_idx)
                .map_or(self.text.len(), |(byte_idx, _)| byte_idx)
        } else {
            let from_end = self.char_count.saturating_sub(char_idx);
            self.text
                .char_indices()
                .rev()
                .nth(from_end.saturating_sub(1))
                .map_or(self.text.len(), |(byte_idx, _)| byte_idx)
        }
    }
}

fn legacy_undoer_populated(state: &TextEditState) -> bool {
    let undoer = state.undoer();
    let probe_a = (
        CCursorRange::one(egui::text::CCursor::new(usize::MAX)),
        "\0__bonsai_probe_a__".to_owned(),
    );
    let probe_b = (
        CCursorRange::one(egui::text::CCursor::new(usize::MAX.saturating_sub(1))),
        "\0__bonsai_probe_b__".to_owned(),
    );
    undoer.has_undo(&probe_a)
        || undoer.has_undo(&probe_b)
        || undoer.has_redo(&probe_a)
        || undoer.has_redo(&probe_b)
}

fn inspect_frame_events(
    ctx: &egui::Context,
    _id: egui::Id,
) -> (bool, f64, FrameInputMode, Option<bool>) {
    ctx.input(|input| {
        let mut has_mutating = false;
        let mut mode = FrameInputMode::Idle;
        let mut history_action = None;

        for event in &input.events {
            match event {
                egui::Event::Paste(_)
                | egui::Event::Cut
                | egui::Event::AccessKitActionRequest(_) => {
                    has_mutating = true;
                    mode = FrameInputMode::NonCoalescingEdit;
                }
                egui::Event::Text(_) => {
                    has_mutating = true;
                    if mode == FrameInputMode::Idle {
                        mode = FrameInputMode::TypingOrDelete;
                    }
                }
                egui::Event::Ime(ime_event) => {
                    has_mutating = true;
                    match ime_event {
                        egui::ImeEvent::Preedit { text, .. } => {
                            if text.is_empty() {
                                mode = FrameInputMode::ImeCancel;
                            } else {
                                mode = FrameInputMode::ImePreedit;
                            }
                        }
                        egui::ImeEvent::Commit(_) => {
                            mode = FrameInputMode::ImeCommit;
                        }
                        egui::ImeEvent::DeleteSurrounding { .. } => {
                            mode = FrameInputMode::NonCoalescingEdit;
                        }
                        #[expect(deprecated, reason = "egui 0.37 deprecated ImeEvent::Enabled.")]
                        egui::ImeEvent::Enabled => {}
                        #[expect(deprecated, reason = "egui 0.37 deprecated ImeEvent::Disabled.")]
                        egui::ImeEvent::Disabled => {
                            if mode == FrameInputMode::Idle {
                                mode = FrameInputMode::ImeCancel;
                            }
                        }
                    }
                }
                egui::Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } => {
                    let is_nav = matches!(
                        key,
                        egui::Key::ArrowLeft
                            | egui::Key::ArrowRight
                            | egui::Key::ArrowUp
                            | egui::Key::ArrowDown
                            | egui::Key::Home
                            | egui::Key::End
                            | egui::Key::PageUp
                            | egui::Key::PageDown
                            | egui::Key::Escape
                    );
                    if !is_nav {
                        has_mutating = true;
                        if (modifiers.matches_logically(egui::Modifiers::COMMAND)
                            && *key == egui::Key::Y)
                            || (modifiers.matches_logically(
                                egui::Modifiers::SHIFT | egui::Modifiers::COMMAND,
                            ) && *key == egui::Key::Z)
                        {
                            history_action = Some(true);
                            mode = FrameInputMode::NonCoalescingEdit;
                        } else if modifiers.matches_logically(egui::Modifiers::COMMAND)
                            && *key == egui::Key::Z
                        {
                            history_action = Some(false);
                            mode = FrameInputMode::NonCoalescingEdit;
                        } else if mode == FrameInputMode::Idle {
                            if *key == egui::Key::Enter {
                                mode = FrameInputMode::NonCoalescingEdit;
                            } else {
                                mode = FrameInputMode::TypingOrDelete;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        (has_mutating, input.time, mode, history_action)
    })
}

pub(super) fn fits(text: &str) -> bool {
    if text.is_ascii() {
        text.len() <= super::INPUT_LIMIT
    } else {
        text.len() <= super::INPUT_LIMIT * 4
            && text.chars().take(super::INPUT_LIMIT + 1).count() <= super::INPUT_LIMIT
    }
}

impl TextBuffer for Buffer<'_> {
    fn is_mutable(&self) -> bool {
        true
    }

    fn as_str(&self) -> &str {
        self.text
    }

    fn insert_text(&mut self, text: &str, index: CharIndex) -> usize {
        if self.rejected || text.is_empty() {
            return 0;
        }
        let remaining = self.limit.saturating_sub(self.char_count);
        // The byte guard avoids scanning an arbitrarily large clipboard payload.
        let count = if text.len() > remaining.saturating_mul(4) {
            remaining.saturating_add(1)
        } else {
            text.chars().take(remaining.saturating_add(1)).count()
        };
        if count > remaining {
            self.rejected = true;
            return 0;
        }
        if count != 0 {
            let copied = (text.len() <= MAX_RETAINED_UNDO_BYTES).then(|| text.to_owned());
            let byte_idx = self.byte_index(index.0);
            if !self.try_record_op(TxOp::Insert {
                byte_idx,
                byte_len: text.len(),
                copied,
            }) {
                return 0;
            }
            self.text.insert_str(byte_idx, text);
            self.char_count = self.char_count.saturating_add(count);
        }
        count
    }

    fn delete_char_range(&mut self, range: Range<CharIndex>) {
        if self.rejected || range.is_empty() {
            return;
        }
        let start = range.start.0.min(self.char_count);
        let end = range.end.0.min(self.char_count);
        if start >= end {
            return;
        }
        let byte_start = self.byte_index(start);
        let byte_end = self.byte_index(end);
        if byte_start >= byte_end {
            return;
        }
        let removed = self.text[byte_start..byte_end].to_owned();
        if !self.try_record_op(TxOp::Delete {
            byte_idx: byte_start,
            removed,
        }) {
            return;
        }
        self.text.drain(byte_start..byte_end);
        self.char_count = self.char_count.saturating_sub(end - start);
    }

    fn clear(&mut self) {
        if self.rejected || self.text.is_empty() {
            return;
        }
        let removed = self.text.clone();
        if !self.try_record_op(TxOp::Delete {
            byte_idx: 0,
            removed,
        }) {
            return;
        }
        self.text.clear();
        self.char_count = 0;
    }

    fn replace_with(&mut self, text: &str) {
        if self.rejected || self.text.as_str() == text {
            return;
        }
        // Undo/redo replaces the whole value. Validate its new contents before
        // computing the UTF-8 char-boundary delta between old and new text.
        if text.len() > self.limit.saturating_mul(4) {
            self.rejected = true;
            return;
        }
        let new_chars = text.chars().take(self.limit.saturating_add(1)).count();
        if new_chars > self.limit {
            self.rejected = true;
            return;
        }

        let (prefix, old_mid_end, new_mid_end) = crate::context::utf8_diff_bounds(self.text, text);
        if prefix < old_mid_end {
            let removed = self.text[prefix..old_mid_end].to_owned();
            if !self.try_record_op(TxOp::Delete {
                byte_idx: prefix,
                removed,
            }) {
                return;
            }
        }
        if prefix < new_mid_end {
            let inserted = &text[prefix..new_mid_end];
            let copied = (inserted.len() <= MAX_RETAINED_UNDO_BYTES).then(|| inserted.to_owned());
            if !self.try_record_op(TxOp::Insert {
                byte_idx: prefix,
                byte_len: inserted.len(),
                copied,
            }) {
                return;
            }
        }
        text.clone_into(self.text);
        self.char_count = new_chars;
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<Buffer<'static>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit_history(
        ctx: &egui::Context,
        id: egui::Id,
        text: &mut String,
        limit: usize,
        modifiers: egui::Modifiers,
    ) -> (bool, Option<CCursorRange>) {
        edit_key_with_time(ctx, id, text, limit, egui::Key::Z, modifiers, None)
    }

    fn edit_key_with_time(
        ctx: &egui::Context,
        id: egui::Id,
        text: &mut String,
        limit: usize,
        key: egui::Key,
        modifiers: egui::Modifiers,
        time: Option<f64>,
    ) -> (bool, Option<CCursorRange>) {
        let mut rejected = false;
        let mut cursor = None;
        let mut full_output = ctx.run_ui(
            egui::RawInput {
                time,
                events: [true, false]
                    .map(|pressed| egui::Event::Key {
                        key,
                        physical_key: None,
                        pressed,
                        repeat: false,
                        modifiers,
                    })
                    .into(),
                ..Default::default()
            },
            |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    let mut buffer = Buffer::new(text, ui.ctx(), id, limit);
                    let mut output = egui::TextEdit::multiline(&mut buffer).id(id).show(ui);
                    rejected = buffer.finish(&mut output);
                    cursor = output.cursor_range;
                });
            },
        );
        full_output.textures_delta.clear();
        (rejected, cursor)
    }

    fn drive_text_edit_events(
        ctx: &egui::Context,
        id: egui::Id,
        text: &mut String,
        limit: usize,
        time: f64,
        events: Vec<egui::Event>,
    ) -> (bool, Option<CCursorRange>) {
        let mut rejected = false;
        let mut cursor = None;
        let mut full_output = ctx.run_ui(
            egui::RawInput {
                time: Some(time),
                events,
                ..Default::default()
            },
            |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    let mut buffer = Buffer::new(text, ui.ctx(), id, limit);
                    let mut output = egui::TextEdit::multiline(&mut buffer).id(id).show(ui);
                    rejected = buffer.finish(&mut output);
                    cursor = output.cursor_range;
                });
            },
        );
        full_output.textures_delta.clear();
        (rejected, cursor)
    }

    #[test]
    fn rejected_whole_value_undo_and_redo_restore_selection_and_history() -> std::io::Result<()> {
        for redo in [false, true] {
            let ctx = egui::Context::default();
            let id = egui::Id::unique("whole-value-history");
            let selection =
                CCursorRange::two(egui::text::CCursor::new(0), egui::text::CCursor::new(3));
            let current = (selection, "e\u{301} 🌱".to_owned());
            let target = (
                CCursorRange::one(egui::text::CCursor::new(6)),
                "日本語🌱e\u{301}".to_owned(),
            );
            // Deliberately inject an out-of-limit history point to exercise
            // whole-value rejection while egui holds its undo mutex.
            let mut history = TextEditUndoer::default();
            if redo {
                history.add_undo(&current);
                history.add_undo(&target);
                assert_eq!(history.undo(&target), Some(&current));
            } else {
                history.add_undo(&target);
                history.add_undo(&current);
            }
            let mut state = TextEditState::default();
            state.cursor.set_char_range(Some(selection));
            state.set_undoer(history);
            state.store(&ctx, id);
            ctx.memory_mut(|memory| memory.request_focus(id));

            let forward = if redo {
                egui::Modifiers::COMMAND | egui::Modifiers::SHIFT
            } else {
                egui::Modifiers::COMMAND
            };
            let backward = if redo {
                egui::Modifiers::COMMAND
            } else {
                egui::Modifiers::COMMAND | egui::Modifiers::SHIFT
            };
            let mut text = current.1.clone();
            assert_eq!(
                edit_history(&ctx, id, &mut text, 5, forward),
                (true, Some(selection))
            );
            assert_eq!(text, current.1);
            let restored = TextEditState::load(&ctx, id)
                .ok_or_else(|| std::io::Error::other("Restored editor state missing"))?;
            assert_eq!(restored.cursor.char_range(), Some(selection));

            // Relax only the fixture's limit. The rejected action must remain
            // available, and its reverse must still restore the exact selection.
            assert_eq!(
                edit_history(&ctx, id, &mut text, 6, forward),
                (false, Some(target.0))
            );
            assert_eq!(text, target.1);
            assert_eq!(
                edit_history(&ctx, id, &mut text, 6, backward),
                (false, Some(selection))
            );
            assert_eq!(text, current.1);
        }
        Ok(())
    }

    #[test]
    fn draft_whole_replacements_keep_unicode_limits_and_atomic_rollback_bytes() {
        let ctx = egui::Context::default();
        let original = "e\u{301} 🌱";
        let mut text = original.to_owned();
        let mut buffer = Buffer::new(&mut text, &ctx, egui::Id::unique("replacement"), 5);

        buffer.replace_with(original);
        assert_eq!(buffer.insert_text("", CharIndex(0)), 0);
        assert!(!buffer.has_tx_changes(), "Unchanged input needs no delta");

        buffer.replace_with("日本語🌱e");
        assert_eq!(buffer.text.as_str(), "日本語🌱e");
        assert!(buffer.has_tx_changes());
        assert!(!buffer.rejected);

        buffer.replace_with("日本語🌱e\u{301}");
        buffer.clear();
        assert!(buffer.rejected);
        assert_eq!(buffer.text.as_str(), "日本語🌱e");
        buffer.rollback_tx();
        assert_eq!(buffer.text.as_str(), original);

        let mut text = original.to_owned();
        let mut buffer = Buffer::new(&mut text, &ctx, egui::Id::unique("clear"), 5);
        buffer.clear();
        buffer.clear();
        assert_eq!(buffer.text, "");
        assert!(buffer.has_tx_changes());
        buffer.replace_with("🌱e\u{301}");
        assert_eq!(buffer.text.as_str(), "🌱e\u{301}");
        buffer.rollback_tx();
        assert_eq!(buffer.text.as_str(), original);

        let mut text = original.to_owned();
        let mut buffer = Buffer::new(&mut text, &ctx, egui::Id::unique("rejected"), 5);
        buffer.replace_with("日本語🌱e\u{301}");
        assert!(buffer.rejected);
        assert_eq!(buffer.text.as_str(), original);
        assert!(
            !buffer.has_tx_changes(),
            "Rejected replacement changed no bytes"
        );
    }

    #[test]
    fn draft_navigation_preserves_history_without_copying_undo_snapshots() -> std::io::Result<()> {
        let ctx = egui::Context::default();
        let id = egui::Id::unique("draft-navigation");
        let cursor = CCursorRange::one(egui::text::CCursor::new(0));
        let older = (cursor, "Earlier café 🌱 draft".to_owned());
        let current = (cursor, "Current 界 draft".to_owned());
        let mut undoer = TextEditUndoer::default();
        undoer.add_undo(&older);
        undoer.add_undo(&current);
        let mut state = TextEditState::default();
        state.set_undoer(undoer);
        state.store(&ctx, id);

        for key in [
            egui::Key::ArrowLeft,
            egui::Key::ArrowRight,
            egui::Key::ArrowUp,
            egui::Key::ArrowDown,
            egui::Key::Home,
            egui::Key::End,
            egui::Key::PageUp,
            egui::Key::PageDown,
            egui::Key::Escape,
        ] {
            let mut text = current.1.clone();
            let mut full_output = ctx.run_pass(
                egui::RawInput {
                    events: vec![egui::Event::Key {
                        key,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::SHIFT | egui::Modifiers::CTRL,
                    }],
                    ..Default::default()
                },
                |ctx| {
                    let buffer = Buffer::new(&mut text, ctx, id, 100);
                    assert!(buffer.undoer.is_none());
                },
            );
            full_output.textures_delta.clear();
            assert_eq!(text, current.1);
        }
        let mut history = TextEditState::load(&ctx, id)
            .ok_or_else(|| std::io::Error::other("Draft undo state missing"))?
            .undoer();
        assert_eq!(history.undo(&current), Some(&older));
        assert_eq!(history.redo(&older), Some(&current));

        // An editing event in the same frame must still take a rollback
        // snapshot, even when navigation precedes it.
        let mut text = current.1.clone();
        let mut full_output = ctx.run_pass(
            egui::RawInput {
                events: vec![
                    egui::Event::Key {
                        key: egui::Key::ArrowLeft,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                    },
                    egui::Event::Paste("Replacement".to_owned()),
                ],
                ..Default::default()
            },
            |ctx| {
                let buffer = Buffer::new(&mut text, ctx, id, 100);
                assert!(buffer.undoer.is_some());
            },
        );
        full_output.textures_delta.clear();
        Ok(())
    }

    #[test]
    fn draft_buffer_keeps_complete_unicode_at_limit_and_rejects_next_character() {
        let ctx = egui::Context::default();
        let mut text = "🌱".repeat(super::super::INPUT_LIMIT - 1);
        let mut buffer = Buffer::new(
            &mut text,
            &ctx,
            egui::Id::unique("draft"),
            super::super::INPUT_LIMIT,
        );
        assert_eq!(
            buffer.insert_text("界", CharIndex(super::super::INPUT_LIMIT - 1)),
            1
        );
        assert_eq!(
            buffer.insert_text("x", CharIndex(super::super::INPUT_LIMIT)),
            0
        );
        assert!(buffer.rejected);
        assert_eq!(buffer.text.chars().count(), super::super::INPUT_LIMIT);
    }

    #[test]
    fn draft_buffer_selection_overflow_retains_original_and_blocks_later_mutations() {
        let ctx = egui::Context::default();
        let mut text = "café 🌱 end".to_owned();
        let original = text.clone();
        let mut buffer = Buffer::new(&mut text, &ctx, egui::Id::unique("draft"), 12);
        buffer.delete_char_range(CharIndex(5)..CharIndex(6));
        assert_eq!(buffer.insert_text("oversized replacement", CharIndex(5)), 0);
        buffer.delete_char_range(CharIndex(0)..CharIndex(4));
        assert!(buffer.has_tx_changes());
        assert_eq!(buffer.text.as_str(), "café  end");
        buffer.rollback_tx();
        assert_eq!(buffer.text.as_str(), original);
    }

    #[test]
    fn inverse_op_history_undoes_redoes_and_rejects_oversized_restore() {
        let ctx = egui::Context::default();
        let id = egui::Id::unique("inverse-ops");
        let mut text = String::new();
        ctx.memory_mut(|mem| mem.request_focus(id));

        let _ = drive_text_edit_events(
            &ctx,
            id,
            &mut text,
            50,
            0.0,
            vec![egui::Event::Paste("Alpha".to_owned())],
        );
        let _ = drive_text_edit_events(
            &ctx,
            id,
            &mut text,
            50,
            2.0,
            vec![egui::Event::Paste(" café 🌱".to_owned())],
        );
        assert_eq!(text, "Alpha café 🌱");

        // Tight limit rejects redo if target exceeds limit, while preserving history.
        let (rejected, _) = edit_key_with_time(
            &ctx,
            id,
            &mut text,
            15,
            egui::Key::Z,
            egui::Modifiers::COMMAND,
            Some(3.0),
        );
        assert!(!rejected);
        assert_eq!(text, "Alpha");

        // Reject redo when limit is tightened below target length ("Alpha café 🌱" = 13 chars).
        let (rejected, _) = edit_key_with_time(
            &ctx,
            id,
            &mut text,
            6,
            egui::Key::Y,
            egui::Modifiers::COMMAND,
            Some(4.0),
        );
        assert!(rejected);
        assert_eq!(text, "Alpha");

        // Relaxing limit completes the redo cleanly.
        let (rejected, _) = edit_key_with_time(
            &ctx,
            id,
            &mut text,
            20,
            egui::Key::Z,
            egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
            Some(5.0),
        );
        assert!(!rejected);
        assert_eq!(text, "Alpha café 🌱");
    }

    #[test]
    fn inverse_op_history_evicts_oldest_at_one_mib_and_handles_ime_and_external_switch() {
        assert_eq!(MAX_RETAINED_UNDO_BYTES, 1_048_576);
        let mut direct_history = DraftHistory::default();
        for step in 0..4 {
            direct_history.push_undo_group(UndoGroup {
                ops: vec![EditOp::Insert {
                    byte_idx: step * 300 * 1024,
                    text: "a".repeat(300 * 1024),
                }],
                before_cursor: None,
                after_cursor: None,
                timestamp: f64::from(u32::try_from(step).unwrap_or(0)),
                byte_cost: 300 * 1024,
                coalesceable: false,
            });
        }
        assert!(direct_history.retained_bytes <= MAX_RETAINED_UNDO_BYTES);
        assert_eq!(direct_history.undo_stack.len(), 3);

        let ctx = egui::Context::default();
        let id = egui::Id::unique("budget-ime-switch");
        let test_budget = 4 * 1024;
        ctx.data_mut(|data| {
            data.insert_temp(
                id.with("draft_history"),
                Arc::new(Mutex::new(DraftHistory::with_budget(test_budget))),
            );
        });
        let mut text = String::new();
        ctx.memory_mut(|mem| mem.request_focus(id));

        // 4 x 1.2 KiB pastes = 4.8 KiB > 4 KiB test budget -> oldest group evicted
        // without paying multi-megabyte egui TextEdit layout cost.
        let chunk_1200 = "a".repeat(1200);
        for step in 0..4 {
            let _ = drive_text_edit_events(
                &ctx,
                id,
                &mut text,
                super::super::INPUT_LIMIT,
                f64::from(step * 2),
                vec![egui::Event::Paste(chunk_1200.clone())],
            );
        }
        let history_arc =
            ctx.data(|data| data.get_temp::<Arc<Mutex<DraftHistory>>>(id.with("draft_history")));
        if let Some(arc) = history_arc
            && let Ok(guard) = arc.lock()
        {
            assert!(guard.retained_bytes <= test_budget);
            assert!(guard.undo_stack.len() <= 3);
        }

        // External draft switch invalidates retained undo history cleanly.
        text = "Switched draft ".to_owned();
        let _ = drive_text_edit_events(&ctx, id, &mut text, 100, 10.0, Vec::new());

        // Multi-frame IME composition commits as a single atomic undo group.
        let _ = drive_text_edit_events(
            &ctx,
            id,
            &mut text,
            100,
            11.0,
            vec![egui::Event::Ime(egui::ImeEvent::Preedit {
                text: "ni".to_owned(),
                active_range_chars: None,
            })],
        );
        let _ = drive_text_edit_events(
            &ctx,
            id,
            &mut text,
            100,
            11.1,
            vec![egui::Event::Ime(egui::ImeEvent::Preedit {
                text: "你".to_owned(),
                active_range_chars: None,
            })],
        );
        let _ = drive_text_edit_events(
            &ctx,
            id,
            &mut text,
            100,
            11.2,
            vec![egui::Event::Ime(egui::ImeEvent::Commit("你好".to_owned()))],
        );
        assert_eq!(text, "Switched draft 你好");

        // One Ctrl+Z removes the entire committed IME phrase in a single group.
        let _ = edit_key_with_time(
            &ctx,
            id,
            &mut text,
            100,
            egui::Key::Z,
            egui::Modifiers::COMMAND,
            Some(12.0),
        );
        assert_eq!(text, "Switched draft ");

        // Earlier pre-switch paste history must not replay onto the switched draft.
        let _ = edit_key_with_time(
            &ctx,
            id,
            &mut text,
            100,
            egui::Key::Z,
            egui::Modifiers::COMMAND,
            Some(13.0),
        );
        assert_eq!(text, "Switched draft ");
    }
}
