//! Cancellable, bounded Myers O(ND) diff over Unicode token spans on a background thread.

use std::{
    error::Error,
    fmt, io,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread::{self, JoinHandle},
};

/// Hard byte ceiling per comparison input (`4 * INPUT_LIMIT` = 4 MiB of UTF-8 text).
pub(super) const MAX_DIFF_INPUT_BYTES: usize = super::INPUT_LIMIT * 4;

/// Maximum Myers edit distance before aborting cleanly with [`DiffError::EditLimitExceeded`].
pub(super) const MAX_EDIT_DISTANCE: usize = 4_096;

/// Classification of a contiguous span in a [`UnicodeDiff`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DiffOp {
    /// Identical text retained across both left and right snapshots.
    Equal,
    /// Text present only in the left snapshot.
    Delete,
    /// Text present only in the right snapshot.
    Insert,
}

/// One ordered, coalesced UTF-8 text span produced by [`diff_unicode_spans`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DiffSpan {
    /// Edit operation for this span.
    pub(super) op: DiffOp,
    /// Exact UTF-8 slice content aligned to Unicode scalar and `\r\n` boundaries.
    pub(super) text: String,
}

/// Lossless sequence of Unicode-aligned edit spans between two text snapshots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct UnicodeDiff {
    /// Ordered, coalesced diff spans.
    pub(super) spans: Vec<DiffSpan>,
    /// Number of equal token spans before coalescing.
    pub(super) equal_tokens: usize,
    /// Number of deleted token spans before coalescing.
    pub(super) deleted_tokens: usize,
    /// Number of inserted token spans before coalescing.
    pub(super) inserted_tokens: usize,
}

impl UnicodeDiff {
    fn reconstruct(&self, side_op: DiffOp) -> String {
        let include = |op| op == DiffOp::Equal || op == side_op;
        let capacity: usize = self
            .spans
            .iter()
            .filter(|span| include(span.op))
            .map(|span| span.text.len())
            .sum();
        let mut out = String::with_capacity(capacity);
        for span in &self.spans {
            if include(span.op) {
                out.push_str(&span.text);
            }
        }
        out
    }

    /// Losslessly reconstructs the exact original left input string from the diff spans.
    #[must_use]
    pub(super) fn reconstruct_left(&self) -> String {
        self.reconstruct(DiffOp::Delete)
    }

    /// Losslessly reconstructs the exact original right input string from the diff spans.
    #[must_use]
    pub(super) fn reconstruct_right(&self) -> String {
        self.reconstruct(DiffOp::Insert)
    }

    /// Returns `true` when at least one deleted or inserted span is present.
    #[must_use]
    pub(super) const fn has_changes(&self) -> bool {
        self.deleted_tokens > 0 || self.inserted_tokens > 0
    }
}

/// Bounded failure modes returned by [`diff_unicode_spans`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DiffError {
    /// Cooperative cancellation was requested via the shared `AtomicBool`.
    Cancelled,
    /// Either input exceeded [`MAX_DIFF_INPUT_BYTES`].
    InputTooLarge,
    /// The Myers edit distance exceeded [`MAX_EDIT_DISTANCE`].
    EditLimitExceeded,
    /// Background diff worker exited or panicked before sending an outcome.
    WorkerFailed,
}

impl fmt::Display for DiffError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cancelled => "Branch comparison was cancelled.",
            Self::InputTooLarge => "Comparison input exceeds the 4 MiB draft diff limit.",
            Self::EditLimitExceeded => "Texts differ too extensively for inline token comparison.",
            Self::WorkerFailed => "Branch comparison stopped unexpectedly.",
        })
    }
}

impl Error for DiffError {}

/// Immutable origin identity for a snapshot participating in branch/draft diffing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum BranchDiffOrigin {
    /// Snapshot of a conversation's unsent composer draft.
    Draft {
        /// Owning conversation identifier.
        conversation_id: u64,
    },
    /// Snapshot of a single transcript message inside a conversation/branch.
    Message {
        /// Owning conversation identifier.
        conversation_id: u64,
        /// Transcript message identifier within that conversation.
        message_id: u64,
        /// Committed history position at snapshot time, if committed.
        history_index: Option<usize>,
    },
    /// Explicit caller-labelled text snapshot (for example an edited prompt draft).
    Snapshot {
        /// Owning conversation identifier.
        conversation_id: u64,
        /// Human-readable snapshot label.
        label: String,
    },
}

/// Immutable provenance and text snapshot for one side of a branch/draft diff.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BranchDiffSource {
    /// Provenance identifying where this snapshot came from.
    pub(super) origin: BranchDiffOrigin,
    /// Display title shown in the comparison dialog.
    pub(super) title: String,
    /// Immutable UTF-8 text snapshot at comparison start.
    pub(super) text: String,
}

/// Completed outcome emitted by a [`BranchDiffJob`] worker thread.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BranchDiffOutcome {
    /// Monotonic job generation used to reject stale acceptance attempts.
    pub(super) generation: u64,
    /// Immutable provenance and snapshot of the left side.
    pub(super) left_source: BranchDiffSource,
    /// Immutable provenance and snapshot of the right side.
    pub(super) right_source: BranchDiffSource,
    /// Bounded diff result or cancellation/limit error.
    pub(super) result: Result<UnicodeDiff, DiffError>,
}

/// Cancellable background diff handle running off the `egui` UI thread.
pub(super) struct BranchDiffJob {
    generation: u64,
    left_source: BranchDiffSource,
    right_source: BranchDiffSource,
    cancel: Arc<AtomicBool>,
    receiver: Receiver<BranchDiffOutcome>,
    worker: Option<JoinHandle<()>>,
    completed: Option<BranchDiffOutcome>,
}

impl BranchDiffJob {
    /// Spawns a dedicated background `std::thread` to diff `left_source` and `right_source`.
    ///
    /// # Errors
    /// Returns an `io::Error` if the worker thread could not be spawned.
    pub(super) fn start(
        generation: u64,
        left_source: BranchDiffSource,
        right_source: BranchDiffSource,
        repaint: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> io::Result<Self> {
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let worker_left = left_source.clone();
        let worker_right = right_source.clone();
        let (sender, receiver) = mpsc::sync_channel(1);

        let worker = thread::Builder::new()
            .name("bonsai-branch-diff".to_owned())
            .spawn(move || {
                let result =
                    diff_unicode_spans(&worker_left.text, &worker_right.text, &worker_cancel);
                let _sent = sender.send(BranchDiffOutcome {
                    generation,
                    left_source: worker_left,
                    right_source: worker_right,
                    result,
                });
                if let Some(notify) = repaint {
                    notify();
                }
            })?;

        Ok(Self {
            generation,
            left_source,
            right_source,
            cancel,
            receiver,
            worker: Some(worker),
            completed: None,
        })
    }

    /// Returns the monotonic generation identifier for this job.
    #[must_use]
    pub(super) const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the immutable left-hand source provenance and snapshot.
    #[must_use]
    pub(super) const fn left_source(&self) -> &BranchDiffSource {
        &self.left_source
    }

    /// Returns the immutable right-hand source provenance and snapshot.
    #[must_use]
    pub(super) const fn right_source(&self) -> &BranchDiffSource {
        &self.right_source
    }

    /// Signals cooperative cancellation to the worker thread without blocking the caller.
    pub(super) fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    /// Returns `true` if cancellation was requested.
    #[must_use]
    pub(super) fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    /// Returns `true` while the background thread is still running and uncancelled.
    #[must_use]
    pub(super) fn is_running(&self) -> bool {
        self.completed.is_none() && !self.is_cancelled()
    }

    /// Non-blocking poll for completion, reaping the joined worker once done.
    pub(super) fn poll(&mut self) -> Option<&BranchDiffOutcome> {
        // Polonius lets an already-completed job lend `self.completed` directly
        // while the pending path ends the borrow to store a received outcome.
        if let Some(outcome) = self.completed.as_ref() {
            return Some(outcome);
        }
        if self.worker.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(handle) = self.worker.take()
        {
            let _joined = handle.join();
        }
        match self.receiver.try_recv() {
            Ok(outcome) => {
                if let Some(handle) = self.worker.take() {
                    let _joined = handle.join();
                }
                Some(self.completed.insert(outcome))
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                if let Some(handle) = self.worker.take() {
                    let _joined = handle.join();
                }
                let result = if self.is_cancelled() {
                    Err(DiffError::Cancelled)
                } else {
                    Err(DiffError::WorkerFailed)
                };
                Some(self.completed.insert(BranchDiffOutcome {
                    generation: self.generation,
                    left_source: self.left_source.clone(),
                    right_source: self.right_source.clone(),
                    result,
                }))
            }
        }
    }
}

impl Drop for BranchDiffJob {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
        if let Some(handle) = self.worker.take() {
            let _joined = handle.join();
        }
    }
}

const fn is_line_break(ch: char) -> bool {
    matches!(ch, '\n' | '\r' | '\u{0085}' | '\u{2028}' | '\u{2029}')
}

const fn is_combining_or_joiner(ch: char) -> bool {
    matches!(
        ch,
        '\u{0300}'..='\u{036F}'
            | '\u{0483}'..='\u{0489}'
            | '\u{0591}'..='\u{05BD}'
            | '\u{05BF}'
            | '\u{05C1}'..='\u{05C2}'
            | '\u{05C4}'..='\u{05C5}'
            | '\u{05C7}'
            | '\u{0610}'..='\u{061A}'
            | '\u{064B}'..='\u{065F}'
            | '\u{0670}'
            | '\u{06D6}'..='\u{06DC}'
            | '\u{06DF}'..='\u{06E4}'
            | '\u{06E7}'..='\u{06E8}'
            | '\u{06EA}'..='\u{06ED}'
            | '\u{1AB0}'..='\u{1AFF}'
            | '\u{1DC0}'..='\u{1DFF}'
            | '\u{200C}'..='\u{200D}'
            | '\u{20D0}'..='\u{20FF}'
            | '\u{3099}'..='\u{309C}'
            | '\u{FE00}'..='\u{FE0F}'
            | '\u{FE20}'..='\u{FE2F}'
            | '\u{1F3FB}'..='\u{1F3FF}'
    )
}

const fn is_cjk_or_kana(ch: char) -> bool {
    matches!(
        ch,
        '\u{3040}'..='\u{309F}'
            | '\u{30A0}'..='\u{30FF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{AC00}'..='\u{D7AF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{20000}'..='\u{2CEAF}'
    )
}

/// Splits `text` into contiguous `char_indices`-aligned Unicode word, horizontal
/// whitespace, newline (keeping `\r\n` atomic), and symbol/CJK tokens.
fn tokenize_unicode_spans(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut pos = 0_usize;
    let bytes_len = text.len();

    while pos < bytes_len {
        let rest = &text[pos..];
        if rest.starts_with("\r\n") {
            tokens.push(&text[pos..pos + 2]);
            pos += 2;
            continue;
        }
        let mut iter = rest.char_indices();
        let Some((_, first)) = iter.next() else {
            break;
        };
        let mut end = pos + first.len_utf8();

        if is_line_break(first) {
            tokens.push(&text[pos..end]);
            pos = end;
            continue;
        }

        if first.is_whitespace() {
            for (rel_idx, ch) in iter {
                if !ch.is_whitespace() || is_line_break(ch) {
                    break;
                }
                end = pos + rel_idx + ch.len_utf8();
            }
            tokens.push(&text[pos..end]);
            pos = end;
            continue;
        }

        if (first.is_alphanumeric() || first == '_') && !is_cjk_or_kana(first) {
            for (rel_idx, ch) in iter {
                if !((ch.is_alphanumeric() || ch == '_' || is_combining_or_joiner(ch))
                    && !is_cjk_or_kana(ch))
                {
                    break;
                }
                end = pos + rel_idx + ch.len_utf8();
            }
            tokens.push(&text[pos..end]);
            pos = end;
            continue;
        }

        // Single symbol, punctuation, CJK/Kana, or emoji (keeping ZWJ/combining clusters intact).
        let mut expect_zwj_next = false;
        for (rel_idx, ch) in iter {
            if expect_zwj_next {
                expect_zwj_next = false;
                end = pos + rel_idx + ch.len_utf8();
                continue;
            }
            if is_combining_or_joiner(ch) {
                end = pos + rel_idx + ch.len_utf8();
                expect_zwj_next = ch == '\u{200D}';
            } else {
                break;
            }
        }
        tokens.push(&text[pos..end]);
        pos = end;
    }

    tokens
}

#[derive(Clone, Copy)]
enum TokenEdit<'a> {
    Equal(&'a str),
    Delete(&'a str),
    Insert(&'a str),
}

fn push_coalesced(
    spans: &mut Vec<DiffSpan>,
    op: DiffOp,
    token: &str,
    equal_tokens: &mut usize,
    deleted_tokens: &mut usize,
    inserted_tokens: &mut usize,
) {
    match op {
        DiffOp::Equal => *equal_tokens += 1,
        DiffOp::Delete => *deleted_tokens += 1,
        DiffOp::Insert => *inserted_tokens += 1,
    }
    if let Some(last) = spans.last_mut()
        && last.op == op
    {
        last.text.push_str(token);
    } else {
        spans.push(DiffSpan {
            op,
            text: token.to_owned(),
        });
    }
}

fn myers_middle<'a>(
    a: &[&'a str],
    b: &[&'a str],
    max_edit_distance: usize,
    cancel: &AtomicBool,
    mut on_step: impl FnMut(usize),
) -> Result<Vec<TokenEdit<'a>>, DiffError> {
    let n = a.len();
    let m = b.len();
    if n == 0 && m == 0 {
        return Ok(Vec::new());
    }
    if n == 0 {
        if m > max_edit_distance {
            return Err(DiffError::EditLimitExceeded);
        }
        return Ok(b.iter().copied().map(TokenEdit::Insert).collect());
    }
    if m == 0 {
        if n > max_edit_distance {
            return Err(DiffError::EditLimitExceeded);
        }
        return Ok(a.iter().copied().map(TokenEdit::Delete).collect());
    }

    let max_d = max_edit_distance.min(n + m);
    if n.abs_diff(m) > max_d {
        return Err(DiffError::EditLimitExceeded);
    }

    let offset = max_d;
    let mut v = vec![0_usize; max_d * 2 + 1];
    let mut trace_data: Vec<u32> = Vec::new();
    let mut trace_starts: Vec<usize> = Vec::new();
    let mut solved_d: Option<usize> = None;
    let mut inner_checks = 0_usize;

    for d in 0..=max_d {
        if cancel.load(Ordering::Relaxed) {
            return Err(DiffError::Cancelled);
        }
        on_step(d);

        if d > 0 {
            let prev_d = d - 1;
            let prev_d_isize = isize::try_from(prev_d).map_err(|_| DiffError::EditLimitExceeded)?;
            trace_starts.push(trace_data.len());
            trace_data
                .try_reserve(d)
                .map_err(|_| DiffError::EditLimitExceeded)?;
            for idx in 0..d {
                let offset_step =
                    isize::try_from(idx * 2).map_err(|_| DiffError::EditLimitExceeded)?;
                let k = -prev_d_isize + offset_step;
                let v_idx = offset
                    .checked_add_signed(k)
                    .ok_or(DiffError::EditLimitExceeded)?;
                let val = u32::try_from(v[v_idx]).map_err(|_| DiffError::EditLimitExceeded)?;
                trace_data.push(val);
            }
        }

        let d_isize = isize::try_from(d).map_err(|_| DiffError::EditLimitExceeded)?;
        let mut k = -d_isize;
        while k <= d_isize {
            inner_checks = inner_checks.wrapping_add(1);
            if inner_checks.is_multiple_of(256) && cancel.load(Ordering::Relaxed) {
                return Err(DiffError::Cancelled);
            }

            let k_idx = offset
                .checked_add_signed(k)
                .ok_or(DiffError::EditLimitExceeded)?;
            let mut x = if d == 0 {
                0
            } else if k == -d_isize || (k != d_isize && v[k_idx - 1] < v[k_idx + 1]) {
                v[k_idx + 1]
            } else {
                v[k_idx - 1] + 1
            };
            let x_isize = isize::try_from(x).map_err(|_| DiffError::EditLimitExceeded)?;
            let y_isize = x_isize - k;
            if y_isize < 0 {
                k += 2;
                continue;
            }
            let mut y = usize::try_from(y_isize).map_err(|_| DiffError::EditLimitExceeded)?;
            while x < n && y < m && a[x] == b[y] {
                x += 1;
                y += 1;
            }
            v[k_idx] = x;
            if x >= n && y >= m {
                solved_d = Some(d);
                break;
            }
            k += 2;
        }
        if solved_d.is_some() {
            break;
        }
    }

    let final_d = solved_d.ok_or(DiffError::EditLimitExceeded)?;
    let mut reverse_edits = Vec::new();
    let mut x = n;
    let mut y = m;

    for d in (1..=final_d).rev() {
        let prev_d = d - 1;
        let start = trace_starts[prev_d];
        let prev_v = &trace_data[start..start + d];
        let prev_d_isize = isize::try_from(prev_d).map_err(|_| DiffError::EditLimitExceeded)?;
        let d_isize = isize::try_from(d).map_err(|_| DiffError::EditLimitExceeded)?;
        let x_isize = isize::try_from(x).map_err(|_| DiffError::EditLimitExceeded)?;
        let y_isize = isize::try_from(y).map_err(|_| DiffError::EditLimitExceeded)?;
        let k = x_isize - y_isize;

        let prev_x_at = |diag: isize| -> Result<usize, DiffError> {
            let slot = usize::try_from((diag + prev_d_isize) / 2)
                .map_err(|_| DiffError::EditLimitExceeded)?;
            usize::try_from(prev_v[slot]).map_err(|_| DiffError::EditLimitExceeded)
        };

        let down = k == -d_isize || (k != d_isize && prev_x_at(k - 1)? < prev_x_at(k + 1)?);
        let prev_k = if down { k + 1 } else { k - 1 };
        let prev_x = prev_x_at(prev_k)?;
        let prev_x_isize = isize::try_from(prev_x).map_err(|_| DiffError::EditLimitExceeded)?;
        let prev_y =
            usize::try_from(prev_x_isize - prev_k).map_err(|_| DiffError::EditLimitExceeded)?;

        while x > prev_x && y > prev_y {
            x -= 1;
            y -= 1;
            reverse_edits.push(TokenEdit::Equal(a[x]));
        }
        if down {
            reverse_edits.push(TokenEdit::Insert(b[prev_y]));
        } else {
            reverse_edits.push(TokenEdit::Delete(a[prev_x]));
        }
        x = prev_x;
        y = prev_y;
    }

    while x > 0 && y > 0 {
        x -= 1;
        y -= 1;
        reverse_edits.push(TokenEdit::Equal(a[x]));
    }

    reverse_edits.reverse();
    Ok(reverse_edits)
}

fn diff_unicode_spans_inner(
    left: &str,
    right: &str,
    max_edit_distance: usize,
    cancel: &AtomicBool,
    on_step: impl FnMut(usize),
) -> Result<UnicodeDiff, DiffError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(DiffError::Cancelled);
    }
    if left.len() > MAX_DIFF_INPUT_BYTES || right.len() > MAX_DIFF_INPUT_BYTES {
        return Err(DiffError::InputTooLarge);
    }
    if left == right {
        let mut equal_tokens = 0_usize;
        let mut spans = Vec::new();
        if !left.is_empty() {
            equal_tokens = tokenize_unicode_spans(left).len();
            spans.push(DiffSpan {
                op: DiffOp::Equal,
                text: left.to_owned(),
            });
        }
        return Ok(UnicodeDiff {
            spans,
            equal_tokens,
            deleted_tokens: 0,
            inserted_tokens: 0,
        });
    }

    let left_tokens = tokenize_unicode_spans(left);
    let right_tokens = tokenize_unicode_spans(right);

    let mut prefix = 0_usize;
    let min_len = left_tokens.len().min(right_tokens.len());
    while prefix < min_len && left_tokens[prefix] == right_tokens[prefix] {
        prefix += 1;
    }

    let mut suffix = 0_usize;
    while suffix + prefix < min_len
        && left_tokens[left_tokens.len() - 1 - suffix]
            == right_tokens[right_tokens.len() - 1 - suffix]
    {
        suffix += 1;
    }

    let left_mid = &left_tokens[prefix..left_tokens.len() - suffix];
    let right_mid = &right_tokens[prefix..right_tokens.len() - suffix];
    let middle_edits = myers_middle(left_mid, right_mid, max_edit_distance, cancel, on_step)?;

    let mut spans = Vec::new();
    let mut equal_tokens = 0_usize;
    let mut deleted_tokens = 0_usize;
    let mut inserted_tokens = 0_usize;

    for &tok in &left_tokens[..prefix] {
        push_coalesced(
            &mut spans,
            DiffOp::Equal,
            tok,
            &mut equal_tokens,
            &mut deleted_tokens,
            &mut inserted_tokens,
        );
    }

    // Group contiguous Deletes before Inserts inside replacement runs for readability while
    // keeping lossless left/right ordering.
    let mut pending_deletes: Vec<&str> = Vec::new();
    let mut pending_inserts: Vec<&str> = Vec::new();
    let flush_pending = |spans: &mut Vec<DiffSpan>,
                         pending_deletes: &mut Vec<&str>,
                         pending_inserts: &mut Vec<&str>,
                         equal_tokens: &mut usize,
                         deleted_tokens: &mut usize,
                         inserted_tokens: &mut usize| {
        for tok in pending_deletes.drain(..) {
            push_coalesced(
                spans,
                DiffOp::Delete,
                tok,
                equal_tokens,
                deleted_tokens,
                inserted_tokens,
            );
        }
        for tok in pending_inserts.drain(..) {
            push_coalesced(
                spans,
                DiffOp::Insert,
                tok,
                equal_tokens,
                deleted_tokens,
                inserted_tokens,
            );
        }
    };

    for edit in middle_edits {
        match edit {
            TokenEdit::Equal(tok) => {
                flush_pending(
                    &mut spans,
                    &mut pending_deletes,
                    &mut pending_inserts,
                    &mut equal_tokens,
                    &mut deleted_tokens,
                    &mut inserted_tokens,
                );
                push_coalesced(
                    &mut spans,
                    DiffOp::Equal,
                    tok,
                    &mut equal_tokens,
                    &mut deleted_tokens,
                    &mut inserted_tokens,
                );
            }
            TokenEdit::Delete(tok) => pending_deletes.push(tok),
            TokenEdit::Insert(tok) => pending_inserts.push(tok),
        }
    }
    flush_pending(
        &mut spans,
        &mut pending_deletes,
        &mut pending_inserts,
        &mut equal_tokens,
        &mut deleted_tokens,
        &mut inserted_tokens,
    );

    for &tok in &left_tokens[left_tokens.len() - suffix..] {
        push_coalesced(
            &mut spans,
            DiffOp::Equal,
            tok,
            &mut equal_tokens,
            &mut deleted_tokens,
            &mut inserted_tokens,
        );
    }

    Ok(UnicodeDiff {
        spans,
        equal_tokens,
        deleted_tokens,
        inserted_tokens,
    })
}

/// Computes a bounded Myers O(ND) diff over Unicode word, whitespace, and newline spans.
///
/// Cooperative cancellation is checked via `cancel` both before tokenization and
/// periodically throughout the Myers search loop.
///
/// # Errors
/// Returns [`DiffError::Cancelled`] if `cancel` becomes `true`,
/// [`DiffError::InputTooLarge`] if either slice exceeds [`MAX_DIFF_INPUT_BYTES`], or
/// [`DiffError::EditLimitExceeded`] if the token edit distance exceeds [`MAX_EDIT_DISTANCE`].
pub(super) fn diff_unicode_spans(
    left: &str,
    right: &str,
    cancel: &AtomicBool,
) -> Result<UnicodeDiff, DiffError> {
    diff_unicode_spans_inner(left, right, MAX_EDIT_DISTANCE, cancel, |_| {})
}

#[cfg(test)]
mod tests {
    use super::{
        BranchDiffJob, BranchDiffOrigin, BranchDiffSource, DiffError, DiffOp, MAX_DIFF_INPUT_BYTES,
        diff_unicode_spans, diff_unicode_spans_inner, tokenize_unicode_spans,
    };
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::{Duration, Instant},
    };

    #[test]
    fn tokenizer_never_splits_crlf_utf8_emoji_zwj_or_combining_marks() {
        let sample = "Hello\r\n日本語テスト\r\nمرحبا بِكُمْ\r\ne\u{0301} 👨‍👩‍👧‍👦 🌱!";
        let tokens = tokenize_unicode_spans(sample);
        assert_eq!(tokens.concat(), sample);
        assert_eq!(tokens.iter().filter(|&&tok| tok == "\r\n").count(), 3);
        assert!(!tokens.contains(&"\r"));
        assert!(!tokens.contains(&"\n"));
        assert!(tokens.contains(&"👨‍👩‍👧‍👦"));
        assert!(tokens.contains(&"e\u{0301}"));
        assert!(tokens.contains(&"بِكُمْ"));
    }

    #[test]
    fn unicode_emoji_cjk_rtl_and_crlf_diff_reconstructs_losslessly() -> Result<(), DiffError> {
        let cases = [
            ("", ""),
            ("identical 🌱", "identical 🌱"),
            (
                "Line one\r\n日本語テキスト\r\nشُكْرًا لك 👨‍👩‍👧 🌱",
                "Line one\r\n日本語テスト\r\nمَرْحَبًا لك 🚀",
            ),
            (
                "Alpha beta\r\ngamma\ndelta",
                "Alpha delta\ngamma\r\nepsilon",
            ),
            ("e\u{0301}clair", "éclair"),
        ];
        let cancel = AtomicBool::new(false);
        for (left, right) in cases {
            let diff = diff_unicode_spans(left, right, &cancel)?;
            assert_eq!(diff.reconstruct_left(), left);
            assert_eq!(diff.reconstruct_right(), right);
            if left == right {
                assert!(!diff.has_changes());
            } else {
                assert!(diff.has_changes());
            }
        }
        Ok(())
    }

    #[test]
    fn diff_respects_precancel_mid_search_cancel_size_bound_and_edit_cap() {
        let precancel = AtomicBool::new(true);
        assert_eq!(
            diff_unicode_spans("left", "right", &precancel),
            Err(DiffError::Cancelled)
        );

        let oversized = "a".repeat(MAX_DIFF_INPUT_BYTES + 1);
        let cancel = AtomicBool::new(false);
        assert_eq!(
            diff_unicode_spans(&oversized, "small", &cancel),
            Err(DiffError::InputTooLarge)
        );
        assert_eq!(
            diff_unicode_spans("small", &oversized, &cancel),
            Err(DiffError::InputTooLarge)
        );

        // Deterministic mid-computation cancellation inside the Myers step loop.
        let left_many: String = (0..200).map(|i| format!("l{i} ")).collect();
        let right_many: String = (0..200).map(|i| format!("r{i} ")).collect();
        let mid_cancel = AtomicBool::new(false);
        let mut observed_steps = 0_usize;
        let res = diff_unicode_spans_inner(
            &left_many,
            &right_many,
            super::MAX_EDIT_DISTANCE,
            &mid_cancel,
            |d| {
                observed_steps = d;
                if d == 12 {
                    mid_cancel.store(true, Ordering::Relaxed);
                }
            },
        );
        assert_eq!(res, Err(DiffError::Cancelled));
        assert_eq!(observed_steps, 12);

        // Edit distance cap exceeded when differing tokens surpass limit.
        let cap_cancel = AtomicBool::new(false);
        let res = diff_unicode_spans_inner(&left_many, &right_many, 16, &cap_cancel, |_| {});
        assert_eq!(res, Err(DiffError::EditLimitExceeded));
    }

    #[test]
    fn background_branch_diff_job_completes_and_cancels_off_thread() -> std::io::Result<()> {
        let left = BranchDiffSource {
            origin: BranchDiffOrigin::Message {
                conversation_id: 1,
                message_id: 2,
                history_index: Some(1),
            },
            title: "Original branch".to_owned(),
            text: "Hello world\r\n日本語 🌱".to_owned(),
        };
        let right = BranchDiffSource {
            origin: BranchDiffOrigin::Draft { conversation_id: 1 },
            title: "Current draft".to_owned(),
            text: "Hello brave world\r\n日本語 🚀".to_owned(),
        };
        let notified = Arc::new(AtomicBool::new(false));
        let notify_flag = Arc::clone(&notified);
        let mut job = BranchDiffJob::start(
            7,
            left.clone(),
            right.clone(),
            Some(Arc::new(move || {
                notify_flag.store(true, Ordering::Release);
            })),
        )?;

        let deadline = Instant::now() + Duration::from_secs(5);
        while job.poll().is_none() {
            assert!(Instant::now() < deadline, "diff job timed out");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(notified.load(Ordering::Acquire));
        let outcome = job
            .poll()
            .ok_or_else(|| std::io::Error::other("missing outcome"))?;
        assert_eq!(outcome.generation, 7);
        assert_eq!(outcome.left_source, left);
        assert_eq!(outcome.right_source, right);
        let diff = outcome
            .result
            .as_ref()
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        assert_eq!(diff.reconstruct_left(), left.text);
        assert_eq!(diff.reconstruct_right(), right.text);
        assert!(
            diff.spans
                .iter()
                .any(|s| s.op == DiffOp::Insert && s.text.contains("brave"))
        );

        // Cancellation on a large job sets cancelled state cleanly without panicking.
        let large_left = BranchDiffSource {
            origin: BranchDiffOrigin::Draft { conversation_id: 2 },
            title: "Left".to_owned(),
            text: (0..2_200).map(|i| format!("a{i} ")).collect(),
        };
        let large_right = BranchDiffSource {
            origin: BranchDiffOrigin::Draft { conversation_id: 3 },
            title: "Right".to_owned(),
            text: (0..2_200).map(|i| format!("b{i} ")).collect(),
        };
        let mut cancel_job = BranchDiffJob::start(8, large_left, large_right, None)?;
        cancel_job.cancel();
        assert!(cancel_job.is_cancelled());
        assert!(!cancel_job.is_running());
        let cancel_deadline = Instant::now() + Duration::from_secs(5);
        while cancel_job.poll().is_none() {
            assert!(Instant::now() < cancel_deadline, "cancel job timed out");
            std::thread::sleep(Duration::from_millis(2));
        }
        if let Some(outcome) = cancel_job.poll() {
            assert!(matches!(
                outcome.result,
                Err(DiffError::Cancelled | DiffError::EditLimitExceeded)
            ));
        }

        // A disconnected/panicked worker transitions poll() to a terminal error and stops running.
        let (sender, receiver) = std::sync::mpsc::sync_channel::<super::BranchDiffOutcome>(1);
        let panicked = std::thread::Builder::new()
            .name("bonsai-branch-diff-panic-test".to_owned())
            .spawn(move || {
                let _hold = sender;
                std::panic::resume_unwind(Box::new("simulated branch diff panic"));
            })?;
        let mut failed_job = BranchDiffJob {
            generation: 9,
            left_source: BranchDiffSource {
                origin: BranchDiffOrigin::Draft { conversation_id: 1 },
                title: "Left".to_owned(),
                text: "a".to_owned(),
            },
            right_source: BranchDiffSource {
                origin: BranchDiffOrigin::Draft { conversation_id: 1 },
                title: "Right".to_owned(),
                text: "b".to_owned(),
            },
            cancel: Arc::new(AtomicBool::new(false)),
            receiver,
            worker: Some(panicked),
            completed: None,
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while failed_job.poll().is_none() {
            assert!(Instant::now() < deadline, "failed job timed out");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(!failed_job.is_running());
        assert_eq!(
            failed_job.poll().map(|o| &o.result),
            Some(&Err(DiffError::WorkerFailed))
        );
        Ok(())
    }
}
