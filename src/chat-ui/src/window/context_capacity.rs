//! Quiet composer guidance; estimates never participate in request admission.
use super::{ModelLifecycle, OUTPUT_TOKENS, RichText, State, controls, egui};
use crate::context::{self, Estimate, Pressure};

#[derive(Default)]
pub(super) struct DraftCache {
    units: context::IncrementalDraftUnits,
    attachment_input: String,
    has_attachments: bool,
    attachments_revision: u64,
    estimate: Option<Estimate>,
    extra_effort: bool,
}

impl State {
    pub(super) fn context_estimate(&mut self) -> Estimate {
        let history = *self
            .context_history
            .get_or_insert_with(|| context::history_tokens(&self.committed));
        let output = self.selected_output_cap();
        let draft_matches = if self.draft_attachments.is_empty() {
            !self.context_draft.has_attachments && self.context_draft.units.matches(&self.input)
        } else {
            self.context_draft.has_attachments && self.context_draft.attachment_input == self.input
        };
        if let Some(estimate) = self.context_draft.estimate
            && estimate.history == history
            && estimate.output == usize::try_from(output).unwrap_or(usize::MAX)
            && self.context_draft.extra_effort == self.extra_effort
            && estimate.capacity == self.context_tokens
            && self.context_draft.attachments_revision == self.draft_attachments_revision
            && draft_matches
        {
            return estimate;
        }
        let estimate = if self.draft_attachments.is_empty() {
            self.context_draft.has_attachments = false;
            self.context_draft.attachment_input = String::new();
            let draft_tokens = self.context_draft.units.tokens(&self.input);
            Estimate::from_draft_tokens(
                history,
                draft_tokens,
                0,
                output,
                self.extra_effort,
                self.context_tokens,
            )
        } else {
            let (formatted_text, images) =
                crate::attachment_import::format_user_turn(&self.input, &self.draft_attachments);
            self.context_draft.has_attachments = true;
            self.input
                .clone_into(&mut self.context_draft.attachment_input);
            let draft_tokens = self.context_draft.units.tokens(&formatted_text);
            Estimate::from_draft_tokens(
                history,
                draft_tokens,
                images.len(),
                output,
                self.extra_effort,
                self.context_tokens,
            )
        };
        self.context_draft.attachments_revision = self.draft_attachments_revision;
        self.context_draft.extra_effort = self.extra_effort;
        self.context_draft.estimate = Some(estimate);
        estimate
    }

    fn selected_output_cap(&self) -> u32 {
        OUTPUT_TOKENS
            .get(self.output_budget)
            .copied()
            .unwrap_or(4096)
    }

    fn context_pending(&self) -> bool {
        self.context_tokens.is_none()
            || (self.request_conversation == Some(self.active_conversation)
                && matches!(
                    self.model,
                    ModelLifecycle::Generating
                        | ModelLifecycle::Stopping
                        | ModelLifecycle::StoppingModel
                        | ModelLifecycle::AwaitingTeardown
                ))
    }

    pub(super) fn context_toolbar(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let estimate = self.context_estimate();
        let pending = self.context_pending();
        let label = ui
            .push_id(("context-capacity", self.active_conversation), |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    let label = ui.label(RichText::new("Message").small().weak());
                    let response = context_button(ui, estimate, pending);
                    let frame = egui::Frame::popup(ui.style())
                        .corner_radius(14)
                        .inner_margin(14);
                    // Keep details above both toolbar actions. An unconstrained
                    // tall popup can be moved down by viewport containment and
                    // intercept the neighboring Thinking button's pointer input.
                    let detail_height = (response.rect.top()
                        - ui.ctx().content_rect().top()
                        - frame.total_margin().sum().y
                        - 6.0)
                        .max(0.0);
                    egui::Popup::menu(&response)
                        .accessible_name("Context details")
                        .align(egui::RectAlign::TOP_START)
                        .align_alternatives(&[egui::RectAlign::TOP_END])
                        .width(320.0)
                        .gap(6.0)
                        .frame(frame)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .show(|ui| {
                            egui::ScrollArea::vertical()
                                .id_salt("context-details")
                                .max_height(detail_height)
                                .min_scrolled_height(0.0)
                                .auto_shrink([false, true])
                                .show(ui, |ui| self.context_details(ui, estimate, pending));
                        });
                    self.thinking_button(ui);
                    label
                })
                .inner
            })
            .inner;
        if !pending && estimate.pressure() != Pressure::Comfortable {
            self.context_guidance(ui, estimate);
        }
        label
    }

    fn context_details(&mut self, ui: &mut egui::Ui, estimate: Estimate, pending: bool) {
        ui.heading("Space for this draft");
        ui.label(
            RichText::new("Estimate · exact token count checked when you send")
                .small()
                .weak(),
        );
        ui.add_space(8.0);
        if estimate.capacity.is_none() {
            ui.label("Context capacity is available after the model starts.");
        } else if pending {
            ui.label(
                "Updates when this response finishes. The unfinished turn is not included below.",
            );
        } else {
            ui.label(format!(
                "~{} of {} tokens reserved",
                tokens(estimate.total),
                tokens(estimate.capacity.unwrap_or_default())
            ));
        }
        capacity_meter(ui, estimate);
        ui.add_space(8.0);
        egui::Grid::new("context-breakdown")
            .num_columns(2)
            .spacing([18.0, 6.0])
            .show(ui, |ui| {
                for (label, amount, approximate) in [
                    ("Completed history & reasoning", estimate.history, true),
                    ("This draft & message framing", estimate.draft, true),
                    ("Extra effort planning", estimate.workflow, true),
                    ("Reply limit (includes thinking)", estimate.output, false),
                    ("Safety reserve", context::RESERVE_TOKENS, false),
                ] {
                    if amount == 0 && label == "Extra effort planning" {
                        continue;
                    }
                    ui.label(RichText::new(label).small());
                    ui.label(
                        RichText::new(format!(
                            "{}{}",
                            if approximate { "~" } else { "" },
                            tokens(amount)
                        ))
                        .small(),
                    );
                    ui.end_row();
                }
            });
        ui.add_space(8.0);
        if !pending && estimate.overflow() == 0 {
            ui.label(format!(
                "About {} tokens of headroom after the reply reservation.",
                tokens(estimate.remaining())
            ));
        }
        ui.label(RichText::new("Language and formatting affect tokenization. This estimate can be high or low; it does not guarantee that a request fits.").small().weak());
        if self.extra_effort {
            ui.label("First pass only. Extra effort's later passes add generated notes and drafts and need more room.");
        }
        ui.label(RichText::new("Reply limits are in Settings → Response → Advanced response controls. Changing them keeps your conversation and draft. Message actions use their own prompts.").small().weak());
        if estimate.output > 1_024 {
            ui.label(
                RichText::new(format!(
                    "Choosing a 1,024-token reply instead leaves {} more tokens for your prompt. The reply limit includes thinking.",
                    tokens(estimate.output - 1_024)
                ))
                .small()
                .weak(),
            );
        }
        ui.label(RichText::new("Capacity is selected when the model starts. Closing memory-heavy apps before the next model start or Retry may leave room for a larger window. This model's current capacity stays fixed.").small().weak());
        if let Some(capacity) = estimate.capacity
            && let Some(footprint) = estimate_runtime_footprint(PlannerRuntimeShape {
                context_tokens: capacity,
                weight_packing: WeightPacking::Pq2_0,
                kv_layout: KvCacheLayout::Rsq4,
                kv_sharing: KvSharingMode::CowRollback,
                evictable_mapped_weight_bytes: WeightPacking::Pq2_0.weight_bytes(),
                npu_resident_arena_bytes: 0,
                npu_released_host_pq2_bytes: 0,
                branch_checkpoints: 1,
            })
        {
            ui.label(
                RichText::new(format_footprint_diagnostic(&footprint))
                    .small()
                    .weak(),
            );
        }
        if !pending && estimate.pressure() != Pressure::Comfortable {
            if estimate.suggested_cap().is_some() {
                ui.label("A smaller reply limit could leave more room for this message.");
            }
            ui.add_space(8.0);
            self.context_actions(ui);
        }
    }

    fn context_guidance(&mut self, ui: &mut egui::Ui, estimate: Estimate) {
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(if estimate.pressure() == Pressure::Over {
                    "This message may be too long for this conversation."
                } else {
                    "This conversation is getting full."
                })
                .small()
                .color(ui.visuals().warn_fg_color),
            );
            self.context_actions(ui);
        });
    }

    fn context_actions(&mut self, ui: &mut egui::Ui) {
        if ui
            .add_enabled(
                self.model.workspace_available(),
                egui::Button::new("Start fresh").small(),
            )
            .on_hover_text("Open a new conversation. This conversation and its draft are kept.")
            .clicked()
        {
            self.new_chat();
            egui::Popup::close_all(ui.ctx());
        }
    }
}

fn context_button(ui: &mut egui::Ui, estimate: Estimate, pending: bool) -> egui::Response {
    let fraction = estimate
        .capacity
        .map_or(0.0, |capacity| fraction(estimate.total, capacity));
    let ring = egui::Atom::paint(egui::Vec2::splat(16.0), move |ui, args| {
        context_ring(
            ui,
            args.rect.center(),
            fraction,
            estimate.pressure(),
            pending,
        );
    });
    let response =
        controls::toolbar_action(ui, (RichText::new("Context:").size(13.0), ring), false)
            .accessible_name("Context");
    let description = if estimate.capacity.is_none() {
        "Context capacity is available after the model starts. Open for details.".to_owned()
    } else if pending {
        "Context updates when this response finishes. Open for details.".to_owned()
    } else {
        let percent = estimate.total.saturating_mul(100) / estimate.capacity.unwrap_or(1);
        format!("About {percent}% of context reserved, including the next reply. Open for details.")
    };
    let _ = response.ctx.accesskit_node_builder(response.id, |node| {
        node.set_description(description.clone());
    });
    response.on_hover_text(description)
}

fn context_ring(
    ui: &egui::Ui,
    center: egui::Pos2,
    fraction: f32,
    pressure: Pressure,
    pending: bool,
) {
    const RADIUS: f32 = 6.5;
    ui.painter().circle_stroke(
        center,
        RADIUS,
        egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
    );
    if pending {
        // An unfilled ring with a dash means the unfinished turn is not yet
        // counted. Do not display a stale fraction or animate an idle widget.
        ui.painter().line_segment(
            [center - egui::vec2(2.0, 0.0), center + egui::vec2(2.0, 0.0)],
            egui::Stroke::new(1.5, ui.visuals().weak_text_color()),
        );
        return;
    }
    let color = if pressure == Pressure::Comfortable {
        ui.visuals().text_color()
    } else {
        ui.visuals().warn_fg_color
    };
    if fraction > 0.0 {
        let points = (0..=48_u16)
            .map(|step| {
                let angle = std::f32::consts::TAU.mul_add(
                    fraction * f32::from(step) / 48.0,
                    -std::f32::consts::FRAC_PI_2,
                );
                let (sin, cos) = angle.sin_cos();
                center + egui::vec2(cos, sin) * RADIUS
            })
            .collect();
        ui.painter()
            .add(egui::Shape::line(points, egui::Stroke::new(2.5, color)));
    }
}

fn capacity_meter(ui: &mut egui::Ui, estimate: Estimate) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 6.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 3.0, ui.visuals().faint_bg_color);
    let Some(capacity) = estimate.capacity else {
        return;
    };
    let accent = ui.visuals().selection.bg_fill;
    let mut used = 0_usize;
    for (amount, color) in [
        (estimate.history, ui.visuals().weak_text_color()),
        (estimate.draft.saturating_add(estimate.workflow), accent),
        (estimate.output, accent.gamma_multiply(0.55)),
        (context::RESERVE_TOKENS, ui.visuals().text_color()),
    ] {
        let start = used.min(capacity);
        used = used.saturating_add(amount);
        let end = used.min(capacity);
        let left = rect.width().mul_add(fraction(start, capacity), rect.left());
        let right = rect.width().mul_add(fraction(end, capacity), rect.left());
        if right > left {
            ui.painter().rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(left, rect.top()),
                    egui::pos2(right, rect.bottom()),
                ),
                1.0,
                color,
            );
        }
    }
}

// Normalize before narrowing, so larger model windows cannot truncate the
// token count. The ring needs only a bounded screen-space fraction.
fn fraction(tokens: usize, capacity: usize) -> f32 {
    if capacity == 0 {
        return 0.0;
    }
    let units = tokens.min(capacity).saturating_mul(usize::from(u16::MAX)) / capacity;
    f32::from(u16::try_from(units).unwrap_or(u16::MAX)) / f32::from(u16::MAX)
}

fn tokens(amount: usize) -> String {
    let raw = amount.to_string();
    let mut formatted = String::with_capacity(raw.len() + raw.len() / 3);
    for (index, character) in raw.chars().enumerate() {
        if index != 0 && (raw.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(character);
    }
    formatted
}

const FIXED_RUNTIME_BYTES: u64 = 11 * 1024 * 1024 * 1024;
const PQ2_WEIGHT_BYTES: u64 = 7_206_168_928;
const PTQ1_WEIGHT_BYTES: u64 = 5_946_648_928;
const PQ2_TO_PTQ1_SAVING_BYTES: u64 = PQ2_WEIGHT_BYTES - PTQ1_WEIGHT_BYTES;
const HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES: u64 = 2_888_663_040;
const RECURRENT_BYTES: u64 = 156_893_184;
const ACTIVE_AND_ROLLBACK_RECURRENT_COPIES: u64 = 2;
const MAX_REUSABLE_BRANCH_CHECKPOINTS: u8 = 8;
const KV_COW_PAGE_TOKENS: usize = 64;
const SEQUENTIAL_KV_BOOKKEEPING_BYTES_PER_TOKEN: u64 = 24 * 4 + 128;

/// Weight packing option surfaced by context capacity footprint diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WeightPacking {
    Pq2_0,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Alternative weight packing variant covered by footprint tests."
        )
    )]
    Ptq1_0,
}

impl WeightPacking {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Pq2_0 => "PQ2_0",
            Self::Ptq1_0 => "PTQ1_0",
        }
    }

    pub(super) const fn weight_bytes(self) -> u64 {
        match self {
            Self::Pq2_0 => PQ2_WEIGHT_BYTES,
            Self::Ptq1_0 => PTQ1_WEIGHT_BYTES,
        }
    }

    pub(super) const fn fixed_runtime_bytes(self) -> u64 {
        match self {
            Self::Pq2_0 => FIXED_RUNTIME_BYTES,
            Self::Ptq1_0 => FIXED_RUNTIME_BYTES - PQ2_TO_PTQ1_SAVING_BYTES,
        }
    }
}

/// Attention KV layout surfaced by context capacity footprint diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KvCacheLayout {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Alternative KV layout variant covered by footprint tests."
        )
    )]
    F16,
    Rsq4,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Alternative KV layout variant covered by footprint tests."
        )
    )]
    Rsq4K3v,
}

impl KvCacheLayout {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::Rsq4 => "K4/V4",
            Self::Rsq4K3v => "K4/V3",
        }
    }

    pub(super) const fn bytes_per_token(self) -> u64 {
        match self {
            Self::F16 => 65_536,
            Self::Rsq4 => 16_896,
            Self::Rsq4K3v => 14_848,
        }
    }
}

/// KV cache sharing mode between active and rollback histories.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KvSharingMode {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Alternative KV sharing mode covered by footprint tests."
        )
    )]
    DivergentRollback,
    CowRollback,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Alternative KV sharing mode covered by footprint tests."
        )
    )]
    AppendOnlyCow,
}

impl KvSharingMode {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::DivergentRollback => "2x KV",
            Self::CowRollback => "CoW KV",
            Self::AppendOnlyCow => "Append-only CoW KV",
        }
    }
}

/// Input parameters for pure steady-state physical RAM vs private Job commit diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PlannerRuntimeShape {
    pub(super) context_tokens: usize,
    pub(super) weight_packing: WeightPacking,
    pub(super) kv_layout: KvCacheLayout,
    pub(super) kv_sharing: KvSharingMode,
    pub(super) evictable_mapped_weight_bytes: u64,
    pub(super) npu_resident_arena_bytes: u64,
    pub(super) npu_released_host_pq2_bytes: u64,
    pub(super) branch_checkpoints: u8,
}

/// Computed steady-state physical RAM and private Windows Job commit estimates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PlannerFootprintEstimate {
    pub(super) context_tokens: usize,
    pub(super) weight_packing: WeightPacking,
    pub(super) kv_layout: KvCacheLayout,
    pub(super) kv_sharing: KvSharingMode,
    pub(super) steady_state_physical_bytes: u64,
    pub(super) private_job_commit_bytes: u64,
    pub(super) evictable_mapped_bytes: u64,
    pub(super) kv_bytes: u64,
    pub(super) recurrent_bytes: u64,
    pub(super) npu_arena_bytes: u64,
    pub(super) npu_released_host_bytes: u64,
    pub(super) branch_checkpoints: u8,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Pure planner diagnostics exercised by context_capacity tests."
    )
)]
impl PlannerFootprintEstimate {
    pub(super) const fn physical_savings_over(&self, baseline: &Self) -> u64 {
        baseline
            .steady_state_physical_bytes
            .saturating_sub(self.steady_state_physical_bytes)
    }

    pub(super) const fn job_commit_savings_over(&self, baseline: &Self) -> u64 {
        baseline
            .private_job_commit_bytes
            .saturating_sub(self.private_job_commit_bytes)
    }
}

/// Compute estimated steady-state physical RAM and private Job commit for `shape`.
///
/// Returns `None` when `branch_checkpoints > 8`, when NPU PQ2 host deduplication
/// is requested with non-PQ2 weights or exceeds `2_888_663_040` bytes, when
/// `evictable_mapped_weight_bytes` exceeds remaining host weights/runtime, or on overflow.
pub(super) fn estimate_runtime_footprint(
    shape: PlannerRuntimeShape,
) -> Option<PlannerFootprintEstimate> {
    if shape.branch_checkpoints > MAX_REUSABLE_BRANCH_CHECKPOINTS
        || shape.npu_released_host_pq2_bytes > HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES
    {
        return None;
    }
    if shape.npu_released_host_pq2_bytes > 0 && shape.weight_packing != WeightPacking::Pq2_0 {
        return None;
    }
    let host_weights_and_fixed = shape
        .weight_packing
        .fixed_runtime_bytes()
        .checked_sub(shape.npu_released_host_pq2_bytes)?;
    if shape.evictable_mapped_weight_bytes > host_weights_and_fixed {
        return None;
    }
    let kv_bytes = if shape.context_tokens == 0 {
        0
    } else {
        let ctx = u64::try_from(shape.context_tokens).ok()?;
        let kv_per_token = shape.kv_layout.bytes_per_token();
        let bookkeeping = ctx.checked_mul(SEQUENTIAL_KV_BOOKKEEPING_BYTES_PER_TOKEN)?;
        let kv_payload = match shape.kv_sharing {
            KvSharingMode::DivergentRollback => ctx.checked_mul(2)?.checked_mul(kv_per_token)?,
            KvSharingMode::CowRollback => {
                let tail = u64::try_from(shape.context_tokens.min(KV_COW_PAGE_TOKENS)).ok()?;
                ctx.checked_add(tail)?.checked_mul(kv_per_token)?
            }
            KvSharingMode::AppendOnlyCow => ctx.checked_mul(kv_per_token)?,
        };
        kv_payload.checked_add(bookkeeping)?
    };
    let recurrent_copies =
        ACTIVE_AND_ROLLBACK_RECURRENT_COPIES.checked_add(u64::from(shape.branch_checkpoints))?;
    let recurrent_bytes = recurrent_copies.checked_mul(RECURRENT_BYTES)?;
    let host_total = host_weights_and_fixed
        .checked_add(recurrent_bytes)?
        .checked_add(kv_bytes)?;
    let private_job_commit_bytes = host_total.checked_sub(shape.evictable_mapped_weight_bytes)?;
    let steady_state_physical_bytes = host_total.checked_add(shape.npu_resident_arena_bytes)?;
    Some(PlannerFootprintEstimate {
        context_tokens: shape.context_tokens,
        weight_packing: shape.weight_packing,
        kv_layout: shape.kv_layout,
        kv_sharing: shape.kv_sharing,
        steady_state_physical_bytes,
        private_job_commit_bytes,
        evictable_mapped_bytes: shape.evictable_mapped_weight_bytes,
        kv_bytes,
        recurrent_bytes,
        npu_arena_bytes: shape.npu_resident_arena_bytes,
        npu_released_host_bytes: shape.npu_released_host_pq2_bytes,
        branch_checkpoints: shape.branch_checkpoints,
    })
}

/// Format an integer byte count in binary `GiB` rounded to two decimal places.
pub(super) fn format_gib_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    let hundredths = (u128::from(bytes) * 100 + u128::from(GIB / 2)) / u128::from(GIB);
    let whole = hundredths / 100;
    let frac = hundredths % 100;
    format!("{whole}.{frac:02} GiB")
}

/// Format a compact diagnostic summary contrasting steady-state physical RAM vs private Job commit.
pub(super) fn format_footprint_diagnostic(estimate: &PlannerFootprintEstimate) -> String {
    let mut parts = vec![
        estimate.weight_packing.label().to_owned(),
        format!(
            "{} {}",
            estimate.kv_layout.label(),
            estimate.kv_sharing.label()
        ),
    ];
    if estimate.npu_released_host_bytes > 0 || estimate.npu_arena_bytes > 0 {
        parts.push(format!(
            "NPU dedup -{} host / +{} arena",
            format_gib_bytes(estimate.npu_released_host_bytes),
            format_gib_bytes(estimate.npu_arena_bytes)
        ));
    }
    parts.push(format!(
        "Est. physical {}",
        format_gib_bytes(estimate.steady_state_physical_bytes)
    ));
    let commit = if estimate.evictable_mapped_bytes > 0 {
        format!(
            "Job commit {} ({} mapped)",
            format_gib_bytes(estimate.private_job_commit_bytes),
            format_gib_bytes(estimate.evictable_mapped_bytes)
        )
    } else {
        format!(
            "Job commit {}",
            format_gib_bytes(estimate.private_job_commit_bytes)
        )
    };
    parts.push(commit);
    let branch_label = if estimate.branch_checkpoints == 1 {
        "1 branch checkpoint".to_owned()
    } else {
        format!("{} branch checkpoints", estimate.branch_checkpoints)
    };
    parts.push(branch_label);
    parts.join(" · ")
}

#[cfg(test)]
mod tests;
