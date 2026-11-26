//! Conversation response controls shared by Settings and the composer popup.
use super::super::OUTPUT_TOKENS;
use super::{RichText, State, controls, egui, settings_card};

impl State {
    pub(in crate::window) fn thinking_button(&mut self, ui: &mut egui::Ui) {
        let help = match (self.enable_thinking, self.extra_effort) {
            (true, true) => {
                "Thinking and extra effort are on. Change how Swift thinks before replying."
            }
            (true, false) => "Thinking is on. Change how Swift thinks before replying.",
            (false, true) => "Extra effort is on. Change how Swift thinks before replying.",
            (false, false) => {
                "Choose whether Swift thinks before replying or takes extra time to review."
            }
        };
        let response =
            controls::toolbar_button(ui, "Thinking", self.enable_thinking || self.extra_effort)
                .on_hover_text(help);
        let opening = response.clicked()
            && !egui::Popup::is_id_open(ui.ctx(), egui::Popup::default_response_id(&response));
        if opening {
            // Keyboard activation belongs to the choices after opening. If
            // generation disables both, Enter must not toggle the still-focused
            // opener closed and expose Stop to the following Escape.
            response.surrender_focus();
        }
        egui::Popup::menu(&response)
            .accessible_name("Thinking options")
            .align(egui::RectAlign::TOP_START)
            .width(280.0)
            .gap(6.0)
            .frame(
                egui::Frame::popup(ui.style())
                    .corner_radius(14)
                    .inner_margin(14),
            )
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                ui.label(RichText::new("Thinking").strong().size(16.0));
                ui.label(RichText::new("For this conversation").small().weak());
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);
                self.thinking_preferences(ui, opening);
                if self.model.busy() {
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new("Available when this response finishes.")
                            .small()
                            .weak(),
                    );
                }
            });
    }

    fn thinking_preferences(&mut self, ui: &mut egui::Ui, focus: bool) {
        self.recovery_dirty |= ThinkingPreferences {
            thinking: &mut self.enable_thinking,
            extra_effort: &mut self.extra_effort,
        }
        .show(ui, !self.model.busy(), focus);
    }

    pub(super) fn response_preferences(&mut self, ui: &mut egui::Ui) {
        ui.label(
            RichText::new("Changes affect the next response and every Extra effort pass.")
                .small()
                .weak(),
        );
        if self.model.busy() {
            ui.add_space(8.0);
            ui.label("Response controls unlock when the current response finishes.");
        }
        ui.add_space(8.0);
        settings_card(ui, |ui| {
            self.thinking_preferences(ui, false);
        });
        ui.add_space(8.0);
        settings_card(ui, |ui| {
            egui::CollapsingHeader::new("Advanced response controls")
                .id_salt("settings-advanced-response")
                .show(ui, |ui| {
                    ui.add_space(8.0);
                    self.recovery_dirty |=
                        reply_preferences(ui, &mut self.output_budget, !self.model.busy());
                    ui.add_space(12.0);
                    ui.separator();
                    let daisyrose_changed = daisyrose_preferences(
                        ui,
                        self.daisyrose_available,
                        self.daisyrose_default_scale_tenths,
                        self.daisyrose_max_rank,
                        self.daisyrose_sha256_prefix.as_deref(),
                        &mut self.daisyrose_enabled,
                        &mut self.daisyrose_scale,
                        !self.model.busy(),
                    );
                    self.daisyrose_customized |= daisyrose_changed;
                    self.recovery_dirty |= daisyrose_changed;
                    ui.add_space(12.0);
                    ui.separator();
                    self.recovery_dirty |= cognitive_preferences(
                        ui,
                        &mut self.cognitive,
                        self.cognitive_controls_available,
                        self.enable_thinking,
                        !self.model.busy(),
                    );
                });
        });
    }
}

/// Borrow only the values edited by the shared Thinking control.
struct ThinkingPreferences<'a> {
    thinking: &'a mut bool,
    extra_effort: &'a mut bool,
}

impl ThinkingPreferences<'_> {
    fn show(self, ui: &mut egui::Ui, editable: bool, focus: bool) -> bool {
        ui.add_enabled_ui(editable, |ui| {
            ui.spacing_mut().interact_size.y = 28.0;
            let thinking = ui.checkbox(self.thinking, "Thinking");
            if focus && editable {
                thinking.request_focus();
            }
            let mut changed = thinking.changed();
            ui.label(
                RichText::new("Reason before answering. Uses part of the reply limit.")
                    .small()
                    .weak(),
            );
            ui.add_space(12.0);
            changed |= ui.checkbox(self.extra_effort, "Extra effort").changed();
            ui.label(
                RichText::new("Plan, draft and review. Experimental; takes longer and may not improve answers.")
                    .small()
                    .weak(),
            );
            changed
        })
        .inner
    }
}

fn reply_preferences(ui: &mut egui::Ui, output_budget: &mut usize, editable: bool) -> bool {
    ui.label(RichText::new("Reply length").strong());
    ui.add_space(8.0);
    let before = *output_budget;
    ui.add_enabled_ui(editable, |ui| {
        ui.horizontal(|ui| {
            let label = ui.label("Output cap");
            egui::ComboBox::from_id_salt("output-cap")
                .selected_text(format!(
                    "{} tokens",
                    OUTPUT_TOKENS.get(*output_budget).copied().unwrap_or(4096)
                ))
                .show_ui(ui, |ui| {
                    for (index, tokens) in OUTPUT_TOKENS.iter().enumerate() {
                        ui.selectable_value(output_budget, index, format!("{tokens} tokens"));
                    }
                })
                .response
                .labelled_by(label.id);
        });
    });
    ui.add_space(8.0);
    ui.label(RichText::new("The output cap includes thinking. A smaller cap leaves more room for your conversation. Open Context for the capacity selected for this model.").small().weak());
    before != *output_budget
}

fn loaded_daisyrose_summary(max_rank: Option<u16>, sha256_prefix: Option<&str>) -> Option<String> {
    match (max_rank, sha256_prefix) {
        (Some(rank), Some(prefix)) => Some(format!("Loaded: rank \u{2264}{rank} · {prefix}")),
        (Some(rank), None) => Some(format!("Loaded: rank \u{2264}{rank}")),
        (None, Some(prefix)) => Some(format!("Loaded: {prefix}")),
        (None, None) => None,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Borrowed UI settings arguments avoid coupling presentation to State."
)]
fn daisyrose_preferences(
    ui: &mut egui::Ui,
    available: bool,
    default_scale_tenths: u8,
    max_rank: Option<u16>,
    sha256_prefix: Option<&str>,
    enabled: &mut bool,
    scale: &mut f32,
    editable: bool,
) -> bool {
    ui.add_space(12.0);
    ui.separator();
    ui.add_space(8.0);
    ui.label(RichText::new("DaisyRose multi-rank LoRA").strong());
    ui.label(
        RichText::new(
            "Optional stackable NpuFriendly adapter applied alongside the pinned adapter when enabled.",
        )
        .small()
        .weak(),
    );
    ui.add_space(8.0);
    let mut changed = ui
        .add_enabled_ui(editable && available, |ui| {
            ui.checkbox(enabled, "Enable DaisyRose adapter").changed()
        })
        .inner;
    changed |= ui
        .add_enabled_ui(editable && available && *enabled, |ui| {
            ui.add(
                egui::Slider::new(scale, 0.0..=2.0)
                    .step_by(0.1)
                    .fixed_decimals(1)
                    .text("DaisyRose scale"),
            )
            .changed()
        })
        .inner;
    let help = if !available {
        "Unavailable for this launch (not configured, opted out with --no-daisyrose-lora, or omitted under memory limits)."
            .to_owned()
    } else if *enabled {
        format!(
            "0 disables contribution. Default: {}.{}. Applies to your next response.",
            default_scale_tenths / 10,
            default_scale_tenths % 10
        )
    } else if default_scale_tenths > 0 {
        format!(
            "Disabled for this conversation. New conversations default to {}.{}. Enable DaisyRose above to adjust its scale.",
            default_scale_tenths / 10,
            default_scale_tenths % 10
        )
    } else {
        "Enable DaisyRose above to adjust its scale.".to_owned()
    };
    ui.label(RichText::new(help).small().weak());
    if available && let Some(summary) = loaded_daisyrose_summary(max_rank, sha256_prefix) {
        ui.label(RichText::new(summary).small().weak());
    }
    changed
}

fn cognitive_preferences(
    ui: &mut egui::Ui,
    cognitive: &mut crate::client::CognitiveOptions,
    runtime_supported: bool,
    thinking_enabled: bool,
    editable: bool,
) -> bool {
    use crate::client::{ExtraEffortStrategy, ReasoningEffort, SamplingPreset, ThinkingBudgetMode};

    ui.add_space(8.0);
    ui.label(RichText::new("Cognitive & sampling controls").strong());
    ui.add_space(6.0);
    let before = *cognitive;
    ui.add_enabled_ui(editable, |ui| {
        ui.horizontal(|ui| {
            let label = ui.label("Extra effort strategy");
            egui::ComboBox::from_id_salt("extra-effort-strategy")
                .selected_text(cognitive.extra_effort_strategy.label())
                .show_ui(ui, |ui| {
                    for option in [
                        ExtraEffortStrategy::PlanDraftReview,
                        ExtraEffortStrategy::MultiDraftSynthesis,
                    ] {
                        ui.selectable_value(
                            &mut cognitive.extra_effort_strategy,
                            option,
                            option.label(),
                        );
                    }
                })
                .response
                .labelled_by(label.id);
        });
    });
    ui.add_space(4.0);
    ui.add_enabled_ui(editable && runtime_supported, |ui| {
        ui.horizontal(|ui| {
            let label = ui.label("Reasoning effort");
            egui::ComboBox::from_id_salt("reasoning-effort")
                .selected_text(cognitive.reasoning_effort.label())
                .show_ui(ui, |ui| {
                    for option in [
                        ReasoningEffort::Low,
                        ReasoningEffort::Medium,
                        ReasoningEffort::XHigh,
                    ] {
                        ui.selectable_value(
                            &mut cognitive.reasoning_effort,
                            option,
                            option.label(),
                        );
                    }
                })
                .response
                .labelled_by(label.id);
        });
        ui.add_space(4.0);
        ui.add_enabled_ui(thinking_enabled, |ui| {
            ui.horizontal(|ui| {
                let label = ui.label("Thinking budget");
                egui::ComboBox::from_id_salt("thinking-budget")
                    .selected_text(cognitive.thinking_budget.label())
                    .show_ui(ui, |ui| {
                        for option in [
                            ThinkingBudgetMode::Unbounded,
                            ThinkingBudgetMode::Concise,
                            ThinkingBudgetMode::Balanced,
                            ThinkingBudgetMode::Deep,
                        ] {
                            ui.selectable_value(
                                &mut cognitive.thinking_budget,
                                option,
                                option.label(),
                            );
                        }
                    })
                    .response
                    .labelled_by(label.id);
            });
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let label = ui.label("Sampling preset");
            egui::ComboBox::from_id_salt("sampling-preset")
                .selected_text(cognitive.sampling_preset.label())
                .show_ui(ui, |ui| {
                    for option in [
                        SamplingPreset::Canonical,
                        SamplingPreset::Precise,
                        SamplingPreset::BalancedDry,
                        SamplingPreset::Creative,
                    ] {
                        ui.selectable_value(&mut cognitive.sampling_preset, option, option.label());
                    }
                })
                .response
                .labelled_by(label.id);
        });
    });
    if !runtime_supported {
        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "Reasoning effort, soft thinking budget, and calibrated sampling presets require \
                 a compatible runtime release.",
            )
            .small()
            .weak(),
        );
    }
    before != *cognitive
}
