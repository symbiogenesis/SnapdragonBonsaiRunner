//! Composer attachment `+` popup menu, background import lifecycle and draft chips.
#![forbid(unsafe_code)]

use std::time::Duration;

use super::{State, controls, egui, error_ui, lightbox, transcript::aspect_fit_rect};
use crate::{
    attachment_import::{
        Attachment, MAX_ATTACHMENTS_PER_MESSAGE, MAX_TOTAL_TEXT_ATTACHMENT_BYTES, Outcome,
        PickerKind, Session,
    },
    client::MAX_REQUEST_IMAGES,
};
use eframe::egui::{Color32, RichText};

impl State {
    pub(super) fn start_attachment_import(&mut self, kind: PickerKind) {
        if self.attachment_import.is_some() || self.audio_import.is_some() || self.audio_file_active
        {
            self.attachment_notice = Some((
                "Wait for or cancel the active attachment import first. Your draft was kept."
                    .to_owned(),
                true,
            ));
            return;
        }
        if self.active_conversation_busy() {
            return;
        }
        let remaining_slots =
            MAX_ATTACHMENTS_PER_MESSAGE.saturating_sub(self.draft_attachments.len());
        if remaining_slots == 0 {
            self.attachment_notice = Some((
                format!(
                    "Each message supports up to {MAX_ATTACHMENTS_PER_MESSAGE} attachments. Remove one to add another."
                ),
                true,
            ));
            return;
        }
        let used_text: usize = self
            .draft_attachments
            .iter()
            .map(Attachment::text_bytes)
            .sum();
        let remaining_text = MAX_TOTAL_TEXT_ATTACHMENT_BYTES.saturating_sub(used_text);
        if kind == PickerKind::Text && remaining_text == 0 {
            self.attachment_notice = Some((
                "Text attachments reached the 512 KiB per-message limit. Remove one to add more."
                    .to_owned(),
                true,
            ));
            return;
        }
        match Session::start(
            kind,
            self.native_owner.unwrap_or_default(),
            remaining_slots,
            remaining_text,
        ) {
            Ok(session) => {
                self.attachment_import = Some(session);
                self.attachment_notice = None;
            }
            Err(_) => {
                self.attachment_notice = Some((
                    "Could not open the attachment importer. Your draft was kept.".to_owned(),
                    true,
                ));
            }
        }
    }

    pub(super) fn cancel_attachment_import(&mut self) {
        if let Some(mut session) = self.attachment_import.take() {
            session.cancel();
        }
    }

    pub(super) fn poll_attachment_import(&mut self, ctx: &egui::Context) {
        let Some(outcome) = self.attachment_import.as_mut().and_then(Session::poll) else {
            return;
        };
        self.attachment_import = None;
        match outcome {
            Outcome::Cancelled => {}
            Outcome::Failed(message) => {
                self.attachment_notice = Some((message, true));
            }
            Outcome::Imported {
                attachments,
                notice,
            } => {
                let cap_notice = self.append_draft_attachments(attachments);
                self.attachment_notice = cap_notice.or(notice).map(|text| (text, false));
            }
        }
        ctx.request_repaint();
    }

    /// Appends imported attachments strictly to the active draft without auto-sending.
    pub(super) fn append_draft_attachments(
        &mut self,
        attachments: Vec<Attachment>,
    ) -> Option<String> {
        let mut used_images: usize = self
            .committed
            .iter()
            .map(|message| message.images.len())
            .fold(0, usize::saturating_add)
            .saturating_add(
                self.draft_attachments
                    .iter()
                    .map(Attachment::image_count)
                    .fold(0, usize::saturating_add),
            );
        let mut added = false;
        let mut skipped_for_slots = 0_usize;
        let mut skipped_for_images = 0_usize;

        for attachment in attachments {
            if self.draft_attachments.len() >= MAX_ATTACHMENTS_PER_MESSAGE {
                skipped_for_slots += 1;
                continue;
            }
            let image_cost = attachment.image_count();
            if image_cost > 0 && used_images.saturating_add(image_cost) > MAX_REQUEST_IMAGES {
                skipped_for_images += 1;
                continue;
            }
            used_images = used_images.saturating_add(image_cost);
            self.draft_attachments.push(attachment);
            added = true;
        }
        if added {
            self.draft_attachments_revision = self.draft_attachments_revision.wrapping_add(1);
            self.retain_draft();
            self.touch_conversation(self.active_conversation);
            self.focus_composer = true;
        }
        if skipped_for_images > 0 {
            Some(format!(
                "Skipped {skipped_for_images} media attachment(s): each conversation request supports up to {MAX_REQUEST_IMAGES} images total."
            ))
        } else if skipped_for_slots > 0 {
            Some(format!(
                "Skipped {skipped_for_slots} attachment(s): each message supports up to {MAX_ATTACHMENTS_PER_MESSAGE} attachments."
            ))
        } else {
            None
        }
    }

    pub(super) fn attachment_menu_button(&mut self, ui: &mut egui::Ui) {
        if self.audio_file_active {
            if ui
                .add(controls::action_button("Cancel audio").selected(true))
                .accessible_name("Cancel audio file transcription")
                .on_hover_text(
                    "Cancel audio file transcription - Escape. Your draft and any retained transcript are kept.",
                )
                .clicked()
            {
                self.cancel_audio_file();
                self.speech_notice = false;
            }
            return;
        }
        if self.attachment_import.is_some() {
            if ui
                .add(controls::action_button("Cancel import").selected(true))
                .accessible_name("Cancel attachment import")
                .on_hover_text("Cancel attachment import - Escape. Your draft was kept.")
                .clicked()
            {
                self.cancel_attachment_import();
            }
            return;
        }

        let updates_installing = self
            .updates
            .as_ref()
            .is_some_and(super::updates::Updates::installing);
        let enabled = !updates_installing && self.audio_import.is_none();
        let icon_color = if enabled {
            ui.visuals().text_color()
        } else {
            ui.visuals().weak_text_color()
        };
        let plus_icon = egui::Atom::paint(egui::Vec2::splat(16.0), move |ui, args| {
            let c = args.rect.center();
            let half = 6.0;
            let stroke = egui::Stroke::new(1.75, icon_color);
            ui.painter().line_segment(
                [c - egui::vec2(half, 0.0), c + egui::vec2(half, 0.0)],
                stroke,
            );
            ui.painter().line_segment(
                [c - egui::vec2(0.0, half), c + egui::vec2(0.0, half)],
                stroke,
            );
        });
        let response = ui
            .add_enabled(
                enabled,
                egui::Button::new(plus_icon)
                    .min_size(egui::vec2(36.0, 36.0))
                    .corner_radius(8),
            )
            .accessible_name("Add attachment")
            .on_hover_text("Attach audio, text, or image and video files");

        let opening = response.clicked()
            && !egui::Popup::is_id_open(ui.ctx(), egui::Popup::default_response_id(&response));
        if opening {
            response.surrender_focus();
        }
        let popup_width = (ui.ctx().content_rect().width() - 24.0).clamp(190.0, 290.0);
        egui::Popup::menu(&response)
            .accessible_name("Attachment options")
            .align(egui::RectAlign::TOP_START)
            .width(popup_width)
            .gap(6.0)
            .frame(
                egui::Frame::popup(ui.style())
                    .corner_radius(12)
                    .inner_margin(10),
            )
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                ui.set_min_width(popup_width - 20.0);
                let asr_ready = !self.preview && self.audio_file_input;
                let asr_loading = !self.preview
                    && !self.audio_file_input
                    && self.canary_configured
                    && !self.speech_checked;
                let can_audio = asr_ready
                    && !self.model.busy()
                    && !self.speaking
                    && !self.dictation_requested
                    && self.audio_import.is_none();
                let can_draft_attach = !self.active_conversation_busy()
                    && self.attachment_import.is_none()
                    && self.audio_import.is_none();

                ui.horizontal(|ui| {
                    if attachment_option_button(
                        ui,
                        AttachmentOptionIcon::Audio,
                        "Audio files",
                        can_audio,
                    )
                    .clicked()
                    {
                        ui.close();
                        self.start_audio_import(crate::audio_import::Source::Picker {
                            owner: self.native_owner.unwrap_or_default(),
                        });
                    }
                    if !asr_ready {
                        if asr_loading {
                            ui.add(egui::Spinner::new().size(14.0));
                            ui.label(RichText::new("Loading ASR…").small().weak());
                            ui.ctx().request_repaint_after(Duration::from_millis(50));
                        } else {
                            ui.label(RichText::new("ASR unavailable").small().weak());
                        }
                    }
                });
                if attachment_option_button(
                    ui,
                    AttachmentOptionIcon::Text,
                    "Text files",
                    can_draft_attach,
                )
                .clicked()
                {
                    ui.close();
                    self.start_attachment_import(PickerKind::Text);
                }
                if attachment_option_button(
                    ui,
                    AttachmentOptionIcon::Media,
                    "Images and Videos",
                    can_draft_attach,
                )
                .clicked()
                {
                    ui.close();
                    self.start_attachment_import(PickerKind::Media);
                }
            });
    }

    pub(super) fn composer_attachments(&mut self, ui: &mut egui::Ui) {
        if self.attachment_import.is_some() {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new().size(14.0));
                ui.label(RichText::new("Importing attachment…").small().weak());
            });
            ui.add_space(4.0);
        }
        if let Some((message, is_error)) = self.attachment_notice.clone() {
            ui.horizontal_wrapped(|ui| {
                if is_error {
                    error_ui::summary(ui, &message);
                } else {
                    ui.label(RichText::new(&message).small().weak());
                }
                if ui.small_button("Dismiss").clicked() {
                    self.attachment_notice = None;
                }
            });
            ui.add_space(4.0);
        }
        if self.draft_attachments.is_empty() {
            return;
        }
        let mut remove_index = None;
        let mut inspect_index = None;
        ui.horizontal_wrapped(|ui| {
            for (index, attachment) in self.draft_attachments.iter().enumerate() {
                let chip_frame = egui::Frame::new()
                    .fill(ui.visuals().extreme_bg_color)
                    .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                    .corner_radius(8)
                    .inner_margin(egui::Margin::symmetric(8, 5));
                chip_frame.show(ui, |ui| {
                    ui.set_max_width(220.0);
                    ui.horizontal(|ui| {
                        if let Some(png) = attachment.thumbnail_png()
                            && let Some(tex) = self.image_cache.png_bytes_texture(ui.ctx(), png)
                        {
                            let (_, thumb_rect) = ui.allocate_space(egui::vec2(28.0, 28.0));
                            let fitted = aspect_fit_rect(thumb_rect, tex.size_vec2());
                            ui.painter().image(
                                tex.id(),
                                fitted,
                                egui::Rect::from_min_max(
                                    egui::pos2(0.0, 0.0),
                                    egui::pos2(1.0, 1.0),
                                ),
                                Color32::WHITE,
                            );
                        } else {
                            let badge = if attachment.is_media() { "IMG" } else { "TXT" };
                            ui.label(RichText::new(badge).small().strong().weak());
                        }
                        let summary = attachment_summary(attachment);
                        ui.vertical(|ui| {
                            let name_button = controls::quiet_button(
                                RichText::new(truncate_name(attachment.name(), 22)).small(),
                            );
                            if attachment.is_media() {
                                if ui
                                    .add(name_button)
                                    .accessible_name(format!("Inspect {}", attachment.name()))
                                    .clicked()
                                {
                                    inspect_index = Some(index);
                                }
                            } else {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(truncate_name(attachment.name(), 22)).small(),
                                    )
                                    .truncate(),
                                );
                            }
                            ui.label(RichText::new(summary).small().weak());
                        });
                        if ui
                            .add(controls::quiet_button("×"))
                            .accessible_name(format!("Remove {}", attachment.name()))
                            .on_hover_text("Remove attachment")
                            .clicked()
                        {
                            remove_index = Some(index);
                        }
                    });
                });
            }
        });
        ui.add_space(6.0);
        if let Some(index) = remove_index {
            self.draft_attachments.remove(index);
            self.draft_attachments_revision = self.draft_attachments_revision.wrapping_add(1);
            self.retain_draft();
            self.touch_conversation(self.active_conversation);
            self.focus_composer = true;
        } else if let Some(index) = inspect_index
            && let Some(attachment) = self.draft_attachments.get(index)
        {
            self.lightbox = lightbox::Lightbox::from_attachment(attachment);
        }
    }
}

pub(super) fn attachment_summary(attachment: &Attachment) -> String {
    match attachment {
        Attachment::Text { text, .. } => {
            if text.len() >= 1024 {
                format!("{} KiB text", text.len().div_ceil(1024))
            } else {
                format!("{} B text", text.len())
            }
        }
        Attachment::Image { width, height, .. } => format!("{width}×{height} image"),
        Attachment::Video { duration_ms, .. } => {
            let frames = attachment.frame_count();
            format!(
                "{}.{}s · {frames} frames",
                duration_ms / 1000,
                (duration_ms % 1000) / 100
            )
        }
    }
}

pub(super) fn truncate_name(name: &str, max_chars: usize) -> String {
    let mut chars = name.chars();
    let collected: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{collected}…")
    } else {
        collected
    }
}

#[derive(Clone, Copy)]
enum AttachmentOptionIcon {
    Audio,
    Text,
    Media,
}

fn attachment_option_button(
    ui: &mut egui::Ui,
    icon: AttachmentOptionIcon,
    label: &'static str,
    enabled: bool,
) -> egui::Response {
    let icon_atom = egui::Atom::paint(egui::Vec2::splat(16.0), move |ui, args| {
        paint_attachment_option_icon(ui, args.rect, icon, args.fallback_text_color);
    });
    ui.add_enabled(
        enabled,
        egui::Button::selectable(false, (icon_atom, label))
            .gap(8.0)
            .corner_radius(8),
    )
    .accessible_name(label)
}

fn paint_attachment_option_icon(
    ui: &egui::Ui,
    rect: egui::Rect,
    icon: AttachmentOptionIcon,
    color: Color32,
) {
    let painter = ui.painter_at(rect);
    let point = |x, y| rect.min + egui::vec2(x, y);
    let stroke = egui::Stroke::new(1.4, color);
    match icon {
        AttachmentOptionIcon::Audio => {
            for (x, half_h) in [
                (2.5_f32, 2.5_f32),
                (5.5, 5.5),
                (8.0, 3.5),
                (10.5, 6.0),
                (13.5, 3.0),
            ] {
                painter.line_segment([point(x, 8.0 - half_h), point(x, 8.0 + half_h)], stroke);
            }
        }
        AttachmentOptionIcon::Text => {
            painter.add(egui::Shape::closed_line(
                vec![
                    point(3.0, 1.5),
                    point(10.0, 1.5),
                    point(13.0, 4.5),
                    point(13.0, 14.5),
                    point(3.0, 14.5),
                ],
                stroke,
            ));
            painter.add(egui::Shape::line(
                vec![point(10.0, 1.5), point(10.0, 4.5), point(13.0, 4.5)],
                stroke,
            ));
            painter.line_segment([point(5.5, 8.0), point(10.5, 8.0)], stroke);
            painter.line_segment([point(5.5, 11.0), point(9.5, 11.0)], stroke);
        }
        AttachmentOptionIcon::Media => {
            painter.rect_stroke(
                egui::Rect::from_min_max(point(1.5, 2.5), point(14.5, 13.5)),
                2.0,
                stroke,
                egui::StrokeKind::Middle,
            );
            painter.circle_stroke(point(5.2, 6.2), 1.3, stroke);
            painter.add(egui::Shape::line(
                vec![
                    point(2.5, 12.5),
                    point(6.5, 8.5),
                    point(9.2, 11.0),
                    point(11.5, 8.8),
                    point(14.0, 11.8),
                ],
                stroke,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::{
        Harness,
        kittest::{NodeT as _, Queryable as _},
    };

    #[test]
    fn attachment_option_buttons_paint_consistent_vector_icons_for_each_kind() {
        for (icon, label, enabled) in [
            (AttachmentOptionIcon::Audio, "Audio files", false),
            (AttachmentOptionIcon::Audio, "Audio files", true),
            (AttachmentOptionIcon::Text, "Text files", true),
            (AttachmentOptionIcon::Media, "Images and Videos", true),
        ] {
            let mut baseline = Harness::builder()
                .with_size([260.0, 60.0])
                .with_context_setup(super::super::configure)
                .build_ui(|ui| {
                    ui.add_enabled(
                        enabled,
                        egui::Button::selectable(false, label)
                            .gap(8.0)
                            .corner_radius(8),
                    )
                    .accessible_name(label);
                });
            baseline.step();
            let baseline_width = baseline
                .get_by_role_and_label(egui::Role::Button, label)
                .rect()
                .width();

            let mut with_icon = Harness::builder()
                .with_size([260.0, 60.0])
                .with_context_setup(super::super::configure)
                .build_ui(|ui| {
                    attachment_option_button(ui, icon, label, enabled);
                });
            with_icon.step();

            let node = with_icon.get_by_role_and_label(egui::Role::Button, label);
            let icon_width = node.rect().width();
            assert_eq!(node.accesskit_node().is_disabled(), !enabled);
            assert!(
                icon_width >= baseline_width + 20.0,
                "{label} ({enabled}): expected 16pt icon + 8pt gap width increase, got {icon_width} vs {baseline_width}"
            );
        }
    }
}
