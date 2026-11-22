//! Full-window lightbox modal for inspecting generated SVG artwork and attached media.
#![forbid(unsafe_code)]

use std::sync::Arc;

use super::{
    controls,
    egui::{self, Color32, RichText, Vec2},
    image_cache::{ImageCache, PrehashedSvg, RasterSlot},
    transcript::{aspect_fit_rect, paint_fitted_texture},
};
use crate::attachment_import::Attachment;

const HEADER_HEIGHT: f32 = 36.0;
const MODAL_PADDING: f32 = 16.0;
const CANVAS_PADDING: f32 = 8.0;

enum Content {
    Svg {
        svg: PrehashedSvg,
        suggested_stem: String,
    },
    Media {
        title: String,
        frames: Vec<(String, Arc<str>)>,
        current: usize,
    },
}

/// Active full-window lightbox modal state for one generated image or media attachment.
pub(super) struct Lightbox {
    intrinsic_size: Vec2,
    content: Content,
}

/// User action emitted by the lightbox modal during a frame.
pub(super) enum LightboxAction {
    /// Dismiss the modal and return focus/input to the main workspace.
    Close,
    /// Start exporting the viewed SVG via the native Save dialog.
    Save {
        svg: PrehashedSvg,
        suggested_stem: String,
    },
}

impl Lightbox {
    /// Creates a lightbox viewer for the selected inline SVG image.
    pub(super) fn new(svg: PrehashedSvg, intrinsic_size: Vec2, suggested_stem: String) -> Self {
        Self {
            intrinsic_size,
            content: Content::Svg {
                svg,
                suggested_stem,
            },
        }
    }

    /// Creates a lightbox viewer for an attached image or multi-keyframe video.
    pub(super) fn from_attachment(attachment: &Attachment) -> Option<Self> {
        let urls = attachment.image_data_urls();
        let total = attachment.frame_count();
        match attachment {
            Attachment::Text { .. } => None,
            Attachment::Image {
                name,
                width,
                height,
                ..
            } => {
                let url = urls.into_iter().next()?;
                Some(Self {
                    intrinsic_size: dimension_vec(*width, *height),
                    content: Content::Media {
                        title: name.clone(),
                        frames: vec![(String::new(), url)],
                        current: 0,
                    },
                })
            }
            Attachment::Video {
                name,
                width,
                height,
                frames,
                ..
            } => {
                let entries: Vec<(String, Arc<str>)> = frames
                    .iter()
                    .zip(urls)
                    .enumerate()
                    .map(|(index, (frame, url))| {
                        let label = format!(
                            "Frame {} / {} · {}.{}s",
                            index + 1,
                            total,
                            frame.timestamp_ms / 1000,
                            (frame.timestamp_ms % 1000) / 100
                        );
                        (label, url)
                    })
                    .collect();
                if entries.is_empty() {
                    return None;
                }
                Some(Self {
                    intrinsic_size: dimension_vec(*width, *height),
                    content: Content::Media {
                        title: name.clone(),
                        frames: entries,
                        current: 0,
                    },
                })
            }
        }
    }

    /// Renders the near-full-window lightbox overlay and returns any requested user action.
    pub(super) fn show(
        &mut self,
        ctx: &egui::Context,
        image_cache: &mut ImageCache,
        export_busy: bool,
    ) -> Option<LightboxAction> {
        let viewport = ctx.content_rect();
        if !viewport.is_positive() {
            return None;
        }
        let modal_width =
            (viewport.width() * 0.94).clamp(viewport.width().min(280.0), viewport.width());
        let modal_height =
            (viewport.height() * 0.92).clamp(viewport.height().min(240.0), viewport.height());
        let modal_rect =
            egui::Rect::from_center_size(viewport.center(), egui::vec2(modal_width, modal_height));
        let mut action = None;

        egui::Area::new(egui::Id::unique("image-lightbox"))
            .order(egui::Order::Foreground)
            .fixed_pos(viewport.min)
            .show(ctx, |ui| {
                let scrim = ui.allocate_rect(viewport, egui::Sense::click());
                ui.painter()
                    .rect_filled(viewport, 0.0, Color32::from_black_alpha(196));
                if scrim.clicked()
                    && scrim
                        .interact_pointer_pos()
                        .is_some_and(|pos| !modal_rect.contains(pos))
                {
                    action = Some(LightboxAction::Close);
                }

                let visuals = ui.visuals().clone();
                ui.painter().rect(
                    modal_rect,
                    12.0,
                    visuals.panel_fill,
                    visuals.widgets.noninteractive.bg_stroke,
                    egui::StrokeKind::Inside,
                );
                let modal_response = ui.interact(
                    modal_rect,
                    egui::Id::unique("image-lightbox-dialog"),
                    egui::Sense::click(),
                );
                let dialog_label = match &self.content {
                    Content::Svg { .. } => "Expanded generated image",
                    Content::Media { .. } => "Attachment preview",
                };
                modal_response.widget_info(|| {
                    egui::WidgetInfo::labeled(egui::Role::Dialog, true, dialog_label)
                });

                let inner_rect = modal_rect.shrink(MODAL_PADDING);
                if !inner_rect.is_positive() {
                    return;
                }
                let mut modal_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .id_salt("image-lightbox-body")
                        .max_rect(inner_rect)
                        .layout(egui::Layout::top_down(egui::Align::Min)),
                );
                modal_ui.set_clip_rect(inner_rect);

                self.header(&mut modal_ui, image_cache, export_busy, &mut action);
                modal_ui.separator();
                modal_ui.add_space(CANVAS_PADDING);

                let canvas_rect = modal_ui.available_rect_before_wrap();
                let _allocated = modal_ui.allocate_rect(canvas_rect, egui::Sense::hover());
                modal_ui
                    .painter()
                    .rect_filled(canvas_rect, 8.0, visuals.extreme_bg_color);
                if canvas_rect.width() <= 16.0 || canvas_rect.height() <= 16.0 {
                    return;
                }
                let target_rect =
                    aspect_fit_rect(canvas_rect.shrink(CANVAS_PADDING), self.intrinsic_size);
                let bounds = canvas_rect.shrink(CANVAS_PADDING);
                match &self.content {
                    Content::Svg { svg, .. } => {
                        show_svg_canvas(
                            ctx,
                            &mut modal_ui,
                            &visuals,
                            image_cache,
                            svg,
                            target_rect,
                            bounds,
                            canvas_rect,
                        );
                    }
                    Content::Media {
                        frames, current, ..
                    } => {
                        if let Some((_, url)) = frames.get(*current)
                            && let Some(tex) = image_cache.data_url_texture(ctx, url)
                        {
                            paint_fitted_texture(&modal_ui, bounds, &tex);
                        } else {
                            let mut unavailable_ui =
                                modal_ui.new_child(egui::UiBuilder::new().max_rect(bounds).layout(
                                    egui::Layout::centered_and_justified(egui::Direction::TopDown),
                                ));
                            unavailable_ui.label(
                                RichText::new("Attachment preview unavailable.")
                                    .small()
                                    .weak(),
                            );
                        }
                    }
                }
            });

        action
    }

    fn header(
        &mut self,
        modal_ui: &mut egui::Ui,
        image_cache: &ImageCache,
        export_busy: bool,
        action: &mut Option<LightboxAction>,
    ) {
        modal_ui.allocate_ui_with_layout(
            egui::vec2(modal_ui.available_width(), HEADER_HEIGHT),
            egui::Layout::left_to_right(egui::Align::Center),
            |header| {
                match &self.content {
                    Content::Svg { .. } => {
                        header.heading("Generated image");
                    }
                    Content::Media {
                        title,
                        frames,
                        current,
                    } => {
                        header.heading(title);
                        if let Some((frame_label, _)) = frames.get(*current)
                            && !frame_label.is_empty()
                        {
                            header.label(RichText::new(frame_label).small().weak());
                        }
                    }
                }
                header.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |buttons| {
                        if buttons.add(controls::action_button("Close")).clicked() {
                            *action = Some(LightboxAction::Close);
                        }
                        match &mut self.content {
                            Content::Svg {
                                svg,
                                suggested_stem,
                            } => {
                                let save_enabled = !export_busy && !image_cache.is_rejected(svg);
                                let save = buttons
                                    .add_enabled(
                                        save_enabled,
                                        controls::quiet_button("Save image…"),
                                    )
                                    .on_hover_text("Save image as PNG or SVG");
                                if save.clicked() {
                                    *action = Some(LightboxAction::Save {
                                        svg: svg.clone(),
                                        suggested_stem: suggested_stem.clone(),
                                    });
                                }
                            }
                            Content::Media {
                                frames, current, ..
                            } if frames.len() > 1 => {
                                if buttons.add(controls::quiet_button("Next frame")).clicked() {
                                    *current = (*current + 1) % frames.len();
                                }
                                if buttons
                                    .add(controls::quiet_button("Previous frame"))
                                    .clicked()
                                {
                                    *current = current.checked_sub(1).unwrap_or(frames.len() - 1);
                                }
                            }
                            Content::Media { .. } => {}
                        }
                    },
                );
            },
        );
    }
}

fn dimension_vec(width: u32, height: u32) -> Vec2 {
    let clamped_w = u16::try_from(width).unwrap_or(u16::MAX).max(1);
    let clamped_h = u16::try_from(height).unwrap_or(u16::MAX).max(1);
    egui::vec2(f32::from(clamped_w), f32::from(clamped_h))
}

#[expect(
    clippy::too_many_arguments,
    reason = "SVG canvas helper isolates fallback scrolling from modal framing."
)]
fn show_svg_canvas(
    ctx: &egui::Context,
    modal_ui: &mut egui::Ui,
    visuals: &egui::Visuals,
    image_cache: &mut ImageCache,
    svg: &PrehashedSvg,
    target_rect: egui::Rect,
    bounds: egui::Rect,
    canvas_rect: egui::Rect,
) {
    match image_cache.get_or_request(ctx, svg, target_rect.size()) {
        RasterSlot::Ready(tex) => paint_fitted_texture(modal_ui, bounds, &tex),
        RasterSlot::Loading {
            fallback: Some(tex),
        } => {
            paint_fitted_texture(modal_ui, bounds, &tex);
            let badge = egui::Rect::from_min_size(
                canvas_rect.max - egui::vec2(28.0, 28.0),
                Vec2::splat(20.0),
            );
            egui::Spinner::new().paint_at(modal_ui, badge);
        }
        RasterSlot::Loading { fallback: None } => {
            egui::Spinner::new().paint_at(
                modal_ui,
                egui::Rect::from_center_size(canvas_rect.center(), Vec2::splat(28.0)),
            );
        }
        RasterSlot::FallbackSource => {
            let text_rect = canvas_rect.shrink(12.0);
            let mut fallback_ui = modal_ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(text_rect)
                    .layout(egui::Layout::top_down(egui::Align::Min)),
            );
            fallback_ui.set_clip_rect(text_rect);
            fallback_ui.label(
                RichText::new("SVG preview unavailable — source:")
                    .small()
                    .weak(),
            );
            fallback_ui.add_space(4.0);
            egui::ScrollArea::vertical()
                .id_salt("lightbox-svg-source")
                .show(&mut fallback_ui, |ui| {
                    ui.add(
                        egui::Label::new(
                            RichText::new(&**svg)
                                .monospace()
                                .small()
                                .color(visuals.weak_text_color()),
                        )
                        .selectable(true)
                        .wrap(),
                    );
                });
        }
    }
}
