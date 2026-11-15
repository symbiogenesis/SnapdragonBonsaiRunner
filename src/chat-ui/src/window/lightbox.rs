//! Full-window lightbox modal for inspecting and saving generated SVG images.
#![forbid(unsafe_code)]

use super::{
    controls,
    egui::{self, Color32, RichText, Vec2},
    image_cache::{ImageCache, PrehashedSvg, RasterSlot},
    transcript::aspect_fit_rect,
};

const HEADER_HEIGHT: f32 = 36.0;
const MODAL_PADDING: f32 = 16.0;
const CANVAS_PADDING: f32 = 8.0;

/// Active full-window lightbox modal state for one generated image.
pub(super) struct Lightbox {
    svg: PrehashedSvg,
    intrinsic_size: Vec2,
    suggested_stem: String,
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
    /// Creates a lightbox viewer for the selected inline image.
    pub(super) fn new(svg: PrehashedSvg, intrinsic_size: Vec2, suggested_stem: String) -> Self {
        Self {
            svg,
            intrinsic_size,
            suggested_stem,
        }
    }

    /// Renders the near-full-window lightbox overlay and returns any requested user action.
    pub(super) fn show(
        &self,
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
                modal_response.widget_info(|| {
                    egui::WidgetInfo::labeled(egui::Role::Dialog, true, "Expanded generated image")
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

                modal_ui.allocate_ui_with_layout(
                    egui::vec2(inner_rect.width(), HEADER_HEIGHT),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |header| {
                        header.heading("Generated image");
                        header.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |buttons| {
                                if buttons.add(controls::action_button("Close")).clicked() {
                                    action = Some(LightboxAction::Close);
                                }
                                let save_enabled =
                                    !export_busy && !image_cache.is_rejected(&self.svg);
                                let save = buttons
                                    .add_enabled(
                                        save_enabled,
                                        controls::quiet_button("Save image…"),
                                    )
                                    .on_hover_text("Save image as PNG or SVG");
                                if save.clicked() {
                                    action = Some(LightboxAction::Save {
                                        svg: self.svg.clone(),
                                        suggested_stem: self.suggested_stem.clone(),
                                    });
                                }
                            },
                        );
                    },
                );
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
                match image_cache.get_or_request(ctx, &self.svg, target_rect.size()) {
                    RasterSlot::Ready(tex) => paint_texture(&modal_ui, bounds, &tex),
                    RasterSlot::Loading {
                        fallback: Some(tex),
                    } => {
                        paint_texture(&modal_ui, bounds, &tex);
                        let badge = egui::Rect::from_min_size(
                            canvas_rect.max - egui::vec2(28.0, 28.0),
                            Vec2::splat(20.0),
                        );
                        egui::Spinner::new().paint_at(&modal_ui, badge);
                    }
                    RasterSlot::Loading { fallback: None } => {
                        egui::Spinner::new().paint_at(
                            &modal_ui,
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
                                        RichText::new(&*self.svg)
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
            });

        action
    }
}

fn paint_texture(ui: &egui::Ui, bounds: egui::Rect, texture: &egui::TextureHandle) {
    let fitted = aspect_fit_rect(bounds, texture.size_vec2());
    ui.painter().image(
        texture.id(),
        fitted,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        Color32::WHITE,
    );
}
