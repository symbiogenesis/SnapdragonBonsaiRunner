//! Microphone presentation has no access to capture or application state.
use super::{controls, egui};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Idle,
    Preparing,
    Recording,
    Finishing,
}

pub(super) fn button(ui: &mut egui::Ui, phase: Phase, enabled: bool) -> egui::Response {
    let active = matches!(phase, Phase::Recording | Phase::Finishing);
    let hover_text = match phase {
        Phase::Finishing => "Cancel voice message · Escape",
        Phase::Recording => "Finish recording and transcribe · Escape cancels",
        Phase::Preparing => "Preparing speech automatically.",
        Phase::Idle => "Record a voice message · Qwen3-ASR 1.7B",
    };
    let response = ui
        .add_enabled(
            enabled,
            controls::action_button("")
                .min_size(egui::Vec2::splat(36.0))
                .selected(active),
        )
        .accessible_name("Microphone")
        .on_hover_text(hover_text)
        .on_disabled_hover_text(hover_text);
    let center = response.rect.center();
    let stroke = if active && response.enabled() {
        ui.visuals().selection.stroke
    } else {
        ui.style().interact(&response).fg_stroke
    };
    if matches!(phase, Phase::Preparing | Phase::Finishing) {
        egui::Spinner::new().color(stroke.color).paint_at(
            ui,
            egui::Rect::from_center_size(center, egui::Vec2::splat(20.0)),
        );
    } else if phase == Phase::Recording {
        let painter = ui.painter_at(response.rect);
        painter.rect_filled(
            egui::Rect::from_center_size(center, egui::Vec2::splat(12.0)),
            2.0,
            stroke.color,
        );
    } else {
        let painter = ui.painter_at(response.rect);
        painter.rect_stroke(
            egui::Rect::from_center_size(center - egui::vec2(0.0, 3.0), egui::vec2(7.0, 12.0)),
            4.0,
            stroke,
            egui::StrokeKind::Middle,
        );
        let cradle = [
            (-7.0, -2.0),
            (-7.0, 0.0),
            (-6.5, 2.7),
            (-5.0, 5.0),
            (-2.7, 6.5),
            (0.0, 7.0),
            (2.7, 6.5),
            (5.0, 5.0),
            (6.5, 2.7),
            (7.0, 0.0),
            (7.0, -2.0),
        ]
        .map(|(x, y)| center + egui::vec2(x, y));
        for pair in cradle.windows(2) {
            painter.line_segment([pair[0], pair[1]], stroke);
        }
        painter.line_segment(
            [
                center + egui::vec2(0.0, 7.0),
                center + egui::vec2(0.0, 10.0),
            ],
            stroke,
        );
        painter.line_segment(
            [
                center + egui::vec2(-4.0, 10.0),
                center + egui::vec2(4.0, 10.0),
            ],
            stroke,
        );
    }
    response
}
