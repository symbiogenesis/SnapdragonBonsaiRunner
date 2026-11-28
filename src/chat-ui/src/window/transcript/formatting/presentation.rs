//! Native reading typography, independent of the offline parser's style hints.
use super::egui;
use std::{ops::Range, sync::Arc};

pub(super) const CODE_BLOCK: &str = "Bonsai Code Block";

fn block_code(format: &egui::TextFormat) -> bool {
    matches!(&format.font_id.family, egui::FontFamily::Name(name) if name.as_ref() == CODE_BLOCK)
}

pub(super) fn gap_format(base: &egui::TextFormat, size: f32) -> egui::TextFormat {
    egui::TextFormat {
        font_id: egui::FontId::proportional(1.0),
        line_height: Some(size * 0.65),
        ..base.clone()
    }
}

pub(super) fn visit_runs(
    text: &str,
    prev_ended_with_newline: bool,
    format: &egui::TextFormat,
    mut visit: impl FnMut(Range<usize>, bool) -> bool,
) -> bool {
    if text.is_empty() {
        return true;
    }
    if block_code(format) || !text.contains('\n') {
        return visit(0..text.len(), false);
    }
    let bytes = text.as_bytes();
    let mut cursor = 0_usize;
    for pos in memchr::memchr_iter(b'\n', bytes) {
        let prev_is_newline = if pos > 0 {
            bytes[pos - 1] == b'\n'
        } else {
            prev_ended_with_newline
        };
        if prev_is_newline {
            if cursor < pos && !visit(cursor..pos, false) {
                return false;
            }
            if !visit(pos..pos + 1, true) {
                return false;
            }
            cursor = pos + 1;
        }
    }
    if cursor < text.len() {
        visit(cursor..text.len(), false)
    } else {
        true
    }
}

pub(super) fn body_format(size: f32, visuals: &egui::Visuals) -> egui::TextFormat {
    egui::TextFormat {
        line_height: Some(size * 1.55),
        ..egui::TextFormat::simple(egui::FontId::proportional(size), visuals.text_color())
    }
}

pub(super) fn text_format(
    span: super::wire::StyleSpan,
    size: f32,
    visuals: &egui::Visuals,
) -> egui::TextFormat {
    let strong = span.is_strong();
    let code_block = span.is_code_block();
    let code = code_block || span.is_code();
    let font_size = f32::from(span.size()) * (size / 16.0);
    let family = if code_block {
        egui::FontFamily::Name(CODE_BLOCK.into())
    } else if code {
        egui::FontFamily::Monospace
    } else if strong {
        egui::FontFamily::Name("Bonsai Strong".into())
    } else {
        egui::FontFamily::Proportional
    };
    let color = if strong {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    let mut format = egui::TextFormat::simple(egui::FontId::new(font_size, family), color);
    format.background = if code && !code_block {
        visuals.code_bg_color
    } else {
        egui::Color32::TRANSPARENT
    };
    format.expand_bg = if code { 2.0 } else { 1.0 };
    format.line_height = Some(if span.size() > 16 {
        font_size * 1.3
    } else if code {
        size * 1.5
    } else {
        size * 1.55
    });
    format.italics = span.is_italic();
    if span.is_strike() {
        format.strikethrough = egui::Stroke::new(1.0, color);
    }
    format
}

/// Paragraph gaps use less space than a full empty line of reading text. This
/// changes no displayed bytes, so source Find and selection retain their maps.
pub(super) fn append(
    job: &mut egui::text::LayoutJob,
    text: &str,
    format: egui::TextFormat,
    size: f32,
) {
    if text.is_empty() {
        return;
    }
    if block_code(&format) || !text.contains('\n') {
        job.append(text, 0.0, format);
        return;
    }
    let gap = gap_format(&format, size);
    visit_runs(text, job.text.ends_with('\n'), &format, |sub, is_gap| {
        job.append(
            &text[sub],
            0.0,
            if is_gap { gap.clone() } else { format.clone() },
        );
        true
    });
}

/// Cache full-width code surfaces with the galley. Walk rendered UTF-8 once;
/// reflow and Find rebuild these bounds with their owning text layout.
pub(super) fn code_panels(galley: &egui::Galley) -> Arc<[egui::Rect]> {
    if !galley
        .job
        .sections
        .iter()
        .any(|section| block_code(&section.format))
    {
        return Arc::default();
    }
    let mut panels: Vec<egui::Rect> = Vec::new();
    let mut byte = 0;
    let mut section_index = 0;
    let mut previous_code = false;
    for row in &galley.rows {
        let end = byte
            + row
                .glyphs
                .iter()
                .map(|glyph| glyph.chr.len_utf8())
                .sum::<usize>()
            + usize::from(row.ends_with_newline);
        while galley
            .job
            .sections
            .get(section_index)
            .is_some_and(|section| section.byte_range.end.0 <= byte)
        {
            section_index += 1;
        }
        let code = galley.job.sections[section_index..]
            .iter()
            .take_while(|section| section.byte_range.start.0 < end)
            .any(|section| block_code(&section.format));
        if code {
            let rect = egui::Rect::from_min_max(
                egui::pos2(0.0, row.rect().top()),
                egui::pos2(galley.job.wrap.max_width, row.rect().bottom()),
            );
            if previous_code && let Some(previous) = panels.last_mut() {
                *previous = previous.union(rect);
            } else {
                panels.push(rect);
            }
        }
        previous_code = code;
        byte = end;
    }
    Arc::from(panels)
}
