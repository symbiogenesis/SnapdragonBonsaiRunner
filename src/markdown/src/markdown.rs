//! Render only the documented native subset; other constructs retain source.
use super::{Builder, ImageBlock};
use markdown_syntax::{Block, CodeBlock, CodeBlockKind, FenceMarker, Inline};

impl Builder<'_> {
    pub(super) fn blocks(
        &mut self,
        blocks: &[Block],
        format: &egui::TextFormat,
        depth: usize,
        tight: bool,
    ) {
        for (index, block) in blocks.iter().enumerate() {
            if self.limited {
                return;
            }
            if depth == 0
                && self.quote_depth == 0
                && self.images_emitted < super::svg::MAX_SVG_IMAGES
                && let Block::CodeBlock(node) = block
                && let Some(image) = self.extract_svg_image(node)
            {
                self.push_image(image);
                continue;
            }
            if index != 0 {
                self.newlines(if tight { 1 } else { 2 });
            }
            match block {
                Block::Paragraph(node) => self.inlines(&node.children, format),
                Block::Heading(node) => {
                    let mut heading = format.clone();
                    heading.font_id.size = match node.depth {
                        1 => 28.0,
                        2 => 24.0,
                        3 => 21.0,
                        4 => 19.0,
                        5 => 17.0,
                        _ => 16.0,
                    } * (self.normal.font_id.size / 16.0);
                    heading.color = self.visuals.strong_text_color();
                    self.inlines(&node.children, &heading);
                }
                Block::BlockQuote(node) => {
                    self.quote_depth += 1;
                    self.blocks(&node.children, format, depth, false);
                    self.quote_depth -= 1;
                }
                Block::List(node) => {
                    let mut number = node
                        .children
                        .first()
                        .and_then(|item| item.number)
                        .unwrap_or(1);
                    for (index, item) in node.children.iter().enumerate() {
                        if index != 0 {
                            self.newlines(if node.tight { 1 } else { 2 });
                        }
                        self.generated(&"    ".repeat(depth), format);
                        if node.ordered {
                            self.generated(&format!("{number}. "), format);
                            number = number.saturating_add(1);
                        } else {
                            self.generated("• ", format);
                        }
                        if let Some(checked) = item.checked {
                            self.generated(if checked { "☑ " } else { "☐ " }, format);
                        }
                        self.blocks(&item.children, format, depth + 1, node.tight);
                    }
                }
                Block::CodeBlock(node) => {
                    let mut code = self.code_format(format);
                    // A finite wire flag lets the trusted UI distinguish a
                    // code surface from an inline monospace span.
                    code.font_id.family = egui::FontFamily::Name("Bonsai Code Block".into());
                    // Map each de-indented code line back to its actual source,
                    // excluding the fence/info string and container prefixes.
                    let mut start = node.meta.span.map_or(0, |span| span.start);
                    let end = node.meta.span.map_or(start, |span| span.end);
                    if matches!(node.kind, CodeBlockKind::Fenced { .. }) {
                        start += self
                            .source
                            .get(start..end)
                            .and_then(|source| source.find('\n'))
                            .map_or(end - start, |offset| offset + 1);
                    }
                    for line in node.value.split_inclusive('\n') {
                        let Some(rest) = self.source.get(start..end) else {
                            self.limited = true;
                            return;
                        };
                        let physical_end = rest.find('\n').map_or(end, |offset| start + offset + 1);
                        let value = line.trim_end_matches('\n');
                        if !value.is_empty() {
                            let range = self
                                .source
                                .get(start..physical_end)
                                .and_then(|source| source.find(value))
                                .map(|offset| start + offset..start + offset + value.len());
                            let Some(range) = range else {
                                // Split indentation tabs can synthesize spaces.
                                // Keep exact source rather than search later lines
                                // repeatedly or associate text with the wrong line.
                                self.limited = true;
                                return;
                            };
                            self.append(value, Some(range), &code);
                        }
                        start = physical_end;
                        if line.ends_with('\n') {
                            self.generated("\n", &code);
                        }
                    }
                }
                Block::ThematicBreak(_) => self.generated("────", format),
                Block::Table(node) => {
                    for (row_index, row) in node.rows.iter().enumerate() {
                        if row_index != 0 {
                            self.newlines(1);
                        }
                        let mut cell_format = format.clone();
                        if row_index == 0 {
                            cell_format.color = self.visuals.strong_text_color();
                        }
                        for (cell_index, cell) in row.cells.iter().enumerate() {
                            if cell_index != 0 {
                                self.generated("  │  ", format);
                            }
                            self.inlines(&cell.children, &cell_format);
                        }
                    }
                }
                Block::BlankLine(_) => self.newlines(2),
                Block::Definition(_) => {}
                // HTML and extensions stay visible and inert. Never route any
                // model output into a browser, image loader or code executor.
                _ => self.literal(block.span(), format),
            }
        }
    }

    fn extract_svg_image(&self, node: &CodeBlock) -> Option<ImageBlock> {
        let CodeBlockKind::Fenced { marker, length } = node.kind else {
            return None;
        };
        let language = node.info.as_deref()?.split_whitespace().next()?;
        if !language.eq_ignore_ascii_case("svg") {
            return None;
        }
        let span = node.meta.span?;
        let block = self.source.get(span.start..span.end)?;
        let open_end_rel = block.find('\n')? + 1;
        let content_start = span.start.checked_add(open_end_rel)?;
        let after_open = block.get(open_end_rel..)?;
        let without_trailing_nl = after_open.strip_suffix('\n').unwrap_or(after_open);
        let without_trailing_nl = without_trailing_nl
            .strip_suffix('\r')
            .unwrap_or(without_trailing_nl);
        let last_line_start_rel = without_trailing_nl.rfind('\n').map_or(0, |pos| pos + 1);
        let last_line = without_trailing_nl.get(last_line_start_rel..)?;
        let marker_char = match marker {
            FenceMarker::Backtick => '`',
            FenceMarker::Tilde => '~',
        };
        let trimmed = last_line.trim_start_matches(' ');
        let leading_spaces = last_line.len() - trimmed.len();
        let fence_count = trimmed.chars().take_while(|ch| *ch == marker_char).count();
        let after_fence = trimmed.get(fence_count..)?;
        if leading_spaces > 3
            || fence_count < length
            || !after_fence.bytes().all(|byte| matches!(byte, b' ' | b'\t'))
        {
            return None;
        }
        let content_end = content_start.checked_add(last_line_start_rel)?;
        if content_start >= content_end {
            return None;
        }
        let svg = self.source.get(content_start..content_end)?;
        let (intrinsic_width, intrinsic_height) = super::svg::inspect(svg)?;
        Some(ImageBlock {
            source: content_start..content_end,
            intrinsic_width,
            intrinsic_height,
        })
    }

    fn code_format(&self, format: &egui::TextFormat) -> egui::TextFormat {
        let mut code = format.clone();
        code.font_id = egui::FontId::monospace(format.font_id.size);
        code.background = self.visuals.code_bg_color;
        code.italics = false;
        code
    }

    fn inlines(&mut self, inlines: &[Inline], format: &egui::TextFormat) {
        for inline in inlines {
            if self.limited {
                return;
            }
            match inline {
                Inline::Text(node) => self.value(&node.value, node.meta.span, format),
                Inline::Escape(node) => {
                    self.value(node.value.encode_utf8(&mut [0; 4]), node.meta.span, format)
                }
                Inline::CharacterReference(node) => {
                    if let Some(value) = node.value() {
                        self.value(&value, node.meta.span, format);
                    } else {
                        self.literal(node.meta.span, format);
                    }
                }
                Inline::Emphasis(node) => {
                    let mut emphasis = format.clone();
                    emphasis.italics = true;
                    self.inlines(&node.children, &emphasis);
                }
                Inline::Strong(node) => {
                    let mut strong = format.clone();
                    strong.color = self.visuals.strong_text_color();
                    self.inlines(&node.children, &strong);
                }
                Inline::Delete(node) => {
                    let mut deleted = format.clone();
                    deleted.strikethrough = egui::Stroke::new(1.0, format.color);
                    self.inlines(&node.children, &deleted);
                }
                Inline::Code(node) => {
                    let code = self.code_format(format);
                    let span = node.meta.span.and_then(|span| {
                        let source = self.source.get(span.start..span.end)?;
                        let width = source.bytes().take_while(|byte| *byte == b'`').count();
                        Some(markdown_syntax::Span::new(
                            span.start + width,
                            span.end.saturating_sub(width),
                        ))
                    });
                    self.value(&node.value, span, &code);
                }
                Inline::Link(node) => self.inlines(&node.children, format),
                Inline::LinkReference(node) => self.inlines(&node.children, format),
                Inline::Image(node) => self.inlines(&node.alt, format),
                Inline::ImageReference(node) => self.inlines(&node.alt, format),
                Inline::Autolink(node) => self.value(&node.text, node.meta.span, format),
                Inline::SoftBreak(_) | Inline::LineBreak(_) => {
                    self.value("\n", inline.span(), format)
                }
                _ => self.literal(inline.span(), format),
            }
        }
    }
}
