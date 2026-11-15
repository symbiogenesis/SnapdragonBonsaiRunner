//! Conversation-only messages with cached layout and viewport culling.
//! Application diagnostics never enter this model.
use eframe::egui::{self, Color32, FontId, RichText, Vec2};
use std::{ops::Range, sync::Arc};

#[cfg(test)]
mod cache_tests;
mod formatting;
mod iteration;
mod layout;
mod message_controls;
mod search;

#[derive(Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Message actions, speech and image export have independent availability gates."
)]
pub(super) struct MessageControls<'a> {
    pub(super) history: &'a [super::client::Message],
    pub(super) can_send: bool,
    pub(super) can_reload: bool,
    pub(super) can_branch: bool,
    pub(super) speech_loading: Option<u64>,
    pub(super) speech_preparing: bool,
    pub(super) export_busy: bool,
}

pub(super) enum TranscriptAction {
    ReadAloud(u64),
    Message(super::iteration::MessageAction),
    OpenLightbox {
        svg: Arc<str>,
        intrinsic_size: Vec2,
        suggested_stem: String,
    },
    SaveImage {
        svg: Arc<str>,
        suggested_stem: String,
    },
}

const MAX_BYTES: usize = 4 * 1024 * 1024;
pub(super) const MAX_MESSAGES: usize = 512;
pub(super) const MESSAGE_INSET: Vec2 = Vec2::new(16.0, 12.0);
const MESSAGE_ACTION_SIZE: f32 = 32.0;
const MESSAGE_CONTENT_GAP: f32 = 6.0;
pub(super) const MESSAGE_GAP: f32 = 12.0;
const GENERATING_IMAGE_HEIGHT: f32 = 120.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) enum Role {
    User,
    Assistant,
}

pub(super) struct TextSegmentCache {
    pub(super) body: Arc<egui::Galley>,
    pub(super) code_panels: Vec<egui::Rect>,
    pub(in crate::window::transcript) source_map: Arc<formatting::SourceMap>,
    pub(super) underline: Vec<[egui::Pos2; 2]>,
}

pub(super) enum CachedSegment {
    Text(TextSegmentCache),
    Image {
        svg: Arc<str>,
        intrinsic_size: Vec2,
        preview_size: Vec2,
        card_size: Vec2,
        index: usize,
    },
    GeneratingImage {
        card_size: Vec2,
    },
}

impl CachedSegment {
    fn height(&self) -> f32 {
        match self {
            Self::Text(text) => text.body.size().y,
            Self::Image { card_size, .. } | Self::GeneratingImage { card_size } => card_size.y,
        }
    }
}

struct LayoutCache {
    width: f32,
    pixels_per_point: f32,
    text_size: f32,
    color: Color32,
    code_background: Option<Color32>,
    strong_color: Option<Color32>,
    reasoning_color: Color32,
    body_revision: u64,
    pending: bool,
    // Search owns the ranges. For an unchanged body and effective query, the
    // visible ranges are always the same prefix; eviction may change its count.
    highlights: HighlightKey,
    selected: Option<Range<usize>>,
    // Only the active result owns underline geometry. Navigation reuses the
    // highlighted galley instead of rebuilding every match's layout section.
    highlight_fill: Color32,
    highlight_color: Color32,
    reasoning_bytes: usize,
    segments: Vec<CachedSegment>,
    reasoning: Option<Arc<egui::Galley>>,
}

impl LayoutCache {
    #[cfg(test)]
    pub(super) fn first_text(&self) -> Option<&TextSegmentCache> {
        self.segments.iter().find_map(|segment| match segment {
            CachedSegment::Text(text) => Some(text),
            CachedSegment::Image { .. } | CachedSegment::GeneratingImage { .. } => None,
        })
    }

    pub(super) fn text_offset_y(&self, source_offset: usize) -> Option<f32> {
        let mut y = 0.0_f32;
        for (index, segment) in self.segments.iter().enumerate() {
            if index > 0 {
                y += MESSAGE_CONTENT_GAP;
            }
            match segment {
                CachedSegment::Text(text) => {
                    if handles_source_offset(&self.segments, index, source_offset) {
                        let display = text.source_map.offset(source_offset);
                        let characters = text.body.job.text.get(..display)?.chars().count();
                        return Some(
                            y + text
                                .body
                                .pos_from_cursor(egui::text::CCursor::new(characters))
                                .center()
                                .y,
                        );
                    }
                    y += text.body.size().y;
                }
                CachedSegment::Image { card_size, .. }
                | CachedSegment::GeneratingImage { card_size } => {
                    y += card_size.y;
                }
            }
        }
        None
    }
}

fn handles_source_offset(segments: &[CachedSegment], index: usize, source_offset: usize) -> bool {
    !segments[index + 1..].iter().any(|later| match later {
        CachedSegment::Text(text) => text
            .source_map
            .first_source_start()
            .is_some_and(|start| start <= source_offset),
        CachedSegment::Image { .. } | CachedSegment::GeneratingImage { .. } => false,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct HighlightKey {
    query_revision: u64,
    count: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Message {
    #[serde(skip)]
    pub(super) id: u64,
    pub(super) role: Role,
    pub(super) body: String,
    #[serde(default)]
    history_index: Option<usize>,
    // Identifies a live turn across recovery without guessing from duplicate text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    turn_prefix: Option<usize>,
    // A failed prompt can be retried only against its original history prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    retry_prefix: Option<usize>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) interrupted: bool,
    #[serde(skip)]
    body_revision: u64,
    reasoning: String,
    #[serde(skip)]
    pending: bool,
    #[serde(skip)]
    show_reasoning: bool,
    #[serde(skip)]
    cache: Option<LayoutCache>,
    #[serde(skip)]
    markdown: formatting::Pending,
    #[serde(skip)]
    measured_height: Option<f32>,
    #[cfg(test)]
    #[serde(skip)]
    body_layouts: usize,
    #[cfg(test)]
    #[serde(skip)]
    cache_updates: usize,
    #[cfg(test)]
    #[serde(skip)]
    underline_layouts: usize,
    #[cfg(test)]
    #[serde(skip)]
    highlight_range_visits: std::cell::Cell<usize>,
    #[cfg(test)]
    #[serde(skip)]
    history_checks: std::cell::Cell<usize>,
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "Serde skip_serializing_if predicates receive a reference to the field."
)]
const fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Default, serde::Serialize)]
pub(super) struct Transcript {
    pub(super) messages: Vec<Message>,
    // All production text mutations go through this type. Keep the running
    // total out of recovery data and rebuild it when messages are restored.
    #[serde(skip)]
    text_bytes: usize,
    #[serde(skip)]
    next_id: u64,
    #[serde(skip)]
    history_revision: u64,
    #[serde(skip)]
    search: search::Search,
    #[serde(skip)]
    viewport: layout::Viewport,
    // Startup waits for a completed presentation frame of the visible rows,
    // without draining formatting work for the rest of the saved conversation.
    #[serde(skip)]
    presentation: Option<(u64, bool)>,
    #[cfg(test)]
    #[serde(skip)]
    trim_message_visits: usize,
    #[cfg(test)]
    #[serde(skip)]
    trim_batches: usize,
}

impl<'de> serde::Deserialize<'de> for Transcript {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct StoredTranscript {
            #[serde(deserialize_with = "super::persistence::deserialize_transcript_messages")]
            messages: Vec<Message>,
        }
        let stored = StoredTranscript::deserialize(deserializer)?;
        let mut transcript = Self::default();
        for mut message in stored.messages {
            transcript.next_id = transcript.next_id.wrapping_add(1);
            message.id = transcript.next_id;
            transcript.text_bytes = transcript.text_bytes.saturating_add(message.text_bytes());
            transcript.messages.push(message);
        }
        if !transcript.valid_saved() {
            return Err(serde::de::Error::custom(
                "Saved transcript exceeds its text or message limit.",
            ));
        }
        Ok(transcript)
    }
}

impl Transcript {
    pub(super) fn invalidate_presentation(&mut self) {
        self.presentation = None;
    }

    pub(super) fn presentation_pending(&self, ctx: &egui::Context) -> bool {
        !self.messages.is_empty()
            && self
                .presentation
                .is_none_or(|(frame, pending)| pending || frame >= ctx.cumulative_frame_nr())
    }

    pub(super) const fn history_revision(&self) -> u64 {
        self.history_revision
    }

    pub(super) fn valid_saved(&self) -> bool {
        self.messages.len() <= MAX_MESSAGES
            && self.messages.iter().fold(0_usize, |total, message| {
                total
                    .saturating_add(message.body.len())
                    .saturating_add(message.reasoning.len())
            }) <= MAX_BYTES
    }

    pub(super) fn clear(&mut self) {
        self.history_revision = self.history_revision.wrapping_add(1);
        self.messages.clear();
        self.text_bytes = 0;
        self.search = search::Search::default();
        self.viewport = layout::Viewport::default();
        self.presentation = None;
    }

    pub(super) fn remove_empty_response(&mut self) {
        let removed = self.messages.pop_if(|message| {
            message.pending = false;
            message.role == Role::Assistant
                && message.body.is_empty()
                && message.reasoning.is_empty()
        });
        if removed.is_some() {
            self.history_revision = self.history_revision.wrapping_add(1);
            self.search.invalidate();
        }
    }

    pub(super) fn push(&mut self, role: Role, body: String) {
        self.presentation = None;
        self.history_revision = self.history_revision.wrapping_add(1);
        self.search.invalidate();
        self.next_id = self.next_id.wrapping_add(1);
        self.text_bytes = self.text_bytes.saturating_add(body.len());
        self.messages.push(Message {
            id: self.next_id,
            role,
            body,
            history_index: None,
            turn_prefix: None,
            retry_prefix: None,
            interrupted: false,
            body_revision: 0,
            reasoning: String::new(),
            pending: role == Role::Assistant,
            show_reasoning: false,
            cache: None,
            markdown: formatting::Pending::default(),
            measured_height: None,
            #[cfg(test)]
            body_layouts: 0,
            #[cfg(test)]
            cache_updates: 0,
            #[cfg(test)]
            underline_layouts: 0,
            #[cfg(test)]
            highlight_range_visits: std::cell::Cell::new(0),
            #[cfg(test)]
            history_checks: std::cell::Cell::new(0),
        });
        self.trim();
    }

    pub(super) fn replace_response(&mut self, text: &str, reasoning: Option<&str>) {
        if self
            .messages
            .last()
            .is_none_or(|message| message.role != Role::Assistant)
        {
            self.push(Role::Assistant, String::new());
        }
        if let Some(message) = self.messages.last_mut() {
            message.measured_height = Some(message.height());
            self.history_revision = self.history_revision.wrapping_add(1);
            self.search.replaced(message.id);
            self.text_bytes = self.text_bytes.saturating_sub(message.text_bytes());
            text.clone_into(&mut message.body);
            message.body_revision = message.body_revision.wrapping_add(1);
            reasoning
                .unwrap_or_default()
                .clone_into(&mut message.reasoning);
            self.text_bytes = self.text_bytes.saturating_add(message.text_bytes());
            message.cache = None;
        }
        self.trim();
        self.remove_empty_response();
    }

    pub(super) fn append(&mut self, text: &str, reasoning: bool) {
        if !reasoning {
            self.search.appended();
        }
        if let Some(message) = self.messages.last_mut()
            && message.role == Role::Assistant
        {
            if reasoning {
                message.reasoning.push_str(text);
            } else {
                message.body.push_str(text);
                if !text.is_empty() {
                    self.history_revision = self.history_revision.wrapping_add(1);
                    message.body_revision = message.body_revision.wrapping_add(1);
                }
            }
            self.text_bytes = self.text_bytes.saturating_add(text.len());
            // Layout keys track answer revisions and reasoning lengths independently.
        }
        self.trim();
    }

    pub(super) fn stage(&mut self) {
        let turn_prefix = self.messages.last().and_then(|message| message.turn_prefix);
        if let Some(message) = self.messages.last_mut() {
            message.pending = false;
        }
        if self.messages.last().is_some_and(|message| {
            message.role == Role::Assistant
                && (!message.body.is_empty() || !message.reasoning.is_empty())
        }) {
            self.push(Role::Assistant, String::new());
            if let Some(message) = self.messages.last_mut() {
                message.turn_prefix = turn_prefix;
            }
        }
    }

    fn trim(&mut self) {
        if self.text_bytes <= MAX_BYTES && self.messages.len() <= MAX_MESSAGES {
            return;
        }
        let mut removed = 0;
        // Bound message ownership and per-frame traversal even when messages
        // contain little or no text. Preserve the final two messages under the
        // byte policy; inspect only the evicted prefix and move the suffix once.
        for message in self
            .messages
            .iter()
            .take(self.messages.len().saturating_sub(2))
        {
            if self.text_bytes <= MAX_BYTES && self.messages.len() - removed <= MAX_MESSAGES {
                break;
            }
            #[cfg(test)]
            {
                self.trim_message_visits += 1;
            }
            self.text_bytes = self.text_bytes.saturating_sub(message.text_bytes());
            removed += 1;
        }
        if removed > 0 {
            self.history_revision = self.history_revision.wrapping_add(1);
            self.search.invalidate();
            drop(self.messages.drain(..removed));
            #[cfg(test)]
            {
                self.trim_batches += 1;
            }
        }
    }

    pub(super) fn plain_text(&self) -> String {
        let messages = self
            .messages
            .iter()
            .filter(|message| !message.body.is_empty())
            .map(|message| {
                (
                    if message.role == Role::User {
                        "You"
                    } else {
                        "Swift Bonsai"
                    },
                    message.body.as_str(),
                )
            });
        let capacity = messages
            .clone()
            .map(|(label, body)| label.len() + body.len() + 3)
            .sum::<usize>()
            .saturating_sub(2);
        let mut text = String::with_capacity(capacity);
        for (label, body) in messages {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(label);
            text.push('\n');
            text.push_str(body);
        }
        text
    }

    pub(super) const fn find_open(&self) -> bool {
        self.search.is_open()
    }

    pub(super) const fn focus_find(&mut self) {
        self.search.focus();
    }

    pub(super) fn find_query(&mut self, query: &str) {
        query.clone_into(&mut self.search.query);
        self.search.focus();
        self.search.query_changed(&self.messages);
    }

    pub(super) fn sync_find_query(&mut self, query: &str) {
        self.search.sync_query(query, &self.messages);
    }

    pub(super) fn close_find(&mut self) {
        self.search.sync_query("", &self.messages);
    }

    pub(super) fn search_bar(&mut self, ui: &mut egui::Ui) -> bool {
        self.search.bar(ui, &self.messages)
    }

    pub(super) fn latest_button(&mut self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let size = Vec2::splat(36.0);
        let rect = egui::Rect::from_center_size(
            egui::pos2(viewport.center().x, viewport.bottom() - 30.0),
            size,
        );
        let response = ui
            .scope(|ui| {
                ui.spacing_mut().button_padding = Vec2::ZERO;
                ui.place(rect, egui::Button::new(()).min_size(size).corner_radius(18))
            })
            .inner
            .accessible_name("Latest")
            .on_hover_text("Jump to the latest message");
        let center = response.rect.center();
        let stroke = ui.style().interact(&response).fg_stroke;
        let painter = ui.painter_at(response.rect);
        painter.line_segment(
            [center - egui::vec2(0.0, 7.0), center + egui::vec2(0.0, 7.0)],
            stroke,
        );
        painter.add(egui::Shape::line(
            vec![
                center + egui::vec2(-6.0, 1.0),
                center + egui::vec2(0.0, 7.0),
                center + egui::vec2(6.0, 1.0),
            ],
            stroke,
        ));
        if response.clicked() {
            self.search.latest = true;
            self.search.target = None;
            ui.ctx().request_repaint();
        }
    }

    pub(super) fn close_find_on_escape(&mut self, ctx: &egui::Context) -> bool {
        self.search.close_on_escape(ctx)
    }

    pub(super) fn speech_text(&self, id: u64) -> Option<String> {
        self.messages
            .iter()
            .find(|message| message.id == id)
            .and_then(Message::speech_body)
    }

    #[cfg(test)]
    pub(super) fn show(
        &mut self,
        ui: &mut egui::Ui,
        width: f32,
        viewport: egui::Rect,
        speaking_message: Option<u64>,
        speech_enabled: bool,
    ) -> Option<u64> {
        match self.show_with_actions(
            ui,
            width,
            viewport,
            speaking_message,
            speech_enabled,
            None,
            None,
        ) {
            Some(TranscriptAction::ReadAloud(id)) => Some(id),
            _ => None,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Transcript presentation borrows speech state, controls and shared image texture cache explicitly."
    )]
    pub(super) fn show_with_actions(
        &mut self,
        ui: &mut egui::Ui,
        width: f32,
        viewport: egui::Rect,
        speaking_message: Option<u64>,
        speech_enabled: bool,
        controls: Option<MessageControls<'_>>,
        mut image_cache: Option<&mut super::image_cache::ImageCache>,
    ) -> Option<TranscriptAction> {
        let mut action = None;
        self.search.refresh(&self.messages);
        let width = width.max(120.0);
        let rows = self.prepare_layout(ui, width, viewport, controls.as_ref());
        let mut presentation_pending = false;
        for (message, row) in self.messages.iter_mut().zip(rows.messages) {
            let (rect, _) =
                ui.allocate_exact_size(Vec2::new(width, row.height), egui::Sense::hover());
            // Compensate painting and hit testing in this pass too; the scroll
            // adjustment below makes the same correction persistent next frame.
            let rect = rect.translate(egui::vec2(0.0, -rows.correction));
            let target = self
                .search
                .target
                .as_ref()
                .filter(|found| found.message == message.id);
            // egui clears native label selection when its widget disappears.
            // Keep a source-current galley visible during deferred geometry-only
            // reflow; exact Find scrolling still waits for current geometry.
            let retained = message
                .cache
                .as_ref()
                .is_some_and(|cache| cache.body_revision == message.body_revision);
            let visible = ui.is_rect_visible(rect) || target.is_some();
            if visible {
                presentation_pending |=
                    !row.ready || (message.role == Role::Assistant && message.markdown.changed());
            }
            if (row.ready || retained) && visible {
                let (found_rect, message_action) = message.show(
                    ui,
                    rect,
                    target.filter(|_| row.ready).map(|found| found.range.start),
                    controls.as_ref(),
                    speaking_message,
                    speech_enabled,
                    image_cache.as_deref_mut(),
                );
                if let Some(message_action) = message_action {
                    action = Some(message_action);
                }
                if let Some(found_rect) = found_rect {
                    ui.scroll_to_rect(found_rect, Some(egui::Align::Center));
                    self.search.target = None;
                }
            }
            ui.add_space((MESSAGE_GAP - ui.spacing().item_spacing.y).max(0.0));
        }
        self.presentation = Some((ui.ctx().cumulative_frame_nr(), presentation_pending));
        if rows.correction != 0.0 {
            ui.scroll_with_delta_animation(
                egui::vec2(0.0, -rows.correction),
                egui::style::ScrollAnimation::none(),
            );
        }
        if self.search.latest {
            ui.scroll_to_cursor(Some(egui::Align::Max));
            self.search.latest = false;
        }
        action
    }
}

impl Message {
    const fn text_bytes(&self) -> usize {
        self.body.len().saturating_add(self.reasoning.len())
    }

    fn speech_body(&self) -> Option<String> {
        if self.role != Role::Assistant || self.pending {
            return None;
        }
        let spoken = super::svg_fence::strip_svg_fences(&self.body);
        let trimmed = spoken.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    }

    fn can_speak(&self) -> bool {
        if self.role != Role::Assistant || self.pending {
            return false;
        }
        if !self.body.contains("```") && !self.body.contains("~~~") {
            return !self.body.trim().is_empty();
        }
        !super::svg_fence::strip_svg_fences(&self.body)
            .trim()
            .is_empty()
    }

    fn speaker(
        &self,
        ui: &mut egui::Ui,
        speaking_message: Option<u64>,
        speech_enabled: bool,
        loading: bool,
    ) -> bool {
        let playing = speaking_message == Some(self.id);
        let label = if playing {
            "Stop reading this response"
        } else {
            "Read this response aloud"
        };
        let response = ui.add_enabled(
            playing || (speech_enabled && self.can_speak()),
            super::controls::quiet_button("")
                .min_size(Vec2::splat(MESSAGE_ACTION_SIZE))
                .selected(playing),
        );
        response.widget_info(|| {
            egui::WidgetInfo::labeled(egui::Role::Button, response.enabled(), label)
        });
        let response = response.on_hover_text(if loading && playing {
            "Preparing speech. Select to cancel."
        } else if loading {
            "Preparing speech automatically."
        } else if self.pending && !playing {
            "Available when this response finishes"
        } else {
            label
        });
        let painter = ui.painter_at(response.rect);
        let color = ui.style().interact(&response).fg_stroke.color;
        let center = response.rect.center();
        if loading {
            egui::Spinner::new()
                .color(color)
                .paint_at(ui, egui::Rect::from_center_size(center, Vec2::splat(20.0)));
        } else if playing {
            painter.rect_filled(
                egui::Rect::from_center_size(center, Vec2::splat(10.0)),
                1.0,
                color,
            );
        } else {
            painter.rect_filled(
                egui::Rect::from_min_max(
                    center + egui::vec2(-8.0, -3.0),
                    center + egui::vec2(-4.0, 3.0),
                ),
                0.0,
                color,
            );
            let points = [(-4.0, -3.0), (1.0, -7.0), (1.0, 7.0), (-4.0, 3.0)]
                .map(|(x, y)| center + egui::vec2(x, y));
            painter.add(egui::Shape::convex_polygon(
                points.to_vec(),
                color,
                egui::Stroke::NONE,
            ));
            for radius in [5.0_f32, 9.0] {
                let points = [-0.8_f32, -0.4, 0.0, 0.4, 0.8].map(|angle| {
                    center + egui::vec2(radius.mul_add(angle.cos(), 1.0), radius * angle.sin())
                });
                painter.add(egui::Shape::line(
                    points.to_vec(),
                    egui::Stroke::new(1.5, color),
                ));
            }
        }
        response.clicked()
    }

    fn layout(
        &mut self,
        ui: &egui::Ui,
        width: f32,
        matches: &[search::Match],
        highlights: HighlightKey,
        selected: Option<&Range<usize>>,
    ) -> &LayoutCache {
        let streaming_visible_prefix = (self.role == Role::Assistant && self.pending)
            .then(|| super::svg_fence::split_streaming_svg(&self.body))
            .and_then(|(prefix, open)| open.then_some(prefix.len()));
        let parse_source =
            streaming_visible_prefix.map_or(self.body.as_str(), |end| &self.body[..end]);
        let parse_revision =
            (self.body_revision << 1) | u64::from(streaming_visible_prefix.is_some());
        let markdown_changed =
            self.role == Role::Assistant && self.markdown.refresh(ui, parse_revision, parse_source);
        let color = ui.visuals().text_color();
        let code_background = (self.role == Role::Assistant).then_some(ui.visuals().code_bg_color);
        let strong_color =
            (self.role == Role::Assistant).then_some(ui.visuals().strong_text_color());
        let reasoning_color = ui.visuals().weak_text_color();
        let pixels_per_point = ui.ctx().pixels_per_point();
        let text_size = super::appearance::text_size(ui.ctx());
        let highlight_fill = ui.visuals().selection.bg_fill;
        let highlight_color = ui.visuals().selection.stroke.color;
        let reasoning_visible = self.show_reasoning && !self.reasoning.is_empty();
        // Polonius lets a complete cache hit lend self.cache directly while a
        // partial miss ends the borrow and reclaims reusable galley owners below.
        let (body_unchanged, reasoning_unchanged, underline_unchanged) =
            if let Some(cache) = self.cache.as_ref() {
                let same_geometry = cache.width.to_bits() == width.to_bits()
                    && cache.pixels_per_point.to_bits() == pixels_per_point.to_bits()
                    && cache.text_size.to_bits() == text_size.to_bits();
                let body_unchanged = same_geometry
                    && !markdown_changed
                    && cache.body_revision == self.body_revision
                    && cache.pending == self.pending
                    && cache.color == color
                    && cache.code_background == code_background
                    && cache.strong_color == strong_color
                    && cache.highlights == highlights
                    && (matches.is_empty()
                        || (cache.highlight_fill == highlight_fill
                            && cache.highlight_color == highlight_color));
                // Hidden reasoning has no galley. Its streamed bytes and weak
                // color cannot invalidate the visible body or Find geometry;
                // opening it still checks the complete current text below.
                let reasoning_unchanged = !reasoning_visible
                    || (same_geometry
                        && cache.reasoning_bytes == self.reasoning.len()
                        && cache.reasoning_color == reasoning_color);
                let underline_unchanged = body_unchanged && cache.selected.as_ref() == selected;
                if body_unchanged
                    && reasoning_unchanged
                    && underline_unchanged
                    && cache.reasoning.is_some() == reasoning_visible
                {
                    return cache;
                }
                (body_unchanged, reasoning_unchanged, underline_unchanged)
            } else {
                (false, false, false)
            };
        #[cfg(test)]
        {
            self.cache_updates += 1;
        }
        // Move reusable owners on a partial update. A complete cache hit above
        // does not reconstruct the cache or adjust any galley's Arc count.
        let (previous_segments, previous_reasoning) =
            self.cache.take().map_or((None, None), |cache| {
                (Some(cache.segments), cache.reasoning)
            });
        let mut segments = if body_unchanged && let Some(segments) = previous_segments {
            segments
        } else {
            #[cfg(test)]
            {
                self.body_layouts += 1;
            }
            self.layout_body(ui, width, text_size, matches, streaming_visible_prefix)
        };
        if !underline_unchanged || !body_unchanged {
            if let Some(selected) = selected {
                #[cfg(test)]
                {
                    self.underline_layouts += 1;
                }
                let segment_count = segments.len();
                for index in 0..segment_count {
                    let active_here = handles_source_offset(&segments, index, selected.start);
                    if let CachedSegment::Text(text) = &mut segments[index] {
                        text.underline = if active_here {
                            active_underline(
                                &text.body.job.text,
                                &text.body,
                                &text.source_map.range(selected.clone()),
                                pixels_per_point,
                            )
                        } else {
                            Vec::new()
                        };
                    }
                }
            } else {
                for segment in &mut segments {
                    if let CachedSegment::Text(text) = segment {
                        text.underline.clear();
                    }
                }
            }
        }
        let reasoning = if !reasoning_visible {
            None
        } else if reasoning_unchanged && let Some(reasoning) = previous_reasoning {
            Some(reasoning)
        } else {
            Some(ui.fonts_mut(|fonts| {
                fonts.layout(
                    self.reasoning.clone(),
                    FontId::proportional(text_size * 0.875),
                    reasoning_color,
                    width,
                )
            }))
        };
        self.cache.insert(LayoutCache {
            width,
            pixels_per_point,
            text_size,
            color,
            code_background,
            strong_color,
            reasoning_color,
            body_revision: self.body_revision,
            pending: self.pending,
            highlights,
            selected: selected.cloned(),
            highlight_fill,
            highlight_color,
            reasoning_bytes: self.reasoning.len(),
            segments,
            reasoning,
        })
    }

    fn layout_current(
        &self,
        ui: &egui::Ui,
        width: f32,
        highlights: HighlightKey,
        selected: Option<&Range<usize>>,
    ) -> bool {
        let visible_reasoning = self.show_reasoning && !self.reasoning.is_empty();
        self.cache.as_ref().is_some_and(|cache| {
            cache.width.to_bits() == width.to_bits()
                && cache.pixels_per_point.to_bits() == ui.ctx().pixels_per_point().to_bits()
                && cache.text_size.to_bits() == super::appearance::text_size(ui.ctx()).to_bits()
                && cache.body_revision == self.body_revision
                && cache.pending == self.pending
                && (self.role != Role::Assistant || !self.markdown.changed())
                && cache.color == ui.visuals().text_color()
                && cache.code_background
                    == (self.role == Role::Assistant).then_some(ui.visuals().code_bg_color)
                && cache.strong_color
                    == (self.role == Role::Assistant).then_some(ui.visuals().strong_text_color())
                && cache.highlights == highlights
                && (highlights.count == 0
                    || (cache.highlight_fill == ui.visuals().selection.bg_fill
                        && cache.highlight_color == ui.visuals().selection.stroke.color))
                && cache.selected.as_ref() == selected
                && cache.reasoning.is_some() == visible_reasoning
                && (!visible_reasoning
                    || (cache.reasoning_bytes == self.reasoning.len()
                        && cache.reasoning_color == ui.visuals().weak_text_color()))
        })
    }

    fn layout_body(
        &self,
        ui: &egui::Ui,
        width: f32,
        text_size: f32,
        matches: &[search::Match],
        streaming_visible_prefix: Option<usize>,
    ) -> Vec<CachedSegment> {
        #[cfg(test)]
        self.highlight_range_visits
            .set(self.highlight_range_visits.get() + matches.len());
        let parse_slice =
            streaming_visible_prefix.map_or(self.body.as_str(), |end| &self.body[..end]);
        let (body_segments, show_generating) = formatting::layout_document(
            &self.body,
            self.role == Role::Assistant,
            width,
            text_size,
            ui.visuals(),
            matches,
            self.markdown.parsed_for(parse_slice),
            streaming_visible_prefix,
        );
        let mut image_index = 0_usize;
        let mut cached = Vec::with_capacity(body_segments.len() + usize::from(show_generating));
        for segment in body_segments {
            match segment {
                formatting::BodySegment::Text(layout) => {
                    let body = ui.fonts_mut(|fonts| fonts.layout_job(layout.job));
                    let code_panels = formatting::code_panels(&body);
                    cached.push(CachedSegment::Text(TextSegmentCache {
                        body,
                        code_panels,
                        source_map: layout.source_map,
                        underline: Vec::new(),
                    }));
                }
                formatting::BodySegment::Image {
                    svg,
                    intrinsic_size,
                } => {
                    let (preview_size, card_size) = image_card_layout(width, intrinsic_size);
                    cached.push(CachedSegment::Image {
                        svg,
                        intrinsic_size,
                        preview_size,
                        card_size,
                        index: image_index,
                    });
                    image_index += 1;
                }
            }
        }
        if show_generating {
            cached.push(CachedSegment::GeneratingImage {
                card_size: generating_card_size(width),
            });
        }
        cached
    }

    pub(super) fn height(&self) -> f32 {
        let Some(cache) = &self.cache else {
            return self.measured_height.unwrap_or(100.0);
        };
        let content_height = if cache.segments.is_empty() {
            20.0
        } else if cache.segments.len() == 1 {
            cache.segments[0].height().max(20.0)
        } else {
            let mut sum = 0.0_f32;
            for (index, segment) in cache.segments.iter().enumerate() {
                if index > 0 {
                    sum += MESSAGE_CONTENT_GAP;
                }
                sum += segment.height();
            }
            sum
        };
        // One shared header holds identity and actions for either role. Speech
        // never reserves a separate row, including while it is unavailable.
        let mut height = 2.0_f32.mul_add(MESSAGE_INSET.y, MESSAGE_ACTION_SIZE)
            + MESSAGE_CONTENT_GAP
            + content_height;
        if self.markdown.failed() {
            height += MESSAGE_CONTENT_GAP + 60.0;
        }
        if !self.reasoning.is_empty() {
            height += MESSAGE_CONTENT_GAP + MESSAGE_ACTION_SIZE;
            if self.show_reasoning
                && let Some(reasoning) = &cache.reasoning
            {
                height += reasoning.size().y + MESSAGE_CONTENT_GAP;
            }
        }
        height
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Message presentation needs row rect, Find target, action controls, speech state and optional image cache."
    )]
    fn show(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        target: Option<usize>,
        controls: Option<&MessageControls<'_>>,
        speaking_message: Option<u64>,
        speech_enabled: bool,
        mut image_cache: Option<&mut super::image_cache::ImageCache>,
    ) -> (Option<egui::Rect>, Option<TranscriptAction>) {
        let mut found_rect = None;
        let fill = if self.role == Role::User {
            ui.visuals().faint_bg_color
        } else {
            ui.visuals().extreme_bg_color
        };
        ui.painter().rect_filled(rect, 12.0, fill);
        let mut child = ui.new_child(
            egui::UiBuilder::new()
                .id_salt(("message", self.id))
                .max_rect(rect.shrink2(MESSAGE_INSET))
                .layout(egui::Layout::top_down(egui::Align::Min)),
        );
        child.set_clip_rect(rect.intersect(ui.clip_rect()));
        child.spacing_mut().item_spacing = Vec2::splat(MESSAGE_CONTENT_GAP);
        child.spacing_mut().button_padding = egui::vec2(8.0, 4.0);
        child.spacing_mut().interact_size.y = MESSAGE_ACTION_SIZE;
        let mut action = self.header(&mut child, controls, speaking_message, speech_enabled);
        let export_busy = controls.is_some_and(|c| c.export_busy);
        if let Some(cache) = &self.cache {
            if self.body.is_empty() && self.pending {
                child.label(RichText::new("Responding…").weak().italics());
            } else if !self.body.is_empty() {
                for (seg_index, segment) in cache.segments.iter().enumerate() {
                    match segment {
                        CachedSegment::Text(text) => {
                            let target_here = target.filter(|offset| {
                                handles_source_offset(&cache.segments, seg_index, *offset)
                            });
                            if let Some(rect) = render_text_segment(
                                &mut child,
                                text,
                                cache.highlight_color,
                                target_here,
                            ) {
                                found_rect = Some(rect);
                            }
                        }
                        CachedSegment::Image { .. } => {
                            if let Some(card_action) = render_image_card(
                                &mut child,
                                self.id,
                                segment,
                                image_cache.as_deref_mut(),
                                export_busy,
                            ) {
                                action = Some(card_action);
                            }
                        }
                        CachedSegment::GeneratingImage { card_size } => {
                            render_generating_card(&mut child, *card_size);
                        }
                    }
                }
                if self.markdown.failed() {
                    let message = if self.markdown.cleanup_failed() {
                        "Formatting stopped; showing source. Restart the app."
                    } else {
                        "Formatting unavailable; showing source."
                    };
                    child.label(RichText::new(message).small().weak());
                }
            }
            if !self.reasoning.is_empty() {
                let label = if self.show_reasoning {
                    "Hide thinking"
                } else {
                    "Show thinking"
                };
                if child
                    .add(
                        super::controls::quiet_button(label)
                            .min_size(egui::vec2(0.0, MESSAGE_ACTION_SIZE)),
                    )
                    .clicked()
                {
                    self.show_reasoning = !self.show_reasoning;
                    child.ctx().request_repaint();
                }
                if self.show_reasoning
                    && let Some(reasoning) = &cache.reasoning
                {
                    child.add(egui::Label::new(Arc::clone(reasoning)).selectable(true));
                }
            }
        }
        if let Some(controls) = controls
            && let Some(retry) = self.retry_controls(&mut child, controls)
        {
            action = Some(TranscriptAction::Message(retry));
        }
        (found_rect, action)
    }
}

fn image_card_layout(width: f32, intrinsic: Vec2) -> (Vec2, Vec2) {
    let card_max_w = width.clamp(120.0, 560.0);
    let max_preview_w = (card_max_w - 16.0).max(96.0);
    let max_preview_h = 380.0_f32;
    let iw = intrinsic.x.max(1.0);
    let ih = intrinsic.y.max(1.0);
    let scale = (max_preview_w / iw).min(max_preview_h / ih);
    let preview_w = (iw * scale).clamp(96.0, max_preview_w).round();
    let preview_h = (ih * scale).clamp(72.0, max_preview_h).round();
    let card_w = (preview_w + 16.0).min(card_max_w);
    let card_h = 8.0 + preview_h + MESSAGE_CONTENT_GAP + MESSAGE_ACTION_SIZE + 8.0;
    (egui::vec2(preview_w, preview_h), egui::vec2(card_w, card_h))
}

fn generating_card_size(width: f32) -> Vec2 {
    egui::vec2(width.clamp(160.0, 560.0), GENERATING_IMAGE_HEIGHT)
}

fn render_text_segment(
    ui: &mut egui::Ui,
    text: &TextSegmentCache,
    highlight_color: Color32,
    target: Option<usize>,
) -> Option<egui::Rect> {
    // Reserve the paint slot before selectable text, so code surfaces
    // remain behind glyphs, selection and Find highlights.
    let code_surface = ui.painter().add(egui::Shape::Noop);
    let response = ui.add(egui::Label::new(Arc::clone(&text.body)).selectable(true));
    ui.painter().set(
        code_surface,
        egui::Shape::Vec(
            text.code_panels
                .iter()
                .map(|panel| {
                    egui::Shape::rect_filled(
                        panel
                            .translate(response.rect.min.to_vec2())
                            .expand2(egui::vec2(6.0, 3.0)),
                        6.0,
                        ui.visuals().code_bg_color,
                    )
                })
                .collect(),
        ),
    );
    for line in &text.underline {
        ui.painter().line_segment(
            line.map(|point| point + response.rect.min.to_vec2()),
            egui::Stroke::new(2.0, highlight_color),
        );
    }
    target.map(|offset| {
        let display = text.source_map.offset(offset);
        let characters = text.body.job.text[..display].chars().count();
        text.body
            .pos_from_cursor(egui::text::CCursor::new(characters))
            .translate(response.rect.min.to_vec2())
    })
}

pub(super) fn aspect_fit_rect(bounds: egui::Rect, texture_size: Vec2) -> egui::Rect {
    let tw = texture_size.x.max(1.0);
    let th = texture_size.y.max(1.0);
    let scale = (bounds.width() / tw).min(bounds.height() / th);
    let size = egui::vec2(
        (tw * scale).min(bounds.width()),
        (th * scale).min(bounds.height()),
    );
    egui::Rect::from_center_size(bounds.center(), size)
}

fn render_image_card(
    ui: &mut egui::Ui,
    message_id: u64,
    segment: &CachedSegment,
    image_cache: Option<&mut super::image_cache::ImageCache>,
    export_busy: bool,
) -> Option<TranscriptAction> {
    let CachedSegment::Image {
        svg,
        intrinsic_size,
        preview_size,
        card_size,
        index,
    } = segment
    else {
        return None;
    };
    let suggested_stem = format!("bonsai-image-{}", index + 1);
    let (card_rect, _) = ui.allocate_exact_size(*card_size, egui::Sense::hover());
    let visuals = ui.visuals().clone();
    ui.painter().rect(
        card_rect,
        10.0,
        visuals.faint_bg_color,
        visuals.widgets.noninteractive.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let preview_rect =
        egui::Rect::from_min_size(card_rect.min + egui::vec2(8.0, 8.0), *preview_size);
    ui.painter()
        .rect_filled(preview_rect, 6.0, visuals.extreme_bg_color);

    let slot = image_cache.map(|cache| cache.get_or_request(ui.ctx(), svg, *preview_size));
    let is_fallback = matches!(slot, Some(super::image_cache::RasterSlot::FallbackSource));
    match slot {
        Some(
            super::image_cache::RasterSlot::Ready(tex)
            | super::image_cache::RasterSlot::Loading {
                fallback: Some(tex),
            },
        ) => {
            let fitted = aspect_fit_rect(preview_rect, tex.size_vec2());
            ui.painter().image(
                tex.id(),
                fitted,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        Some(super::image_cache::RasterSlot::Loading { fallback: None }) | None => {
            egui::Spinner::new().paint_at(
                ui,
                egui::Rect::from_center_size(preview_rect.center(), Vec2::splat(24.0)),
            );
        }
        Some(super::image_cache::RasterSlot::FallbackSource) => {
            let text_rect = preview_rect.shrink(8.0);
            let mut preview_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(text_rect)
                    .layout(egui::Layout::top_down(egui::Align::Min)),
            );
            preview_ui.set_clip_rect(text_rect);
            preview_ui.label(
                RichText::new("SVG preview unavailable — source:")
                    .small()
                    .weak(),
            );
            preview_ui.label(
                RichText::new(svg.as_ref())
                    .monospace()
                    .small()
                    .color(visuals.weak_text_color()),
            );
        }
    }

    let mut action = None;
    let preview_id = ui.make_persistent_id(("svg-preview", message_id, *index));
    let preview_response = ui
        .interact(preview_rect, preview_id, egui::Sense::click())
        .on_hover_cursor(egui::CursorIcon::ZoomIn)
        .on_hover_text("Double-click to view larger");
    preview_response.widget_info(|| {
        egui::WidgetInfo::labeled(
            egui::Role::Image,
            true,
            "Generated image. Double-click to expand.",
        )
    });
    if preview_response.double_clicked() {
        action = Some(TranscriptAction::OpenLightbox {
            svg: Arc::clone(svg),
            intrinsic_size: *intrinsic_size,
            suggested_stem: suggested_stem.clone(),
        });
    }

    let footer_rect = egui::Rect::from_min_max(
        egui::pos2(
            card_rect.min.x + 8.0,
            preview_rect.max.y + MESSAGE_CONTENT_GAP,
        ),
        egui::pos2(card_rect.max.x - 8.0, card_rect.max.y - 8.0),
    );
    let mut footer = ui.new_child(
        egui::UiBuilder::new()
            .id_salt(("svg-footer", message_id, *index))
            .max_rect(footer_rect)
            .layout(egui::Layout::right_to_left(egui::Align::Center)),
    );
    let save_enabled = !export_busy && !is_fallback;
    let save_button = footer
        .add_enabled(
            save_enabled,
            super::controls::quiet_button("Save…").min_size(egui::vec2(0.0, MESSAGE_ACTION_SIZE)),
        )
        .on_hover_text("Save image as PNG or SVG");
    save_button.widget_info(|| {
        egui::WidgetInfo::labeled(egui::Role::Button, save_enabled, "Save generated image")
    });
    if save_button.clicked() {
        action = Some(TranscriptAction::SaveImage {
            svg: Arc::clone(svg),
            suggested_stem: suggested_stem.clone(),
        });
    }

    let expand_button = footer
        .add(super::controls::quiet_button("Expand").min_size(egui::vec2(0.0, MESSAGE_ACTION_SIZE)))
        .on_hover_text("Open image in full-window view");
    expand_button.widget_info(|| {
        egui::WidgetInfo::labeled(egui::Role::Button, true, "Expand generated image")
    });
    if expand_button.clicked() {
        action = Some(TranscriptAction::OpenLightbox {
            svg: Arc::clone(svg),
            intrinsic_size: *intrinsic_size,
            suggested_stem,
        });
    }

    action
}

fn render_generating_card(ui: &mut egui::Ui, card_size: Vec2) {
    let (card_rect, _) = ui.allocate_exact_size(card_size, egui::Sense::hover());
    let visuals = ui.visuals().clone();
    ui.painter().rect(
        card_rect,
        10.0,
        visuals.faint_bg_color,
        visuals.widgets.noninteractive.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let mut inner = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(card_rect.shrink(12.0))
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    inner.add(egui::Spinner::new());
    inner.add_space(6.0);
    inner.label(RichText::new("Generating image…").weak());
}

// Display byte offsets map to Unicode scalar indices in the galley. Retain only
// the selected glyph runs, splitting wrapped rows and font fallback baselines
// in the same way as epaint's TextFormat underline. Newlines own no glyph.
fn active_underline(
    body: &str,
    galley: &egui::Galley,
    selected: &Range<usize>,
    pixels_per_point: f32,
) -> Vec<[egui::Pos2; 2]> {
    let start = body[..selected.start].chars().count();
    let end = start + body[selected.clone()].chars().count();
    let mut offset = 0;
    let mut lines: Vec<[egui::Pos2; 2]> = Vec::new();
    let stroke = egui::Stroke::new(2.0, Color32::PLACEHOLDER);
    for row in &galley.rows {
        if offset >= end {
            break;
        }
        let first = start.saturating_sub(offset).min(row.glyphs.len());
        let last = end.saturating_sub(offset).min(row.glyphs.len());
        let mut run: Option<[egui::Pos2; 2]> = None;
        for glyph in &row.glyphs[first..last] {
            let mut y = glyph.logical_rect().bottom();
            stroke.round_center_to_pixel(pixels_per_point, &mut y);
            let right = egui::pos2(glyph.max_x(), y) + row.pos.to_vec2();
            if let Some(line) = &mut run
                && line[0].y.to_bits() == right.y.to_bits()
            {
                line[1] = right;
            } else {
                if let Some(line) = run.take() {
                    lines.push(line);
                }
                run = Some([egui::pos2(glyph.pos.x, y) + row.pos.to_vec2(), right]);
            }
        }
        if let Some(line) = run {
            lines.push(line);
        }
        offset += row.char_count_including_newline().0;
    }
    lines
}

#[cfg(test)]
mod persistence_tests {
    use super::*;

    #[test]
    fn restored_transcript_retains_text_without_layout_or_pending_generation()
    -> Result<(), serde_json::Error> {
        let mut transcript = Transcript::default();
        transcript.push(Role::User, "界 🌱".to_owned());
        transcript.push(Role::Assistant, "Interrupted draft".to_owned());
        transcript.append("Retained reasoning", true);
        let bytes = serde_json::to_vec(&transcript)?;
        let mut restored: Transcript = serde_json::from_slice(&bytes)?;
        assert_eq!(restored.plain_text(), transcript.plain_text());
        assert_eq!(restored.messages[1].reasoning, "Retained reasoning");
        assert!(!restored.messages[1].pending);
        assert!(
            restored
                .messages
                .iter()
                .all(|message| message.cache.is_none())
        );
        assert_ne!(restored.messages[0].id, restored.messages[1].id);
        restored.push(Role::User, "New turn".to_owned());
        assert_ne!(restored.messages[1].id, restored.messages[2].id);
        let text = String::from_utf8_lossy(&bytes);
        assert!(!text.contains("cache"));
        assert!(!text.contains("pending"));
        assert!(!text.contains("next_id"));
        Ok(())
    }
}
