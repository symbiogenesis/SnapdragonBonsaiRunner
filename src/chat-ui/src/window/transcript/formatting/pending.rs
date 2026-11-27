//! Per-message asynchronous parser state. Production never parses in the UI.
use super::{MAX_MARKDOWN_BYTES, egui, wire};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

const STREAM_COALESCE_MIN_WINDOW: Duration = Duration::from_millis(80);
const STREAM_COALESCE_MEDIUM_WINDOW: Duration = Duration::from_millis(160);
const STREAM_COALESCE_LARGE_WINDOW: Duration = Duration::from_millis(280);
const STREAM_COALESCE_MEDIUM_BYTES: usize = 8 * 1024;
const STREAM_COALESCE_LARGE_BYTES: usize = 32 * 1024;

const fn stream_coalesce_window(source_len: usize) -> Duration {
    if source_len < STREAM_COALESCE_MEDIUM_BYTES {
        STREAM_COALESCE_MIN_WINDOW
    } else if source_len < STREAM_COALESCE_LARGE_BYTES {
        STREAM_COALESCE_MEDIUM_WINDOW
    } else {
        STREAM_COALESCE_LARGE_WINDOW
    }
}

fn coalesce_delay(
    streaming: bool,
    last_dispatch: Option<Instant>,
    now: Instant,
    dispatched_len: usize,
    source: &str,
) -> Option<Duration> {
    if !streaming {
        return None;
    }
    let last = last_dispatch?;
    let elapsed = now.saturating_duration_since(last);
    let window = stream_coalesce_window(source.len());
    if elapsed >= window {
        return None;
    }
    let mut tail_start = dispatched_len.saturating_sub(1).min(source.len());
    while tail_start > 0 && !source.is_char_boundary(tail_start) {
        tail_start -= 1;
    }
    if !source[tail_start..].contains("\n\n") {
        return Some(window - elapsed);
    }
    (elapsed < STREAM_COALESCE_MIN_WINDOW).then_some(STREAM_COALESCE_MIN_WINDOW - elapsed)
}

#[derive(Default)]
pub(in crate::window::transcript) struct Pending {
    revision: Option<u64>,
    parsed: Option<Arc<wire::Document>>,
    parsed_source: Option<Arc<str>>,
    failed: bool,
    cleanup_failed: bool,
    service: Option<Arc<crate::markdown_service::Service>>,
    ticket: Option<crate::markdown_service::Ticket>,
    ticket_source: Option<Arc<str>>,
    request_dirty: bool,
    last_dispatch: Option<Instant>,
    dispatched_len: usize,
    #[cfg(test)]
    asynchronous_fixture: bool,
}

impl Pending {
    pub(in crate::window::transcript) fn parsed_for(
        &self,
        source: &str,
    ) -> Option<(&Arc<wire::Document>, usize)> {
        let parsed = self.parsed.as_ref()?;
        let prefix = self.parsed_source.as_deref()?;
        source.starts_with(prefix).then_some((parsed, prefix.len()))
    }

    pub(in crate::window::transcript) const fn failed(&self) -> bool {
        self.failed
    }

    pub(in crate::window::transcript) const fn cleanup_failed(&self) -> bool {
        self.cleanup_failed
    }

    fn begin_revision(&mut self, revision: u64, source: &str) -> bool {
        self.revision = Some(revision);
        if source.is_empty() || source.len() > MAX_MARKDOWN_BYTES {
            self.parsed = None;
            self.parsed_source = None;
            self.failed = false;
            self.ticket_source = None;
            self.request_dirty = false;
            self.last_dispatch = None;
            self.dispatched_len = 0;
            return true;
        }
        if self.parsed_source.as_deref() == Some(source) {
            self.request_dirty = false;
            self.failed = false;
            return false;
        }
        if self.ticket_source.as_deref() == Some(source) {
            self.request_dirty = false;
            self.failed = false;
            return false;
        }
        let preserves_parsed = self
            .parsed_source
            .as_ref()
            .is_some_and(|prefix| source.starts_with(prefix.as_ref()));
        let preserves_ticket = self
            .ticket_source
            .as_ref()
            .is_some_and(|prefix| source.starts_with(prefix.as_ref()));
        if !preserves_parsed {
            self.parsed = None;
            self.parsed_source = None;
        }
        if !preserves_ticket {
            self.ticket_source = None;
        }
        if !preserves_parsed && !preserves_ticket {
            self.last_dispatch = None;
            self.dispatched_len = 0;
        }
        self.failed = false;
        self.request_dirty = true;
        !preserves_ticket
    }

    pub(in crate::window::transcript) fn changed(&self) -> bool {
        #[cfg(test)]
        if !self.asynchronous_fixture {
            return self.revision.is_none();
        }
        self.request_dirty
            || self.ticket.is_some()
            || self
                .service
                .as_ref()
                .is_some_and(|service| service.cleanup_failed() != self.cleanup_failed)
    }

    #[cfg(test)]
    fn refresh_fixture(&mut self, _ui: &egui::Ui, revision: u64, source: &str) -> bool {
        if self.revision == Some(revision) {
            return false;
        }
        self.revision = Some(revision);
        if !source.is_empty()
            && source.len() <= MAX_MARKDOWN_BYTES
            && self.parsed_source.as_deref() == Some(source)
        {
            return false;
        }
        // Headless UI fixtures exercise the same validated DTO without starting
        // real sandboxes. Native containment remains an independent gate.
        let layout = (source.len() <= MAX_MARKDOWN_BYTES)
            .then(|| wire::decode(source, &super::parse_wire(source)))
            .flatten()
            .map(Arc::new);
        self.parsed_source = layout.as_ref().map(|_| Arc::from(source));
        self.parsed = layout;
        true
    }

    pub(in crate::window::transcript) fn refresh(
        &mut self,
        ui: &egui::Ui,
        revision: u64,
        source: &str,
        streaming: bool,
    ) -> bool {
        #[cfg(test)]
        if !self.asynchronous_fixture {
            return self.refresh_fixture(ui, revision, source);
        }
        use crate::markdown_service::{Admission, Service, Status};
        let mut changed = false;
        if let Some(service) = &self.service {
            let failed = service.cleanup_failed();
            changed = self.cleanup_failed != failed;
            self.cleanup_failed = failed;
        }
        if self.revision != Some(revision) {
            if self.begin_revision(revision, source)
                && let Some(service) = &self.service
                && let Some(ticket) = self.ticket.take()
            {
                service.release(ticket);
            }
            changed = true;
            if source.is_empty() || source.len() > MAX_MARKDOWN_BYTES {
                return changed;
            }
        }
        if let Some(service) = &self.service
            && let Some(ticket) = self.ticket
        {
            match service.poll(ticket) {
                Status::Pending => {}
                Status::Ready(bytes) => {
                    let ticket_source = self.ticket_source.take();
                    let decoded = ticket_source
                        .as_deref()
                        .and_then(|parsed_src| wire::decode(parsed_src, &bytes))
                        .map(Arc::new);
                    if let Some(layout) = decoded {
                        self.parsed = Some(layout);
                        self.parsed_source = ticket_source;
                        self.failed = false;
                    } else {
                        self.parsed = None;
                        self.parsed_source = None;
                        self.failed = true;
                        self.request_dirty = false;
                        service.reject();
                    }
                    service.release(ticket);
                    self.ticket = None;
                    changed = true;
                }
                Status::Failed => {
                    self.parsed = None;
                    self.parsed_source = None;
                    self.failed = true;
                    self.request_dirty = false;
                    service.release(ticket);
                    self.ticket = None;
                    self.ticket_source = None;
                    changed = true;
                    ui.ctx().request_repaint_after(Duration::from_secs(5));
                }
            }
        }
        if self.ticket.is_none() && !self.failed && self.request_dirty {
            let Some(service) = Service::get_or_init(&mut self.service, ui.ctx()) else {
                // Recovered messages can precede the asynchronously prepared
                // broker image. Startup will repaint when it becomes available.
                return changed;
            };
            let now = Instant::now();
            if let Some(delay) = coalesce_delay(
                streaming,
                self.last_dispatch,
                now,
                self.dispatched_len,
                source,
            ) {
                ui.ctx().request_repaint_after(delay);
                return changed;
            }
            let request_source: Arc<str> = Arc::from(source);
            match service.request(Arc::clone(&request_source)) {
                Admission::Admitted(ticket) => {
                    self.ticket = Some(ticket);
                    self.ticket_source = Some(request_source);
                    self.request_dirty = false;
                    self.last_dispatch = Some(now);
                    self.dispatched_len = source.len();
                }
                Admission::InvalidInput | Admission::Unavailable => {
                    self.failed = true;
                    self.request_dirty = false;
                    changed = true;
                }
                Admission::Busy => {
                    // Backpressure is temporary. Retry admission on an upcoming frame
                    // without retaining an extra source clone or restarting a failed worker.
                    ui.ctx().request_repaint_after(Duration::from_millis(25));
                }
            }
        }
        changed
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(service) = &self.service
            && let Some(ticket) = self.ticket.take()
        {
            service.release(ticket);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        markdown_service::Service,
        window::{State, client, mpsc, startup_tests, transcript::Role},
    };
    use egui_kittest::Harness;

    #[test]
    fn prefix_extension_retains_parsed_layout_and_active_ticket() {
        let mut pending = Pending::default();
        let first = "# Title\n";
        assert!(pending.begin_revision(1, first));
        assert!(pending.request_dirty);

        // Simulate dispatch of the first ticket and completion of parsed prefix.
        pending.ticket_source = Some(Arc::from(first));
        pending.request_dirty = false;
        let wire_bytes = super::super::parse_wire(first);
        pending.parsed = wire::decode(first, &wire_bytes).map(Arc::new);
        pending.parsed_source = pending.ticket_source.take();

        // Dispatch a second ticket, then extend the stream again while in flight.
        let second = "# Title\n**bold**";
        assert!(pending.begin_revision(2, second));
        pending.ticket_source = Some(Arc::from(second));
        pending.request_dirty = false;

        let third = "# Title\n**bold** tail";
        assert!(!pending.begin_revision(3, third));
        assert_eq!(pending.ticket_source.as_deref(), Some(second));
        assert!(pending.request_dirty);
        assert_eq!(
            pending.parsed_for(third).map(|(_, len)| len),
            Some(first.len())
        );

        // Non-prefix edit invalidates both the retained prefix and active ticket.
        let replaced = "Completely replaced";
        assert!(pending.begin_revision(4, replaced));
        assert!(pending.ticket_source.is_none());
        assert!(pending.parsed.is_none());
        assert!(pending.parsed_source.is_none());
        assert!(pending.request_dirty);
    }

    #[test]
    fn adaptive_stream_coalesce_window_scales_with_length_and_enforces_paragraph_floor() {
        assert_eq!(stream_coalesce_window(0), Duration::from_millis(80));
        assert_eq!(
            stream_coalesce_window(STREAM_COALESCE_MEDIUM_BYTES - 1),
            Duration::from_millis(80)
        );
        assert_eq!(
            stream_coalesce_window(STREAM_COALESCE_MEDIUM_BYTES),
            Duration::from_millis(160)
        );
        assert_eq!(
            stream_coalesce_window(STREAM_COALESCE_LARGE_BYTES - 1),
            Duration::from_millis(160)
        );
        assert_eq!(
            stream_coalesce_window(STREAM_COALESCE_LARGE_BYTES),
            Duration::from_millis(280)
        );

        let start = Instant::now();
        let small = "abc";
        assert_eq!(
            coalesce_delay(false, Some(start), start, 0, small),
            None,
            "non-streaming must dispatch immediately"
        );
        assert_eq!(
            coalesce_delay(
                true,
                Some(start),
                start + Duration::from_millis(30),
                0,
                small
            ),
            Some(Duration::from_millis(50))
        );

        let medium_plain = "x".repeat(STREAM_COALESCE_MEDIUM_BYTES);
        assert_eq!(
            coalesce_delay(
                true,
                Some(start),
                start + Duration::from_millis(100),
                0,
                &medium_plain,
            ),
            Some(Duration::from_millis(60))
        );
        assert_eq!(
            coalesce_delay(
                true,
                Some(start),
                start + Duration::from_millis(160),
                0,
                &medium_plain,
            ),
            None
        );

        // Paragraph break fast-dispatch still honors the 80 ms floor.
        let mut medium_break = "x".repeat(STREAM_COALESCE_MEDIUM_BYTES);
        medium_break.push_str("\n\nnext");
        assert_eq!(
            coalesce_delay(
                true,
                Some(start),
                start + Duration::from_millis(50),
                STREAM_COALESCE_MEDIUM_BYTES,
                &medium_break,
            ),
            Some(Duration::from_millis(30))
        );
        assert_eq!(
            coalesce_delay(
                true,
                Some(start),
                start + Duration::from_millis(80),
                STREAM_COALESCE_MEDIUM_BYTES,
                &medium_break,
            ),
            None
        );

        let large_plain = "界".repeat(STREAM_COALESCE_LARGE_BYTES / 3 + 1);
        assert_eq!(
            coalesce_delay(
                true,
                Some(start),
                start + Duration::from_millis(200),
                3,
                &large_plain,
            ),
            Some(Duration::from_millis(80))
        );
        assert_eq!(
            coalesce_delay(
                true,
                Some(start),
                start + Duration::from_millis(280),
                3,
                &large_plain,
            ),
            None
        );
    }

    #[test]
    fn markdown_async_configuration_formats_before_model_preparation() -> std::io::Result<()> {
        let source = "# Heading 界\n\nA **bold** answer 🌱.";
        for failed in [false, true] {
            let (request, directory) = startup_tests::fixture_with_broker(true)?;
            let (commands, _incoming) = mpsc::channel();
            let (events, incoming_events) = client::event_channel();
            let mut state = State::pending_startup(commands, incoming_events, request);
            drop(events);
            state.input = "Draft while formatting starts".into();
            state.transcript.push(Role::Assistant, source.to_owned());
            state
                .transcript
                .push(Role::Assistant, "**Offscreen**".into());
            for message in &mut state.transcript.messages {
                message.markdown.asynchronous_fixture = true;
            }
            let mut chat = Harness::builder()
                .with_size([640.0, 80.0])
                .with_context_setup(crate::window::configure_test_context)
                .build_ui_state(
                    |ui, state: &mut State| {
                        egui::ScrollArea::vertical().show_viewport(ui, |ui, viewport| {
                            state.transcript.show(ui, 600.0, viewport, None, false);
                        });
                    },
                    state,
                );
            chat.step();
            chat.step();
            chat.step();
            assert!(!chat.state().transcript.messages[0].markdown.failed());
            assert_eq!(
                chat.state().transcript.messages[0]
                    .cache
                    .as_ref()
                    .and_then(crate::window::transcript::LayoutCache::first_text)
                    .map(|text| text.body.job.text.as_str()),
                Some(source)
            );
            assert!(
                chat.output().viewport_output[&egui::ViewportId::ROOT].repaint_delay
                    > Duration::from_millis(16)
            );
            let respond = Service::fixture(&chat.ctx);
            chat.step();
            let context = chat.ctx.clone();
            chat.state_mut().poll_startup(&context);
            chat.state_mut().drain_events();
            assert!(chat.state().startup_request.is_some());
            assert!(chat.state().startup.is_none() && chat.state().worker.is_none());
            assert!(!chat.state().warning && chat.state().speech.is_none());
            assert_eq!(chat.state().input, "Draft while formatting starts");
            assert!(chat.state().markdown_preparing(&context));
            assert!(
                chat.output().viewport_output[&egui::ViewportId::ROOT].repaint_delay
                    > Duration::from_millis(16)
            );
            if failed {
                Service::get(&context)
                    .ok_or_else(|| std::io::Error::other("Missing parser"))?
                    .reject();
            } else {
                assert_eq!(&*respond()?, source);
                // Receiving a packet is insufficient: apply it and draw a frame.
                assert!(chat.state().markdown_preparing(&context));
            }
            assert!(context.has_requested_repaint());
            chat.step();
            let message = &chat.state().transcript.messages[0];
            assert_eq!(
                message
                    .cache
                    .as_ref()
                    .and_then(crate::window::transcript::LayoutCache::first_text)
                    .map(|text| text.body.job.text.as_str()),
                Some(if failed {
                    source
                } else {
                    "Heading 界\n\nA bold answer 🌱."
                })
            );
            assert_eq!(message.body, source);
            assert_eq!(message.markdown.failed(), failed);
            assert!(!chat.state().markdown_preparing(&context));
            if !failed {
                assert!(chat.state().transcript.messages[1].markdown.changed());
                assert_eq!(&*respond()?, "**Offscreen**");
            }
            drop(chat);
            drop(respond);
            drop(context);
            std::fs::remove_dir_all(directory)?;
        }
        Ok(())
    }
}
