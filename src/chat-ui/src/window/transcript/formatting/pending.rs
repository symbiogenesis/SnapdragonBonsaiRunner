//! Per-message asynchronous parser state. Production never parses in the UI.
use super::{Layout, MAX_MARKDOWN_BYTES, egui, wire};
use std::sync::Arc;

#[derive(Default)]
pub(in crate::window::transcript) struct Pending {
    revision: Option<u64>,
    parsed: Option<Arc<Layout>>,
    parsed_source: Option<Arc<str>>,
    failed: bool,
    cleanup_failed: bool,
    service: Option<Arc<crate::markdown_service::Service>>,
    ticket: Option<crate::markdown_service::Ticket>,
    ticket_source: Option<Arc<str>>,
    request_source: Option<Arc<str>>,
    #[cfg(test)]
    asynchronous_fixture: bool,
}

impl Pending {
    pub(in crate::window::transcript) fn parsed_for(
        &self,
        source: &str,
    ) -> Option<(&Layout, usize)> {
        let parsed = self.parsed.as_deref()?;
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
            self.request_source = None;
            return true;
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
        self.failed = false;
        self.request_source = Some(Arc::from(source));
        !preserves_ticket
    }

    pub(in crate::window::transcript) fn changed(&self) -> bool {
        #[cfg(test)]
        if !self.asynchronous_fixture {
            return self.revision.is_none();
        }
        self.request_source.is_some()
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
    ) -> bool {
        #[cfg(test)]
        if !self.asynchronous_fixture {
            return self.refresh_fixture(ui, revision, source);
        }
        use crate::markdown_service::{Service, Status};
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
                Status::Pending => ui
                    .ctx()
                    .request_repaint_after(std::time::Duration::from_millis(16)),
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
                        self.request_source = None;
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
                    self.request_source = None;
                    service.release(ticket);
                    self.ticket = None;
                    self.ticket_source = None;
                    changed = true;
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_secs(5));
                }
            }
        }
        if self.ticket.is_none()
            && !self.failed
            && let Some(request_source) = &self.request_source
        {
            if self.service.is_none() {
                self.service = Service::get(ui.ctx());
            }
            let Some(service) = &self.service else {
                // Recovered messages can precede the asynchronously prepared
                // broker image. Startup will repaint when it becomes available.
                return changed;
            };
            let ticket = service.request(Arc::clone(request_source));
            if ticket.admitted() {
                self.ticket = Some(ticket);
                self.ticket_source = self.request_source.take();
                // Admission follows polling. An idle transcript needs another
                // frame to collect even the first completed parse.
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(16));
            } else if service.unavailable() {
                self.failed = true;
                self.request_source = None;
                changed = true;
            } else {
                // Backpressure is temporary. Retain one immutable source owner
                // and retry admission, without restarting a failed worker.
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(25));
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
        assert_eq!(pending.request_source.as_deref(), Some(first));

        // Simulate dispatch of the first ticket and completion of parsed prefix.
        pending.ticket_source = pending.request_source.take();
        let wire_bytes = super::super::parse_wire(first);
        pending.parsed = wire::decode(first, &wire_bytes).map(Arc::new);
        pending.parsed_source = pending.ticket_source.take();

        // Dispatch a second ticket, then extend the stream again while in flight.
        let second = "# Title\n**bold**";
        assert!(pending.begin_revision(2, second));
        pending.ticket_source = pending.request_source.take();

        let third = "# Title\n**bold** tail";
        assert!(!pending.begin_revision(3, third));
        assert_eq!(pending.ticket_source.as_deref(), Some(second));
        assert_eq!(pending.request_source.as_deref(), Some(third));
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
        assert_eq!(pending.request_source.as_deref(), Some(replaced));
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
                    .map(|cache| cache.body.job.text.as_str()),
                Some(source)
            );
            assert!(
                chat.output().viewport_output[&egui::ViewportId::ROOT].repaint_delay
                    > std::time::Duration::from_millis(16)
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
                    <= std::time::Duration::from_millis(16)
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
            chat.step();
            let message = &chat.state().transcript.messages[0];
            assert_eq!(
                message
                    .cache
                    .as_ref()
                    .map(|cache| cache.body.job.text.as_str()),
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
