//! Per-message asynchronous parser state. Production never parses in the UI.
use super::{Layout, MAX_MARKDOWN_BYTES, egui, wire};
use std::sync::Arc;

#[derive(Default)]
pub(in crate::window::transcript) struct Pending {
    revision: Option<u64>,
    parsed: Option<Arc<Layout>>,
    failed: bool,
    cleanup_failed: bool,
    #[cfg(not(test))]
    service: Option<Arc<crate::markdown_service::Service>>,
    #[cfg(not(test))]
    ticket: Option<crate::markdown_service::Ticket>,
    #[cfg(not(test))]
    request_source: Option<Arc<str>>,
}

impl Pending {
    pub(in crate::window::transcript) fn parsed(&self) -> Option<&Layout> {
        self.parsed.as_deref()
    }

    pub(in crate::window::transcript) const fn failed(&self) -> bool {
        self.failed
    }

    pub(in crate::window::transcript) const fn cleanup_failed(&self) -> bool {
        self.cleanup_failed
    }

    #[cfg(test)]
    pub(in crate::window::transcript) fn changed(&self) -> bool {
        self.revision.is_none()
    }

    #[cfg(not(test))]
    pub(in crate::window::transcript) fn changed(&self) -> bool {
        self.request_source.is_some()
            || self.ticket.is_some()
            || self
                .service
                .as_ref()
                .is_some_and(|service| service.cleanup_failed() != self.cleanup_failed)
    }

    #[cfg(test)]
    pub(in crate::window::transcript) fn refresh(
        &mut self,
        _ui: &egui::Ui,
        revision: u64,
        source: &str,
    ) -> bool {
        if self.revision == Some(revision) {
            return false;
        }
        self.revision = Some(revision);
        // Headless UI fixtures exercise the same validated DTO without starting
        // real sandboxes. Native containment remains an independent gate.
        self.parsed = (source.len() <= MAX_MARKDOWN_BYTES)
            .then(|| wire::decode(source, &super::parse_wire(source)))
            .flatten()
            .map(Arc::new);
        true
    }

    #[cfg(not(test))]
    pub(in crate::window::transcript) fn refresh(
        &mut self,
        ui: &egui::Ui,
        revision: u64,
        source: &str,
    ) -> bool {
        use crate::markdown_service::{Service, Status};
        let mut changed = false;
        if let Some(service) = &self.service {
            let failed = service.cleanup_failed();
            changed = self.cleanup_failed != failed;
            self.cleanup_failed = failed;
        }
        if self.revision != Some(revision) {
            if let Some(service) = &self.service
                && let Some(ticket) = self.ticket.take()
            {
                service.release(ticket);
            }
            self.revision = Some(revision);
            self.parsed = None;
            self.failed = false;
            self.request_source = None;
            changed = true;
            if source.is_empty() || source.len() > MAX_MARKDOWN_BYTES {
                return changed;
            }
            self.request_source = Some(Arc::from(source));
        }
        if let Some(request_source) = &self.request_source {
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
                self.request_source = None;
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
        if let Some(service) = &self.service
            && let Some(ticket) = self.ticket
        {
            match service.poll(ticket) {
                Status::Pending => ui
                    .ctx()
                    .request_repaint_after(std::time::Duration::from_millis(16)),
                Status::Ready(bytes) => {
                    self.parsed = wire::decode(source, &bytes).map(Arc::new);
                    self.failed = self.parsed.is_none();
                    if self.failed {
                        service.reject();
                    }
                    service.release(ticket);
                    self.ticket = None;
                    changed = true;
                }
                Status::Failed => {
                    self.failed = true;
                    service.release(ticket);
                    self.ticket = None;
                    changed = true;
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_secs(5));
                }
            }
        }
        changed
    }
}

#[cfg(not(test))]
impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(service) = &self.service
            && let Some(ticket) = self.ticket.take()
        {
            service.release(ticket);
        }
    }
}
