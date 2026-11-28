//! Bounded worker-to-window delivery with cancellation-aware stream pressure.

use super::{Command, Event, Failure, IO_POLL, MAX_BODY, MAX_EVENT, Result};
use std::{
    borrow::Cow,
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, RecvTimeoutError, TryRecvError},
    },
};

const STREAM_EVENTS: usize = 64;
const CONTROL_RESERVE: usize = 16;
const COALESCE_BYTES: usize = 16 * 1024;

type RepaintCallback = Arc<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct Queue {
    events: VecDeque<Event>,
    stream_bytes: usize,
    sender_closed: bool,
    receiver_closed: bool,
    repaint_scheduled: bool,
    repaint: Option<RepaintCallback>,
    #[cfg(test)]
    wait_started: Option<std::sync::mpsc::Sender<()>>,
}

impl Queue {
    fn take_repaint(&mut self) -> Option<RepaintCallback> {
        if self.repaint_scheduled {
            None
        } else {
            let callback = self.repaint.clone()?;
            self.repaint_scheduled = true;
            Some(callback)
        }
    }
}

/// The single inference worker's bounded notification endpoint.
pub(crate) struct EventSender(Arc<Mutex<Queue>>);

/// The native window's nonblocking notification endpoint.
pub(crate) struct EventReceiver(Arc<Mutex<Queue>>);

/// Create one FIFO; lifecycle events retain room when streaming is saturated.
pub(crate) fn channel() -> (EventSender, EventReceiver) {
    channel_with_wakeup(None)
}

/// Create one FIFO that invokes `repaint` outside the queue lock whenever a worker event is posted.
pub(crate) fn channel_with_repaint(repaint: RepaintCallback) -> (EventSender, EventReceiver) {
    channel_with_wakeup(Some(repaint))
}

fn channel_with_wakeup(repaint: Option<RepaintCallback>) -> (EventSender, EventReceiver) {
    let shared = Arc::new(Mutex::new(Queue {
        repaint,
        ..Queue::default()
    }));
    (EventSender(Arc::clone(&shared)), EventReceiver(shared))
}

fn notify_repaint(repaint: Option<RepaintCallback>) {
    if let Some(callback) = repaint {
        callback();
    }
}

const fn stream_bytes(event: &Event) -> usize {
    match event {
        Event::Text(text) | Event::Reasoning(text) => text.len(),
        _ => 0,
    }
}

impl EventSender {
    // A request has fewer than CONTROL_RESERVE lifecycle notifications. Another
    // request requires the window to consume its preceding completion/retry.
    // Unexpected exhaustion fails closed instead of blocking process teardown.
    pub(crate) fn send(&self, event: Event) -> std::result::Result<(), ()> {
        if matches!(event, Event::Text(_) | Event::Reasoning(_)) {
            return if self.try_stream(&mut Some(event)).map_err(|_| ())? {
                Ok(())
            } else {
                Err(())
            };
        }
        if matches!(&event, Event::Draft(text) | Event::RetainedAnswer(text) if text.len() > MAX_BODY)
        {
            return Err(());
        }
        let mut queue = self.0.lock().map_err(|_| ())?;
        if queue.receiver_closed {
            return Err(());
        }
        if let Event::Continuing(part) = event {
            if let Some(existing) = queue
                .events
                .iter_mut()
                .rev()
                .take_while(|queued| !matches!(queued, Event::Stage(_)))
                .find(|queued| matches!(queued, Event::Continuing(_)))
            {
                *existing = Event::Continuing(part);
                let repaint = queue.take_repaint();
                drop(queue);
                notify_repaint(repaint);
                return Ok(());
            }
            if queue.events.len() >= STREAM_EVENTS + CONTROL_RESERVE {
                return Ok(());
            }
        } else if queue.events.len() >= STREAM_EVENTS + CONTROL_RESERVE {
            return Err(());
        }
        queue.events.push_back(event);
        let repaint = queue.take_repaint();
        drop(queue);
        notify_repaint(repaint);
        Ok(())
    }

    fn try_stream_cow(&self, text: &mut Option<Cow<'_, str>>, reasoning: bool) -> Result<bool> {
        let mut queue = self.0.lock().map_err(|_| Failure::Shutdown)?;
        if queue.receiver_closed {
            return Err(Failure::Shutdown);
        }
        let bytes = text.as_deref().map_or(0, str::len);
        if bytes > MAX_EVENT {
            return Err(Failure::Message("A streaming update exceeded its limit."));
        }
        if queue.stream_bytes + bytes > MAX_EVENT {
            return Ok(false);
        }
        let combined = match (queue.events.back_mut(), text.as_deref(), reasoning) {
            (Some(Event::Text(tail)), Some(delta), false)
            | (Some(Event::Reasoning(tail)), Some(delta), true)
                if tail.len() + delta.len() <= COALESCE_BYTES =>
            {
                tail.push_str(delta);
                true
            }
            _ => false,
        };
        if combined {
            queue.stream_bytes += bytes;
            let repaint = queue.take_repaint();
            drop(queue);
            *text = None;
            notify_repaint(repaint);
            return Ok(true);
        }
        if queue.events.len() >= STREAM_EVENTS {
            return Ok(false);
        }
        let mut posted = false;
        if let Some(delta) = text.take() {
            let owned = delta.into_owned();
            queue.stream_bytes += bytes;
            queue.events.push_back(if reasoning {
                Event::Reasoning(owned)
            } else {
                Event::Text(owned)
            });
            posted = true;
        }
        let repaint = if posted { queue.take_repaint() } else { None };
        drop(queue);
        notify_repaint(repaint);
        Ok(true)
    }

    fn try_stream(&self, event: &mut Option<Event>) -> Result<bool> {
        let mut queue = self.0.lock().map_err(|_| Failure::Shutdown)?;
        if queue.receiver_closed {
            return Err(Failure::Shutdown);
        }
        let bytes = event.as_ref().map_or(0, stream_bytes);
        if bytes > MAX_EVENT {
            return Err(Failure::Message("A streaming update exceeded its limit."));
        }
        if queue.stream_bytes + bytes > MAX_EVENT {
            return Ok(false);
        }
        let pending = event.as_ref();
        let tail = queue.events.back_mut();
        let combined = match (tail, pending) {
            (Some(Event::Text(tail)), Some(Event::Text(text)))
            | (Some(Event::Reasoning(tail)), Some(Event::Reasoning(text)))
                if tail.len() + text.len() <= COALESCE_BYTES =>
            {
                tail.push_str(text);
                true
            }
            _ => false,
        };
        if combined {
            queue.stream_bytes += bytes;
            let repaint = queue.take_repaint();
            drop(queue);
            *event = None;
            notify_repaint(repaint);
            return Ok(true);
        }
        if queue.events.len() >= STREAM_EVENTS {
            return Ok(false);
        }
        let mut posted = false;
        if let Some(event) = event.take() {
            queue.stream_bytes += bytes;
            queue.events.push_back(event);
            posted = true;
        }
        let repaint = if posted { queue.take_repaint() } else { None };
        drop(queue);
        notify_repaint(repaint);
        Ok(true)
    }

    pub(super) fn stream_text(
        &self,
        text: Cow<'_, str>,
        reasoning: bool,
        commands: &Receiver<Command>,
    ) -> Result<()> {
        let mut pending = Some(text);
        loop {
            super::check_commands(commands)?;
            if self.try_stream_cow(&mut pending, reasoning)? {
                return Ok(());
            }
            #[cfg(test)]
            if let Ok(mut queue) = self.0.lock()
                && let Some(started) = queue.wait_started.take()
            {
                let _sent = started.send(());
            }
            match commands.recv_timeout(IO_POLL) {
                Ok(Command::StopModel) => return Err(Failure::ModelStopped),
                Ok(Command::Stop) => return Err(Failure::Stopped),
                Ok(Command::Shutdown) | Err(RecvTimeoutError::Disconnected) => {
                    return Err(Failure::Shutdown);
                }
                Ok(_) | Err(RecvTimeoutError::Timeout) => {}
            }
        }
    }

    #[cfg(test)]
    pub(super) fn stream(&self, event: Event, commands: &Receiver<Command>) -> Result<()> {
        let mut event = Some(event);
        loop {
            super::check_commands(commands)?;
            if self.try_stream(&mut event)? {
                return Ok(());
            }
            // The native window is behind. Do not read another socket byte or
            // generate more notifications; Stop/Shutdown wake this wait directly.
            #[cfg(test)]
            if let Ok(mut queue) = self.0.lock()
                && let Some(started) = queue.wait_started.take()
            {
                let _sent = started.send(());
            }
            match commands.recv_timeout(IO_POLL) {
                Ok(Command::StopModel) => return Err(Failure::ModelStopped),
                Ok(Command::Stop) => return Err(Failure::Stopped),
                Ok(Command::Shutdown) | Err(RecvTimeoutError::Disconnected) => {
                    return Err(Failure::Shutdown);
                }
                Ok(_) | Err(RecvTimeoutError::Timeout) => {}
            }
        }
    }
}

impl Drop for EventSender {
    fn drop(&mut self) {
        let repaint = if let Ok(mut queue) = self.0.lock() {
            queue.sender_closed = true;
            queue.take_repaint()
        } else {
            None
        };
        notify_repaint(repaint);
    }
}

impl EventReceiver {
    pub(crate) fn drain_into(
        &self,
        out: &mut Vec<Event>,
        max: usize,
    ) -> std::result::Result<(bool, bool), TryRecvError> {
        let mut queue = self.0.lock().map_err(|_| TryRecvError::Disconnected)?;
        queue.repaint_scheduled = false;
        for _ in 0..max {
            let Some(event) = queue.events.pop_front() else {
                break;
            };
            queue.stream_bytes -= stream_bytes(&event);
            out.push(event);
        }
        let has_more = !queue.events.is_empty();
        let disconnected = !has_more && queue.sender_closed;
        Ok((disconnected, has_more))
    }

    #[cfg(test)]
    pub(crate) fn try_recv(&self) -> std::result::Result<Event, TryRecvError> {
        let mut queue = self.0.lock().map_err(|_| TryRecvError::Disconnected)?;
        queue.repaint_scheduled = false;
        if let Some(event) = queue.events.pop_front() {
            queue.stream_bytes -= stream_bytes(&event);
            Ok(event)
        } else if queue.sender_closed {
            Err(TryRecvError::Disconnected)
        } else {
            Err(TryRecvError::Empty)
        }
    }

    #[cfg(test)]
    pub(crate) fn try_iter(&self) -> impl Iterator<Item = Event> {
        std::iter::from_fn(|| self.try_recv().ok())
    }
}

impl Drop for EventReceiver {
    fn drop(&mut self) {
        if let Ok(mut queue) = self.0.lock() {
            queue.receiver_closed = true;
            queue.events.clear();
            queue.stream_bytes = 0;
        }
    }
}

#[cfg(test)]
mod tests;
