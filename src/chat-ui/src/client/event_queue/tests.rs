//! Queue accounting and cancellation fixtures; no inference or timing claims.

use super::*;
use crate::client::{Completion, Message};
use std::{sync::mpsc, thread, time::Duration};

fn fill(sender: &EventSender) -> Result<()> {
    for index in 0..STREAM_EVENTS {
        let mut event = Some(if index.is_multiple_of(2) {
            Event::Text("x".into())
        } else {
            Event::Reasoning("y".into())
        });
        assert!(sender.try_stream(&mut event)?);
        assert!(event.is_none());
    }
    Ok(())
}

#[test]
fn stream_entries_and_payload_are_bounded_and_lifecycle_order_is_retained() -> Result<()> {
    let (sender, receiver) = channel();
    fill(&sender)?;
    let mut pending = Some(Event::Text("pending".into()));
    assert!(!sender.try_stream(&mut pending)?);
    assert!(pending.is_some());
    sender
        .send(Event::Stage("Review"))
        .map_err(|()| Failure::Shutdown)?;
    sender
        .send(Event::Draft("Retained draft".into()))
        .map_err(|()| Failure::Shutdown)?;
    sender
        .send(Event::Committed(vec![Message {
            role: "assistant",
            content: "Committed".into(),
            reasoning_content: None,
            images: Arc::default(),
        }]))
        .map_err(|()| Failure::Shutdown)?;
    sender
        .send(Event::Complete(Completion::Finished))
        .map_err(|()| Failure::Shutdown)?;
    assert_eq!(
        receiver.try_iter().take(STREAM_EVENTS).count(),
        STREAM_EVENTS
    );
    assert!(matches!(receiver.try_recv(), Ok(Event::Stage("Review"))));
    assert!(matches!(receiver.try_recv(), Ok(Event::Draft(text)) if text == "Retained draft"));
    assert!(
        matches!(receiver.try_recv(), Ok(Event::Committed(history)) if history[0].content.as_ref() == "Committed")
    );
    assert!(matches!(
        receiver.try_recv(),
        Ok(Event::Complete(Completion::Finished))
    ));
    assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

    let mut maximum = Some(Event::Text("x".repeat(MAX_EVENT)));
    assert!(sender.try_stream(&mut maximum)?);
    let mut surplus = Some(Event::Reasoning("y".into()));
    assert!(!sender.try_stream(&mut surplus)?);
    assert!(receiver.try_recv().is_ok());
    assert!(sender.try_stream(&mut surplus)?);
    Ok(())
}

#[test]
fn tiny_deltas_coalesce_without_crossing_kind_or_lifecycle_boundaries() -> Result<()> {
    let (sender, receiver) = channel();
    for _ in 0..=(COALESCE_BYTES * 2) {
        assert!(sender.try_stream(&mut Some(Event::Text("x".into())))?);
    }
    assert!(sender.try_stream(&mut Some(Event::Reasoning("why".into())))?);
    sender
        .send(Event::Stage("New stage"))
        .map_err(|()| Failure::Shutdown)?;
    assert!(sender.try_stream(&mut Some(Event::Text("new".into())))?);
    let events: Vec<_> = receiver.try_iter().collect();
    assert_eq!(events.len(), 6);
    for (event, expected) in events[..3].iter().zip([COALESCE_BYTES, COALESCE_BYTES, 1]) {
        assert!(matches!(event, Event::Text(text) if text.len() == expected));
    }
    assert!(matches!(&events[3], Event::Reasoning(text) if text == "why"));
    assert!(matches!(events[4], Event::Stage("New stage")));
    assert!(matches!(&events[5], Event::Text(text) if text == "new"));
    Ok(())
}

#[test]
fn generic_send_keeps_stream_bounds_and_retained_answers_have_a_separate_body_limit() -> Result<()>
{
    let (sender, receiver) = channel();
    sender
        .send(Event::Text("x".repeat(MAX_EVENT)))
        .map_err(|()| Failure::Shutdown)?;
    assert_eq!(sender.send(Event::Text("surplus".into())), Err(()));
    sender
        .send(Event::RetainedAnswer("r".repeat(MAX_BODY)))
        .map_err(|()| Failure::Shutdown)?;
    sender
        .send(Event::Complete(Completion::TokenLimit))
        .map_err(|()| Failure::Shutdown)?;
    assert!(matches!(receiver.try_recv(), Ok(Event::Text(text)) if text.len() == MAX_EVENT));
    assert!(
        matches!(receiver.try_recv(), Ok(Event::RetainedAnswer(text)) if text.len() == MAX_BODY)
    );
    assert!(matches!(
        receiver.try_recv(),
        Ok(Event::Complete(Completion::TokenLimit))
    ));
    assert!(sender.send(Event::Text("x".repeat(MAX_EVENT + 1))).is_err());
    assert!(
        sender
            .send(Event::RetainedAnswer("x".repeat(MAX_BODY + 1)))
            .is_err()
    );
    assert!(sender.send(Event::Draft("x".repeat(MAX_BODY + 1))).is_err());
    assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    Ok(())
}

#[test]
fn saturated_delivery_remains_cancellable_and_terminal_notifications_fit() -> Result<()> {
    for shutdown in [false, true] {
        let (sender, receiver) = channel();
        fill(&sender)?;
        let (commands, incoming) = mpsc::channel();
        let (finished, result) = mpsc::channel();
        let (started, waiting) = mpsc::channel();
        sender.0.lock().map_err(|_| Failure::Shutdown)?.wait_started = Some(started);
        let worker = thread::spawn(move || {
            let outcome = sender.stream(Event::Text("surplus".into()), &incoming);
            if !shutdown {
                assert!(
                    sender
                        .send(Event::Stopping("Original prompt".into()))
                        .is_ok()
                );
                assert!(sender.send(Event::Stopped).is_ok());
            }
            assert!(finished.send(outcome).is_ok());
        });
        assert!(waiting.recv_timeout(Duration::from_secs(3)).is_ok());
        assert!(
            commands
                .send(if shutdown {
                    Command::Shutdown
                } else {
                    Command::Stop
                })
                .is_ok()
        );
        assert_eq!(
            result.recv_timeout(Duration::from_secs(3)),
            Ok(Err(if shutdown {
                Failure::Shutdown
            } else {
                Failure::Stopped
            }))
        );
        assert!(worker.join().is_ok());
        assert_eq!(
            receiver.try_iter().take(STREAM_EVENTS).count(),
            STREAM_EVENTS
        );
        if !shutdown {
            assert!(
                matches!(receiver.try_recv(), Ok(Event::Stopping(text)) if text == "Original prompt")
            );
            assert!(matches!(receiver.try_recv(), Ok(Event::Stopped)));
        }
        assert!(matches!(
            receiver.try_recv(),
            Err(TryRecvError::Disconnected)
        ));
    }
    Ok(())
}

#[test]
fn backpressured_delivery_resumes_with_exact_fifo_bytes_after_consumption() -> Result<()> {
    let (sender, receiver) = channel();
    fill(&sender)?;
    let (_commands, incoming) = mpsc::channel();
    let (started, waiting) = mpsc::channel();
    sender.0.lock().map_err(|_| Failure::Shutdown)?.wait_started = Some(started);
    let (finished, outcome) = mpsc::channel();
    let worker = thread::spawn(move || {
        let result = sender.stream(Event::Text("resumed".into()), &incoming);
        assert!(finished.send(result).is_ok());
    });
    assert!(waiting.recv_timeout(Duration::from_secs(3)).is_ok());
    assert!(matches!(receiver.try_recv(), Ok(Event::Text(text)) if text == "x"));
    assert_eq!(outcome.recv_timeout(Duration::from_secs(3)), Ok(Ok(())));
    assert!(worker.join().is_ok());
    let events: Vec<_> = receiver.try_iter().collect();
    assert_eq!(events.len(), STREAM_EVENTS);
    for (index, event) in events[..STREAM_EVENTS - 1].iter().enumerate() {
        if index.is_multiple_of(2) {
            assert!(matches!(event, Event::Reasoning(text) if text == "y"));
        } else {
            assert!(matches!(event, Event::Text(text) if text == "x"));
        }
    }
    assert!(matches!(&events[STREAM_EVENTS - 1], Event::Text(text) if text == "resumed"));
    Ok(())
}

#[test]
fn saturated_closed_consumer_and_control_overflow_fail_without_waiting() -> Result<()> {
    let (sender, receiver) = channel();
    fill(&sender)?;
    let (_commands, incoming) = mpsc::channel();
    for _ in 0..CONTROL_RESERVE {
        sender
            .send(Event::Stage("Reserved"))
            .map_err(|()| Failure::Shutdown)?;
    }
    assert_eq!(sender.send(Event::Stopped), Err(()));
    drop(receiver);
    assert_eq!(
        sender.stream(Event::Text("surplus".into()), &incoming),
        Err(Failure::Shutdown)
    );
    Ok(())
}

#[test]
fn saturated_delivery_stops_when_command_sender_disconnects() -> Result<()> {
    let (sender, receiver) = channel();
    fill(&sender)?;
    let (commands, incoming) = mpsc::channel();
    drop(commands);
    assert_eq!(
        sender.stream(Event::Text("surplus".into()), &incoming),
        Err(Failure::Shutdown)
    );
    assert_eq!(receiver.try_iter().count(), STREAM_EVENTS);
    Ok(())
}

#[test]
fn event_sender_invokes_repaint_wakeup_on_posted_events_and_drop() -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let wakeups = Arc::new(AtomicUsize::new(0));
    let callback_wakeups = Arc::clone(&wakeups);
    let (sender, receiver) = channel_with_repaint(Arc::new(move || {
        callback_wakeups.fetch_add(1, Ordering::Relaxed);
    }));
    // Replace wakeup with one that also verifies the queue mutex is released before invocation.
    let probe_wakeups = Arc::clone(&wakeups);
    let shared = Arc::clone(&sender.0);
    sender.0.lock().map_err(|_| Failure::Shutdown)?.repaint = Some(Arc::new(move || {
        assert!(
            shared.try_lock().is_ok(),
            "Repaint wakeup must run outside the queue lock"
        );
        probe_wakeups.fetch_add(1, Ordering::Relaxed);
    }));

    sender
        .send(Event::Stage("Thinking…"))
        .map_err(|()| Failure::Shutdown)?;
    assert_eq!(wakeups.load(Ordering::Relaxed), 1);

    // Subsequent sends before the UI drains reuse the pending edge-triggered wakeup.
    sender
        .send(Event::Text("a".into()))
        .map_err(|()| Failure::Shutdown)?;
    assert_eq!(wakeups.load(Ordering::Relaxed), 1);

    // Coalesced text delta also keeps the single pending wakeup until drained.
    sender
        .send(Event::Text("b".into()))
        .map_err(|()| Failure::Shutdown)?;
    assert_eq!(wakeups.load(Ordering::Relaxed), 1);

    // Draining resets the edge trigger so the next send schedules a fresh wakeup.
    let mut drained = Vec::new();
    assert_eq!(receiver.drain_into(&mut drained, 64), Ok((false, false)));
    assert_eq!(drained.len(), 2);
    assert!(matches!(drained[0], Event::Stage("Thinking…")));
    assert!(matches!(&drained[1], Event::Text(text) if text == "ab"));

    sender
        .send(Event::Text("c".into()))
        .map_err(|()| Failure::Shutdown)?;
    assert_eq!(wakeups.load(Ordering::Relaxed), 2);
    assert!(matches!(receiver.try_recv(), Ok(Event::Text(text)) if text == "c"));

    // Saturate the stream queue; rejected backpressure send must not spuriously wake.
    fill(&sender)?;
    let mut saturated = Vec::new();
    assert_eq!(receiver.drain_into(&mut saturated, 64), Ok((false, false)));
    assert_eq!(saturated.len(), STREAM_EVENTS);
    fill(&sender)?;
    let count_when_full = wakeups.load(Ordering::Relaxed);
    assert_eq!(sender.send(Event::Text("overflow".into())), Err(()));
    assert_eq!(wakeups.load(Ordering::Relaxed), count_when_full);

    // Draining resets the edge trigger before drop so sender shutdown wakes the UI.
    saturated.clear();
    assert_eq!(receiver.drain_into(&mut saturated, 64), Ok((false, false)));
    drop(sender);
    assert_eq!(wakeups.load(Ordering::Relaxed), count_when_full + 1);
    drop(receiver);
    Ok(())
}

#[test]
fn continuing_coalesces_per_stage_without_consuming_control_reserve() -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Coalescing across intervening Text/Reasoning on a stream-full queue leaves room for Committed+Complete.
    let (sender, receiver) = channel();
    fill(&sender)?;
    for part in 2..=34_u32 {
        sender
            .send(Event::Continuing(part))
            .map_err(|()| Failure::Shutdown)?;
    }
    sender
        .send(Event::Committed(vec![Message {
            role: "assistant",
            content: "Final".into(),
            reasoning_content: None,
            images: Arc::default(),
        }]))
        .map_err(|()| Failure::Shutdown)?;
    sender
        .send(Event::Complete(Completion::Finished))
        .map_err(|()| Failure::Shutdown)?;
    assert_eq!(
        receiver.try_iter().take(STREAM_EVENTS).count(),
        STREAM_EVENTS
    );
    assert!(matches!(receiver.try_recv(), Ok(Event::Continuing(34))));
    assert!(matches!(receiver.try_recv(), Ok(Event::Committed(_))));
    assert!(matches!(
        receiver.try_recv(),
        Ok(Event::Complete(Completion::Finished))
    ));
    assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

    // Stage-boundary isolation: Review's Continuing never overwrites Draft's undrained Continuing.
    let (sender_stage, receiver_stage) = channel();
    for event in [
        Event::Stage("Draft"),
        Event::Text("d1".into()),
        Event::Continuing(2),
        Event::Draft("d1 d2".into()),
        Event::Stage("Review"),
        Event::Text("r1".into()),
        Event::Continuing(2),
        Event::Continuing(3),
    ] {
        sender_stage.send(event).map_err(|()| Failure::Shutdown)?;
    }
    let events: Vec<_> = receiver_stage.try_iter().collect();
    assert_eq!(events.len(), 7);
    assert!(matches!(events[0], Event::Stage("Draft")));
    assert!(matches!(&events[1], Event::Text(t) if t == "d1"));
    assert!(matches!(events[2], Event::Continuing(2)));
    assert!(matches!(&events[3], Event::Draft(t) if t == "d1 d2"));
    assert!(matches!(events[4], Event::Stage("Review")));
    assert!(matches!(&events[5], Event::Text(t) if t == "r1"));
    assert!(matches!(events[6], Event::Continuing(3)));

    // Full queue without an existing Continuing in the current stage drops silently (Ok),
    // while other control events return Err(()), and dropped receiver still fails closed.
    let (sender_full, receiver_full) = channel();
    fill(&sender_full)?;
    sender_full
        .send(Event::Continuing(2))
        .map_err(|()| Failure::Shutdown)?;
    sender_full
        .send(Event::Stage("New stage"))
        .map_err(|()| Failure::Shutdown)?;
    for _ in 0..(CONTROL_RESERVE - 2) {
        sender_full
            .send(Event::Stage("Reserved"))
            .map_err(|()| Failure::Shutdown)?;
    }
    assert_eq!(sender_full.send(Event::Continuing(2)), Ok(()));
    assert_eq!(sender_full.send(Event::Stopped), Err(()));
    drop(receiver_full);
    assert_eq!(sender_full.send(Event::Continuing(3)), Err(()));

    // Coalescing invokes repaint outside the queue lock and keeps stream_bytes = 0 while
    // separating adjacent Event::Text deltas.
    let wakeups = Arc::new(AtomicUsize::new(0));
    let probe_wakeups = Arc::clone(&wakeups);
    let (sender_wake, receiver_wake) = channel_with_repaint(Arc::new(|| {}));
    let shared = Arc::clone(&sender_wake.0);
    sender_wake.0.lock().map_err(|_| Failure::Shutdown)?.repaint = Some(Arc::new(move || {
        assert!(
            shared.try_lock().is_ok(),
            "Continuing coalescing repaint must run outside the queue lock"
        );
        probe_wakeups.fetch_add(1, Ordering::Relaxed);
    }));
    sender_wake
        .send(Event::Text("a".into()))
        .map_err(|()| Failure::Shutdown)?;
    sender_wake
        .send(Event::Continuing(2))
        .map_err(|()| Failure::Shutdown)?;
    let mut drained = Vec::new();
    assert_eq!(
        receiver_wake.drain_into(&mut drained, 8),
        Ok((false, false))
    );
    assert_eq!(wakeups.load(Ordering::Relaxed), 1);
    sender_wake
        .send(Event::Continuing(3))
        .map_err(|()| Failure::Shutdown)?;
    sender_wake
        .send(Event::Continuing(4))
        .map_err(|()| Failure::Shutdown)?;
    assert_eq!(wakeups.load(Ordering::Relaxed), 2);
    sender_wake
        .send(Event::Text("b".into()))
        .map_err(|()| Failure::Shutdown)?;
    assert_eq!(
        sender_wake
            .0
            .lock()
            .map_err(|_| Failure::Shutdown)?
            .stream_bytes,
        1
    );
    let events_wake: Vec<_> = receiver_wake.try_iter().collect();
    assert_eq!(events_wake.len(), 2);
    assert!(matches!(events_wake[0], Event::Continuing(4)));
    assert!(matches!(&events_wake[1], Event::Text(t) if t == "b"));
    Ok(())
}
