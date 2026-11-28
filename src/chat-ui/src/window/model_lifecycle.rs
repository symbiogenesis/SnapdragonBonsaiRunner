//! Admission and teardown state for the single owned model worker.

use crate::client::Event;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ModelLifecycle {
    Unavailable,
    Preparing,
    Restarting,
    Ready,
    Generating,
    AwaitingTeardown,
    Stopping,
    StoppingModel,
    Stopped,
    Retryable,
}

impl ModelLifecycle {
    pub(super) const fn ready(self) -> bool {
        matches!(self, Self::Ready | Self::Generating)
    }

    pub(super) const fn busy(self) -> bool {
        matches!(self, Self::Generating)
    }

    pub(super) const fn workspace_available(self) -> bool {
        // A generating turn retains its original conversation while local
        // navigation and drafts remain available. Teardown still settles first.
        matches!(
            self,
            Self::Unavailable
                | Self::Preparing
                | Self::Ready
                | Self::Generating
                | Self::Stopped
                | Self::Retryable
        )
    }

    pub(super) const fn stopping(self) -> bool {
        matches!(self, Self::Stopping | Self::StoppingModel)
    }

    pub(super) const fn retryable(self) -> bool {
        matches!(self, Self::Retryable)
    }

    pub(super) const fn reloadable(self) -> bool {
        matches!(self, Self::Retryable | Self::Stopped)
    }

    pub(super) const fn waiting_model(self) -> bool {
        matches!(
            self,
            Self::Preparing
                | Self::Restarting
                | Self::AwaitingTeardown
                | Self::Stopping
                | Self::StoppingModel
        )
    }

    /// Events arrive in order from one worker. Retry is sent only after its
    /// teardown notice; attaching a replacement worker replaces its channel.
    /// Reject stale events that cannot belong to the current lifecycle phase.
    pub(super) const fn accepts(self, event: &Event) -> bool {
        match event {
            Event::Starting => matches!(self, Self::Preparing | Self::Restarting),
            Event::Restarting => matches!(self, Self::Stopping),
            Event::ContextCapacity(_)
            | Event::VisionAvailable(_)
            | Event::DaisyRoseAvailable(_)
            | Event::CognitiveControlsAvailable(_)
            | Event::Ready => {
                matches!(self, Self::Preparing)
            }
            Event::Text(_)
            | Event::Reasoning(_)
            | Event::Continuing(_)
            | Event::Stage(_)
            | Event::Draft(_)
            | Event::RetainedAnswer(_)
            | Event::Complete(_) => matches!(self, Self::Generating),
            Event::Committed(_) | Event::Stopping(_) => {
                matches!(
                    self,
                    Self::Generating | Self::Stopping | Self::StoppingModel
                )
            }
            Event::Stopped => matches!(self, Self::Stopping),
            Event::ModelStopped => matches!(
                self,
                Self::StoppingModel | Self::AwaitingTeardown | Self::Stopped | Self::Retryable
            ),
            Event::Rejected { retry_required, .. } => {
                matches!(
                    self,
                    Self::Generating | Self::Stopping | Self::StoppingModel
                ) || (*retry_required
                    && matches!(
                        self,
                        Self::AwaitingTeardown | Self::Stopped | Self::Retryable | Self::Restarting
                    ))
            }
            Event::Retryable(_) => !matches!(self, Self::Unavailable | Self::Restarting),
            Event::Failed(_) => !matches!(self, Self::Unavailable),
        }
    }
}
