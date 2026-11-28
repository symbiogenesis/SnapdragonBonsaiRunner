//! Conversation preference and cancellable incremental answer playback.
use super::{State, VOICE_LIMIT, egui, speech};

impl State {
    fn valid_voice(&mut self) -> bool {
        let voice = self.voice_profile.trim();
        if !voice.is_empty() && !voice.contains('\0') && voice.chars().count() <= VOICE_LIMIT {
            return true;
        }
        "Describe the voice in Settings before reading aloud.".clone_into(&mut self.speech_status);
        self.speech_notice = false;
        self.open_settings(super::settings::Page::Voice);
        false
    }

    pub(super) fn toggle_streaming_read_aloud(&mut self) {
        self.speech_notice = false;
        if self.streaming_read_aloud {
            self.streaming_read_aloud = false;
            if self.streaming_active {
                self.stop_speech();
            }
        } else if !self.speech_output {
            if !self.speech_checked && self.tts_configured {
                "Read aloud will be available when speech preparation completes."
                    .clone_into(&mut self.speech_status);
            } else {
                let message = self.read_aloud_help();
                message.clone_into(&mut self.speech_status);
                self.open_voice_setup();
            }
        } else if self.valid_voice() {
            self.streaming_read_aloud = true;
            self.streaming_suppressed = false;
            "New responses will be read aloud as they arrive.".clone_into(&mut self.speech_status);
            if self.model.busy()
                && self
                    .request_conversation
                    .is_none_or(|id| id == self.active_conversation)
                && !self.last_answer.is_empty()
            {
                self.stop_speech();
                self.streaming_suppressed = false;
                self.stream_answer(self.last_answer.clone());
            }
        }
        self.recovery_dirty = true;
        self.record_speech_status();
    }

    pub(super) fn stream_answer(&mut self, text: String) {
        if !self.model.busy()
            || self
                .request_conversation
                .is_some_and(|id| id != self.active_conversation)
            || !self.streaming_read_aloud
            || self.streaming_suppressed
            || !self.speech_output
            || text.is_empty()
        {
            return;
        }
        let super::svg_fence::FilteredDelta::Forward(text) = self.streaming_svg_filter.push(text)
        else {
            return;
        };
        if !self.streaming_active {
            if !self.valid_voice() {
                self.streaming_suppressed = true;
                return;
            }
            self.cancel_microphone();
            if !self.speech_command(speech::Command::BeginStream {
                voice_description: self.voice_profile.trim().to_owned(),
            }) {
                return;
            }
            self.streaming_active = true;
            self.streaming_utf16 = 0;
            self.speaking = true;
            self.speech_loading = true;
            self.speaking_message = self.transcript.messages.last().map(|message| message.id);
            self.speech_notice = false;
            "Reading the response as it arrives.".clone_into(&mut self.speech_status);
        }
        let units = text.encode_utf16().take(speech::MAX_TEXT + 1).count();
        if units > speech::MAX_TEXT.saturating_sub(self.streaming_utf16) {
            self.speech_command(speech::Command::FinishStream);
            self.streaming_active = false;
            self.streaming_utf16 = 0;
            self.streaming_suppressed = true;
            return;
        }
        self.streaming_utf16 += units;
        self.speech_command(speech::Command::AppendStream { text });
    }

    pub(super) fn replace_streaming_stage(&mut self) {
        self.streaming_svg_filter.reset();
        self.streaming_utf16 = 0;
        if self.streaming_active {
            self.stop_speech();
            self.streaming_suppressed = false;
        }
    }

    pub(super) fn read_aloud_controls(&mut self, ui: &mut egui::Ui) {
        let mut enabled = self.streaming_read_aloud;
        if ui
            .add_enabled(
                self.speech_output || enabled,
                egui::Checkbox::new(&mut enabled, "Always Read Aloud"),
            )
            .on_hover_text("Read new responses in short chunks using your Voice profile. Speaker buttons can replay completed responses at any time.")
            .changed()
        {
            self.toggle_streaming_read_aloud();
        }
        if self.speaking && ui.button("Stop speech").clicked() {
            self.stop_speech();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        client,
        window::{ModelLifecycle, transcript::Role},
    };
    use std::{io, sync::mpsc, time::Duration};

    fn fixture() -> io::Result<(State, mpsc::Receiver<speech::Command>)> {
        let (commands, _incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::create(commands, incoming_events, None, None);
        let (session, speech_commands) = speech::Session::fixture()?;
        state.speech = Some(session);
        state.speech_output = true;
        state.preview = false;
        state.model = ModelLifecycle::Generating;
        state.transcript.push(Role::Assistant, String::new());
        Ok((state, speech_commands))
    }

    fn next(commands: &mpsc::Receiver<speech::Command>) -> io::Result<speech::Command> {
        commands
            .recv_timeout(Duration::from_secs(2))
            .map_err(io::Error::other)
    }

    #[test]
    fn streaming_read_aloud_forwards_visible_deltas_before_completion_and_finishes_once()
    -> io::Result<()> {
        let (mut state, commands) = fixture()?;
        assert!(!state.streaming_read_aloud);
        state.toggle_streaming_read_aloud();
        state.model_event(client::Event::Reasoning("Private reasoning".to_owned()));
        state.model_event(client::Event::Draft("Retained draft".to_owned()));
        let delta = "Hello ".to_owned();
        let delta_pointer = delta.as_ptr();
        state.model_event(client::Event::Text(delta));
        assert!(state.model.busy() && state.streaming_active);
        assert!(matches!(next(&commands)?, speech::Command::StopListening));
        assert!(
            matches!(next(&commands)?, speech::Command::BeginStream { voice_description } if voice_description == state.voice_profile)
        );
        assert!(
            matches!(next(&commands)?, speech::Command::AppendStream { text } if text == "Hello " && text.as_ptr() == delta_pointer)
        );
        state.model_event(client::Event::Text("world.".to_owned()));
        assert!(
            matches!(next(&commands)?, speech::Command::AppendStream { text } if text == "world.")
        );
        state.model_event(client::Event::Complete(client::Completion::TokenLimit));
        assert!(matches!(next(&commands)?, speech::Command::FinishStream));
        assert!(!state.model.busy());
        assert_eq!(state.last_answer, "Hello world.");
        Ok(())
    }

    #[test]
    fn streaming_read_aloud_manual_replay_and_stop_suppress_the_current_response() -> io::Result<()>
    {
        let (mut state, commands) = fixture()?;
        state.transcript.append("Older answer", false);
        state.transcript.remove_empty_response();
        let older = state.transcript.messages[0].id;
        state.transcript.push(Role::Assistant, String::new());
        state.toggle_streaming_read_aloud();
        state.model_event(client::Event::Text("New answer".to_owned()));
        for _ in 0..3 {
            next(&commands)?;
        }
        state.toggle_read_aloud(older);
        assert!(matches!(next(&commands)?, speech::Command::StopListening));
        assert!(matches!(next(&commands)?, speech::Command::StopSpeaking));
        assert!(
            matches!(next(&commands)?, speech::Command::Speak { text, .. } if text == "Older answer")
        );
        assert!(state.streaming_read_aloud && state.streaming_suppressed);
        assert!(!state.streaming_active);
        state.model_event(client::Event::Text(" continues".to_owned()));
        state.model_event(client::Event::Complete(client::Completion::Finished));
        assert_eq!(state.speaking_message, Some(older));
        state.toggle_read_aloud(older);
        assert!(matches!(next(&commands)?, speech::Command::StopListening));
        assert!(matches!(next(&commands)?, speech::Command::StopSpeaking));
        assert!(!state.speaking);
        assert!(commands.recv_timeout(Duration::from_millis(20)).is_err());
        Ok(())
    }

    #[test]
    fn streaming_read_aloud_stage_replacement_cancels_and_rejection_never_restarts()
    -> io::Result<()> {
        let (mut state, commands) = fixture()?;
        state.toggle_streaming_read_aloud();
        state.model_event(client::Event::Text("Draft answer".to_owned()));
        for _ in 0..3 {
            next(&commands)?;
        }
        state.model_event(client::Event::Stage("Review"));
        assert!(matches!(next(&commands)?, speech::Command::StopListening));
        assert!(matches!(next(&commands)?, speech::Command::StopSpeaking));
        state.model_event(client::Event::Text("Revised answer".to_owned()));
        assert!(matches!(next(&commands)?, speech::Command::StopListening));
        assert!(matches!(
            next(&commands)?,
            speech::Command::BeginStream { .. }
        ));
        assert!(
            matches!(next(&commands)?, speech::Command::AppendStream { text } if text == "Revised answer")
        );
        state.model_event(client::Event::Rejected {
            message: "Context limit".to_owned(),
            text: "Prompt".to_owned(),
            retry_required: false,
        });
        assert!(matches!(next(&commands)?, speech::Command::StopListening));
        assert!(matches!(next(&commands)?, speech::Command::StopSpeaking));
        state.model_event(client::Event::Text("late text".to_owned()));
        assert!(!state.streaming_active);
        assert!(commands.recv_timeout(Duration::from_millis(20)).is_err());
        Ok(())
    }

    #[test]
    fn streaming_read_aloud_toggle_off_and_current_speaker_cancel_unfinished_audio()
    -> io::Result<()> {
        let (mut state, commands) = fixture()?;
        state.toggle_streaming_read_aloud();
        state.model_event(client::Event::Text("Some text".to_owned()));
        for _ in 0..3 {
            next(&commands)?;
        }
        let current = state.transcript.messages[0].id;
        state.toggle_read_aloud(current);
        assert!(!state.speaking);
        assert!(state.streaming_read_aloud && state.streaming_suppressed);
        assert!(matches!(next(&commands)?, speech::Command::StopListening));
        assert!(matches!(next(&commands)?, speech::Command::StopSpeaking));
        state.toggle_streaming_read_aloud();
        assert!(!state.streaming_read_aloud);
        state.model_event(client::Event::Text(" continues silently".to_owned()));
        assert!(commands.recv_timeout(Duration::from_millis(20)).is_err());
        Ok(())
    }

    #[test]
    fn svg_fences_are_stripped_from_streaming_read_aloud_and_replay_text() -> io::Result<()> {
        let (mut state, commands) = fixture()?;
        state.toggle_streaming_read_aloud();
        for chunk in [
            "Here is an emblem:\n\n",
            "```svg\n<svg viewBox=\"0 0 10 10\">",
            "<circle cx=\"5\" cy=\"5\" r=\"4\"/></svg>\n```\n\n",
            "All set.",
        ] {
            state.model_event(client::Event::Text(chunk.to_owned()));
        }
        state.model_event(client::Event::Complete(client::Completion::Finished));
        let mut spoken = String::new();
        while let Ok(command) = commands.recv_timeout(Duration::from_millis(50)) {
            if let speech::Command::AppendStream { text } = command {
                spoken.push_str(&text);
            }
        }
        assert_eq!(spoken, "Here is an emblem:\n\n\nAll set.");
        assert!(!spoken.contains("<svg"));
        assert!(!spoken.contains("circle"));

        let message_id = state.transcript.messages[0].id;
        assert_eq!(
            state.transcript.speech_text(message_id).as_deref(),
            Some("Here is an emblem:\n\nAll set.")
        );

        state.model = ModelLifecycle::Generating;
        state
            .transcript
            .push(Role::Assistant, "```svg\n<svg/>\n```".to_owned());
        state.model_event(client::Event::Complete(client::Completion::Finished));
        let svg_only_id = state.transcript.messages[1].id;
        assert_eq!(state.transcript.speech_text(svg_only_id), None);
        Ok(())
    }

    #[test]
    fn streaming_read_aloud_stays_continuous_across_automatic_token_limit_continuations()
    -> io::Result<()> {
        let (mut state, speech_commands) = fixture()?;
        let (model_commands, incoming_model) = mpsc::channel();
        state.commands = model_commands;
        state.context_tokens = Some(65_536);
        state.model = ModelLifecycle::Ready;
        state.transcript.clear();
        state.toggle_streaming_read_aloud();

        state.input = "Read a two-part story".to_owned();
        state.submit();
        assert!(matches!(
            incoming_model.try_recv(),
            Ok(client::Command::Send { .. })
        ));
        assert!(matches!(
            next(&speech_commands)?,
            speech::Command::StopSpeaking
        ));

        state.model_event(client::Event::Text("First part. ".to_owned()));
        assert!(matches!(
            next(&speech_commands)?,
            speech::Command::StopListening
        ));
        assert!(matches!(
            next(&speech_commands)?,
            speech::Command::BeginStream { .. }
        ));
        assert!(
            matches!(next(&speech_commands)?, speech::Command::AppendStream { text } if text == "First part. ")
        );

        state.model_event(client::Event::Continuing(2));
        assert!(state.model.busy() && state.streaming_active);
        // Part 1 is not cut off with StopSpeaking or closed early with FinishStream.
        assert!(
            speech_commands
                .recv_timeout(Duration::from_millis(20))
                .is_err()
        );

        state.model_event(client::Event::Text("Second part.".to_owned()));
        assert!(
            matches!(next(&speech_commands)?, speech::Command::AppendStream { text } if text == "Second part.")
        );
        let committed = vec![
            client::Message {
                role: "user",
                content: "Read a two-part story".into(),
                images: std::sync::Arc::default(),
                reasoning_content: None,
            },
            client::Message {
                role: "assistant",
                content: "First part. Second part.".into(),
                images: std::sync::Arc::default(),
                reasoning_content: None,
            },
        ];
        state.model_event(client::Event::Committed(committed));
        state.model_event(client::Event::Complete(client::Completion::Finished));
        assert!(matches!(
            next(&speech_commands)?,
            speech::Command::FinishStream
        ));
        Ok(())
    }
}
