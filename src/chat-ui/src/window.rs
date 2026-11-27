//! Native egui chat presentation. Backend work and audio remain off the UI loop.
#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep presentation internals parent-only."
)]

use crate::{
    client::{self, Command, Completion, Config, Event},
    speech,
};
use eframe::egui::{self, RichText};
use std::{
    collections::VecDeque,
    io,
    sync::mpsc::{self, Receiver, Sender},
    thread::JoinHandle,
    time::{Duration, Instant},
};

mod appearance;
pub(crate) mod atomic_file;
mod attachments;
mod audio_files;
#[cfg(test)]
mod background_tests;
mod branch_diff;
mod branding;
mod coalescing_store;
mod composer;
mod composer_insertion;
#[cfg(test)]
mod composer_unicode_tests;
mod context_capacity;
mod controls;
mod draft_buffer;
#[cfg(test)]
mod ergonomics_tests;
mod error_ui;
#[cfg(test)]
mod error_ui_tests;
mod graphics;
mod history_search;
#[cfg(test)]
mod history_search_tests;
mod image_cache;
pub(crate) use image_cache::fnv1a64;
#[cfg(test)]
mod image_ui_tests;
#[cfg(test)]
mod interaction_tests;
mod iteration;
#[cfg(test)]
mod iteration_ui_tests;
#[cfg(test)]
mod keyboard_tests;
mod lightbox;
mod microphone;
mod model_lifecycle;
mod navigation;
mod persistence;
#[allow(
    unsafe_code,
    reason = "Audited UI-thread native window placement with a borrowed eframe window handle."
)]
mod placement;
#[allow(
    unsafe_code,
    reason = "Audited Windows high-contrast and fatal-dialog boundary."
)]
mod platform;
mod profile;
mod read_aloud;
mod settings;
#[cfg(test)]
mod settings_ui_tests;
mod setup;
mod sidebar_status;
mod svg_fence;
#[cfg(test)]
mod tests;
mod transcript;

mod updates;
mod workspace;

use model_lifecycle::ModelLifecycle;
use transcript::{Role, Transcript};
use workspace::Conversation;

const OUTPUT_TOKENS: [u32; 4] = [512, 1024, 2048, 4096];
const DEFAULT_VOICE: &str = "A warm, clear, natural adult voice with a calm conversational pace.";
// Draft capacity is independent of the broker's serialized byte/token admission.
// Four-byte Unicode scalars bound this native buffer to 4 MiB of UTF-8 text.
const INPUT_LIMIT: usize = 1_048_576;
const VOICE_LIMIT: usize = 1_024;
const MAX_DIAGNOSTICS: usize = 128;
const MAX_DIAGNOSTIC_CHARS: usize = 4_096;
pub(super) const CONVERSATION_DELETED_NOTICE_DURATION: Duration = Duration::from_secs(4);

#[derive(Clone, Copy, Debug)]
pub(super) struct ConversationNotice {
    text: &'static str,
    auto_dismiss_after: Option<Duration>,
    dismiss_at: Option<f64>,
}

impl ConversationNotice {
    pub(super) const fn persistent(text: &'static str) -> Self {
        Self {
            text,
            auto_dismiss_after: None,
            dismiss_at: None,
        }
    }

    pub(super) const fn auto_dismiss(text: &'static str, duration: Duration) -> Self {
        Self {
            text,
            auto_dismiss_after: Some(duration),
            dismiss_at: None,
        }
    }
}

fn initial_theme_check() -> Instant {
    // Synchronize native appearance on the first frame without waiting two seconds.
    Instant::now()
        .checked_sub(Duration::from_secs(2))
        .unwrap_or_else(Instant::now)
}

fn runtime_description(config: &Config) -> String {
    let engine = if config.native_engine {
        "Rust"
    } else {
        "Publisher"
    };
    let weights = if let Some(variant) = config.bonsai25_variant {
        match variant {
            client::Bonsai25Variant::Pq2 => "Bonsai 2.5 PQ2",
            client::Bonsai25Variant::Ptq1 => "Bonsai 2.5 PTQ1",
        }
    } else {
        "PQ2"
    };
    if config.native_engine {
        format!(
            "{engine} CPU · {weights} · {} row workers",
            config.native_row_workers
        )
    } else {
        format!("{engine} CPU · {weights}")
    }
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent broker, audio and presentation state is explicit."
)]
struct State {
    commands: Sender<Command>,
    events: client::EventReceiver,
    drained_events: Vec<Event>,
    repaint_ctx: Option<egui::Context>,
    worker: Option<JoinHandle<()>>,
    markdown_image: Option<std::path::PathBuf>,
    markdown_configured: std::cell::Cell<bool>,
    pending_config: Option<Config>,
    startup_request: Option<crate::startup::Request>,
    startup: Option<crate::startup::Session>,
    setup: setup::Setup,
    startup_failure: Option<std::sync::Arc<std::sync::Mutex<Option<String>>>>,
    updates: Option<updates::Updates>,
    update_exit: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    model: ModelLifecycle,
    awaiting_first_token: bool,
    vision_available: bool,
    preview: bool,
    input: String,
    input_revision: u64,
    draft_attachments: Vec<crate::attachment_import::Attachment>,
    draft_attachments_revision: u64,
    transcript: Transcript,
    committed: Vec<client::Message>,
    context_tokens: Option<usize>,
    context_history: Option<usize>,
    context_draft: context_capacity::DraftCache,
    conversations: Vec<Conversation>,
    conversation_search: history_search::Search,
    active_conversation: u64,
    request_conversation: Option<u64>,
    auto_continuing: bool,
    routing_request: bool,
    next_conversation: u64,
    rename_conversation: Option<(u64, String)>,
    delete_conversation: Option<u64>,
    edit_message: Option<iteration::EditMessage>,
    branch_comparison: Option<iteration::BranchComparison>,
    branch_diff_generation: u64,
    conversation_notice: Option<ConversationNotice>,
    recovery: Option<crate::recovery::Recovery>,
    placement: Option<placement::Placement>,
    save_conversations: bool,
    recovery_dirty: bool,
    recovery_flush_now: bool,
    recovery_erase_pending: bool,
    recovery_notice: String,
    last_recovery_save: Instant,
    last_workspace_bytes: std::cell::Cell<usize>,
    restore_history: bool,
    status: String,
    status_detail: String,
    runtime_description: String,
    weight_packing: context_capacity::WeightPacking,
    warning: bool,
    alerts: error_ui::View,
    diagnostics: VecDeque<String>,
    diagnostics_revision: u64,
    show_diagnostics: bool,
    show_settings: bool,
    settings_view: settings::View,
    appearance: appearance::Preferences,
    appearance_store: Option<appearance::Store>,
    profile_store: Option<profile::Store>,
    sidebar_open: bool,
    sidebar_focus: bool,
    help: Option<(&'static str, &'static str)>,
    extra_effort: bool,
    enable_thinking: bool,
    output_budget: usize,
    daisyrose_scale: f32,
    daisyrose_enabled: bool,
    daisyrose_available: bool,
    daisyrose_default_scale_tenths: u8,
    daisyrose_max_rank: Option<u16>,
    daisyrose_sha256_prefix: Option<String>,
    daisyrose_customized: bool,
    cognitive: client::CognitiveOptions,
    cognitive_controls_available: bool,
    memory: crate::context_cards::MemoryStore,
    memory_revision: u64,
    memory_draft: settings::MemoryEditorDraft,
    focus_composer: bool,
    composing: bool,
    composer_id: Option<egui::Id>,
    composer_cursor: Option<egui::text::CCursorRange>,
    pending_cursor: Option<egui::text::CCursorRange>,
    speech: Option<speech::Session>,
    speech_checked: bool,
    speech_input: bool,
    audio_file_input: bool,
    audio_import: Option<crate::audio_import::Session>,
    attachment_import: Option<crate::attachment_import::Session>,
    attachment_notice: Option<(String, bool)>,
    audio_file_generation: u64,
    audio_file_active: bool,
    audio_file_notice: Option<String>,
    audio_file_progress: Option<speech::FileProgress>,
    audio_file_review_required: bool,
    audio_file_transcript: String,
    audio_file_error: bool,
    show_audio_transcript: bool,
    native_owner: Option<isize>,
    audio_paste: Option<crate::audio_import::PasteHook>,
    audio_paste_checked: bool,
    image_cache: image_cache::ImageCache,
    lightbox: Option<lightbox::Lightbox>,
    image_export: Option<crate::image_export::Export>,
    speech_output: bool,
    canary_configured: bool,
    tts_configured: bool,
    canary_managed: bool,
    tts_managed: bool,
    speech_status: String,
    // Only observed failures enter the shared error panel; progress stays local.
    speech_notice: bool,
    voice_profile: String,
    listening: bool,
    dictation_requested: bool,
    dictation_finishing: bool,
    pending_voice_transcript: Option<(u64, String)>,
    microphone: Option<microphone::Microphone>,
    microphone_generation: Option<u64>,
    speaking: bool,
    speech_loading: bool,
    speaking_message: Option<u64>,
    streaming_read_aloud: bool,
    streaming_active: bool,
    streaming_utf16: usize,
    streaming_suppressed: bool,
    streaming_svg_filter: svg_fence::StreamingSvgFilter,
    answer_complete: bool,
    last_answer: String,
    retained_draft: String,
    stage: Option<&'static str>,
    last_theme_check: Instant,
}

const BUFFER_HIGH_WATER_BYTES: usize = 64 * 1024;
const BUFFER_RETAINED_BYTES: usize = 4 * 1024;

fn clear_and_shrink_buffer(text: &mut String) {
    text.clear();
    shrink_excess_buffer(text);
}

fn shrink_excess_buffer(text: &mut String) {
    if text.capacity() > BUFFER_HIGH_WATER_BYTES && text.len() <= BUFFER_HIGH_WATER_BYTES / 2 {
        text.shrink_to(text.len().max(BUFFER_RETAINED_BYTES));
    }
}

impl State {
    fn create(
        commands: Sender<Command>,
        events: client::EventReceiver,
        config: Option<&Config>,
        microphone: Option<microphone::Microphone>,
    ) -> Self {
        let speech = config
            .and_then(|config| speech::start(config.canary.clone(), config.qwen_tts.clone()).ok());
        let mut state = Self::initial_state(commands, events, microphone, speech);
        if let Some(config) = config {
            state.model = ModelLifecycle::Preparing;
            state.preview = false;
            state.apply_model_config(config);
            state.apply_speech_flags(config);
        }
        state
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep the complete native state initialization and speech ownership flags in one transaction."
    )]
    fn initial_state(
        commands: Sender<Command>,
        events: client::EventReceiver,
        microphone: Option<microphone::Microphone>,
        speech: Option<speech::Session>,
    ) -> Self {
        Self {
            commands,
            events,
            drained_events: Vec::new(),
            repaint_ctx: None,
            worker: None,
            markdown_image: None,
            markdown_configured: std::cell::Cell::new(false),
            pending_config: None,
            startup_request: None,
            startup: None,
            setup: setup::Setup::default(),
            startup_failure: None,
            updates: None,
            update_exit: None,
            model: ModelLifecycle::Unavailable,
            awaiting_first_token: false,
            vision_available: false,
            preview: true,
            input: String::new(),
            input_revision: 0,
            draft_attachments: Vec::new(),
            draft_attachments_revision: 0,
            transcript: Transcript::default(),
            committed: Vec::new(),
            context_tokens: None,
            context_history: None,
            context_draft: context_capacity::DraftCache::default(),
            conversations: vec![Conversation::new(1)],
            conversation_search: history_search::Search::default(),
            active_conversation: 1,
            request_conversation: None,
            auto_continuing: false,
            routing_request: false,
            next_conversation: 2,
            rename_conversation: None,
            delete_conversation: None,
            edit_message: None,
            branch_comparison: None,
            branch_diff_generation: 0,
            conversation_notice: None,
            recovery: None,
            placement: None,
            save_conversations: false,
            recovery_dirty: false,
            recovery_flush_now: false,
            recovery_erase_pending: false,
            recovery_notice: String::new(),
            last_recovery_save: Instant::now(),
            last_workspace_bytes: std::cell::Cell::new(0),
            restore_history: false,
            status: "Preparing your local model…".to_owned(),
            status_detail: String::new(),
            runtime_description: "Model-free preview".to_owned(),
            weight_packing: context_capacity::WeightPacking::Pq2_0,
            warning: false,
            alerts: error_ui::View::default(),
            diagnostics: VecDeque::new(),
            diagnostics_revision: 0,
            show_diagnostics: false,
            show_settings: false,
            settings_view: settings::View::default(),
            appearance: appearance::Preferences::default(),
            appearance_store: None,
            profile_store: None,
            sidebar_open: false,
            sidebar_focus: false,
            help: None,
            extra_effort: false,
            enable_thinking: false,
            output_budget: 3,
            daisyrose_scale: client::DEFAULT_DAISYROSE_SCALE,
            daisyrose_enabled: false,
            daisyrose_available: false,
            daisyrose_default_scale_tenths: 0,
            daisyrose_max_rank: None,
            daisyrose_sha256_prefix: None,
            daisyrose_customized: false,
            cognitive: client::CognitiveOptions::default(),
            cognitive_controls_available: false,
            memory: crate::context_cards::MemoryStore::default(),
            memory_revision: 0,
            memory_draft: settings::MemoryEditorDraft::default(),
            focus_composer: false,
            composing: false,
            composer_id: None,
            composer_cursor: None,
            pending_cursor: None,
            speech_checked: speech.is_none(),
            speech,
            speech_input: false,
            audio_file_input: false,
            audio_import: None,
            attachment_import: None,
            attachment_notice: None,
            audio_file_generation: 0,
            audio_file_active: false,
            audio_file_notice: None,
            audio_file_progress: None,
            audio_file_review_required: false,
            audio_file_transcript: String::new(),
            audio_file_error: false,
            show_audio_transcript: false,
            native_owner: None,
            audio_paste: None,
            audio_paste_checked: false,
            image_cache: image_cache::ImageCache::default(),
            lightbox: None,
            image_export: None,
            speech_output: false,
            canary_configured: false,
            tts_configured: false,
            canary_managed: false,
            tts_managed: false,
            speech_status: "Select Microphone or a response's speaker button for setup.".to_owned(),
            speech_notice: false,
            voice_profile: DEFAULT_VOICE.to_owned(),
            listening: false,
            dictation_requested: false,
            dictation_finishing: false,
            pending_voice_transcript: None,
            microphone,
            microphone_generation: None,
            speaking: false,
            speech_loading: false,
            speaking_message: None,
            streaming_read_aloud: false,
            streaming_active: false,
            streaming_utf16: 0,
            streaming_suppressed: false,
            streaming_svg_filter: svg_fence::StreamingSvgFilter::new(),
            answer_complete: false,
            last_answer: String::new(),
            retained_draft: String::new(),
            stage: None,
            last_theme_check: initial_theme_check(),
        }
    }

    fn pending_startup(
        commands: Sender<Command>,
        events: client::EventReceiver,
        request: crate::startup::Request,
    ) -> Self {
        let mut state = Self::create(commands, events, None, None);
        // The launcher already prepared this image. Formatting recovered
        // answers must not wait for model provisioning or depend on its result.
        state.markdown_image.clone_from(&request.markdown_image);
        state.updates = updates::Updates::for_startup(&request);
        state.preview = false;
        state.canary_configured = request.canary_configured;
        state.tts_configured = request.tts_configured;
        state.speech_checked = !state.canary_configured && !state.tts_configured;
        if !state.speech_checked {
            "Preparing speech services…".clone_into(&mut state.speech_status);
        }
        "Preparing runtime selection…".clone_into(&mut state.runtime_description);
        state.model = ModelLifecycle::Preparing;
        state.daisyrose_available = false;
        state.daisyrose_max_rank = None;
        state.daisyrose_sha256_prefix = None;
        state.record(&format!(
            "Swift Bonsai {}. Data directory: {}",
            env!("CARGO_PKG_VERSION"),
            request.data.display()
        ));
        state.setup.request = Some(request.clone());
        state.startup_request = Some(request);
        state.set_status(
            "Preparing your local model… You can draft a message now.",
            false,
        );
        state
    }

    fn configure_markdown_once(&self, ctx: &egui::Context, image: &std::path::Path) {
        if !self.markdown_configured.get() {
            crate::markdown_service::Service::configure(ctx, image);
            self.markdown_configured.set(true);
        }
    }

    fn markdown_preparing(&self, ctx: &egui::Context) -> bool {
        self.markdown_image.as_ref().is_some_and(|image| {
            self.configure_markdown_once(ctx, image);
            crate::markdown_service::Service::get(ctx).is_some_and(|service| {
                service.warm_up()
                    || (!service.unavailable() && self.transcript.presentation_pending(ctx))
            })
        })
    }

    fn poll_startup(&mut self, ctx: &egui::Context) {
        if (self.startup_request.is_some() || self.pending_config.is_some())
            && self.markdown_preparing(ctx)
        {
            // Present the visible restored answers before model preparation
            // competes for CPU and disk. A handshake alone has no styled text.
            ctx.request_repaint_after(Duration::from_millis(16));
            return;
        }
        if let Some(image) = self.markdown_image.as_deref() {
            self.configure_markdown_once(ctx, image);
        }
        if let Some(config) = self.pending_config.take()
            && let Err(error) = self.attach_configuration(config, false)
        {
            self.fail_startup(&error);
        }
        if let Some(request) = self.startup_request.take() {
            match crate::startup::Session::start(request) {
                Ok(session) => self.startup = Some(session),
                Err(error) => self.fail_startup(&error),
            }
        }
        if self.worker.is_none() {
            self.setup.progress = self
                .startup
                .as_ref()
                .and_then(crate::startup::Session::progress);
        }
        let Some(result) = self
            .startup
            .as_ref()
            .and_then(crate::startup::Session::poll)
        else {
            return;
        };
        match result {
            Ok(crate::startup::Progress::Model {
                config,
                speech_pending,
            }) => {
                if self.setup.pausing {
                    if !speech_pending {
                        self.startup.take();
                        self.fail_startup(&io::Error::new(
                            io::ErrorKind::Interrupted,
                            "Setup paused.",
                        ));
                    }
                    return;
                }
                self.setup.broker = Some(config.broker.clone());
                self.setup.error = None;
                self.alerts.clear_dismissed(error_ui::AlertKind::Setup);
                self.setup.progress = None;
                if !speech_pending {
                    self.startup.take();
                }
                if let Err(error) = self.attach_configuration(config, speech_pending) {
                    self.startup.take();
                    self.fail_startup(&error);
                }
            }
            Ok(crate::startup::Progress::Speech(result)) => {
                self.startup.take();
                if self.setup.pausing && self.worker.is_none() {
                    self.fail_startup(&result.err().unwrap_or_else(|| {
                        io::Error::new(io::ErrorKind::Interrupted, "Setup paused.")
                    }));
                    return;
                }
                match result {
                    Ok(config) => self.attach_speech_configuration(&config),
                    Err(error) => self.fail_speech_preparation(&error),
                }
            }
            Err(error) => {
                self.startup.take();
                if self.worker.is_some() {
                    self.fail_speech_preparation(&error);
                } else {
                    self.fail_startup(&error);
                }
            }
        }
    }

    fn fail_speech_preparation(&mut self, error: &io::Error) {
        self.speech_checked = true;
        self.setup.speech_error = Some(error.to_string());
        self.alerts
            .clear_dismissed(error_ui::AlertKind::SpeechSetup);
        "Speech setup could not finish. Text chat is still available. Select Retry speech setup."
            .clone_into(&mut self.speech_status);
        Self::record_into(
            &mut self.diagnostics,
            &mut self.diagnostics_revision,
            &self.speech_status,
        );
        self.record(&error.to_string());
    }

    fn fail_startup(&mut self, error: &io::Error) {
        self.speech_checked = true;
        self.setup.error = Some(error.to_string());
        self.alerts.clear_dismissed(error_ui::AlertKind::Setup);
        self.setup.progress = None;
        self.setup.pausing = false;
        if let Some(failure) = &self.startup_failure
            && let Ok(mut failure) = failure.lock()
        {
            *failure = (error.kind() != io::ErrorKind::Interrupted).then(|| error.to_string());
        }
        self.fail(setup::failure_message(error));
        self.record(&error.to_string());
    }

    fn attach_configuration(&mut self, config: Config, speech_pending: bool) -> io::Result<()> {
        self.attach_configuration_with(config, speech_pending, |config, incoming, events| {
            std::thread::Builder::new()
                .name("swift-chat-client".to_owned())
                .spawn(move || client::run(config, incoming, events))
        })
    }

    fn attach_configuration_with(
        &mut self,
        config: Config,
        speech_pending: bool,
        start: impl FnOnce(Config, Receiver<Command>, client::EventSender) -> io::Result<JoinHandle<()>>,
    ) -> io::Result<()> {
        let (commands, incoming) = mpsc::channel();
        let (events, incoming_events) = match &self.repaint_ctx {
            Some(ctx) => {
                let repaint_ctx = ctx.clone();
                client::event_channel_with_repaint(std::sync::Arc::new(move || {
                    repaint_ctx.request_repaint();
                }))
            }
            None => client::event_channel(),
        };
        self.apply_model_config(&config);
        self.daisyrose_default_scale_tenths = 0;
        self.daisyrose_max_rank = None;
        self.daisyrose_sha256_prefix = None;
        if !speech_pending {
            self.attach_speech_configuration(&config);
        }
        self.worker = Some(start(config, incoming, events)?);
        self.commands = commands;
        self.events = incoming_events;
        self.auto_continuing = false;
        self.set_status("Preparing your local model…", false);
        Ok(())
    }

    fn apply_model_config(&mut self, config: &Config) {
        self.markdown_image = Some(config.broker.clone());
        self.runtime_description = runtime_description(config);
        self.weight_packing = match config.bonsai25_variant {
            Some(client::Bonsai25Variant::Pq2) => context_capacity::WeightPacking::Bonsai25Pq2,
            Some(client::Bonsai25Variant::Ptq1) => context_capacity::WeightPacking::Bonsai25Ptq1,
            None => context_capacity::WeightPacking::Pq2_0,
        };
        self.daisyrose_available = config.daisyrose_lora.is_some();
        self.daisyrose_max_rank = None;
        self.daisyrose_sha256_prefix = None;
    }

    fn apply_speech_flags(&mut self, config: &Config) {
        self.canary_configured = config.canary.is_some();
        self.tts_configured = config.qwen_tts.is_some();
        self.canary_managed = config
            .canary
            .as_ref()
            .is_some_and(|speech| speech.launch.is_some());
        self.tts_managed = config
            .qwen_tts
            .as_ref()
            .is_some_and(|speech| speech.launch.is_some());
    }

    fn attach_speech_configuration(&mut self, config: &Config) {
        self.setup.speech_error = None;
        self.alerts
            .clear_dismissed(error_ui::AlertKind::SpeechSetup);
        self.speech_notice = false;
        self.apply_speech_flags(config);
        self.speech = speech::start(config.canary.clone(), config.qwen_tts.clone()).ok();
        self.speech_checked = self.speech.is_none();
        if self.speech_checked {
            self.offer_speech_repair(false, false);
        }
        if self.speech_checked && self.speaking {
            self.speaking = false;
            self.speech_loading = false;
            self.speaking_message = None;
        }
        self.microphone = microphone::Microphone::new().ok();
    }
    fn offer_speech_repair(&mut self, input: bool, output: bool) {
        if ((self.canary_managed && !input) || (self.tts_managed && !output))
            && self.startup.is_none()
            && self.startup_request.is_none()
            && self.setup.request.is_some()
            && self.setup.broker.is_some()
            && self.setup.speech_error.is_none()
        {
            self.setup.speech_error = Some("Swift could not finish preparing speech. Select Retry speech setup to check and repair its app-managed files. Text chat and your draft remain available.".into());
            self.alerts
                .clear_dismissed(error_ui::AlertKind::SpeechSetup);
            self.speech_notice = true;
        }
    }

    fn set_status(&mut self, message: &str, warning: bool) {
        self.record(message);
        message.clone_into(&mut self.status_detail);
        self.status = message
            .split("Protected credentials and technical evidence:")
            .next()
            .unwrap_or(message)
            .trim()
            .chars()
            .take(MAX_DIAGNOSTIC_CHARS)
            .collect();
        self.warning = warning;
        if warning {
            self.alerts.clear_dismissed(error_ui::AlertKind::Model);
        }
    }

    fn record_into(diagnostics: &mut VecDeque<String>, revision: &mut u64, message: &str) {
        if diagnostics
            .back()
            .is_none_or(|previous| previous != message)
        {
            diagnostics.push_back(message.chars().take(MAX_DIAGNOSTIC_CHARS).collect());
            diagnostics.retain_back(MAX_DIAGNOSTICS);
            *revision = revision.wrapping_add(1);
        }
    }

    fn record(&mut self, message: &str) {
        Self::record_into(
            &mut self.diagnostics,
            &mut self.diagnostics_revision,
            message,
        );
    }

    pub(super) fn record_speech_status(&mut self) {
        Self::record_into(
            &mut self.diagnostics,
            &mut self.diagnostics_revision,
            &self.speech_status,
        );
    }

    fn reset_model_capabilities(&mut self) {
        self.context_tokens = None;
        self.vision_available = false;
        self.daisyrose_available = false;
        self.daisyrose_default_scale_tenths = 0;
        self.daisyrose_max_rank = None;
        self.daisyrose_sha256_prefix = None;
    }

    pub(super) fn set_draft_attachments(
        &mut self,
        attachments: Vec<crate::attachment_import::Attachment>,
    ) {
        self.draft_attachments = attachments;
        self.draft_attachments_revision = self.draft_attachments_revision.wrapping_add(1);
    }

    pub(super) fn take_draft_attachments(&mut self) -> Vec<crate::attachment_import::Attachment> {
        self.draft_attachments_revision = self.draft_attachments_revision.wrapping_add(1);
        std::mem::take(&mut self.draft_attachments)
    }

    pub(super) fn replace_draft_attachments(
        &mut self,
        attachments: Vec<crate::attachment_import::Attachment>,
    ) -> Vec<crate::attachment_import::Attachment> {
        self.draft_attachments_revision = self.draft_attachments_revision.wrapping_add(1);
        std::mem::replace(&mut self.draft_attachments, attachments)
    }

    pub(super) fn history_has_media_up_to(&self, prefix: usize) -> bool {
        self.committed
            .iter()
            .take(prefix)
            .any(|message| !message.images.is_empty())
    }

    pub(super) fn has_unsent_or_history_media(&self) -> bool {
        self.draft_attachments
            .iter()
            .any(crate::attachment_import::Attachment::is_media)
            || self.history_has_media_up_to(self.committed.len())
    }

    fn submit(&mut self) {
        self.submit_turn(None);
    }

    fn submit_turn(&mut self, retry: Option<u64>) {
        if self.preview {
            self.set_status(
                "Preview only. Launch the app normally to chat with your local model.",
                false,
            );
            return;
        }
        if self.model != ModelLifecycle::Ready
            || (self.input.trim().is_empty() && self.draft_attachments.is_empty())
        {
            return;
        }
        if !self.vision_available && self.has_unsent_or_history_media() {
            self.set_status(
                "Images and videos require the vision projector (mmproj). Remove media attachments or relaunch without --no-mmproj.",
                true,
            );
            return;
        }
        let text = self.input.trim().to_owned();
        if self.dictation_requested {
            self.stop_speech();
        } else {
            self.stop_speech_output();
        }
        clear_and_shrink_buffer(&mut self.last_answer);
        clear_and_shrink_buffer(&mut self.retained_draft);
        self.attachment_notice = None;
        self.stage = None;
        self.answer_complete = false;
        self.streaming_svg_filter.reset();
        let max_completion_tokens = OUTPUT_TOKENS
            .get(self.output_budget)
            .copied()
            .unwrap_or(4096);
        let (wire_text, images) =
            crate::attachment_import::format_user_turn(&text, &self.draft_attachments);
        let memory = self.memory.resolve(&self.committed, &wire_text);
        let system_appendix = memory.pinned_appendix.map(std::sync::Arc::<str>::from);
        let turn_appendix = memory.triggered_appendix.map(std::sync::Arc::<str>::from);
        self.daisyrose_customized = true;
        if self
            .commands
            .send(Command::Send {
                text: wire_text,
                images,
                max_completion_tokens,
                extra_effort: self.extra_effort,
                enable_thinking: self.enable_thinking,
                daisyrose_scale: self.effective_daisyrose_scale(),
                cognitive: self.cognitive,
                system_appendix,
                turn_appendix,
            })
            .is_err()
        {
            self.fail("The model worker stopped. Close and restart the app.");
            return;
        }
        if let Some(conversation) = self
            .conversations
            .iter_mut()
            .find(|conversation| conversation.id == self.active_conversation)
            && !conversation.renamed
            && self.transcript.messages.is_empty()
        {
            let title_source = if text.is_empty() {
                self.draft_attachments
                    .first()
                    .map_or("Attachment", crate::attachment_import::Attachment::name)
            } else {
                text.as_str()
            };
            conversation.title = title_source
                .split_whitespace()
                .flat_map(|word| std::iter::once(' ').chain(word.chars()))
                .skip(1)
                .take(48)
                .collect();
        }
        if let Some(id) = retry {
            self.transcript.restart_turn(id, self.committed.len());
        } else {
            let attachments = self.take_draft_attachments();
            self.transcript
                .start_turn_with_attachments(text, attachments, self.committed.len());
        }
        self.request_conversation = Some(self.active_conversation);
        self.auto_continuing = false;
        self.touch_conversation(self.active_conversation);
        self.input.clear();
        self.input_revision = self.input_revision.wrapping_add(1);
        self.place_composer_cursor(0);
        self.model = ModelLifecycle::Generating;
        self.awaiting_first_token = true;
        // A turn begun without speech stays silent even if preparation finishes
        // mid-answer. Only an explicit read or a later turn may start playback.
        self.streaming_suppressed = !self.speech_output;
        self.set_status("Responding…", false);
    }

    fn new_chat(&mut self) {
        if !self.model.workspace_available() {
            return;
        }
        let Some(next) = self.next_conversation.checked_add(1) else {
            self.conversation_notice = Some(ConversationNotice::persistent(
                "Cannot create another conversation in this workspace.",
            ));
            return;
        };
        self.update_worker_conversation(Command::NewChat);
        self.sidebar_open = false;
        self.stop_speech();
        self.save_conversation();
        self.conversation_notice = None;
        self.clear_conversation_search();
        let id = self.next_conversation;
        self.next_conversation = next;
        self.conversations.insert(0, Conversation::new(id));
        self.active_conversation = id;
        self.clear_conversation();
        self.recovery_dirty = true;
        if self.model == ModelLifecycle::Ready {
            self.set_status("Ready for a new conversation.", false);
        }
    }

    fn retry(&mut self) {
        if !self.model.reloadable()
            || self
                .updates
                .as_ref()
                .is_some_and(updates::Updates::installing)
        {
            return;
        }
        if self.commands.send(Command::Retry).is_err() {
            self.fail("The model worker stopped. Close and restart the app.");
            return;
        }
        self.stop_speech();
        self.reset_model_capabilities();
        self.model = ModelLifecycle::Restarting;
        self.set_status("Restarting the local model…", false);
    }

    fn stop_generation(&mut self) {
        if !self.model.busy() {
            return;
        }
        if self.commands.send(Command::Stop).is_err() {
            self.fail("The model worker stopped. Close and restart the app.");
            return;
        }
        self.stop_speech();
        self.with_request_conversation(|state| {
            state.model = ModelLifecycle::Stopping;
            state.transcript.remove_empty_response();
            state.set_status("Stopping the response…", false);
            state.touch_conversation(state.active_or_routed_conversation());
        });
    }

    fn handle_escape(&mut self, ctx: &egui::Context) {
        if self.update_confirmation_visible() {
            return;
        }
        // The transcript viewer and lightbox own Escape and background shortcuts,
        // even on their first frame. Closing them must not cancel or stop a request.
        if self.show_audio_transcript || self.alerts.details.is_some() {
            return;
        }
        if self.lightbox.is_some() {
            if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
                self.lightbox = None;
                ctx.request_repaint();
            }
            return;
        }
        // Voice cancellation takes effect before overlays consume Escape. Keep
        // the key available so the same action also dismisses Find or a modal.
        if self.dictation_requested && plain_key_pressed(ctx, egui::Key::Escape) {
            if self.audio_file_active {
                self.cancel_audio_file();
            } else {
                self.cancel_microphone();
                "Voice message cancelled.".clone_into(&mut self.speech_status);
            }
        }
        if self.attachment_import.is_some() && plain_key_pressed(ctx, egui::Key::Escape) {
            self.cancel_attachment_import();
            return;
        }
        if self.has_conversation_dialog()
            || self.branch_comparison.is_some()
            || self.show_settings
            || egui::Popup::is_any_open(ctx)
        {
            return;
        }
        if self.conversation_search.clear_on_escape(ctx) {
            self.sync_conversation_find();
            return;
        }
        if self.sidebar_open && ctx.content_rect().width() < navigation::DOCK_THRESHOLD {
            return;
        }
        if self.transcript.close_find_on_escape(ctx) {
            self.focus_composer = true;
        } else if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
        {
            if self.active_conversation_busy() {
                self.stop_generation();
            } else if self.speaking || self.dictation_requested {
                self.stop_speech();
            }
        }
    }

    pub(super) const fn has_conversation_dialog(&self) -> bool {
        self.rename_conversation.is_some()
            || self.delete_conversation.is_some()
            || self.edit_message.is_some()
    }

    const fn has_overlay(&self) -> bool {
        self.show_settings
            || self.show_audio_transcript
            || self.lightbox.is_some()
            || self.alerts.details.is_some()
            || self.branch_comparison.is_some()
    }

    fn has_dialog_or_popup(&self, ctx: &egui::Context) -> bool {
        self.has_overlay() || self.has_conversation_dialog() || egui::Popup::is_any_open(ctx)
    }

    fn handle_input_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.input(|input| input.viewport().close_requested()) {
            if self.dictation_requested {
                self.cancel_microphone();
            }
            self.cancel_attachment_import();
            self.cancel_branch_comparison();
            self.image_export = None;
        }
        ctx.input(|input| {
            for event in &input.events {
                match event {
                    egui::Event::Ime(egui::ImeEvent::Preedit { text, .. }) => {
                        self.composing = !text.is_empty();
                    }
                    egui::Event::Ime(egui::ImeEvent::Commit(_)) => self.composing = false,
                    _ => {}
                }
            }
        });
        if self.update_confirmation_visible() {
            return;
        }
        if !self.has_dialog_or_popup(ctx) {
            if ctx.input_mut(|input| input.consume_key(egui::Modifiers::CTRL, egui::Key::N)) {
                self.new_chat();
            }
            // Run before the sidebar and composer text editors consume Ctrl+F.
            if ctx.input_mut(|input| input.consume_key(egui::Modifiers::CTRL, egui::Key::F)) {
                self.sidebar_open = false;
                self.sidebar_focus = false;
                self.focus_composer = false;
                self.transcript.focus_find();
                ctx.request_repaint();
            }
            if ctx.input_mut(|input| {
                input.consume_key(egui::Modifiers::CTRL | egui::Modifiers::SHIFT, egui::Key::B)
            }) {
                self.toggle_navigation(ctx);
            }
        }
        self.handle_escape(ctx);
    }

    fn retain_draft(&mut self) {
        if !self.retained_draft.is_empty() {
            self.last_answer.clone_from(&self.retained_draft);
            self.answer_complete = true;
        }
    }

    fn restore_committed_last_answer(&mut self) {
        if let Some(answer) = self
            .committed
            .last()
            .filter(|entry| entry.role == "assistant")
        {
            answer.content.as_ref().clone_into(&mut self.last_answer);
            shrink_excess_buffer(&mut self.last_answer);
            self.answer_complete = !self.transcript.latest_interrupted(&self.committed);
        }
    }

    fn finish_streaming_speech(&mut self) {
        if self.streaming_active {
            if let Some(tail) = self.streaming_svg_filter.finish()
                && !tail.is_empty()
            {
                let units = tail.encode_utf16().take(speech::MAX_TEXT + 1).count();
                if units <= speech::MAX_TEXT.saturating_sub(self.streaming_utf16) {
                    self.streaming_utf16 += units;
                    self.speech_command(speech::Command::AppendStream { text: tail });
                }
            }
            self.speech_command(speech::Command::FinishStream);
        } else {
            self.streaming_svg_filter.reset();
        }
        self.streaming_utf16 = 0;
    }

    fn fail(&mut self, message: &str) {
        if matches!(
            self.model,
            ModelLifecycle::Generating
                | ModelLifecycle::Stopping
                | ModelLifecycle::StoppingModel
                | ModelLifecycle::AwaitingTeardown
        ) {
            self.with_request_conversation(|state| state.fail_current_conversation(message));
        } else {
            self.fail_current_conversation(message);
        }
    }

    fn fail_current_conversation(&mut self, message: &str) {
        let auto_continuing = std::mem::replace(&mut self.auto_continuing, false);
        if !self.routing_request {
            self.stop_speech();
        }
        let mutating_turn = matches!(
            self.model,
            ModelLifecycle::Generating
                | ModelLifecycle::Stopping
                | ModelLifecycle::StoppingModel
                | ModelLifecycle::AwaitingTeardown
        );
        let adopted = mutating_turn && self.adopt_interrupted();
        if mutating_turn && !adopted && auto_continuing {
            self.transcript.discard_turn(self.committed.len());
            self.restore_committed_last_answer();
        }
        self.context_tokens = None;
        self.vision_available = false;
        self.awaiting_first_token = false;
        self.model = ModelLifecycle::Unavailable;
        self.transcript.remove_empty_response();
        if !adopted && !auto_continuing && !self.transcript.latest_interrupted(&self.committed) {
            self.retain_draft();
        }
        if mutating_turn {
            self.touch_conversation(self.active_or_routed_conversation());
        }
        self.set_status(message, true);
    }

    fn reject(&mut self, message: &str, text: &str, retry_required: bool) {
        let auto_continuing = std::mem::replace(&mut self.auto_continuing, false);
        if !self.routing_request
            && let Some(id) = self.request_conversation
            && id != self.active_conversation
        {
            if !auto_continuing {
                // A worker can reject a queued Send after announcing teardown.
                // Restore its prompt only to the conversation that sent it.
                let mut restored = false;
                if let Some(conversation) = self
                    .conversations
                    .iter_mut()
                    .find(|conversation| conversation.id == id)
                {
                    if let Some(saved) = conversation.saved.as_mut() {
                        let mut input_revision = 0_u64;
                        let mut attachment_revision = 0_u64;
                        restored = restore_rejected_draft_for(
                            &mut saved.input,
                            &mut input_revision,
                            &mut saved.draft_attachments,
                            &mut attachment_revision,
                            &saved.transcript,
                            saved.committed.len(),
                            text,
                        );
                    }
                    if restored {
                        conversation.invalidate_json_cache();
                    }
                }
                if restored {
                    self.touch_conversation(id);
                }
            }
            return;
        }
        let adopted = retry_required && self.adopt_interrupted();
        if auto_continuing && !adopted {
            self.transcript.discard_turn(self.committed.len());
            self.restore_committed_last_answer();
        } else if !retry_required {
            self.transcript.interrupt_turn(self.committed.len());
        }
        self.touch_conversation(self.active_or_routed_conversation());
        if self.model.stopping() || self.model == ModelLifecycle::Restarting {
            // A rejection can queue just before Stop or behind the first
            // teardown notice. Preserve explicit Stop/retry ownership until
            // teardown ends or the worker acknowledges the new attempt.
            if self.model.stopping() && retry_required {
                self.model = ModelLifecycle::AwaitingTeardown;
            }
            if !adopted && !auto_continuing {
                self.restore_rejected_draft_into_active(text);
            }
            return;
        }
        if !self.routing_request {
            if auto_continuing && !retry_required && self.stage.is_none() {
                if self.streaming_active {
                    self.speaking_message = self.transcript.messages.last().map(|entry| entry.id);
                }
                self.finish_streaming_speech();
            } else {
                self.stop_speech();
            }
        }
        self.model = if retry_required {
            if self.model.retryable() {
                ModelLifecycle::Retryable
            } else {
                ModelLifecycle::AwaitingTeardown
            }
        } else {
            ModelLifecycle::Ready
        };
        self.transcript.remove_empty_response();
        if !adopted && !auto_continuing {
            self.restore_rejected_draft_into_active(text);
        }
        if auto_continuing && !retry_required {
            self.set_status(
                "Response reached the selected output cap, including thinking. You can ask Swift to continue.",
                false,
            );
            self.record(message);
        } else {
            self.set_status(
                if adopted {
                    "Response interrupted. The partial reply is included in this conversation; retry the model to continue."
                } else {
                    message
                },
                true,
            );
            message.clone_into(&mut self.status_detail);
        }
        self.focus_composer = !retry_required && !auto_continuing;
    }

    fn drain_events(&mut self) {
        if self.worker.is_none()
            && (self.startup_request.is_some()
                || self.pending_config.is_some()
                || self.startup.is_some())
        {
            return;
        }
        // Bound per-frame work even when the worker refills the bounded queue.
        let mut batch = std::mem::take(&mut self.drained_events);
        let outcome = self.events.drain_into(&mut batch, 64);
        for event in batch.drain(..) {
            self.model_event(event);
        }
        self.drained_events = batch;
        match outcome {
            Ok((true, _)) | Err(mpsc::TryRecvError::Disconnected) => {
                if !self.preview
                    && (self.model.ready() || self.model.busy() || self.model.waiting_model())
                {
                    self.fail("The model worker stopped. Close and restart the app.");
                }
            }
            Ok((false, has_more)) => {
                if has_more && let Some(ctx) = &self.repaint_ctx {
                    ctx.request_repaint();
                }
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    fn model_event(&mut self, event: Event) {
        if !self.model.accepts(&event) {
            return;
        }
        if matches!(
            self.model,
            ModelLifecycle::Generating
                | ModelLifecycle::Stopping
                | ModelLifecycle::StoppingModel
                | ModelLifecycle::AwaitingTeardown
        ) || matches!(event, Event::Rejected { .. })
        {
            self.with_request_conversation(|state| state.apply_model_event(event));
        } else {
            self.apply_model_event(event);
        }
        self.restore_selected_history();
    }

    fn apply_model_event(&mut self, event: Event) {
        self.recovery_dirty = true;
        match event {
            Event::Starting => {
                self.auto_continuing = false;
                self.awaiting_first_token = false;
                self.reset_model_capabilities();
                self.model = ModelLifecycle::Preparing;
                self.set_status("Preparing your local model…", false);
            }
            Event::Restarting => {
                self.auto_continuing = false;
                self.awaiting_first_token = false;
                self.reset_model_capabilities();
                self.model = ModelLifecycle::Restarting;
                self.set_status("Restarting the local model…", false);
            }
            Event::Retryable(message) => {
                self.awaiting_first_token = false;
                self.fail(&message);
                self.model = ModelLifecycle::Retryable;
                self.recovery_flush_now = true;
            }
            Event::Stopping(text) => {
                self.awaiting_first_token = false;
                self.stopping_response(&text);
                self.touch_conversation(self.active_or_routed_conversation());
            }
            Event::Stopped => {
                self.awaiting_first_token = false;
                self.stopped_response();
                self.recovery_flush_now = true;
            }
            Event::ModelStopped => {
                self.awaiting_first_token = false;
                self.stopped_model();
                self.recovery_flush_now = true;
            }
            Event::ContextCapacity(tokens) => self.context_tokens = Some(tokens),
            Event::VisionAvailable(available) => self.vision_available = available,
            Event::DaisyRoseAvailable(readiness) => {
                self.daisyrose_available = readiness.available;
                self.daisyrose_default_scale_tenths = if readiness.available {
                    readiness.default_scale_tenths
                } else {
                    0
                };
                self.daisyrose_max_rank = if readiness.available {
                    readiness.max_rank.filter(|rank| (1..=64).contains(rank))
                } else {
                    None
                };
                self.daisyrose_sha256_prefix = if readiness.available {
                    readiness
                        .sha256_prefix
                        .as_deref()
                        .and_then(client::valid_daisyrose_sha256_prefix)
                } else {
                    None
                };
                if readiness.default_enabled()
                    && !self.daisyrose_customized
                    && self.committed.is_empty()
                    && self.transcript.messages.is_empty()
                    && !self.daisyrose_enabled
                    && (self.daisyrose_scale - client::DEFAULT_DAISYROSE_SCALE).abs() < f32::EPSILON
                {
                    self.daisyrose_enabled = true;
                    self.daisyrose_scale = readiness.default_scale();
                }
            }
            Event::CognitiveControlsAvailable(available) => {
                self.cognitive_controls_available = available;
            }
            Event::Ready => {
                self.auto_continuing = false;
                self.awaiting_first_token = false;
                self.model = ModelLifecycle::Ready;
                self.set_status("Ready when you are.", false);
            }
            Event::Rejected {
                message,
                text,
                retry_required,
            } => {
                self.awaiting_first_token = false;
                self.reject(&message, &text, retry_required);
                self.recovery_flush_now = true;
            }
            Event::Stage(label) => {
                self.replace_streaming_stage();
                self.stage = Some(label);
                clear_and_shrink_buffer(&mut self.last_answer);
                self.answer_complete = false;
                self.awaiting_first_token = true;
                self.transcript.stage();
                self.set_status(label, false);
                self.touch_conversation(self.active_or_routed_conversation());
            }
            Event::Draft(text) => {
                self.retained_draft = text;
                self.touch_conversation(self.active_or_routed_conversation());
            }
            Event::Text(text) | Event::RetainedAnswer(text) => {
                if !text.is_empty() {
                    self.awaiting_first_token = false;
                }
                self.last_answer.push_str(&text);
                self.transcript.append(&text, false);
                self.stream_answer(text);
                self.stage
                    .unwrap_or("Writing…")
                    .clone_into(&mut self.status);
                self.touch_conversation(self.active_or_routed_conversation());
            }
            Event::Reasoning(text) => {
                if !text.is_empty() {
                    self.awaiting_first_token = false;
                }
                self.transcript.append(&text, true);
                self.touch_conversation(self.active_or_routed_conversation());
            }
            Event::Committed(history) => {
                self.commit_history(history);
                self.touch_conversation(self.active_or_routed_conversation());
            }
            Event::Complete(completion) => {
                self.auto_continuing = false;
                self.awaiting_first_token = false;
                self.transcript.remove_empty_response();
                self.model = ModelLifecycle::Ready;
                self.answer_complete = true;
                self.recovery_flush_now = true;
                match completion {
                    Completion::Finished => {
                        self.finish_streaming_speech();
                        self.set_status("Ready when you are.", false);
                        self.touch_conversation(self.active_or_routed_conversation());
                    }
                    Completion::TokenLimit
                        if !self.retained_draft.is_empty()
                            && self.last_answer == self.retained_draft =>
                    {
                        self.finish_streaming_speech();
                        self.set_status(
                            "Review did not finish. Your complete draft is retained.",
                            false,
                        );
                        self.touch_conversation(self.active_or_routed_conversation());
                    }
                    Completion::TokenLimit => {
                        self.transcript.mark_token_limit(&self.committed);
                        if !self.try_auto_continue() {
                            self.finish_streaming_speech();
                            self.set_status(
                                "Response reached the selected output cap, including thinking. You can ask Swift to continue.",
                                false,
                            );
                            self.touch_conversation(self.active_or_routed_conversation());
                        }
                    }
                }
            }
            Event::Failed(message) => {
                self.awaiting_first_token = false;
                self.fail(&message);
                self.recovery_flush_now = true;
            }
        }
    }

    fn commit_history(&mut self, history: Vec<client::Message>) {
        if self.model.stopping()
            && let Some(answer) = history.last().filter(|answer| answer.role == "assistant")
        {
            // A fully completed turn can win just before the stop click. Keep
            // its visible answer consistent with the authoritative history,
            // without reopening Send or overwriting the cleanup status.
            self.transcript
                .replace_response(&answer.content, answer.reasoning_content.as_deref());
            answer.content.as_ref().clone_into(&mut self.last_answer);
            shrink_excess_buffer(&mut self.last_answer);
            self.answer_complete = true;
        }
        self.transcript.bind_committed(&self.committed, &history);
        self.committed = history;
        self.context_history = None;
    }

    fn stopping_response(&mut self, text: &str) {
        let auto_continuing = std::mem::replace(&mut self.auto_continuing, false);
        if !self.routing_request {
            self.stop_speech();
        }
        let stopping_model = self.model == ModelLifecycle::StoppingModel;
        self.model = if stopping_model {
            ModelLifecycle::StoppingModel
        } else {
            ModelLifecycle::Stopping
        };
        let adopted = self.adopt_interrupted();
        if !adopted {
            if auto_continuing {
                self.transcript.discard_turn(self.committed.len());
                self.restore_committed_last_answer();
            } else {
                self.restore_rejected_draft_into_active(text);
            }
        }
        self.transcript.remove_empty_response();
        self.set_status(
            if stopping_model {
                "Response stopped. Releasing the local model…"
            } else {
                "Stopping the response…"
            },
            false,
        );
    }

    fn stopped_response(&mut self) {
        self.auto_continuing = false;
        self.model = ModelLifecycle::Ready;
        self.set_status(client::STOPPED_MESSAGE, false);
        self.focus_composer = true;
    }

    fn stopped_model(&mut self) {
        self.auto_continuing = false;
        self.reset_model_capabilities();
        self.model = ModelLifecycle::Stopped;
        self.set_status("Local model stopped.", false);
        self.focus_composer = true;
    }

    fn adopt_interrupted(&mut self) -> bool {
        if self.transcript.adopt_interrupted(&mut self.committed) {
            if let Some(answer) = self.committed.last() {
                self.last_answer = answer.content.to_string();
                self.answer_complete = false;
            }
            self.restore_history = true;
            self.context_history = None;
            self.recovery_dirty = true;
            true
        } else {
            self.transcript.interrupt_turn(self.committed.len());
            false
        }
    }

    fn restore_rejected_draft_into_active(&mut self, wire_text: &str) {
        if restore_rejected_draft_for(
            &mut self.input,
            &mut self.input_revision,
            &mut self.draft_attachments,
            &mut self.draft_attachments_revision,
            &self.transcript,
            self.committed.len(),
            wire_text,
        ) {
            self.place_composer_cursor(self.input.chars().count());
        }
        self.retain_draft();
    }

    fn clear_speech_session(&mut self, file_error: &'static str, status: &'static str) {
        self.speech_input = false;
        self.audio_file_input = false;
        self.speech_output = false;
        if self.audio_file_active {
            self.fail_audio_file(file_error);
        } else {
            self.cancel_microphone();
        }
        self.speaking = false;
        self.speech_loading = false;
        self.speaking_message = None;
        self.streaming_active = false;
        self.streaming_utf16 = 0;
        self.streaming_suppressed = true;
        status.clone_into(&mut self.speech_status);
        self.speech_notice = true;
    }

    fn speech_command(&mut self, command: speech::Command) -> bool {
        if self
            .speech
            .as_ref()
            .is_some_and(|session| session.commands.send(command).is_ok())
        {
            return true;
        }
        self.clear_speech_session(
            "The speech worker stopped. Restart the app and try the recording again.",
            "Speech worker stopped. Restart the app to retry.",
        );
        false
    }

    fn stop_speech(&mut self) {
        let was_active = self.speaking || self.dictation_requested || self.listening;
        if self.audio_file_active {
            self.cancel_audio_file();
        } else {
            self.cancel_microphone();
        }
        self.stop_speech_output();
        if was_active {
            "Speech stopped. Microphone off.".clone_into(&mut self.speech_status);
            self.speech_notice = false;
        }
    }

    fn stop_speech_output(&mut self) {
        if let Some(session) = &self.speech {
            let _result = session.commands.send(speech::Command::StopSpeaking);
        }
        self.speaking = false;
        self.speech_loading = false;
        self.speaking_message = None;
        self.streaming_active = false;
        self.streaming_utf16 = 0;
        self.streaming_suppressed = true;
        self.streaming_svg_filter.reset();
    }

    fn cancel_microphone(&mut self) {
        self.release_completed_input();
        if let Some(session) = &self.speech {
            let _result = session.commands.send(speech::Command::StopListening);
        }
    }

    // Successful input releases capture and UI ownership without discarding the
    // worker's short, monitored ASR lease. Explicit cancellation also sends StopListening.
    fn release_completed_input(&mut self) {
        if let Some(import) = &self.audio_import {
            import.cancel();
        }
        self.audio_file_active = false;
        self.audio_file_progress = None;
        if let Some(session) = &self.speech {
            session.set_input_generation(None);
        }
        if let Some(microphone) = &mut self.microphone {
            microphone.stop();
        }
        self.microphone_generation = None;
        self.dictation_requested = false;
        self.dictation_finishing = false;
        self.pending_voice_transcript = None;
        self.listening = false;
    }

    fn toggle_dictation(&mut self) {
        self.speech_notice = false;
        if self.audio_file_active {
            self.cancel_audio_file();
        } else if self.dictation_requested {
            self.finish_dictation();
        } else if !self.speech_input && (self.speech_checked || !self.canary_configured) {
            let message = self.microphone_help();
            message.clone_into(&mut self.speech_status);
            self.help = Some(("Microphone setup", message));
        } else if !self.speech_input {
            "Enable the microphone after speech preparation completes."
                .clone_into(&mut self.speech_status);
        } else if !self.model.busy() && !self.speaking {
            let Some(microphone) = &mut self.microphone else {
                return;
            };
            if let Ok(generation) = microphone.start() {
                if let Some(session) = &self.speech {
                    session.set_input_generation(Some(generation));
                }
                self.microphone_generation = Some(generation);
                self.dictation_requested = true;
                "Requesting microphone permission?".clone_into(&mut self.speech_status);
                self.focus_composer = true;
            } else {
                self.speech_notice = true;
                "Microphone permission is unavailable. Check Windows privacy settings."
                    .clone_into(&mut self.speech_status);
            }
        }
        self.record_speech_status();
    }

    fn finish_dictation(&mut self) {
        if self.dictation_finishing {
            self.cancel_microphone();
            "Voice message cancelled.".clone_into(&mut self.speech_status);
            return;
        }
        match self.microphone.as_mut().map(microphone::Microphone::finish) {
            Some(Ok(true)) => {
                self.dictation_finishing = true;
                "Finishing voice message… Escape cancels.".clone_into(&mut self.speech_status);
            }
            Some(Err(_)) => {
                self.speech_notice = true;
                self.cancel_microphone();
                "Could not finish microphone capture. Try again."
                    .clone_into(&mut self.speech_status);
            }
            Some(Ok(false)) | None => {
                self.cancel_microphone();
                "Microphone off. No voice message sent.".clone_into(&mut self.speech_status);
            }
        }
    }

    fn toggle_read_aloud(&mut self, message_id: u64) {
        if self.speaking && self.speaking_message == Some(message_id) {
            self.stop_speech();
            return;
        }
        let Some(text) = self.transcript.speech_text(message_id) else {
            return;
        };
        self.speech_notice = false;
        if self.speech_output {
            let voice_description = self.voice_profile.trim().to_owned();
            if voice_description.is_empty() {
                "Describe the voice in Settings before reading aloud."
                    .clone_into(&mut self.speech_status);
                self.open_settings(settings::Page::Voice);
                self.record_speech_status();
                return;
            }
            self.stop_speech();
            if self.speech_command(speech::Command::Speak {
                text,
                voice_description,
            }) {
                self.speaking = true;
                self.speech_loading = true;
                self.speaking_message = Some(message_id);
                "Reading the selected response aloud.".clone_into(&mut self.speech_status);
            }
        } else if !self.speech_checked && self.tts_configured {
            "Read aloud will be available when speech preparation completes."
                .clone_into(&mut self.speech_status);
        } else {
            let message = self.read_aloud_help();
            message.clone_into(&mut self.speech_status);
            self.open_voice_setup();
        }
        self.record_speech_status();
    }

    const fn microphone_help(&self) -> &'static str {
        if !self.canary_configured {
            "Speech input is disabled for this launch. Swift normally prepares speech automatically. Open Settings > Voice to review speech setup. Nothing is recording."
        } else if self.microphone.is_none() {
            "Windows microphone access is unavailable. Review Windows microphone privacy settings. Nothing is recording."
        } else {
            "Swift prepares speech input automatically. Wait for preparation to finish, or use Retry speech setup in Settings > Voice if a setup error is shown. Nothing is recording."
        }
    }

    const fn read_aloud_help(&self) -> &'static str {
        if self.tts_configured && !self.speech_checked {
            "Swift is preparing read aloud automatically. You can keep using text chat while it prepares."
        } else if self.tts_configured {
            "Swift manages read aloud automatically. If setup needs attention, use Retry speech setup in Settings > Voice. Once setup finishes, select a response's speaker button to try again. Your Voice profile chooses how it sounds."
        } else {
            "Read aloud is disabled for this launch. Swift normally prepares speech automatically with its default settings. Your conversations remain available."
        }
    }

    fn drain_speech_events(&mut self) {
        for _ in 0..32 {
            let Some(session) = &self.speech else { break };
            match session.events.try_recv() {
                Ok(event) => self.speech_event(event),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.speech_checked = true;
                    self.clear_speech_session(
                        "The speech worker disconnected. Restart the app and try the recording again.",
                        "Speech unavailable. Text chat remains available.",
                    );
                    break;
                }
            }
        }
    }

    fn speech_event(&mut self, event: speech::Event) {
        let previous = self.speech_status.clone();
        match event {
            speech::Event::Diagnostic(message) => self.record(&message),
            speech::Event::Capabilities { input, output } => {
                self.offer_speech_repair(input, output);
                self.speech_checked = true;
                self.speech_input = input && self.microphone.is_some();
                self.audio_file_input = input;
                self.speech_output = output;
                match (self.speech_input, output) {
                    (true, true) => "Microphone and read aloud are ready.",
                    (true, false) => "Microphone ready. Open Settings for read aloud setup.",
                    (false, true) => "Response speakers ready. Select Microphone for setup.",
                    (false, false) => {
                        "Speech is unavailable. Open Settings > Voice to review setup and retry."
                    }
                }
                .clone_into(&mut self.speech_status);
                if !output && self.speaking {
                    self.speaking = false;
                    self.speech_loading = false;
                    self.speaking_message = None;
                }
            }
            speech::Event::Listening(listening) => {
                self.listening = listening && self.dictation_requested;
                // StopListening also acknowledges file cleanup. It must not
                // overwrite file progress, success, cancellation or errors.
                if !self.audio_file_active && self.listening {
                    "Recording. Click the microphone to transcribe; Escape cancels."
                        .clone_into(&mut self.speech_status);
                } else if !self.audio_file_active && self.dictation_requested {
                    "Microphone enabled.".clone_into(&mut self.speech_status);
                }
            }
            speech::Event::FileProgress {
                generation,
                progress,
            } => self.update_audio_file_progress(generation, progress),
            speech::Event::FilePart { generation, text } => self.audio_file_part(generation, &text),
            speech::Event::FileFinished { generation } => {
                if self.audio_file_active && self.microphone_generation == Some(generation) {
                    self.audio_file_review_required |= self.show_audio_transcript;
                }
                self.finish_audio_file(generation);
            }
            speech::Event::Recognized { generation, text } => {
                if self.dictation_requested
                    && self.dictation_finishing
                    && self.microphone_generation == Some(generation)
                {
                    // User input in this frame must settle before this can send.
                    self.pending_voice_transcript = Some((generation, text));
                }
            }
            speech::Event::Speaking {
                generation,
                speaking,
            } => self.speaking_status(generation, speaking),
            speech::Event::Failed {
                generation,
                message,
            } => {
                if self.is_current_speech_output(generation) {
                    self.speech_notice = true;
                }
                self.output_status(generation, false, message);
            }
            speech::Event::OutputBuffering {
                generation,
                buffering,
            } => {
                if self.is_current_speech_output(generation) && self.speaking {
                    self.speech_loading = buffering;
                }
            }
            speech::Event::InputFailed {
                generation,
                message,
            } => {
                if self.dictation_requested && self.microphone_generation == Some(generation) {
                    if self.audio_file_active {
                        self.fail_audio_file(message);
                    } else {
                        self.cancel_microphone();
                    }
                    message.clone_into(&mut self.speech_status);
                    self.speech_notice = true;
                }
            }
            speech::Event::Status(message) if self.dictation_requested => {
                message.clone_into(&mut self.speech_status);
            }
            speech::Event::Status(_) => {}
        }
        if self.speech_status != previous {
            self.record_speech_status();
        }
    }

    fn update_audio_file_progress(
        &mut self,
        generation: u64,
        progress: Option<speech::FileProgress>,
    ) {
        if !self.audio_file_active || self.microphone_generation != Some(generation) {
            return;
        }
        if let Some(progress) = progress {
            self.audio_file_progress = Some(progress);
        }
        self.audio_file_notice = Some(
            if progress.is_some() {
                "Recognizing speech in this segment. Cancel or Escape stops transcription."
            } else {
                "Preparing the next audio segment. Cancel or Escape stops transcription."
            }
            .to_owned(),
        );
    }

    fn speaking_status(&mut self, generation: u64, speaking: bool) {
        let message = if speaking {
            if self.streaming_active {
                "Reading the response as it arrives."
            } else {
                "Reading the selected response aloud."
            }
        } else {
            "Read aloud stopped."
        };
        self.output_status(generation, speaking, message);
    }

    fn is_current_speech_output(&self, generation: u64) -> bool {
        self.speech
            .as_ref()
            .is_some_and(|session| session.is_current_output(generation))
    }

    fn output_status(&mut self, generation: u64, speaking: bool, message: &str) {
        if self.is_current_speech_output(generation) {
            self.speaking = speaking;
            if !speaking {
                self.speech_loading = false;
                self.speaking_message = None;
                self.streaming_active = false;
                self.streaming_suppressed = true;
            }
            message.clone_into(&mut self.speech_status);
        }
    }

    fn accept_transcript(&mut self, generation: u64, text: &str) {
        if !self.dictation_requested
            || self.microphone_generation != Some(generation)
            || !self.dictation_finishing
        {
            return;
        }
        let from_file = self.audio_file_active;
        self.release_completed_input();
        let text = text.trim();
        if text.is_empty() {
            "No speech recognized. Your draft was kept.".clone_into(&mut self.speech_status);
            if from_file {
                self.audio_file_notice = Some(self.speech_status.clone());
            }
            return;
        }
        let complete = self.insert_dictation(text);
        if from_file && !complete {
            self.audio_file_notice = Some("Transcription complete. View or copy the full transcript below. It exceeds the chat prompt size, so no message was sent. Your draft was kept.".to_owned());
            return;
        }
        if !complete {
            "Voice message exceeds the prompt limit. Your draft was kept; no message was sent."
                .clone_into(&mut self.speech_status);
        } else if from_file
            && self.model.ready()
            && !self.composing
            && self.rename_conversation.is_none()
            && self.delete_conversation.is_none()
            && self.edit_message.is_none()
            && self.branch_comparison.is_none()
            && !self.show_settings
            && !self.show_audio_transcript
            && self.alerts.details.is_none()
            && !self.audio_file_review_required
        {
            self.submit();
            if self.model.busy() {
                "Audio file transcribed and sent.".clone_into(&mut self.speech_status);
            }
        } else {
            if from_file && self.model.ready() {
                "Audio file transcribed. Review the draft, then select Send."
            } else if from_file {
                "Audio file transcribed. Select Send when the model is ready."
            } else if self.model.ready() {
                "Voice message transcribed. Review the draft, then select Send."
            } else {
                "Voice message transcribed. Select Send when the model is ready."
            }
            .clone_into(&mut self.speech_status);
        }
        if from_file {
            self.audio_file_notice = Some(self.speech_status.clone());
        }
        self.focus_composer = !self.show_audio_transcript && !self.show_settings;
    }

    fn flush_voice_transcript(&mut self, ctx: &egui::Context) {
        if self.audio_file_active
            && self.show_audio_transcript
            && self.pending_voice_transcript.is_some()
        {
            self.audio_file_review_required = true;
        }
        // A press can become a Cancel or conversation switch on release. Leave
        // the transcript staged until the native widgets have handled that click.
        if ctx.input(|input| input.pointer.any_down()) {
            return;
        }
        if let Some((generation, text)) = self.pending_voice_transcript.take() {
            self.accept_transcript(generation, &text);
            ctx.request_repaint();
        }
    }

    fn place_composer_cursor(&mut self, index: usize) {
        self.recovery_dirty = true;
        let cursor = egui::text::CCursorRange::one(egui::text::CCursor::new(index));
        self.composer_cursor = Some(cursor);
        self.pending_cursor = Some(cursor);
    }

    fn transcript_insertion<'a>(&self, text: &'a str) -> Option<composer_insertion::Plan<'a>> {
        let selection = self.composer_cursor.map(|cursor| {
            let range = cursor.as_sorted_char_range();
            usize::from(range.start)..usize::from(range.end)
        });
        composer_insertion::Plan::new(&self.input, selection, text, INPUT_LIMIT)
    }

    fn insert_dictation(&mut self, text: &str) -> bool {
        let Some(plan) = self.transcript_insertion(text) else {
            return false;
        };
        let cursor = plan.apply(&mut self.input);
        self.input_revision = self.input_revision.wrapping_add(1);
        self.place_composer_cursor(cursor);
        self.touch_conversation(self.active_conversation);
        true
    }

    fn poll_microphone(&mut self) {
        if self.audio_file_active {
            return;
        }
        let Some(event) = self
            .microphone
            .as_mut()
            .and_then(microphone::Microphone::poll)
        else {
            return;
        };
        let previous = self.speech_status.clone();
        self.microphone_event(event);
        if self.speech_status != previous {
            self.record_speech_status();
        }
    }

    fn microphone_event(&mut self, event: microphone::Event) {
        match event {
            microphone::Event::Recording { generation } => {
                if self.dictation_requested
                    && self.microphone_generation == Some(generation)
                    && self.speech_command(speech::Command::PrepareInput { generation })
                {
                    "Recording. Click the microphone to transcribe; Escape cancels. Finishes after 25 seconds."
                        .clone_into(&mut self.speech_status);
                }
            }
            microphone::Event::Status(message) if self.dictation_requested => {
                message.clone_into(&mut self.speech_status);
            }
            microphone::Event::Status(_) => {}
            microphone::Event::Failed(message) => {
                self.speech_notice = true;
                self.cancel_microphone();
                message.clone_into(&mut self.speech_status);
            }
            microphone::Event::Clip { generation, wav } => {
                if self.dictation_requested && self.microphone_generation == Some(generation) {
                    self.listening = false;
                    self.dictation_finishing = true;
                    if self.speech_command(speech::Command::Transcribe { generation, wav }) {
                        "Transcribing with Qwen3-ASR 1.7B… Escape cancels."
                            .clone_into(&mut self.speech_status);
                    }
                }
            }
        }
    }
}

impl Drop for State {
    fn drop(&mut self) {
        if let Some(store) = &mut self.profile_store {
            let _queued = store.queue(&self.voice_profile);
        }
        self.profile_store.take();
        if let Some(store) = &mut self.appearance_store {
            let _queued = store.queue(self.appearance);
        }
        self.appearance_store.take();
        self.audio_paste.take();
        self.cancel_attachment_import();
        self.branch_comparison.take();
        self.updates.take();
        self.pending_config.take();
        self.startup_request.take();
        self.startup.take();
        self.stop_speech();
        if let Some(recovery) = &self.recovery {
            if self.recovery_erase_pending {
                recovery.finish_erase();
            } else if self.save_conversations {
                recovery.finish_with(|| self.workspace_bytes().ok());
            }
        }
        self.microphone.take();
        self.speech.take();
        let _sent = self.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _joined = worker.join();
        }
    }
}

fn restore_rejected_draft_for(
    input: &mut String,
    input_revision: &mut u64,
    draft_attachments: &mut Vec<crate::attachment_import::Attachment>,
    draft_attachments_revision: &mut u64,
    transcript: &Transcript,
    prefix: usize,
    wire_text: &str,
) -> bool {
    if let Some(turn_user) = transcript
        .messages
        .iter()
        .rfind(|message| message.role == Role::User && message.matches_turn_prefix(prefix))
        && !turn_user.attachments.is_empty()
    {
        let mut restored = false;
        if input.is_empty() && !turn_user.body.is_empty() && draft_buffer::fits(&turn_user.body) {
            turn_user.body.clone_into(input);
            *input_revision = input_revision.wrapping_add(1);
            restored = true;
        }
        if draft_attachments.is_empty() {
            draft_attachments.clone_from(&turn_user.attachments);
            *draft_attachments_revision = draft_attachments_revision.wrapping_add(1);
            restored = true;
        }
        return restored;
    }
    if input.is_empty() && draft_buffer::fits(wire_text) {
        wire_text.clone_into(input);
        *input_revision = input_revision.wrapping_add(1);
        true
    } else {
        false
    }
}

fn configure(ctx: &egui::Context) {
    appearance::configure(ctx);
}

#[cfg(test)]
fn test_font_provider() -> std::sync::Arc<egui_system_font_provider::SystemFontProvider> {
    use std::sync::{Arc, LazyLock};

    // Enumerate the same installed fonts once. Each harness still owns its
    // context, font atlas, styles and layout caches.
    static PROVIDER: LazyLock<Arc<egui_system_font_provider::SystemFontProvider>> =
        LazyLock::new(|| Arc::new(egui_system_font_provider::SystemFontProvider::new()));
    Arc::clone(&PROVIDER)
}

#[cfg(test)]
fn configure_test_context(ctx: &egui::Context) {
    appearance::prepare_fonts();
    configure(ctx);
    ctx.add_font_provider(test_font_provider());
}

impl State {
    fn sidebar(&mut self, ui: &mut egui::Ui) {
        settle_panel_height(
            ui,
            "sidebar-footer",
            egui::Margin {
                left: 0,
                right: 0,
                top: 8,
                bottom: 0,
            },
            |ui| self.sidebar_footer(ui),
        );
        ui.add_space(4.0);
        branding::logo(ui);
        ui.add_space(8.0);
        if ui
            .add_enabled(
                self.model.workspace_available(),
                egui::Button::new("+ New conversation")
                    .min_size(egui::vec2(ui.available_width(), 44.0))
                    .corner_radius(10),
            )
            .on_hover_text("Start fresh · Ctrl+N")
            .clicked()
        {
            self.new_chat();
        }
        ui.add_space(8.0);
        self.conversation_list(ui);
    }

    fn sidebar_footer(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().item_spacing.y = 4.0;
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.ctx().content_rect().width() >= navigation::DOCK_THRESHOLD {
                self.navigation_button(ui, false);
            }
            let pending = self
                .updates
                .as_ref()
                .is_some_and(updates::Updates::available);
            let response = ui
                .add_sized([ui.available_width(), 36.0], egui::Button::new("Settings"))
                .accessible_name(if pending {
                    "Settings, update pending"
                } else {
                    "Settings"
                });
            if pending {
                let center = egui::pos2(response.rect.right() - 12.0, response.rect.top() + 10.0);
                ui.painter()
                    .circle_filled(center, 4.0, ui.visuals().selection.stroke.color);
            }
            let response = response.on_hover_text(if pending {
                "Settings · update pending"
            } else {
                "Settings"
            });
            if response.clicked() {
                self.open_settings(if pending {
                    settings::Page::Application
                } else {
                    settings::Page::Response
                });
            }
        });
        ui.add_space(4.0);
        self.sidebar_status(ui);
    }

    fn runtime_controls(&self, ui: &mut egui::Ui) {
        ui.heading("Runtime");
        ui.label(&self.runtime_description);
        ui.label(
            RichText::new("Development build. Full-context, sustained-performance and output-quality qualification remain pending.")
                .small()
                .weak(),
        );
        ui.add_space(16.0);
    }

    fn can_install_update(&self) -> bool {
        self.startup.is_none()
            && self.startup_request.is_none()
            && (matches!(
                self.model,
                ModelLifecycle::Ready | ModelLifecycle::Stopped | ModelLifecycle::Retryable
            ) || self.setup_stopped_for_update())
    }

    fn update_controls(&mut self, ui: &mut egui::Ui) {
        let can_install = self.can_install_update();
        let setup_stopped = self.setup_stopped_for_update();
        let Some(updates) = &mut self.updates else {
            updates::manual_controls(ui);
            return;
        };
        updates.controls(ui, can_install);
        if self.model == ModelLifecycle::Unavailable && !setup_stopped {
            ui.label("If model preparation failed, close Swift before updating the application.");
        }
    }

    fn begin_update(&mut self) {
        let Some(updates) = &mut self.updates else {
            return;
        };
        updates.wait_for_stop();
        self.stop_speech();
        if matches!(
            self.model,
            ModelLifecycle::Generating | ModelLifecycle::Stopping
        ) {
            if self.commands.send(Command::StopModel).is_err() {
                self.fail("The model worker stopped. Close and restart the app before updating.");
            } else {
                self.with_request_conversation(|state| {
                    state.model = ModelLifecycle::StoppingModel;
                    state.transcript.remove_empty_response();
                    state.set_status("Stopping the local model before updating...", false);
                    state.touch_conversation(state.active_or_routed_conversation());
                });
            }
        } else if self.model == ModelLifecycle::Ready {
            if self.commands.send(Command::StopModel).is_err() {
                self.fail("The model worker stopped. Close and restart the app before updating.");
            } else {
                self.model = ModelLifecycle::StoppingModel;
                self.set_status("Stopping the local model before updating...", false);
            }
        }
    }

    fn update_confirmation_visible(&self) -> bool {
        self.updates
            .as_ref()
            .is_some_and(updates::Updates::confirmation_pending)
    }

    fn close_with_pending_update(&mut self, ctx: &egui::Context) {
        if !ctx.input(|input| input.viewport().close_requested()) {
            return;
        }
        if self.prepare_update_on_close() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        }
    }

    fn prepare_update_on_close(&mut self) -> bool {
        let Some(updates) = &self.updates else {
            return false;
        };
        if updates.should_close() {
            return false;
        }
        if updates.closing() {
            return updates.installing();
        }
        // Preparation and unknown model ownership retain ordinary shutdown.
        // A live response can stop through the same acknowledged model teardown
        // as explicit installation, preserving its usable partial reply.
        let can_stop = self.startup.is_none()
            && self.startup_request.is_none()
            && (matches!(
                self.model,
                ModelLifecycle::Ready
                    | ModelLifecycle::Generating
                    | ModelLifecycle::Stopping
                    | ModelLifecycle::StoppingModel
                    | ModelLifecycle::AwaitingTeardown
                    | ModelLifecycle::Stopped
                    | ModelLifecycle::Retryable
            ) || self.setup_stopped_for_update());
        if can_stop
            && self
                .updates
                .as_mut()
                .is_some_and(updates::Updates::approve_on_close)
        {
            self.begin_update();
            return true;
        }
        false
    }

    fn poll_updates(&mut self, ctx: &egui::Context) {
        let setup_stopped = self.setup_stopped_for_update();
        let Some(updates) = &mut self.updates else {
            return;
        };
        if self.model.reloadable() || setup_stopped {
            updates.finish_stop(true);
        } else if self.model == ModelLifecycle::Unavailable {
            updates.finish_stop(false);
        }
        let diagnostic = updates.poll();
        if updates.should_close() {
            if let Some(approved) = &self.update_exit {
                approved.store(true, std::sync::atomic::Ordering::Release);
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if updates.closing() && !updates.installing() {
            // A publication/teardown failure must neither replace bytes nor
            // trap a user who asked to exit. No approved update exit is set.
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if let Some(message) = diagnostic {
            self.record(&message);
        }
    }

    fn diagnostics(&mut self, ctx: &egui::Context) {
        if !self.show_diagnostics {
            return;
        }
        let mut open = true;
        egui::Window::new("Diagnostics")
            .open(&mut open)
            .default_width(640.0)
            .max_width((ctx.content_rect().width() - 32.0).clamp(240.0, 640.0))
            .max_height((ctx.content_rect().height() - 32.0).clamp(120.0, 360.0))
            .show(ctx, |ui| {
                self.diagnostics_contents(ui);
            });
        self.show_diagnostics = open;
    }

    fn conversation_notices(&mut self, ui: &mut egui::Ui) {
        if let Some(notice) = &mut self.conversation_notice
            && let Some(duration) = notice.auto_dismiss_after
        {
            let now = ui.input(|input| input.time);
            let deadline = *notice
                .dismiss_at
                .get_or_insert_with(|| now + duration.as_secs_f64());
            if !now.is_finite() || now >= deadline {
                self.conversation_notice = None;
                ui.ctx().request_repaint();
            } else if let Ok(remaining) = Duration::try_from_secs_f64(deadline - now) {
                ui.ctx().request_repaint_after(remaining);
            }
        }
        if let Some(notice) = self.conversation_notice {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(notice.text).small());
                if ui.small_button("Dismiss").clicked() {
                    self.conversation_notice = None;
                }
            });
        }
        if self.preview {
            ui.label(
                RichText::new("PREVIEW  ·  Launch normally to use your local model.")
                    .small()
                    .weak(),
            );
        }
    }

    fn empty_state(&mut self, ui: &mut egui::Ui) {
        ui.add_space((ui.available_height() * 0.18).max(20.0));
        ui.vertical_centered(|ui| {
            ui.label(
                RichText::new("B")
                    .size(44.0)
                    .color(ui.visuals().selection.bg_fill),
            );
            ui.add_space(12.0);
            ui.label(RichText::new("What’s on your mind?").strong().size(30.0));
            ui.add_space(8.0);
            ui.label(RichText::new("Write, explore, or work through an idea together.").weak());
            ui.add_space(28.0);
            for (title, prompt) in [
                (
                    "Find the right words",
                    "Help me write a clear, thoughtful message about ",
                ),
                ("Explore an idea", "Help me think through an idea: "),
                ("Make a plan", "Help me create a practical plan for "),
            ] {
                if ui
                    .add(egui::Button::new(title).min_size(egui::vec2(280.0, 42.0)))
                    .clicked()
                {
                    prompt.clone_into(&mut self.input);
                    self.input_revision = self.input_revision.wrapping_add(1);
                    self.place_composer_cursor(self.input.chars().count());
                    self.touch_conversation(self.active_conversation);
                    self.focus_composer = true;
                }
            }
        });
    }

    fn start_image_export(
        &mut self,
        ctx: &egui::Context,
        svg: std::sync::Arc<str>,
        suggested_stem: String,
    ) {
        if self.image_export.is_some() {
            return;
        }
        let owner = self.native_owner.unwrap_or(0);
        match crate::image_export::Export::start(ctx, owner, svg, suggested_stem) {
            Ok(export) => {
                self.image_export = Some(export);
                self.set_status("Choose where to save the image…", false);
            }
            Err(_) => {
                self.set_status("Could not open the image save dialog.", true);
            }
        }
    }

    fn poll_image_export(&mut self, ctx: &egui::Context) {
        let Some(export) = &mut self.image_export else {
            return;
        };
        match export.poll() {
            crate::image_export::Outcome::Pending => {}
            crate::image_export::Outcome::Cancelled => {
                self.image_export = None;
                self.set_status("Image save cancelled.", false);
                ctx.request_repaint();
            }
            crate::image_export::Outcome::Saved(path) => {
                self.image_export = None;
                self.set_status(&format!("Saved image to {}.", path.display()), false);
                ctx.request_repaint();
            }
            crate::image_export::Outcome::Failed(message) => {
                self.image_export = None;
                self.set_status(message, true);
                ctx.request_repaint();
            }
        }
    }

    fn show_transcript(&mut self, ui: &mut egui::Ui, width: f32, viewport: egui::Rect) {
        let available = !self
            .updates
            .as_ref()
            .is_some_and(updates::Updates::installing);
        let controls = transcript::MessageControls {
            history: &self.committed,
            can_send: available && !self.preview && self.model == ModelLifecycle::Ready,
            can_reload: available && !self.preview && self.model.reloadable(),
            can_branch: available && self.model.workspace_available(),
            speech_loading: self.speaking_message.filter(|_| self.speech_loading),
            speech_preparing: self.tts_configured && !self.speech_checked,
            export_busy: self.image_export.is_some(),
        };
        match self.transcript.show_with_actions(
            ui,
            width,
            viewport,
            self.speaking_message,
            self.speech_output,
            Some(controls),
            Some(&mut self.image_cache),
        ) {
            Some(transcript::TranscriptAction::ReadAloud(id)) => self.toggle_read_aloud(id),
            Some(transcript::TranscriptAction::Message(action)) => self.message_action(action),
            Some(transcript::TranscriptAction::OpenLightbox {
                svg,
                intrinsic_size,
                suggested_stem,
            }) => {
                self.lightbox = Some(lightbox::Lightbox::new(svg, intrinsic_size, suggested_stem));
                ui.ctx().request_repaint();
            }
            Some(transcript::TranscriptAction::InspectAttachment { message_id, index }) => {
                if let Some(attachment) = self
                    .transcript
                    .messages
                    .iter()
                    .find(|message| message.id == message_id)
                    .and_then(|message| message.attachments.get(index))
                {
                    self.lightbox = lightbox::Lightbox::from_attachment(attachment);
                    ui.ctx().request_repaint();
                }
            }
            Some(transcript::TranscriptAction::SaveImage {
                svg,
                suggested_stem,
            }) => self.start_image_export(ui.ctx(), svg.into_arc(), suggested_stem),
            None => {}
        }
    }

    fn application_notices(&mut self, ui: &mut egui::Ui) {
        self.setup_controls(ui);
        self.conversation_notices(ui);
        self.alert_panel(ui);
        let can_install = self.can_install_update();
        if !self.show_settings
            && let Some(updates) = &mut self.updates
        {
            updates.nudge(ui, can_install);
        }
    }

    fn draw_workspace(&mut self, ui: &mut egui::Ui) {
        let narrow = ui.available_width() < navigation::DOCK_THRESHOLD;
        self.docked_navigation(ui, narrow);
        let horizontal_margin = if narrow { 6 } else { 12 };
        let find_open = self.transcript.find_open();
        egui::CentralPanel::default()
            .frame(egui::Frame::new().inner_margin(egui::Margin {
                left: 0,
                right: 0,
                top: if find_open { 12 } else { 0 },
                bottom: 12,
            }))
            .show(ui, |ui| {
                horizontal_inset(ui, horizontal_margin, "workspace-notices", |ui| {
                    self.application_notices(ui);
                });
                settle_panel_height(
                    ui,
                    "composer-panel",
                    egui::Margin {
                        left: horizontal_margin,
                        right: horizontal_margin,
                        top: 12,
                        bottom: 4,
                    },
                    |ui| self.composer(ui),
                );
                let empty = self.transcript.messages.is_empty();
                // Even an empty Find scope advances egui's layout cursor.
                let closed_find = find_open
                    && horizontal_inset(ui, horizontal_margin, "workspace-find", |ui| {
                        ui.push_id(("conversation-find", self.active_conversation), |ui| {
                            self.transcript.search_bar(ui)
                        })
                        .inner
                    });
                if closed_find {
                    self.focus_composer = true;
                    ui.ctx().request_repaint();
                }
                let scroll = egui::ScrollArea::vertical()
                    .id_salt(("conversation-scroll", self.active_conversation))
                    .animated(false)
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .content_margin(egui::Margin::symmetric(horizontal_margin, 0))
                    .show_viewport(ui, |ui, viewport| {
                        let content_top = ui.next_widget_position().y;
                        if empty {
                            self.empty_state(ui);
                        } else {
                            ui.add_space(transcript::MESSAGE_GAP);
                            let width = ui.available_width().min(self.appearance.column_width());
                            let margin = ((ui.available_width() - width) * 0.5).max(0.0);
                            ui.horizontal(|ui| {
                                ui.add_space(margin);
                                ui.vertical(|ui| {
                                    ui.set_width(width);
                                    ui.push_id(
                                        ("conversation-text", self.active_conversation),
                                        |ui| {
                                            let viewport = viewport.translate(egui::vec2(
                                                0.0,
                                                content_top - ui.next_widget_position().y,
                                            ));
                                            self.show_transcript(ui, width, viewport);
                                        },
                                    );
                                });
                            });
                        }
                        viewport.top()
                    });
                if (scroll.inner - scroll.state.clamped_offset().y).abs() >= 0.5 {
                    self.transcript.invalidate_presentation();
                }
                if !empty
                    && scroll.content_size.y
                        - scroll.inner_rect.height()
                        - scroll.state.clamped_offset().y
                        > 2.0
                {
                    self.transcript.latest_button(ui, scroll.inner_rect);
                }
            });
    }

    fn draw(&mut self, ui: &mut egui::Ui) {
        // Disable the workspace on the first modal frame too: egui's modal
        // focus layer is otherwise inherited only from the preceding pass.
        let drawer = self.sidebar_open && ui.available_width() < navigation::DOCK_THRESHOLD;
        let update_confirmation = self.update_confirmation_visible();
        ui.add_enabled_ui(
            !self.has_overlay() && !update_confirmation && !drawer,
            |ui| {
                self.draw_workspace(ui);
            },
        );
        let ctx = ui.ctx().clone();
        if update_confirmation {
            let can_install = self.can_install_update();
            if self
                .updates
                .as_mut()
                .is_some_and(|updates| updates.confirmation(&ctx, can_install))
            {
                self.begin_update();
            } else if !self.update_confirmation_visible() {
                if !self.show_settings {
                    self.focus_composer = true;
                }
                ctx.request_repaint();
            }
            return;
        }
        if self.show_settings {
            self.settings(&ctx);
            return;
        }
        self.navigation_drawer(&ctx);
        self.diagnostics(&ctx);
        self.conversation_dialogs(&ctx);
        self.iteration_dialog(&ctx);
        self.alert_details_dialog(&ctx);
        self.audio_transcript_dialog(&ctx);
        if let Some((title, message)) = self.help {
            let mut open = true;
            let width = (ctx.content_rect().width() - 80.0).clamp(240.0, 480.0);
            egui::Window::new(title)
                .open(&mut open)
                .frame(controls::dialog_frame(&ctx))
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .collapsible(false)
                .resizable(false)
                .default_width(width)
                .max_width(width)
                .show(&ctx, |ui| {
                    ui.set_width(width);
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                    ui.add_space(8.0);
                    egui::ScrollArea::vertical()
                        .id_salt("audio-setup-help")
                        .max_height((ctx.content_rect().height() - 180.0).clamp(80.0, 320.0))
                        .show(ui, |ui| {
                            ui.add(egui::Label::new(message).wrap());
                        });
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(4.0);
                    ui.allocate_ui_with_layout(
                        egui::vec2(ui.available_width(), 36.0),
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if ui.add(controls::action_button("Close")).clicked() {
                                self.help = None;
                                self.speech_notice = false;
                            }
                        },
                    );
                });
            if !open {
                self.help = None;
                self.speech_notice = false;
            }
        }
        if let Some(lightbox_action) = self.lightbox.as_mut().and_then(|lightbox| {
            lightbox.show(&ctx, &mut self.image_cache, self.image_export.is_some())
        }) {
            match lightbox_action {
                lightbox::LightboxAction::Close => {
                    self.lightbox = None;
                    ctx.request_repaint();
                }
                lightbox::LightboxAction::Save {
                    svg,
                    suggested_stem,
                } => self.start_image_export(&ctx, svg.into_arc(), suggested_stem),
            }
        }
    }
}

/// Apply symmetric horizontal margins without advancing the parent's vertical
/// cursor when no child widgets allocate height.
fn horizontal_inset<R>(
    ui: &mut egui::Ui,
    margin: i8,
    id_salt: &'static str,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let mut bounds = ui.available_rect_before_wrap();
    let inset = f32::from(margin);
    bounds.min.x += inset;
    bounds.max.x = (bounds.max.x - inset).max(bounds.min.x);
    let mut child = ui.new_child(egui::UiBuilder::new().id_salt(id_salt).max_rect(bounds));
    let result = add_contents(&mut child);
    let used = child.min_rect();
    if used.height() > 0.0 {
        ui.allocate_rect(used, egui::Sense::hover());
    }
    result
}

pub(super) fn plain_key_pressed(ctx: &egui::Context, key: egui::Key) -> bool {
    ctx.input(|input| {
        input.events.iter().any(|event| {
            matches!(event, egui::Event::Key {
                key: pressed_key,
                pressed: true,
                modifiers,
                ..
            } if *pressed_key == key && *modifiers == egui::Modifiers::NONE)
        })
    })
}

fn settle_panel_height(
    ui: &mut egui::Ui,
    id: &'static str,
    margin: egui::Margin,
    content: impl FnOnce(&mut egui::Ui),
) {
    let previous = egui::containers::panel::PanelState::load(ui.ctx(), ui.make_persistent_id(id))
        .map(|state| state.size().y);
    let panel = egui::Panel::bottom(id)
        .frame(egui::Frame::new().inner_margin(margin))
        .show(ui, content);
    // Settle the measured bottom anchor after first layout or a notice change.
    if previous.is_none_or(|height| (height - panel.response.rect.height()).abs() > 0.5) {
        ui.ctx().request_repaint();
    }
}

impl eframe::App for State {
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.repaint_ctx.is_none() {
            self.repaint_ctx = Some(ctx.clone());
        }
        self.poll_placement(frame, ctx.input(|input| input.viewport().close_requested()));
        let was_ready = self.model.ready();
        let was_busy = self.model.busy();
        self.poll_startup(ctx);
        self.drain_events();
        self.close_with_pending_update(ctx);
        self.poll_updates(ctx);
        if ((!was_ready && self.model.ready())
            || (was_busy && !self.model.busy() && self.model.ready()))
            && !self.has_overlay()
            && ctx.memory(|memory| {
                memory
                    .focused()
                    .is_none_or(|id| Some(id) == self.composer_id)
            })
            && !ctx
                .plugin::<egui::text_selection::LabelSelectionState>()
                .lock()
                .has_selection()
            && !ctx.input(|input| input.pointer.any_down())
        {
            self.focus_composer = true;
        }
        self.prepare_audio_paste(frame);
        self.handle_input_shortcuts(ctx);
        self.handle_audio_paste(ctx);
        self.poll_audio_import();
        self.poll_attachment_import(ctx);
        self.poll_branch_comparison(ctx);
        self.poll_microphone();
        self.drain_speech_events();
        self.poll_recovery();
        self.poll_appearance(ctx);
        self.poll_profile();
        self.image_cache.poll(ctx);
        self.poll_image_export(ctx);
        if self.last_theme_check.elapsed() >= Duration::from_secs(1) {
            self.appearance.apply(ctx);
            self.last_theme_check = Instant::now();
        }
        // No frame-rate polling: wait between events when idle. While waiting for
        // the first token of an active turn, throttle to 125 ms (8 Hz) and let
        // `event_channel_with_repaint` wake the UI immediately on token arrival.
        ctx.request_repaint_after(self.next_repaint_delay());
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.draw(ui);
        self.flush_voice_transcript(ui.ctx());
    }
}

impl State {
    fn next_repaint_delay(&self) -> Duration {
        let active_generation = !self.preview && self.model.busy();
        let waiting = self.image_cache.has_pending()
            || self.image_export.is_some()
            || self.attachment_import.is_some()
            || self
                .branch_comparison
                .as_ref()
                .is_some_and(iteration::BranchComparison::is_running)
            || (!self.preview
                && (self.model.waiting_model()
                    || self.speaking
                    || self.dictation_requested
                    || self.audio_import.is_some()
                    || !self.speech_checked));
        if active_generation && !self.awaiting_first_token {
            Duration::from_millis(150)
        } else if waiting {
            Duration::from_millis(50)
        } else if active_generation {
            Duration::from_millis(125)
        } else {
            Duration::from_secs(1)
        }
    }
}

fn native_options() -> io::Result<eframe::NativeOptions> {
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../assets/bonsai.png"))
        .map_err(io::Error::other)?;
    Ok(eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_icon(icon)
            .with_inner_size([1080.0, 780.0])
            .with_min_inner_size([480.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,
        ..eframe::NativeOptions::default()
    })
}

fn open_persisted_stores(
    data: &std::path::Path,
) -> impl FnOnce(&mut State, &eframe::CreationContext<'_>) + use<> {
    let recovery = crate::recovery::Recovery::open(data);
    let placement = placement::Placement::open(data);
    let appearance = appearance::Store::open(data);
    let profile = profile::Store::open(data);
    move |state, cc| {
        state.repaint_ctx = Some(cc.egui_ctx.clone());
        state.attach_recovery(recovery);
        state.attach_placement(placement, cc);
        state.attach_appearance(appearance, &cc.egui_ctx);
        state.attach_profile(profile);
    }
}

const WINDOW_TITLE: &str = "Snapdragon Bonsai";
const PREVIEW_WINDOW_TITLE: &str = "Snapdragon Bonsai · Preview";

fn run_native_window(
    title: &str,
    build: impl FnOnce(&eframe::CreationContext<'_>, Sender<Command>, client::EventReceiver) -> State
    + 'static,
) -> io::Result<()> {
    appearance::prepare_fonts();
    eframe::run_native(
        title,
        native_options()?,
        Box::new(move |cc| {
            graphics::install(cc);
            configure(&cc.egui_ctx);
            let (commands, _incoming) = mpsc::channel();
            let (_events, incoming_events) = client::event_channel();
            Ok(Box::new(build(cc, commands, incoming_events)))
        }),
    )
    .map_err(|error| io::Error::other(error.to_string()))
}

pub(super) fn run(config: Config) -> io::Result<()> {
    let attach_stores = open_persisted_stores(&config.data);
    run_native_window(WINDOW_TITLE, move |cc, commands, incoming_events| {
        let microphone = microphone::Microphone::new().ok();
        let mut state = State::create(commands, incoming_events, None, microphone);
        state.model = ModelLifecycle::Preparing;
        state.preview = false;
        state.apply_model_config(&config);
        state.pending_config = Some(config);
        attach_stores(&mut state, cc);
        state
    })
}

fn prepare_startup_directory(data: &std::path::Path) -> io::Result<()> {
    bonsai_windows_credentials::validate_external_path(data)?;
    std::fs::create_dir_all(data)?;
    bonsai_windows_credentials::validate_external_path(data)
}

fn startup_exit_result(update_approved: bool, failure: Option<String>) -> io::Result<bool> {
    // Approval is published only after a verified queue and acknowledged owned
    // teardown. A historical, safely stopped setup failure cannot suppress that
    // dedicated exit, while ordinary shutdown must still report the failure.
    if update_approved {
        return Ok(true);
    }
    failure.map_or(Ok(false), |error| Err(io::Error::other(error)))
}

pub(super) fn run_startup(request: crate::startup::Request) -> io::Result<bool> {
    prepare_startup_directory(&request.data)?;
    // Retain the established recovery ordering: restore before accepting a draft.
    let attach_stores = open_persisted_stores(&request.data);
    let failure = std::sync::Arc::new(std::sync::Mutex::new(None));
    let reported_failure = std::sync::Arc::clone(&failure);
    let update_exit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let approved_update = std::sync::Arc::clone(&update_exit);
    run_native_window(WINDOW_TITLE, move |cc, commands, incoming_events| {
        let mut state = State::pending_startup(commands, incoming_events, request);
        state.startup_failure = Some(reported_failure);
        state.update_exit = Some(approved_update);
        attach_stores(&mut state, cc);
        state
    })?;
    let error = failure
        .lock()
        .map_err(|_| io::Error::other("Cannot read native preparation result."))?
        .take();
    startup_exit_result(
        update_exit.load(std::sync::atomic::Ordering::Acquire),
        error,
    )
}
/// Opens only a seeded interface; no backend, credential or audio operation runs.
pub(super) fn run_preview() -> io::Result<()> {
    run_native_window(PREVIEW_WINDOW_TITLE, |_, commands, incoming_events| {
        let mut state = State::create(commands, incoming_events, None, None);
        state.model = ModelLifecycle::Ready;
        "Interface preview".clone_into(&mut state.status);
        state.transcript.push(
            Role::User,
            "I have a few ideas for a short story, but I’m not sure where to start.".to_owned(),
        );
        state.transcript.push(Role::Assistant, "Start with the moment something changes.\n\nA quiet bookshop receives a letter addressed to someone who has never existed. Its owner recognizes the handwriting as their own.\n\nWhat draws you in more: the mystery of the letter, or the person who finds it?".to_owned());
        state.last_answer = state
            .transcript
            .messages
            .last()
            .map_or(String::new(), |message| message.body.clone());
        state.answer_complete = true;
        state
    })
}

pub(super) fn show_error(message: &str) {
    platform::show_error(message);
}

#[cfg(test)]
mod startup_tests {
    use super::*;
    use std::{ffi::OsString, fs, path::PathBuf};

    #[test]
    fn approved_update_exit_takes_precedence_over_historical_setup_failure() -> io::Result<()> {
        assert!(startup_exit_result(
            true,
            Some("Safely stopped setup failure".into())
        )?);
        assert!(startup_exit_result(true, None)?);
        assert!(!startup_exit_result(false, None)?);
        let failure = startup_exit_result(false, Some("Setup download failed".into()))
            .err()
            .ok_or_else(|| io::Error::other("Ordinary close lost the setup failure"))?;
        assert_eq!(failure.to_string(), "Setup download failed");
        Ok(())
    }

    #[test]
    fn first_launch_custom_data_directory_opens_persistence_before_setup() -> io::Result<()> {
        let (_request, directory) = fixture()?;
        let data = directory.join("new-data");
        assert!(!data.exists());
        prepare_startup_directory(&data)?;
        let recovery = crate::recovery::Recovery::open(&data)?;
        let appearance = appearance::Store::open(&data)?;
        drop(recovery);
        drop(appearance);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn setup_failures_allow_updates_only_after_confirmed_cleanup_and_pause_exits_normally()
    -> io::Result<()> {
        let (request, directory) = fixture()?;
        let (commands, _incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::pending_startup(commands, incoming_events, request);
        let failure = std::sync::Arc::new(std::sync::Mutex::new(None));
        state.startup_failure = Some(std::sync::Arc::clone(&failure));
        assert!(!state.can_install_update());
        state.startup_request.take();
        state.fail_startup(&io::Error::other("Download failed"));
        assert!(
            state.can_install_update(),
            "A failed download must not block an application fix"
        );
        state.fail_startup(&io::Error::other(
            "Setup process cleanup could not be confirmed.",
        ));
        assert!(!state.can_install_update());
        state.fail_startup(&io::Error::new(
            io::ErrorKind::Interrupted,
            "Paused by user",
        ));
        assert!(state.can_install_update());
        assert!(
            failure
                .lock()
                .map_err(|_| io::Error::other("Poisoned fixture"))?
                .is_none(),
            "Pausing must not turn normal close into a launch error"
        );
        state.worker = Some(std::thread::spawn(|| {}));
        assert!(
            !state.can_install_update(),
            "Unknown model ownership must block replacement"
        );
        drop(state);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    fn fixture() -> io::Result<(crate::startup::Request, PathBuf)> {
        fixture_with_broker(false)
    }

    pub(super) fn fixture_with_broker(
        prepared: bool,
    ) -> io::Result<(crate::startup::Request, PathBuf)> {
        let directory = std::env::temp_dir().join(format!(
            "Bonsai-UI-Startup-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(io::Error::other)?
                .as_nanos()
        ));
        fs::create_dir(&directory)?;
        let request = directory.join("request.json");
        let tool = directory.join("tool.exe");
        fs::write(&request, "{}")?;
        fs::write(&tool, [])?;
        if prepared {
            let parser = directory.join("nonadjacent-parser");
            fs::create_dir(&parser)?;
            let broker = parser.join("broker.exe");
            fs::write(&broker, b"inert broker path fixture")?;
            fs::write(
                &request,
                serde_json::to_vec(&serde_json::json!({"BrokerPath": broker}))?,
            )?;
        }
        let arguments = vec![
            OsString::from("--startup-tool"),
            tool.into_os_string(),
            OsString::from("--startup-request"),
            request.into_os_string(),
            OsString::from("--startup-data-directory"),
            directory.clone().into_os_string(),
        ];
        Ok((crate::startup::Request::parse(&arguments)?, directory))
    }

    #[test]
    fn configured_markdown_image_is_available_before_model_setup_and_after_failure()
    -> io::Result<()> {
        let (request, directory) = fixture_with_broker(true)?;
        let expected = directory.join("nonadjacent-parser/broker.exe");
        let (commands, _incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::pending_startup(commands, incoming_events, request);
        assert_eq!(state.markdown_image.as_deref(), Some(expected.as_path()));
        assert!(state.worker.is_none() && state.startup.is_none());
        state.fail_startup(&io::Error::other("Synthetic model preparation failure"));
        assert_eq!(state.markdown_image.as_deref(), Some(expected.as_path()));
        drop(state);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn model_events_and_text_chat_continue_while_speech_prepares_or_fails() -> io::Result<()> {
        let (request, directory) = fixture()?;
        let (commands, incoming) = mpsc::channel();
        let (events, incoming_events) = client::event_channel();
        let mut state = State::pending_startup(commands, incoming_events, request);
        state.startup_request.take();
        let (session, progress) = crate::startup::Session::fixture();
        state.startup = Some(session);
        state.worker = Some(std::thread::spawn(|| {}));
        events
            .send(Event::Ready)
            .map_err(|()| io::Error::other("Model fixture disconnected"))?;
        state.drain_events();
        assert_eq!(state.model, ModelLifecycle::Ready);
        state.input = "Chat before speech is ready".into();
        state.submit();
        assert!(matches!(incoming.try_recv(), Ok(Command::Send { .. })));
        assert_eq!(state.model, ModelLifecycle::Generating);
        assert!(state.streaming_suppressed);
        let conversation = state.request_conversation;
        state.input = "A newer draft".into();
        let failure = std::sync::Arc::new(std::sync::Mutex::new(None));
        state.startup_failure = Some(std::sync::Arc::clone(&failure));
        progress
            .send(Ok(crate::startup::Progress::Speech(Err(io::Error::other(
                "Synthetic speech preparation failure",
            )))))
            .map_err(|_| io::Error::other("Preparation fixture disconnected"))?;
        state.poll_startup(&egui::Context::default());
        assert!(state.startup.is_none() && state.speech_checked);
        assert_eq!(state.model, ModelLifecycle::Generating);
        assert_eq!(state.request_conversation, conversation);
        assert_eq!(state.input, "A newer draft");
        assert!(state.speech_status.contains("Retry speech setup"));
        assert!(state.setup.speech_error.is_some());
        assert!(
            failure
                .lock()
                .map_err(|_| io::Error::other("Poisoned fixture"))?
                .is_none()
        );
        drop(state);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn pending_native_startup_preserves_draft_through_failure_and_configuration() -> io::Result<()>
    {
        let (request, directory) = fixture()?;
        let (commands, incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::pending_startup(commands, incoming_events, request);
        state.input = "A message drafted while applications prepare".into();
        state.enable_thinking = true;
        state.output_budget = 1;
        state.voice_profile = "My chosen voice".into();
        assert!(!state.preview && !state.model.ready() && state.model.waiting_model());
        assert!(state.worker.is_none() && state.speech.is_none() && state.microphone.is_none());
        state.submit();
        state.drain_events();
        assert!(incoming.try_recv().is_err());
        assert!(state.model.waiting_model() && !state.warning);
        let failure = std::sync::Arc::new(std::sync::Mutex::new(None));
        state.startup_failure = Some(std::sync::Arc::clone(&failure));
        assert!(
            failure
                .lock()
                .map_err(|_| io::Error::other("Poisoned fixture."))?
                .is_none()
        );
        state.fail_startup(&io::Error::other("Synthetic setup failure"));
        assert_eq!(
            failure
                .lock()
                .map_err(|_| io::Error::other("Poisoned fixture."))?
                .as_deref(),
            Some("Synthetic setup failure")
        );
        assert!(!state.model.waiting_model() && state.warning && !state.model.retryable());
        assert_eq!(state.input, "A message drafted while applications prepare");
        assert!(state.transcript.messages.is_empty());
        state.startup_request.take();
        state.model = ModelLifecycle::Preparing;
        let config = Config {
            broker: directory.join("broker.exe"),
            data: directory.clone(),
            evidence: directory.clone(),
            frontend_port: 5173,
            backend_port: 8081,
            require_broker_pipe: false,
            daisyrose_lora: None,
            no_daisyrose_lora: false,
            no_mmproj: false,
            bonsai25_variant: None,
            native_engine: false,
            native_row_workers: 1,
            prompt_reuse: client::PromptReuse::Disabled,
            prompt_reuse_checkpoints: None,
            prompt_reuse_checkpoint_bytes: None,
            kv_cache: None,
            cpu_prefill_tokens: None,
            cpu_prefill_gdn: None,
            spec_type: None,
            draft_max: None,
            cpu_family: None,
            placement: None,
            canary: None,
            qwen_tts: None,
        };
        state.attach_configuration_with(config, false, |_, incoming, events| {
            thread_for_fixture(incoming, events)
        })?;
        assert_eq!(state.input, "A message drafted while applications prepare");
        assert!(state.enable_thinking && state.output_budget == 1);
        assert_eq!(state.voice_profile, "My chosen voice");
        assert!(!state.preview && !state.model.ready());
        drop(state);
        fs::remove_dir_all(&directory)?;
        Ok(())
    }

    #[test]
    fn setup_retry_button_preserves_draft_history_options_and_clears_failed_exit() -> io::Result<()>
    {
        use egui_kittest::{Harness, kittest::Queryable as _};
        let (request, directory) = fixture()?;
        let (commands, incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::pending_startup(commands, incoming_events, request);
        state.startup_request.take();
        state.input = "Keep my setup draft 🌱".into();
        state.enable_thinking = true;
        state.output_budget = 1;
        state
            .transcript
            .push(Role::Assistant, "Recovered conversation".into());
        state.transcript.remove_empty_response();
        let failure = std::sync::Arc::new(std::sync::Mutex::new(None));
        state.startup_failure = Some(std::sync::Arc::clone(&failure));
        state.fail_startup(&io::Error::other(
            "Pinned download failed: connection interrupted",
        ));
        let mut harness = Harness::builder()
            .with_size([480.0, 480.0])
            .build_ui_state(|ui, state: &mut State| state.application_notices(ui), state);
        harness
            .get_by_role_and_label(egui::Role::Button, "Retry setup")
            .click();
        // Retried setup shows a spinner that deliberately keeps requesting frames.
        harness.run_steps(2);
        let state = harness.state();
        assert!(state.startup_request.is_some());
        assert_eq!(state.model, ModelLifecycle::Preparing);
        assert!(state.setup.error.is_none());
        assert_eq!(state.input, "Keep my setup draft 🌱");
        assert_eq!(state.transcript.messages[0].body, "Recovered conversation");
        assert!(state.enable_thinking && state.output_budget == 1);
        assert!(
            incoming.try_recv().is_err(),
            "Setup retry must never send a draft"
        );
        assert!(
            failure
                .lock()
                .map_err(|_| io::Error::other("Poisoned fixture"))?
                .is_none()
        );
        drop(harness);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn setup_retry_rejects_unconfirmed_cleanup_and_existing_model_worker() -> io::Result<()> {
        let (request, directory) = fixture()?;
        let (commands, _incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::pending_startup(commands, incoming_events, request);
        state.startup_request.take();
        state.fail_startup(&io::Error::other(
            "Setup process cleanup could not be confirmed.",
        ));
        state.retry_setup();
        assert!(state.startup_request.is_none());
        assert_eq!(state.model, ModelLifecycle::Unavailable);
        state.fail_startup(&io::Error::other("download failed"));
        state.worker = Some(std::thread::spawn(|| {}));
        state.retry_setup();
        assert!(
            state.startup_request.is_none(),
            "Never overlap model ownership with setup"
        );
        drop(state);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn pausing_setup_discards_queued_model_handoff_and_keeps_speech_cleanup_gate() -> io::Result<()>
    {
        for speech_pending in [false, true] {
            let (request, directory) = fixture()?;
            let (commands, _incoming) = mpsc::channel();
            let (_events, incoming_events) = client::event_channel();
            let mut state = State::pending_startup(commands, incoming_events, request);
            state.startup_request.take();
            state.input = "Keep the draft while pausing".into();
            state.setup.pausing = true;
            let (session, progress) = crate::startup::Session::fixture();
            state.startup = Some(session);
            let config = Config {
                broker: directory.join("broker.exe"),
                data: directory.clone(),
                evidence: directory.clone(),
                frontend_port: 5173,
                backend_port: 8081,
                require_broker_pipe: false,
                daisyrose_lora: None,
                no_daisyrose_lora: false,
                no_mmproj: false,
                bonsai25_variant: None,
                native_engine: true,
                native_row_workers: 4,
                prompt_reuse: client::PromptReuse::Disabled,
                prompt_reuse_checkpoints: None,
                prompt_reuse_checkpoint_bytes: None,
                kv_cache: None,
                cpu_prefill_tokens: None,
                cpu_prefill_gdn: None,
                spec_type: None,
                draft_max: None,
                cpu_family: None,
                placement: None,
                canary: None,
                qwen_tts: None,
            };
            progress
                .send(Ok(crate::startup::Progress::Model {
                    config,
                    speech_pending,
                }))
                .map_err(|_| io::Error::other("Fixture disconnected"))?;
            state.poll_startup(&egui::Context::default());
            assert!(
                state.worker.is_none(),
                "Pause must beat an unread model handoff"
            );
            assert_eq!(state.input, "Keep the draft while pausing");
            if speech_pending {
                assert!(
                    state.startup.is_some(),
                    "Wait for speech cleanup before retry"
                );
                progress
                    .send(Ok(crate::startup::Progress::Speech(Err(io::Error::other(
                        "Setup process cleanup could not be confirmed.",
                    )))))
                    .map_err(|_| io::Error::other("Fixture disconnected"))?;
                state.poll_startup(&egui::Context::default());
                state.retry_setup();
                assert!(state.startup_request.is_none());
            } else {
                assert!(state.startup.is_none());
                state.retry_setup();
                assert!(state.startup_request.is_some());
            }
            drop(state);
            fs::remove_dir_all(directory)?;
        }
        Ok(())
    }

    #[test]
    fn pending_native_startup_blocks_read_aloud_until_speech_is_available() -> io::Result<()> {
        let (request, directory) = fixture()?;
        let (commands, _incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::pending_startup(commands, incoming_events, request);
        assert!(state.canary_configured && state.tts_configured && !state.speech_checked);
        state
            .transcript
            .push(Role::Assistant, "Recovered answer to read aloud.".into());
        state.transcript.remove_empty_response();
        let message_id = state.transcript.messages[0].id;
        state.toggle_dictation();
        assert!(state.help.is_none());
        assert_eq!(
            state.speech_status,
            "Enable the microphone after speech preparation completes."
        );
        state.toggle_streaming_read_aloud();
        assert!(state.help.is_none() && !state.streaming_read_aloud);
        state.toggle_read_aloud(message_id);
        assert!(state.help.is_none());
        assert!(!state.speaking && state.speaking_message.is_none());
        let (session, speech_commands) = speech::Session::fixture()?;
        state.speech = Some(session);
        state.speech_event(speech::Event::Capabilities {
            input: true,
            output: true,
        });
        assert!(state.speech_checked && state.speech_output);
        assert!(speech_commands.try_recv().is_err());
        assert!(!state.model.ready());
        state.toggle_streaming_read_aloud();
        assert!(state.streaming_read_aloud);
        state.toggle_read_aloud(message_id);
        assert!(matches!(
            speech_commands
                .recv_timeout(Duration::from_secs(2))
                .map_err(io::Error::other)?,
            speech::Command::StopListening
        ));
        assert!(matches!(
            speech_commands
                .recv_timeout(Duration::from_secs(2))
                .map_err(io::Error::other)?,
            speech::Command::StopSpeaking
        ));
        assert!(matches!(
            speech_commands.recv_timeout(Duration::from_secs(2)).map_err(io::Error::other)?,
            speech::Command::Speak { text, voice_description }
                if text == "Recovered answer to read aloud." && voice_description == state.voice_profile
        ));
        drop(state);
        fs::remove_dir_all(&directory)?;
        Ok(())
    }

    #[test]
    fn high_water_answer_and_draft_buffers_shrink_on_clear_and_short_restore() -> io::Result<()> {
        let (request, directory) = fixture()?;
        let (commands, _incoming) = mpsc::channel();
        let (_events, incoming_events) = client::event_channel();
        let mut state = State::pending_startup(commands, incoming_events, request);

        state.last_answer = "a".repeat(256 * 1024);
        state.retained_draft = "d".repeat(256 * 1024);
        assert!(state.last_answer.capacity() > BUFFER_HIGH_WATER_BYTES);
        assert!(state.retained_draft.capacity() > BUFFER_HIGH_WATER_BYTES);

        state.clear_conversation();
        assert!(state.last_answer.is_empty());
        assert!(state.retained_draft.is_empty());
        assert!(state.last_answer.capacity() <= BUFFER_HIGH_WATER_BYTES);
        assert!(state.retained_draft.capacity() <= BUFFER_HIGH_WATER_BYTES);

        state.last_answer = "b".repeat(256 * 1024);
        state.committed.push(client::Message {
            role: "assistant",
            content: "short".into(),
            reasoning_content: None,
            images: std::sync::Arc::from([]),
        });
        state.restore_committed_last_answer();
        assert_eq!(state.last_answer, "short");
        assert!(state.last_answer.capacity() <= BUFFER_HIGH_WATER_BYTES);

        // Steady-state capacity below high-water is preserved across clears.
        let mut moderate = String::with_capacity(8 * 1024);
        moderate.push_str("moderate");
        let kept_cap = moderate.capacity();
        clear_and_shrink_buffer(&mut moderate);
        assert_eq!(moderate.capacity(), kept_cap);

        drop(state);
        fs::remove_dir_all(&directory)?;
        Ok(())
    }

    fn thread_for_fixture(
        incoming: Receiver<Command>,
        events: client::EventSender,
    ) -> io::Result<JoinHandle<()>> {
        std::thread::Builder::new()
            .name("startup-regression".into())
            .spawn(move || {
                let _retained_events = events;
                while let Ok(command) = incoming.recv() {
                    if matches!(command, Command::Shutdown) {
                        break;
                    }
                }
            })
    }
}
