//! Native multi-file context attachments: UTF-8 text files and isolated image/video decoding.

use std::{
    ffi::OsString,
    io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use base64ct::{Base64, Encoding as _};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

use crate::com_dialog::{
    Cancel, FilterSpec, MultiDialogError, MultiDialogSpec, run_multi_open_modal,
};

mod image;
mod isolated;
mod text;
#[allow(
    unsafe_code,
    reason = "Audited isolated AppContainer WIC and Media Foundation decoding boundary."
)]
mod worker;

/// Maximum number of attachments permitted on a single prompt message.
pub(crate) const MAX_ATTACHMENTS_PER_MESSAGE: usize = crate::com_dialog::MAX_PICKED_FILES;
/// Maximum byte length accepted for any single UTF-8 text/code file attachment.
pub(crate) const MAX_TEXT_FILE_BYTES: usize = 256 * 1024;
/// Maximum combined normalized UTF-8 byte length across all text attachments in one message.
pub(crate) const MAX_TOTAL_TEXT_ATTACHMENT_BYTES: usize = 512 * 1024;
/// Maximum source file byte size accepted before launching an isolated image/video worker.
pub(crate) const MAX_SOURCE_MEDIA_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum width or height after downscaling attached standalone images.
pub(crate) const MAX_IMAGE_EDGE: u32 = 1_280;
/// Maximum width or height after downscaling extracted video keyframes.
pub(crate) const MAX_VIDEO_FRAME_EDGE: u32 = 768;
/// Maximum width or height for UI preview thumbnails persisted in history.
pub(crate) const MAX_THUMBNAIL_EDGE: u32 = 256;
/// Maximum number of keyframes extracted from a single attached video.
pub(crate) const MAX_VIDEO_KEYFRAMES: usize = 6;
/// Maximum encoded PNG byte length per model-bound image or frame.
pub(crate) const MAX_MODEL_PNG_BYTES: usize = 12 * 1024 * 1024;
/// Maximum combined encoded PNG byte length across all frames of one attached video.
pub(crate) const MAX_TOTAL_VIDEO_PNG_BYTES: usize = 9 * 1024 * 1024;
const MAX_THUMBNAIL_PNG_BYTES: usize = 512 * 1024;
const IMPORT_TIMEOUT: Duration = Duration::from_secs(25);
const POLL: Duration = Duration::from_millis(10);

const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "md", "markdown", "json", "csv", "tsv", "log", "yaml", "yml", "toml", "xml", "rs",
    "ps1", "py", "c", "h", "cpp", "hpp", "js", "ts", "html", "css", "sql", "ini", "cfg", "conf",
];
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"];
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "m4v", "webm", "mkv", "avi", "wmv"];

const TEXT_FILTERS: [FilterSpec<'static>; 1] = [FilterSpec {
    name: "Text & code files (*.txt;*.md;*.json;*.csv;*.log;*.yaml;*.toml;*.xml;*.rs;*.ps1;...)",
    pattern: "*.txt;*.md;*.markdown;*.json;*.csv;*.tsv;*.log;*.yaml;*.yml;*.toml;*.xml;*.rs;*.ps1;*.py;*.c;*.h;*.cpp;*.hpp;*.js;*.ts;*.html;*.css;*.sql;*.ini;*.cfg;*.conf",
}];
const MEDIA_FILTERS: [FilterSpec<'static>; 1] = [FilterSpec {
    name: "Images & videos (*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif;*.mp4;*.mov;*.mkv;*.webm;...)",
    pattern: "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif;*.tif;*.tiff;*.mp4;*.mov;*.m4v;*.webm;*.mkv;*.avi;*.wmv",
}];

/// Submenu selection category for the native multi-file attachment picker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PickerKind {
    Text,
    Media,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MediaKind {
    Image,
    Video,
}

impl MediaKind {
    const fn as_arg(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
        }
    }

    fn from_arg(value: &str) -> Option<Self> {
        match value {
            "image" => Some(Self::Image),
            "video" => Some(Self::Video),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FileCategory {
    Text,
    Media(MediaKind),
}

/// One extracted video frame stored as a bounded PNG data URL with its timeline offset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct VideoFrame {
    pub(crate) timestamp_ms: u64,
    pub(crate) data_url: Arc<str>,
}

/// Immutable PNG thumbnail bytes paired with a once-computed FNV-1a content hash.
#[derive(Clone, Debug, Eq)]
pub(crate) struct PrehashedPng {
    bytes: Arc<[u8]>,
    hash: u64,
}

impl PrehashedPng {
    /// Wraps `bytes` alongside its precomputed FNV-1a content hash.
    pub(crate) fn new(bytes: Arc<[u8]>) -> Self {
        let hash = crate::window::fnv1a64(&bytes);
        Self { bytes, hash }
    }

    /// Returns the precomputed FNV-1a content hash of the PNG payload.
    pub(crate) const fn hash(&self) -> u64 {
        self.hash
    }
}

impl PartialEq for PrehashedPng {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && self.bytes.len() == other.bytes.len()
            && (Arc::ptr_eq(&self.bytes, &other.bytes) || self.bytes.as_ref() == other.bytes.as_ref())
    }
}

impl std::ops::Deref for PrehashedPng {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl AsRef<[u8]> for PrehashedPng {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl From<Arc<[u8]>> for PrehashedPng {
    fn from(bytes: Arc<[u8]>) -> Self {
        Self::new(bytes)
    }
}

impl From<Vec<u8>> for PrehashedPng {
    fn from(bytes: Vec<u8>) -> Self {
        Self::new(Arc::from(bytes))
    }
}

impl<const N: usize> From<[u8; N]> for PrehashedPng {
    fn from(bytes: [u8; N]) -> Self {
        Self::new(Arc::from(bytes))
    }
}

/// Context attachment attached to a draft or persisted prompt message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum Attachment {
    Text {
        name: String,
        text: String,
    },
    Image {
        name: String,
        width: u32,
        height: u32,
        data_url: Arc<str>,
        #[serde(with = "base64_bytes")]
        thumbnail_png: PrehashedPng,
    },
    Video {
        name: String,
        width: u32,
        height: u32,
        duration_ms: u64,
        frames: Arc<[VideoFrame]>,
        #[serde(with = "base64_bytes")]
        thumbnail_png: PrehashedPng,
    },
}

impl Attachment {
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Text { name, .. } | Self::Image { name, .. } | Self::Video { name, .. } => name,
        }
    }

    pub(crate) const fn is_media(&self) -> bool {
        matches!(self, Self::Image { .. } | Self::Video { .. })
    }

    pub(crate) fn image_count(&self) -> usize {
        match self {
            Self::Text { .. } => 0,
            Self::Image { .. } => 1,
            Self::Video { frames, .. } => frames.len(),
        }
    }

    pub(crate) fn thumbnail_png(&self) -> Option<&PrehashedPng> {
        match self {
            Self::Text { .. } => None,
            Self::Image { thumbnail_png, .. } | Self::Video { thumbnail_png, .. } => {
                Some(thumbnail_png)
            }
        }
    }

    pub(crate) fn image_data_urls(&self) -> Vec<Arc<str>> {
        match self {
            Self::Text { .. } => Vec::new(),
            Self::Image { data_url, .. } => vec![Arc::clone(data_url)],
            Self::Video { frames, .. } => frames
                .iter()
                .map(|frame| Arc::clone(&frame.data_url))
                .collect(),
        }
    }

    pub(crate) fn frame_count(&self) -> usize {
        self.image_count()
    }

    pub(crate) fn text_bytes(&self) -> usize {
        match self {
            Self::Text { text, .. } => text.len(),
            Self::Image { .. } | Self::Video { .. } => 0,
        }
    }
}

fn longest_backtick_run(text: &str) -> usize {
    let mut best = 0_usize;
    let mut current = 0_usize;
    for ch in text.chars() {
        if ch == '`' {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

/// Formats one user turn's caption and attachments into broker prompt text and ordered image URLs.
pub(crate) fn format_user_turn(
    caption: &str,
    attachments: &[Attachment],
) -> (String, Arc<[Arc<str>]>) {
    if attachments.is_empty() {
        return (caption.to_owned(), Arc::default());
    }
    let mut sections = Vec::new();
    let mut images = Vec::new();
    for attachment in attachments {
        match attachment {
            Attachment::Text { name, text } => {
                let fence = "`".repeat(longest_backtick_run(text).max(2) + 1);
                if text.ends_with('\n') {
                    sections.push(format!("File: {name}\n{fence}\n{text}{fence}"));
                } else {
                    sections.push(format!("File: {name}\n{fence}\n{text}\n{fence}"));
                }
            }
            Attachment::Image { data_url, .. } => {
                images.push(Arc::clone(data_url));
            }
            Attachment::Video { name, frames, .. } => {
                let mut timestamps = Vec::with_capacity(frames.len());
                for frame in frames.iter() {
                    images.push(Arc::clone(&frame.data_url));
                    timestamps.push(format!(
                        "{}.{}s",
                        frame.timestamp_ms / 1000,
                        (frame.timestamp_ms % 1000) / 100
                    ));
                }
                if !frames.is_empty() {
                    sections.push(format!(
                        "Video: {name} ({} keyframes: {})",
                        frames.len(),
                        timestamps.join(", ")
                    ));
                }
            }
        }
    }
    if !caption.is_empty() {
        sections.push(caption.to_owned());
    }
    (sections.join("\n\n"), images.into())
}

/// Result delivered once the native picker and bounded file import finish.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Dialog cancelled or closed with no files selected.
    Cancelled,
    /// Imported attachments ready to append to the active draft, with an optional partial notice.
    Imported {
        attachments: Vec<Attachment>,
        notice: Option<String>,
    },
    /// Selection or decoding failed before adding attachments.
    Failed(String),
}

#[derive(Default)]
pub(crate) struct Shared {
    pub(crate) cancelled: AtomicBool,
    picker: Cancel,
}

/// Active background native picker + attachment import job.
pub(crate) struct Session {
    shared: Arc<Shared>,
    receiver: Receiver<Outcome>,
    worker: Option<JoinHandle<()>>,
    complete: bool,
}

impl Session {
    pub(crate) fn start(
        kind: PickerKind,
        owner: isize,
        remaining_slots: usize,
        remaining_text_bytes: usize,
    ) -> io::Result<Self> {
        let shared = Arc::new(Shared::default());
        let worker_shared = Arc::clone(&shared);
        let (sender, receiver) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("bonsai-attachment-import".to_owned())
            .spawn(move || {
                run_picker_import(
                    kind,
                    owner,
                    remaining_slots,
                    remaining_text_bytes,
                    &worker_shared,
                    &sender,
                );
            })?;
        Ok(Self {
            shared,
            receiver,
            worker: Some(worker),
            complete: false,
        })
    }

    pub(crate) fn cancel(&mut self) {
        self.shared.picker.cancel(&self.shared.cancelled);
    }

    pub(crate) fn poll(&mut self) -> Option<Outcome> {
        if self.complete {
            return None;
        }
        if self.worker.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(worker) = self.worker.take()
        {
            let _joined = worker.join();
        }
        self.shared.picker.reap_finished();
        let outcome = match self.receiver.try_recv() {
            Ok(outcome) => outcome,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => Outcome::Failed(
                "Attachment import stopped unexpectedly. Your draft was kept.".to_owned(),
            ),
        };
        self.complete = true;
        Some(outcome)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.shared.picker.cancel(&self.shared.cancelled);
        self.shared.picker.join();
        if let Some(worker) = self.worker.take() {
            let _joined = worker.join();
        }
    }
}

pub(crate) fn run_worker(arguments: &[OsString]) -> std::process::ExitCode {
    worker::run_worker(arguments)
}

pub(crate) fn classify_path(path: &Path) -> Option<FileCategory> {
    let extension = path.extension().and_then(std::ffi::OsStr::to_str)?;
    if TEXT_EXTENSIONS
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    {
        return Some(FileCategory::Text);
    }
    if IMAGE_EXTENSIONS
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    {
        return Some(FileCategory::Media(MediaKind::Image));
    }
    if VIDEO_EXTENSIONS
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    {
        return Some(FileCategory::Media(MediaKind::Video));
    }
    None
}

fn run_picker_import(
    kind: PickerKind,
    owner: isize,
    remaining_slots: usize,
    remaining_text_bytes: usize,
    shared: &Shared,
    sender: &SyncSender<Outcome>,
) {
    let outcome = picker_and_import(kind, owner, remaining_slots, remaining_text_bytes, shared);
    let _sent = sender.try_send(outcome);
}

fn picker_and_import(
    kind: PickerKind,
    owner: isize,
    remaining_slots: usize,
    remaining_text_bytes: usize,
    shared: &Shared,
) -> Outcome {
    if remaining_slots == 0 {
        return Outcome::Failed(format!(
            "Up to {MAX_ATTACHMENTS_PER_MESSAGE} attachments per message are supported. Remove one to add more."
        ));
    }
    let spec = match kind {
        PickerKind::Text => MultiDialogSpec {
            title: "Attach text or code files",
            ok_button_label: "Attach",
            filters: &TEXT_FILTERS,
        },
        PickerKind::Media => MultiDialogSpec {
            title: "Attach images or videos",
            ok_button_label: "Attach",
            filters: &MEDIA_FILTERS,
        },
    };
    let paths = match run_multi_open_modal(owner, &shared.picker, &shared.cancelled, &spec) {
        Ok(paths) => paths,
        Err(MultiDialogError::Cancelled) => return Outcome::Cancelled,
        Err(MultiDialogError::TooManyFiles { selected, max }) => {
            return Outcome::Failed(format!(
                "Select up to {max} files at a time ({selected} selected). Your draft was kept."
            ));
        }
        Err(MultiDialogError::Failed) => {
            return Outcome::Failed(
                "The file picker could not open. Your draft was kept.".to_owned(),
            );
        }
    };
    import_selected_paths(kind, &paths, remaining_slots, remaining_text_bytes, shared)
}

fn import_selected_paths(
    kind: PickerKind,
    paths: &[PathBuf],
    remaining_slots: usize,
    mut remaining_text_bytes: usize,
    shared: &Shared,
) -> Outcome {
    if paths.len() > remaining_slots {
        return Outcome::Failed(format!(
            "Only {remaining_slots} attachment slot(s) remain on this draft (selected {}).",
            paths.len()
        ));
    }
    let mut imported = Vec::with_capacity(paths.len());
    let mut first_error = None;
    let mut skipped_count = 0usize;
    for path in paths {
        if shared.cancelled.load(Ordering::Acquire) {
            return Outcome::Cancelled;
        }
        let result = import_one_path(kind, path, remaining_text_bytes, shared);
        match result {
            Ok(attachment) => {
                remaining_text_bytes = remaining_text_bytes.saturating_sub(attachment.text_bytes());
                imported.push(attachment);
            }
            Err(message) => {
                if first_error.is_none() {
                    first_error = Some(message);
                }
                skipped_count += 1;
            }
        }
    }
    if imported.is_empty() {
        return match first_error {
            Some(message) if shared.cancelled.load(Ordering::Acquire) => {
                let _ = message;
                Outcome::Cancelled
            }
            Some(message) => Outcome::Failed(message),
            None => Outcome::Cancelled,
        };
    }
    let notice = first_error.map(|error| {
        if skipped_count <= 1 {
            error
        } else {
            format!(
                "Added {} file(s); skipped {skipped_count}: {error}",
                imported.len()
            )
        }
    });
    Outcome::Imported {
        attachments: imported,
        notice,
    }
}

fn import_one_path(
    kind: PickerKind,
    path: &Path,
    remaining_text_bytes: usize,
    shared: &Shared,
) -> Result<Attachment, String> {
    let category = classify_path(path).ok_or_else(|| {
        format!(
            "Unsupported file type for \"{}\". Your draft was kept.",
            display_name(path)
        )
    })?;
    match (kind, category) {
        (PickerKind::Text, FileCategory::Text) => {
            text::import_text_file(path, remaining_text_bytes)
        }
        (PickerKind::Media, FileCategory::Media(media_kind)) => {
            isolated::import_media(path, media_kind, shared, IMPORT_TIMEOUT)
        }
        (PickerKind::Text, FileCategory::Media(_)) => Err(format!(
            "\"{}\" is a media file; choose Images and Videos to attach it.",
            display_name(path)
        )),
        (PickerKind::Media, FileCategory::Text) => Err(format!(
            "\"{}\" is a text file; choose Text files to attach it.",
            display_name(path)
        )),
    }
}

pub(crate) fn display_name(path: &Path) -> String {
    let raw = path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("attachment");
    let cleaned: String = raw
        .chars()
        .map(|ch| if ch.is_control() { '_' } else { ch })
        .take(128)
        .collect();
    if cleaned.is_empty() {
        "attachment".to_owned()
    } else {
        cleaned
    }
}

mod base64_bytes {
    use super::*;

    pub(super) fn serialize<S: Serializer>(
        bytes: &PrehashedPng,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&Base64::encode_string(bytes.as_ref()))
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<PrehashedPng, D::Error> {
        let encoded = String::deserialize(deserializer)?;
        let decoded = Base64::decode_vec(&encoded).map_err(D::Error::custom)?;
        Ok(PrehashedPng::from(decoded))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Attachment, FileCategory, MAX_TOTAL_TEXT_ATTACHMENT_BYTES, MediaKind, Outcome, PickerKind,
        PrehashedPng, Shared, VideoFrame, classify_path, format_user_turn, import_selected_paths,
    };
    use std::{fs, io, path::Path, sync::Arc};

    #[test]
    fn extension_classification_separates_text_image_and_video() {
        assert_eq!(
            classify_path(Path::new(r"C:\notes\README.MD")),
            Some(FileCategory::Text)
        );
        assert_eq!(
            classify_path(Path::new(r"C:\photos\frame.PNG")),
            Some(FileCategory::Media(MediaKind::Image))
        );
        assert_eq!(
            classify_path(Path::new(r"C:\clips\demo.mp4")),
            Some(FileCategory::Media(MediaKind::Video))
        );
        assert_eq!(classify_path(Path::new(r"C:\bin\app.exe")), None);
        assert_eq!(classify_path(Path::new(r"C:\audio\clip.wav")), None);
    }

    #[test]
    fn attachment_json_roundtrip_preserves_compact_base64_thumbnails() -> io::Result<()> {
        let sample = vec![
            Attachment::Text {
                name: "notes.md".to_owned(),
                text: "line 1\nline 2".to_owned(),
            },
            Attachment::Image {
                name: "photo.png".to_owned(),
                width: 640,
                height: 480,
                data_url: Arc::from("data:image/png;base64,iVBORw0KGgo="),
                thumbnail_png: PrehashedPng::from([0x89_u8, b'P', b'N', b'G']),
            },
            Attachment::Video {
                name: "clip.mp4".to_owned(),
                width: 768,
                height: 432,
                duration_ms: 4200,
                frames: Arc::from([VideoFrame {
                    timestamp_ms: 500,
                    data_url: Arc::from("data:image/png;base64,iVBORw0KGgo="),
                }]),
                thumbnail_png: PrehashedPng::from([0x89_u8, b'P', b'N', b'G']),
            },
        ];
        let json = serde_json::to_string(&sample).map_err(io::Error::other)?;
        assert!(json.contains("\"iVBORw==\""));
        let decoded: Vec<Attachment> = serde_json::from_str(&json).map_err(io::Error::other)?;
        assert_eq!(decoded, sample);
        Ok(())
    }

    #[test]
    fn selected_text_paths_enforce_slot_and_total_byte_budgets() -> io::Result<()> {
        let directory = std::env::temp_dir().join(format!(
            "bonsai-attach-sel-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&directory)?;
        let valid = directory.join("hello.txt");
        let ignored_media = directory.join("pic.png");
        fs::write(&valid, b"\xEF\xBB\xBFhello\r\nworld")?;
        fs::write(&ignored_media, b"not-used")?;

        let shared = Shared::default();
        let too_many = import_selected_paths(
            PickerKind::Text,
            &[valid.clone(), ignored_media.clone()],
            1,
            MAX_TOTAL_TEXT_ATTACHMENT_BYTES,
            &shared,
        );
        assert!(matches!(too_many, Outcome::Failed(_)));

        let partial = import_selected_paths(
            PickerKind::Text,
            &[valid, ignored_media],
            4,
            MAX_TOTAL_TEXT_ATTACHMENT_BYTES,
            &shared,
        );
        let Outcome::Imported {
            attachments,
            notice,
        } = partial
        else {
            return Err(io::Error::other(format!(
                "Expected partial import outcome, got {partial:?}"
            )));
        };
        assert_eq!(attachments.len(), 1);
        assert_eq!(
            attachments[0],
            Attachment::Text {
                name: "hello.txt".to_owned(),
                text: "hello\nworld".to_owned(),
            }
        );
        assert!(notice.is_some());
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn format_user_turn_builds_dynamic_text_fences_video_metadata_and_ordered_images() {
        assert_eq!(
            format_user_turn("Plain prompt", &[]),
            ("Plain prompt".to_owned(), Arc::default())
        );

        let image_only = Attachment::Image {
            name: "photo.png".to_owned(),
            width: 32,
            height: 32,
            data_url: Arc::from("data:image/png;base64,AAAA"),
            thumbnail_png: PrehashedPng::from([1_u8, 2]),
        };
        let (text, images) = format_user_turn("", std::slice::from_ref(&image_only));
        assert_eq!(text, "");
        assert_eq!(images.as_ref(), &[Arc::from("data:image/png;base64,AAAA")]);

        let fenced_text = Attachment::Text {
            name: "snippet.md".to_owned(),
            text: "before\n```rust\nlet x = 1;\n```\nafter".to_owned(),
        };
        let video = Attachment::Video {
            name: "demo.mp4".to_owned(),
            width: 64,
            height: 48,
            duration_ms: 2_500,
            frames: Arc::from([
                VideoFrame {
                    timestamp_ms: 0,
                    data_url: Arc::from("data:image/png;base64,BBBB"),
                },
                VideoFrame {
                    timestamp_ms: 1_540,
                    data_url: Arc::from("data:image/png;base64,CCCC"),
                },
            ]),
            thumbnail_png: PrehashedPng::from([3_u8]),
        };
        let (text, images) = format_user_turn("Inspect these", &[fenced_text, image_only, video]);
        assert_eq!(
            text,
            "File: snippet.md\n````\nbefore\n```rust\nlet x = 1;\n```\nafter\n````\n\nVideo: demo.mp4 (2 keyframes: 0.0s, 1.5s)\n\nInspect these"
        );
        assert_eq!(
            images.as_ref(),
            &[
                Arc::from("data:image/png;base64,AAAA"),
                Arc::from("data:image/png;base64,BBBB"),
                Arc::from("data:image/png;base64,CCCC"),
            ]
        );
    }
}
