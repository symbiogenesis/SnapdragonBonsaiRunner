//! Encrypted device-wide preferences, independent of conversation saving.
use super::{State, VOICE_LIMIT, coalescing_store::CoalescingStore};
use serde::{Deserialize, Serialize};
use std::{io, path::Path, time::Duration};

const MAX_BYTES: usize = 8_192;
const MAX_SEALED: u64 = 73_728;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    version: u32,
    voice_profile: String,
}

pub(super) struct Opened {
    store: Store,
    saved: Option<String>,
    load_failed: bool,
}

pub(super) struct Store(CoalescingStore<String>);

impl Store {
    pub(super) fn open(data: &Path) -> io::Result<Opened> {
        let path = data.join("chat-profile.dpapi");
        bonsai_windows_credentials::validate_external_path(&path)?;
        let (saved, load_failed) = read(&path).map_or((None, true), |saved| (saved, false));
        let store = CoalescingStore::spawn(
            "swift-chat-profile",
            "Profile saving worker stopped.",
            saved.clone(),
            Some(Duration::from_secs(3)),
            move |voice| save(&path, &voice),
        )?;
        Ok(Opened {
            store: Self(store),
            saved,
            load_failed,
        })
    }

    pub(super) fn queue(&mut self, voice: &str) -> io::Result<()> {
        if voice.chars().count() > VOICE_LIMIT {
            return Err(io::Error::other("Voice profile exceeds its size limit."));
        }
        self.0.queue_by(
            voice,
            |latest, candidate| latest == candidate,
            str::to_owned,
        )
    }
}

impl State {
    pub(super) fn attach_profile(&mut self, opened: io::Result<Opened>) {
        match opened {
            Ok(mut opened) => {
                if let Some(voice) = opened.saved {
                    self.voice_profile = voice;
                } else if opened.load_failed {
                    // Establish a baseline without replacing unreadable bytes.
                    // Only a subsequent explicit voice edit may replace them.
                    opened.store.0.set_latest(Some(self.voice_profile.clone()));
                    self.record("Could not restore your voice profile. The existing file is preserved until you change the voice in Settings.");
                }
                // Missing profiles migrate the voice restored from older workspaces.
                self.profile_store = Some(opened.store);
                self.poll_profile();
            }
            Err(_) => self.record("Could not open profile preferences. Voice changes cannot be saved for the next launch."),
        }
    }

    pub(super) fn poll_profile(&mut self) {
        let Some(store) = &mut self.profile_store else {
            return;
        };
        let failed = store.0.take_error();
        if store.queue(&self.voice_profile).is_err() || failed {
            self.record(
                "Could not save your voice profile. Your conversations are saved separately.",
            );
        }
    }
}

fn read(path: &Path) -> io::Result<Option<String>> {
    let Some(file) = crate::bounded_file::open_regular_optional(path)? else {
        return Ok(None);
    };
    let sealed = crate::bounded_file::read(file, MAX_SEALED)?
        .ok_or_else(|| io::Error::other("Saved profile exceeds its storage bound."))?;
    let bytes = crate::scrub::ScrubbingBytes::new(
        bonsai_windows_credentials::unprotect_private_data(&sealed)?,
    );
    if bytes.len() > MAX_BYTES {
        return Err(io::Error::other("Saved profile exceeds its size limit."));
    }
    let saved: Saved = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
    if saved.version != 1 || saved.voice_profile.chars().count() > VOICE_LIMIT {
        return Err(io::Error::other("Invalid saved profile."));
    }
    Ok(Some(saved.voice_profile))
}

fn save(path: &Path, voice: &str) -> io::Result<()> {
    bonsai_windows_credentials::validate_external_path(path)?;
    let bytes = crate::scrub::ScrubbingBytes::new(
        serde_json::to_vec(&Saved {
            version: 1,
            voice_profile: voice.to_owned(),
        })
        .map_err(io::Error::other)?,
    );
    if voice.chars().count() > VOICE_LIMIT || bytes.len() > MAX_BYTES {
        return Err(io::Error::other("Voice profile exceeds its size limit."));
    }
    let sealed = bonsai_windows_credentials::protect_private_data(&bytes)?;
    super::atomic_file::replace(path, &sealed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn fixture() -> io::Result<std::path::PathBuf> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("Bonsai-profile-{stamp}"));
        fs::create_dir(&directory)?;
        Ok(directory)
    }

    fn state() -> State {
        let (commands, _incoming) = std::sync::mpsc::channel();
        let (_events, incoming) = crate::client::event_channel();
        let mut state = State::create(commands, incoming, None, None);
        state.preview = false;
        state
    }

    #[test]
    fn profile_voice_survives_restart_with_conversation_saving_off() -> io::Result<()> {
        let directory = fixture()?;
        let mut original = state();
        original.attach_recovery(crate::recovery::Recovery::open(&directory));
        original.attach_profile(Store::open(&directory));
        original.voice_profile = "A quiet, clear voice 雨 🌱".to_owned();
        let chosen = original.voice_profile.clone();
        original.input = "Do not save this conversation".to_owned();
        original.save_conversations = false;
        original.recovery_erase_pending = true;
        assert!(!original.save_conversations);
        drop(original);
        let path = directory.join("chat-profile.dpapi");
        assert!(
            !fs::read(&path)?
                .windows(chosen.len())
                .any(|bytes| bytes == chosen.as_bytes())
        );
        let mut restored = state();
        restored.attach_recovery(crate::recovery::Recovery::open(&directory));
        restored.attach_profile(Store::open(&directory));
        assert_eq!(restored.voice_profile, chosen);
        assert_eq!(restored.input, "");
        assert!(!restored.save_conversations && !restored.speaking && !restored.listening);
        assert!(!directory.join("chat-recovery.dpapi").exists());
        drop(restored);
        fs::remove_file(path)?;
        fs::remove_file(directory.join("chat-recovery.lock"))?;
        fs::remove_dir(directory)
    }

    #[test]
    fn profile_migrates_legacy_voice_and_takes_precedence_over_recovery() -> io::Result<()> {
        let directory = fixture()?;
        let mut legacy = state();
        legacy.attach_recovery(crate::recovery::Recovery::open(&directory));
        legacy.voice_profile = "Voice restored from the legacy workspace".to_owned();
        drop(legacy);
        let mut migrated = state();
        migrated.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert_eq!(
            migrated.voice_profile,
            "Voice restored from the legacy workspace"
        );
        migrated.attach_profile(Store::open(&directory));
        drop(migrated);
        // An older application may still write its historical workspace field.
        let mut stale = state();
        stale.attach_recovery(crate::recovery::Recovery::open(&directory));
        stale.voice_profile = "Older recovered workspace voice".to_owned();
        drop(stale);
        let mut restored = state();
        restored.attach_recovery(crate::recovery::Recovery::open(&directory));
        assert_eq!(restored.voice_profile, "Older recovered workspace voice");
        restored.attach_profile(Store::open(&directory));
        assert_eq!(
            restored.voice_profile,
            "Voice restored from the legacy workspace"
        );
        drop(restored);
        fs::remove_file(directory.join("chat-profile.dpapi"))?;
        fs::remove_file(directory.join("chat-recovery.dpapi"))?;
        fs::remove_file(directory.join("chat-recovery.lock"))?;
        fs::remove_dir(directory)
    }

    #[test]
    fn profile_preserves_corrupt_bytes_until_a_voice_edit() -> io::Result<()> {
        let directory = fixture()?;
        let path = directory.join("chat-profile.dpapi");
        fs::write(&path, b"unreadable profile")?;
        let mut unchanged = state();
        unchanged.attach_profile(Store::open(&directory));
        unchanged.poll_profile();
        drop(unchanged);
        assert_eq!(fs::read(&path)?, b"unreadable profile");
        let mut changed = state();
        changed.attach_profile(Store::open(&directory));
        changed.voice_profile = "An explicitly changed voice".to_owned();
        drop(changed);
        assert_eq!(read(&path)?.as_deref(), Some("An explicitly changed voice"));
        fs::remove_file(path)?;
        fs::remove_dir(directory)
    }
}
