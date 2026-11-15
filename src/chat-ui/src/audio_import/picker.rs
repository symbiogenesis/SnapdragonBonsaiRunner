//! Single-file Shell audio picker with agile cross-thread cancellation.
#![deny(unsafe_code)]

use std::{path::PathBuf, sync::atomic::AtomicBool};

pub(super) use crate::com_dialog::Cancel;
use crate::com_dialog::{DialogError, DialogMode, DialogSpec, FilterSpec, run_modal};

use super::{CANCELLED, CODEC_FAILURE};

const PICKER_UNAVAILABLE: &str =
    "The audio file picker could not open. Try pasting a copied file instead.";
const AUDIO_FILTERS: [FilterSpec<'static>; 1] = [FilterSpec {
    name: "Audio files (*.mp3;*.wav;*.m4a;*.aac;*.flac;*.wma)",
    pattern: "*.mp3;*.wav;*.m4a;*.aac;*.flac;*.wma",
}];
const OPEN_SPEC: DialogSpec<'static> = DialogSpec {
    mode: DialogMode::Open,
    title: "Add audio prompt",
    filters: &AUDIO_FILTERS,
};

pub(super) fn pick_file(
    owner: isize,
    cancel: &Cancel,
    cancelled: &AtomicBool,
) -> Result<Option<PathBuf>, &'static str> {
    let selection =
        run_modal(owner, cancel, cancelled, &OPEN_SPEC).map_err(|error| match error {
            DialogError::Cancelled => CANCELLED,
            DialogError::Unavailable => PICKER_UNAVAILABLE,
            DialogError::ResultInvalid => CODEC_FAILURE,
        })?;
    Ok(selection.map(|selected| selected.path))
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::Duration,
    };
    use windows::{
        Win32::UI::Shell::{
            FILEOPENDIALOGOPTIONS, FOS_ALLOWMULTISELECT, FOS_DONTADDTORECENT, FOS_FILEMUSTEXIST,
            FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR, FOS_PATHMUSTEXIST, FOS_STRICTFILETYPES,
        },
        core::HRESULT,
    };

    use super::{CANCELLED, Cancel, OPEN_SPEC, PICKER_UNAVAILABLE, pick_file};
    use crate::com_dialog::{
        ActiveDialog, Apartment, DialogError, ModalDialog, classify_picker_show, dialog_options,
    };

    const PICKER_CANCELLED: HRESULT =
        HRESULT::from_win32(windows::Win32::Foundation::ERROR_CANCELLED.0);

    #[test]
    fn picker_options_and_cancellation_classification_preserve_single_file_contract() {
        let options = dialog_options(
            FILEOPENDIALOGOPTIONS(FOS_ALLOWMULTISELECT.0 | FOS_STRICTFILETYPES.0),
            false,
        );
        assert!(options.contains(FOS_DONTADDTORECENT));
        assert!(options.contains(FOS_FILEMUSTEXIST));
        assert!(options.contains(FOS_FORCEFILESYSTEM));
        assert!(options.contains(FOS_NOCHANGEDIR));
        assert!(options.contains(FOS_PATHMUSTEXIST));
        assert!(options.contains(FOS_STRICTFILETYPES));
        assert!(!options.contains(FOS_ALLOWMULTISELECT));

        assert_eq!(classify_picker_show(Ok(()), false), Ok(true));
        assert_eq!(
            classify_picker_show(Err(windows::core::Error::from(PICKER_CANCELLED)), false),
            Ok(false)
        );
        assert_eq!(
            classify_picker_show(Ok(()), true),
            Err(DialogError::Cancelled)
        );
        assert_eq!(
            classify_picker_show(Err(windows::core::Error::from(PICKER_CANCELLED)), true),
            Err(DialogError::Cancelled)
        );
        assert_eq!(
            classify_picker_show(
                Err(windows::core::Error::from(HRESULT(-2_147_467_259))),
                false
            ),
            Err(DialogError::Unavailable)
        );
    }

    #[test]
    fn pre_cancelled_session_aborts_picker_before_opening_dialog() {
        let cancel = Cancel::default();
        let cancelled = AtomicBool::new(false);
        cancel.cancel(&cancelled);
        assert_eq!(pick_file(0, &cancel, &cancelled), Err(CANCELLED));
        cancel.join();
    }

    #[test]
    fn published_agile_dialog_cancels_and_joins_across_apartments() -> io::Result<()> {
        let state = Arc::new((Cancel::default(), AtomicBool::new(false)));
        let worker_state = Arc::clone(&state);
        let (published_tx, published_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("bonsai-audio-picker-test".to_owned())
            .spawn(move || -> Result<(), &'static str> {
                let apartment = Apartment::sta().map_err(|_| PICKER_UNAVAILABLE)?;
                let Ok(dialog) = ModalDialog::new(&apartment, &OPEN_SPEC) else {
                    let _sent = published_tx.send(false);
                    return Ok(());
                };
                let active = ActiveDialog::publish(&worker_state.0, &worker_state.1, &dialog)
                    .map_err(|_| CANCELLED)?;
                let _sent = published_tx.send(true);
                while !worker_state.1.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                drop(active);
                Ok(())
            })?;
        if published_rx
            .recv_timeout(Duration::from_secs(2))
            .map_err(io::Error::other)?
        {
            state.0.cancel(&state.1);
            // Duplicate cancellation must remain idempotent.
            state.0.cancel(&state.1);
        }
        let outcome = worker
            .join()
            .map_err(|_| io::Error::other("picker test worker panicked"))?;
        state.0.join();
        assert_eq!(outcome, Ok(()));
        Ok(())
    }
}
