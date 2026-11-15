//! Single-file Shell audio picker with agile cross-thread cancellation.
#![deny(unsafe_code)]

use std::{
    num::NonZeroIsize,
    path::PathBuf,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};
use windows::{
    Win32::{
        Foundation::ERROR_CANCELLED,
        UI::Shell::{
            FILEOPENDIALOGOPTIONS, FOS_ALLOWMULTISELECT, FOS_DONTADDTORECENT, FOS_FILEMUSTEXIST,
            FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR, FOS_PATHMUSTEXIST, IFileDialog,
        },
    },
    core::{AgileReference, HRESULT},
};

use super::{Apartment, CANCELLED, CODEC_FAILURE, POLL};

const PICKER_UNAVAILABLE: &str =
    "The audio file picker could not open. Try pasting a copied file instead.";
const PICKER_CANCELLED: HRESULT = HRESULT::from_win32(ERROR_CANCELLED.0);

#[derive(Default)]
pub(super) struct Cancel(Mutex<CancelState>);

#[derive(Default)]
struct CancelState {
    active: Option<(AgileReference<IFileDialog>, Arc<AtomicBool>)>,
    closer: Option<JoinHandle<()>>,
}

impl Cancel {
    pub(super) fn cancel(&self, cancelled: &AtomicBool) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        cancelled.store(true, Ordering::Release);
        let Some((dialog, showing)) = state.active.take() else {
            return;
        };
        if let Ok(closer) = thread::Builder::new()
            .name("bonsai-audio-picker-close".to_owned())
            .spawn(move || com::close_dialog(dialog, &showing))
        {
            state.closer = Some(closer);
        }
    }

    pub(super) fn reap_finished(&self) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if state.closer.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(closer) = state.closer.take()
        {
            let _joined = closer.join();
        }
    }

    pub(super) fn join(&self) {
        let closer = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .closer
            .take();
        if let Some(closer) = closer {
            let _joined = closer.join();
        }
    }
}

struct ActiveDialog<'a> {
    cancel: &'a Cancel,
    showing: Arc<AtomicBool>,
}

impl<'a> ActiveDialog<'a> {
    fn publish(
        cancel: &'a Cancel,
        cancelled: &AtomicBool,
        dialog: &com::OpenDialog<'_>,
    ) -> Result<Self, &'static str> {
        let agile = dialog.agile_handle()?;
        let showing = Arc::new(AtomicBool::new(true));
        let mut state = cancel.0.lock().unwrap_or_else(PoisonError::into_inner);
        if cancelled.load(Ordering::Acquire) {
            return Err(CANCELLED);
        }
        state.active = Some((agile, Arc::clone(&showing)));
        drop(state);
        Ok(Self { cancel, showing })
    }
}

impl Drop for ActiveDialog<'_> {
    fn drop(&mut self) {
        self.showing.store(false, Ordering::Release);
        let active = self
            .cancel
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .active
            .take();
        drop(active);
    }
}

pub(super) fn pick_file(
    owner: isize,
    cancel: &Cancel,
    cancelled: &AtomicBool,
) -> Result<Option<PathBuf>, &'static str> {
    if cancelled.load(Ordering::Acquire) {
        return Err(CANCELLED);
    }
    let apartment = Apartment::sta().map_err(|_| PICKER_UNAVAILABLE)?;
    let dialog = com::OpenDialog::new(&apartment)?;
    let active = ActiveDialog::publish(cancel, cancelled, &dialog)?;
    let show_result = dialog.show(NonZeroIsize::new(owner));
    drop(active);
    if !classify_picker_show(show_result, cancelled.load(Ordering::Acquire))? {
        return Ok(None);
    }
    dialog.selected_path().map(Some)
}

const fn picker_options(existing: FILEOPENDIALOGOPTIONS) -> FILEOPENDIALOGOPTIONS {
    FILEOPENDIALOGOPTIONS(
        (existing.0
            | FOS_DONTADDTORECENT.0
            | FOS_FILEMUSTEXIST.0
            | FOS_FORCEFILESYSTEM.0
            | FOS_NOCHANGEDIR.0
            | FOS_PATHMUSTEXIST.0)
            & !FOS_ALLOWMULTISELECT.0,
    )
}

fn classify_picker_show(
    result: windows::core::Result<()>,
    cancelled: bool,
) -> Result<bool, &'static str> {
    if cancelled {
        return Err(CANCELLED);
    }
    match result {
        Ok(()) => Ok(true),
        Err(error) if error.code() == PICKER_CANCELLED => Ok(false),
        Err(_) => Err(PICKER_UNAVAILABLE),
    }
}

#[allow(
    unsafe_code,
    reason = "Thin RAII boundary over COM IFileOpenDialog, IShellItem, and CoTaskMemFree."
)]
mod com {
    use std::{
        ffi::{OsString, c_void},
        num::NonZeroIsize,
        os::windows::ffi::OsStringExt as _,
        path::PathBuf,
        ptr,
        sync::atomic::{AtomicBool, Ordering},
        thread,
    };
    use windows::{
        Win32::{
            Foundation::HWND,
            System::Com::{CLSCTX_INPROC_SERVER, CoCreateInstance, CoTaskMemFree},
            UI::Shell::{
                Common::COMDLG_FILTERSPEC, FileOpenDialog, IFileDialog, IFileOpenDialog,
                IShellItem, SIGDN_FILESYSPATH,
            },
        },
        core::{AgileReference, Interface as _, PCWSTR, PWSTR, w},
    };

    use super::{
        Apartment, CODEC_FAILURE, PICKER_CANCELLED, PICKER_UNAVAILABLE, POLL, picker_options,
    };

    const DIALOG_TITLE: PCWSTR = w!("Add audio prompt");
    const AUDIO_FILTER: [COMDLG_FILTERSPEC; 1] = [COMDLG_FILTERSPEC {
        pszName: w!("Audio files (*.mp3;*.wav;*.m4a;*.aac;*.flac;*.wma)"),
        pszSpec: w!("*.mp3;*.wav;*.m4a;*.aac;*.flac;*.wma"),
    }];

    pub(super) struct OpenDialog<'a> {
        inner: IFileOpenDialog,
        _apartment: &'a Apartment,
    }

    impl<'a> OpenDialog<'a> {
        pub(super) fn new(apartment: &'a Apartment) -> Result<Self, &'static str> {
            // SAFETY: In-process COM activation on the caller's STA-initialized thread.
            let inner: IFileOpenDialog =
                unsafe { CoCreateInstance(&FileOpenDialog, None, CLSCTX_INPROC_SERVER) }
                    .map_err(|_| PICKER_UNAVAILABLE)?;
            // SAFETY: Valid COM dialog pointer on its owning STA thread.
            let existing = unsafe { inner.GetOptions() }.map_err(|_| PICKER_UNAVAILABLE)?;
            let options = picker_options(existing);
            // SAFETY: Applies sanitized single-file filesystem flags on the owning STA thread.
            unsafe { inner.SetOptions(options) }.map_err(|_| PICKER_UNAVAILABLE)?;
            // SAFETY: AUDIO_FILTER points to static NUL-terminated UTF-16 literals.
            unsafe { inner.SetFileTypes(&AUDIO_FILTER) }.map_err(|_| PICKER_UNAVAILABLE)?;
            // SAFETY: Selects the sole 1-based entry configured in AUDIO_FILTER.
            unsafe { inner.SetFileTypeIndex(1) }.map_err(|_| PICKER_UNAVAILABLE)?;
            // SAFETY: DIALOG_TITLE is a static NUL-terminated UTF-16 literal.
            unsafe { inner.SetTitle(DIALOG_TITLE) }.map_err(|_| PICKER_UNAVAILABLE)?;
            Ok(Self {
                inner,
                _apartment: apartment,
            })
        }

        pub(super) fn agile_handle(&self) -> Result<AgileReference<IFileDialog>, &'static str> {
            let base: IFileDialog = self.inner.cast().map_err(|_| PICKER_UNAVAILABLE)?;
            AgileReference::new(&base).map_err(|_| PICKER_UNAVAILABLE)
        }

        pub(super) fn show(&self, owner: Option<NonZeroIsize>) -> windows::core::Result<()> {
            let owner_hwnd = owner.map(|raw| {
                HWND(ptr::with_exposed_provenance_mut::<c_void>(
                    raw.get().cast_unsigned(),
                ))
            });
            // SAFETY: Modal dialog runs on its owning STA thread; owner is null or live window.
            unsafe { self.inner.Show(owner_hwnd) }
        }

        pub(super) fn selected_path(&self) -> Result<PathBuf, &'static str> {
            // SAFETY: Called only after Show succeeded with single-file FOS_FILEMUSTEXIST.
            let item: IShellItem = unsafe { self.inner.GetResult() }.map_err(|_| CODEC_FAILURE)?;
            // SAFETY: Valid IShellItem; CoTaskMemString frees the returned buffer on drop.
            let raw =
                unsafe { item.GetDisplayName(SIGDN_FILESYSPATH) }.map_err(|_| CODEC_FAILURE)?;
            CoTaskMemString(raw).into_path_buf()
        }
    }

    struct CoTaskMemString(PWSTR);

    impl CoTaskMemString {
        fn into_path_buf(self) -> Result<PathBuf, &'static str> {
            if self.0.is_null() {
                return Err(CODEC_FAILURE);
            }
            // SAFETY: Non-null NUL-terminated UTF-16 string owned by self until Drop.
            let wide = unsafe { self.0.as_wide() };
            if wide.is_empty() {
                return Err(CODEC_FAILURE);
            }
            Ok(PathBuf::from(OsString::from_wide(wide)))
        }
    }

    impl Drop for CoTaskMemString {
        fn drop(&mut self) {
            if !self.0.is_null() {
                let raw = self.0.0.cast();
                // SAFETY: Frees the buffer allocated by IShellItem::GetDisplayName.
                unsafe { CoTaskMemFree(Some(raw)) };
            }
        }
    }

    pub(super) fn close_dialog(dialog: AgileReference<IFileDialog>, showing: &AtomicBool) {
        let Ok(_apartment) = Apartment::mta() else {
            return;
        };
        let dialog = dialog;
        while showing.load(Ordering::Acquire) {
            if let Ok(resolved) = dialog.resolve() {
                // SAFETY: Marshaled IFileDialog proxy on an MTA worker; Close is
                // documented by Microsoft for cross-thread modal dismissal.
                let _closed = unsafe { resolved.Close(PICKER_CANCELLED) };
            }
            if showing.load(Ordering::Acquire) {
                thread::sleep(POLL);
            }
        }
    }
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
    use windows::{Win32::UI::Shell::FOS_STRICTFILETYPES, core::HRESULT};

    use super::{
        ActiveDialog, Apartment, CANCELLED, Cancel, FILEOPENDIALOGOPTIONS, FOS_ALLOWMULTISELECT,
        FOS_DONTADDTORECENT, FOS_FILEMUSTEXIST, FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR,
        FOS_PATHMUSTEXIST, PICKER_CANCELLED, PICKER_UNAVAILABLE, classify_picker_show, com,
        pick_file, picker_options,
    };

    #[test]
    fn picker_options_and_cancellation_classification_preserve_single_file_contract() {
        let options = picker_options(FILEOPENDIALOGOPTIONS(
            FOS_ALLOWMULTISELECT.0 | FOS_STRICTFILETYPES.0,
        ));
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
        assert_eq!(classify_picker_show(Ok(()), true), Err(CANCELLED));
        assert_eq!(
            classify_picker_show(Err(windows::core::Error::from(PICKER_CANCELLED)), true),
            Err(CANCELLED)
        );
        assert_eq!(
            classify_picker_show(
                Err(windows::core::Error::from(HRESULT(-2_147_467_259))),
                false
            ),
            Err(PICKER_UNAVAILABLE)
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
                let Ok(dialog) = com::OpenDialog::new(&apartment) else {
                    let _sent = published_tx.send(false);
                    return Ok(());
                };
                let active = ActiveDialog::publish(&worker_state.0, &worker_state.1, &dialog)?;
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
