//! Single-file Shell save picker with agile cross-thread cancellation.
#![deny(unsafe_code)]

use std::{
    ffi::OsString,
    num::NonZeroIsize,
    path::PathBuf,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use windows::{
    Win32::{
        Foundation::ERROR_CANCELLED,
        UI::Shell::{
            FILEOPENDIALOGOPTIONS, FOS_ALLOWMULTISELECT, FOS_DONTADDTORECENT, FOS_FORCEFILESYSTEM,
            FOS_NOCHANGEDIR, FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST, FOS_STRICTFILETYPES,
            IFileDialog,
        },
    },
    core::{AgileReference, HRESULT},
};

use crate::audio_import::Apartment;

pub(super) const CANCELLED: &str = "Image save cancelled.";
const PICKER_UNAVAILABLE: &str = "The image save dialog could not open.";
const DESTINATION_FAILED: &str = "Could not resolve the selected save destination.";
const PICKER_CANCELLED: HRESULT = HRESULT::from_win32(ERROR_CANCELLED.0);
const POLL: Duration = Duration::from_millis(10);
const DEFAULT_STEM: &str = "bonsai-image";
const MAX_STEM_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Format {
    Png,
    Svg,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Selection {
    pub(super) path: PathBuf,
    pub(super) format: Format,
}

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
            .name("bonsai-image-save-close".to_owned())
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
        dialog: &com::SaveDialog<'_>,
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

pub(super) fn pick_save_target(
    owner: isize,
    cancel: &Cancel,
    cancelled: &AtomicBool,
    suggested_stem: &str,
) -> Result<Option<Selection>, &'static str> {
    if cancelled.load(Ordering::Acquire) {
        return Err(CANCELLED);
    }
    let apartment = Apartment::sta().map_err(|_| PICKER_UNAVAILABLE)?;
    let dialog = com::SaveDialog::new(&apartment, suggested_stem)?;
    let active = ActiveDialog::publish(cancel, cancelled, &dialog)?;
    let show_result = dialog.show(NonZeroIsize::new(owner));
    drop(active);
    if !classify_picker_show(show_result, cancelled.load(Ordering::Acquire))? {
        return Ok(None);
    }
    dialog.selection().map(Some)
}

const fn picker_options(existing: FILEOPENDIALOGOPTIONS) -> FILEOPENDIALOGOPTIONS {
    FILEOPENDIALOGOPTIONS(
        (existing.0
            | FOS_DONTADDTORECENT.0
            | FOS_FORCEFILESYSTEM.0
            | FOS_NOCHANGEDIR.0
            | FOS_OVERWRITEPROMPT.0
            | FOS_PATHMUSTEXIST.0
            | FOS_STRICTFILETYPES.0)
            & !FOS_ALLOWMULTISELECT.0,
    )
}

fn sanitize_stem(raw: &str) -> String {
    let sanitized: String = raw
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_')
        .take(MAX_STEM_BYTES)
        .collect();
    if sanitized.is_empty() {
        DEFAULT_STEM.to_owned()
    } else {
        sanitized
    }
}

fn resolve_selection(path: PathBuf, type_index: u32) -> Result<Selection, &'static str> {
    if path.file_name().is_none() {
        return Err(DESTINATION_FAILED);
    }
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        if ext.eq_ignore_ascii_case("svg") {
            return Ok(Selection {
                path,
                format: Format::Svg,
            });
        }
        if ext.eq_ignore_ascii_case("png") {
            return Ok(Selection {
                path,
                format: Format::Png,
            });
        }
    }
    let (suffix, format) = if type_index == 2 {
        (".svg", Format::Svg)
    } else {
        (".png", Format::Png)
    };
    let mut name: OsString = path.into_os_string();
    name.push(suffix);
    Ok(Selection {
        path: PathBuf::from(name),
        format,
    })
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
    reason = "Thin RAII boundary over COM IFileSaveDialog, IShellItem, and CoTaskMemFree."
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
                Common::COMDLG_FILTERSPEC, FileSaveDialog, IFileDialog, IFileSaveDialog,
                IShellItem, SIGDN_FILESYSPATH,
            },
        },
        core::{AgileReference, Interface as _, PCWSTR, PWSTR, w},
    };

    use super::{
        Apartment, DESTINATION_FAILED, PICKER_CANCELLED, PICKER_UNAVAILABLE, POLL, Selection,
        picker_options, resolve_selection, sanitize_stem,
    };

    const DIALOG_TITLE: PCWSTR = w!("Save generated image");
    const DEFAULT_EXTENSION: PCWSTR = w!("png");
    const IMAGE_FILTERS: [COMDLG_FILTERSPEC; 2] = [
        COMDLG_FILTERSPEC {
            pszName: w!("PNG image (*.png)"),
            pszSpec: w!("*.png"),
        },
        COMDLG_FILTERSPEC {
            pszName: w!("SVG vector image (*.svg)"),
            pszSpec: w!("*.svg"),
        },
    ];

    pub(super) struct SaveDialog<'a> {
        inner: IFileSaveDialog,
        _apartment: &'a Apartment,
    }

    impl<'a> SaveDialog<'a> {
        pub(super) fn new(
            apartment: &'a Apartment,
            suggested_stem: &str,
        ) -> Result<Self, &'static str> {
            // SAFETY: In-process COM activation on the caller's STA-initialized thread.
            let inner: IFileSaveDialog =
                unsafe { CoCreateInstance(&FileSaveDialog, None, CLSCTX_INPROC_SERVER) }
                    .map_err(|_| PICKER_UNAVAILABLE)?;
            // SAFETY: Valid COM dialog pointer on its owning STA thread.
            let existing = unsafe { inner.GetOptions() }.map_err(|_| PICKER_UNAVAILABLE)?;
            let options = picker_options(existing);
            // SAFETY: Applies sanitized single-file filesystem save flags on the owning STA thread.
            unsafe { inner.SetOptions(options) }.map_err(|_| PICKER_UNAVAILABLE)?;
            // SAFETY: IMAGE_FILTERS contains static NUL-terminated UTF-16 filter literals.
            unsafe { inner.SetFileTypes(&IMAGE_FILTERS) }.map_err(|_| PICKER_UNAVAILABLE)?;
            // SAFETY: Selects the 1-based PNG entry configured in IMAGE_FILTERS.
            unsafe { inner.SetFileTypeIndex(1) }.map_err(|_| PICKER_UNAVAILABLE)?;
            // SAFETY: DEFAULT_EXTENSION is a static NUL-terminated UTF-16 literal.
            unsafe { inner.SetDefaultExtension(DEFAULT_EXTENSION) }
                .map_err(|_| PICKER_UNAVAILABLE)?;
            // SAFETY: DIALOG_TITLE is a static NUL-terminated UTF-16 literal.
            unsafe { inner.SetTitle(DIALOG_TITLE) }.map_err(|_| PICKER_UNAVAILABLE)?;
            let default_name: Vec<u16> = format!("{}.png", sanitize_stem(suggested_stem))
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect();
            // SAFETY: `default_name` stays alive and NUL-terminated for the synchronous COM call.
            unsafe { inner.SetFileName(PCWSTR(default_name.as_ptr())) }
                .map_err(|_| PICKER_UNAVAILABLE)?;
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

        pub(super) fn selection(&self) -> Result<Selection, &'static str> {
            // SAFETY: Valid COM dialog pointer on its owning STA thread after Show succeeded.
            let type_index = unsafe { self.inner.GetFileTypeIndex() }.unwrap_or(1);
            // SAFETY: Called only after Show succeeded.
            let item: IShellItem =
                unsafe { self.inner.GetResult() }.map_err(|_| DESTINATION_FAILED)?;
            // SAFETY: Valid IShellItem; CoTaskMemString frees the returned buffer on drop.
            let raw = unsafe { item.GetDisplayName(SIGDN_FILESYSPATH) }
                .map_err(|_| DESTINATION_FAILED)?;
            let path = CoTaskMemString(raw).into_path_buf()?;
            resolve_selection(path, type_index)
        }
    }

    struct CoTaskMemString(PWSTR);

    impl CoTaskMemString {
        fn into_path_buf(self) -> Result<PathBuf, &'static str> {
            if self.0.is_null() {
                return Err(DESTINATION_FAILED);
            }
            // SAFETY: Non-null NUL-terminated UTF-16 string owned by self until Drop.
            let wide = unsafe { self.0.as_wide() };
            if wide.is_empty() {
                return Err(DESTINATION_FAILED);
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
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::Duration,
    };
    use windows::{
        Win32::UI::Shell::{FILEOPENDIALOGOPTIONS, FOS_ALLOWMULTISELECT, FOS_FILEMUSTEXIST},
        core::HRESULT,
    };

    use super::{
        ActiveDialog, Apartment, CANCELLED, Cancel, DEFAULT_STEM, FOS_DONTADDTORECENT,
        FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR, FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST,
        FOS_STRICTFILETYPES, Format, PICKER_CANCELLED, PICKER_UNAVAILABLE, Selection,
        classify_picker_show, com, pick_save_target, picker_options, resolve_selection,
        sanitize_stem,
    };

    #[test]
    fn save_picker_options_stem_sanitization_and_selection_resolution() {
        let options = picker_options(FILEOPENDIALOGOPTIONS(FOS_ALLOWMULTISELECT.0));
        assert!(options.contains(FOS_DONTADDTORECENT));
        assert!(options.contains(FOS_FORCEFILESYSTEM));
        assert!(options.contains(FOS_NOCHANGEDIR));
        assert!(options.contains(FOS_OVERWRITEPROMPT));
        assert!(options.contains(FOS_PATHMUSTEXIST));
        assert!(options.contains(FOS_STRICTFILETYPES));
        assert!(!options.contains(FOS_FILEMUSTEXIST));
        assert!(!options.contains(FOS_ALLOWMULTISELECT));

        assert_eq!(sanitize_stem("bonsai/chart: #1?"), "bonsaichart1");
        assert_eq!(sanitize_stem("///"), DEFAULT_STEM);
        assert_eq!(sanitize_stem(&"a".repeat(100)).len(), 64);

        assert_eq!(
            resolve_selection(PathBuf::from(r"C:\tmp\chart"), 1),
            Ok(Selection {
                path: PathBuf::from(r"C:\tmp\chart.png"),
                format: Format::Png,
            })
        );
        assert_eq!(
            resolve_selection(PathBuf::from(r"C:\tmp\chart"), 2),
            Ok(Selection {
                path: PathBuf::from(r"C:\tmp\chart.svg"),
                format: Format::Svg,
            })
        );
        assert_eq!(
            resolve_selection(PathBuf::from(r"C:\tmp\chart.SVG"), 1),
            Ok(Selection {
                path: PathBuf::from(r"C:\tmp\chart.SVG"),
                format: Format::Svg,
            })
        );
        assert_eq!(
            resolve_selection(PathBuf::from(r"C:\tmp\chart.v1"), 1),
            Ok(Selection {
                path: PathBuf::from(r"C:\tmp\chart.v1.png"),
                format: Format::Png,
            })
        );

        assert_eq!(classify_picker_show(Ok(()), false), Ok(true));
        assert_eq!(
            classify_picker_show(Err(windows::core::Error::from(PICKER_CANCELLED)), false),
            Ok(false)
        );
        assert_eq!(classify_picker_show(Ok(()), true), Err(CANCELLED));
        assert_eq!(
            classify_picker_show(
                Err(windows::core::Error::from(HRESULT(-2_147_467_259))),
                false,
            ),
            Err(PICKER_UNAVAILABLE)
        );
    }

    #[test]
    fn pre_cancelled_save_aborts_before_opening_dialog() {
        let cancel = Cancel::default();
        let cancelled = AtomicBool::new(false);
        cancel.cancel(&cancelled);
        assert_eq!(
            pick_save_target(0, &cancel, &cancelled, "chart-1"),
            Err(CANCELLED)
        );
        cancel.join();
    }

    #[test]
    fn published_agile_save_dialog_cancels_and_joins_across_apartments() -> io::Result<()> {
        let state = Arc::new((Cancel::default(), AtomicBool::new(false)));
        let worker_state = Arc::clone(&state);
        let (published_tx, published_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("bonsai-image-save-test".to_owned())
            .spawn(move || -> Result<(), &'static str> {
                let apartment = Apartment::sta().map_err(|_| PICKER_UNAVAILABLE)?;
                let Ok(dialog) = com::SaveDialog::new(&apartment, "bonsai-image-1") else {
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
            .map_err(|_| io::Error::other("save picker test worker panicked"))?;
        state.0.join();
        assert_eq!(outcome, Ok(()));
        Ok(())
    }
}
