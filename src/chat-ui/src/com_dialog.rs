//! Shared WinRT/COM apartment, `CoTaskMemFree` string, and modal `IFileDialog` boundaries.

use std::{
    ffi::{OsString, c_void},
    marker::PhantomData,
    num::NonZeroIsize,
    os::windows::ffi::OsStringExt as _,
    path::PathBuf,
    ptr,
    rc::Rc,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use windows::{
    Win32::{
        Foundation::{ERROR_CANCELLED, HWND, RPC_E_CHANGED_MODE},
        System::{
            Com::{
                CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx,
                CoTaskMemFree, CoUninitialize,
            },
            WinRT::{RO_INIT_MULTITHREADED, RO_INIT_SINGLETHREADED, RoInitialize, RoUninitialize},
        },
        UI::Shell::{
            Common::COMDLG_FILTERSPEC, FILEOPENDIALOGOPTIONS, FOS_ALLOWMULTISELECT,
            FOS_DONTADDTORECENT, FOS_FILEMUSTEXIST, FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR,
            FOS_NODEREFERENCELINKS, FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST, FOS_STRICTFILETYPES,
            FileOpenDialog, FileSaveDialog, IFileDialog, IFileOpenDialog, IShellItem,
            IShellItemArray, SIGDN_FILESYSPATH,
        },
    },
    core::{AgileReference, Error, HRESULT, Interface as _, PCWSTR, PWSTR},
};

/// Maximum number of files accepted from a single multi-file picker selection.
pub(crate) const MAX_PICKED_FILES: usize = 8;

const PICKER_CANCELLED: HRESULT = HRESULT::from_win32(ERROR_CANCELLED.0);
const POLL: Duration = Duration::from_millis(10);

enum Uninit {
    WinRt,
    Com,
    None,
}

/// RAII apartment guard for worker and UI threads entering WinRT or COM.
pub(crate) struct Apartment {
    uninit: Uninit,
    _not_send: PhantomData<Rc<()>>,
}

impl Apartment {
    /// Initializes a single-threaded WinRT/COM apartment owned by this guard.
    pub(crate) fn sta() -> windows::core::Result<Self> {
        // SAFETY: This thread owns its single-threaded apartment until Drop.
        unsafe { RoInitialize(RO_INIT_SINGLETHREADED) }?;
        Ok(Self {
            uninit: Uninit::WinRt,
            _not_send: PhantomData,
        })
    }

    /// Initializes a multithreaded WinRT/COM apartment owned by this guard.
    pub(crate) fn mta() -> windows::core::Result<Self> {
        // SAFETY: This thread owns its multithreaded apartment until Drop.
        unsafe { RoInitialize(RO_INIT_MULTITHREADED) }?;
        Ok(Self {
            uninit: Uninit::WinRt,
            _not_send: PhantomData,
        })
    }

    /// Enters a multithreaded COM apartment, borrowing an existing apartment on mode change.
    pub(crate) fn com_mta_or_existing() -> windows::core::Result<Self> {
        // SAFETY: Initializes COM on the caller thread; balanced by CoUninitialize when owned.
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if hr.is_ok() {
            Ok(Self {
                uninit: Uninit::Com,
                _not_send: PhantomData,
            })
        } else if hr == RPC_E_CHANGED_MODE {
            Ok(Self {
                uninit: Uninit::None,
                _not_send: PhantomData,
            })
        } else {
            Err(Error::from(hr))
        }
    }
}

impl Drop for Apartment {
    fn drop(&mut self) {
        match self.uninit {
            Uninit::WinRt => {
                // SAFETY: Matches the successful `RoInitialize` on this same thread.
                unsafe { RoUninitialize() };
            }
            Uninit::Com => {
                // SAFETY: Matches the successful `CoInitializeEx` on this same thread.
                unsafe { CoUninitialize() };
            }
            Uninit::None => {}
        }
    }
}

/// Owned UTF-16 `CoTaskMemAlloc` string freed automatically via `CoTaskMemFree`.
pub(crate) struct CoTaskMemString(PWSTR);

impl CoTaskMemString {
    /// Wraps a `PWSTR` returned by a COM/Shell API.
    pub(crate) const fn from_pwstr(raw: PWSTR) -> Self {
        Self(raw)
    }

    /// Converts a non-null, non-empty string into a `PathBuf` before freeing it.
    pub(crate) fn into_path_buf(self) -> Option<PathBuf> {
        if self.0.is_null() {
            return None;
        }
        // SAFETY: Non-null NUL-terminated UTF-16 string owned by self until Drop.
        let wide = unsafe { self.0.as_wide() };
        if wide.is_empty() {
            return None;
        }
        Some(PathBuf::from(OsString::from_wide(wide)))
    }
}

impl Drop for CoTaskMemString {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let raw = self.0.0.cast();
            // SAFETY: Frees the buffer allocated by COM/Shell for this owner.
            unsafe { CoTaskMemFree(Some(raw)) };
        }
    }
}

/// Classification of modal file dialog failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DialogError {
    /// Operation was cancelled via `Cancel` before or during modal presentation.
    Cancelled,
    /// Dialog could not be initialized, configured, or shown.
    Unavailable,
    /// Selected Shell item path could not be retrieved after acceptance.
    ResultInvalid,
}

/// File filter tuple passed to `IFileDialog::SetFileTypes`.
pub(crate) struct FilterSpec<'a> {
    pub(crate) name: &'a str,
    pub(crate) pattern: &'a str,
}

/// Open versus Save configuration for a modal `IFileDialog`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DialogMode<'a> {
    Open,
    Save {
        default_extension: &'a str,
        default_file_name: &'a str,
    },
}

/// Declarative configuration for a single-file modal Shell dialog.
pub(crate) struct DialogSpec<'a> {
    pub(crate) mode: DialogMode<'a>,
    pub(crate) title: &'a str,
    pub(crate) filters: &'a [FilterSpec<'a>],
}

/// Declarative configuration for a multi-file modal Open dialog.
pub(crate) struct MultiDialogSpec<'a> {
    pub(crate) title: &'a str,
    pub(crate) ok_button_label: &'a str,
    pub(crate) filters: &'a [FilterSpec<'a>],
}

/// Classification of multi-file modal Open dialog failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MultiDialogError {
    Cancelled,
    TooManyFiles { selected: usize, max: usize },
    Failed,
}

/// Accepted path and 1-based selected filter index returned by `run_modal`.
pub(crate) struct DialogSelection {
    pub(crate) path: PathBuf,
    pub(crate) file_type_index: u32,
}

/// Thread-safe handle for dismissing an active modal `IFileDialog` across apartments.
#[derive(Default)]
pub(crate) struct Cancel(Mutex<CancelState>);

#[derive(Default)]
struct CancelState {
    active: Option<(AgileReference<IFileDialog>, Arc<AtomicBool>)>,
    closer: Option<JoinHandle<()>>,
}

impl Cancel {
    /// Marks the dialog cancelled and spawns an MTA closer if a modal dialog is active.
    pub(crate) fn cancel(&self, cancelled: &AtomicBool) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        cancelled.store(true, Ordering::Release);
        let Some((dialog, showing)) = state.active.take() else {
            return;
        };
        if let Ok(closer) = thread::Builder::new()
            .name("bonsai-com-dialog-close".to_owned())
            .spawn(move || close_dialog(dialog, &showing))
        {
            state.closer = Some(closer);
        }
    }

    /// Reaps a finished cross-thread closer handle without blocking.
    pub(crate) fn reap_finished(&self) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if state.closer.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(closer) = state.closer.take()
        {
            let _joined = closer.join();
        }
    }

    /// Waits for any spawned closer thread to exit.
    pub(crate) fn join(&self) {
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

/// Guard publishing an active modal `IFileDialog` for cross-thread cancellation.
pub(crate) struct ActiveDialog<'a> {
    cancel: &'a Cancel,
    showing: Arc<AtomicBool>,
}

impl<'a> ActiveDialog<'a> {
    /// Registers `dialog` with `cancel` unless `cancelled` is already set.
    pub(crate) fn publish(
        cancel: &'a Cancel,
        cancelled: &AtomicBool,
        dialog: &ModalDialog<'_>,
    ) -> Result<Self, DialogError> {
        let agile = dialog.agile_handle()?;
        let showing = Arc::new(AtomicBool::new(true));
        let mut state = cancel.0.lock().unwrap_or_else(PoisonError::into_inner);
        if cancelled.load(Ordering::Acquire) {
            return Err(DialogError::Cancelled);
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

/// STA-owned COM `IFileDialog` configured for either opening or saving one filesystem file.
pub(crate) struct ModalDialog<'a> {
    inner: IFileDialog,
    _apartment: &'a Apartment,
}

fn publish_and_show(
    owner: isize,
    cancel: &Cancel,
    cancelled: &AtomicBool,
    dialog: &ModalDialog<'_>,
) -> Result<bool, DialogError> {
    let active = ActiveDialog::publish(cancel, cancelled, dialog)?;
    let show_result = dialog.show(NonZeroIsize::new(owner));
    drop(active);
    classify_picker_show(show_result, cancelled.load(Ordering::Acquire))
}

fn shell_item_path(item: &IShellItem) -> Option<PathBuf> {
    // SAFETY: Valid IShellItem; CoTaskMemString frees the returned buffer on drop.
    let raw = unsafe { item.GetDisplayName(SIGDN_FILESYSPATH) }.ok()?;
    CoTaskMemString::from_pwstr(raw).into_path_buf()
}

impl<'a> ModalDialog<'a> {
    /// Instantiates and configures a modal `IFileDialog` on `apartment`.
    pub(crate) fn new(
        apartment: &'a Apartment,
        spec: &DialogSpec<'_>,
    ) -> Result<Self, DialogError> {
        let clsid = match spec.mode {
            DialogMode::Open => &FileOpenDialog,
            DialogMode::Save { .. } => &FileSaveDialog,
        };
        // SAFETY: In-process COM activation on the caller's STA-initialized thread.
        let inner: IFileDialog = unsafe { CoCreateInstance(clsid, None, CLSCTX_INPROC_SERVER) }
            .map_err(|_| DialogError::Unavailable)?;
        // SAFETY: Valid COM dialog pointer on its owning STA thread.
        let existing = unsafe { inner.GetOptions() }.map_err(|_| DialogError::Unavailable)?;
        let options = dialog_options(existing, matches!(spec.mode, DialogMode::Save { .. }));
        // SAFETY: Applies sanitized single-file filesystem flags on the owning STA thread.
        unsafe { inner.SetOptions(options) }.map_err(|_| DialogError::Unavailable)?;
        configure_filters_and_title(&inner, spec.filters, spec.title)?;
        if let DialogMode::Save {
            default_extension,
            default_file_name,
        } = spec.mode
        {
            let ext = encode_wide_nul(default_extension);
            // SAFETY: `ext` remains alive and NUL-terminated during SetDefaultExtension.
            unsafe { inner.SetDefaultExtension(PCWSTR(ext.as_ptr())) }
                .map_err(|_| DialogError::Unavailable)?;
            let name = encode_wide_nul(default_file_name);
            // SAFETY: `name` remains alive and NUL-terminated during SetFileName.
            unsafe { inner.SetFileName(PCWSTR(name.as_ptr())) }
                .map_err(|_| DialogError::Unavailable)?;
        }
        Ok(Self {
            inner,
            _apartment: apartment,
        })
    }

    fn agile_handle(&self) -> Result<AgileReference<IFileDialog>, DialogError> {
        AgileReference::new(&self.inner).map_err(|_| DialogError::Unavailable)
    }

    fn show(&self, owner: Option<NonZeroIsize>) -> windows::core::Result<()> {
        let owner_hwnd = owner.map(|raw| {
            HWND(ptr::without_provenance_mut::<c_void>(
                raw.get().cast_unsigned(),
            ))
        });
        // SAFETY: Modal dialog runs on its owning STA thread; owner is null or live window.
        unsafe { self.inner.Show(owner_hwnd) }
    }

    fn selection(&self) -> Result<DialogSelection, DialogError> {
        // SAFETY: Valid COM dialog pointer on its owning STA thread after Show succeeded.
        let file_type_index = unsafe { self.inner.GetFileTypeIndex() }.unwrap_or(1);
        // SAFETY: Called only after Show succeeded.
        let item: IShellItem =
            unsafe { self.inner.GetResult() }.map_err(|_| DialogError::ResultInvalid)?;
        let path = shell_item_path(&item).ok_or(DialogError::ResultInvalid)?;
        Ok(DialogSelection {
            path,
            file_type_index,
        })
    }
}

/// Runs a modal Open or Save `IFileDialog` on a fresh STA apartment with cross-thread cancel.
pub(crate) fn run_modal(
    owner: isize,
    cancel: &Cancel,
    cancelled: &AtomicBool,
    spec: &DialogSpec<'_>,
) -> Result<Option<DialogSelection>, DialogError> {
    if cancelled.load(Ordering::Acquire) {
        return Err(DialogError::Cancelled);
    }
    let apartment = Apartment::sta().map_err(|_| DialogError::Unavailable)?;
    let dialog = ModalDialog::new(&apartment, spec)?;
    if !publish_and_show(owner, cancel, cancelled, &dialog)? {
        return Ok(None);
    }
    dialog.selection().map(Some)
}

/// Computes sanitized single-file filesystem options for Open or Save Shell dialogs.
pub(crate) const fn dialog_options(
    existing: FILEOPENDIALOGOPTIONS,
    is_save: bool,
) -> FILEOPENDIALOGOPTIONS {
    let specific = if is_save {
        FOS_OVERWRITEPROMPT.0 | FOS_STRICTFILETYPES.0
    } else {
        FOS_FILEMUSTEXIST.0
    };
    FILEOPENDIALOGOPTIONS(
        (existing.0
            | FOS_DONTADDTORECENT.0
            | FOS_FORCEFILESYSTEM.0
            | FOS_NOCHANGEDIR.0
            | FOS_PATHMUSTEXIST.0
            | specific)
            & !FOS_ALLOWMULTISELECT.0,
    )
}

/// Computes sanitized multi-file Open filesystem options while rejecting link dereferencing.
pub(crate) const fn multi_open_dialog_options(
    existing: FILEOPENDIALOGOPTIONS,
) -> FILEOPENDIALOGOPTIONS {
    FILEOPENDIALOGOPTIONS(
        existing.0
            | FOS_ALLOWMULTISELECT.0
            | FOS_DONTADDTORECENT.0
            | FOS_FILEMUSTEXIST.0
            | FOS_FORCEFILESYSTEM.0
            | FOS_NOCHANGEDIR.0
            | FOS_NODEREFERENCELINKS.0
            | FOS_PATHMUSTEXIST.0,
    )
}

fn validate_picked_count(count: u32) -> Result<u32, MultiDialogError> {
    if count == 0 {
        return Err(MultiDialogError::Cancelled);
    }
    let selected = usize::try_from(count).unwrap_or(usize::MAX);
    if selected > MAX_PICKED_FILES {
        return Err(MultiDialogError::TooManyFiles {
            selected,
            max: MAX_PICKED_FILES,
        });
    }
    Ok(count)
}

/// Runs a modal multi-file `IFileOpenDialog` on a fresh STA apartment with cross-thread cancel.
pub(crate) fn run_multi_open_modal(
    owner: isize,
    cancel: &Cancel,
    cancelled: &AtomicBool,
    spec: &MultiDialogSpec<'_>,
) -> Result<Vec<PathBuf>, MultiDialogError> {
    if cancelled.load(Ordering::Acquire) {
        return Err(MultiDialogError::Cancelled);
    }
    let apartment = Apartment::sta().map_err(|_| MultiDialogError::Failed)?;
    // SAFETY: In-process COM activation of IFileOpenDialog on the caller's STA thread.
    let open_dialog: IFileOpenDialog =
        unsafe { CoCreateInstance(&FileOpenDialog, None, CLSCTX_INPROC_SERVER) }
            .map_err(|_| MultiDialogError::Failed)?;
    let base: IFileDialog = open_dialog.cast().map_err(|_| MultiDialogError::Failed)?;
    // SAFETY: Valid IFileDialog interface on its owning STA thread.
    let existing = unsafe { base.GetOptions() }.map_err(|_| MultiDialogError::Failed)?;
    let options = multi_open_dialog_options(existing);
    // SAFETY: Applies multi-file filesystem flags on the owning STA thread.
    unsafe { base.SetOptions(options) }.map_err(|_| MultiDialogError::Failed)?;
    configure_filters_and_title(&base, spec.filters, spec.title)
        .map_err(|_| MultiDialogError::Failed)?;
    if !spec.ok_button_label.is_empty() {
        let label = encode_wide_nul(spec.ok_button_label);
        // SAFETY: `label` remains alive and NUL-terminated during SetOkButtonLabel.
        unsafe { base.SetOkButtonLabel(PCWSTR(label.as_ptr())) }
            .map_err(|_| MultiDialogError::Failed)?;
    }
    let modal = ModalDialog {
        inner: base,
        _apartment: &apartment,
    };
    match publish_and_show(owner, cancel, cancelled, &modal) {
        Ok(true) => {}
        Ok(false) | Err(DialogError::Cancelled) => return Err(MultiDialogError::Cancelled),
        Err(DialogError::Unavailable | DialogError::ResultInvalid) => {
            return Err(MultiDialogError::Failed);
        }
    }
    // SAFETY: Dialog accepted cleanly on the owning STA thread; fetch IShellItemArray results.
    let results: IShellItemArray =
        unsafe { open_dialog.GetResults() }.map_err(|_| MultiDialogError::Failed)?;
    // SAFETY: Valid IShellItemArray owned on the STA thread.
    let count = validate_picked_count(
        unsafe { results.GetCount() }.map_err(|_| MultiDialogError::Failed)?,
    )?;
    let capacity = usize::try_from(count).unwrap_or(MAX_PICKED_FILES);
    let mut paths = Vec::with_capacity(capacity);
    for index in 0..count {
        // SAFETY: `index` is strictly below `count` from GetCount on this STA thread.
        let item: IShellItem =
            unsafe { results.GetItemAt(index) }.map_err(|_| MultiDialogError::Failed)?;
        let path = shell_item_path(&item).ok_or(MultiDialogError::Failed)?;
        paths.push(path);
    }
    Ok(paths)
}

fn configure_filters_and_title(
    inner: &IFileDialog,
    filters: &[FilterSpec<'_>],
    title: &str,
) -> Result<(), DialogError> {
    if !filters.is_empty() {
        let wide_filters: Vec<(Vec<u16>, Vec<u16>)> = filters
            .iter()
            .map(|filter| {
                (
                    encode_wide_nul(filter.name),
                    encode_wide_nul(filter.pattern),
                )
            })
            .collect();
        let com_filters: Vec<COMDLG_FILTERSPEC> = wide_filters
            .iter()
            .map(|(name, pattern)| COMDLG_FILTERSPEC {
                pszName: PCWSTR(name.as_ptr()),
                pszSpec: PCWSTR(pattern.as_ptr()),
            })
            .collect();
        // SAFETY: `wide_filters` remains alive and NUL-terminated during SetFileTypes.
        unsafe { inner.SetFileTypes(&com_filters) }.map_err(|_| DialogError::Unavailable)?;
        // SAFETY: Selects the first 1-based entry configured in `com_filters`.
        unsafe { inner.SetFileTypeIndex(1) }.map_err(|_| DialogError::Unavailable)?;
    }
    let title = encode_wide_nul(title);
    // SAFETY: `title` remains alive and NUL-terminated during SetTitle.
    unsafe { inner.SetTitle(PCWSTR(title.as_ptr())) }.map_err(|_| DialogError::Unavailable)
}

/// Classifies `IFileDialog::Show` completion while giving external cancellation priority.
pub(crate) fn classify_picker_show(
    result: windows::core::Result<()>,
    cancelled: bool,
) -> Result<bool, DialogError> {
    if cancelled {
        return Err(DialogError::Cancelled);
    }
    match result {
        Ok(()) => Ok(true),
        Err(error) if error.code() == PICKER_CANCELLED => Ok(false),
        Err(_) => Err(DialogError::Unavailable),
    }
}

fn encode_wide_nul(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

fn close_dialog(dialog: AgileReference<IFileDialog>, showing: &AtomicBool) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_open_options_enable_multiselect_and_link_safety() {
        let flags = multi_open_dialog_options(FILEOPENDIALOGOPTIONS(0));
        for required in [
            FOS_ALLOWMULTISELECT,
            FOS_DONTADDTORECENT,
            FOS_FILEMUSTEXIST,
            FOS_FORCEFILESYSTEM,
            FOS_NOCHANGEDIR,
            FOS_NODEREFERENCELINKS,
            FOS_PATHMUSTEXIST,
        ] {
            assert_ne!(flags.0 & required.0, 0);
        }
    }

    #[test]
    fn multi_open_count_validation_enforces_zero_and_upper_bound() {
        let filters = [FilterSpec {
            name: "Images",
            pattern: "*.png;*.jpg",
        }];
        let spec = MultiDialogSpec {
            title: "Attach files",
            ok_button_label: "Attach",
            filters: &filters,
        };
        assert_eq!(spec.title, "Attach files");
        assert_eq!(spec.ok_button_label, "Attach");
        assert_eq!(spec.filters.len(), 1);
        assert_eq!(validate_picked_count(0), Err(MultiDialogError::Cancelled));
        assert_eq!(validate_picked_count(1), Ok(1));
        assert_eq!(validate_picked_count(8), Ok(8));
        assert_eq!(
            validate_picked_count(9),
            Err(MultiDialogError::TooManyFiles {
                selected: 9,
                max: MAX_PICKED_FILES,
            })
        );
    }
}
