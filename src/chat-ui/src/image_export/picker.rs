//! Single-file Shell save picker with agile cross-thread cancellation.
#![deny(unsafe_code)]

use std::{ffi::OsString, path::PathBuf, sync::atomic::AtomicBool};

pub(super) use crate::com_dialog::Cancel;
use crate::com_dialog::{DialogError, DialogMode, DialogSpec, FilterSpec, run_modal};

pub(super) const CANCELLED: &str = "Image save cancelled.";
const PICKER_UNAVAILABLE: &str = "The image save dialog could not open.";
const DESTINATION_FAILED: &str = "Could not resolve the selected save destination.";
const DEFAULT_STEM: &str = "bonsai-image";
const MAX_STEM_BYTES: usize = 64;
const IMAGE_FILTERS: [FilterSpec<'static>; 2] = [
    FilterSpec {
        name: "PNG image (*.png)",
        pattern: "*.png",
    },
    FilterSpec {
        name: "SVG vector image (*.svg)",
        pattern: "*.svg",
    },
];

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

pub(super) fn pick_save_target(
    owner: isize,
    cancel: &Cancel,
    cancelled: &AtomicBool,
    suggested_stem: &str,
) -> Result<Option<Selection>, &'static str> {
    let default_file_name = format!("{}.png", sanitize_stem(suggested_stem));
    let spec = DialogSpec {
        mode: DialogMode::Save {
            default_extension: "png",
            default_file_name: &default_file_name,
        },
        title: "Save generated image",
        filters: &IMAGE_FILTERS,
    };
    let Some(selection) =
        run_modal(owner, cancel, cancelled, &spec).map_err(|error| match error {
            DialogError::Cancelled => CANCELLED,
            DialogError::Unavailable => PICKER_UNAVAILABLE,
            DialogError::ResultInvalid => DESTINATION_FAILED,
        })?
    else {
        return Ok(None);
    };
    resolve_selection(selection.path, selection.file_type_index).map(Some)
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
        Win32::UI::Shell::{
            FILEOPENDIALOGOPTIONS, FOS_ALLOWMULTISELECT, FOS_DONTADDTORECENT, FOS_FILEMUSTEXIST,
            FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR, FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST,
            FOS_STRICTFILETYPES,
        },
        core::HRESULT,
    };

    use super::{
        CANCELLED, Cancel, DEFAULT_STEM, DialogMode, DialogSpec, Format, IMAGE_FILTERS,
        PICKER_UNAVAILABLE, Selection, pick_save_target, resolve_selection, sanitize_stem,
    };
    use crate::com_dialog::{
        ActiveDialog, Apartment, DialogError, ModalDialog, classify_picker_show, dialog_options,
    };

    const PICKER_CANCELLED: HRESULT =
        HRESULT::from_win32(windows::Win32::Foundation::ERROR_CANCELLED.0);

    #[test]
    fn save_picker_options_stem_sanitization_and_selection_resolution() {
        let options = dialog_options(FILEOPENDIALOGOPTIONS(FOS_ALLOWMULTISELECT.0), true);
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
        assert_eq!(
            classify_picker_show(Ok(()), true),
            Err(DialogError::Cancelled)
        );
        assert_eq!(
            classify_picker_show(
                Err(windows::core::Error::from(HRESULT(-2_147_467_259))),
                false,
            ),
            Err(DialogError::Unavailable)
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
                let spec = DialogSpec {
                    mode: DialogMode::Save {
                        default_extension: "png",
                        default_file_name: "bonsai-image-1.png",
                    },
                    title: "Save generated image",
                    filters: &IMAGE_FILTERS,
                };
                let Ok(dialog) = ModalDialog::new(&apartment, &spec) else {
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
            .map_err(|_| io::Error::other("save picker test worker panicked"))?;
        state.0.join();
        assert_eq!(outcome, Ok(()));
        Ok(())
    }
}
