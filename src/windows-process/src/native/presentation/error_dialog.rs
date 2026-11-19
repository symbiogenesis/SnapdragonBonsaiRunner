//! Fixed native startup failure UI, independent of graphics initialization.

use std::borrow::Cow;
use std::cell::Cell;
use std::ptr;
use windows_sys::{
    Win32::{
        Foundation::{GlobalFree, HANDLE, HGLOBAL, HINSTANCE, HWND, LPARAM, WPARAM},
        System::{
            LibraryLoader::GetModuleHandleW,
            Memory::{GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalUnlock},
            Ole::CF_UNICODETEXT,
        },
        UI::WindowsAndMessaging::{
            BS_DEFPUSHBUTTON, DLGTEMPLATE, DS_CENTER, DS_MODALFRAME, DS_SETFONT, ES_AUTOVSCROLL,
            ES_MULTILINE, ES_READONLY, GWLP_USERDATA, IDCANCEL, IDOK, MB_ICONERROR, MB_OK,
            WM_CLOSE, WM_COMMAND, WM_INITDIALOG, WS_BORDER, WS_CAPTION, WS_CHILD, WS_SYSMENU,
            WS_TABSTOP, WS_VISIBLE, WS_VSCROLL,
        },
    },
    core::w,
};

use crate::native::library::SystemLibrary;

const PREVIEW_ID: u16 = 1000;
const COPY_ID: u16 = 1001;
const FEEDBACK_ID: u16 = 1002;
const PREVIEW_CHARS: usize = 2048;

type DialogProc = unsafe extern "system" fn(HWND, u32, WPARAM, LPARAM) -> isize;
type DialogBoxIndirectParamWFn = unsafe extern "system" fn(
    HINSTANCE,
    *const DLGTEMPLATE,
    HWND,
    Option<DialogProc>,
    LPARAM,
) -> isize;
type MessageBoxWFn = unsafe extern "system" fn(HWND, *const u16, *const u16, u32) -> i32;
type SetDlgItemTextWFn = unsafe extern "system" fn(HWND, i32, *const u16) -> i32;
type GetWindowLongPtrWFn = unsafe extern "system" fn(HWND, i32) -> isize;
type SetWindowLongPtrWFn = unsafe extern "system" fn(HWND, i32, isize) -> isize;
type EndDialogFn = unsafe extern "system" fn(HWND, isize) -> i32;
type OpenClipboardFn = unsafe extern "system" fn(HWND) -> i32;
type EmptyClipboardFn = unsafe extern "system" fn() -> i32;
type SetClipboardDataFn = unsafe extern "system" fn(u32, HANDLE) -> HANDLE;
type CloseClipboardFn = unsafe extern "system" fn() -> i32;

struct User32Api {
    dialog_box_indirect_param_w: DialogBoxIndirectParamWFn,
    message_box_w: MessageBoxWFn,
    set_dlg_item_text_w: SetDlgItemTextWFn,
    get_window_long_ptr_w: GetWindowLongPtrWFn,
    set_window_long_ptr_w: SetWindowLongPtrWFn,
    end_dialog: EndDialogFn,
    open_clipboard: OpenClipboardFn,
    empty_clipboard: EmptyClipboardFn,
    set_clipboard_data: SetClipboardDataFn,
    close_clipboard: CloseClipboardFn,
    _library: SystemLibrary,
}

impl User32Api {
    fn load() -> std::io::Result<Self> {
        let library = SystemLibrary::load_system32("user32.dll", "system user32.dll unavailable")?;
        macro_rules! symbol {
            ($name:literal, $kind:ty) => {{
                let pointer = library.symbol($name)?;
                // SAFETY: Exact documented Windows ARM64 user32.dll export signature;
                // `_library` remains owned for the lifetime of this struct.
                unsafe {
                    std::mem::transmute::<unsafe extern "system" fn() -> isize, $kind>(pointer)
                }
            }};
        }
        Ok(Self {
            dialog_box_indirect_param_w: symbol!(
                c"DialogBoxIndirectParamW",
                DialogBoxIndirectParamWFn
            ),
            message_box_w: symbol!(c"MessageBoxW", MessageBoxWFn),
            set_dlg_item_text_w: symbol!(c"SetDlgItemTextW", SetDlgItemTextWFn),
            get_window_long_ptr_w: symbol!(c"GetWindowLongPtrW", GetWindowLongPtrWFn),
            set_window_long_ptr_w: symbol!(c"SetWindowLongPtrW", SetWindowLongPtrWFn),
            end_dialog: symbol!(c"EndDialog", EndDialogFn),
            open_clipboard: symbol!(c"OpenClipboard", OpenClipboardFn),
            empty_clipboard: symbol!(c"EmptyClipboard", EmptyClipboardFn),
            set_clipboard_data: symbol!(c"SetClipboardData", SetClipboardDataFn),
            close_clipboard: symbol!(c"CloseClipboard", CloseClipboardFn),
            _library: library,
        })
    }
}

thread_local! {
    static ACTIVE_USER32: Cell<*const User32Api> = const { Cell::new(ptr::null()) };
}

struct ActiveUser32Guard;

impl ActiveUser32Guard {
    fn install(api: &User32Api) -> Self {
        ACTIVE_USER32.set(&raw const *api);
        Self
    }
}

impl Drop for ActiveUser32Guard {
    fn drop(&mut self) {
        ACTIVE_USER32.set(ptr::null());
    }
}

struct Contents {
    full: Vec<u16>,
    preview: Vec<u16>,
}

fn terminated(text: &str) -> Vec<u16> {
    // A diagnostic containing NUL must not silently lose everything after it
    // when Windows consumes null-terminated text or CF_UNICODETEXT.
    text.replace('\0', "\\0")
        .encode_utf16()
        .chain(Some(0))
        .collect()
}

fn preview(message: &str, limit: usize) -> Cow<'_, str> {
    const SUFFIX: &str = "\n\n[Preview shortened. Copy error includes the complete message.]";
    match message.char_indices().nth(limit) {
        Some((end, _)) => {
            let mut text = String::with_capacity(end + SUFFIX.len());
            text.push_str(&message[..end]);
            text.push_str(SUFFIX);
            Cow::Owned(text)
        }
        None => Cow::Borrowed(message),
    }
}

pub(super) fn show(message: &str) {
    // Load user32.dll only when presenting a fatal graphical dialog so headless
    // CLI/probe binaries retain zero static GUI DLL imports under Win32k denial.
    let Ok(user32) = User32Api::load() else {
        return;
    };
    let contents = Contents {
        full: terminated(message),
        preview: terminated(&preview(message, PREVIEW_CHARS)),
    };
    let template = template();
    let parameter = isize::try_from((&raw const contents).expose_provenance());
    // SAFETY: A null name requests this process's module without transferring ownership.
    let instance = unsafe { GetModuleHandleW(ptr::null()) };
    let result = parameter.map_or(-1, |parameter| {
        let _guard = ActiveUser32Guard::install(&user32);
        // SAFETY: The DWORD-aligned template contains only standard native controls.
        // All template/text storage, `user32`, and the callback context remain live
        // until this synchronous modal call returns on this thread.
        unsafe {
            (user32.dialog_box_indirect_param_w)(
                instance,
                template.as_ptr().cast::<DLGTEMPLATE>(),
                ptr::null_mut(),
                Some(dialog_proc),
                parameter,
            )
        }
    });
    if result == -1 {
        // Even a platform failure must not put an unbounded diagnostic in a message box.
        let short: String = message
            .chars()
            .take(160)
            .map(|character| {
                if character.is_control() {
                    ' '
                } else {
                    character
                }
            })
            .collect();
        let fallback = terminated(&format!(
            "Swift Bonsai could not start.\n\n{short}\n\nThe detailed error controls could not open."
        ));
        // SAFETY: Both terminated strings and `user32` remain live throughout the synchronous call.
        let _ = unsafe {
            (user32.message_box_w)(
                ptr::null_mut(),
                fallback.as_ptr(),
                w!("Swift Bonsai 2"),
                MB_OK | MB_ICONERROR,
            )
        };
    }
}

unsafe extern "system" fn dialog_proc(
    dialog: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> isize {
    let api_pointer = ACTIVE_USER32.with(Cell::get);
    if api_pointer.is_null() {
        return 0;
    }
    // SAFETY: `ActiveUser32Guard` keeps the borrowed `User32Api` alive on this
    // thread's stack for the complete synchronous `DialogBoxIndirectParamW` call.
    let user32 = unsafe { &*api_pointer };
    if message == WM_INITDIALOG {
        // SAFETY: This dialog owns its user-data slot and receives the live Contents
        // pointer only from the synchronous DialogBoxIndirectParamW invocation above.
        let _ = unsafe { (user32.set_window_long_ptr_w)(dialog, GWLP_USERDATA, lparam) };
    }
    // SAFETY: Only WM_INITDIALOG above writes this dialog's private user-data slot.
    let parameter = unsafe { (user32.get_window_long_ptr_w)(dialog, GWLP_USERDATA) };
    let pointer =
        ptr::with_exposed_provenance::<Contents>(usize::try_from(parameter).unwrap_or_default());
    if !pointer.is_null() {
        // SAFETY: The caller owns Contents for the complete modal lifetime. The
        // callback borrows it immutably and makes no reentrant mutation of it.
        let contents = unsafe { &*pointer };
        if message == WM_INITDIALOG {
            // SAFETY: PREVIEW_ID identifies the template's edit control; text stays live.
            let _ = unsafe {
                (user32.set_dlg_item_text_w)(
                    dialog,
                    i32::from(PREVIEW_ID),
                    contents.preview.as_ptr(),
                )
            };
            return 1;
        }
        if message == WM_COMMAND && wparam & 0xffff == usize::from(COPY_ID) {
            let feedback = if copy(user32, dialog, &contents.full) {
                w!("Copied the complete error.")
            } else {
                w!("Clipboard unavailable. Try Copy error again.")
            };
            // SAFETY: The template's fixed feedback control copies this static text.
            let _ =
                unsafe { (user32.set_dlg_item_text_w)(dialog, i32::from(FEEDBACK_ID), feedback) };
            return 1;
        }
    }
    if message == WM_CLOSE
        || (message == WM_COMMAND && matches!(i32::try_from(wparam & 0xffff), Ok(IDOK | IDCANCEL)))
    {
        // SAFETY: This callback runs only for its own live modal dialog.
        let _ = unsafe { (user32.end_dialog)(dialog, 0) };
        return 1;
    }
    0
}

struct Clipboard<'a>(&'a User32Api);

impl Drop for Clipboard<'_> {
    fn drop(&mut self) {
        // SAFETY: This guard exists only after OpenClipboard succeeded on this thread.
        let _ = unsafe { (self.0.close_clipboard)() };
    }
}

struct Allocation(HGLOBAL);

impl Drop for Allocation {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: Non-null storage is unlocked and still owned by this guard;
            // successful SetClipboardData clears it before Windows takes ownership.
            let _ = unsafe { GlobalFree(self.0) };
        }
    }
}

fn copy(user32: &User32Api, dialog: HWND, text: &[u16]) -> bool {
    let Some(bytes) = text.len().checked_mul(size_of::<u16>()) else {
        return false;
    };
    // SAFETY: The byte count is checked and GMEM_MOVEABLE is required for clipboard transfer.
    let mut allocation = Allocation(unsafe { GlobalAlloc(GMEM_MOVEABLE, bytes) });
    if allocation.0.is_null() {
        return false;
    }
    // SAFETY: This is a live, exclusively owned movable allocation.
    let destination = unsafe { GlobalLock(allocation.0) }.cast::<u16>();
    if destination.is_null() {
        return false;
    }
    // SAFETY: The source and newly allocated destination do not overlap; the
    // checked allocation size covers every UTF-16 unit including the terminator.
    unsafe { ptr::copy_nonoverlapping(text.as_ptr(), destination, text.len()) };
    // SAFETY: The allocation has exactly the one lock obtained above.
    let _ = unsafe { GlobalUnlock(allocation.0) };
    // SAFETY: The live dialog provides the valid owner required by EmptyClipboard.
    if unsafe { (user32.open_clipboard)(dialog) } == 0 {
        return false;
    }
    let _clipboard = Clipboard(user32);
    // SAFETY: This thread holds the clipboard open and intentionally replaces its contents.
    if unsafe { (user32.empty_clipboard)() } == 0 {
        return false;
    }
    // SAFETY: CF_UNICODETEXT accepts unlocked movable storage containing
    // terminated UTF-16. Ownership transfers only when this call succeeds.
    if unsafe { (user32.set_clipboard_data)(u32::from(CF_UNICODETEXT), allocation.0) }.is_null() {
        return false;
    }
    allocation.0 = ptr::null_mut();
    true
}

fn push_dword(words: &mut Vec<u16>, value: u32) {
    let bytes = value.to_le_bytes();
    words.extend([
        u16::from_le_bytes([bytes[0], bytes[1]]),
        u16::from_le_bytes([bytes[2], bytes[3]]),
    ]);
}

fn control(words: &mut Vec<u16>, id: u16, class: u16, style: u32, rect: [u16; 4], text: &str) {
    if !words.len().is_multiple_of(2) {
        words.push(0);
    }
    push_dword(words, WS_CHILD | WS_VISIBLE | style);
    push_dword(words, 0);
    words.extend(rect);
    words.extend([id, 0xffff, class]);
    words.extend(terminated(text));
    words.push(0); // No creation data.
}

fn template() -> Vec<u32> {
    let mut words = Vec::new();
    push_dword(
        &mut words,
        WS_CAPTION | WS_SYSMENU | (DS_MODALFRAME | DS_SETFONT | DS_CENTER).cast_unsigned(),
    );
    push_dword(&mut words, 0);
    words.extend([6, 0, 0, 280, 164, 0, 0]); // Count, fixed bounds, no menu/default dialog class.
    words.extend(terminated("Swift Bonsai 2"));
    words.push(9);
    words.extend(terminated("Segoe UI"));
    control(
        &mut words,
        1003,
        0x82,
        0,
        [12, 10, 256, 14],
        "Swift Bonsai could not start",
    );
    control(
        &mut words,
        1004,
        0x82,
        0,
        [12, 29, 256, 12],
        "Error details (preview)",
    );
    control(
        &mut words,
        PREVIEW_ID,
        0x81,
        WS_BORDER
            | WS_TABSTOP
            | WS_VSCROLL
            | (ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY).cast_unsigned(),
        [12, 44, 256, 66],
        "",
    );
    control(
        &mut words,
        FEEDBACK_ID,
        0x82,
        0,
        [12, 115, 256, 18],
        "Copy error keeps the complete message. Review paths before sharing.",
    );
    control(
        &mut words,
        COPY_ID,
        0x80,
        WS_TABSTOP,
        [116, 140, 82, 16],
        "&Copy error",
    );
    control(
        &mut words,
        1,
        0x80,
        WS_TABSTOP | BS_DEFPUSHBUTTON.cast_unsigned(),
        [206, 140, 62, 16],
        "Close",
    );
    if !words.len().is_multiple_of(2) {
        words.push(0);
    }
    // Vec<u32> guarantees the DWORD alignment required by dialog templates.
    words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u32::from(pair[0]) | (u32::from(pair[1]) << 16))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read as _};

    fn word(input: &mut Cursor<&[u8]>) -> std::io::Result<u16> {
        let mut bytes = [0; 2];
        input.read_exact(&mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn dword(input: &mut Cursor<&[u8]>) -> std::io::Result<u32> {
        let mut bytes = [0; 4];
        input.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn text(input: &mut Cursor<&[u8]>) -> Result<String, Box<dyn std::error::Error>> {
        let mut units = Vec::new();
        loop {
            let unit = word(input)?;
            if unit == 0 {
                return Ok(String::from_utf16(&units)?);
            }
            units.push(unit);
        }
    }

    #[test]
    fn startup_error_template_encodes_aligned_accessible_native_controls()
    -> Result<(), Box<dyn std::error::Error>> {
        // Decode the standard DLGTEMPLATE wire layout independently of the
        // encoder. Opening a native modal would require interactive acceptance.
        let bytes: Vec<_> = template().into_iter().flat_map(u32::to_le_bytes).collect();
        let mut input = Cursor::new(bytes.as_slice());
        let style = dword(&mut input)?;
        assert_ne!(style & DS_SETFONT.cast_unsigned(), 0);
        assert_eq!(dword(&mut input)?, 0);
        assert_eq!(word(&mut input)?, 6);
        assert_eq!([word(&mut input)?, word(&mut input)?], [0, 0]);
        let width = word(&mut input)?;
        let height = word(&mut input)?;
        assert_eq!([word(&mut input)?, word(&mut input)?], [0, 0]);
        assert_eq!(text(&mut input)?, "Swift Bonsai 2");
        assert!(word(&mut input)? > 0);
        assert_eq!(text(&mut input)?, "Segoe UI");

        for (id, class, required, label) in [
            (1003, 0x82, 0, "Swift Bonsai could not start"),
            (1004, 0x82, 0, "Error details (preview)"),
            (
                1000,
                0x81,
                WS_TABSTOP | WS_VSCROLL | (ES_MULTILINE | ES_READONLY).cast_unsigned(),
                "",
            ),
            (
                1002,
                0x82,
                0,
                "Copy error keeps the complete message. Review paths before sharing.",
            ),
            (1001, 0x80, WS_TABSTOP, "&Copy error"),
            (
                1,
                0x80,
                WS_TABSTOP | BS_DEFPUSHBUTTON.cast_unsigned(),
                "Close",
            ),
        ] {
            if !input.position().is_multiple_of(4) {
                assert_eq!(word(&mut input)?, 0, "control alignment padding");
            }
            let style = dword(&mut input)?;
            assert_eq!(style & (WS_CHILD | WS_VISIBLE), WS_CHILD | WS_VISIBLE);
            assert_eq!(
                style & required,
                required,
                "required flags for control {id}"
            );
            assert_eq!(dword(&mut input)?, 0);
            let x = word(&mut input)?;
            let y = word(&mut input)?;
            let control_width = word(&mut input)?;
            let control_height = word(&mut input)?;
            assert!(control_width > 0 && control_height > 0);
            assert!(u32::from(x) + u32::from(control_width) <= u32::from(width));
            assert!(u32::from(y) + u32::from(control_height) <= u32::from(height));
            assert_eq!(word(&mut input)?, id);
            assert_eq!([word(&mut input)?, word(&mut input)?], [0xffff, class]);
            assert_eq!(text(&mut input)?, label);
            assert_eq!(word(&mut input)?, 0, "unexpected control creation data");
        }
        if !input.position().is_multiple_of(4) {
            assert_eq!(word(&mut input)?, 0, "final DWORD alignment padding");
        }
        assert_eq!(input.position(), u64::try_from(bytes.len())?);
        Ok(())
    }

    #[test]
    fn startup_error_preview_is_bounded_without_losing_full_copy_text() {
        let message = format!("{}\0tail", "錯誤🌱\n".repeat(1000));
        let shortened = preview(&message, PREVIEW_CHARS);
        assert!(shortened.chars().count() < PREVIEW_CHARS + 100);
        assert!(
            shortened.ends_with("[Preview shortened. Copy error includes the complete message.]")
        );
        let full = terminated(&message);
        assert_eq!(full.last(), Some(&0));
        assert_eq!(full.iter().filter(|unit| **unit == 0).count(), 1);
        assert!(String::from_utf16_lossy(&full).ends_with("\\0tail\0"));
        assert_eq!(preview("Short error", PREVIEW_CHARS), "Short error");
        assert!(matches!(
            preview("Short error", PREVIEW_CHARS),
            Cow::Borrowed("Short error")
        ));
    }
}
