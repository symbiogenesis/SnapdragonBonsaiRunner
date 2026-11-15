//! Full inherited UTF-16 environments with deterministic Windows-name overlays.

use std::os::windows::ffi::OsStrExt as _;
use std::{cmp::Ordering, collections::BTreeMap, ffi::OsString, io};

use windows_sys::Win32::Globalization::{
    CSTR_EQUAL, CSTR_GREATER_THAN, CSTR_LESS_THAN, CompareStringOrdinal,
};

use crate::invalid;

const PROTECTED: &[&str] = &[
    "SystemDrive",
    "SystemRoot",
    "windir",
    "ProgramData",
    "LOCALAPPDATA",
    "APPDATA",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "ALLUSERSPROFILE",
    "PUBLIC",
    "ComSpec",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramW6432",
    "CommonProgramFiles",
    "CommonProgramFiles(x86)",
    "CommonProgramW6432",
];

fn compare(left: &[u16], right: &[u16]) -> io::Result<Ordering> {
    let left_len =
        i32::try_from(left.len()).map_err(|_| invalid("environment name is too long"))?;
    let right_len =
        i32::try_from(right.len()).map_err(|_| invalid("environment name is too long"))?;
    // SAFETY: Both slices remain readable for their checked UTF-16 lengths.
    // Explicit lengths preserve unpaired UTF-16 units; ordinal case comparison
    // is independent of locale and does not modify or retain either slice.
    match unsafe { CompareStringOrdinal(left.as_ptr(), left_len, right.as_ptr(), right_len, 1) } {
        CSTR_LESS_THAN => Ok(Ordering::Less),
        CSTR_EQUAL => Ok(Ordering::Equal),
        CSTR_GREATER_THAN => Ok(Ordering::Greater),
        _ => Err(io::Error::last_os_error()),
    }
}

fn is_protected(name: &[u16]) -> io::Result<bool> {
    let mut stack = [0_u16; 32];
    for protected in PROTECTED {
        let mut len = 0;
        for unit in protected.encode_utf16() {
            if let Some(slot) = stack.get_mut(len) {
                *slot = unit;
                len += 1;
            }
        }
        if compare(name, &stack[..len])?.is_eq() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn sort_entries(entries: &mut [(Vec<u16>, Vec<u16>)]) -> io::Result<()> {
    let mut comparison_error = None;
    entries.sort_by(|left, right| {
        if comparison_error.is_some() {
            return Ordering::Equal;
        }
        match compare(&left.0, &right.0) {
            Ok(order) => order,
            Err(error) => {
                comparison_error = Some(error);
                Ordering::Equal
            }
        }
    });
    if let Some(error) = comparison_error {
        return Err(error);
    }
    Ok(())
}

fn push_entry(block: &mut Vec<u16>, name: &[u16], value: &[u16]) {
    block.extend_from_slice(name);
    block.push(u16::from(b'='));
    block.extend_from_slice(value);
    block.push(0);
}

pub(super) fn build(
    inherited: impl IntoIterator<Item = (OsString, OsString)>,
    overlay: &BTreeMap<String, String>,
) -> io::Result<Vec<u16>> {
    let mut overrides = Vec::with_capacity(overlay.len());
    for (name, value) in overlay {
        if name.is_empty() || name.contains(['=', '\0']) || value.contains('\0') {
            return Err(invalid("invalid child environment entry"));
        }
        let name: Vec<_> = name.encode_utf16().collect();
        if is_protected(&name)? {
            return Err(invalid(
                "startup cannot replace a Windows system/profile environment variable",
            ));
        }
        overrides.push((name, value.encode_utf16().collect()));
    }
    sort_entries(&mut overrides)?;
    for pair in overrides.windows(2) {
        if compare(&pair[0].0, &pair[1].0)?.is_eq() {
            return Err(invalid(
                "duplicate case-insensitive child environment override",
            ));
        }
    }

    let mut entries: Vec<(Vec<u16>, Vec<u16>)> = inherited
        .into_iter()
        .map(|(name, value)| (name.encode_wide().collect(), value.encode_wide().collect()))
        .collect();
    sort_entries(&mut entries)?;

    let mut block = Vec::new();
    let mut inherited_index = 0;
    let mut override_index = 0;
    while inherited_index < entries.len() && override_index < overrides.len() {
        let entry = &entries[inherited_index];
        let replacement = &overrides[override_index];
        match compare(&entry.0, &replacement.0)? {
            Ordering::Less => {
                push_entry(&mut block, &entry.0, &entry.1);
                inherited_index += 1;
            }
            Ordering::Equal => {
                inherited_index += 1;
            }
            Ordering::Greater => {
                push_entry(&mut block, &replacement.0, &replacement.1);
                override_index += 1;
            }
        }
    }
    for (name, value) in &entries[inherited_index..] {
        push_entry(&mut block, name, value);
    }
    for (name, value) in &overrides[override_index..] {
        push_entry(&mut block, name, value);
    }
    if block.is_empty() {
        block.push(0);
    }
    block.push(0);
    Ok(block)
}

pub(super) fn build_with_removed(
    inherited: impl IntoIterator<Item = (OsString, OsString)>,
    overlay: &BTreeMap<String, String>,
    removed: &[String],
) -> io::Result<Vec<u16>> {
    if removed.len() > 256
        || removed.iter().any(|name| {
            name.is_empty() || name.len() > 260 || !name.is_ascii() || name.contains(['=', '\0'])
        })
    {
        return Err(invalid("Invalid child environment removal selection."));
    }
    for name in removed {
        if PROTECTED
            .iter()
            .any(|protected| name.eq_ignore_ascii_case(protected))
        {
            return Err(invalid(
                "Cannot remove a Windows system/profile environment variable.",
            ));
        }
    }
    build(
        inherited.into_iter().filter(|(name, _)| {
            !removed
                .iter()
                .any(|selected| name.to_string_lossy().eq_ignore_ascii_case(selected))
        }),
        overlay,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::windows::ffi::OsStringExt as _;

    #[test]
    fn unicode_environment_preserves_profile_drive_entries_and_unpaired_units() -> io::Result<()> {
        let unusual = OsString::from_wide(&[0xd800, 88]);
        let inherited = vec![
            ("Path".into(), "original".into()),
            ("SystemRoot".into(), "C:\\Windows".into()),
            ("LOCALAPPDATA".into(), "C:\\profile\\local".into()),
            ("=C:".into(), "C:\\drive-working".into()),
            (unusual.clone(), unusual),
        ];
        let overlay = BTreeMap::from([
            ("pAtH".into(), "tool-path".into()),
            ("BONSAI_UNICODE".into(), "\u{30d1}\u{30b9}\u{1f332}".into()),
        ]);
        let block = build(inherited, &overlay)?;
        let entries: Vec<_> = block
            .split(|unit| *unit == 0)
            .filter(|entry| !entry.is_empty())
            .collect();
        for expected in [
            "pAtH=tool-path",
            "SystemRoot=C:\\Windows",
            "LOCALAPPDATA=C:\\profile\\local",
            "=C:=C:\\drive-working",
            "BONSAI_UNICODE=\u{30d1}\u{30b9}\u{1f332}",
        ] {
            assert!(entries.contains(&expected.encode_utf16().collect::<Vec<_>>().as_slice()));
        }
        assert!(
            !entries.contains(
                &"Path=original"
                    .encode_utf16()
                    .collect::<Vec<_>>()
                    .as_slice()
            )
        );
        assert!(entries.contains(&[0xd800, 88, 61, 0xd800, 88].as_slice()));
        assert!(block.ends_with(&[0, 0]));
        Ok(())
    }

    #[test]
    fn protected_duplicate_and_malformed_overrides_reject() {
        for (name, value) in [
            ("systemROOT", "C:\\unsafe"),
            ("AppData", "C:\\unsafe"),
            ("ProgramData", "C:\\unsafe"),
            ("USERPROFILE", "C:\\unsafe"),
            ("", "empty"),
            ("=C:", "drive override"),
            ("A=B", "equals"),
            ("NUL\0NAME", "nul"),
            ("NAME", "NUL\0VALUE"),
        ] {
            assert!(build(Vec::new(), &BTreeMap::from([(name.into(), value.into())])).is_err());
        }
        assert!(
            build(
                Vec::new(),
                &BTreeMap::from([
                    ("Path".into(), "first".into()),
                    ("PATH".into(), "second".into()),
                ])
            )
            .is_err()
        );
    }
}
