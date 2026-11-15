//! Reject external speech/MSVC runtime imports when creating new distributions.
//!
//! This reads PE metadata only; it never loads the candidate image or a DLL.

use std::{
    io::{Read, Seek},
    path::Path,
};

use crate::pe_format::{PeLayout, PePolicy, SeekSource};

const MAX_IMPORTS: usize = 4096;
const MAX_IMPORT_BYTES: usize = 1024 * 1024;
const MAX_DLL_NAME: usize = 256;

enum PortableError {
    Invalid,
    Forbidden(String),
}

impl From<String> for PortableError {
    fn from(_: String) -> Self {
        Self::Invalid
    }
}

pub(super) fn verify(path: &Path) -> crate::Result<()> {
    let file = super::open(path)?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    verify_reader(file, length)
}

fn invalid() -> String {
    "Invalid or unsupported portable PE import metadata.".into()
}

fn verify_dll_name(
    layout: &PeLayout,
    source: &mut impl crate::pe_format::ByteSource,
    rva: u64,
) -> Result<(), PortableError> {
    let mut bytes = [0_u8; MAX_DLL_NAME];
    let name = layout.read_c_bytes_at(source, rva, true, &mut bytes, "Unterminated DLL name")?;
    if name
        .iter()
        .any(|byte| !byte.is_ascii_alphanumeric() && !b"._-".contains(byte))
    {
        return Err(PortableError::Invalid);
    }
    let name = std::str::from_utf8(name)
        .map_err(|_| PortableError::Invalid)?
        .to_ascii_lowercase();
    if forbidden(&name) {
        return Err(PortableError::Forbidden(format!(
            "Portable executable imports {name}. Rebuild all application components with the static runtime and Rust TTS before packaging."
        )));
    }
    Ok(())
}

fn verify_reader<R: Read + Seek>(source: R, length: u64) -> crate::Result<()> {
    let mut source = SeekSource::new(source, length);
    let layout = PeLayout::parse(&mut source, &PePolicy::PORTABLE).map_err(|_| invalid())?;
    for delayed in [false, true] {
        layout
            .visit_import_descriptors::<_, PortableError>(
                &mut source,
                delayed,
                MAX_IMPORT_BYTES,
                MAX_IMPORTS,
                true,
                |source, descriptor| {
                    let rva = layout.resolve_import_name_rva(
                        delayed,
                        descriptor.attributes,
                        descriptor.name_raw,
                        true,
                    )?;
                    verify_dll_name(&layout, source, rva)
                },
            )
            .map_err(|error| match error {
                PortableError::Invalid => invalid(),
                PortableError::Forbidden(message) => message,
            })?;
    }
    Ok(())
}

fn forbidden(name: &str) -> bool {
    [
        "vcruntime",
        "msvcp",
        "concrt",
        "vcomp",
        "ggml",
        "qwen",
        "onnxruntime",
        "ucrtbased",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
        || (name.starts_with("msvcr") && name != "msvcrt.dll")
}

#[cfg(test)]
mod tests {
    use super::verify_reader;
    use std::io::Cursor;

    fn image(normal: Option<&str>, delayed: Option<&str>) -> Vec<u8> {
        let mut bytes = vec![0; 1024];
        bytes[..2].copy_from_slice(b"MZ");
        bytes[60..64].copy_from_slice(&128_u32.to_le_bytes());
        bytes[128..134].copy_from_slice(&[b'P', b'E', 0, 0, 0x64, 0xaa]);
        bytes[134..136].copy_from_slice(&1_u16.to_le_bytes());
        bytes[148..150].copy_from_slice(&240_u16.to_le_bytes());
        bytes[152..154].copy_from_slice(&0x20b_u16.to_le_bytes());
        bytes[212..216].copy_from_slice(&512_u32.to_le_bytes());
        bytes[260..264].copy_from_slice(&16_u32.to_le_bytes());
        bytes[400..404].copy_from_slice(&512_u32.to_le_bytes());
        bytes[404..408].copy_from_slice(&0x1000_u32.to_le_bytes());
        bytes[408..412].copy_from_slice(&512_u32.to_le_bytes());
        bytes[412..416].copy_from_slice(&512_u32.to_le_bytes());
        if let Some(name) = normal {
            bytes[272..276].copy_from_slice(&0x1000_u32.to_le_bytes());
            bytes[276..280].copy_from_slice(&40_u32.to_le_bytes());
            bytes[524..528].copy_from_slice(&0x1080_u32.to_le_bytes());
            bytes[640..640 + name.len()].copy_from_slice(name.as_bytes());
        }
        if let Some(name) = delayed {
            bytes[368..372].copy_from_slice(&0x1100_u32.to_le_bytes());
            bytes[372..376].copy_from_slice(&64_u32.to_le_bytes());
            bytes[768..772].copy_from_slice(&1_u32.to_le_bytes());
            bytes[772..776].copy_from_slice(&0x1180_u32.to_le_bytes());
            bytes[896..896 + name.len()].copy_from_slice(name.as_bytes());
        }
        bytes
    }

    fn check(bytes: Vec<u8>) -> crate::Result<()> {
        let length = u64::try_from(bytes.len()).map_err(|error| error.to_string())?;
        verify_reader(Cursor::new(bytes), length)
    }

    #[test]
    fn system_crt_imports_pass_and_external_runtimes_fail_in_both_tables() -> crate::Result<()> {
        check(image(None, None))?;
        for name in [
            "KERNEL32.dll",
            "ucrtbase.dll",
            "api-ms-win-crt-runtime-l1-1-0.dll",
            "msvcrt.dll",
        ] {
            check(image(Some(name), Some(name)))?;
        }
        for name in [
            "VCRUNTIME140.dll",
            "msvcp140.dll",
            "concrt140.dll",
            "vcomp140.dll",
            "ggml-base.dll",
            "qwen-core.dll",
            "onnxruntime.dll",
            "ucrtbased.dll",
            "msvcr120.dll",
        ] {
            assert!(check(image(Some(name), None)).is_err());
            assert!(check(image(Some("KERNEL32.dll"), Some(name))).is_err());
        }
        Ok(())
    }

    #[test]
    fn delayed_virtual_addresses_are_checked_without_truncating_the_base_address()
    -> crate::Result<()> {
        let mut bytes = image(None, Some("vcruntime140.dll"));
        bytes[176..184].copy_from_slice(&0x1000_0000_u64.to_le_bytes());
        bytes[768..772].fill(0);
        bytes[772..776].copy_from_slice(&0x1000_1180_u32.to_le_bytes());
        assert!(check(bytes.clone()).is_err());
        bytes[896..912].fill(0);
        bytes[896..908].copy_from_slice(b"KERNEL32.dll");
        check(bytes.clone())?;
        bytes[176..184].copy_from_slice(&0x1_1000_0000_u64.to_le_bytes());
        assert!(check(bytes).is_err());
        Ok(())
    }

    #[test]
    fn malformed_tables_names_and_optional_headers_fail_closed() {
        let mut bytes = image(Some("KERNEL32.dll"), None);
        bytes[276..280].copy_from_slice(&20_u32.to_le_bytes());
        assert!(check(bytes).is_err()); // No terminating descriptor in the directory.
        let mut bytes = image(Some("KERNEL32.dll"), None);
        bytes[524..528].copy_from_slice(&0x2000_u32.to_le_bytes());
        assert!(check(bytes).is_err()); // No initialized bytes at the DLL-name RVA.
        let mut bytes = image(Some("KERNEL32.dll"), None);
        bytes[272..276].copy_from_slice(&0x11f8_u32.to_le_bytes());
        assert!(check(bytes).is_err()); // Table straddles the section's raw end.
        let mut bytes = image(None, Some("KERNEL32.dll"));
        bytes[768..772].copy_from_slice(&2_u32.to_le_bytes());
        assert!(check(bytes).is_err()); // Unknown delay-descriptor attributes.
        let mut bytes = image(Some("KERNEL32.dll"), None);
        bytes[640..896].fill(b'a');
        assert!(check(bytes).is_err()); // Bounded names must terminate.
        let mut bytes = image(None, None);
        bytes[152..154].copy_from_slice(&0x10b_u16.to_le_bytes());
        assert!(check(bytes).is_err());
        assert!(check(vec![0; 63]).is_err());
    }
}
