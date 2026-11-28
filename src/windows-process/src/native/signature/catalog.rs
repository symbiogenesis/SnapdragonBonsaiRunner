//! Installed catalog member verification using the retained file handle.
//!
//! Reference: <https://learn.microsoft.com/windows/win32/api/mscat/nf-mscat-cryptcatadminenumcatalogfromhash>

use std::fs::File;
use std::io;
use std::os::windows::io::AsRawHandle as _;
use std::ptr;
use windows_sys::Win32::Security::Cryptography::Catalog::{
    CATALOG_INFO, CryptCATAdminAcquireContext2, CryptCATAdminCalcHashFromFileHandle2,
    CryptCATAdminEnumCatalogFromHash, CryptCATAdminReleaseCatalogContext,
    CryptCATAdminReleaseContext, CryptCATCatalogInfoFromContext,
};
use windows_sys::Win32::Security::WinTrust::{
    DRIVER_ACTION_VERIFY, WINTRUST_CATALOG_INFO, WINTRUST_DATA_0, WTD_CHOICE_CATALOG,
};

struct Admin(isize);
impl Drop for Admin {
    fn drop(&mut self) {
        // SAFETY: The acquired catalog administration context is owned exactly once.
        let _ = unsafe { CryptCATAdminReleaseContext(self.0, 0) };
    }
}
struct Catalog<'a> {
    admin: &'a Admin,
    context: isize,
}
impl Drop for Catalog<'_> {
    fn drop(&mut self) {
        if self.context != 0 {
            // SAFETY: This catalog context was obtained from the still-live matching
            // admin and has not yet been consumed by `CryptCATAdminEnumCatalogFromHash`.
            let _ = unsafe { CryptCATAdminReleaseCatalogContext(self.admin.0, self.context, 0) };
        }
    }
}

fn enumerate_catalogs<T>(
    mut next_catalog: impl FnMut() -> Option<isize>,
    mut verify_member: impl FnMut(isize) -> io::Result<T>,
) -> io::Result<T> {
    const MAX_CATALOGS: usize = 16;
    let mut last_error = None;
    for _ in 0..MAX_CATALOGS {
        let Some(context) = next_catalog() else {
            break;
        };
        match verify_member(context) {
            Ok(signer) => return Ok(signer),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        super::super::invalid("No installed catalog binds the retained file hash.")
    }))
}

pub(super) fn verify(file: &File, filename: &[u16]) -> io::Result<crate::AuthenticodeSigner> {
    let mut admin = 0;
    let subsystem = DRIVER_ACTION_VERIFY;
    let algorithm = windows_sys::core::w!("SHA256");
    // SAFETY: The output is writable; static algorithm and subsystem inputs remain live.
    super::super::check(unsafe {
        CryptCATAdminAcquireContext2(
            &raw mut admin,
            &raw const subsystem,
            algorithm,
            ptr::null(),
            0,
        )
    })?;
    let admin = Admin(admin);
    let mut hash = [0_u8; 64];
    let mut hash_bytes = 64_u32;
    // SAFETY: The retained file and catalog context are live; the destination has
    // exactly the declared writable capacity and the API is synchronous.
    super::super::check(unsafe {
        CryptCATAdminCalcHashFromFileHandle2(
            admin.0,
            file.as_raw_handle(),
            &raw mut hash_bytes,
            hash.as_mut_ptr(),
            0,
        )
    })?;
    if hash_bytes != 32 {
        return Err(super::super::invalid("Catalog SHA-256 length changed."));
    }
    let mut member: Vec<u16> = hash[..32]
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|part| u16::from(b"0123456789ABCDEF"[usize::from(part)]))
        .collect();
    member.push(0);

    let mut catalog = Catalog {
        admin: &admin,
        context: 0,
    };
    let catalog_cell = std::cell::RefCell::new(&mut catalog);
    enumerate_catalogs(
        || {
            let mut cat = catalog_cell.borrow_mut();
            let mut prev_context = std::mem::replace(&mut cat.context, 0);
            // SAFETY: The live administrator receives the exact SHA-256 bytes and either
            // 0 (fresh search) or the prior context whose ownership transfers to
            // `CryptCATAdminEnumCatalogFromHash` (which releases it).
            let context = unsafe {
                CryptCATAdminEnumCatalogFromHash(
                    admin.0,
                    hash.as_ptr(),
                    hash_bytes,
                    0,
                    &raw mut prev_context,
                )
            };
            if context == 0 {
                None
            } else {
                cat.context = context;
                Some(context)
            }
        },
        |context| verify_catalog_member(&admin, context, file, filename, &member, &hash, hash_bytes),
    )
}

fn verify_catalog_member(
    admin: &Admin,
    catalog_context: isize,
    file: &File,
    filename: &[u16],
    member: &[u16],
    hash: &[u8; 64],
    hash_bytes: u32,
) -> io::Result<crate::AuthenticodeSigner> {
    let mut info = CATALOG_INFO {
        cbStruct: super::super::win_size::<CATALOG_INFO>(),
        ..CATALOG_INFO::default()
    };
    // SAFETY: The selected catalog remains owned; info is correctly sized and writable.
    super::super::check(unsafe {
        CryptCATCatalogInfoFromContext(catalog_context, &raw mut info, 0)
    })?;
    if !info.wszCatalogFile.contains(&0) {
        return Err(super::super::invalid("Catalog path was not terminated."));
    }
    let mut calc_hash = *hash;
    let mut catalog_info = WINTRUST_CATALOG_INFO {
        cbStruct: super::super::win_size::<WINTRUST_CATALOG_INFO>(),
        pcwszCatalogFilePath: info.wszCatalogFile.as_ptr(),
        pcwszMemberTag: member.as_ptr(),
        pcwszMemberFilePath: filename.as_ptr(),
        hMemberFile: file.as_raw_handle(),
        pbCalculatedFileHash: calc_hash.as_mut_ptr(),
        cbCalculatedFileHash: hash_bytes,
        hCatAdmin: admin.0,
        ..WINTRUST_CATALOG_INFO::default()
    };
    super::verify_wintrust(
        WTD_CHOICE_CATALOG,
        WINTRUST_DATA_0 {
            pCatalog: &raw mut catalog_info,
        },
        "Installed catalog trust or membership was rejected.",
        "Installed catalog trust state did not close.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_catalogs_falls_back_through_candidates_and_bounds_max_catalogs() {
        // 0 catalogs -> "No installed catalog binds..."
        let none = enumerate_catalogs::<()>(|| None, |_| Ok(()));
        assert!(
            none.err()
                .is_some_and(|e| e.to_string().contains("No installed catalog"))
        );

        // 1st fails, 2nd succeeds
        let mut idx = 0_isize;
        let second = enumerate_catalogs(
            || {
                idx += 1;
                Some(idx)
            },
            |ctx| {
                if ctx == 1 {
                    Err(io::Error::other("first rejected"))
                } else {
                    Ok(ctx)
                }
            },
        );
        assert_eq!(second.ok(), Some(2));

        // More than 16 failing catalogs stops at MAX_CATALOGS = 16 and returns last_error
        let mut many = 0_isize;
        let capped = enumerate_catalogs::<()>(
            || {
                many += 1;
                Some(many)
            },
            |ctx| Err(io::Error::other(format!("fail-{ctx}"))),
        );
        assert_eq!(many, 16);
        assert_eq!(capped.err().map(|e| e.to_string()), Some("fail-16".into()));
    }
}
