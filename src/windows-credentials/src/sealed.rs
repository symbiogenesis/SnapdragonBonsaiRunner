//! User-scoped Windows encryption for optional private conversation recovery.
use std::ops::Deref;
use std::{fmt, io, ptr};
use windows_sys::Win32::{
    Foundation::LocalFree,
    Security::Cryptography::{
        CRYPT_INTEGER_BLOB, CRYPTPROTECT_UI_FORBIDDEN, CryptProtectData, CryptUnprotectData,
    },
};
use zeroize::Zeroize as _;

const MAX_PLAIN: usize = 8 * 1024 * 1024;
const MAX_SEALED: usize = MAX_PLAIN + 65_536;
const PURPOSE: &[u8] = b"SwiftBonsai2.ChatRecovery.v1";

/// Owned DPAPI plaintext buffer that zeroizes its contents when dropped.
pub struct ZeroizingBytes(Vec<u8>);

impl Deref for ZeroizingBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[u8]> for ZeroizingBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for ZeroizingBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ZeroizingBytes")
            .field(&self.0.len())
            .finish()
    }
}

impl Drop for ZeroizingBytes {
    fn drop(&mut self) {
        self.0.as_mut_slice().zeroize();
    }
}

/// Encrypts bounded conversation data for the current Windows user with DPAPI.
///
/// The result can be stored outside Git without writing plaintext to disk. This
/// is user-scoped protection, not isolation from other processes of that user.
/// Windows owns the encryption keys; no password or machine-wide scope is used.
///
/// # Errors
/// Rejects empty data, data over 8 MiB, or a Windows data-protection failure.
pub fn protect_private_data(data: &[u8]) -> io::Result<Vec<u8>> {
    transform(data, true)
}

/// Authenticates and decrypts this application's DPAPI conversation data into a
/// zeroizing buffer.
///
/// # Errors
/// Rejects empty, oversized, corrupted or inaccessible encrypted data, including
/// data protected for another Windows user. Plaintext is bounded to 8 MiB.
pub fn unprotect_private_data_zeroizing(data: &[u8]) -> io::Result<ZeroizingBytes> {
    transform(data, false).map(ZeroizingBytes)
}

/// Authenticates and decrypts this application's DPAPI conversation data.
///
/// # Errors
/// Rejects empty, oversized, corrupted or inaccessible encrypted data, including
/// data protected for another Windows user. Plaintext is bounded to 8 MiB.
pub fn unprotect_private_data(data: &[u8]) -> io::Result<Vec<u8>> {
    unprotect_private_data_zeroizing(data).map(|mut bytes| std::mem::take(&mut bytes.0))
}

fn borrowed(data: &[u8]) -> io::Result<CRYPT_INTEGER_BLOB> {
    Ok(CRYPT_INTEGER_BLOB {
        cbData: u32::try_from(data.len()).map_err(io::Error::other)?,
        pbData: data.as_ptr().cast_mut(),
    })
}

struct Output(CRYPT_INTEGER_BLOB);

impl Drop for Output {
    fn drop(&mut self) {
        if !self.0.pbData.is_null() {
            // SAFETY: DPAPI returned this allocation and its byte length. It is
            // uniquely owned here and remains live until LocalFree below.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    self.0.pbData,
                    usize::try_from(self.0.cbData).unwrap_or_default(),
                )
            };
            bytes.zeroize();
            // SAFETY: DPAPI allocations require LocalFree, exactly once.
            unsafe { LocalFree(self.0.pbData.cast()) };
        }
    }
}

fn transform(data: &[u8], protect: bool) -> io::Result<Vec<u8>> {
    let limit = if protect { MAX_PLAIN } else { MAX_SEALED };
    if data.is_empty() || data.len() > limit {
        return Err(super::invalid("Private recovery data exceeds its bound."));
    }
    let input = borrowed(data)?;
    let entropy = borrowed(PURPOSE)?;
    let mut output = Output(CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: ptr::null_mut(),
    });
    let success = if protect {
        // SAFETY: input and entropy are immutable live slices. DPAPI does not
        // modify them despite the legacy blob's mutable pointer type. Output
        // is an owned empty slot, and prompts and machine-wide scope are off.
        unsafe {
            CryptProtectData(
                &raw const input,
                ptr::null(),
                &raw const entropy,
                ptr::null(),
                ptr::null(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &raw mut output.0,
            )
        }
    } else {
        // SAFETY: the same bounded input/output ownership contract applies to
        // decryption. No description allocation or interactive prompt is used.
        unsafe {
            CryptUnprotectData(
                &raw const input,
                ptr::null_mut(),
                &raw const entropy,
                ptr::null(),
                ptr::null(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &raw mut output.0,
            )
        }
    };
    if success == 0 {
        return Err(io::Error::last_os_error());
    }
    let length = usize::try_from(output.0.cbData).map_err(io::Error::other)?;
    if output.0.pbData.is_null() || length > if protect { MAX_SEALED } else { MAX_PLAIN } {
        return Err(io::Error::other(
            "Windows returned invalid private recovery data.",
        ));
    }
    // SAFETY: successful DPAPI output owns exactly cbData initialized bytes;
    // copy before Output scrubs and frees the allocation.
    Ok(unsafe { std::slice::from_raw_parts(output.0.pbData, length) }.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_recovery_round_trip_and_tamper_rejection() -> io::Result<()> {
        let plain = "A private draft: 🌱 界".as_bytes();
        let mut sealed = protect_private_data(plain)?;
        assert!(!sealed.windows(plain.len()).any(|window| window == plain));
        assert_eq!(unprotect_private_data(&sealed)?, plain);
        let zeroized = unprotect_private_data_zeroizing(&sealed)?;
        assert_eq!(&*zeroized, plain);
        assert_eq!(zeroized.as_ref(), plain);
        assert!(!format!("{zeroized:?}").contains("private draft"));
        drop(zeroized);
        if let Some(last) = sealed.last_mut() {
            *last ^= 1;
        }
        assert!(unprotect_private_data(&sealed).is_err());
        assert!(unprotect_private_data_zeroizing(&sealed).is_err());
        assert!(protect_private_data(&[]).is_err());
        assert!(unprotect_private_data(b"not a Windows protected blob").is_err());
        assert!(unprotect_private_data_zeroizing(b"not a Windows protected blob").is_err());
        assert!(protect_private_data(&vec![0; MAX_PLAIN + 1]).is_err());
        Ok(())
    }
}
