//! Streaming SHA-256 through Windows' CPU-dispatched CNG implementation.

use std::{io, ptr};

use windows_sys::Win32::Security::Cryptography::{BCRYPT_SHA256_ALG_HANDLE, BCryptHash};

/// A streaming SHA-256 hash using Windows' CPU-dispatched CNG implementation.
///
/// Streaming delegates to [`bonsai_native_hash::Hasher`]; one-shot slice hashing
/// uses stateless `BCryptHash`.
pub struct Sha256(bonsai_native_hash::Hasher);

impl Sha256 {
    /// Create an empty SHA-256 state without opening an algorithm provider.
    ///
    /// # Errors
    /// Returns an unavailable provider or native allocation failure.
    pub fn new() -> io::Result<Self> {
        bonsai_native_hash::Hasher::new().map(Self)
    }

    /// Append exactly the supplied bytes to this hash.
    ///
    /// # Errors
    /// Rejects an input exceeding the native length bound or a provider failure.
    pub fn update(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.0.update(bytes)
    }

    /// Consume this state and return its exact 32-byte SHA-256 digest.
    ///
    /// # Errors
    /// Returns a native finalization failure. The hash is destroyed on all paths.
    pub fn finish(self) -> io::Result<[u8; 32]> {
        self.0.finish()
    }

    /// Compute the 32-byte SHA-256 digest of a slice in one call via Windows CNG.
    ///
    /// # Errors
    /// Returns a provider allocation, update or finalization failure.
    pub fn digest_bytes(bytes: &[u8]) -> io::Result<[u8; 32]> {
        if let Ok(length) = u32::try_from(bytes.len()) {
            let mut digest = [0_u8; 32];
            let input = if bytes.is_empty() {
                ptr::null()
            } else {
                bytes.as_ptr()
            };
            // SAFETY: BCRYPT_SHA256_ALG_HANDLE is the OS SHA-256 pseudo-handle,
            // input is readable for length bytes (or null when empty), and
            // digest is writable for the exact 32-byte SHA-256 output size.
            check(unsafe {
                BCryptHash(
                    BCRYPT_SHA256_ALG_HANDLE,
                    ptr::null(),
                    0,
                    input,
                    length,
                    digest.as_mut_ptr(),
                    32,
                )
            })?;
            return Ok(digest);
        }
        let mut hasher = Self::new()?;
        for chunk in bytes.chunks(0x4000_0000) {
            hasher.update(chunk)?;
        }
        hasher.finish()
    }
}

fn check(status: i32) -> io::Result<()> {
    if status >= 0 {
        Ok(())
    } else {
        Err(io::Error::other("Windows SHA-256 operation failed."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(parts: &[&[u8]]) -> io::Result<String> {
        let mut hash = Sha256::new()?;
        for part in parts {
            hash.update(part)?;
        }
        Ok(super::super::hex(&hash.finish()?))
    }

    #[test]
    fn cng_sha256_matches_known_empty_short_and_multiblock_vectors() -> io::Result<()> {
        assert_eq!(
            digest(&[])?,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            super::super::hex(&Sha256::digest_bytes(b"")?),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest(&[b"", b"a", b"", b"bc", b""])?,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            super::super::hex(&Sha256::digest_bytes(b"abc")?),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest(&[b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"])?,
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            super::super::hex(&Sha256::digest_bytes(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )?),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let block = [b'a'; 1000];
        let mut hash = Sha256::new()?;
        for _ in 0..1000 {
            hash.update(&block)?;
        }
        assert_eq!(
            super::super::hex(&hash.finish()?),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
        Ok(())
    }
}
