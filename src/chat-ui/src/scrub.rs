//! Explicit zeroizing wrapper for transient plaintext credentials and DPAPI buffers.

#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "Internal HTTP and recovery modules share this crate-private byte scrubber."
)]

use std::ops::{Deref, DerefMut};

/// Owned byte buffer whose storage is overwritten before deallocation.
#[derive(Default)]
pub(crate) struct ScrubbingBytes(Vec<u8>);

impl ScrubbingBytes {
    /// Wrap an owned plaintext or credential vector so every byte is scrubbed on drop.
    pub(crate) const fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

impl From<Vec<u8>> for ScrubbingBytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

impl Deref for ScrubbingBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl DerefMut for ScrubbingBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl AsRef<[u8]> for ScrubbingBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Drop for ScrubbingBytes {
    fn drop(&mut self) {
        self.0.fill(0);
        std::hint::black_box(&*self.0);
    }
}
