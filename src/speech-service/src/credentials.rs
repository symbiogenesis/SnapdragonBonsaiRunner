//! Retained private credential reader; authentication material has no Debug view.

#[cfg(windows)]
pub(super) use bonsai_windows_credentials::RetainedCredential as Credential;

#[cfg(not(windows))]
use std::{io, path::Path};

#[cfg(not(windows))]
enum Never {}

#[cfg(not(windows))]
pub(super) struct Credential(Never);

#[cfg(not(windows))]
impl Credential {
    pub(super) fn open(_path: &Path) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Protected companion credentials require Windows.",
        ))
    }

    pub(super) const fn bytes(&self) -> &[u8; 64] {
        match self.0 {}
    }
}
