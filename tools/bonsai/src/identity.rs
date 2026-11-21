//! Canonical lowercase identities for maintenance receipts and reviewed pins.
#![expect(
    clippy::redundant_pub_crate,
    reason = "Private maintenance modules expose only a crate-local implementation API."
)]

pub(crate) struct Hasher(bonsai_native_hash::Hasher);

impl Hasher {
    pub(crate) fn new() -> crate::Result<Self> {
        bonsai_native_hash::Hasher::new()
            .map(Self)
            .map_err(|error| error.to_string())
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) -> crate::Result<()> {
        self.0.update(bytes).map_err(|error| error.to_string())
    }

    pub(crate) fn finalize_bytes(self) -> crate::Result<[u8; 32]> {
        self.0.finish().map_err(|error| error.to_string())
    }

    pub(crate) fn finalize_hex(self) -> crate::Result<String> {
        Ok(hex(&self.finalize_bytes()?))
    }
}

pub(super) fn hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

pub(super) fn sha256(bytes: &[u8]) -> String {
    #[cfg(windows)]
    {
        bonsai_windows_credentials::Sha256::digest_bytes(bytes)
            .map_or_else(|_| String::new(), |digest| hex(&digest))
    }
    #[cfg(not(windows))]
    {
        bonsai_native_hash::hash_bytes(bytes).map_or_else(|_| String::new(), |digest| hex(&digest))
    }
}

pub(super) fn is_lower_hex(text: &str, length: usize) -> bool {
    text.len() == length
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Streams a reader once, hashing every byte while enforcing optional length,
/// expected lowercase SHA-256 digest and cancellation checkpoints.
pub(crate) fn hash_reader_checked<R: std::io::Read + ?Sized>(
    reader: &mut R,
    expected_bytes: Option<u64>,
    expected_sha256: Option<&str>,
    checkpoint: &mut (impl FnMut() -> crate::Result<()> + ?Sized),
) -> crate::Result<(u64, String)> {
    let mut hasher = Hasher::new()?;
    let buffer_len = expected_bytes.map_or(65_536, |bytes| {
        usize::try_from(bytes)
            .unwrap_or(1_048_576)
            .clamp(1, 1_048_576)
    });
    let mut buffer = vec![0_u8; buffer_len];
    let mut total = 0_u64;
    loop {
        checkpoint()?;
        let count = match reader.read(&mut buffer) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.to_string()),
        };
        if count == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(count).map_err(|error| error.to_string())?)
            .ok_or("Hashed byte count overflowed.")?;
        if expected_bytes.is_some_and(|limit| total > limit) {
            return Err("Stream exceeded its expected byte length.".into());
        }
        hasher.update(&buffer[..count])?;
    }
    checkpoint()?;
    if expected_bytes.is_some_and(|limit| total != limit) {
        return Err("Stream length differs from its expected byte length.".into());
    }
    let digest = hasher.finalize_hex()?;
    if let Some(expected) = expected_sha256
        && digest != expected
    {
        return Err("Stream SHA-256 differs from its expected digest.".into());
    }
    Ok((total, digest))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{hash_reader_checked, sha256};

    #[test]
    fn hash_reader_checked_verifies_length_digest_and_checkpoints() -> crate::Result<()> {
        let payload = b"bonsai-stream";
        let digest = sha256(payload);
        let mut cursor = Cursor::new(payload);
        let mut ticks = 0_u32;
        let (bytes, computed) = hash_reader_checked(
            &mut cursor,
            Some(u64::try_from(payload.len()).map_err(|error| error.to_string())?),
            Some(&digest),
            &mut || {
                ticks += 1;
                Ok(())
            },
        )?;
        assert_eq!(bytes, 13);
        assert_eq!(computed, digest);
        assert!(ticks >= 2);
        assert!(
            hash_reader_checked(
                &mut Cursor::new(payload),
                Some(12),
                Some(&digest),
                &mut || Ok(())
            )
            .is_err()
        );
        assert!(
            hash_reader_checked(
                &mut Cursor::new(payload),
                Some(13),
                Some(&"0".repeat(64)),
                &mut || Ok(())
            )
            .is_err()
        );
        assert!(
            hash_reader_checked(&mut Cursor::new(payload), None, None, &mut || Err(
                "cancelled".into()
            ))
            .is_err()
        );
        Ok(())
    }
}
