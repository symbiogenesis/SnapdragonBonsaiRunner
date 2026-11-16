//! Bounded canonical serialization, including allocation growth before rejection.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep the canonical serializer explicitly crate scoped."
)]

#[cfg(test)]
use crate::MAX_BODY_BYTES;
use crate::Rejection;
use std::io::{self, Write};

struct Buffer {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl Buffer {
    fn with_hint_and_limit(hint: usize, limit: usize) -> Result<Self, Rejection> {
        let initial = hint.min(limit);
        let mut bytes = Vec::new();
        if initial > 0 {
            bytes
                .try_reserve_exact(initial)
                .map_err(|_| Rejection::BodySize)?;
        }
        Ok(Self {
            bytes,
            limit,
            exceeded: false,
        })
    }

    #[cfg(test)]
    const fn with_limit(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            exceeded: false,
        }
    }
}

impl Write for Buffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other(
                "Canonical request exceeds its byte limit.",
            ));
        }
        let needed = self.bytes.len() + bytes.len();
        if needed > self.bytes.capacity() {
            // Vec's ordinary geometric growth can reserve nearly twice the
            // accepted body size. Cap every requested allocation, including
            // the final write that will be rejected for fixed policy overhead.
            let min_cap = 128.min(self.limit);
            let target = needed
                .max(self.bytes.capacity().saturating_mul(2))
                .clamp(min_cap, self.limit);
            self.bytes
                .try_reserve_exact(target - self.bytes.len())
                .map_err(|_| io::Error::other("Could not allocate canonical request."))?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn serialize(value: &impl serde::Serialize) -> Result<Vec<u8>, Rejection> {
    serialize_with_hint_and_limit(value, 0, MAX_BODY_BYTES)
}

pub(super) fn serialize_with_hint_and_limit(
    value: &impl serde::Serialize,
    hint: usize,
    limit: usize,
) -> Result<Vec<u8>, Rejection> {
    let mut output = Buffer::with_hint_and_limit(hint, limit)?;
    serde_json::to_writer(&mut output, value).map_err(|_| {
        if output.exceeded {
            Rejection::BodySize
        } else {
            Rejection::ChatSchema
        }
    })?;
    Ok(output.bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_buffer_caps_growth_and_rejects_without_retaining_excess() -> io::Result<()> {
        let mut output = Buffer::with_limit(MAX_BODY_BYTES);
        // Awkward writes make ordinary Vec doubling overshoot the power-of-two
        // body limit on its final growth, rather than landing exactly on it.
        let chunk = [b'x'; 4093];
        for _ in 0..MAX_BODY_BYTES / chunk.len() {
            output.write_all(&chunk)?;
            assert!(output.bytes.capacity() <= MAX_BODY_BYTES);
        }
        output.write_all(&chunk[..MAX_BODY_BYTES % chunk.len()])?;
        assert!(output.bytes.capacity() <= MAX_BODY_BYTES);
        assert_eq!(output.bytes.len(), MAX_BODY_BYTES);
        assert!(output.write_all(b"x").is_err());
        assert!(output.exceeded);
        assert_eq!(output.bytes.len(), MAX_BODY_BYTES);
        Ok(())
    }

    #[test]
    fn canonical_buffer_preserves_json_bytes_and_exact_limit() -> Result<(), serde_json::Error> {
        let text = "雪🌱\n\u{0000}\"\\";
        let expected = serde_json::to_vec(text)?;
        assert_eq!(serialize(&text), Ok(expected));
        assert_eq!(
            serialize(&"x".repeat(MAX_BODY_BYTES - 2)).map(|bytes| bytes.len()),
            Ok(MAX_BODY_BYTES)
        );
        assert_eq!(
            serialize(&"x".repeat(MAX_BODY_BYTES - 1)),
            Err(Rejection::BodySize)
        );
        Ok(())
    }

    #[test]
    fn canonical_buffer_hint_preallocates_within_limit_and_rejects_excess() -> Result<(), Rejection>
    {
        let buffer = Buffer::with_hint_and_limit(512, 1024)?;
        assert!(buffer.bytes.capacity() >= 512);
        assert!(buffer.bytes.capacity() <= 1024);

        let clamped = Buffer::with_hint_and_limit(8192, 1024)?;
        assert_eq!(clamped.bytes.capacity(), 1024);

        let payload = "hinted-json-payload-雪";
        let hinted = serialize_with_hint_and_limit(&payload, 64, 256)?;
        let expected = serialize(&payload)?;
        assert_eq!(hinted, expected);
        assert!(hinted.capacity() >= 64);
        assert!(hinted.capacity() <= 256);

        assert_eq!(
            serialize_with_hint_and_limit(&"x".repeat(65), 128, 64),
            Err(Rejection::BodySize)
        );
        Ok(())
    }
}
