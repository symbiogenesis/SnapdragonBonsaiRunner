//! Strict bounded chunk framing for the documented local PCM stream.

use super::{Failure, Result};
use crate::http::{AllowedFraming, BodyPolicy, BodyReader, HeadPolicy, TransportError};
use std::io;

const MAX_PCM: usize = 48_000 * 2 * 120;
#[cfg(test)]
const INVALID: Failure = Failure::Message("The local speech stream was incomplete or invalid.");

struct PcmSource<R, C> {
    read: R,
    check: C,
    failure: Option<Failure>,
}

impl<R, C> PcmSource<R, C>
where
    C: FnMut() -> Result<()>,
{
    fn poll(&mut self) -> Result<()> {
        (self.check)()
    }

    fn check(&mut self) -> io::Result<()> {
        if let Err(failure) = (self.check)() {
            self.failure = Some(failure);
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "Local PCM check cancelled",
            ));
        }
        Ok(())
    }
}

impl<R, C> io::Read for PcmSource<R, C>
where
    R: FnMut(&mut [u8]) -> Result<usize>,
    C: FnMut() -> Result<()>,
{
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.check()?;
        let count = match (self.read)(bytes) {
            Ok(count) => count,
            Err(failure) => {
                self.failure = Some(failure);
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "Local PCM read failed",
                ));
            }
        };
        self.check()?;
        Ok(count)
    }
}

pub(super) fn decode(
    read: impl FnMut(&mut [u8]) -> Result<usize>,
    check: impl FnMut() -> Result<()>,
    mut sink: impl FnMut(&[u8]) -> Result<()>,
    map_error: fn(TransportError) -> Failure,
) -> Result<()> {
    let stream_error = |transport| {
        if matches!(transport, TransportError::Status(_)) {
            map_error(transport)
        } else {
            map_error(TransportError::InvalidChunkFraming)
        }
    };
    let head_policy = HeadPolicy {
        framing: AllowedFraming::ChunkedOnly,
        require_connection_close: true,
        allow_htab: false,
    };
    let body_policy = BodyPolicy {
        chunk_line_limit: 10,
        max_body: MAX_PCM,
        require_even_chunks: true,
        require_non_empty: true,
    };
    let mut source = PcmSource {
        read,
        check,
        failure: None,
    };
    // Run caller cancellation check before initial socket header fill.
    source.poll()?;
    let (_head, mut body) = match BodyReader::new(
        &mut source,
        16_384,
        head_policy,
        |code| {
            if code == 200 {
                Ok(MAX_PCM)
            } else {
                Err(TransportError::Status(code))
            }
        },
        |_, value| value.eq_ignore_ascii_case(b"audio/pcm"),
        body_policy,
    ) {
        Ok(parts) => parts,
        Err(transport) => {
            return Err(source
                .failure
                .take()
                .unwrap_or_else(|| stream_error(transport)));
        }
    };
    let mut carry: Option<u8> = None;
    loop {
        body.inner_mut().poll()?;
        match body.next_payload() {
            Ok(Some(bytes)) => {
                let mut rest = bytes;
                if let Some(first) = carry.take()
                    && let Some((&second, tail)) = rest.split_first()
                {
                    sink(&[first, second])?;
                    rest = tail;
                }
                let even = rest.len() & !1;
                if even != 0 {
                    sink(&rest[..even])?;
                }
                if rest.len() > even {
                    carry = Some(rest[even]);
                }
            }
            Ok(None) => {
                if carry.is_some() {
                    return Err(stream_error(TransportError::InvalidChunkFraming));
                }
                return body.verify_eof().map_err(|transport| {
                    body.inner_mut()
                        .failure
                        .take()
                        .unwrap_or_else(|| stream_error(transport))
                });
            }
            Err(transport) => {
                return Err(body
                    .inner_mut()
                    .failure
                    .take()
                    .unwrap_or_else(|| stream_error(transport)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Failure, INVALID, MAX_PCM};
    use crate::canary::map_asr_transport;
    use std::{cell::Cell, io::Read};

    const HEAD: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";

    fn decode(
        read: impl FnMut(&mut [u8]) -> super::Result<usize>,
        check: impl FnMut() -> super::Result<()>,
        sink: impl FnMut(&[u8]) -> super::Result<()>,
    ) -> super::Result<()> {
        super::decode(read, check, sink, map_asr_transport)
    }

    #[test]
    fn pcm_span_lends_ready_samples_without_consuming_chunk_framing() {
        let bytes = [HEAD, b"2\r\n\x01\x80"].concat();
        let mut reader = bytes.as_slice();
        let mut delivered = Vec::new();
        let result = decode(
            |data| reader.read(data).map_err(|_| Failure::Cancelled),
            || Ok(()),
            |chunk| {
                delivered.extend_from_slice(chunk);
                Ok(())
            },
        );
        assert_eq!(delivered, [1, 128]);
        assert_eq!(result, Err(INVALID));
    }

    #[test]
    fn pcm_split_sample_observes_cancellation_before_refill() {
        let bytes = [HEAD, b"2\r\n\x01"].concat();
        let mut reader = bytes.as_slice();
        let reads = Cell::new(0);
        let cancelled = Cell::new(false);
        let result = decode(
            |data| {
                reads.set(reads.get() + 1);
                let count = reader.read(data).map_err(|_| Failure::Cancelled)?;
                if count == 0 {
                    cancelled.set(true);
                }
                Ok(count)
            },
            || {
                if cancelled.get() {
                    Err(Failure::Cancelled)
                } else {
                    Ok(())
                }
            },
            |_| Ok(()),
        );
        assert_eq!(result, Err(Failure::Cancelled));
        assert_eq!(reads.get(), 2);
    }

    #[test]
    fn pcm_stream_requires_terminal_chunk_and_rejects_ambiguous_framing() {
        for (body, valid) in [
            (b"4\r\n\x01\x00\x02\x00\r\n0\r\n\r\n".as_slice(), true),
            (b"4\r\n\x01\x00\x02\x00\r\n".as_slice(), false),
            (b"3\r\nabc\r\n0\r\n\r\n".as_slice(), false),
            (b"0\r\n\r\n".as_slice(), false),
            (b"2\r\nxx\r\n0\r\n\r\nextra".as_slice(), false),
            (b"2;ext=x\r\nxx\r\n0\r\n\r\n".as_slice(), false),
            (b"2\r\nxx\r\n0\r\nX: y\r\n\r\n".as_slice(), false),
            (b"ffffffff\r\n".as_slice(), false),
        ] {
            let bytes = [HEAD, body].concat();
            let mut reader = bytes.as_slice();
            let mut pcm = Vec::new();
            let result = decode(
                |data| reader.read(data).map_err(|_| Failure::Cancelled),
                || Ok(()),
                |chunk| {
                    pcm.extend_from_slice(chunk);
                    Ok(())
                },
            );
            assert_eq!(result.is_ok(), valid);
            if valid {
                assert_eq!(pcm, [1, 0, 2, 0]);
            }
        }
        let malformed = [
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n".as_slice(),
            &HEAD[17..],
            b"2\r\nxx\r\n0\r\n\r\n",
        ]
        .concat();
        let mut reader = malformed.as_slice();
        assert!(
            decode(
                |data| reader.read(data).map_err(|_| Failure::Cancelled),
                || Ok(()),
                |_| Ok(())
            )
            .is_err()
        );
    }

    #[test]
    fn pcm_stream_delivers_before_completion_and_propagates_sink_cancellation() {
        let bytes = [HEAD, b"2\r\nxx\r\n"].concat();
        let mut reader = bytes.as_slice();
        let mut delivered = false;
        let result = decode(
            |data| reader.read(data).map_err(|_| Failure::Cancelled),
            || Ok(()),
            |_| {
                delivered = true;
                Err(Failure::Cancelled)
            },
        );
        assert!(delivered);
        assert_eq!(result, Err(Failure::Cancelled));
    }

    #[test]
    fn pcm_framing_reuses_one_buffer_without_byte_sized_source_reads() -> Result<(), Failure> {
        let bytes = [HEAD, b"2\r\nxx\r\n2\r\nyy\r\n0\r\n\r\n"].concat();
        let mut reader = bytes.as_slice();
        let mut reads = 0;
        let mut pcm = Vec::new();
        decode(
            |data| {
                assert_eq!(data.len(), 16_384);
                reads += 1;
                reader.read(data).map_err(|_| Failure::Cancelled)
            },
            || Ok(()),
            |chunk| {
                pcm.extend_from_slice(chunk);
                Ok(())
            },
        )?;
        assert_eq!(pcm, b"xxyy");
        assert_eq!(reads, 2, "one source read plus the required EOF check");
        Ok(())
    }

    #[test]
    fn pcm_ready_prefix_arrives_before_remaining_chunk_or_framing() -> Result<(), Failure> {
        let bytes = [HEAD, b"8\r\n\x01\x00\x02\x00\x03\x00\x04\x00\r\n0\r\n\r\n"].concat();
        let prefix = HEAD.len() + 3 + 4;
        let mut offset = 0;
        let delivered = Cell::new(false);
        let mut pcm = Vec::new();
        decode(
            |data| {
                let limit = if offset < prefix {
                    prefix
                } else {
                    assert!(
                        delivered.get(),
                        "ready samples must precede another source read"
                    );
                    bytes.len()
                };
                let count = data.len().min(limit - offset);
                data[..count].copy_from_slice(&bytes[offset..offset + count]);
                offset += count;
                Ok(count)
            },
            || Ok(()),
            |chunk| {
                assert_eq!(chunk.len() % 2, 0);
                delivered.set(true);
                pcm.extend_from_slice(chunk);
                Ok(())
            },
        )?;
        assert_eq!(pcm, [1, 0, 2, 0, 3, 0, 4, 0]);
        Ok(())
    }

    #[test]
    fn pcm_fragmentation_preserves_complete_samples_and_terminal_validation() -> Result<(), Failure>
    {
        let bytes = [
            HEAD,
            b"6\r\n\x01\x80\x02\x00\xff\x7f\r\n2\r\n\x00\x80\r\n0\r\n\r\n",
        ]
        .concat();
        for fragment in [1, 2, 3, 7, 16_384] {
            let mut reader = bytes.as_slice();
            let mut pcm = Vec::new();
            decode(
                |data| {
                    let count = fragment.min(data.len());
                    reader
                        .read(&mut data[..count])
                        .map_err(|_| Failure::Cancelled)
                },
                || Ok(()),
                |chunk| {
                    assert_ne!(chunk.len(), 0);
                    assert_eq!(chunk.len() % 2, 0);
                    pcm.extend_from_slice(chunk);
                    Ok(())
                },
            )?;
            assert_eq!(pcm, [1, 128, 2, 0, 255, 127, 0, 128]);
        }
        Ok(())
    }

    #[test]
    fn pcm_buffered_data_still_observes_cancellation_and_rejects_truncation() {
        let bytes = [HEAD, b"2\r\nxx\r\n2\r\nyy\r\n0\r\n\r\n"].concat();
        let mut reader = bytes.as_slice();
        let cancelled = Cell::new(false);
        let mut delivered = 0;
        let result = decode(
            |data| reader.read(data).map_err(|_| Failure::Cancelled),
            || {
                if cancelled.get() {
                    Err(Failure::Cancelled)
                } else {
                    Ok(())
                }
            },
            |_| {
                delivered += 1;
                cancelled.set(true);
                Ok(())
            },
        );
        assert_eq!(result, Err(Failure::Cancelled));
        assert_eq!(delivered, 1);
        for body in [
            b"4\r\nxx".as_slice(),
            b"4\r\nxxx".as_slice(),
            b"2\r\nxxX\n0\r\n\r\n".as_slice(),
            b"2\r\nxx\r\n0\r\n\r\nextra".as_slice(),
            format!("{:x}\r\n", MAX_PCM + 2).as_bytes(),
        ] {
            let bytes = [HEAD, body].concat();
            let mut reader = bytes.as_slice();
            assert_eq!(
                decode(
                    |data| reader.read(data).map_err(|_| Failure::Cancelled),
                    || Ok(()),
                    |_| Ok(()),
                ),
                Err(INVALID)
            );
        }
    }
}
