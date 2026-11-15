//! Basic PE machine probing; file admission and architecture policy stay with callers.

pub(super) use crate::pe_format::{PeProbeError as Error, read_pe_machine as machine};

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor};

    use super::{Error, machine};

    fn image(architecture: u16) -> Vec<u8> {
        let mut bytes = vec![0; 104];
        bytes[..2].copy_from_slice(b"MZ");
        bytes[60..64].copy_from_slice(&80_u32.to_le_bytes());
        bytes[80..84].copy_from_slice(b"PE\0\0");
        bytes[84..86].copy_from_slice(&architecture.to_le_bytes());
        bytes
    }

    fn probe(bytes: &[u8]) -> Result<u16, Error> {
        machine(
            &mut Cursor::new(bytes),
            u64::try_from(bytes.len()).map_err(|_| Error::Offset)?,
        )
    }

    #[test]
    fn pe_machine_preserves_architectures_without_admission_policy() {
        for architecture in [0xaa64, 0xa641, 0xa64e, 0x8664, 0x014c, 0xffff] {
            assert_eq!(probe(&image(architecture)).ok(), Some(architecture));
        }
    }

    #[test]
    fn pe_machine_rejects_truncation_and_offsets_outside_the_header_extent() {
        let bytes = image(0xaa64);
        for length in 0..bytes.len() {
            assert!(probe(&bytes[..length]).is_err());
        }
        for offset in [0_u32, 63, 81, u32::MAX] {
            let mut bytes = bytes.clone();
            bytes[60..64].copy_from_slice(&offset.to_le_bytes());
            assert!(matches!(probe(&bytes), Err(Error::Offset)));
        }
    }

    #[test]
    fn pe_machine_distinguishes_dos_and_pe_signatures_and_read_failure() {
        let mut bytes = image(0xaa64);
        bytes[0] = 0;
        assert!(matches!(probe(&bytes), Err(Error::DosSignature)));
        bytes[0] = b'M';
        bytes[80] = 0;
        assert!(matches!(probe(&bytes), Err(Error::PeSignature)));
        assert!(
            matches!(probe(&[]), Err(Error::DosRead(error)) if error.kind() == io::ErrorKind::UnexpectedEof)
        );
        // A stale/inaccurate admitted length cannot turn truncated bytes into a
        // valid machine observation; the final header read must still succeed.
        assert!(
            matches!(machine(&mut Cursor::new(&bytes[..82]), 104), Err(Error::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof)
        );
    }
}
