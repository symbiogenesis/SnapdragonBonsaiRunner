//! Hardened little-endian slice readers and PE32+ ARM64 layout primitives.

use std::io::{self, Read, Seek, SeekFrom};

pub(crate) const PE_SIGNATURE: u32 = 0x0000_4550;
pub(crate) const MACHINE_ARM64: u16 = 0xaa64;
pub(crate) const PE32_PLUS_MAGIC: u16 = 0x020b;
pub(crate) const SUBSYSTEM_WINDOWS_GUI: u16 = 2;
pub(crate) const MAX_SECTIONS: usize = 96;
pub(crate) const MAX_DIRECTORIES: usize = 16;
pub(crate) const DATA_DIR_EXPORT: usize = 0;
pub(crate) const DATA_DIR_IMPORT: usize = 1;
pub(crate) const DATA_DIR_DELAY_IMPORT: usize = 13;

pub(crate) fn extent<'a>(data: &'a [u8], offset: usize, size: usize) -> crate::Result<&'a [u8]> {
    let end = offset.checked_add(size).ok_or("Binary extent overflow.")?;
    data.get(offset..end)
        .ok_or_else(|| "Binary extent outside retained input.".into())
}

fn array_at<const N: usize>(data: &[u8], offset: usize) -> crate::Result<[u8; N]> {
    extent(data, offset, N)?
        .try_into()
        .map_err(|_| "Binary extent outside retained input.".into())
}

pub(crate) fn u16_at(data: &[u8], offset: usize) -> crate::Result<u16> {
    Ok(u16::from_le_bytes(array_at(data, offset)?))
}

pub(crate) fn u32_at(data: &[u8], offset: usize) -> crate::Result<u32> {
    Ok(u32::from_le_bytes(array_at(data, offset)?))
}

pub(crate) fn u64_at(data: &[u8], offset: usize) -> crate::Result<u64> {
    Ok(u64::from_le_bytes(array_at(data, offset)?))
}

pub(crate) fn i16_at(data: &[u8], offset: usize) -> crate::Result<i16> {
    Ok(i16::from_le_bytes(array_at(data, offset)?))
}

pub(crate) fn i32_at(data: &[u8], offset: usize) -> crate::Result<i32> {
    Ok(i32::from_le_bytes(array_at(data, offset)?))
}

pub(crate) fn usize_at(data: &[u8], offset: usize) -> crate::Result<usize> {
    usize::try_from(u32_at(data, offset)?).map_err(|error| error.to_string())
}

pub(crate) fn c_bytes<'a>(data: &'a [u8], max_len: usize, error: &str) -> crate::Result<&'a [u8]> {
    let end = data
        .iter()
        .take(max_len.saturating_add(1))
        .position(|byte| *byte == 0)
        .ok_or(error)?;
    Ok(&data[..end])
}

/// Bounds-checked sequential little-endian slice reader.
pub(crate) struct LeCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> LeCursor<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(crate) const fn position(&self) -> usize {
        self.position
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.position >= self.bytes.len()
    }

    pub(crate) fn take(&mut self, size: usize) -> crate::Result<&'a [u8]> {
        let slice = extent(self.bytes, self.position, size)?;
        self.position += size;
        Ok(slice)
    }

    pub(crate) fn skip(&mut self, size: usize) -> crate::Result<()> {
        self.take(size).map(|_| ())
    }

    pub(crate) fn read_array<const N: usize>(&mut self) -> crate::Result<[u8; N]> {
        self.take(N)?
            .try_into()
            .map_err(|_| "Binary extent outside retained input.".into())
    }

    pub(crate) fn read_u8(&mut self) -> crate::Result<u8> {
        Ok(self.read_array::<1>()?[0])
    }

    pub(crate) fn read_u16(&mut self) -> crate::Result<u16> {
        Ok(u16::from_le_bytes(self.read_array::<2>()?))
    }

    pub(crate) fn read_u32(&mut self) -> crate::Result<u32> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }
}

#[derive(Debug)]
pub(crate) enum PeProbeError {
    DosRead(io::Error),
    DosSignature,
    Offset,
    Io(io::Error),
    PeSignature,
}

pub(crate) fn parse_dos_pe_offset(
    dos: &[u8; 64],
    length: u64,
    min_pe_bytes: u64,
) -> Result<u64, PeProbeError> {
    let [b'M', b'Z', .., o0, o1, o2, o3] = *dos else {
        return Err(PeProbeError::DosSignature);
    };
    let offset = u64::from(u32::from_le_bytes([o0, o1, o2, o3]));
    if offset < 64
        || offset
            .checked_add(min_pe_bytes)
            .is_none_or(|end| end > length)
    {
        return Err(PeProbeError::Offset);
    }
    Ok(offset)
}

pub(crate) fn read_pe_machine(
    reader: &mut (impl Read + Seek),
    length: u64,
) -> Result<u16, PeProbeError> {
    let mut dos = [0; 64];
    reader.read_exact(&mut dos).map_err(PeProbeError::DosRead)?;
    let offset = parse_dos_pe_offset(&dos, length, 24)?;
    reader
        .seek(SeekFrom::Start(offset))
        .map_err(PeProbeError::Io)?;
    let mut pe = [0; 6];
    reader.read_exact(&mut pe).map_err(PeProbeError::Io)?;
    let [b'P', b'E', 0, 0, m0, m1] = pe else {
        return Err(PeProbeError::PeSignature);
    };
    Ok(u16::from_le_bytes([m0, m1]))
}

pub(crate) fn require_arm64_gui_subsystem(reader: &mut (impl Read + Seek)) -> crate::Result<()> {
    reader.rewind().map_err(|error| error.to_string())?;
    let mut dos = [0; 64];
    reader
        .read_exact(&mut dos)
        .map_err(|error| error.to_string())?;
    let offset = u64::from(u32_at(&dos, 60)?);
    if dos[..2] != *b"MZ" || offset < 64 {
        return Err("Invalid launcher DOS header.".into());
    }
    reader
        .seek(SeekFrom::Start(offset))
        .map_err(|error| error.to_string())?;
    let mut pe = [0; 94];
    reader
        .read_exact(&mut pe)
        .map_err(|error| error.to_string())?;
    if u32_at(&pe, 0)? != PE_SIGNATURE
        || u16_at(&pe, 4)? != MACHINE_ARM64
        || u16_at(&pe, 20)? < 70
        || u16_at(&pe, 24)? != PE32_PLUS_MAGIC
        || u16_at(&pe, 92)? != SUBSYSTEM_WINDOWS_GUI
    {
        return Err("Bundling requires the Windows GUI-subsystem bonsai-launcher image.".into());
    }
    Ok(())
}

pub(crate) trait ByteSource {
    fn byte_len(&self) -> u64;
    fn read_exact_at(&mut self, offset: u64, dst: &mut [u8]) -> crate::Result<()>;
}

impl ByteSource for &[u8] {
    fn byte_len(&self) -> u64 {
        u64::try_from(self.len()).unwrap_or(u64::MAX)
    }

    fn read_exact_at(&mut self, offset: u64, dst: &mut [u8]) -> crate::Result<()> {
        let start = usize::try_from(offset).map_err(|_| "PE extent outside file")?;
        dst.copy_from_slice(extent(self, start, dst.len())?);
        Ok(())
    }
}

pub(crate) struct SeekSource<R> {
    reader: R,
    length: u64,
}

impl<R: Read + Seek> SeekSource<R> {
    pub(crate) const fn new(reader: R, length: u64) -> Self {
        Self { reader, length }
    }
}

impl<R: Read + Seek> ByteSource for SeekSource<R> {
    fn byte_len(&self) -> u64 {
        self.length
    }

    fn read_exact_at(&mut self, offset: u64, dst: &mut [u8]) -> crate::Result<()> {
        let size = u64::try_from(dst.len()).map_err(|_| "PE extent outside file")?;
        if offset.checked_add(size).is_none_or(|end| end > self.length) {
            return Err("PE extent outside file".into());
        }
        self.reader
            .seek(SeekFrom::Start(offset))
            .map_err(|error| error.to_string())?;
        self.reader
            .read_exact(dst)
            .map_err(|error| error.to_string())
    }
}

pub(crate) struct PePolicy {
    pub(crate) min_directories: usize,
    pub(crate) max_directories: usize,
    pub(crate) max_optional_size: usize,
    pub(crate) require_size_of_headers: bool,
    pub(crate) validate_section_layout: bool,
}

impl PePolicy {
    pub(crate) const STATIC_AUDIT: Self = Self {
        min_directories: 14,
        max_directories: MAX_DIRECTORIES,
        max_optional_size: 4096,
        require_size_of_headers: false,
        validate_section_layout: false,
    };

    pub(crate) const PORTABLE: Self = Self {
        min_directories: 0,
        max_directories: MAX_DIRECTORIES,
        max_optional_size: 4096,
        require_size_of_headers: true,
        validate_section_layout: true,
    };
}

pub(crate) struct PeSection {
    name: [u8; 8],
    pub(crate) virtual_size: u64,
    pub(crate) virtual_address: u64,
    pub(crate) raw_size: usize,
    pub(crate) raw_offset: u64,
}

impl PeSection {
    pub(crate) fn name_bytes(&self) -> &[u8] {
        let end = self
            .name
            .iter()
            .rposition(|byte| *byte != 0)
            .map_or(0, |index| index + 1);
        &self.name[..end]
    }
}

pub(crate) struct ImportDescriptor {
    pub(crate) attributes: u32,
    pub(crate) timestamp: u32,
    pub(crate) name_raw: u32,
    pub(crate) ilt_rva: u32,
    pub(crate) iat_rva: u32,
}

pub(crate) struct PeLayout {
    pub(crate) base_address: u64,
    pub(crate) headers_size: u32,
    directories: [(u32, u32); MAX_DIRECTORIES],
    sections: Vec<PeSection>,
}

fn sections_overlap(a: &PeSection, b: &PeSection) -> bool {
    let ranges = |start: u64, size: u64, other_start: u64, other_size: u64| {
        size > 0
            && other_size > 0
            && start < other_start.saturating_add(other_size)
            && other_start < start.saturating_add(size)
    };
    let a_mapped = a
        .virtual_size
        .max(u64::try_from(a.raw_size).unwrap_or(u64::MAX));
    let b_mapped = b
        .virtual_size
        .max(u64::try_from(b.raw_size).unwrap_or(u64::MAX));
    let a_raw = u64::try_from(a.raw_size).unwrap_or(u64::MAX);
    let b_raw = u64::try_from(b.raw_size).unwrap_or(u64::MAX);
    ranges(a.virtual_address, a_mapped, b.virtual_address, b_mapped)
        || ranges(a.raw_offset, a_raw, b.raw_offset, b_raw)
}

impl PeLayout {
    pub(crate) fn parse(source: &mut impl ByteSource, policy: &PePolicy) -> crate::Result<Self> {
        let total = source.byte_len();
        let mut dos = [0_u8; 64];
        source.read_exact_at(0, &mut dos)?;
        let pe_offset = parse_dos_pe_offset(&dos, total, 24).map_err(|error| match error {
            PeProbeError::DosSignature => "Missing DOS signature",
            _ => "PE extent outside file",
        })?;

        let mut coff = [0_u8; 24];
        source.read_exact_at(pe_offset, &mut coff)?;
        let mut coff_cursor = LeCursor::new(&coff);
        if coff_cursor.read_u32()? != PE_SIGNATURE || coff_cursor.read_u16()? != MACHINE_ARM64 {
            return Err("Expected native ARM64 PE".into());
        }
        let section_count = usize::from(coff_cursor.read_u16()?);
        coff_cursor.skip(12)?;
        let optional_size = usize::from(coff_cursor.read_u16()?);
        if section_count > MAX_SECTIONS {
            return Err("Too many PE sections".into());
        }
        if optional_size < 112 || optional_size > policy.max_optional_size {
            return Err("Truncated PE32+ optional header".into());
        }

        let mut optional_buf = [0_u8; 4096];
        let optional = &mut optional_buf[..optional_size];
        source.read_exact_at(pe_offset + 24, optional)?;
        if u16_at(optional, 0)? != PE32_PLUS_MAGIC {
            return Err("Expected PE32+".into());
        }
        let base_address = u64_at(optional, 24)?;
        let headers_size = u32_at(optional, 60)?;
        let directory_count = usize_at(optional, 108)?;
        let max_directories = ((optional_size - 112) / 8).min(policy.max_directories);
        if !(policy.min_directories..=max_directories).contains(&directory_count) {
            return Err("PE directory count exceeds optional header or omits import tables".into());
        }
        let mut directories = [(0_u32, 0_u32); MAX_DIRECTORIES];
        for (index, slot) in directories.iter_mut().take(directory_count).enumerate() {
            *slot = (
                u32_at(optional, 112 + index * 8)?,
                u32_at(optional, 116 + index * 8)?,
            );
        }

        let sections_offset = pe_offset + 24 + u64::try_from(optional_size).unwrap_or(u64::MAX);
        let sections_bytes = section_count * 40;
        let sections_end = sections_offset
            .checked_add(u64::try_from(sections_bytes).unwrap_or(u64::MAX))
            .ok_or("PE extent outside file")?;
        if policy.require_size_of_headers
            && (u64::from(headers_size) < sections_end || u64::from(headers_size) > total)
        {
            return Err("PE headers extent outside file".into());
        }

        let mut sections_buf = [0_u8; MAX_SECTIONS * 40];
        let sections_raw = &mut sections_buf[..sections_bytes];
        source.read_exact_at(sections_offset, sections_raw)?;
        let mut sections_cursor = LeCursor::new(sections_raw);
        let mut sections = Vec::with_capacity(section_count);
        for _ in 0..section_count {
            let name = sections_cursor.read_array::<8>()?;
            let virtual_size_u32 = sections_cursor.read_u32()?;
            let virtual_address_u32 = sections_cursor.read_u32()?;
            let raw_size_u32 = sections_cursor.read_u32()?;
            let raw_offset_u32 = sections_cursor.read_u32()?;
            sections_cursor.skip(16)?;

            if u64::from(raw_offset_u32)
                .checked_add(u64::from(raw_size_u32))
                .is_none_or(|end| end > total)
            {
                return Err("PE extent outside file".into());
            }
            let entry = PeSection {
                name,
                virtual_size: u64::from(virtual_size_u32),
                virtual_address: u64::from(virtual_address_u32),
                raw_size: usize::try_from(raw_size_u32).map_err(|error| error.to_string())?,
                raw_offset: u64::from(raw_offset_u32),
            };
            if policy.validate_section_layout {
                let mapped_size = virtual_size_u32.max(raw_size_u32);
                if virtual_address_u32.checked_add(mapped_size).is_none()
                    || (mapped_size > 0 && virtual_address_u32 < headers_size)
                    || sections.iter().any(|other| sections_overlap(&entry, other))
                {
                    return Err("PE sections overlap or exceed address bounds".into());
                }
            }
            sections.push(entry);
        }

        Ok(Self {
            base_address,
            headers_size,
            directories,
            sections,
        })
    }

    pub(crate) fn directory(&self, index: usize) -> (u32, u32) {
        self.directories.get(index).copied().unwrap_or((0, 0))
    }

    pub(crate) fn sections(&self) -> &[PeSection] {
        &self.sections
    }

    pub(crate) fn map_rva(
        &self,
        rva: u64,
        size: usize,
        allow_header_rvas: bool,
    ) -> crate::Result<(u64, usize)> {
        if allow_header_rvas && rva < u64::from(self.headers_size) {
            let available = u64::from(self.headers_size) - rva;
            if u64::try_from(size).map_err(|error| error.to_string())? <= available {
                return Ok((
                    rva,
                    usize::try_from(available).map_err(|error| error.to_string())?,
                ));
            }
            return Err("PE RVA extent outside raw section".into());
        }
        for section in &self.sections {
            let raw_size = u64::try_from(section.raw_size).map_err(|error| error.to_string())?;
            if rva >= section.virtual_address
                && rva < section.virtual_address + section.virtual_size.max(raw_size)
            {
                let delta = usize::try_from(rva - section.virtual_address)
                    .map_err(|error| error.to_string())?;
                if delta > section.raw_size || size > section.raw_size - delta {
                    return Err("PE RVA extent outside raw section".into());
                }
                let offset = section
                    .raw_offset
                    .checked_add(u64::try_from(delta).map_err(|error| error.to_string())?)
                    .ok_or("PE extent outside file")?;
                return Ok((offset, section.raw_size - delta));
            }
        }
        Err("Unmapped PE RVA".into())
    }

    pub(crate) fn resolve_import_name_rva(
        &self,
        delayed: bool,
        attributes: u32,
        name_raw: u32,
        strict_delay_attributes: bool,
    ) -> crate::Result<u64> {
        let mut rva = u64::from(name_raw);
        if delayed {
            if strict_delay_attributes && attributes & !1 != 0 {
                return Err("Unsupported delay-import descriptor attributes".into());
            }
            if attributes & 1 == 0 {
                rva = rva
                    .checked_sub(self.base_address)
                    .ok_or("Unmapped PE RVA")?;
            }
        }
        Ok(rva)
    }

    pub(crate) fn read_c_bytes_at<'a>(
        &self,
        source: &mut impl ByteSource,
        rva: u64,
        allow_header_rvas: bool,
        scratch: &'a mut [u8],
        error: &str,
    ) -> crate::Result<&'a [u8]> {
        if rva == 0 {
            return Err("Unmapped PE RVA".into());
        }
        let (offset, available) = self.map_rva(rva, 1, allow_header_rvas)?;
        let count = available.min(scratch.len());
        source.read_exact_at(offset, &mut scratch[..count])?;
        let end = scratch[..count]
            .iter()
            .position(|byte| *byte == 0)
            .filter(|end| *end > 0)
            .ok_or(error)?;
        Ok(&scratch[..end])
    }

    pub(crate) fn visit_import_descriptors<S: ByteSource, E: From<String>>(
        &self,
        source: &mut S,
        delayed: bool,
        max_import_bytes: usize,
        max_descriptors: usize,
        allow_header_rvas: bool,
        mut visitor: impl FnMut(&mut S, ImportDescriptor) -> Result<(), E>,
    ) -> Result<(), E> {
        let directory_index = if delayed {
            DATA_DIR_DELAY_IMPORT
        } else {
            DATA_DIR_IMPORT
        };
        let (rva, size_u32) = self.directory(directory_index);
        let size = usize::try_from(size_u32).map_err(|error| error.to_string())?;
        if rva == 0 && size == 0 {
            return Ok(());
        }
        let width = if delayed { 32 } else { 20 };
        if rva == 0 || !(width..=max_import_bytes).contains(&size) {
            return Err(String::from("Invalid import directory extent").into());
        }
        let (start, _) = self.map_rva(u64::from(rva), size, allow_header_rvas)?;
        let mut record = [0_u8; 32];
        for index in 0..(size / width).min(max_descriptors) {
            let offset = start
                .checked_add(u64::try_from(index * width).map_err(|error| error.to_string())?)
                .ok_or_else(|| String::from("PE extent outside file"))?;
            source.read_exact_at(offset, &mut record[..width])?;
            if record[..width].iter().all(|byte| *byte == 0) {
                return Ok(());
            }
            let descriptor = if delayed {
                ImportDescriptor {
                    attributes: u32_at(&record, 0)?,
                    timestamp: 0,
                    name_raw: u32_at(&record, 4)?,
                    iat_rva: u32_at(&record, 12)?,
                    ilt_rva: u32_at(&record, 16)?,
                }
            } else {
                ImportDescriptor {
                    attributes: 0,
                    ilt_rva: u32_at(&record, 0)?,
                    timestamp: u32_at(&record, 4)?,
                    name_raw: u32_at(&record, 12)?,
                    iat_rva: u32_at(&record, 16)?,
                }
            };
            visitor(source, descriptor)?;
        }
        Err(String::from("Unterminated import directory").into())
    }
}
