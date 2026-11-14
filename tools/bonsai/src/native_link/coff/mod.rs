//! Bounded COFF symbol observations, never a source or linked-code absence proof.

use crate::Result;

const SYMBOL_LIMIT: usize = 1_000_000;
const NAME_LIMIT: usize = 4096;
const MEMBER_LIMIT: usize = 65_536;
const BIGOBJ_CLASS: [u8; 16] = [
    0xc7, 0xa1, 0xba, 0xd1, 0xee, 0xba, 0xa9, 0x4b, 0xaf, 0x20, 0xfa, 0xf6, 0x6a, 0xa4, 0xdc, 0xb8,
];

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Observation {
    pub(super) member_offset: Option<usize>,
    pub(super) symbol: String,
    pub(super) binding: &'static str,
}

fn extent(data: &[u8], offset: usize, size: usize) -> Result<&[u8]> {
    data.get(offset..offset.checked_add(size).ok_or("COFF extent overflow.")?)
        .ok_or_else(|| "COFF extent outside retained input.".into())
}

fn u16_at(data: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        extent(data, offset, 2)?
            .try_into()
            .map_err(|_| "COFF u16.")?,
    ))
}

fn u32_at(data: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        extent(data, offset, 4)?
            .try_into()
            .map_err(|_| "COFF u32.")?,
    ))
}

fn size_at(data: &[u8], offset: usize) -> Result<usize> {
    usize::try_from(u32_at(data, offset)?).map_err(|error| error.to_string())
}

fn name(data: &[u8]) -> Result<&[u8]> {
    let end = data
        .iter()
        .take(NAME_LIMIT + 1)
        .position(|byte| *byte == 0)
        .ok_or("COFF symbol name is unterminated or exceeds its bound.")?;
    Ok(&data[..end])
}

fn known(symbol: &[u8]) -> bool {
    let symbol = symbol.strip_prefix(b"__imp_").unwrap_or(symbol);
    matches!(
        symbol,
        b"_CxxThrowException"
            | b"__CxxFrameHandler3"
            | b"__CxxFrameHandler4"
            | b"__CxxDetectRethrow"
            | b"__CxxQueryExceptionSize"
            | b"__CxxExceptionFilter"
            | b"__CxxRegisterExceptionObject"
            | b"__CxxUnregisterExceptionObject"
            | b"__CxxCallUnwindDtor"
            | b"__CxxCallUnwindDelDtor"
            | b"__CxxCallUnwindVecDtor"
            | b"__std_exception_copy"
            | b"__std_exception_destroy"
            | b"__GSHandlerCheck_EH"
            | b"__GSHandlerCheck_EH4"
    )
}

fn observe(
    result: &mut Vec<Observation>,
    member_offset: Option<usize>,
    symbol: &[u8],
    binding: &'static str,
) -> Result<()> {
    if known(symbol) {
        if result.len() >= SYMBOL_LIMIT {
            return Err("COFF runtime observations exceed their bound.".into());
        }
        result.push(Observation {
            member_offset,
            symbol: std::str::from_utf8(symbol)
                .map_err(|error| error.to_string())?
                .into(),
            binding,
        });
    }
    Ok(())
}

fn object(data: &[u8], member: Option<usize>, result: &mut Vec<Observation>) -> Result<()> {
    extent(data, 0, 20)?;
    let anonymous = u16_at(data, 0)? == 0 && u16_at(data, 2)? == u16::MAX;
    let (header, sections, table, count, width) = if anonymous {
        if u16_at(data, 6)? != 0xaa64 {
            return Err("COFF runtime inspection requires native ARM64 inputs.".into());
        }
        if u16_at(data, 4)? == 0 {
            let size = size_at(data, 12)?;
            let payload = extent(data, 20, size)?;
            let flags = u16_at(data, 18)?;
            let name_type = (flags >> 2) & 7;
            if data.len() != 20 + size || flags & !0x1f != 0 || flags & 3 == 3 || name_type > 4 {
                return Err("Unsupported COFF short import header.".into());
            }
            let symbol = name(payload)?;
            let dll = name(&payload[symbol.len() + 1..])?;
            if symbol.is_empty() || dll.is_empty() {
                return Err("COFF short import has an empty symbol or DLL.".into());
            }
            let tail = &payload[symbol.len() + dll.len() + 2..];
            if (name_type != 4 && !tail.is_empty())
                || (name_type == 4 && (tail.is_empty() || name(tail)?.len() + 1 != tail.len()))
            {
                return Err("COFF short import payload differs from its name type.".into());
            }
            observe(result, member, symbol, "import")?;
            if name_type == 4 {
                observe(result, member, name(tail)?, "import-target")?;
            }
            return Ok(());
        }
        if u16_at(data, 4)? != 2 || extent(data, 12, 16)? != BIGOBJ_CLASS {
            return Err("Unsupported anonymous COFF or link-time-code-generation input.".into());
        }
        (
            56,
            size_at(data, 44)?,
            size_at(data, 48)?,
            size_at(data, 52)?,
            20,
        )
    } else {
        if u16_at(data, 0)? != 0xaa64 || u16_at(data, 16)? != 0 {
            return Err("COFF runtime inspection requires native ARM64 object headers.".into());
        }
        (
            20,
            usize::from(u16_at(data, 2)?),
            size_at(data, 8)?,
            size_at(data, 12)?,
            18,
        )
    };
    if sections > 65_536 || count > SYMBOL_LIMIT {
        return Err("COFF section/symbol count exceeds its bound.".into());
    }
    let headers = header + sections * 40;
    extent(data, 0, headers)?;
    if count == 0 && table == 0 {
        return Ok(());
    }
    if table < headers {
        return Err("COFF symbol table overlaps object headers.".into());
    }
    let symbols = extent(data, table, count * width)?;
    let string_offset = table + count * width;
    let string_size = size_at(data, string_offset)?;
    if !(4..=16 * 1024 * 1024).contains(&string_size) {
        return Err("COFF string table exceeds its bounds.".into());
    }
    let strings = extent(data, string_offset, string_size)?;
    let mut index = 0;
    while index < count {
        let entry = extent(symbols, index * width, width)?;
        let symbol = if u32_at(entry, 0)? == 0 {
            let offset = size_at(entry, 4)?;
            if offset < 4 {
                return Err("COFF long symbol name points into the string header.".into());
            }
            name(
                strings
                    .get(offset..)
                    .ok_or("COFF long symbol offset outside string table.")?,
            )?
        } else {
            let inline = &entry[..8];
            &inline[..inline.iter().position(|byte| *byte == 0).unwrap_or(8)]
        };
        let storage = entry[width - 2];
        let auxiliary = usize::from(entry[width - 1]);
        if index + 1 + auxiliary > count {
            return Err("COFF auxiliary symbols exceed the symbol table.".into());
        }
        if storage == 2 || storage == 105 {
            let section = if width == 18 {
                i32::from(i16::from_le_bytes([entry[12], entry[13]]))
            } else {
                i32::from_le_bytes(entry[12..16].try_into().map_err(|_| "COFF section.")?)
            };
            if section < -1 || (storage == 105 && (section != 0 || auxiliary != 1)) {
                return Err(
                    "COFF external symbol has an unsupported section or weak record.".into(),
                );
            }
            if section > 0
                && usize::try_from(section).map_err(|error| error.to_string())? > sections
            {
                return Err("COFF external symbol refers to an absent section.".into());
            }
            let binding = if storage == 105 {
                "weak-external"
            } else if section == 0 && u32_at(entry, 8)? == 0 {
                "undefined"
            } else if section == 0 {
                "common"
            } else {
                "defined"
            };
            observe(result, member, symbol, binding)?;
        }
        index += 1 + auxiliary;
    }
    Ok(())
}

/// Inspect every object/member, including library members the linker may not select.
pub(super) fn inspect(data: &[u8], archive: bool) -> Result<Vec<Observation>> {
    let mut result = Vec::new();
    if !archive {
        object(data, None, &mut result)?;
        return Ok(result);
    }
    if !data.starts_with(b"!<arch>\n") {
        return Err("COFF runtime inspection requires an ordinary MSVC archive.".into());
    }
    let mut offset = 8;
    let mut members = 0;
    let mut object_members = 0;
    let mut linker_members = 0;
    let mut long_names = false;
    while offset < data.len() {
        members += 1;
        if members > MEMBER_LIMIT {
            return Err("COFF archive member count exceeds its bound.".into());
        }
        let header = extent(data, offset, 60)?;
        if &header[58..] != b"`\n" {
            return Err("COFF archive member header is malformed.".into());
        }
        let size = std::str::from_utf8(&header[48..58])
            .map_err(|error| error.to_string())?
            .trim()
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        let payload = extent(data, offset + 60, size)?;
        let label = std::str::from_utf8(&header[..16])
            .map_err(|error| error.to_string())?
            .trim_end();
        if label == "/" {
            linker_members += 1;
            if linker_members > 2 || object_members != 0 || long_names {
                return Err("COFF archive linker member order is unsupported.".into());
            }
        } else if label == "//" {
            if long_names || object_members != 0 {
                return Err("COFF archive long-name member order is unsupported.".into());
            }
            long_names = true;
        } else {
            if label.is_empty() || label.starts_with("#1/") || label == "/SYM64/" {
                return Err("Unsupported COFF archive member naming format.".into());
            }
            object(payload, Some(offset), &mut result)?;
            object_members += 1;
        }
        offset += 60 + size;
        if size % 2 != 0 {
            if extent(data, offset, 1)? != b"\n" {
                return Err("COFF archive member padding is malformed.".into());
            }
            offset += 1;
        }
    }
    if object_members == 0 {
        return Err("COFF archive has no object members.".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
