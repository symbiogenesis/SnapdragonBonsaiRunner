//! Inspect original PE32+ lookup tables; never resolve or load an import.

use super::{Image, ImportSymbol, Result, mapped, module};
use crate::pe_format::{c_bytes, extent, u64_at};

const IMPORT_LIMIT: usize = 65_536;
const NAME_LIMIT: usize = 4096;
const ORDINAL_FLAG: u64 = 1 << 63;

fn symbol(image: &Image, value: u64) -> Result<ImportSymbol<'_>> {
    if value & ORDINAL_FLAG != 0 {
        if value & !(ORDINAL_FLAG | 0xffff) != 0 {
            return Err("PE ordinal import contains reserved bits".into());
        }
        return Ok(ImportSymbol::Ordinal(
            u16::try_from(value & 0xffff).map_err(|error| error.to_string())?,
        ));
    }
    if value > 0x7fff_ffff {
        return Err("PE name import contains reserved bits".into());
    }
    let (start, available) = mapped(image, value, 3)?;
    let slice = extent(&image.data, start + 2, (available - 2).min(NAME_LIMIT + 1))?;
    let bytes = c_bytes(
        slice,
        NAME_LIMIT,
        "PE import symbol is empty, unterminated or exceeds its bound",
    )?;
    if bytes.is_empty() {
        return Err("PE import symbol is empty, unterminated or exceeds its bound".into());
    }
    if !bytes.iter().all(u8::is_ascii_graphic) {
        return Err("PE import symbol must contain only ASCII graphic bytes".into());
    }
    Ok(ImportSymbol::Name(
        std::str::from_utf8(bytes).map_err(|error| error.to_string())?,
    ))
}

fn table(
    image: &Image,
    lookup: u64,
    address: u64,
    count: &mut usize,
    mut visitor: impl FnMut(ImportSymbol<'_>) -> Result<()>,
) -> Result<()> {
    if lookup == 0 || address == 0 {
        return Err("PE symbol inspection requires lookup and address tables".into());
    }
    let (start, available) = mapped(image, lookup, 8)?;
    for index in 0..(available / 8).min(IMPORT_LIMIT + 1) {
        let value = u64_at(&image.data, start + index * 8)?;
        if value == 0 {
            // Bound images may replace IAT entries. Check its complete extent
            // and terminator, without interpreting those addresses as names.
            let (iat, _) = mapped(image, address, (index + 1) * 8)?;
            if u64_at(&image.data, iat + index * 8)? != 0 {
                return Err("PE import address table lacks its matching terminator".into());
            }
            return Ok(());
        }
        *count += 1;
        if *count > IMPORT_LIMIT {
            return Err("PE import symbols exceed their aggregate bound".into());
        }
        visitor(symbol(image, value)?)?;
    }
    Err("PE import lookup table is unterminated or exceeds its bound".into())
}

/// Visit original names and unresolved ordinals in one normal/delay directory.
/// Delay descriptors require the modern RVA form. An absent ordinary lookup
/// table can use an unbound IAT; a bound IAT cannot recover original names.
pub(super) fn visit_import_symbols(
    image: &Image,
    delayed: bool,
    mut visitor: impl FnMut(&str, ImportSymbol<'_>) -> Result<()>,
) -> Result<usize> {
    let mut count = 0;
    image.layout.visit_import_descriptors(
        &mut image.data.as_slice(),
        delayed,
        65_536,
        IMPORT_LIMIT,
        false,
        |_, descriptor| {
            if delayed && descriptor.attributes != 1 {
                return Err("PE symbol inspection requires RVA-based delay descriptors".into());
            }
            let name = module(image, u64::from(descriptor.name_raw))?;
            let address = u64::from(descriptor.iat_rva);
            let lookup = if descriptor.ilt_rva == 0 && !delayed && descriptor.timestamp == 0 {
                address
            } else {
                u64::from(descriptor.ilt_rva)
            };
            table(image, lookup, address, &mut count, |symbol| {
                visitor(&name, symbol)
            })
        },
    )?;
    Ok(count)
}
