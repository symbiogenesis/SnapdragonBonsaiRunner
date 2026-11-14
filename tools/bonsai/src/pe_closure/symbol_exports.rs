//! Bounded original export-table observations; no loader or API-set inference.

use std::collections::BTreeMap;

use super::{Image, Result, extent, mapped, offset_at, u16_at, u32_at};

const EXPORT_LIMIT: usize = 65_536;
const TEXT_LIMIT: usize = 4096;
const TOTAL_TEXT_LIMIT: usize = 16 * 1024 * 1024;

pub(crate) struct Export {
    pub(crate) names: Vec<String>,
    pub(crate) rva: u32,
    pub(crate) forwarder: Option<String>,
}

#[derive(Default)]
pub(crate) struct Table {
    pub(crate) ordinals: BTreeMap<u32, Export>,
    pub(crate) names: BTreeMap<String, u32>,
}

fn text(image: &Image, rva: u32, bound: usize, total: &mut usize) -> Result<String> {
    let (offset, available) = mapped(image, u64::from(rva), 1)?;
    let bytes = extent(
        &image.data,
        offset,
        available.min(bound).min(TEXT_LIMIT + 1),
    )?;
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .filter(|end| *end > 0 && *end <= TEXT_LIMIT)
        .ok_or("PE export text is empty, unterminated or exceeds its bound")?;
    if !bytes[..end].iter().all(u8::is_ascii_graphic) {
        return Err("PE export text must contain only ASCII graphic bytes".into());
    }
    *total = total
        .checked_add(end)
        .filter(|total| *total <= TOTAL_TEXT_LIMIT)
        .ok_or("PE export names exceed their aggregate byte bound")?;
    String::from_utf8(bytes[..end].to_vec()).map_err(|error| error.to_string())
}

pub(crate) fn inspect(image: &Image) -> Result<Table> {
    let directory = image.optional + 112;
    let rva = u32_at(&image.data, directory)?;
    let size = offset_at(&image.data, directory + 4)?;
    if rva == 0 && size == 0 {
        return Ok(Table::default());
    }
    if rva == 0 || size < 40 {
        return Err("Invalid PE export directory extent".into());
    }
    let end = u64::from(rva)
        .checked_add(u64::try_from(size).map_err(|error| error.to_string())?)
        .filter(|end| *end <= u64::from(u32::MAX) + 1)
        .ok_or("PE export directory RVA overflows")?;
    let (header, _) = mapped(image, u64::from(rva), size)?;
    let base = u32_at(&image.data, header + 16)?;
    let functions = offset_at(&image.data, header + 20)?;
    let names = offset_at(&image.data, header + 24)?;
    if functions > EXPORT_LIMIT || names > EXPORT_LIMIT || (functions == 0 && names != 0) {
        return Err("PE export counts exceed their bounds".into());
    }
    if functions == 0 {
        return Ok(Table::default());
    }
    let (addresses, _) = mapped(
        image,
        u64::from(u32_at(&image.data, header + 28)?),
        functions * 4,
    )?;
    let mut result = Table::default();
    let mut total_text = 0;
    for index in 0..functions {
        let ordinal = base
            .checked_add(u32::try_from(index).map_err(|error| error.to_string())?)
            .ok_or("PE export ordinal overflows")?;
        let target = u32_at(&image.data, addresses + index * 4)?;
        if target == 0 {
            continue;
        }
        let forwarder = if u64::from(target) >= u64::from(rva) && u64::from(target) < end {
            Some(text(
                image,
                target,
                usize::try_from(end - u64::from(target)).map_err(|error| error.to_string())?,
                &mut total_text,
            )?)
        } else {
            mapped(image, u64::from(target), 1)?;
            None
        };
        result.ordinals.insert(
            ordinal,
            Export {
                names: Vec::new(),
                rva: target,
                forwarder,
            },
        );
    }
    if names == 0 {
        return Ok(result);
    }
    let (pointers, _) = mapped(
        image,
        u64::from(u32_at(&image.data, header + 32)?),
        names * 4,
    )?;
    let (indices, _) = mapped(
        image,
        u64::from(u32_at(&image.data, header + 36)?),
        names * 2,
    )?;
    let mut previous: Option<String> = None;
    for index in 0..names {
        let name = text(
            image,
            u32_at(&image.data, pointers + index * 4)?,
            TEXT_LIMIT + 1,
            &mut total_text,
        )?;
        if previous.as_ref().is_some_and(|previous| previous >= &name) {
            return Err("PE export names must be distinct and sorted".into());
        }
        previous = Some(name.clone());
        let slot = u32::from(u16_at(&image.data, indices + index * 2)?);
        if usize::try_from(slot).map_err(|error| error.to_string())? >= functions {
            return Err("PE named export ordinal index exceeds its table".into());
        }
        let ordinal = base
            .checked_add(slot)
            .ok_or("PE export ordinal overflows")?;
        result
            .ordinals
            .get_mut(&ordinal)
            .ok_or("PE named export points to an absent address")?
            .names
            .push(name.clone());
        result.names.insert(name, ordinal);
    }
    Ok(result)
}
