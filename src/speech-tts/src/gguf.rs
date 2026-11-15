//! Bounded GGUF tokenizer metadata and shared GGUF wire primitives.
#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep GGUF metadata parsing private to the speech crate."
)]

use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

const MAGIC: &[u8; 4] = b"GGUF";
const MAX_TENSORS: u64 = 1_000_000;
const MAX_KV_ENTRIES: u64 = 65_536;
const MAX_KEY_BYTES: u64 = 65_536;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_RETAINED_STRING_BYTES: usize = 64 * 1024 * 1024;

const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;
const TYPE_FLOAT64: u32 = 12;

pub(super) fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid speech tokenizer metadata.",
    )
}

enum Entry {
    StringArray(Vec<Box<[u8]>>),
    OtherArray,
    U32(u32),
    OtherScalar,
}

/// Bounded GGUF metadata table retaining string arrays and `u32` scalars.
pub(super) struct Gguf {
    keys: HashMap<Box<[u8]>, usize>,
    entries: Vec<Entry>,
}

impl Gguf {
    pub(super) fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        Self::from_reader(BufReader::new(file))
    }

    pub(super) fn from_reader<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut magic = [0_u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid());
        }
        let version = read_u32(&mut reader)?;
        if !matches!(version, 2 | 3) {
            return Err(invalid());
        }
        let tensor_count = read_u64(&mut reader)?;
        let kv_count = read_u64(&mut reader)?;
        if tensor_count > MAX_TENSORS || kv_count > MAX_KV_ENTRIES {
            return Err(invalid());
        }
        let kv_len = usize::try_from(kv_count).map_err(|_| invalid())?;
        let mut keys = HashMap::with_capacity(kv_len);
        let mut entries = Vec::with_capacity(kv_len);
        let mut retained_bytes = 0_usize;

        for index in 0..kv_len {
            let key = read_key(&mut reader)?;
            if keys.insert(key, index).is_some() {
                return Err(invalid());
            }
            let kind = read_u32(&mut reader)?;
            let entry = match kind {
                TYPE_UINT32 => Entry::U32(read_u32(&mut reader)?),
                TYPE_BOOL => {
                    read_bool_scalar(&mut reader)?;
                    Entry::OtherScalar
                }
                TYPE_STRING => {
                    validate_string(&mut reader)?;
                    Entry::OtherScalar
                }
                TYPE_ARRAY => read_array(&mut reader, &mut retained_bytes)?,
                scalar => {
                    let width = scalar_width(scalar).ok_or_else(invalid)?;
                    skip_bytes(&mut reader, width)?;
                    Entry::OtherScalar
                }
            };
            entries.push(entry);
        }

        Ok(Self { keys, entries })
    }

    pub(super) fn find(&self, name: &[u8]) -> Option<usize> {
        self.keys.get(name).copied()
    }

    pub(super) fn string_count(&self, key: usize) -> io::Result<usize> {
        match self.entries.get(key) {
            Some(Entry::StringArray(values)) => Ok(values.len()),
            _ => Err(invalid()),
        }
    }

    pub(super) fn string(&self, key: usize, index: usize) -> io::Result<&[u8]> {
        match self.entries.get(key) {
            Some(Entry::StringArray(values)) => {
                values.get(index).map(Box::as_ref).ok_or_else(invalid)
            }
            _ => Err(invalid()),
        }
    }

    pub(super) fn u32_value(&self, key: usize) -> io::Result<u32> {
        match self.entries.get(key) {
            Some(Entry::U32(value)) => Ok(*value),
            _ => Err(invalid()),
        }
    }
}

pub(super) const fn scalar_width(kind: u32) -> Option<u64> {
    match kind {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => Some(1),
        TYPE_UINT16 | TYPE_INT16 => Some(2),
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => Some(4),
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

pub(super) fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

pub(super) fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_key(reader: &mut impl Read) -> io::Result<Box<[u8]>> {
    let len = read_u64(reader)?;
    if len == 0 || len > MAX_KEY_BYTES {
        return Err(invalid());
    }
    let len = usize::try_from(len).map_err(|_| invalid())?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|_| invalid())?;
    bytes.resize(len, 0);
    reader.read_exact(&mut bytes)?;
    if bytes.contains(&0) || std::str::from_utf8(&bytes).is_err() {
        return Err(invalid());
    }
    Ok(bytes.into_boxed_slice())
}

fn read_bool_scalar(reader: &mut impl Read) -> io::Result<()> {
    let mut byte = [0_u8; 1];
    reader.read_exact(&mut byte)?;
    if byte[0] > 1 {
        return Err(invalid());
    }
    Ok(())
}

fn read_string_checked(reader: &mut impl Read, max_len: u64) -> io::Result<Vec<u8>> {
    let len = read_u64(reader)?;
    if len > max_len {
        return Err(invalid());
    }
    let len = usize::try_from(len).map_err(|_| invalid())?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(len).map_err(|_| invalid())?;
    raw.resize(len, 0);
    reader.read_exact(&mut raw)?;
    if raw.contains(&0) || std::str::from_utf8(&raw).is_err() {
        return Err(invalid());
    }
    Ok(raw)
}

fn validate_string(reader: &mut impl Read) -> io::Result<()> {
    let _ = read_string_checked(reader, MAX_STRING_BYTES)?;
    Ok(())
}

fn read_array(reader: &mut impl Read, retained_bytes: &mut usize) -> io::Result<Entry> {
    let element_kind = read_u32(reader)?;
    let count = read_u64(reader)?;
    if count > MAX_ARRAY_ELEMENTS {
        return Err(invalid());
    }
    if element_kind == TYPE_STRING {
        let count = usize::try_from(count).map_err(|_| invalid())?;
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(|_| invalid())?;
        for _ in 0..count {
            let raw = read_string_checked(reader, MAX_STRING_BYTES)?;
            *retained_bytes = retained_bytes
                .checked_add(raw.len())
                .filter(|&total| total <= MAX_RETAINED_STRING_BYTES)
                .ok_or_else(invalid)?;
            values.push(raw.into_boxed_slice());
        }
        return Ok(Entry::StringArray(values));
    }
    if element_kind == TYPE_BOOL {
        validate_bool_bytes(reader, count)?;
        return Ok(Entry::OtherArray);
    }
    let width = scalar_width(element_kind).ok_or_else(invalid)?;
    let total = count.checked_mul(width).ok_or_else(invalid)?;
    skip_bytes(reader, total)?;
    Ok(Entry::OtherArray)
}

fn validate_bool_bytes(reader: &mut impl Read, count: u64) -> io::Result<()> {
    let mut read = 0_u64;
    let mut scratch = [0_u8; 1024];
    while read < count {
        let step = usize::try_from((count - read).min(1024)).map_err(|_| invalid())?;
        reader.read_exact(&mut scratch[..step])?;
        if scratch[..step].iter().any(|&byte| byte > 1) {
            return Err(invalid());
        }
        read += u64::try_from(step).map_err(|_| invalid())?;
    }
    Ok(())
}

fn skip_bytes(reader: &mut impl Read, count: u64) -> io::Result<()> {
    if count == 0 {
        return Ok(());
    }
    if io::copy(&mut reader.take(count), &mut io::sink())? != count {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{Gguf, TYPE_ARRAY, TYPE_BOOL, TYPE_STRING, TYPE_UINT32};

    fn push_str(out: &mut Vec<u8>, bytes: &[u8]) {
        out.extend(u64::try_from(bytes.len()).unwrap_or(0).to_le_bytes());
        out.extend(bytes);
    }

    fn header(kv_count: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(b"GGUF");
        out.extend(3_u32.to_le_bytes());
        out.extend(0_u64.to_le_bytes());
        out.extend(kv_count.to_le_bytes());
        out
    }

    #[test]
    fn rejects_invalid_bool_scalar_and_bool_array() {
        let mut valid = header(2);
        push_str(&mut valid, b"flag");
        valid.extend(TYPE_BOOL.to_le_bytes());
        valid.push(1);
        push_str(&mut valid, b"flags");
        valid.extend(TYPE_ARRAY.to_le_bytes());
        valid.extend(TYPE_BOOL.to_le_bytes());
        valid.extend(2_u64.to_le_bytes());
        valid.extend([0_u8, 1_u8]);
        assert!(Gguf::from_reader(Cursor::new(valid)).is_ok());

        let mut bad_scalar = header(1);
        push_str(&mut bad_scalar, b"flag");
        bad_scalar.extend(TYPE_BOOL.to_le_bytes());
        bad_scalar.push(2);
        assert!(Gguf::from_reader(Cursor::new(bad_scalar)).is_err());

        let mut bad_array = header(1);
        push_str(&mut bad_array, b"flags");
        bad_array.extend(TYPE_ARRAY.to_le_bytes());
        bad_array.extend(TYPE_BOOL.to_le_bytes());
        bad_array.extend(2_u64.to_le_bytes());
        bad_array.extend([0_u8, 255_u8]);
        assert!(Gguf::from_reader(Cursor::new(bad_array)).is_err());
    }

    #[test]
    fn rejects_non_utf8_embedded_nul_strings_and_duplicate_keys() {
        // Duplicate key
        let mut dup = header(2);
        for val in [1_u32, 2_u32] {
            push_str(&mut dup, b"k");
            dup.extend(TYPE_UINT32.to_le_bytes());
            dup.extend(val.to_le_bytes());
        }
        assert!(Gguf::from_reader(Cursor::new(dup)).is_err());

        // Non-UTF-8 key and embedded NUL key
        for bad_key in [b"\xff".as_slice(), b"a\0b".as_slice()] {
            let mut data = header(1);
            push_str(&mut data, bad_key);
            data.extend(TYPE_UINT32.to_le_bytes());
            data.extend(1_u32.to_le_bytes());
            assert!(Gguf::from_reader(Cursor::new(data)).is_err());
        }

        // Scalar string: non-UTF-8 and embedded NUL
        for bad_val in [b"\xfe\xff".as_slice(), b"ok\0tail".as_slice()] {
            let mut data = header(1);
            push_str(&mut data, b"s");
            data.extend(TYPE_STRING.to_le_bytes());
            push_str(&mut data, bad_val);
            assert!(Gguf::from_reader(Cursor::new(data)).is_err());
        }

        // String array: non-UTF-8 and embedded NUL (must fail instead of truncating)
        for bad_elem in [b"\x80".as_slice(), b"tok\0suffix".as_slice()] {
            let mut data = header(1);
            push_str(&mut data, b"tokens");
            data.extend(TYPE_ARRAY.to_le_bytes());
            data.extend(TYPE_STRING.to_le_bytes());
            data.extend(1_u64.to_le_bytes());
            push_str(&mut data, bad_elem);
            assert!(Gguf::from_reader(Cursor::new(data)).is_err());
        }
    }
}
