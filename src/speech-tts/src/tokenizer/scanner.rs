//! Exact ordered predicates and byte positions of pinned Qwen pretokenization.

use std::io;

fn codepoint(bytes: &[u8], index: usize) -> io::Result<(u32, usize)> {
    let first = bytes[index];
    let (width, mask) = match first {
        0xc0..=0xdf => (2, 0x1f),
        0xe0..=0xef => (3, 0x0f),
        0xf0..=0xf7 => (4, 0x07),
        _ => return Ok((u32::from(first), 1)),
    };
    let available = bytes
        .get(index..index + width)
        .ok_or_else(|| super::invalid("Truncated Qwen tokenizer codepoint."))?;
    let mut value = u32::from(first & mask);
    for &byte in &available[1..] {
        value = (value << 6) | u32::from(byte & 0x3f);
    }
    Ok((value, width))
}

const fn letter(codepoint: u32) -> bool {
    matches!(codepoint, 0x41..=0x5a | 0x61..=0x7a | 0xc0..=0xd6 | 0xd8..=0xf6 | 0xf8..=0x24f | 0x370..=0x1fff | 0x2c00..=0x2dff | 0x3040..=0x9fff | 0xac00..=0xd7af | 0xf900..=0xfaff | 0x1_0000..)
}

const fn digit(codepoint: u32) -> bool {
    matches!(codepoint, 0x30..=0x39)
}

const fn whitespace(codepoint: u32) -> bool {
    matches!(
        codepoint,
        0x20 | 0x09..=0x0d | 0xa0 | 0x2000..=0x2002 | 0x200b | 0x3000
    )
}

const fn newline(codepoint: u32) -> bool {
    matches!(codepoint, 0x0a | 0x0d)
}

fn letters_end(text: &[u8], mut index: usize) -> io::Result<usize> {
    while index < text.len() {
        let (next, width) = codepoint(text, index)?;
        if !letter(next) {
            break;
        }
        index += width;
    }
    Ok(index)
}

fn contraction(text: &[u8], start: usize) -> io::Result<Option<usize>> {
    for suffix in [b"ll".as_slice(), b"re", b"ve", b"s", b"t", b"m", b"d"] {
        let end = start + suffix.len();
        if text
            .get(start..end)
            .is_some_and(|bytes| bytes.eq_ignore_ascii_case(suffix))
            && (end == text.len() || !letter(codepoint(text, end)?.0))
        {
            return Ok(Some(end));
        }
    }
    Ok(None)
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep the scanner's ordered pretokenization rules and scalar whitespace branching together for review."
)]
fn next_chunk<'a>(text: &'a [u8], index: &mut usize) -> io::Result<Option<&'a [u8]>> {
    if *index >= text.len() {
        return Ok(None);
    }
    let start = *index;
    let (current, width) = codepoint(text, start)?;
    if matches!(current, 0x27 | 0x2019)
        && start + width < text.len()
        && let Some(end) = contraction(text, start + width)?
    {
        *index = end;
        return Ok(Some(&text[start..end]));
    }
    if letter(current) {
        *index = letters_end(text, start + width)?;
        return Ok(Some(&text[start..*index]));
    }
    if !newline(current) && !letter(current) && !digit(current) && !whitespace(current) {
        let after = start + width;
        if after < text.len() {
            let (next, next_width) = codepoint(text, after)?;
            if letter(next) {
                *index = letters_end(text, after + next_width)?;
                return Ok(Some(&text[start..*index]));
            }
        }
    }
    if digit(current) {
        *index = start + 1;
        return Ok(Some(&text[start..=start]));
    }
    if newline(current) {
        let mut cursor = start;
        while cursor < text.len() && newline(u32::from(text[cursor])) {
            cursor += 1;
        }
        *index = cursor;
        return Ok(Some(&text[start..cursor]));
    }
    if whitespace(current) {
        let mut previous_whitespace_start = start;
        let mut whitespace_end = start + width;
        let mut whitespace_chars = 1_usize;
        while whitespace_end < text.len() {
            let (next, next_width) = codepoint(text, whitespace_end)?;
            if !whitespace(next) || newline(next) {
                break;
            }
            previous_whitespace_start = whitespace_end;
            whitespace_end += next_width;
            whitespace_chars += 1;
        }
        let followed = if whitespace_end < text.len() {
            let (next, _) = codepoint(text, whitespace_end)?;
            !whitespace(next) && !newline(next)
        } else {
            false
        };
        if followed && whitespace_chars > 1 {
            *index = previous_whitespace_start;
            return Ok(Some(&text[start..*index]));
        }
        let mut cursor = start + width;
        if cursor < text.len() {
            let (next, next_width) = codepoint(text, cursor)?;
            if letter(next) {
                *index = letters_end(text, cursor + next_width)?;
                return Ok(Some(&text[start..*index]));
            }
            if digit(next) {
                *index = cursor;
                return Ok(Some(&text[start..cursor]));
            }
            if !whitespace(next) && !newline(next) {
                while cursor < text.len() {
                    let (next, next_width) = codepoint(text, cursor)?;
                    if whitespace(next) || letter(next) || digit(next) {
                        break;
                    }
                    cursor += next_width;
                }
                while cursor < text.len() && newline(u32::from(text[cursor])) {
                    cursor += 1;
                }
                *index = cursor;
                return Ok(Some(&text[start..cursor]));
            }
        }
        cursor = whitespace_end;
        while cursor < text.len() {
            let (next, next_width) = codepoint(text, cursor)?;
            if !whitespace(next) {
                break;
            }
            cursor += next_width;
        }
        *index = cursor;
        return Ok(Some(&text[start..cursor]));
    }
    let mut cursor = start + width;
    while cursor < text.len() {
        let (next, next_width) = codepoint(text, cursor)?;
        if whitespace(next) || letter(next) || digit(next) || newline(next) {
            break;
        }
        cursor += next_width;
    }
    while cursor < text.len() && newline(u32::from(text[cursor])) {
        cursor += 1;
    }
    *index = cursor;
    Ok(Some(&text[start..cursor]))
}

pub(super) fn pretokenize_iter(text: &[u8]) -> impl Iterator<Item = io::Result<&[u8]>> {
    let mut index = 0;
    let mut failed = false;
    std::iter::from_fn(move || {
        if failed {
            return None;
        }
        match next_chunk(text, &mut index) {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(err) => {
                failed = true;
                Some(Err(err))
            }
        }
    })
}

#[cfg(test)]
pub(super) fn pretokenize(text: &[u8]) -> io::Result<Vec<&[u8]>> {
    let mut chunks = Vec::new();
    chunks
        .try_reserve_exact(text.len())
        .map_err(io::Error::other)?;
    for chunk in pretokenize_iter(text) {
        chunks.push(chunk?);
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::pretokenize;

    #[test]
    fn multibyte_unicode_whitespace_preserves_utf8_scalar_boundaries() -> io::Result<()> {
        assert_eq!(
            pretokenize("\u{00A0}foo".as_bytes())?,
            [b"\xc2\xa0foo".as_slice()]
        );
        assert_eq!(
            pretokenize("\u{3000}bar".as_bytes())?,
            [b"\xe3\x80\x80bar".as_slice()]
        );
        assert_eq!(
            pretokenize("\u{00A0}\u{00A0}foo".as_bytes())?,
            [b"\xc2\xa0".as_slice(), b"\xc2\xa0foo".as_slice()]
        );
        assert_eq!(
            pretokenize("\u{2000}\u{2000}word".as_bytes())?,
            ["\u{2000}".as_bytes(), "\u{2000}word".as_bytes()]
        );
        for input in [
            "a\u{00A0}b",
            " \u{3000}bar",
            "\u{00A0}\u{3000}x",
            "\u{200b}\u{200b}end",
        ] {
            let chunks = pretokenize(input.as_bytes())?;
            for chunk in &chunks {
                assert!(std::str::from_utf8(chunk).is_ok());
            }
            assert_eq!(chunks.concat(), input.as_bytes());
        }
        Ok(())
    }
}
