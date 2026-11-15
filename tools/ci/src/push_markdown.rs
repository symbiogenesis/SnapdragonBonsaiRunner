//! Structural Markdown integrity checks for outgoing Git snapshots.
//!
//! Enforces only defect-level invariants (unclosed fenced code blocks and
//! missing relative link targets outside code/comments) so prose and formatting
//! choices never produce false-positive push blocks.

use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::Result;

const MAX_MARKDOWN_BYTES: u64 = 4 * 1024 * 1024;

pub(crate) fn check(source: &Path, markdown_paths: &[String]) -> Result<()> {
    for relative in markdown_paths {
        let path = source.join(relative);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(format!("Cannot inspect {relative}: {error}")),
        };
        if metadata.is_dir() {
            continue;
        }
        if metadata.len() > MAX_MARKDOWN_BYTES {
            return Err(format!(
                "Push blocked: Markdown file {relative} exceeds 4 MiB bound."
            ));
        }
        let content = fs::read_to_string(&path)
            .map_err(|error| format!("Push blocked: cannot read {relative} as UTF-8: {error}"))?;
        check_document(source, relative, &content)?;
    }
    Ok(())
}

fn check_document(source: &Path, relative: &str, content: &str) -> Result<()> {
    let mut fence: Option<(char, usize, usize)> = None;
    let mut paragraph = String::new();
    let mut paragraph_line = 1_usize;

    for (index, raw_line) in content.split('\n').enumerate() {
        let line_number = index + 1;
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        let stripped = strip_block_prefix(line);

        if let Some((open_ch, open_len, _)) = fence {
            let count = count_leading_char(stripped, open_ch);
            if count >= open_len && stripped[count..].trim().is_empty() {
                fence = None;
            }
            continue;
        }

        if let Some((open_ch, open_len)) = opening_fence(stripped) {
            flush_paragraph(source, relative, paragraph_line, &mut paragraph)?;
            fence = Some((open_ch, open_len, line_number));
            continue;
        }

        if stripped.trim().is_empty() {
            flush_paragraph(source, relative, paragraph_line, &mut paragraph)?;
            continue;
        }

        if paragraph.is_empty() {
            paragraph_line = line_number;
        } else {
            paragraph.push('\n');
        }
        paragraph.push_str(line);
    }

    if let Some((_, _, open_line)) = fence {
        return Err(format!(
            "Push blocked: {relative}:{open_line}: unclosed fenced code block."
        ));
    }

    flush_paragraph(source, relative, paragraph_line, &mut paragraph)
}

fn flush_paragraph(
    source: &Path,
    relative: &str,
    start_line: usize,
    paragraph: &mut String,
) -> Result<()> {
    if paragraph.is_empty() {
        return Ok(());
    }
    check_paragraph_links(source, relative, start_line, paragraph)?;
    paragraph.clear();
    Ok(())
}

fn strip_block_prefix(mut line: &str) -> &str {
    loop {
        let trimmed = line.trim_start_matches([' ', '\t']);
        if let Some(rest) = trimmed.strip_prefix('>') {
            line = rest;
        } else {
            return trimmed;
        }
    }
}

fn count_leading_char(text: &str, expected: char) -> usize {
    text.chars().take_while(|ch| *ch == expected).count()
}

fn opening_fence(stripped: &str) -> Option<(char, usize)> {
    let first = stripped.chars().next()?;
    if first != '`' && first != '~' {
        return None;
    }
    let count = count_leading_char(stripped, first);
    if count < 3 {
        return None;
    }
    let info = &stripped[count..];
    if first == '`' && info.contains('`') {
        return None;
    }
    Some((first, count))
}

fn check_paragraph_links(
    source: &Path,
    relative: &str,
    start_line: usize,
    text: &str,
) -> Result<()> {
    check_reference_definitions(source, relative, start_line, text)?;

    let bytes = text.as_bytes();
    let mut index = 0_usize;
    let mut line = start_line;

    while index < bytes.len() {
        if bytes[index] == b'\n' {
            line += 1;
            index += 1;
            continue;
        }
        if bytes[index] == b'\\' {
            if let Some(ch) = text[index + 1..].chars().next() {
                if ch == '\n' {
                    line += 1;
                }
                index += 1 + ch.len_utf8();
                continue;
            }
            break;
        }
        if text[index..].starts_with("<!--") {
            let after = index + 4;
            if let Some(end) = text[after..].find("-->") {
                line += text[after..after + end].matches('\n').count();
                index = after + end + 3;
                continue;
            }
            break;
        }
        if bytes[index] == b'`' {
            let ticks = count_leading_char(&text[index..], '`');
            if let Some(close) = find_closing_ticks(text, index + ticks, ticks) {
                line += text[index..close + ticks].matches('\n').count();
                index = close + ticks;
                continue;
            }
            index += ticks;
            continue;
        }
        if bytes[index] == b'['
            && let Some(close_bracket) = find_closing_bracket(text, index + 1)
            && bytes.get(close_bracket + 1) == Some(&b'(')
            && let Some((destination, _close_paren)) =
                parse_link_destination(text, close_bracket + 2)
        {
            let link_line = line + text[index..close_bracket + 2].matches('\n').count();
            verify_destination(source, relative, link_line, &destination)?;
            index += 1;
            continue;
        }

        if let Some(ch) = text[index..].chars().next() {
            index += ch.len_utf8();
        } else {
            break;
        }
    }

    Ok(())
}

fn check_reference_definitions(
    source: &Path,
    relative: &str,
    start_line: usize,
    text: &str,
) -> Result<()> {
    for (offset, raw_line) in text.split('\n').enumerate() {
        let trimmed = strip_block_prefix(raw_line);
        if !trimmed.starts_with('[') || trimmed.starts_with("[^") {
            continue;
        }
        let Some(colon_pos) = trimmed.find("]:") else {
            continue;
        };
        if colon_pos <= 1 || trimmed[1..colon_pos].contains(['[', ']']) {
            continue;
        }
        let rest = trimmed[colon_pos + 2..].trim_start();
        if rest.is_empty() {
            continue;
        }
        let destination = if let Some(inner) = rest.strip_prefix('<') {
            let Some(end) = inner.find('>') else {
                continue;
            };
            inner[..end].to_owned()
        } else {
            rest.split_whitespace().next().unwrap_or("").to_owned()
        };
        if !destination.is_empty() {
            verify_destination(source, relative, start_line + offset, &destination)?;
        }
    }
    Ok(())
}

fn find_closing_ticks(text: &str, start: usize, expected_len: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = start;
    while index < bytes.len() {
        if bytes[index] == b'`' {
            let count = count_leading_char(&text[index..], '`');
            if count == expected_len {
                return Some(index);
            }
            index += count;
        } else {
            index += 1;
        }
    }
    None
}

fn find_closing_bracket(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = start;
    let mut depth = 1_usize;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => {
                index += 1;
                if index < bytes.len() {
                    if let Some(ch) = text[index..].chars().next() {
                        index += ch.len_utf8();
                    }
                }
            }
            b'`' => {
                let ticks = count_leading_char(&text[index..], '`');
                if let Some(close) = find_closing_ticks(text, index + ticks, ticks) {
                    index = close + ticks;
                } else {
                    index += ticks;
                }
            }
            b'[' => {
                depth += 1;
                index += 1;
            }
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
                index += 1;
            }
            _ => {
                if let Some(ch) = text[index..].chars().next() {
                    index += ch.len_utf8();
                } else {
                    break;
                }
            }
        }
    }
    None
}

fn parse_link_destination(text: &str, start: usize) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    let mut index = start;
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    if index >= bytes.len() {
        return None;
    }

    if bytes[index] == b'<' {
        index += 1;
        let dest_start = index;
        while index < bytes.len() {
            match bytes[index] {
                b'\\' if index + 1 < bytes.len() => index += 2,
                b'\n' | b'\r' | b'<' => return None,
                b'>' => {
                    let dest = text[dest_start..index].to_owned();
                    let close = find_link_closing_paren(text, index + 1)?;
                    return Some((dest, close));
                }
                _ => {
                    index += text[index..].chars().next()?.len_utf8();
                }
            }
        }
        return None;
    }

    let mut dest = String::new();
    let mut paren_depth = 0_usize;
    while index < bytes.len() {
        let ch = text[index..].chars().next()?;
        match ch {
            '\\' => {
                index += 1;
                if let Some(escaped) = text[index..].chars().next() {
                    dest.push(escaped);
                    index += escaped.len_utf8();
                } else {
                    return None;
                }
            }
            '(' => {
                paren_depth += 1;
                if paren_depth > 32 {
                    return None;
                }
                dest.push('(');
                index += 1;
            }
            ')' => {
                if paren_depth == 0 {
                    return Some((dest, index));
                }
                paren_depth -= 1;
                dest.push(')');
                index += 1;
            }
            c if c.is_ascii_whitespace() || c.is_ascii_control() => {
                if paren_depth != 0 {
                    return None;
                }
                let close = find_link_closing_paren(text, index)?;
                return Some((dest, close));
            }
            other => {
                dest.push(other);
                index += other.len_utf8();
            }
        }
    }
    None
}

fn find_link_closing_paren(text: &str, mut index: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    if index >= bytes.len() {
        return None;
    }
    if bytes[index] == b')' {
        return Some(index);
    }
    let closer = match bytes[index] {
        b'"' => b'"',
        b'\'' => b'\'',
        b'(' => b')',
        _ => return None,
    };
    index += 1;
    while index < bytes.len() {
        if bytes[index] == b'\\' && index + 1 < bytes.len() {
            index += 2;
            continue;
        }
        if bytes[index] == closer {
            index += 1;
            while index < bytes.len() && bytes[index].is_ascii_whitespace() {
                index += 1;
            }
            return (bytes.get(index) == Some(&b')')).then_some(index);
        }
        index += text[index..].chars().next()?.len_utf8();
    }
    None
}

fn has_uri_scheme(target: &str) -> bool {
    let prefix = target.split(['/', '?', '#']).next().unwrap_or("");
    let Some((scheme, _)) = prefix.split_once(':') else {
        return false;
    };
    let mut chars = scheme.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    scheme.len() >= 2
        && first.is_ascii_alphabetic()
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.'))
}

fn percent_decode(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0_usize;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let (Some(hi), Some(lo)) =
                (hex_nibble(bytes[index + 1]), hex_nibble(bytes[index + 2]))
        {
            out.push((hi << 4) | lo);
            index += 3;
            continue;
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8(out).ok()
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn resolve_target_path(
    source: &Path,
    markdown_relative: &str,
    line: usize,
    path_part: &str,
) -> Result<PathBuf> {
    if path_part.contains('\\') {
        return Err(format!(
            "Push blocked: {markdown_relative}:{line}: Markdown link '{path_part}' uses backslashes; use '/'."
        ));
    }
    let mut components: Vec<String> = if path_part.starts_with('/') {
        Vec::new()
    } else {
        Path::new(markdown_relative)
            .parent()
            .into_iter()
            .flat_map(|parent| parent.components())
            .filter_map(|component| match component {
                Component::Normal(part) => part.to_str().map(str::to_owned),
                _ => None,
            })
            .collect()
    };

    for segment in path_part.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                if components.pop().is_none() {
                    return Err(format!(
                        "Push blocked: {markdown_relative}:{line}: relative link '{path_part}' escapes repository root."
                    ));
                }
            }
            other => components.push(other.to_owned()),
        }
    }

    let mut candidate = source.to_path_buf();
    for segment in components {
        candidate.push(segment);
    }
    Ok(candidate)
}

fn verify_destination(
    source: &Path,
    markdown_relative: &str,
    line: usize,
    raw_destination: &str,
) -> Result<()> {
    let trimmed = raw_destination.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') || has_uri_scheme(trimmed) {
        return Ok(());
    }
    let path_part = trimmed.split(['#', '?']).next().unwrap_or("");
    if path_part.is_empty() {
        return Ok(());
    }
    let raw_candidate = resolve_target_path(source, markdown_relative, line, path_part)?;
    if raw_candidate.exists() {
        return Ok(());
    }
    if let Some(decoded) = percent_decode(path_part)
        && decoded != path_part
    {
        let decoded_candidate = resolve_target_path(source, markdown_relative, line, &decoded)?;
        if decoded_candidate.exists() {
            return Ok(());
        }
    }
    Err(format!(
        "Push blocked: {markdown_relative}:{line}: relative link target '{path_part}' does not exist."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> std::result::Result<Self, Box<dyn std::error::Error>> {
            let path = fs::canonicalize(std::env::temp_dir())?.join(format!(
                "Bonsai-PushMarkdown-{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn push_markdown_detects_unclosed_fences_and_ignores_code_links() -> TestResult {
        let scratch = Scratch::new()?;
        fs::create_dir_all(scratch.0.join("docs"))?;
        fs::write(scratch.0.join("docs/real.md"), "# Real\n")?;
        fs::write(
            scratch.0.join("docs/good.md"),
            concat!(
                "# Good\n\n",
                "See [real](real.md#section), [root](/docs/real.md), [web](https://example.com),\n",
                "and inline `[ghost](missing.md)` or `` `[ghost2](missing2.md)` ``.\n",
                "<!-- [commented](missing3.md) -->\n\n",
                "```markdown\n",
                "[fenced](missing4.md)\n",
                "```\n\n",
                "````rust\n",
                "/// ```nested`\n",
                "fn ok() {}\n",
                "````\n",
            ),
        )?;
        assert!(
            check(
                &scratch.0,
                &["docs/good.md".into(), "docs/deleted.md".into()]
            )
            .is_ok()
        );

        fs::write(
            scratch.0.join("docs/unclosed.md"),
            "# Broken\n\n```rust\nfn missing_close() {}\n",
        )?;
        let fence_error = check(&scratch.0, &["docs/unclosed.md".into()])
            .err()
            .ok_or("Expected unclosed fence error.")?;
        assert!(
            fence_error.contains("unclosed fenced code block"),
            "{fence_error}"
        );

        fs::write(
            scratch.0.join("docs/broken-link.md"),
            "# Broken link\n\nSee [nope](does-not-exist.md).\n",
        )?;
        let link_error = check(&scratch.0, &["docs/broken-link.md".into()])
            .err()
            .ok_or("Expected broken link error.")?;
        assert!(link_error.contains("does-not-exist.md"), "{link_error}");

        fs::write(
            scratch.0.join("docs/escape.md"),
            "See [outside](../../outside.md).\n",
        )?;
        let escape_error = check(&scratch.0, &["docs/escape.md".into()])
            .err()
            .ok_or("Expected root-escape link error.")?;
        assert!(
            escape_error.contains("escapes repository root"),
            "{escape_error}"
        );
        Ok(())
    }

    #[test]
    fn push_markdown_passes_every_repository_markdown_file() -> TestResult {
        let root = fs::canonicalize(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))?;
        let mut stack = vec![root.clone()];
        let mut markdown = Vec::new();
        while let Some(dir) = stack.pop() {
            for entry in fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                let rel = path
                    .strip_prefix(&root)?
                    .to_string_lossy()
                    .replace('\\', "/");
                if rel == ".git"
                    || rel == "target"
                    || rel == ".worktrees"
                    || rel.starts_with("third_party/")
                    || rel == "third_party"
                {
                    continue;
                }
                if entry.file_type()?.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|ext| {
                    ext.eq_ignore_ascii_case("md") || ext.eq_ignore_ascii_case("markdown")
                }) {
                    markdown.push(rel);
                }
            }
        }
        assert!(!markdown.is_empty());
        check(&root, &markdown).map_err(Into::into)
    }
}
