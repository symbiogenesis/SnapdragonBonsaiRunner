//! Bounded source sanity against Git's immutable outgoing ref tips.

use std::ffi::OsString;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use crate::Result;

#[cfg(windows)]
#[path = "push_sanity_windows.rs"]
mod windows;

const USAGE: &str =
    "bonsai-ci push-sanity --root PATH [--policy-root PATH] (Git pre-push updates on stdin)";

#[derive(Debug, PartialEq, Eq)]
struct Update {
    tip: String,
    base: String,
}

fn oid(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn updates(text: &str) -> Result<Vec<Update>> {
    let mut result = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 4 || !oid(fields[1]) || !oid(fields[3]) {
            return Err("Push blocked: malformed Git pre-push ref update.".into());
        }
        if !fields[1].bytes().all(|byte| byte == b'0') {
            result.push(Update {
                tip: fields[1].to_ascii_lowercase(),
                base: fields[3].to_ascii_lowercase(),
            });
        }
    }
    Ok(result)
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Selection {
    rust: bool,
    workflows: bool,
    powershell: Vec<String>,
    cache_actions: Vec<&'static str>,
    source_fixture: bool,
}

fn selection(paths: &[String]) -> Selection {
    let owned: Vec<_> = paths
        .iter()
        .filter(|path| !path.starts_with("third_party/"))
        .collect();
    Selection {
        rust: owned.iter().any(|path| {
            extension(path, "rs")
                || path.ends_with("Cargo.toml")
                || path.as_str() == "rust-toolchain.toml"
                || path.starts_with(".cargo/")
        }),
        workflows: owned.iter().any(|path| {
            extension(path, "yml")
                || extension(path, "yaml")
                || path.as_str() == ".github/workflows/actions.lock"
                || path.as_str() == "tools/ci/check_workflow_startup.ps1"
                || path.as_str() == "tools/ci/check_workflow_syntax.ps1"
        }),
        powershell: owned
            .iter()
            .filter(|path| extension(path, "ps1") || extension(path, "psm1"))
            .map(|path| (*path).clone())
            .collect(),
        cache_actions: ["bonsai-tools", "ci-tool"]
            .into_iter()
            .filter(|action| {
                owned.iter().any(|path| {
                    path.as_str() == format!(".github/actions/{action}/action.yml")
                        || [
                            "tools/ci/test_tools_cache_inputs.ps1",
                            "tools/ci/test_support.ps1",
                        ]
                        .contains(&path.as_str())
                })
            })
            .collect(),
        source_fixture: owned.iter().any(|path| {
            [
                ".github/workflows/source-checks.yml",
                "tools/ci/select_source_checks.ps1",
                "tools/ci/test_source_checks.ps1",
                "tools/ci/test_support.ps1",
            ]
            .contains(&path.as_str())
        }),
    }
}

fn extension(path: &str, expected: &str) -> bool {
    Path::new(path)
        .extension()
        .is_some_and(|actual| actual.eq_ignore_ascii_case(expected))
}

pub(super) fn canonical_path(path: PathBuf) -> Result<PathBuf> {
    let path = std::fs::canonicalize(path).map_err(|error| error.to_string())?;
    // Canonical Windows paths have a verbatim prefix. The Job boundary accepts
    // ordinary drive/UNC paths and deliberately rejects device path spellings.
    #[cfg(windows)]
    {
        let text = path.to_str().ok_or("Push sanity paths must be UTF-8.")?;
        if let Some(suffix) = text.strip_prefix("\\\\?\\UNC\\") {
            return Ok(PathBuf::from(format!("\\\\{suffix}")));
        }
        if let Some(suffix) = text.strip_prefix("\\\\?\\") {
            return Ok(PathBuf::from(suffix));
        }
    }
    Ok(path)
}

pub(super) fn run(arguments: &[OsString]) -> Result<String> {
    if !matches!(arguments.len(), 3 | 5 | 7)
        || arguments[1] != "--root"
        || (arguments.len() >= 5 && arguments[3] != "--policy-root")
        || (arguments.len() == 7 && arguments[5] != "--bootstrap-milliseconds")
    {
        return Err(USAGE.into());
    }
    let root = canonical_path(PathBuf::from(&arguments[2]))?;
    let policy = if arguments.len() >= 5 {
        canonical_path(PathBuf::from(&arguments[4]))?
    } else {
        root.clone()
    };
    let bootstrap_milliseconds = if arguments.len() == 7 {
        arguments[6]
            .to_str()
            .ok_or(USAGE)?
            .parse::<u64>()
            .map_err(|_| USAGE)?
    } else {
        0
    };
    let mut input = String::new();
    std::io::stdin()
        .take(65537)
        .read_to_string(&mut input)
        .map_err(|error| error.to_string())?;
    if input.len() > 65536 {
        return Err("Push blocked: ref-update input exceeds 64 KiB.".into());
    }
    let updates = updates(&input)?;
    if updates.is_empty() {
        return Ok("No outgoing source tips (deletions only).".into());
    }
    #[cfg(windows)]
    return windows::check(&root, &policy, &updates, bootstrap_milliseconds);
    #[cfg(not(windows))]
    {
        let _ = (
            root,
            policy,
            updates,
            bootstrap_milliseconds,
            selection(&[]),
        );
        Err(
            "Push sanity requires native Windows ARM64 and the pinned Rust/PowerShell tools."
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_sanity_reads_all_ref_tips_and_skips_only_deletions() -> Result<()> {
        let first = "a".repeat(40);
        let second = "b".repeat(40);
        let zero = "0".repeat(40);
        let text = format!(
            "refs/heads/other {first} refs/heads/main {second}\nrefs/tags/v1 {second} refs/tags/v1 {zero}\n(delete) {zero} refs/heads/old {first}\n"
        );
        assert_eq!(
            updates(&text)?,
            vec![
                Update {
                    tip: first,
                    base: second.clone()
                },
                Update {
                    tip: second,
                    base: zero
                },
            ]
        );
        for bad in [
            "refs/heads/main HEAD refs/heads/main HEAD",
            "only three fields",
        ] {
            assert!(updates(bad).is_err());
        }
        Ok(())
    }

    #[test]
    fn push_sanity_selects_syntax_without_style_checks_for_docs_or_vendors() {
        let paths = ["README.md", "third_party/library/invalid.rs"].map(String::from);
        assert_eq!(selection(&paths), Selection::default());
        let paths = ["scripts/file with spaces.ps1", "src/audio/src/lib.rs"].map(String::from);
        let plan = selection(&paths);
        assert!(plan.rust);
        assert!(!plan.workflows);
        assert_eq!(plan.powershell, ["scripts/file with spaces.ps1"]);
    }

    #[test]
    fn push_sanity_routes_known_environment_and_source_regressions() {
        let plan = selection(&[".github/actions/bonsai-tools/action.yml".into()]);
        assert!(plan.workflows);
        assert_eq!(plan.cache_actions, ["bonsai-tools"]);
        assert!(!plan.source_fixture);
        let plan = selection(&[".github/actions/ci-tool/action.yml".into()]);
        assert_eq!(plan.cache_actions, ["ci-tool"]);
        let plan = selection(&["tools/ci/test_support.ps1".into()]);
        assert_eq!(plan.cache_actions, ["bonsai-tools", "ci-tool"]);
        assert!(plan.source_fixture);
        let plan = selection(&[".github/workflows/source-checks.yml".into()]);
        assert!(plan.workflows && plan.source_fixture);
        assert_eq!(plan.cache_actions, Vec::<&str>::new());
    }
}
