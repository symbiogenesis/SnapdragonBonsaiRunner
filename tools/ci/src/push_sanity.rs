//! Bounded source sanity against Git's immutable outgoing ref tips.

use std::ffi::OsString;
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
    markdown: Vec<String>,
    cache_actions: Vec<&'static str>,
    cache_closures: bool,
    workspace_registry: bool,
    native_boundary: bool,
    hexagon_locks: bool,
    source_fixture: bool,
    foundation_handoff: bool,
    device_concurrency: bool,
    ci_bootstrap_cache: bool,
    rust_toolchain_fixture: bool,
}

fn is_native_source(path: &str) -> bool {
    ["c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx", "m", "mm"]
        .into_iter()
        .any(|ext| extension(path, ext))
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
                || matches!(
                    Path::new(path).file_name().and_then(|name| name.to_str()),
                    Some("rustfmt.toml" | ".rustfmt.toml")
                )
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
        markdown: owned
            .iter()
            .filter(|path| extension(path, "md") || extension(path, "markdown"))
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
        cache_closures: owned.iter().any(|path| {
            extension(path, "rs")
                || path.ends_with("Cargo.toml")
                || path.as_str() == "Cargo.lock"
                || [
                    ".github/actions/check-rust/action.yml",
                    ".github/actions/bonsai-tools/action.yml",
                    ".github/actions/ci-tool/action.yml",
                    "tools/ci/test_tools_cache_inputs.ps1",
                    "tools/ci/test_support.ps1",
                ]
                .contains(&path.as_str())
        }),
        workspace_registry: owned.iter().any(|path| {
            path.ends_with("Cargo.toml")
                || path.as_str() == "Cargo.lock"
                || path.as_str() == "config/rust-packages.json"
                || path.as_str() == ".github/ci/impact.json"
                || path.as_str() == "tools/ci/src/checks/mod.rs"
                || path.as_str() == "tools/ci/src/impact.rs"
                || path.as_str() == "tools/ci/src/workspace.rs"
        }),
        native_boundary: owned.iter().any(|path| {
            is_native_source(path)
                || path.as_str() == ".gitattributes"
                || path.ends_with("/.gitattributes")
                || extension(path, "rs")
                || path.ends_with("Cargo.toml")
                || path.as_str() == "Cargo.lock"
                || path.ends_with("CMakeLists.txt")
                || path.as_str() == "tools/ci/native-boundary.runner.json"
                || path.starts_with(".github/workflows/")
                || path.starts_with("tools/hexagon/")
                || path.starts_with("tools/native-reference/")
        }),
        hexagon_locks: owned.iter().any(|path| {
            path.starts_with("tools/hexagon/")
                || path.starts_with("tools/bonsai/src/hexagon_tools")
                || path.starts_with("tools/bonsai/src/hexagon_policy")
                || path.starts_with("config/hexagon-")
        }),
        source_fixture: owned.iter().any(|path| {
            [
                ".github/workflows/source-checks.yml",
                "tools/ci/select_source_checks.ps1",
                "tools/ci/test_source_checks.ps1",
                "tools/ci/test_support.ps1",
            ]
            .contains(&path.as_str())
        }),
        foundation_handoff: owned.iter().any(|path| {
            [
                ".github/workflows/foundation.yml",
                ".github/workflows/release.yml",
                ".github/ci/impact.json",
                "config/rust-packages.json",
                ".github/actions/check-rust/action.yml",
                ".github/actions/use-bonsai-tools/action.yml",
                "tools/ci/src/checks/mod.rs",
                "tools/ci/test_foundation_handoff.ps1",
                "tools/ci/test_support.ps1",
            ]
            .contains(&path.as_str())
        }),
        device_concurrency: owned.iter().any(|path| {
            [
                ".github/workflows/native-runtime-acceptance.yml",
                ".github/workflows/native-ptq1-numerics.yml",
                ".github/workflows/x2-cpu-evaluation.yml",
                ".github/workflows/cpu-bundle-comparison.yml",
                ".github/workflows/tts-evaluation.yml",
                "tools/ci/test_device_concurrency.ps1",
                "tools/ci/test_support.ps1",
            ]
            .contains(&path.as_str())
        }),
        ci_bootstrap_cache: owned.iter().any(|path| {
            [
                ".github/actions/ci-tool/action.yml",
                "tools/ci/native_ci_cache.ps1",
                "tools/ci/test_native_ci_cache.ps1",
                "tools/ci/test_support.ps1",
            ]
            .contains(&path.as_str())
        }),
        rust_toolchain_fixture: owned.iter().any(|path| {
            [
                ".github/actions/rust-toolchain/action.yml",
                "tools/ci/test_rust_toolchain.ps1",
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
    let mut cursor = super::ArgCursor::new(arguments, USAGE);
    cursor.skip_command()?;
    let root = canonical_path(PathBuf::from(cursor.expect_path("--root")?))?;
    let mut policy = None;
    let mut bootstrap_milliseconds = 0_u64;
    if let Some(flag) = cursor.next_flag()? {
        if flag != "--policy-root" {
            return Err(USAGE.into());
        }
        policy = Some(canonical_path(cursor.next_path()?)?);
    }
    if let Some(flag) = cursor.next_flag()? {
        if flag != "--bootstrap-milliseconds" {
            return Err(USAGE.into());
        }
        bootstrap_milliseconds = cursor.next_u64()?;
    }
    cursor.finish()?;
    let policy = policy.unwrap_or_else(|| root.clone());
    let input = String::from_utf8(crate::impact::workspace::read_bounded_reader(
        std::io::stdin(),
        65536,
        "Push blocked: ref-update input exceeds 64 KiB.",
    )?)
    .map_err(|error| error.to_string())?;
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
        let paths = [
            "README.md",
            "third_party/library/invalid.rs",
            "third_party/library/ignored.md",
        ]
        .map(String::from);
        assert_eq!(
            selection(&paths),
            Selection {
                markdown: vec!["README.md".into()],
                ..Selection::default()
            }
        );
        let paths = ["scripts/file with spaces.ps1", "src/audio/src/lib.rs"].map(String::from);
        let plan = selection(&paths);
        assert!(plan.rust);
        assert!(plan.cache_closures);
        assert!(plan.native_boundary);
        assert!(!plan.workflows);
        assert!(plan.markdown.is_empty());
        assert_eq!(plan.powershell, ["scripts/file with spaces.ps1"]);
    }

    #[test]
    fn push_sanity_selects_registry_boundary_cache_closures_and_hexagon_locks() {
        let registry = selection(&["config/rust-packages.json".into()]);
        assert!(registry.workspace_registry);
        assert!(!registry.rust);

        let closure = selection(&[".github/actions/check-rust/action.yml".into()]);
        assert!(closure.workflows);
        assert!(closure.cache_closures);

        let native = selection(&["tools/native-reference/helper.cpp".into()]);
        assert!(native.native_boundary);
        assert!(!native.rust);

        for path in [".gitattributes", "tools/bonsai/.gitattributes"] {
            let attributes = selection(&[path.into()]);
            assert!(attributes.native_boundary, "{path}");
            assert!(!attributes.rust, "{path}");
        }

        let hexagon = selection(&["tools/hexagon/windows-queue-abi.lock.json".into()]);
        assert!(hexagon.hexagon_locks);
        assert!(hexagon.native_boundary);
    }

    #[test]
    fn push_sanity_routes_known_environment_and_source_regressions() {
        let plan = selection(&[".github/actions/bonsai-tools/action.yml".into()]);
        assert!(plan.workflows);
        assert_eq!(plan.cache_actions, ["bonsai-tools"]);
        assert!(!plan.source_fixture);
        let plan = selection(&[".github/actions/ci-tool/action.yml".into()]);
        assert_eq!(plan.cache_actions, ["ci-tool"]);
        assert!(plan.cache_closures && plan.ci_bootstrap_cache);
        let plan = selection(&["tools/ci/test_support.ps1".into()]);
        assert_eq!(plan.cache_actions, ["bonsai-tools", "ci-tool"]);
        assert!(
            plan.source_fixture
                && plan.foundation_handoff
                && plan.device_concurrency
                && plan.ci_bootstrap_cache
                && plan.rust_toolchain_fixture
        );
        let plan = selection(&[".github/workflows/source-checks.yml".into()]);
        assert!(plan.workflows && plan.source_fixture);
        assert_eq!(plan.cache_actions, Vec::<&str>::new());
        assert!(selection(&[".github/workflows/foundation.yml".into()]).foundation_handoff);
        assert!(selection(&[".github/workflows/x2-cpu-evaluation.yml".into()]).device_concurrency);
        assert!(
            selection(&[".github/actions/rust-toolchain/action.yml".into()]).rust_toolchain_fixture
        );
        let check_rust = selection(&[".github/actions/check-rust/action.yml".into()]);
        assert!(check_rust.foundation_handoff);
    }

    #[test]
    fn push_sanity_formats_changed_root_and_nested_rustfmt_configuration() {
        for path in [
            "rustfmt.toml",
            ".rustfmt.toml",
            "src/audio/rustfmt.toml",
            "src/audio/.rustfmt.toml",
        ] {
            assert!(selection(&[path.into()]).rust, "{path}");
        }
        for path in ["docs/settings.toml", "docs/rustfmt.toml.example"] {
            assert!(!selection(&[path.into()]).rust, "{path}");
        }
    }
}
