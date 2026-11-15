//! CI routing and aggregate-gate regressions using isolated manifest fixtures.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bonsai_ci::{gate, impact};
use serde_json::{Value, json};

type TestResult = Result<(), Box<dyn Error>>;

fn workspace_root() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .filter(|path| !path.is_empty())
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")), PathBuf::from)
        .join("../..")
}

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct WorkspaceCase {
    root: PathBuf,
    temporary_root: PathBuf,
    config: Value,
}

impl WorkspaceCase {
    fn new() -> Result<Self, Box<dyn Error>> {
        let temporary_root = fs::canonicalize(std::env::temp_dir())?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = temporary_root.join(format!(
            "SwiftBonsai-CiRouting-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        Ok(Self {
            root,
            temporary_root,
            config: json!({
                "groups": {"rust": {"packages": ["*"]}},
                "shared": ["Cargo.toml", "Cargo.lock", "rust-toolchain*", ".cargo/**"],
                "ignored": ["*.md", "docs/**"]
            }),
        })
    }

    fn leaf_and_other() -> Result<Self, Box<dyn Error>> {
        let fixture = Self::new()?;
        fixture.workspace(&["leaf", "other"], "", false)?;
        fixture.manifest("leaf", "leaf", "")?;
        fixture.manifest("other", "other", "")?;
        Ok(fixture)
    }

    fn write(&self, path: &str, contents: &str) -> std::io::Result<()> {
        let destination = self.root.join(path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(destination, contents)
    }

    fn manifest(&self, directory: &str, name: &str, dependencies: &str) -> std::io::Result<()> {
        let path = if directory.is_empty() {
            "Cargo.toml".to_owned()
        } else {
            format!("{directory}/Cargo.toml")
        };
        self.write(
            &path,
            &format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n{dependencies}\n"),
        )
    }

    fn workspace(&self, members: &[&str], extra: &str, root_package: bool) -> TestResult {
        let package = if root_package {
            "[package]\nname = \"root\"\nversion = \"0.1.0\"\n"
        } else {
            ""
        };
        self.write(
            "Cargo.toml",
            &format!(
                "{package}[workspace]\nmembers = {}\n{extra}",
                serde_json::to_string(members)?
            ),
        )?;
        Ok(())
    }

    fn plan(&self, paths: &[&str]) -> Result<impact::Plan, String> {
        self.plan_with_full(paths, false)
    }

    fn plan_with_full(&self, paths: &[&str], full: bool) -> Result<impact::Plan, String> {
        let paths: Vec<_> = paths.iter().map(|path| (*path).to_owned()).collect();
        impact::select(&self.root, &paths, full, &self.config, "changed inputs")
    }
}

impl Drop for WorkspaceCase {
    fn drop(&mut self) {
        // This fixture created the unique direct child; verify its boundary before removal.
        if self.root.parent() == Some(self.temporary_root.as_path())
            && self
                .root
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("SwiftBonsai-CiRouting-"))
        {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

#[test]
fn implicit_workspace_path_members_are_discovered_recursively() -> TestResult {
    let fixture = WorkspaceCase::new()?;
    fixture.workspace(&["app", "other"], "", false)?;
    fixture.manifest(
        "app",
        "app",
        "[dependencies]\nmiddle = { path = \"../middle\" }",
    )?;
    fixture.manifest(
        "middle",
        "middle",
        "[dependencies]\nleaf = { path = \"../leaf\" }",
    )?;
    fixture.manifest("leaf", "leaf", "")?;
    fixture.manifest("other", "other", "")?;
    let plan = fixture.plan(&["leaf/src/lib.rs"])?;
    assert_eq!(plan.packages, ["app", "leaf", "middle"]);
    assert!(!plan.full);
    Ok(())
}

#[test]
fn leaf_change_selects_transitive_consumers_without_sibling() -> TestResult {
    let fixture = WorkspaceCase::new()?;
    fixture.workspace(&["leaf", "middle", "app", "other"], "", false)?;
    fixture.manifest("leaf", "leaf", "")?;
    fixture.manifest(
        "middle",
        "middle",
        "[dependencies]\nleaf = { path = \"../leaf\" }",
    )?;
    fixture.manifest(
        "app",
        "app",
        "[dependencies]\nmiddle = { path = \"../middle\" }",
    )?;
    fixture.manifest("other", "other", "")?;
    let plan = fixture.plan(&["leaf/src/lib.rs"])?;
    assert_eq!(plan.packages, ["app", "leaf", "middle"]);
    assert!(plan.groups["rust"]);
    assert!(!plan.full);
    Ok(())
}

#[test]
fn aliases_and_each_dependency_kind_reach_consumers() -> TestResult {
    let dependencies = [
        (
            "runtime",
            "[dependencies]\nrenamed = { package = \"leaf\", path = \"../leaf\" }",
        ),
        (
            "development",
            "[dev-dependencies]\nleaf = { path = \"../leaf\" }",
        ),
        (
            "build",
            "[build-dependencies]\nleaf = { path = \"../leaf\" }",
        ),
        (
            "target",
            "[target.'cfg(windows)'.dependencies]\nleaf = { path = \"../leaf\" }",
        ),
        (
            "target-development",
            "[target.'cfg(unix)'.dev-dependencies]\nleaf = { path = \"../leaf\" }",
        ),
        (
            "target-build",
            "[target.'cfg(windows)'.build-dependencies]\nleaf = { path = \"../leaf\" }",
        ),
    ];
    let fixture = WorkspaceCase::new()?;
    let mut members = vec!["leaf"];
    members.extend(dependencies.iter().map(|(name, _)| *name));
    fixture.workspace(&members, "", false)?;
    fixture.manifest("leaf", "leaf", "")?;
    for (name, dependency) in dependencies {
        fixture.manifest(name, name, dependency)?;
    }
    members.sort_unstable();
    assert_eq!(fixture.plan(&["leaf/src/lib.rs"])?.packages, members);
    Ok(())
}

#[test]
fn inherited_alias_path_is_relative_to_workspace_root() -> TestResult {
    let fixture = WorkspaceCase::new()?;
    fixture.workspace(
        &["crates/leaf", "applications/app"],
        "[workspace.dependencies]\nrenamed = { package = \"leaf\", path = \"crates/leaf\" }\n",
        false,
    )?;
    fixture.manifest("crates/leaf", "leaf", "")?;
    fixture.manifest(
        "applications/app",
        "app",
        "[dependencies]\nrenamed.workspace = true",
    )?;
    assert_eq!(
        fixture.plan(&["crates/leaf/src/lib.rs"])?.packages,
        ["app", "leaf"]
    );
    Ok(())
}

#[test]
fn nested_member_owns_file_instead_of_root_package() -> TestResult {
    let fixture = WorkspaceCase::new()?;
    fixture.workspace(&["src/nested"], "", true)?;
    fixture.manifest("src/nested", "nested", "")?;
    assert_eq!(
        fixture.plan(&["src/nested/src/lib.rs"])?.packages,
        ["nested"]
    );
    assert_eq!(fixture.plan(&["src/main.rs"])?.packages, ["root"]);
    Ok(())
}

#[test]
fn move_between_crates_selects_old_and_new_dependents() -> TestResult {
    let fixture = WorkspaceCase::new()?;
    fixture.workspace(&["old", "new", "old-app", "new-app", "other"], "", false)?;
    for name in ["old", "new", "other"] {
        fixture.manifest(name, name, "")?;
    }
    for name in ["old", "new"] {
        fixture.manifest(
            &format!("{name}-app"),
            &format!("{name}-app"),
            &format!("[dependencies]\n{name} = {{ path = \"../{name}\" }}"),
        )?;
    }
    // git diff --no-renames supplies both the deletion and addition for a move.
    let plan = fixture.plan(&["old/src/moved.rs", "new/src/moved.rs"])?;
    assert_eq!(plan.packages, ["new", "new-app", "old", "old-app"]);
    assert!(!plan.full);
    Ok(())
}

#[test]
fn deleted_regular_source_still_selects_owner() -> TestResult {
    let fixture = WorkspaceCase::leaf_and_other()?;
    assert!(!fixture.root.join("leaf/src/deleted.rs").exists());
    assert_eq!(fixture.plan(&["leaf/src/deleted.rs"])?.packages, ["leaf"]);
    Ok(())
}

#[test]
fn existing_crate_manifest_selects_only_affected_crate() -> TestResult {
    let fixture = WorkspaceCase::leaf_and_other()?;
    let plan = fixture.plan(&["leaf/Cargo.toml"])?;
    assert_eq!(plan.packages, ["leaf"]);
    assert!(!plan.full);
    Ok(())
}

#[test]
fn unregistered_or_removed_manifest_requires_full_validation() -> TestResult {
    let fixture = WorkspaceCase::leaf_and_other()?;
    for path in ["new/Cargo.toml", "removed/Cargo.toml"] {
        let plan = fixture.plan(&[path])?;
        assert!(plan.full, "{path}");
        assert_eq!(plan.packages, ["leaf", "other"]);
    }
    Ok(())
}

#[test]
fn shared_build_inputs_require_all_groups() -> TestResult {
    let mut fixture = WorkspaceCase::leaf_and_other()?;
    fixture.config["groups"]["scripts"] = json!({"paths": ["scripts/**"]});
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".cargo/config.toml",
    ] {
        let plan = fixture.plan(&[path])?;
        assert!(plan.full, "{path}");
        assert!(plan.groups.values().all(|selected| *selected));
    }
    Ok(())
}

#[test]
fn root_documentation_does_not_start_expensive_jobs() -> TestResult {
    let fixture = WorkspaceCase::leaf_and_other()?;
    let plan = fixture.plan(&["README.md", "docs/validation.md"])?;
    assert_eq!(plan.packages, Vec::<String>::new());
    assert!(plan.groups.values().all(|selected| !selected));
    assert!(!plan.full);
    Ok(())
}

#[test]
fn mixed_documentation_and_code_keeps_code_coverage() -> TestResult {
    let fixture = WorkspaceCase::leaf_and_other()?;
    let plan = fixture.plan(&["docs/validation.md", "leaf/src/lib.rs"])?;
    assert_eq!(plan.packages, ["leaf"]);
    assert!(!plan.full);
    Ok(())
}

#[test]
fn unknown_input_requires_conservative_full_validation() -> TestResult {
    let fixture = WorkspaceCase::leaf_and_other()?;
    let plan = fixture.plan(&["new-code/contract.json"])?;
    assert!(plan.full);
    assert_eq!(plan.packages, ["leaf", "other"]);
    Ok(())
}

#[test]
fn lint_observation_does_not_hide_unclassified_behavior() -> TestResult {
    let mut fixture = WorkspaceCase::leaf_and_other()?;
    fixture.config["groups"]["standards"] = json!({"observes": ["scripts/**/*.ps1"]});
    fixture.config["groups"]["scripts"] = json!({"paths": ["scripts/known.ps1"]});
    assert!(fixture.plan(&["scripts/new-helper.ps1"])?.full);
    let known = fixture.plan(&["scripts/known.ps1"])?;
    assert!(!known.full);
    assert_eq!(known.packages, Vec::<String>::new());
    assert!(known.groups["scripts"]);
    assert!(known.groups["standards"]);
    Ok(())
}

#[test]
fn explicit_full_validation_ignores_empty_diff() -> TestResult {
    let mut fixture = WorkspaceCase::leaf_and_other()?;
    fixture.config["groups"]["scripts"] = json!({"paths": ["scripts/**"]});
    let plan = fixture.plan_with_full(&[], true)?;
    assert!(plan.full);
    assert_eq!(plan.packages, ["leaf", "other"]);
    assert!(plan.groups.values().all(|selected| *selected));
    Ok(())
}

#[test]
fn package_specific_shared_input_selects_its_consumers() -> TestResult {
    let mut fixture = WorkspaceCase::leaf_and_other()?;
    fixture.manifest(
        "other",
        "other",
        "[dependencies]\nleaf = { path = \"../leaf\" }",
    )?;
    fixture.config["package_inputs"] = json!({"leaf": ["fixtures/leaf/**"]});
    let plan = fixture.plan(&["fixtures/leaf/golden.json"])?;
    assert_eq!(plan.packages, ["leaf", "other"]);
    assert!(!plan.full);
    Ok(())
}

#[test]
fn group_dependencies_expand_transitively() -> TestResult {
    let mut fixture = WorkspaceCase::leaf_and_other()?;
    fixture.config["groups"] = json!({
        "release": {"groups": ["scripts"]},
        "scripts": {"groups": ["tools"]},
        "tools": {"packages": ["leaf"]},
        "unrelated": {"packages": ["other"]}
    });
    assert_eq!(
        fixture.plan(&["leaf/src/lib.rs"])?.groups,
        BTreeMap::from([
            ("release".into(), true),
            ("scripts".into(), true),
            ("tools".into(), true),
            ("unrelated".into(), false)
        ])
    );
    Ok(())
}

#[test]
fn package_path_override_does_not_claim_unrelated_root_files() -> TestResult {
    let mut fixture = WorkspaceCase::leaf_and_other()?;
    fixture.config["package_paths"] = json!({"leaf": ["leaf/src/**", "shared/leaf.rs"]});
    assert_eq!(fixture.plan(&["shared/leaf.rs"])?.packages, ["leaf"]);
    assert!(fixture.plan(&["leaf/unknown-input.bin"])?.full);
    Ok(())
}

#[test]
fn runner_ci_native_ui_uses_dedicated_lane_without_maintenance() -> TestResult {
    let root = workspace_root();
    let config: Value = serde_json::from_slice(&fs::read(root.join(".github/ci/impact.json"))?)?;
    let plan = impact::select(
        &root,
        &["src/chat-ui/src/window.rs".into()],
        false,
        &config,
        "changed inputs",
    )?;
    assert_eq!(plan.packages, ["bonsai-chat"]);
    assert!(plan.groups["ui"]);
    assert!(!plan.groups["broker"]);
    assert!(!plan.groups["maintenance"]);
    assert!(!plan.groups["speech"]);
    assert!(!plan.groups["tools"]);
    Ok(())
}

#[test]
fn runner_ci_only_asr_selects_checked_tools_without_unrelated_lanes() -> TestResult {
    let root = workspace_root();
    let config: Value = serde_json::from_slice(&fs::read(root.join(".github/ci/impact.json"))?)?;
    for (package, lane) in [("asr", "speech_asr"), ("tts", "speech_tts")] {
        let plan = impact::select(
            &root,
            &[format!("src/speech-{package}/src/main.rs")],
            false,
            &config,
            "changed inputs",
        )?;
        assert_eq!(plan.packages, [format!("bonsai-speech-{package}")]);
        for group in [lane, "speech"] {
            assert!(plan.groups[group], "{package}: {group}");
        }
        assert_eq!(plan.groups["tools"], package == "asr", "{package}");
        for group in [
            "broker_checks",
            "ui",
            "maintenance",
            "integration",
            "release",
        ] {
            assert!(!plan.groups[group], "{package}: {group}");
        }
    }
    Ok(())
}

#[test]
fn runner_ci_q4_lock_rebuilds_the_embedded_planner_and_tts() -> TestResult {
    let root = workspace_root();
    let config: Value = serde_json::from_slice(&fs::read(root.join(".github/ci/impact.json"))?)?;
    let plan = impact::select(
        &root,
        &["src/speech-tts/model-q4.lock.json".into()],
        false,
        &config,
        "changed Q4 artifact identity",
    )?;
    assert!(plan.packages.contains(&"bonsai-tools".to_owned()));
    assert!(plan.packages.contains(&"bonsai-speech-tts".to_owned()));
    assert!(plan.groups["speech_tts"] && plan.groups["tools"]);
    Ok(())
}

#[test]
fn runner_ci_native_packages_and_embedded_notices_select_checked_cli() -> TestResult {
    let root = workspace_root();
    let config: Value = serde_json::from_slice(&fs::read(root.join(".github/ci/impact.json"))?)?;
    for path in [
        "tools/bonsai/src/chat_package.rs",
        "tools/bonsai/src/chat_package/management.rs",
        "tools/native-reference/reference_numerics.rs",
        "LICENSE",
        "NOTICE",
    ] {
        let plan = impact::select(&root, &[path.into()], false, &config, "chat packaging")?;
        assert!(!plan.full, "{path}");
        assert!(plan.packages.contains(&"bonsai-tools".to_owned()), "{path}");
        assert!(plan.groups["tools"] && plan.groups["rust_lanes"], "{path}");
        assert!(!plan.groups["maintenance"], "{path}");
    }
    Ok(())
}

#[test]
fn runner_ci_shared_changes_assign_every_package_to_exactly_one_lane() -> TestResult {
    let root = workspace_root();
    let config: Value = serde_json::from_slice(&fs::read(root.join(".github/ci/impact.json"))?)?;
    let plan = impact::select(&root, &["Cargo.lock".into()], false, &config, "shared")?;
    assert!(plan.full);
    for package in &plan.packages {
        let mut owners = Vec::new();
        for lane in [
            "broker",
            "ui",
            "speech_asr",
            "speech_tts",
            "maintenance",
            "tools",
        ] {
            let group = &config["groups"][lane];
            let includes = group["packages"]
                .as_array()
                .ok_or("Missing lane packages")?;
            let excluded = group["exclude_packages"]
                .as_array()
                .is_some_and(|names| names.iter().any(|name| name == package));
            if !excluded && includes.iter().any(|name| name == "*" || name == package) {
                owners.push(lane);
            }
        }
        assert_eq!(owners.len(), 1, "{package}: {owners:?}");
    }
    Ok(())
}

#[test]
fn runner_ci_gate_requires_independent_maintenance_cli_and_speech_results() -> TestResult {
    let root = workspace_root();
    let config: Value = serde_json::from_slice(&fs::read(root.join(".github/ci/impact.json"))?)?;
    let plan = impact::select(&root, &[], true, &config, "full")?;
    let jobs = &config["workflows"]["foundation"];
    let mut needs = json!({"changes": {"result": "success", "outputs": {
        "plan": serde_json::to_string(&plan)?
    }}});
    for job in jobs.as_object().ok_or("Missing Foundation job map")?.keys() {
        needs[job] = json!({"result": "success"});
    }
    assert_eq!(gate::failures(&needs, jobs), Vec::<String>::new());
    for job in [
        "maintenance",
        "tools-package",
        "checked-tools",
        "speech",
        "speech-tts",
    ] {
        for result in ["skipped", "failure", "cancelled"] {
            needs[job]["result"] = json!(result);
            assert!(!gate::failures(&needs, jobs).is_empty(), "{job}: {result}");
        }
        needs[job]["result"] = json!("success");
    }
    Ok(())
}

#[test]
fn runner_ci_dependency_checks_remain_independent_of_cli_production() -> TestResult {
    let root = workspace_root();
    let config: Value = serde_json::from_slice(&fs::read(root.join(".github/ci/impact.json"))?)?;
    for path in ["src/artifacts/src/lib.rs", "src/windows-process/src/lib.rs"] {
        let plan = impact::select(&root, &[path.into()], false, &config, "CLI dependency")?;
        assert!(plan.packages.contains(&"bonsai-tools".to_owned()), "{path}");
        assert!(plan.groups["maintenance"] && plan.groups["tools"], "{path}");
    }
    assert_eq!(config["workflows"]["foundation"]["checked-tools"], "tools");
    assert_eq!(config["workflows"]["foundation"]["tools-package"], "tools");
    Ok(())
}

fn collect_rust_sources(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_rust_sources(&path, out)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn parse_rust_string_prefix(s: &str) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    if let Some(rest) = s.strip_prefix('"') {
        let mut out = String::new();
        let mut chars = rest.char_indices();
        while let Some((idx, ch)) = chars.next() {
            if ch == '\\' {
                if let Some((_, escaped)) = chars.next() {
                    out.push(escaped);
                }
            } else if ch == '"' {
                return Some((out, 1 + idx + 1));
            } else {
                out.push(ch);
            }
        }
        return None;
    }
    if bytes.first() == Some(&b'r') {
        let mut hashes = 0;
        while bytes.get(1 + hashes) == Some(&b'#') {
            hashes += 1;
        }
        if bytes.get(1 + hashes) == Some(&b'"') {
            let start = 2 + hashes;
            let closing = format!("\"{}", "#".repeat(hashes));
            if let Some(end_rel) = s[start..].find(&closing) {
                return Some((
                    s[start..start + end_rel].to_owned(),
                    start + end_rel + closing.len(),
                ));
            }
        }
    }
    None
}

fn extract_quoted_argument(text: &str, needle: &str) -> Vec<String> {
    let mut results = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let rest = &text[i..];
        if rest.starts_with("//") {
            i += rest.find('\n').map_or(rest.len(), |n| n + 1);
            continue;
        }
        if rest.starts_with("/*") {
            let mut depth = 1_usize;
            let mut j = i + 2;
            while j + 1 < bytes.len() && depth > 0 {
                if bytes[j] == b'/' && bytes[j + 1] == b'*' {
                    depth += 1;
                    j += 2;
                } else if bytes[j] == b'*' && bytes[j + 1] == b'/' {
                    depth -= 1;
                    j += 2;
                } else {
                    j += 1;
                }
            }
            i = j;
            continue;
        }
        if rest.starts_with("b\"") {
            if let Some((_, consumed)) = parse_rust_string_prefix(&rest[1..]) {
                i += 1 + consumed;
                continue;
            }
        }
        if rest.starts_with("br#") || rest.starts_with("br\"") {
            if let Some((_, consumed)) = parse_rust_string_prefix(&rest[1..]) {
                i += 1 + consumed;
                continue;
            }
        }
        if (bytes[i] == b'"' || bytes[i] == b'r')
            && let Some((_, consumed)) = parse_rust_string_prefix(rest)
        {
            i += consumed;
            continue;
        }
        let char_lit_start = if rest.starts_with("b'") {
            Some(2)
        } else if bytes[i] == b'\'' {
            Some(1)
        } else {
            None
        };
        if let Some(prefix_len) = char_lit_start {
            let tail = &rest[prefix_len..];
            if let Some(after_esc) = tail.strip_prefix('\\') {
                let mut chars = after_esc.char_indices();
                if let Some((_, _)) = chars.next()
                    && let Some((q_idx, '\'')) = chars.next()
                {
                    i += prefix_len + 1 + q_idx + 1;
                    continue;
                }
            } else {
                let mut chars = tail.char_indices();
                if let Some((_, first)) = chars.next()
                    && first != '\''
                    && let Some((q_idx, '\'')) = chars.next()
                {
                    i += prefix_len + q_idx + 1;
                    continue;
                }
            }
        }
        if rest.starts_with(needle) {
            let prev_ident = i > 0
                && !needle.starts_with('.')
                && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            if !prev_ident {
                let after = &rest[needle.len()..];
                let trimmed = after.trim_start();
                if let Some((val, _)) = parse_rust_string_prefix(trimmed) {
                    results.push(val);
                } else if let Some(concat_body) = trimmed.strip_prefix("concat!")
                    && let Some(env_pos) = concat_body.find("env!(\"CARGO_MANIFEST_DIR\"),")
                {
                    let tail =
                        concat_body[env_pos + "env!(\"CARGO_MANIFEST_DIR\"),".len()..].trim_start();
                    if let Some((val, _)) = parse_rust_string_prefix(tail) {
                        results.push(format!("$MANIFEST${val}"));
                    }
                }
                i += needle.len();
                continue;
            }
        }
        i += rest.chars().next().map_or(1, char::len_utf8);
    }
    results
}

#[test]
fn runner_ci_every_embedded_file_selects_its_owning_package() -> TestResult {
    let root = fs::canonicalize(workspace_root())?;
    let config: Value = serde_json::from_slice(&fs::read(root.join(".github/ci/impact.json"))?)?;
    let registry: Value =
        serde_json::from_slice(&fs::read(root.join("config/rust-packages.json"))?)?;
    let packages = registry["Packages"]
        .as_array()
        .ok_or("Missing registered packages")?;
    for pkg in packages {
        let name = pkg["Name"].as_str().ok_or("Missing package name")?;
        let manifest_rel = pkg["Manifest"].as_str().ok_or("Missing package manifest")?;
        let manifest_path = root.join(manifest_rel);
        let pkg_dir = manifest_path.parent().ok_or("Missing package directory")?;
        let pkg_rel = pkg_dir
            .strip_prefix(&root)?
            .to_str()
            .ok_or("Non-UTF-8 package directory")?
            .replace('\\', "/");
        let mut sources = Vec::new();
        collect_rust_sources(pkg_dir, &mut sources)?;
        let manifest_text = fs::read_to_string(&manifest_path)?;
        for line in manifest_text.lines() {
            if let Some(rest) = line.trim().strip_prefix("path = \"")
                && let Some(rel) = rest.strip_suffix('"')
                && Path::new(rel)
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext == "rs")
            {
                sources.push(fs::canonicalize(pkg_dir.join(rel))?);
            }
        }
        let mut external_inputs = std::collections::BTreeSet::new();
        for source in sources {
            let source_rel = source
                .strip_prefix(&root)?
                .to_str()
                .ok_or("Non-UTF-8 source path")?
                .replace('\\', "/");
            if !source_rel.starts_with(&format!("{pkg_rel}/")) {
                external_inputs.insert(source_rel.clone());
            }
            let content = fs::read_to_string(&source)?;
            let source_parent = source.parent().ok_or("Missing source parent")?;
            for needle in ["include_str!(", "include_bytes!("] {
                for raw in extract_quoted_argument(&content, needle) {
                    let resolved = if let Some(suffix) = raw.strip_prefix("$MANIFEST$") {
                        let candidate = pkg_dir.join(suffix.trim_start_matches('/'));
                        fs::canonicalize(&candidate).map_err(|err| {
                            format!("{source_rel} -> {}: {err}", candidate.display())
                        })?
                    } else {
                        let candidate = source_parent.join(&raw);
                        fs::canonicalize(&candidate).map_err(|err| {
                            format!("{source_rel} -> {}: {err}", candidate.display())
                        })?
                    };
                    let rel = resolved
                        .strip_prefix(&root)?
                        .to_str()
                        .ok_or("Non-UTF-8 embedded path")?
                        .replace('\\', "/");
                    if !rel.starts_with(&format!("{pkg_rel}/")) {
                        external_inputs.insert(rel);
                    }
                }
            }
            if source.file_name().and_then(|n| n.to_str()) == Some("build.rs") {
                for raw in extract_quoted_argument(&content, ".set_icon(") {
                    let resolved = fs::canonicalize(pkg_dir.join(raw))?;
                    let rel = resolved
                        .strip_prefix(&root)?
                        .to_str()
                        .ok_or("Non-UTF-8 icon path")?
                        .replace('\\', "/");
                    if !rel.starts_with(&format!("{pkg_rel}/")) {
                        external_inputs.insert(rel);
                    }
                }
            }
        }
        for external in external_inputs {
            let plan = impact::select(
                &root,
                std::slice::from_ref(&external),
                false,
                &config,
                "embedded input check",
            )?;
            assert!(
                plan.packages.iter().any(|selected| selected == name),
                "External input {external} embedded by {name} did not select {name} (selected: {:?})",
                plan.packages
            );
        }
    }
    Ok(())
}

fn needs(selected: bool, result: &str) -> Value {
    json!({
        "changes": {"result": "success", "outputs": {
            "plan": json!({"groups": {"rust": selected}}).to_string()
        }},
        "build": {"result": result}
    })
}

#[test]
fn selected_job_must_succeed() {
    for result in ["success", "skipped", "failure", "cancelled"] {
        let errors = gate::failures(&needs(true, result), &json!({"build": "rust"}));
        assert_eq!(errors.is_empty(), result == "success", "{result}");
    }
}

#[test]
fn only_unselected_skips_are_allowed() {
    for result in ["success", "skipped", "failure", "cancelled"] {
        let errors = gate::failures(&needs(false, result), &json!({"build": "rust"}));
        assert_eq!(
            errors.is_empty(),
            matches!(result, "success" | "skipped"),
            "{result}"
        );
    }
}

#[test]
fn failed_planner_and_malformed_results_cannot_pass() -> TestResult {
    let valid = needs(true, "success");
    let mut cases = Vec::new();
    for result in ["skipped", "failure", "cancelled"] {
        let mut needs = valid.clone();
        needs["changes"]["result"] = result.into();
        cases.push(needs);
    }
    let mut missing_job = valid.clone();
    missing_job
        .as_object_mut()
        .ok_or("Expected needs object")?
        .remove("build");
    cases.push(missing_job);
    let mut unknown_job = valid.clone();
    unknown_job["unchecked"] = json!({"result": "failure"});
    cases.push(unknown_job);
    let mut missing_plan = valid.clone();
    missing_plan["changes"]["outputs"]
        .as_object_mut()
        .ok_or("Expected outputs object")?
        .remove("plan");
    cases.push(missing_plan);
    for plan in ["not-json", "[]", "{\"groups\":{\"rust\":\"false\"}}"] {
        let mut needs = valid.clone();
        needs["changes"]["outputs"]["plan"] = plan.into();
        cases.push(needs);
    }
    for (index, needs) in cases.iter().enumerate() {
        assert!(
            !gate::failures(needs, &json!({"build": "rust"})).is_empty(),
            "case {index}"
        );
    }
    Ok(())
}

#[test]
fn malformed_gate_shapes_and_unknown_groups_fail_closed() {
    let jobs = json!({"build": "rust"});
    for needs in [Value::Null, json!([]), json!(true)] {
        assert_ne!(gate::failures(&needs, &jobs), Vec::<String>::new());
    }
    for jobs in [
        Value::Null,
        json!([]),
        json!({"build": false}),
        json!({"build": "unknown"}),
    ] {
        assert_ne!(
            gate::failures(&needs(true, "success"), &jobs),
            Vec::<String>::new()
        );
    }
    for result in [Value::Null, json!(false), json!({})] {
        let mut malformed = needs(false, "success");
        malformed["build"]["result"] = result;
        assert_ne!(gate::failures(&malformed, &jobs), Vec::<String>::new());
    }
}
