//! Command parsing and GitHub Actions output contracts.

use std::ffi::{OsStr, OsString};
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::{Result, gate, impact};

#[path = "planning.rs"]
mod planning;

#[path = "release_prerequisites.rs"]
mod release_prerequisites;

#[path = "checks/mod.rs"]
mod checks;

#[path = "native_acceptance_guard.rs"]
mod native_acceptance_guard;

#[path = "native_source_inventory.rs"]
mod native_source_inventory;

#[path = "native_archive_inventory.rs"]
mod native_archive_inventory;

#[path = "native_source_closure.rs"]
mod native_source_closure;

#[path = "native_boundary.rs"]
mod native_boundary;

#[path = "push_sanity.rs"]
mod push_sanity;

#[cfg(any(windows, test))]
#[path = "push_markdown.rs"]
mod push_markdown;

#[path = "push_format.rs"]
mod push_format;

const USAGE: &str = "Usage: bonsai-ci impact [--root PATH] [--config PATH] [--workflow FILE] [--before REV] [--after REV] [--full]\n       bonsai-ci gate [--root PATH] [--config PATH] --workflow NAME\n       bonsai-ci rust-select --root PATH\n       bonsai-ci rust-plan --root PATH [--package NAME] [--phase PHASE]\n       bonsai-ci rust-check --root PATH --target PATH --evidence PATH [--package NAME] [--phase PHASE]";

#[derive(Default)]
struct Arguments {
    command: String,
    root: Option<PathBuf>,
    config: Option<PathBuf>,
    workflow: Option<String>,
    before: String,
    after: String,
    full: bool,
}

pub(crate) struct ArgCursor<'a> {
    items: std::slice::Iter<'a, OsString>,
    usage: &'a str,
}

impl<'a> ArgCursor<'a> {
    pub(crate) fn new(arguments: &'a [OsString], usage: &'a str) -> Self {
        Self {
            items: arguments.iter(),
            usage,
        }
    }

    pub(crate) fn skip_command(&mut self) -> Result<&'a str> {
        self.next_str()
    }

    pub(crate) fn next_os(&mut self) -> Result<&'a OsStr> {
        self.items
            .next()
            .map(OsString::as_os_str)
            .ok_or_else(|| self.usage.into())
    }

    pub(crate) fn next_str(&mut self) -> Result<&'a str> {
        self.next_os()?.to_str().ok_or_else(|| self.usage.into())
    }

    pub(crate) fn next_flag(&mut self) -> Result<Option<&'a str>> {
        match self.items.next() {
            Some(flag) => flag.to_str().map(Some).ok_or_else(|| self.usage.into()),
            None => Ok(None),
        }
    }

    pub(crate) fn next_path(&mut self) -> Result<PathBuf> {
        Ok(PathBuf::from(self.next_os()?))
    }

    pub(crate) fn next_u64(&mut self) -> Result<u64> {
        self.next_str()?
            .parse::<u64>()
            .map_err(|_| self.usage.into())
    }

    pub(crate) fn expect_flag(&mut self, expected: &str) -> Result<()> {
        if self.next_str()? != expected {
            return Err(self.usage.into());
        }
        Ok(())
    }

    pub(crate) fn expect_value(&mut self, flag: &str) -> Result<&'a str> {
        self.expect_flag(flag)?;
        self.next_str()
    }

    pub(crate) fn expect_path(&mut self, flag: &str) -> Result<&'a Path> {
        self.expect_flag(flag)?;
        Ok(Path::new(self.next_os()?))
    }

    pub(crate) fn expect_hex(&mut self, flag: &str, length: usize) -> Result<&'a str> {
        let value = self.expect_value(flag)?;
        if !impact::workspace::is_lower_hex(value, length) {
            return Err(self.usage.into());
        }
        Ok(value)
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        if self.items.next().is_some() {
            return Err(self.usage.into());
        }
        Ok(())
    }
}

fn parse(arguments: &[OsString]) -> Result<Arguments> {
    let mut cursor = ArgCursor::new(arguments, USAGE);
    let command = cursor.next_str()?;
    if !["impact", "gate"].contains(&command) {
        return Err(USAGE.into());
    }
    let mut parsed = Arguments {
        command: command.into(),
        after: "HEAD".into(),
        ..Arguments::default()
    };
    while let Some(option) = cursor.next_flag()? {
        if option == "--full" && command == "impact" {
            parsed.full = true;
            continue;
        }
        if !["--root", "--config", "--workflow", "--before", "--after"].contains(&option)
            || (command == "gate" && ["--before", "--after"].contains(&option))
        {
            return Err(USAGE.into());
        }
        match option {
            "--root" => parsed.root = Some(cursor.next_path()?),
            "--config" => parsed.config = Some(cursor.next_path()?),
            "--workflow" => parsed.workflow = Some(cursor.next_str()?.into()),
            "--before" => parsed.before = cursor.next_str()?.into(),
            "--after" => parsed.after = cursor.next_str()?.into(),
            _ => return Err(USAGE.into()),
        }
    }
    if command == "gate" && parsed.workflow.as_ref().is_none_or(String::is_empty) {
        return Err(USAGE.into());
    }
    Ok(parsed)
}

pub(in crate::command) fn append_environment_file(variable: &str, contents: &str) -> Result<()> {
    if let Some(path) = std::env::var_os(variable).filter(|path| !path.is_empty()) {
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .and_then(|mut file| file.write_all(contents.as_bytes()))
            .map_err(|error| format!("Cannot append {variable}: {error}"))?;
    }
    Ok(())
}

const MAX_ACTION_OUTPUT_PATHS: usize = 64;

fn action_outputs(plan: &impact::Plan, config: &Value, serialized: &str) -> Result<String> {
    use std::fmt::Write as _;

    let bounded_plan;
    let plan_output = if plan.paths.len() > MAX_ACTION_OUTPUT_PATHS {
        let mut compact = plan.clone();
        compact.paths.truncate(MAX_ACTION_OUTPUT_PATHS);
        bounded_plan = serde_json::to_string(&compact).map_err(|e| e.to_string())?;
        bounded_plan.as_str()
    } else {
        serialized
    };
    let mut output = format!(
        "plan={plan_output}\npackages={}\n",
        serde_json::to_string(&plan.packages).map_err(|e| e.to_string())?
    );
    for (name, selected) in &plan.groups {
        writeln!(output, "{name}={selected}").map_err(|error| error.to_string())?;
    }
    if let Some(matrices) = config.get("matrices") {
        for (name, choices) in matrices
            .as_object()
            .ok_or("CI matrices must be an object.")?
        {
            let mut values = Vec::new();
            for (choice, group) in choices
                .as_object()
                .ok_or("CI matrix choices must be an object.")?
            {
                let selected = group
                    .as_str()
                    .and_then(|name| plan.groups.get(name))
                    .ok_or("CI matrix references an unknown group.")?;
                if *selected {
                    values.push(choice);
                }
            }
            writeln!(
                output,
                "{name}={}",
                serde_json::to_string(&values).map_err(|e| e.to_string())?
            )
            .map_err(|error| error.to_string())?;
        }
    }
    Ok(output)
}

#[cfg(feature = "github-api")]
const MAX_GITHUB_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;

#[cfg(feature = "github-api")]
fn github_agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(20)))
        .https_only(true)
        .max_redirects(0)
        .max_response_header_size(64 * 1024);
    #[cfg(windows)]
    let config = config.tls_config(
        ureq::tls::TlsConfig::builder()
            .provider(ureq::tls::TlsProvider::NativeTls)
            .root_certs(ureq::tls::RootCerts::PlatformVerifier)
            .build(),
    );
    ureq::Agent::new_with_config(config.build())
}

#[cfg(feature = "github-api")]
const GITHUB_GET_MAX_ATTEMPTS: usize = 4;
#[cfg(feature = "github-api")]
const GITHUB_GET_BASE_DELAY: std::time::Duration = std::time::Duration::from_millis(500);

#[cfg(feature = "github-api")]
fn is_transient_github_error(error: &ureq::Error) -> bool {
    match error {
        ureq::Error::StatusCode(code) => matches!(*code, 408 | 429 | 500..=599),
        ureq::Error::Io(_)
        | ureq::Error::Timeout(_)
        | ureq::Error::ConnectionFailed
        | ureq::Error::HostNotFound
        | ureq::Error::Protocol(_) => true,
        _ => false,
    }
}

#[cfg(feature = "github-api")]
fn sanitize_github_error(error: &ureq::Error) -> String {
    match error {
        ureq::Error::StatusCode(code) => format!("HTTP status {code}"),
        ureq::Error::Io(io_error) => format!("transport I/O: {:?}", io_error.kind()),
        ureq::Error::Timeout(_) => "request timeout".into(),
        ureq::Error::HostNotFound => "host not found".into(),
        ureq::Error::ConnectionFailed => "connection failed".into(),
        ureq::Error::Protocol(_) => "HTTP protocol error".into(),
        ureq::Error::Tls(_) => "TLS error".into(),
        ureq::Error::Http(_) => "HTTP request error".into(),
        ureq::Error::BadUri(_) => "invalid URI".into(),
        ureq::Error::RedirectFailed | ureq::Error::TooManyRedirects => "redirect error".into(),
        ureq::Error::LargeResponseHeader(_, _) => "response header too large".into(),
        ureq::Error::ConnectProxyFailed(_) => "proxy connection failed".into(),
        ureq::Error::InvalidProxyUrl => "invalid proxy configuration".into(),
        ureq::Error::RequireHttpsOnly(_) => "HTTPS required".into(),
        _ => "transport or TLS failure".into(),
    }
}

#[cfg(feature = "github-api")]
fn retry_with_backoff<T, E>(
    max_attempts: usize,
    base_delay: std::time::Duration,
    mut operation: impl FnMut() -> std::result::Result<T, E>,
    is_transient: impl Fn(&E) -> bool,
    mut sleep: impl FnMut(std::time::Duration),
) -> std::result::Result<T, E> {
    let attempts = max_attempts.max(1);
    let mut attempt = 1_usize;
    let mut delay = base_delay;
    loop {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) if attempt < attempts && is_transient(&error) => {
                sleep(delay);
                delay = delay.saturating_mul(2);
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(feature = "github-api")]
fn github_get_json(
    agent: &ureq::Agent,
    url: &str,
    token: &str,
    user_agent: &str,
    api_error: &str,
    response_error: &str,
    malformed_error: &str,
) -> Result<Value> {
    let mut response = retry_with_backoff(
        GITHUB_GET_MAX_ATTEMPTS,
        GITHUB_GET_BASE_DELAY,
        || {
            agent
                .get(url)
                .header("Authorization", &format!("Bearer {token}"))
                .header("Accept", "application/vnd.github+json")
                .header("User-Agent", user_agent)
                .call()
        },
        is_transient_github_error,
        std::thread::sleep,
    )
    .map_err(|error| format!("{api_error} ({})", sanitize_github_error(&error)))?;
    let bytes = impact::workspace::read_bounded_reader(
        response.body_mut().as_reader(),
        MAX_GITHUB_RESPONSE_BYTES,
        response_error,
    )
    .map_err(|_| response_error.to_owned())?;
    serde_json::from_slice(&bytes).map_err(|_| malformed_error.into())
}

fn native_operation(arguments: &[OsString]) -> Option<Result<String>> {
    match arguments.first()?.to_str()? {
        "push-sanity" => Some(push_sanity::run(arguments)),
        "release-prerequisites" => Some(release_prerequisites::run(arguments)),
        "native-boundary-verify" => Some(native_boundary::run(arguments)),
        "native-archive-inventory" => Some(native_archive_inventory::run(arguments)),
        "native-closure-verify" => Some(native_source_closure::run(arguments)),
        "native-closure-collect" => Some(native_source_closure::collect(arguments)),
        "native-closure-capture" => Some(native_source_closure::capture(arguments)),
        "native-source-verify" => Some(native_source_inventory::run(arguments)),
        "native-acceptance-guard" => Some(native_acceptance_guard::run(arguments)),
        "native-regressions" => Some(checks::regressions(arguments)),
        _ => None,
    }
}

/// Execute a CI planning, check or native-admission operation.
///
/// # Errors
/// Rejects invalid arguments/configuration, failed output writes and incomplete
/// or unsuccessful selected gates. Baseline failures select full coverage.
pub fn run(arguments: &[OsString]) -> Result<String> {
    if push_format::selected(arguments) {
        return push_format::run(arguments);
    }
    if let Some(result) = native_operation(arguments) {
        return result;
    }
    if arguments
        .first()
        .and_then(|value| value.to_str())
        .is_some_and(|command| matches!(command, "rust-select" | "rust-plan" | "rust-check"))
    {
        return checks::run(arguments);
    }
    if arguments.len() == 1 && arguments[0] == "--help" {
        return Ok(format!(
            "{USAGE}\n       bonsai-ci rust-select|rust-plan|rust-check (see tools/ci/README.md)\n       bonsai-ci release-prerequisites --revision SHA\n       bonsai-ci native-acceptance-guard --json REQUEST\n       bonsai-ci native-regressions --request -\n       bonsai-ci native-source-verify --root PATH --source PATH --archive PATH\n       bonsai-ci native-archive-inventory --archive PATH --pin PATH --expected-pin-sha256 SHA256\n       bonsai-ci native-closure-collect --request PATH --expected-sha256 SHA256\n       bonsai-ci native-closure-capture --request PATH --expected-sha256 SHA256\n       bonsai-ci native-closure-verify --manifest PATH --expected-sha256 SHA256\n       bonsai-ci native-boundary-verify --root PATH --manifest PATH --expected-sha256 SHA256"
        ));
    }
    let arguments = parse(arguments)?;
    let root = arguments
        .root
        .or_else(|| std::env::var_os("GITHUB_WORKSPACE").map(PathBuf::from))
        .map_or_else(std::env::current_dir, Ok)
        .and_then(fs::canonicalize)
        .map_err(|error| error.to_string())?;
    let config_path = arguments
        .config
        .unwrap_or_else(|| root.join(".github/ci/impact.json"));
    let config: Value =
        serde_json::from_slice(&fs::read(config_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if arguments.command == "gate" {
        let workflow = arguments.workflow.ok_or(USAGE)?;
        let jobs = config
            .get("workflows")
            .and_then(|workflows| workflows.get(&workflow))
            .ok_or("Unknown CI workflow.")?;
        let needs: Value =
            serde_json::from_str(&std::env::var("CI_NEEDS").map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        let errors = gate::failures(&needs, jobs);
        append_environment_file(
            "GITHUB_STEP_SUMMARY",
            &gate::summary(&workflow, &needs, jobs),
        )?;
        return if errors.is_empty() {
            Ok("All selected CI groups passed; unselected groups were skipped.".into())
        } else {
            Err(errors.join("\n"))
        };
    }
    let inputs = planning::inputs(
        &root,
        &planning::Options {
            before: &arguments.before,
            after: &arguments.after,
            workflow: arguments.workflow.as_deref(),
            full: arguments.full,
        },
    );
    let plan = impact::select(&root, &inputs.paths, inputs.full, &config, inputs.reason)?;
    let serialized = serde_json::to_string(&plan).map_err(|error| error.to_string())?;
    let outputs = action_outputs(&plan, &config, &serialized)?;
    let groups: Vec<_> = plan
        .groups
        .iter()
        .filter(|(_, selected)| **selected)
        .map(|(name, _)| name.as_str())
        .collect();
    let summary = format!(
        "Affected crates: {}\n\nSelected groups: {}\n\nSelection: {}\n",
        if plan.packages.is_empty() {
            "none".into()
        } else {
            plan.packages.join(", ")
        },
        if groups.is_empty() {
            "none".into()
        } else {
            groups.join(", ")
        },
        plan.reason
    );
    append_environment_file("GITHUB_OUTPUT", &outputs)?;
    append_environment_file("GITHUB_STEP_SUMMARY", &summary)?;
    Ok(serialized)
}

#[cfg(test)]
fn workspace_root_from(runtime: Option<&OsStr>, compile_time: &str) -> PathBuf {
    runtime
        .filter(|path| !path.is_empty())
        .map_or_else(|| PathBuf::from(compile_time), PathBuf::from)
        .join("../..")
}

#[cfg(test)]
pub(in crate::command) fn test_workspace_root() -> PathBuf {
    workspace_root_from(
        std::env::var_os("CARGO_MANIFEST_DIR").as_deref(),
        env!("CARGO_MANIFEST_DIR"),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ffi::OsStr;
    use std::path::PathBuf;

    use serde_json::json;

    use super::{action_outputs, parse, workspace_root_from};
    use crate::impact::Plan;

    #[test]
    fn test_workspace_root_prefers_runtime_manifest_dir_over_stale_compile_path() {
        assert_eq!(
            workspace_root_from(Some(OsStr::new("live/tools/ci")), "deleted/tools/ci"),
            PathBuf::from("live/tools/ci").join("../..")
        );
        assert_eq!(
            workspace_root_from(Some(OsStr::new("")), "compile/tools/ci"),
            PathBuf::from("compile/tools/ci").join("../..")
        );
        assert_eq!(
            workspace_root_from(None, "compile/tools/ci"),
            PathBuf::from("compile/tools/ci").join("../..")
        );
    }

    #[test]
    fn output_preserves_plan_packages_booleans_and_filtered_matrices() -> crate::Result<()> {
        let plan = Plan {
            packages: vec!["leaf".into()],
            groups: BTreeMap::from([("rust".into(), true), ("scripts".into(), false)]),
            full: false,
            paths: vec!["leaf/src/lib.rs".into()],
            reason: "changed inputs".into(),
        };
        let serialized = serde_json::to_string(&plan).map_err(|e| e.to_string())?;
        let output = action_outputs(
            &plan,
            &json!({"matrices": {"lanes": {"core":"rust", "scripts":"scripts"}}}),
            &serialized,
        )?;
        assert!(output.contains(&format!("plan={serialized}\n")));
        assert!(
            output.contains("packages=[\"leaf\"]\nrust=true\nscripts=false\nlanes=[\"core\"]\n")
        );
        assert!(
            action_outputs(
                &plan,
                &json!({"matrices":{"lanes":{"bad":"missing"}}}),
                &serialized
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn command_arguments_reject_unknown_options_and_incomplete_gate() {
        for arguments in [
            vec![],
            vec!["gate"],
            vec!["impact", "--unknown"],
            vec!["impact", "--after"],
            vec!["gate", "--workflow", "foundation", "--full"],
        ] {
            assert!(parse(&arguments.into_iter().map(Into::into).collect::<Vec<_>>()).is_err());
        }
        assert!(parse(&["impact".into(), "--full".into()]).is_ok());
        assert!(parse(&["gate".into(), "--workflow".into(), "foundation".into()]).is_ok());
    }

    #[test]
    fn action_outputs_bounds_large_path_lists_to_protect_linux_env_limits() -> crate::Result<()> {
        let paths: Vec<String> = (0..1600)
            .map(|index| format!("docs/evidence/unreferenced-{index:04}/summary.json"))
            .collect();
        let plan = Plan {
            packages: vec!["bonsai-broker".into()],
            groups: BTreeMap::from([("broker_checks".into(), true), ("speech".into(), false)]),
            full: true,
            paths,
            reason: "unclassified input (conservative full validation)".into(),
        };
        let serialized = serde_json::to_string(&plan).map_err(|e| e.to_string())?;
        assert!(serialized.len() > 65_536);
        let output = action_outputs(&plan, &json!({}), &serialized)?;
        assert!(output.len() < 8_192);
        let plan_line = output
            .lines()
            .find_map(|line| line.strip_prefix("plan="))
            .ok_or("missing plan output")?;
        let parsed: serde_json::Value =
            serde_json::from_str(plan_line).map_err(|e| e.to_string())?;
        assert_eq!(
            parsed["groups"],
            json!({"broker_checks": true, "speech": false})
        );
        assert_eq!(
            parsed["paths"].as_array().map(Vec::len),
            Some(super::MAX_ACTION_OUTPUT_PATHS)
        );
        Ok(())
    }

    #[cfg(feature = "github-api")]
    #[test]
    fn github_error_classification_sanitization_and_exponential_retry() {
        for status in [408_u16, 429, 500, 502, 503, 504] {
            let err = ureq::Error::StatusCode(status);
            assert!(super::is_transient_github_error(&err), "status {status}");
            assert_eq!(
                super::sanitize_github_error(&err),
                format!("HTTP status {status}")
            );
        }
        for status in [400_u16, 401, 403, 404, 422] {
            let err = ureq::Error::StatusCode(status);
            assert!(!super::is_transient_github_error(&err), "status {status}");
            assert_eq!(
                super::sanitize_github_error(&err),
                format!("HTTP status {status}")
            );
        }
        assert!(super::is_transient_github_error(&ureq::Error::HostNotFound));
        assert!(super::is_transient_github_error(
            &ureq::Error::ConnectionFailed
        ));
        assert!(super::is_transient_github_error(&ureq::Error::Io(
            std::io::Error::new(std::io::ErrorKind::ConnectionReset, "secret-token-url")
        )));
        assert_eq!(
            super::sanitize_github_error(&ureq::Error::BadUri("https://token@secret".into())),
            "invalid URI"
        );
        assert_eq!(
            super::sanitize_github_error(&ureq::Error::Io(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "do not leak"
            ))),
            "transport I/O: TimedOut"
        );

        let mut attempts = 0_usize;
        let mut sleeps = Vec::new();
        let recovered = super::retry_with_backoff(
            4,
            std::time::Duration::from_millis(10),
            || {
                attempts += 1;
                if attempts < 3 {
                    Err(ureq::Error::StatusCode(503))
                } else {
                    Ok("ok")
                }
            },
            super::is_transient_github_error,
            |delay| sleeps.push(delay),
        );
        assert!(recovered.is_ok());
        assert_eq!(attempts, 3);
        assert_eq!(
            sleeps,
            vec![
                std::time::Duration::from_millis(10),
                std::time::Duration::from_millis(20),
            ]
        );

        let mut permanent_attempts = 0_usize;
        let mut permanent_sleeps = Vec::new();
        let rejected: Result<(), ureq::Error> = super::retry_with_backoff(
            4,
            std::time::Duration::from_millis(10),
            || {
                permanent_attempts += 1;
                Err(ureq::Error::StatusCode(404))
            },
            super::is_transient_github_error,
            |delay| permanent_sleeps.push(delay),
        );
        assert!(matches!(rejected, Err(ureq::Error::StatusCode(404))));
        assert_eq!(permanent_attempts, 1);
        assert!(permanent_sleeps.is_empty());
    }
}
