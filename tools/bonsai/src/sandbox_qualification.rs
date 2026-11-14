//! Fixed native sandbox qualification with a hosted prebuilt-only owner handoff.
#![cfg_attr(
    not(any(windows, test)),
    expect(
        dead_code,
        reason = "Native execution requires Windows; fixed admission is covered by portable fixtures."
    )
)]

use serde_json::{Value, json};

#[cfg(windows)]
mod windows;

const FILES: &[&str] = &[
    "sandbox-ownership.exe",
    "sandbox-contained.exe",
    "bonsai-process-test-probe.exe",
    "bonsai-tools.exe",
    "bonsai-ci.exe",
];
const CASES: &[(&str, &str)] = &[
    (
        "sandbox-ownership.exe",
        "native_sandbox_limits_files_and_preserves_loopback_process_identity",
    ),
    (
        "sandbox-ownership.exe",
        "native_sandbox_offline_pipes_enforce_authority_and_close_without_logs",
    ),
    (
        "sandbox-ownership.exe",
        "native_sandbox_denies_external_tcp_connect_by_policy",
    ),
    (
        "sandbox-ownership.exe",
        "native_sandbox_host_loopback_capability_includes_host_interface_services",
    ),
    (
        "sandbox-ownership.exe",
        "native_sandbox_abrupt_owner_death_ends_worker_and_listener",
    ),
    (
        "sandbox-contained.exe",
        "contained_verification_retains_exact_artifact_and_credential",
    ),
];
const TOOLCHAIN: &str = "nightly-2026-11-12-aarch64-pc-windows-msvc";
// Fixed controls contain different numbers of ten-second native waits. Keep
// their conservative remaining estimates attached to the original case order.
const CASE_SECONDS: &[u64] = &[45, 25, 15, 15, 50, 30];
const LOG_CASES: &[(&str, &str)] = &[
    (
        "sandbox-ownership.exe",
        "native_sandbox_stdout_flood_is_capped_and_stopped",
    ),
    (
        "sandbox-ownership.exe",
        "native_sandbox_stderr_flood_is_capped_and_stopped",
    ),
    (
        "sandbox-ownership.exe",
        "native_sandbox_exit_drains_both_logs_and_releases_capture_files",
    ),
];
const LOG_CASE_SECONDS: &[u64] = &[30, 30, 30];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Workload {
    Authority,
    LogCapture,
}

impl Workload {
    fn select(value: Option<&Value>) -> crate::Result<Self> {
        match value {
            None => Ok(Self::Authority),
            Some(Value::String(name)) if name == "authority-v1" => Ok(Self::Authority),
            Some(Value::String(name)) if name == "log-capture-v1" => Ok(Self::LogCapture),
            _ => Err("Unknown sandbox qualification workload.".into()),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Authority => "authority-v1",
            Self::LogCapture => "log-capture-v1",
        }
    }

    const fn cases(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Authority => CASES,
            Self::LogCapture => LOG_CASES,
        }
    }

    const fn seconds(self) -> &'static [u64] {
        match self {
            Self::Authority => CASE_SECONDS,
            Self::LogCapture => LOG_CASE_SECONDS,
        }
    }

    fn estimated_total_seconds(self) -> u64 {
        self.seconds().iter().sum::<u64>() + 60
    }

    const fn scope(self) -> &'static str {
        match self {
            Self::Authority => {
                "Six fixed model-free native sandbox controls; host-loopback authority includes other host services; no model, broker-only transport, throughput or complete application qualification."
            }
            Self::LogCapture => {
                "Three fixed model-free native capture controls: stdout flood, stderr flood and normal-exit tail draining; no scratch/private-temp quota, broker-only transport, model, performance or complete application qualification."
            }
        }
    }
}

fn text<'a>(value: &'a Value, key: &str) -> crate::Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("Missing sandbox string: {key}"))
}

fn selected_artifacts(raw: &str, wanted: &[(&str, &str)]) -> crate::Result<Vec<String>> {
    if raw.len() > 16 * 1024 * 1024 {
        return Err("Sandbox Cargo messages exceed their bound.".into());
    }
    let mut paths = std::collections::BTreeMap::new();
    for line in raw.lines().filter(|line| !line.is_empty()) {
        let value: Value = serde_json::from_str(line).map_err(|error| error.to_string())?;
        if value["reason"] != "compiler-artifact" || value["executable"].is_null() {
            continue;
        }
        for &(name, kind) in wanted {
            if value["target"]["name"] == name {
                let path = text(&value, "executable")?;
                if !std::path::Path::new(path).is_absolute()
                    || !value["target"]["kind"]
                        .as_array()
                        .is_some_and(|kinds| kinds.iter().any(|value| value == kind))
                    || paths.insert(name, path.to_owned()).is_some()
                {
                    return Err("Sandbox Cargo artifact inventory is ambiguous.".into());
                }
            }
        }
    }
    wanted
        .iter()
        .map(|&(name, _)| {
            paths
                .remove(name)
                .ok_or_else(|| format!("Sandbox Cargo artifact missing: {name}"))
        })
        .collect()
}

fn package_identity(
    receipt: &Value,
    revision: &str,
    run: &str,
    attempt: &str,
    workload: Workload,
) -> crate::Result<()> {
    if receipt["SchemaVersion"] != 1
        || receipt["Kind"] != "native-sandbox-prebuilt-package"
        || receipt["Revision"] != revision
        || receipt["RunId"] != run
        || receipt["RunAttempt"] != attempt
        || receipt["Toolchain"] != TOOLCHAIN
        || receipt["ModelExecuted"] != false
        || receipt["MxcVersion"] != "1.0.1"
        // Legacy schema-1 packages omit Workload and retain only the original
        // six authority controls. They can never authorize the log workload.
        || Workload::select(receipt.get("Workload"))? != workload
        || receipt["Cases"] != json!(workload.cases())
    {
        return Err("Sandbox package source/run/workload identity differs.".into());
    }
    let files = receipt["Files"]
        .as_array()
        .filter(|files| files.len() == FILES.len())
        .ok_or("Sandbox package file inventory differs.")?;
    let mut seen = std::collections::BTreeSet::new();
    for file in files {
        let name = text(file, "Name")?;
        if !FILES.contains(&name)
            || !seen.insert(name)
            || file["Bytes"]
                .as_u64()
                .is_none_or(|bytes| bytes == 0 || bytes > 134_217_728)
            || file["Sha256"]
                .as_str()
                .is_none_or(|hash| !crate::identity::is_lower_hex(hash, 64))
        {
            return Err("Sandbox package file identity differs.".into());
        }
    }
    Ok(())
}

fn inventory(binary: &str, output: &str, workload: Workload) -> crate::Result<()> {
    let listed = output.lines().collect::<std::collections::BTreeSet<_>>();
    for &(file, name) in workload.cases() {
        if file == binary && !listed.contains(format!("{name}: test").as_str()) {
            return Err(format!("Required native sandbox case missing: {name}"));
        }
    }
    Ok(())
}

fn remaining_fit(
    now: u64,
    frequency: u64,
    deadline: u64,
    cases: usize,
    workload: Workload,
) -> crate::Result<()> {
    if frequency == 0
        || cases == 0
        || cases > workload.cases().len()
        || workload
            .seconds()
            .get(workload.cases().len() - cases..)
            .map(|seconds| seconds.iter().sum::<u64>())
            .and_then(|seconds| seconds.checked_mul(frequency))
            .and_then(|ticks| now.checked_add(ticks))
            .is_none_or(|end| end > deadline)
    {
        return Err("Remaining sandbox allowance cannot fit the fixed conservative controls; qualification remains pending.".into());
    }
    Ok(())
}

/// Prepare source-bound native sandbox fixtures on hosted CI or qualify that exact package.
///
/// # Errors
/// Rejects changed source/run/workload identities, incomplete package inventories,
/// unsupported execution lanes, missing controls, expired clocks or failed teardown.
pub fn run(request: &Value) -> crate::Result<Value> {
    match request["operation"].as_str() {
        Some("prepare" | "qualify") => {
            #[cfg(windows)]
            {
                windows::run(request)
            }
            #[cfg(not(windows))]
            {
                Err("Sandbox qualification requires native Windows ARM64.".into())
            }
        }
        _ => Err("Unknown sandbox qualification operation.".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sandbox_artifacts_require_every_exact_prebuilt_binary() -> crate::Result<()> {
        let root = std::env::temp_dir();
        let message = |name: &str, kind: &str| {
            json!({"reason":"compiler-artifact","target":{"name":name,"kind":[kind]},"executable":root.join(name)}).to_string()
        };
        let raw = format!(
            "{}\n{}",
            message("ownership", "test"),
            message("bonsai-process-test-probe", "bin")
        );
        let wanted = [("ownership", "test"), ("bonsai-process-test-probe", "bin")];
        assert_eq!(selected_artifacts(&raw, &wanted)?.len(), 2);
        for raw in [
            message("ownership", "test"),
            format!("{raw}\n{}", message("ownership", "test")),
            raw.replace("\"bin\"", "\"test\""),
        ] {
            assert!(selected_artifacts(&raw, &wanted).is_err());
        }
        Ok(())
    }
    #[test]
    fn sandbox_receipt_and_inventory_keep_exact_original_six_controls() -> crate::Result<()> {
        assert_eq!(CASES.len(), 6);
        assert_eq!(FILES.len(), 5);
        let files = FILES
            .iter()
            .map(|name| json!({"Name":name,"Bytes":1,"Sha256":"b".repeat(64)}))
            .collect::<Vec<_>>();
        let receipt = json!({"SchemaVersion":1,"Kind":"native-sandbox-prebuilt-package","Revision":"a".repeat(40),"RunId":"123","RunAttempt":"1","Toolchain":TOOLCHAIN,"ModelExecuted":false,"MxcVersion":"1.0.1","Cases":CASES,"Files":files});
        package_identity(&receipt, &"a".repeat(40), "123", "1", Workload::Authority)?;
        for (field, value) in [
            ("RunId", json!("124")),
            ("Cases", json!([])),
            ("ModelExecuted", json!(0)),
            ("Files", json!([])),
        ] {
            let mut changed = receipt.clone();
            changed[field] = value;
            assert!(
                package_identity(&changed, &"a".repeat(40), "123", "1", Workload::Authority)
                    .is_err()
            );
        }
        let listed = CASES
            .iter()
            .filter(|(file, _)| *file == "sandbox-ownership.exe")
            .map(|(_, name)| format!("{name}: test"))
            .collect::<Vec<_>>()
            .join("\n");
        inventory("sandbox-ownership.exe", &listed, Workload::Authority)?;
        assert!(inventory("sandbox-ownership.exe", "", Workload::Authority).is_err());
        Ok(())
    }

    #[test]
    fn sandbox_admission_charges_each_remaining_control_to_one_original_deadline()
    -> crate::Result<()> {
        assert_eq!(CASE_SECONDS.len(), CASES.len());
        assert_eq!(CASE_SECONDS.iter().sum::<u64>(), 180);
        remaining_fit(1_000, 10, 2_800, 6, Workload::Authority)?;
        assert!(remaining_fit(1_001, 10, 2_800, 6, Workload::Authority).is_err());
        remaining_fit(1_450, 10, 2_800, 5, Workload::Authority)?;
        assert!(remaining_fit(1_451, 10, 2_800, 5, Workload::Authority).is_err());
        for (now, frequency, deadline, count) in [
            (0, 0, 1, 1),
            (0, u64::MAX, u64::MAX, 6),
            (u64::MAX, 1, u64::MAX, 1),
            (0, 1, 1, 0),
            (0, 1, u64::MAX, 7),
        ] {
            assert!(remaining_fit(now, frequency, deadline, count, Workload::Authority).is_err());
        }
        Ok(())
    }

    #[test]
    fn sandbox_log_workload_cannot_expand_or_substitute_original_authority_controls()
    -> crate::Result<()> {
        assert_eq!(Workload::select(None)?, Workload::Authority);
        assert_eq!(
            Workload::select(Some(&json!("authority-v1")))?,
            Workload::Authority
        );
        assert_eq!(
            Workload::select(Some(&json!("log-capture-v1")))?,
            Workload::LogCapture
        );
        for value in [Value::Null, json!("logs"), json!([]), json!(true)] {
            assert!(Workload::select(Some(&value)).is_err());
        }
        let files = FILES
            .iter()
            .map(|name| json!({"Name":name,"Bytes":1,"Sha256":"b".repeat(64)}))
            .collect::<Vec<_>>();
        let mut receipt = json!({"SchemaVersion":1,"Kind":"native-sandbox-prebuilt-package","Revision":"a".repeat(40),"RunId":"123","RunAttempt":"1","Toolchain":TOOLCHAIN,"ModelExecuted":false,"MxcVersion":"1.0.1","Workload":"log-capture-v1","Cases":LOG_CASES,"Files":files});
        package_identity(&receipt, &"a".repeat(40), "123", "1", Workload::LogCapture)?;
        assert!(
            package_identity(&receipt, &"a".repeat(40), "123", "1", Workload::Authority).is_err()
        );
        for (field, value) in [
            ("Workload", Value::Null),
            ("Workload", json!("authority-v1")),
            ("Cases", json!(CASES)),
            ("Cases", json!([LOG_CASES[0], LOG_CASES[1]])),
        ] {
            let mut changed = receipt.clone();
            changed[field] = value;
            assert!(
                package_identity(&changed, &"a".repeat(40), "123", "1", Workload::LogCapture)
                    .is_err()
            );
        }
        receipt
            .as_object_mut()
            .ok_or("fixture receipt must be an object")?
            .remove("Workload");
        assert!(
            package_identity(&receipt, &"a".repeat(40), "123", "1", Workload::LogCapture).is_err()
        );
        let listed = LOG_CASES
            .iter()
            .map(|(_, name)| format!("{name}: test"))
            .collect::<Vec<_>>()
            .join("\n");
        inventory("sandbox-ownership.exe", &listed, Workload::LogCapture)?;
        assert!(inventory("sandbox-ownership.exe", &listed, Workload::Authority).is_err());
        assert!(
            inventory(
                "sandbox-ownership.exe",
                &format!("{}: test", LOG_CASES[0].1),
                Workload::LogCapture,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn sandbox_log_admission_keeps_three_controls_inside_the_original_deadline() -> crate::Result<()>
    {
        let workload = Workload::LogCapture;
        assert_eq!(workload.cases().len(), 3);
        assert_eq!(workload.seconds().len(), 3);
        assert_eq!(workload.estimated_total_seconds(), 150);
        assert_eq!(Workload::Authority.estimated_total_seconds(), 240);
        for index in 0..workload.cases().len() {
            let now = 1_000 + u64::try_from(index).map_err(|error| error.to_string())? * 300;
            remaining_fit(now, 10, 1_900, 3 - index, workload)?;
            assert!(remaining_fit(now + 1, 10, 1_900, 3 - index, workload).is_err());
        }
        for (now, frequency, deadline, count) in [
            (0, 0, 1, 1),
            (0, u64::MAX, u64::MAX, 3),
            (u64::MAX, 1, u64::MAX, 1),
            (0, 1, 1, 0),
            (0, 1, u64::MAX, 4),
        ] {
            assert!(remaining_fit(now, frequency, deadline, count, workload).is_err());
        }
        Ok(())
    }
}
