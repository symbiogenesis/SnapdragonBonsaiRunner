//! Pure post-bootstrap report gates shared by hosted workflow adapters.
use crate::Result;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
#[serde(
    tag = "operation",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum Request {
    RuntimeCapabilities {
        report: Value,
    },
    AdmissionOnly {
        report: Value,
    },
    SwiftAdmission {
        report: Value,
    },
    AdapterAdmission {
        report: Value,
    },
    Ptq1Admission {
        report: Value,
    },
    RuntimeServerPlan {
        report: Value,
    },
    CapturePlan {
        report: Value,
        expected_capture_mode: String,
    },
}
struct Record<'a>(&'a Value);
impl Record<'_> {
    fn equal(&self, pointer: &str, expected: &Value) -> Result<()> {
        if self.0.pointer(pointer) != Some(expected) {
            return Err(format!(
                "Hosted report violates its fixed contract: {pointer}"
            ));
        }
        Ok(())
    }
    fn flag(&self, name: &str, expected: bool) -> Result<()> {
        self.equal(&format!("/{name}"), &Value::Bool(expected))
    }
    fn count(&self, name: &str, expected: u64) -> Result<()> {
        if self.0.get(name).and_then(Value::as_u64) != Some(expected) {
            return Err(format!("Hosted report requires an exact integer: {name}"));
        }
        Ok(())
    }
    fn no_model(&self) -> Result<()> {
        self.flag("ModelProcessStarted", false)
    }
    fn text(&self, name: &str) -> Result<&str> {
        self.0
            .get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("Hosted report requires text: {name}"))
    }
}
fn require(valid: bool, message: &str) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

/// Validate one fixed hosted report and return its original fields unchanged.
///
/// # Errors
/// Rejects missing, coerced or changed fields, unexpected model launch, altered
/// memory counts and broadened work. This performs no host or model operation.
pub fn run(request: Value) -> Result<Value> {
    let request: Request = serde_json::from_value(request).map_err(|error| error.to_string())?;
    let report = match request {
        Request::RuntimeCapabilities { report } => {
            let record = Record(&report);
            record.equal("/CapabilityVerification/Status", &Value::from("verified"))?;
            record.equal("/Capabilities/qualification", &Value::from("unqualified"))?;
            record.equal(
                "/Capabilities/runtime/target/architecture",
                &Value::from("aarch64"),
            )?;
            report
        }
        Request::AdmissionOnly { report } => {
            let record = Record(&report);
            record.no_model()?;
            require(
                report["AvailablePhysicalBytesAtAdmission"]
                    .as_u64()
                    .is_some_and(|bytes| bytes > 0),
                "Admission-only report requires positive observed available memory.",
            )?;
            report
        }
        Request::SwiftAdmission { report } => {
            let record = Record(&report);
            record.no_model()?;
            for (field, count) in [
                ("RequiredAvailableBytes", 16_642_998_272),
                ("ServiceCeilingBytes", 12_348_030_976),
                ("AvailableMemoryFloorBytes", 4_294_967_296),
            ] {
                record.count(field, count)?;
            }
            report
        }
        Request::AdapterAdmission { report } => {
            let record = Record(&report);
            record.no_model()?;
            record.flag("AdapterRequested", true)?;
            record.count("EstimatedAdapterBytes", 9_682_464)?;
            report
        }
        Request::Ptq1Admission { report } => {
            let record = Record(&report);
            record.no_model()?;
            record.count("EstimatedWeightBytes", 5_946_648_928)?;
            require(
                record.text("WeightRepresentation")? == "current-repacked-PTQ1_0",
                "Derived PTQ1 admission representation changed.",
            )?;
            report
        }
        Request::RuntimeServerPlan { report } => {
            let record = Record(&report);
            record.no_model()?;
            record.flag("LaunchEnabled", false)?;
            require(
                record
                    .text("RuntimeHelpVerification")?
                    .starts_with("all-planned-switches-advertised")
                    && record
                        .text("ArtifactVerification")?
                        .ends_with("model-not-checked"),
                "Runtime-only server preflight contract changed.",
            )?;
            report
        }
        Request::CapturePlan {
            report,
            expected_capture_mode,
        } => {
            require(
                ["numerical", "request-timing"].contains(&expected_capture_mode.as_str()),
                "Unsupported fixed capture mode.",
            )?;
            let record = Record(&report);
            record.no_model()?;
            record.flag("DefaultMayChange", false)?;
            require(
                record.text("CaptureMode")? == expected_capture_mode,
                "Capture plan differs from its selected mode.",
            )?;
            if expected_capture_mode == "request-timing" {
                for (field, count) in [
                    ("NativeSamples", 0),
                    ("RustSamples", 1),
                    ("OutputTokens", 1),
                ] {
                    record.count(field, count)?;
                }
            }
            report
        }
    };
    require(report.is_object(), "Hosted report must be an object.")?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn fixtures() -> Vec<(&'static str, Value)> {
        vec![
            (
                "runtime-capabilities",
                json!({"CapabilityVerification":{"Status":"verified"},"Capabilities":{"qualification":"unqualified","runtime":{"target":{"architecture":"aarch64"}}}}),
            ),
            (
                "admission-only",
                json!({"ModelProcessStarted":false,"AvailablePhysicalBytesAtAdmission":1}),
            ),
            (
                "swift-admission",
                json!({"ModelProcessStarted":false,"RequiredAvailableBytes":16_642_998_272_u64,"ServiceCeilingBytes":12_348_030_976_u64,"AvailableMemoryFloorBytes":4_294_967_296_u64}),
            ),
            (
                "adapter-admission",
                json!({"ModelProcessStarted":false,"AdapterRequested":true,"EstimatedAdapterBytes":9_682_464}),
            ),
            (
                "ptq1-admission",
                json!({"ModelProcessStarted":false,"EstimatedWeightBytes":5_946_648_928_u64,"WeightRepresentation":"current-repacked-PTQ1_0"}),
            ),
            (
                "runtime-server-plan",
                json!({"ModelProcessStarted":false,"LaunchEnabled":false,"RuntimeHelpVerification":"all-planned-switches-advertised;checked","ArtifactVerification":"runtime-verified;model-not-checked"}),
            ),
            (
                "capture-plan",
                json!({"ModelProcessStarted":false,"DefaultMayChange":false,"CaptureMode":"request-timing","NativeSamples":0,"RustSamples":1,"OutputTokens":1}),
            ),
        ]
    }
    fn request(operation: &str, report: &Value) -> Value {
        let mut value = json!({"operation":operation,"report":report});
        if operation == "capture-plan" {
            value["expectedCaptureMode"] = json!("request-timing");
        }
        value
    }
    #[test]
    fn hosted_report_gates_preserve_verified_fields_without_host_operations() -> Result<()> {
        for (operation, report) in fixtures() {
            assert_eq!(run(request(operation, &report))?, report);
        }
        Ok(())
    }
    #[test]
    fn hosted_report_gates_reject_changed_coerced_and_missing_contract_fields() -> Result<()> {
        for (operation, pointer, value) in [
            (
                "runtime-capabilities",
                "/Capabilities/runtime/target/architecture",
                json!("x86_64"),
            ),
            (
                "runtime-capabilities",
                "/Capabilities/qualification",
                json!("qualified"),
            ),
            (
                "runtime-capabilities",
                "/CapabilityVerification/Status",
                json!(null),
            ),
            (
                "admission-only",
                "/AvailablePhysicalBytesAtAdmission",
                json!(0),
            ),
            (
                "admission-only",
                "/AvailablePhysicalBytesAtAdmission",
                json!("1"),
            ),
            (
                "swift-admission",
                "/RequiredAvailableBytes",
                json!(16_642_998_272.0),
            ),
            (
                "swift-admission",
                "/ServiceCeilingBytes",
                json!(12_348_030_975_u64),
            ),
            (
                "swift-admission",
                "/AvailableMemoryFloorBytes",
                json!(false),
            ),
            ("adapter-admission", "/AdapterRequested", json!(1)),
            (
                "adapter-admission",
                "/EstimatedAdapterBytes",
                json!(9_682_463),
            ),
            ("ptq1-admission", "/WeightRepresentation", json!("PQ2_0")),
            (
                "ptq1-admission",
                "/EstimatedWeightBytes",
                json!(5_946_648_927_u64),
            ),
            ("runtime-server-plan", "/LaunchEnabled", json!(true)),
            (
                "runtime-server-plan",
                "/RuntimeHelpVerification",
                json!("unchecked"),
            ),
            (
                "runtime-server-plan",
                "/ArtifactVerification",
                json!("model-checked"),
            ),
            ("capture-plan", "/CaptureMode", json!("numerical")),
            ("capture-plan", "/DefaultMayChange", json!(true)),
            ("capture-plan", "/NativeSamples", json!(1)),
            ("capture-plan", "/RustSamples", json!(0)),
            ("capture-plan", "/OutputTokens", json!(2)),
        ] {
            let (_, mut report) = fixtures()
                .into_iter()
                .find(|(kind, _)| *kind == operation)
                .ok_or("Missing hosted report fixture.")?;
            *report
                .pointer_mut(pointer)
                .ok_or("Missing fixture pointer.")? = value;
            assert!(
                run(request(operation, &report)).is_err(),
                "{operation}: {pointer}"
            );
        }
        for (operation, mut report) in fixtures()
            .into_iter()
            .filter(|(_, value)| value.get("ModelProcessStarted").is_some())
        {
            report["ModelProcessStarted"] = json!(true);
            assert!(run(request(operation, &report)).is_err());
        }
        assert!(run(json!({"operation":"admission-only","report":{}})).is_err());
        Ok(())
    }
}
