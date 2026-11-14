//! Independent generated-C and Rust wire receipt controls.
use super::{Result, Value, fields_equal, flags_false, hexagon_policy, json, require};
pub(super) const SHARED: &[&str] = &[
    "schemaVersion",
    "testVectorId",
    "handleHex",
    "start",
    "stop",
];
pub(super) fn layout() -> Value {
    json!({"charBits":8,"pointerBytes":8,"sizeTBytes":8,"uint32Bytes":4,"uint64Bytes":8,"remoteHandle64Bytes":8,"remoteBufferBytes":16,"remoteBufferAlignment":8,"remoteBufferPointerOffset":0,"remoteBufferLengthOffset":8,"remoteArgBytes":16,"remoteArgAlignment":8,"littleEndian":true})
}
pub(super) fn capture(actual: &Value, expected: &Value) -> Result<()> {
    fields_equal(actual, expected, SHARED)?;
    require(
        actual["layout"] == layout()
            && actual["callCount"] == 2
            && actual["unexpectedCalls"] == 0
            && actual["returnCodes"] == json!({"start":0,"stop":0}),
        "Generated C wire failed fixed call/layout checks.",
    )
}
pub(super) fn parse(text: &str) -> Result<Value> {
    require(
        text.len() <= 16384,
        "Wire capture receipt exceeds fixed bound.",
    )?;
    hexagon_policy::data_json(text)
}
pub(super) fn rust_receipt(actual: &Value, captured: &Value) -> Result<()> {
    let fields = [
        "schemaVersion",
        "testVectorId",
        "handleHex",
        "start",
        "stop",
        "driverLoaded",
        "deviceExecuted",
        "sdkCompatibilityQualified",
        "profileEligible",
        "kind",
    ];
    let actual_keys = actual
        .as_object()
        .ok_or("Rust wire receipt must be an object.")?;
    require(
        actual_keys.len() == fields.len()
            && fields.iter().all(|key| actual_keys.contains_key(*key))
            && actual["kind"] == "public-idl-rust-encoding-control",
        "Rust encoding emitted unsupported receipt shape.",
    )?;
    flags_false(
        actual,
        &[
            "driverLoaded",
            "deviceExecuted",
            "sdkCompatibilityQualified",
            "profileEligible",
        ],
    )?;
    fields_equal(actual, captured, SHARED)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_wire_receipts_reject_changed_geometry_calls_and_authority() -> Result<()> {
        let mut expected = hexagon_policy::data_json(include_str!(
            "../../../../hexagon/public-qaic-wire.expected.json"
        ))?;
        expected["layout"] = layout();
        expected["callCount"] = json!(2);
        expected["unexpectedCalls"] = json!(0);
        expected["returnCodes"] = json!({"start":0,"stop":0});
        capture(&expected, &expected)?;
        for (pointer, value) in [
            ("/layout/remoteArgBytes", json!(8)),
            ("/callCount", json!("2")),
            ("/unexpectedCalls", json!(1)),
            ("/returnCodes/stop", json!(1)),
        ] {
            let mut changed = expected.clone();
            *changed
                .pointer_mut(pointer)
                .ok_or("Wire fixture field missing.")? = value;
            assert!(capture(&changed, &expected).is_err());
        }
        let mut rust = json!({"kind":"public-idl-rust-encoding-control","driverLoaded":false,"deviceExecuted":false,"sdkCompatibilityQualified":false,"profileEligible":false});
        for field in SHARED {
            rust[*field] = expected[*field].clone();
        }
        rust_receipt(&rust, &expected)?;
        for field in [
            "driverLoaded",
            "deviceExecuted",
            "sdkCompatibilityQualified",
            "profileEligible",
        ] {
            let mut changed = rust.clone();
            changed[field] = json!(true);
            assert!(rust_receipt(&changed, &expected).is_err());
        }
        rust["unreviewed"] = json!(true);
        assert!(rust_receipt(&rust, &expected).is_err());
        assert!(parse(&" ".repeat(16385)).is_err());
        Ok(())
    }
}
