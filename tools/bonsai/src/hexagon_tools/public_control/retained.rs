//! Exact historical public-wire observations; no C source is compiled or executed.
use super::*;

const ARCHIVE: &str = "docs/evidence/hexagon-public-qaic-41171859264";
const PROVENANCE: &str = "ff5f18b5ff5c3694b7d572bb9c16cf609dac04b63afea61756a761b8a62248d6";

fn same_identity(left: &Value, right: &Value) -> Result<()> {
    fields_equal(left, right, &["bytes", "sha256"])
}

fn pinned_json(path: &Path, expected: &Value, maximum: u64) -> Result<Value> {
    let bytes = read(path, maximum)?;
    require(
        expected["bytes"] == bytes.len() && expected["sha256"] == ids::sha256(&bytes),
        "Parsed retained wire bytes differ from the exact reviewed identity.",
    )?;
    hexagon_policy::data_json_with_limit(
        std::str::from_utf8(&bytes).map_err(|error| error.to_string())?,
        usize::try_from(maximum).map_err(|error| error.to_string())?,
    )
}

fn provider(host: &Value) -> Result<()> {
    require(
        host["schemaVersion"] == 1
            && host["kind"] == "public-QAIC-independent-source-oracle"
            && host["runnerRevision"] == "8c69947b7b98a8d406d15c639d83f926a3b643cc"
            && host["runId"] == "41171859264"
            && host["qaicCommit"] == "a66a0019fb488abae955bbc9b3a9de3dc15962d0"
            && host["outcome"] == "public-generated-C-and-Rust-wire-cross-checked-only"
            && host["generatorCompleted"] == true
            && host["wireCapture"]["passed"] == true
            && host["wireCapture"]["mockCompleted"] == true
            && host["wireCapture"]["compilerIdentityRechecked"] == true
            && host["wireCapture"]["compiler"]["target"] == "x86_64-linux-gnu"
            && host["rustWireControl"]["passed"] == true
            && host["rustWireControl"]["testExecutionCompleted"] == true
            && host["rustWireControl"]["encodingExecutionCompleted"] == true,
        "Retained wire provider lacks the exact successful independent observation.",
    )?;
    flags_false(
        &host["wireCapture"],
        &[
            "sdkLinked",
            "driverLoaded",
            "deviceCodeLoaded",
            "sdkEquivalenceQualified",
            "WindowsArm64AbiQualified",
        ],
    )?;
    Ok(())
}

/// Replays the fixed original provider, inputs and complete outputs without a compiler.
#[expect(
    clippy::too_many_lines,
    reason = "One replay binds the immutable original provider, complete retained input set and both original outputs before admitting the fixed vector."
)]
pub(super) fn verify(root: &Path, public: &Value) -> Result<Value> {
    require(
        public["schemaVersion"] == 2
            && public["qaicCommit"] == "a66a0019fb488abae955bbc9b3a9de3dc15962d0"
            && public["wireCapture"]["mode"] == "retained-independent-observation",
        "Unsupported retained public-wire contract.",
    )?;
    flags_false(public, &["deviceInvocationAuthorized", "profileEligible"])?;
    flags_false(
        &public["wireCapture"],
        &["sdkLinked", "deviceInvocationAuthorized", "profileEligible"],
    )?;
    let archive = root.join(ARCHIVE);
    let provenance = archive.join("provenance.json");
    let provenance_pin = json!({"bytes":15872,"sha256":PROVENANCE});
    let selection = pinned_json(&provenance, &provenance_pin, 1_048_576)?;
    for entry in list(&selection, "retainedFiles")? {
        let path = string(entry, "path")?;
        require(
            !path.is_empty()
                && path
                    .split('/')
                    .all(|part| !part.is_empty() && part != "." && part != "..")
                && path
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"/._-".contains(&byte)),
            "Retained wire selection has an unsupported path.",
        )?;
        pinned(&archive.join(path), entry)?;
    }
    let host_pin = list(&selection, "retainedFiles")?
        .iter()
        .find(|entry| entry["path"] == "host.json")
        .ok_or("Retained provider identity missing.")?;
    let host = pinned_json(&archive.join("host.json"), host_pin, 1_048_576)?;
    provider(&host)?;
    let original = &host["wireCapture"];
    for (field, subdirectory, observed_inputs) in [
        ("headers", "wire-public-inputs", list(original, "headers")?),
        (
            "reviewedGeneratedFiles",
            "generated-public-source",
            list(&host, "generatedFiles")?,
        ),
    ] {
        let selected = list(&public["wireCapture"], field)?;
        require(
            selected.len() == observed_inputs.len(),
            "Retained public source selection is incomplete.",
        )?;
        let mut names = std::collections::BTreeSet::new();
        for input in selected {
            let name = string(input, "name")?;
            require(
                names.insert(name),
                "Retained public source selection contains duplicates.",
            )?;
            let observed_input = observed_inputs
                .iter()
                .find(|item| item["name"] == name)
                .ok_or("Retained public source selection changed.")?;
            same_identity(input, observed_input)?;
            let path = archive.join(subdirectory).join(format!("{name}.txt"));
            pinned(&path, input)
                .map_err(|error| format!("Reviewed public input {}: {error}", path.display()))?;
        }
    }
    same_identity(&public["wireCapture"]["harness"], &original["harness"])?;
    same_identity(
        &public["wireCapture"]["expected"],
        &original["independentExpected"],
    )?;
    // These original source files were restored from their exact existing Git
    // blobs, rather than changing the immutable historical retention receipt.
    pinned(
        &archive.join("wire-public-inputs/public_qaic_wire.c.txt"),
        &original["harness"],
    )?;
    pinned(
        &archive.join("public_idl_control.rs.txt"),
        &host["rustWireControl"]["source"],
    )?;
    let expected_path = root.join("tools/hexagon/public-qaic-wire.expected.json");
    let expected = pinned_json(&expected_path, &original["independentExpected"], 16_384)?;
    let captured = archive.join("wire-capture.log");
    let observed = pinned_json(&captured, &original["captureReceipt"], 16_384)?;
    require(
        observed == original["observed"],
        "Retained C output differs from its provider receipt.",
    )?;
    wire::capture(&observed, &expected)?;
    let rust_path = archive.join("wire-rust-encoding.log");
    let rust = pinned_json(
        &rust_path,
        &host["rustWireControl"]["captureReceipt"],
        16_384,
    )?;
    require(
        rust == host["rustWireControl"]["observed"],
        "Retained Rust output differs from its provider receipt.",
    )?;
    wire::rust_receipt(&rust, &observed)?;
    Ok(
        json!({"kind":"retained-independent-public-wire-replay","passed":true,
        "runId":host["runId"],"runnerRevision":host["runnerRevision"],
        "provenance":provenance_pin,"host":host_pin,
        "originalCompiler":original["compiler"],"originalExecutable":original["executable"],
        "captureReceipt":original["captureReceipt"],"observed":observed,
        "originalSourceInputsVerified":true,"freshCExecution":false,
        "scope":"fixed historical Linux LP64 vector; no fresh native C declaration or calling-convention check",
        "driverLoaded":false,"deviceExecuted":false,"sdkEquivalenceQualified":false,
        "WindowsArm64AbiQualified":false,"profileEligible":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> Result<&'static Path> {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| "Fixture repository unavailable.".into())
    }

    #[test]
    fn retained_public_wire_replays_exact_original_inputs_and_both_observations() -> Result<()> {
        let root = root()?;
        let public = json_file(&root.join("tools/hexagon/public-qaic.lock.json"), 1_048_576)?;
        let replay = verify(root, &public)?;
        assert_eq!(replay["passed"], true);
        assert_eq!(replay["freshCExecution"], false);
        assert_eq!(replay["observed"]["start"]["primaryBytes"], 48);
        assert_eq!(
            replay["observed"]["start"]["primaryHex"],
            "443322110000000018077665544332216453423174635241483726150473625158473625140372611403827100000000"
        );
        assert!(!root.join("tools/hexagon/public_qaic_wire.c").exists());
        Ok(())
    }

    #[test]
    fn retained_public_wire_rejects_changed_inputs_and_missing_archive() -> Result<()> {
        let root = root()?;
        let public = json_file(&root.join("tools/hexagon/public-qaic.lock.json"), 1_048_576)?;
        for pointer in [
            "/wireCapture/harness/sha256",
            "/wireCapture/expected/sha256",
            "/wireCapture/headers/0/sha256",
            "/wireCapture/reviewedGeneratedFiles/0/sha256",
        ] {
            let mut changed = public.clone();
            *changed
                .pointer_mut(pointer)
                .ok_or("Missing fixture identity.")? = json!("0".repeat(64));
            assert!(verify(root, &changed).is_err(), "accepted {pointer}");
        }
        assert!(verify(&root.join("Cargo.toml"), &public).is_err());
        Ok(())
    }

    #[test]
    fn retained_public_wire_rejects_provider_failure_and_authority_changes() -> Result<()> {
        let host = json_file(&root()?.join(ARCHIVE).join("host.json"), 1_048_576)?;
        provider(&host)?;
        for (pointer, value) in [
            ("/runId", json!("different")),
            ("/wireCapture/passed", json!(false)),
            (
                "/wireCapture/compiler/target",
                json!("aarch64-pc-windows-msvc"),
            ),
            ("/wireCapture/compilerIdentityRechecked", json!(false)),
            ("/wireCapture/mockCompleted", json!(false)),
            ("/wireCapture/WindowsArm64AbiQualified", json!(true)),
            ("/rustWireControl/testExecutionCompleted", json!(false)),
        ] {
            let mut changed = host.clone();
            *changed
                .pointer_mut(pointer)
                .ok_or("Missing provider fixture field.")? = value;
            assert!(provider(&changed).is_err(), "accepted {pointer}");
        }
        Ok(())
    }
}
