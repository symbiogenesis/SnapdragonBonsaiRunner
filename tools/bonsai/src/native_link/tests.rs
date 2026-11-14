//! Literal link/package fixtures; these are not captures of compiler execution.

use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> std::result::Result<Self, Box<dyn std::error::Error>> {
        // The real command intentionally accepts ordinary disk paths, not
        // canonical Windows namespace paths (\\?\C:\...).
        let path = std::env::temp_dir().join(format!(
            "Bonsai-NativeLink-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }

    fn file(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> std::result::Result<Identity, Box<dyn std::error::Error>> {
        let path = self.0.join(name);
        fs::write(&path, bytes)?;
        Ok(Identity {
            path,
            bytes: u64::try_from(bytes.len())?,
            sha256: crate::identity::sha256(bytes),
        })
    }

    fn document(
        &self,
        name: &str,
        value: &Value,
    ) -> std::result::Result<Identity, Box<dyn std::error::Error>> {
        self.file(name, &serde_json::to_vec(value)?)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Ok(parent) = fs::canonicalize(std::env::temp_dir())
            && let Ok(target) = fs::canonicalize(&self.0)
            && target.parent() == Some(parent.as_path())
            && fs::symlink_metadata(&self.0)
                .is_ok_and(|metadata| !bonsai_artifacts::is_redirect(&metadata))
        {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

fn identity(value: &Identity) -> Value {
    json!({"path":value.path,"bytes":value.bytes,"sha256":value.sha256})
}

fn put16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn image() -> Vec<u8> {
    let mut bytes = vec![0; 0x1400];
    bytes[..2].copy_from_slice(b"MZ");
    put32(&mut bytes, 60, 0x80);
    put32(&mut bytes, 0x80, 0x4550);
    put16(&mut bytes, 0x84, 0xaa64);
    put16(&mut bytes, 0x86, 1);
    put16(&mut bytes, 0x94, 240);
    put16(&mut bytes, 0x98, 0x20b);
    put32(&mut bytes, 0x98 + 108, 16);
    bytes[0x188..0x18e].copy_from_slice(b".rdata");
    for offset in [0x190, 0x194, 0x198] {
        put32(&mut bytes, offset, 0x1000);
    }
    put32(&mut bytes, 0x19c, 0x400);
    put32(&mut bytes, 0x98 + 120, 0x1000);
    put32(&mut bytes, 0x98 + 124, 40);
    put32(&mut bytes, 0x40c, 0x1100);
    bytes[0x500..0x50d].copy_from_slice(b"kernel32.dll\0");
    put32(&mut bytes, 0x98 + 112 + 13 * 8, 0x1200);
    put32(&mut bytes, 0x98 + 116 + 13 * 8, 64);
    put32(&mut bytes, 0x600, 1);
    put32(&mut bytes, 0x604, 0x1300);
    bytes[0x700..0x70b].copy_from_slice(b"bcrypt.dll\0");
    bytes
}

fn coff_object(defined: bool) -> Vec<u8> {
    let mut bytes = vec![0; 101];
    put16(&mut bytes, 0, 0xaa64);
    put16(&mut bytes, 2, 1);
    put32(&mut bytes, 8, 60);
    put32(&mut bytes, 12, 1);
    put32(&mut bytes, 64, 4);
    put16(&mut bytes, 72, u16::from(defined));
    bytes[76] = 2;
    put32(&mut bytes, 78, 23);
    bytes[82..].copy_from_slice(b"__CxxFrameHandler3\0");
    bytes
}

fn runtime_image() -> Vec<u8> {
    let mut bytes = image();
    put32(&mut bytes, 0x400, 0x1400);
    put32(&mut bytes, 0x410, 0x1440);
    for table in [0x800, 0x840] {
        for (index, value) in [0x1480_u64, 0x14c0, (1 << 63) | 0x11]
            .into_iter()
            .enumerate()
        {
            bytes[table + index * 8..table + index * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }
    }
    bytes[0x882..0x895].copy_from_slice(b"_CxxThrowException\0");
    bytes[0x8c2..0x8d4].copy_from_slice(b"ordinary_function\0");
    put32(&mut bytes, 0x60c, 0x1540);
    put32(&mut bytes, 0x610, 0x1500);
    bytes[0x900..0x908].copy_from_slice(&0x1580_u64.to_le_bytes());
    bytes[0x908..0x910].copy_from_slice(&((1_u64 << 63) | 0x17).to_le_bytes());
    bytes[0x940..0x948].copy_from_slice(&0x1_4000_1000_u64.to_le_bytes());
    bytes[0x948..0x950].copy_from_slice(&0x1_4000_2000_u64.to_le_bytes());
    bytes[0x982..0x995].copy_from_slice(b"__CxxFrameHandler3\0");
    // An unrelated string in the same image cannot become an import witness.
    bytes[0xa00..0xa13].copy_from_slice(b"__CxxFrameHandler4\0");
    bytes
}

fn replace_image(scratch: &Scratch, capture: &mut Capture, bytes: &[u8]) -> TestResult {
    capture.links[0].output = scratch.file("linked.exe", bytes)?;
    let packaged = scratch.file("package/service.exe", bytes)?;
    capture.package_files[0].bytes = packaged.bytes;
    capture.package_files[0].sha256 = packaged.sha256;
    Ok(())
}

fn coff_archive() -> Vec<u8> {
    let mut bytes = b"!<arch>\n".to_vec();
    let mut header = [b' '; 60];
    header[..12].copy_from_slice(b"runtime.obj/");
    header[48..51].copy_from_slice(b"101");
    header[58..].copy_from_slice(b"`\n");
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&coff_object(true));
    bytes.push(b'\n');
    bytes
}

fn fixture(scratch: &Scratch) -> std::result::Result<Capture, Box<dyn std::error::Error>> {
    let object = scratch.file("source.obj", &coff_object(false))?;
    let linker = scratch.file("lld-link.exe", b"literal original linker bytes")?;
    let library = scratch.file("system.lib", &coff_archive())?;
    let source = scratch.document(
        "source-manifest.json",
        &json!({
            "schemaVersion":3,"kind":"selected-clang-native-source-closure",
            "sourceRevision":"a".repeat(40),"target":"aarch64-pc-windows-msvc",
            "configuration":"Release","compiler":{"sha256":"b".repeat(64)},
            "compilationDatabase":{"sha256":"c".repeat(64)},
            "units":[{"object":identity(&object)}]
        }),
    )?;
    let receipt = scratch.document(
        "source-receipt.json",
        &json!({
            "schemaVersion":1,"kind":"selected-native-source-closure-verification","passed":true,
            "manifestSchemaVersion":3,"recursiveGenerationProvenanceChecked":true,
            "selectedBuildInventoryChecked":true,"manifestSha256":source.sha256,
            "sourceRevision":"a".repeat(40),"target":"aarch64-pc-windows-msvc",
            "configuration":"Release","compilerSha256":"b".repeat(64),
            "compilationDatabaseSha256":"c".repeat(64),"translationUnits":1
        }),
    )?;
    fs::create_dir(scratch.0.join("package"))?;
    let output = scratch.file("linked.exe", &image())?;
    let packaged = scratch.file("package/service.exe", &image())?;
    let capture = Capture {
        schema_version: 1,
        kind: "selected-lld-arm64-link-package".into(),
        source_manifest: source,
        source_receipt: receipt,
        linker: linker.clone(),
        links: vec![Link {
            working_directory: scratch.0.clone(),
            arguments: vec![
                linker.path.to_str().ok_or("NonUTF8 scratch")?.into(),
                "/nodefaultlib".into(),
                "/machine:arm64".into(),
                "/manifest:no".into(),
                format!("/out:{}", output.path.display()),
                object.path.to_str().ok_or("NonUTF8 scratch")?.into(),
                library.path.to_str().ok_or("NonUTF8 scratch")?.into(),
            ],
            exit_code: 0,
            inputs: vec![
                Input {
                    kind: InputKind::CompiledObject,
                    identity: object,
                },
                Input {
                    kind: InputKind::Library,
                    identity: library,
                },
            ],
            output,
            package_name: "service.exe".into(),
            normal_imports: vec!["kernel32.dll".into()],
            delay_imports: vec!["bcrypt.dll".into()],
        }],
        package_directory: scratch.0.join("package"),
        package_files: vec![PackageFile {
            name: "service.exe".into(),
            bytes: packaged.bytes,
            sha256: packaged.sha256,
        }],
        system_imports: vec!["kernel32.dll".into(), "bcrypt.dll".into()],
    };
    // Every mutation control starts from a successfully admitted selection.
    // Otherwise a path/setup failure could masquerade as policy rejection.
    verify(&capture, &mut Intake::default())?;
    Ok(capture)
}

fn rejected<T>(result: Result<T>, expected: &str) {
    assert_eq!(result.err().as_deref(), Some(expected));
}

#[test]
fn native_link_literal_package_binds_all_objects_imports_and_limits_claims() -> TestResult {
    let scratch = Scratch::new()?;
    let manifest = fixture(&scratch)?;
    let receipt = verify(&manifest, &mut Intake::default())?;
    assert_eq!(receipt["translationUnits"], 1);
    assert_eq!(receipt["normalAndDelayImportsChecked"], true);
    assert_eq!(receipt["exactFlatPackageChecked"], true);
    assert_eq!(receipt["sourceClosureReexecuted"], false);
    assert_eq!(receipt["independentSourceReceiptReviewRequired"], true);
    assert_eq!(receipt["cppRuntimeAbsenceQualified"], false);
    assert_eq!(receipt["nativeRetirementQualified"], false);
    assert_eq!(receipt["coffRuntimeSymbolsChecked"], false);
    Ok(())
}

#[test]
fn native_link_schema_two_reports_coff_runtime_evidence_without_an_absence_verdict() -> TestResult {
    let scratch = Scratch::new()?;
    let mut manifest = fixture(&scratch)?;
    manifest.schema_version = 2;
    let receipt = verify(&manifest, &mut Intake::default())?;
    assert_eq!(receipt["coffRuntimeSymbolsChecked"], true);
    let inputs = receipt["coffRuntimeSymbolInputs"]
        .as_array()
        .ok_or("Missing COFF inputs")?;
    assert_eq!(inputs.len(), 2);
    for (file, binding, member) in [
        ("source.obj", "undefined", Value::Null),
        ("system.lib", "defined", json!(8)),
    ] {
        let input = inputs
            .iter()
            .find(|input| {
                input["path"]
                    .as_str()
                    .is_some_and(|path| path.ends_with(file))
            })
            .ok_or("Missing COFF input")?;
        assert_eq!(
            input["knownCppEhSymbols"],
            json!([{
                "archiveMemberOffset":member,"symbol":"__CxxFrameHandler3","binding":binding
            }])
        );
        assert!(
            input["sha256"]
                .as_str()
                .is_some_and(|hash| is_lower_hex(hash, 64))
        );
    }
    assert_eq!(receipt["archiveMemberLinkSelectionChecked"], false);
    assert_eq!(receipt["cppRuntimeAbsenceQualified"], false);
    assert_eq!(receipt["nativeRetirementQualified"], false);

    // Schema 1 replays its original scope. Schema 2 refuses opaque input bytes
    // rather than silently treating unsupported inputs as symbol-free.
    manifest.links[0].inputs[1].identity =
        scratch.file("system.lib", b"opaque old library fixture")?;
    manifest.schema_version = 1;
    verify(&manifest, &mut Intake::default())?;
    manifest.schema_version = 2;
    rejected(
        verify(&manifest, &mut Intake::default()),
        "COFF runtime inspection requires an ordinary MSVC archive.",
    );
    Ok(())
}

#[test]
fn native_link_schema_three_observes_named_and_ordinal_normal_and_delay_imports() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = fixture(&scratch)?;
    capture.schema_version = 3;
    replace_image(&scratch, &mut capture, &runtime_image())?;
    let receipt = verify(&capture, &mut Intake::default())?;
    assert_eq!(receipt["coffRuntimeSymbolsChecked"], true);
    assert_eq!(receipt["peRuntimeImportsChecked"], true);
    let image = &receipt["peRuntimeImportImages"][0];
    assert_eq!(image["packageName"], "service.exe");
    assert_eq!(image["sha256"], capture.links[0].output.sha256);
    assert_eq!(image["namedImportCount"], 3);
    assert_eq!(
        image["knownCppEhImports"],
        json!([
            {"module":"kernel32.dll","delayed":false,"symbol":"_CxxThrowException"},
            {"module":"bcrypt.dll","delayed":true,"symbol":"__CxxFrameHandler3"}
        ])
    );
    assert_eq!(
        image["ordinalImports"],
        json!([
            {"module":"kernel32.dll","delayed":false,"ordinal":17},
            {"module":"bcrypt.dll","delayed":true,"ordinal":23}
        ])
    );
    assert_eq!(receipt["ordinalImportTargetsResolved"], false);
    assert_eq!(receipt["cppRuntimeAbsenceQualified"], false);
    assert_eq!(receipt["nativeRetirementQualified"], false);
    Ok(())
}

#[test]
fn native_link_schema_three_uses_only_unbound_iat_when_lookup_is_absent() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = fixture(&scratch)?;
    capture.schema_version = 3;
    let mut bytes = runtime_image();
    put32(&mut bytes, 0x400, 0);
    replace_image(&scratch, &mut capture, &bytes)?;
    let receipt = verify(&capture, &mut Intake::default())?;
    assert_eq!(receipt["peRuntimeImportImages"][0]["namedImportCount"], 3);
    put32(&mut bytes, 0x404, 1);
    replace_image(&scratch, &mut capture, &bytes)?;
    rejected(
        verify(&capture, &mut Intake::default()),
        "PE symbol inspection requires lookup and address tables",
    );
    Ok(())
}

#[test]
fn native_link_schema_three_rejects_unrecoverable_or_malformed_symbol_tables() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = fixture(&scratch)?;
    capture.schema_version = 3;
    for (offset, mutation, expected) in [
        (
            0x800,
            (1_u64 << 63 | 0x1_0011).to_le_bytes().to_vec(),
            "PE ordinal import contains reserved bits",
        ),
        (
            0x800,
            0x8000_1480_u64.to_le_bytes().to_vec(),
            "PE name import contains reserved bits",
        ),
        (
            0x882,
            vec![0],
            "PE import symbol is empty, unterminated or exceeds its bound",
        ),
        (
            0x882,
            vec![b' '],
            "PE import symbol must contain only ASCII graphic bytes",
        ),
        (
            0x858,
            1_u64.to_le_bytes().to_vec(),
            "PE import address table lacks its matching terminator",
        ),
        (
            0x600,
            3_u32.to_le_bytes().to_vec(),
            "PE symbol inspection requires RVA-based delay descriptors",
        ),
        (
            0x610,
            0_u32.to_le_bytes().to_vec(),
            "PE symbol inspection requires lookup and address tables",
        ),
        (
            0x400,
            0x1ffc_u32.to_le_bytes().to_vec(),
            "PE RVA extent outside raw section",
        ),
        (
            0x800,
            0x1ffe_u64.to_le_bytes().to_vec(),
            "PE RVA extent outside raw section",
        ),
    ] {
        let mut bytes = runtime_image();
        bytes[offset..offset + mutation.len()].copy_from_slice(&mutation);
        replace_image(&scratch, &mut capture, &bytes)?;
        rejected(verify(&capture, &mut Intake::default()), expected);
    }
    Ok(())
}

#[test]
fn native_link_schema_three_requires_thunks_without_extending_historical_replay() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = fixture(&scratch)?;
    capture.schema_version = 2;
    let receipt = verify(&capture, &mut Intake::default())?;
    assert_eq!(receipt["peRuntimeImportsChecked"], false);
    capture.schema_version = 3;
    rejected(
        verify(&capture, &mut Intake::default()),
        "PE symbol inspection requires lookup and address tables",
    );
    Ok(())
}

#[test]
fn native_link_explicit_invocation_rejects_hidden_inputs_and_failed_links() -> TestResult {
    let scratch = Scratch::new()?;
    let mut manifest = fixture(&scratch)?;
    for option in [
        "@response.rsp",
        "/libpath:C:/hidden",
        "/defaultlib:hidden.lib",
        "relative.obj",
        "/incremental",
    ] {
        manifest.links[0].arguments.push(option.into());
        rejected(
            verify_arguments(&manifest.links[0], &manifest.linker),
            "Native link has an unsupported or repeated option; response/search/default-library inputs are excluded.",
        );
        manifest.links[0].arguments.pop();
    }
    let ordinary = manifest.links[0].arguments.clone();
    manifest.links[0]
        .arguments
        .retain(|value| value != "/nodefaultlib");
    rejected(
        verify_arguments(&manifest.links[0], &manifest.linker),
        "Native link requires explicit input, ARM64, output and disabled implicit-library/manifest paths.",
    );
    manifest.links[0].arguments = ordinary;
    manifest.links[0].exit_code = 1;
    rejected(
        verify_arguments(&manifest.links[0], &manifest.linker),
        "Native link lacks a successful explicit original invocation.",
    );
    Ok(())
}

#[test]
fn native_link_rejects_object_omission_relabeling_and_changed_receipt() -> TestResult {
    let scratch = Scratch::new()?;
    let mut manifest = fixture(&scratch)?;
    let removed = manifest.links[0].inputs.remove(0);
    rejected(
        verify(&manifest, &mut Intake::default()),
        "Native link inputs differ from its explicit invocation.",
    );
    manifest.links[0].inputs.insert(0, removed);
    manifest.links[0].inputs[0].kind = InputKind::ToolchainObject;
    rejected(
        verify(&manifest, &mut Intake::default()),
        "Native link cannot relabel a selected compiled object.",
    );
    manifest.links[0].inputs[0].kind = InputKind::CompiledObject;
    let mut receipt: Value = serde_json::from_slice(&fs::read(&manifest.source_receipt.path)?)?;
    for field in [
        "manifestSha256",
        "compilerSha256",
        "compilationDatabaseSha256",
    ] {
        let original = receipt[field].clone();
        receipt[field] = "d".repeat(64).into();
        manifest.source_receipt = scratch.document("source-receipt.json", &receipt)?;
        rejected(
            verify(&manifest, &mut Intake::default()),
            "Native link requires the matching independently reviewed schema-3 source receipt.",
        );
        receipt[field] = original;
    }
    Ok(())
}

#[test]
fn native_link_rejects_missing_extra_or_changed_package_bytes() -> TestResult {
    let scratch = Scratch::new()?;
    let mut manifest = fixture(&scratch)?;
    scratch.file("package/extra.txt", b"unexpected")?;
    rejected(
        verify(&manifest, &mut Intake::default()),
        "Native link package has missing or extra entries.",
    );
    fs::remove_file(scratch.0.join("package/extra.txt"))?;
    let original = manifest.package_files[0].sha256.clone();
    manifest.package_files[0].sha256 = "d".repeat(64);
    rejected(
        verify(&manifest, &mut Intake::default()),
        "Artifact length or SHA-256 differs from the compiled lock.",
    );
    manifest.package_files[0].sha256 = original;
    manifest.links[0].output.sha256 = "d".repeat(64);
    rejected(
        verify(&manifest, &mut Intake::default()),
        "Artifact length or SHA-256 differs from the compiled lock.",
    );
    manifest.links[0].output.sha256 = manifest.package_files[0].sha256.clone();
    fs::remove_file(scratch.0.join("package/service.exe"))?;
    rejected(
        verify(&manifest, &mut Intake::default()),
        "Artifact storage could not be opened safely.",
    );
    Ok(())
}

#[test]
fn native_link_rejects_unreviewed_delay_imports_and_unbound_boundaries() -> TestResult {
    let scratch = Scratch::new()?;
    let mut manifest = fixture(&scratch)?;
    manifest.links[0].delay_imports.clear();
    rejected(
        verify(&manifest, &mut Intake::default()),
        "Native link PE imports differ from the reviewed inventory.",
    );
    manifest.links[0].delay_imports.push("bcrypt.dll".into());
    manifest.system_imports.pop();
    rejected(
        verify(&manifest, &mut Intake::default()),
        "Native link PE import is neither packaged nor a declared system boundary.",
    );
    manifest.system_imports.push("bcrypt.dll".into());
    manifest.system_imports.push("unused.dll".into());
    rejected(
        verify(&manifest, &mut Intake::default()),
        "Native link omits source objects/package images or declares unused system imports.",
    );
    assert!(module_set(&["../unsafe.dll".into()]).is_err());
    assert!(module_set(&["kernel32.dll".into(), "kernel32.dll".into()]).is_err());
    Ok(())
}

#[test]
fn native_link_manifest_intake_rejects_digest_tamper_and_unknown_fields() -> TestResult {
    let scratch = Scratch::new()?;
    let mut manifest = scratch.document("link.json", &json!({"schemaVersion":1,"unknown":true}))?;
    assert!(
        run(json!({"manifest":identity(&manifest)}))
            .err()
            .is_some_and(|error| error.contains("unknown field"))
    );
    manifest.sha256 = "d".repeat(64);
    rejected(
        run(json!({"manifest":identity(&manifest)})),
        "Artifact length or SHA-256 differs from the compiled lock.",
    );
    assert!(
        run(json!({"manifest":identity(&manifest),"ignoreErrors":true}))
            .err()
            .is_some_and(|error| error.contains("unknown field"))
    );
    Ok(())
}

#[test]
fn native_link_source_binding_rejects_paired_missing_and_malformed_identities() -> TestResult {
    let scratch = Scratch::new()?;
    let mut manifest = fixture(&scratch)?;
    let original_source: Value =
        serde_json::from_slice(&fs::read(&manifest.source_manifest.path)?)?;
    let original_receipt: Value =
        serde_json::from_slice(&fs::read(&manifest.source_receipt.path)?)?;
    for (source_path, receipt_key) in [
        ("/sourceRevision", "sourceRevision"),
        ("/configuration", "configuration"),
        ("/compiler/sha256", "compilerSha256"),
        ("/compilationDatabase/sha256", "compilationDatabaseSha256"),
    ] {
        for malformed in [
            None,
            Some(Value::Null),
            Some(json!(17)),
            Some(json!("")),
            Some(json!("bad\nidentity")),
        ] {
            let mut source = original_source.clone();
            let mut receipt = original_receipt.clone();
            if let Some(malformed) = malformed {
                *source
                    .pointer_mut(source_path)
                    .ok_or("Missing fixture identity")? = malformed.clone();
                receipt[receipt_key] = malformed;
            } else {
                let (parent, key) = source_path.rsplit_once('/').ok_or("Missing fixture path")?;
                source
                    .pointer_mut(parent)
                    .and_then(Value::as_object_mut)
                    .ok_or("Missing fixture object")?
                    .remove(key);
                receipt
                    .as_object_mut()
                    .ok_or("Missing fixture receipt")?
                    .remove(receipt_key);
            }
            manifest.source_manifest = scratch.document("source-manifest.json", &source)?;
            receipt["manifestSha256"] = manifest.source_manifest.sha256.clone().into();
            manifest.source_receipt = scratch.document("source-receipt.json", &receipt)?;
            rejected(
                source_objects(&manifest, &mut Intake::default()),
                "Native link requires the matching independently reviewed schema-3 source receipt.",
            );
        }
    }
    Ok(())
}
