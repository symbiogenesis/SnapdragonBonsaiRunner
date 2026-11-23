//! Literal link/package fixtures; these are not captures of compiler execution.

use super::*;

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

struct Scratch(crate::bounded_input::ScratchDir);

impl Scratch {
    fn new() -> std::result::Result<Self, Box<dyn std::error::Error>> {
        // The real command intentionally accepts ordinary disk paths, not
        // canonical Windows namespace paths (\\?\C:\...).
        Ok(Self(crate::bounded_input::ScratchDir::new(
            "Bonsai-NativeLink",
        )?))
    }

    fn path(&self) -> &Path {
        self.0.path()
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

fn export_image(
    base: u32,
    names: &[(&str, u16)],
    targets: &[Option<&str>],
) -> std::result::Result<Vec<u8>, std::num::TryFromIntError> {
    let mut bytes = image();
    put32(&mut bytes, 0x98 + 112, 0x1600);
    put32(&mut bytes, 0x98 + 116, 0x400);
    put32(&mut bytes, 0xa10, base);
    put32(&mut bytes, 0xa14, u32::try_from(targets.len())?);
    put32(&mut bytes, 0xa18, u32::try_from(names.len())?);
    put32(&mut bytes, 0xa1c, 0x1640);
    put32(&mut bytes, 0xa20, 0x1680);
    put32(&mut bytes, 0xa24, 0x16c0);
    let mut string = 0xb00;
    for (index, (name, ordinal)) in names.iter().enumerate() {
        put32(
            &mut bytes,
            0xa80 + index * 4,
            u32::try_from(string + 0xc00)?,
        );
        put16(&mut bytes, 0xac0 + index * 2, *ordinal);
        bytes[string..string + name.len()].copy_from_slice(name.as_bytes());
        string += name.len() + 1;
    }
    for (index, target) in targets.iter().enumerate() {
        let rva = if let Some(target) = target {
            let rva = u32::try_from(string + 0xc00)?;
            bytes[string..string + target.len()].copy_from_slice(target.as_bytes());
            string += target.len() + 1;
            rva
        } else {
            0x1e00
        };
        put32(&mut bytes, 0xa40 + index * 4, rva);
    }
    Ok(bytes)
}

fn dependency(scratch: &Scratch, capture: &mut Capture, module: &str, bytes: &[u8]) -> TestResult {
    let identity = scratch.file(module, bytes)?;
    if let Some(existing) = capture
        .dependency_images
        .iter_mut()
        .find(|item| item.module == module)
    {
        existing.identity = identity;
    } else {
        capture.dependency_images.push(DependencyImage {
            module: module.into(),
            identity,
        });
    }
    Ok(())
}

fn runtime_fixture(scratch: &Scratch) -> std::result::Result<Capture, Box<dyn std::error::Error>> {
    let mut capture = fixture(scratch)?;
    capture.schema_version = 4;
    replace_image(scratch, &mut capture, &runtime_image())?;
    dependency(
        scratch,
        &mut capture,
        "kernel32.dll",
        &export_image(
            17,
            &[("_CxxThrowException", 0), ("ordinary_function", 1)],
            &[None, None],
        )?,
    )?;
    dependency(
        scratch,
        &mut capture,
        "bcrypt.dll",
        &export_image(23, &[("__CxxFrameHandler3", 0)], &[None])?,
    )?;
    verify(&capture, &mut Intake::default())?;
    Ok(capture)
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
    fs::create_dir(scratch.path().join("package"))?;
    let output = scratch.file("linked.exe", &image())?;
    let packaged = scratch.file("package/service.exe", &image())?;
    let capture = Capture {
        schema_version: 1,
        kind: "selected-lld-arm64-link-package".into(),
        source_manifest: source,
        source_receipt: receipt,
        linker: linker.clone(),
        links: vec![Link {
            working_directory: scratch.path().to_path_buf(),
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
        package_directory: scratch.path().join("package"),
        package_files: vec![PackageFile {
            name: "service.exe".into(),
            bytes: packaged.bytes,
            sha256: packaged.sha256,
        }],
        system_imports: vec!["kernel32.dll".into(), "bcrypt.dll".into()],
        dependency_images: Vec::new(),
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
fn native_link_schema_four_resolves_ordinal_aliases_against_exact_provider_images() -> TestResult {
    let scratch = Scratch::new()?;
    let capture = runtime_fixture(&scratch)?;
    let receipt = verify(&capture, &mut Intake::default())?;
    assert_eq!(receipt["ordinalImportTargetsResolved"], true);
    assert_eq!(receipt["forwardedImportTargetsResolved"], true);
    let boundary = &receipt["runtimeImportBoundary"];
    assert_eq!(
        boundary["images"].as_array().ok_or("Missing images")?.len(),
        2
    );
    let routes = boundary["routes"].as_array().ok_or("Missing routes")?;
    assert_eq!(routes.len(), 5);
    assert_eq!(routes[2]["importSymbol"], json!({"ordinal":17}));
    assert_eq!(routes[2]["knownCppEhNames"], json!(["_CxxThrowException"]));
    assert_eq!(routes[2]["stages"][0]["exportOrdinal"], 17);
    assert_eq!(routes[4]["delayed"], true);
    assert_eq!(routes[4]["knownCppEhNames"], json!(["__CxxFrameHandler3"]));
    for image in boundary["images"].as_array().ok_or("Missing images")? {
        let supplied = capture
            .dependency_images
            .iter()
            .find(|item| item.module == image["module"])
            .ok_or("Unexpected provider image")?;
        assert_eq!(image["bytes"], supplied.identity.bytes);
        assert_eq!(image["sha256"], supplied.identity.sha256);
    }
    assert_eq!(boundary["recursiveDependencyImportsChecked"], false);
    assert_eq!(boundary["apiSetRoutingQualified"], false);
    assert_eq!(boundary["providerImplementationLanguageQualified"], false);
    assert_eq!(receipt["systemDependencyClosureChecked"], false);
    assert_eq!(receipt["cppRuntimeAbsenceQualified"], false);
    assert_eq!(receipt["nativeRetirementQualified"], false);
    Ok(())
}

#[test]
fn native_link_schema_four_follows_original_name_and_ordinal_forwarders() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = runtime_fixture(&scratch)?;
    dependency(
        &scratch,
        &mut capture,
        "kernel32.dll",
        &export_image(
            17,
            &[("_CxxThrowException", 0), ("ordinary_function", 1)],
            &[Some("helper.#29"), None],
        )?,
    )?;
    dependency(
        &scratch,
        &mut capture,
        "helper.dll",
        &export_image(
            29,
            &[("export_alias", 0)],
            &[Some("terminal.__CxxFrameHandler4")],
        )?,
    )?;
    dependency(
        &scratch,
        &mut capture,
        "terminal.dll",
        &export_image(
            41,
            &[("__CxxFrameHandler4", 0), ("ordinary_alias", 0)],
            &[None],
        )?,
    )?;
    let receipt = verify(&capture, &mut Intake::default())?;
    let route = &receipt["runtimeImportBoundary"]["routes"][2];
    assert_eq!(
        route["knownCppEhNames"],
        json!(["_CxxThrowException", "__CxxFrameHandler4"])
    );
    let stages = route["stages"].as_array().ok_or("Missing stages")?;
    assert_eq!(stages.len(), 3);
    assert_eq!(stages[0]["forwarder"], "helper.#29");
    assert_eq!(stages[1]["requestedSymbol"], json!({"ordinal":29}));
    assert_eq!(stages[1]["forwarder"], "terminal.__CxxFrameHandler4");
    assert_eq!(
        stages[2]["exportNames"],
        json!(["__CxxFrameHandler4", "ordinary_alias"])
    );
    assert_eq!(stages[2]["forwarder"], Value::Null);
    assert_eq!(stages[2]["targetRva"], 0x1e00);
    Ok(())
}

#[test]
fn native_link_schema_four_rejects_missing_cyclic_and_unused_provider_routes() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = runtime_fixture(&scratch)?;
    dependency(
        &scratch,
        &mut capture,
        "kernel32.dll",
        &export_image(
            17,
            &[("_CxxThrowException", 0), ("ordinary_function", 1)],
            &[Some("helper.#29"), None],
        )?,
    )?;
    rejected(
        verify(&capture, &mut Intake::default()),
        "Native link runtime route lacks a retained provider image: helper.dll",
    );
    dependency(
        &scratch,
        &mut capture,
        "helper.dll",
        &export_image(28, &[], &[None])?,
    )?;
    rejected(
        verify(&capture, &mut Intake::default()),
        "Native link runtime route names an absent ordinal: helper.dll!#29",
    );
    dependency(
        &scratch,
        &mut capture,
        "helper.dll",
        &export_image(29, &[], &[Some("kernel32.#17")])?,
    )?;
    rejected(
        verify(&capture, &mut Intake::default()),
        "Native link runtime route cycles or exceeds its depth bound.",
    );
    dependency(
        &scratch,
        &mut capture,
        "helper.dll",
        &export_image(29, &[], &[Some("api-ms-example.#17")])?,
    )?;
    rejected(
        verify(&capture, &mut Intake::default()),
        "Native link runtime route requires separately qualified API-set routing.",
    );
    dependency(
        &scratch,
        &mut capture,
        "helper.dll",
        &export_image(29, &[], &[None])?,
    )?;
    dependency(
        &scratch,
        &mut capture,
        "unused.dll",
        &export_image(1, &[], &[None])?,
    )?;
    rejected(
        verify(&capture, &mut Intake::default()),
        "Native link dependency image selection contains an unused provider.",
    );
    Ok(())
}

#[test]
fn native_link_schema_four_rejects_malformed_exports_forwarders_and_changed_bytes() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = runtime_fixture(&scratch)?;
    let mut bytes = export_image(
        17,
        &[("_CxxThrowException", 0), ("ordinary_function", 1)],
        &[None, None],
    )?;
    put16(&mut bytes, 0xac0, 2);
    dependency(&scratch, &mut capture, "kernel32.dll", &bytes)?;
    rejected(
        verify(&capture, &mut Intake::default()),
        "PE named export ordinal index exceeds its table",
    );
    dependency(
        &scratch,
        &mut capture,
        "kernel32.dll",
        &export_image(
            17,
            &[("_CxxThrowException", 0), ("ordinary_function", 1)],
            &[Some("helper.#65536"), None],
        )?,
    )?;
    rejected(
        verify(&capture, &mut Intake::default()),
        "PE forwarder ordinal exceeds its supported range",
    );
    let path = &capture.dependency_images[0].identity.path;
    let mut changed = fs::read(path)?;
    changed[0x1000] = 1;
    fs::write(path, changed)?;
    assert!(verify(&capture, &mut Intake::default()).is_err());
    Ok(())
}

fn clear_imports(bytes: &mut [u8]) {
    for offset in [0x98 + 120, 0x98 + 112 + 13 * 8] {
        put32(bytes, offset, 0);
        put32(bytes, offset + 4, 0);
    }
    bytes[0x400..0xa00].fill(0);
}

// One original ordinal import, independent from the export fixture at 0xa00.
fn ordinal_dependency(bytes: &mut [u8], module: &str, ordinal: u16, delayed: bool) -> TestResult {
    clear_imports(bytes);
    let (descriptor, name, lookup, address) = if delayed {
        put32(bytes, 0x98 + 112 + 13 * 8, 0x1200);
        put32(bytes, 0x98 + 116 + 13 * 8, 64);
        put32(bytes, 0x600, 1);
        (0x600, 0x700, 0x900, 0x940)
    } else {
        put32(bytes, 0x98 + 120, 0x1000);
        put32(bytes, 0x98 + 124, 40);
        (0x400, 0x500, 0x800, 0x840)
    };
    let rva = |offset: usize| u32::try_from(offset + 0xc00);
    if delayed {
        put32(bytes, descriptor + 4, rva(name)?);
        put32(bytes, descriptor + 12, rva(address)?);
        put32(bytes, descriptor + 16, rva(lookup)?);
    } else {
        put32(bytes, descriptor, rva(lookup)?);
        put32(bytes, descriptor + 12, rva(name)?);
        put32(bytes, descriptor + 16, rva(address)?);
    }
    bytes[name..name + module.len()].copy_from_slice(module.as_bytes());
    for offset in [lookup, address] {
        bytes[offset..offset + 8]
            .copy_from_slice(&((1_u64 << 63) | u64::from(ordinal)).to_le_bytes());
    }
    Ok(())
}

fn recursive_fixture(
    scratch: &Scratch,
) -> std::result::Result<Capture, Box<dyn std::error::Error>> {
    let mut capture = runtime_fixture(scratch)?;
    capture.schema_version = 5;
    for item in &mut capture.dependency_images {
        let mut bytes = fs::read(&item.identity.path)?;
        clear_imports(&mut bytes);
        item.identity = scratch.file(&item.module, &bytes)?;
    }
    Ok(capture)
}

#[test]
fn native_link_schema_five_traverses_dependency_imports_and_forwarded_providers() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = recursive_fixture(&scratch)?;
    let mut kernel = fs::read(&capture.dependency_images[0].identity.path)?;
    ordinal_dependency(&mut kernel, "middle.dll", 5, false)?;
    dependency(&scratch, &mut capture, "kernel32.dll", &kernel)?;
    let mut middle = export_image(5, &[], &[Some("terminal.__CxxFrameHandler4")])?;
    ordinal_dependency(&mut middle, "leaf.dll", 11, true)?;
    dependency(&scratch, &mut capture, "middle.dll", &middle)?;
    for (module, base, name) in [
        ("terminal.dll", 9, "__CxxFrameHandler4"),
        ("leaf.dll", 11, "__CxxFrameHandler3"),
        ("tail.dll", 13, "__std_exception_copy"),
    ] {
        let mut bytes = export_image(base, &[(name, 0)], &[None])?;
        clear_imports(&mut bytes);
        if module == "terminal.dll" {
            // This provider is reached only through a forwarder; its own
            // imports must extend the traversal rather than terminating it.
            ordinal_dependency(&mut bytes, "tail.dll", 13, false)?;
        }
        dependency(&scratch, &mut capture, module, &bytes)?;
    }
    let receipt = verify(&capture, &mut Intake::default())?;
    let boundary = &receipt["runtimeImportBoundary"];
    assert_eq!(boundary["recursiveDependencyImportsChecked"], true);
    let checked = boundary["dependencyImportImages"]
        .as_array()
        .ok_or("Missing dependency imports")?;
    assert_eq!(checked.len(), 6);
    let routes = boundary["routes"].as_array().ok_or("Missing routes")?;
    assert_eq!(routes.len(), 8);
    let route = routes
        .iter()
        .find(|r| r["dependencyModule"] == "kernel32.dll")
        .ok_or("Missing transitive route")?;
    assert_eq!(route["packageName"], Value::Null);
    assert_eq!(route["importerLocation"], "retained-dependency");
    assert_eq!(route["knownCppEhNames"], json!(["__CxxFrameHandler4"]));
    assert_eq!(route["stages"][1]["module"], "terminal.dll");
    let route = routes
        .iter()
        .find(|r| r["dependencyModule"] == "middle.dll")
        .ok_or("Missing delay route")?;
    assert_eq!(route["delayed"], true);
    assert_eq!(route["knownCppEhNames"], json!(["__CxxFrameHandler3"]));
    let route = routes
        .iter()
        .find(|r| r["dependencyModule"] == "terminal.dll")
        .ok_or("Missing forwarded provider imports")?;
    assert_eq!(route["knownCppEhNames"], json!(["__std_exception_copy"]));
    assert_eq!(routes[0]["importerLocation"], "package");
    for field in [
        "systemDependencyClosureChecked",
        "cppRuntimeAbsenceQualified",
        "nativeRetirementQualified",
    ] {
        assert_eq!(receipt[field], false);
    }
    Ok(())
}

#[test]
fn native_link_schema_five_checks_mutual_imports_once_without_changing_schema_four() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = recursive_fixture(&scratch)?;
    for (module, target, ordinal) in [
        ("kernel32.dll", "bcrypt.dll", 23),
        ("bcrypt.dll", "kernel32.dll", 17),
    ] {
        let mut bytes = fs::read(
            &capture
                .dependency_images
                .iter()
                .find(|d| d.module == module)
                .ok_or("Missing fixture")?
                .identity
                .path,
        )?;
        ordinal_dependency(&mut bytes, target, ordinal, false)?;
        dependency(&scratch, &mut capture, module, &bytes)?;
    }
    let receipt = verify(&capture, &mut Intake::default())?;
    assert_eq!(
        receipt["runtimeImportBoundary"]["dependencyImportImages"]
            .as_array()
            .ok_or("Missing imports")?
            .len(),
        2
    );
    assert_eq!(
        receipt["runtimeImportBoundary"]["routes"]
            .as_array()
            .ok_or("Missing routes")?
            .len(),
        7
    );
    capture.schema_version = 4;
    let historical = verify(&capture, &mut Intake::default())?;
    assert_eq!(
        historical["runtimeImportBoundary"]["recursiveDependencyImportsChecked"],
        false
    );
    assert_eq!(
        historical["runtimeImportBoundary"]["dependencyImportImages"],
        Value::Null
    );
    assert_eq!(
        historical["runtimeImportBoundary"]["routes"]
            .as_array()
            .ok_or("Missing historical routes")?
            .len(),
        5
    );
    Ok(())
}

#[test]
fn native_link_schema_five_rejects_missing_apiset_and_malformed_dependency_imports() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = recursive_fixture(&scratch)?;
    let kernel = fs::read(&capture.dependency_images[0].identity.path)?;
    for (target, expected) in [
        (
            "absent.dll",
            "Native link dependency import lacks a retained provider image: absent.dll",
        ),
        (
            "api-ms-example.dll",
            "Native link runtime route requires separately qualified API-set routing.",
        ),
    ] {
        let mut bytes = kernel.clone();
        ordinal_dependency(&mut bytes, target, 17, false)?;
        dependency(&scratch, &mut capture, "kernel32.dll", &bytes)?;
        rejected(verify(&capture, &mut Intake::default()), expected);
    }
    let mut bytes = kernel;
    ordinal_dependency(&mut bytes, "bcrypt.dll", 23, true)?;
    // Nonzero IAT terminator must reject, even for a delayed dependency import.
    bytes[0x948] = 1;
    dependency(&scratch, &mut capture, "kernel32.dll", &bytes)?;
    assert!(verify(&capture, &mut Intake::default()).is_err());
    capture.schema_version = 4;
    assert!(verify(&capture, &mut Intake::default()).is_ok());
    Ok(())
}

#[test]
fn native_link_schema_five_requires_and_traverses_zero_symbol_descriptors() -> TestResult {
    let scratch = Scratch::new()?;
    let mut capture = recursive_fixture(&scratch)?;
    let mut kernel = fs::read(&capture.dependency_images[0].identity.path)?;
    ordinal_dependency(&mut kernel, "zero.dll", 1, false)?;
    // Empty lookup and IAT still leave an original import descriptor/module.
    kernel[0x800..0x808].fill(0);
    kernel[0x840..0x848].fill(0);
    dependency(&scratch, &mut capture, "kernel32.dll", &kernel)?;
    rejected(
        verify(&capture, &mut Intake::default()),
        "Native link dependency import lacks a retained provider image: zero.dll",
    );
    let mut zero = export_image(1, &[], &[None])?;
    ordinal_dependency(&mut zero, "bcrypt.dll", 23, true)?;
    dependency(&scratch, &mut capture, "zero.dll", &zero)?;
    let receipt = verify(&capture, &mut Intake::default())?;
    let boundary = &receipt["runtimeImportBoundary"];
    assert_eq!(
        boundary["dependencyImportImages"]
            .as_array()
            .ok_or("Missing dependencies")?
            .len(),
        3
    );
    let routes = boundary["routes"].as_array().ok_or("Missing routes")?;
    assert_eq!(routes.len(), 6);
    let route = routes
        .iter()
        .find(|r| r["dependencyModule"] == "zero.dll")
        .ok_or("Empty descriptor bypassed dependency traversal")?;
    assert_eq!(route["delayed"], true);
    assert_eq!(route["knownCppEhNames"], json!(["__CxxFrameHandler3"]));
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
    let cases = [
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
    ];
    let base_image = runtime_image();
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .clamp(1, cases.len());
    let chunk_len = cases.len().div_ceil(workers);
    std::thread::scope(|scope| -> TestResult {
        let handles: Vec<_> = cases
            .chunks(chunk_len)
            .map(|chunk| {
                let base = &base_image;
                scope.spawn(move || -> Result<()> {
                    let scratch = Scratch::new().map_err(|error| error.to_string())?;
                    let mut capture = fixture(&scratch).map_err(|error| error.to_string())?;
                    capture.schema_version = 3;
                    for (offset, mutation, expected) in chunk {
                        let mut bytes = base.clone();
                        bytes[*offset..*offset + mutation.len()].copy_from_slice(mutation);
                        replace_image(&scratch, &mut capture, &bytes)
                            .map_err(|error| error.to_string())?;
                        rejected(verify(&capture, &mut Intake::default()), expected);
                    }
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            handle
                .join()
                .map_err(|_| "native_link worker panicked".to_owned())??;
        }
        Ok(())
    })
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
    fs::remove_file(scratch.path().join("package/extra.txt"))?;
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
    fs::remove_file(scratch.path().join("package/service.exe"))?;
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
