//! Native Windows ARM64 public queue ABI and explicit wire-branch compiler controls.
use super::*;
fn arm64(path: &Path) -> Result<()> {
    crate::startup_cache::assert_arm64(path)
}
fn version(text: &str, count: std::ops::RangeInclusive<usize>) -> Option<Vec<u32>> {
    let parts = text
        .split('.')
        .map(str::parse)
        .collect::<std::result::Result<Vec<u32>, _>>()
        .ok()?;
    count.contains(&parts.len()).then_some(parts)
}
fn compiler(control: &mut Control, report: &mut Value) -> Result<PathBuf> {
    let programs = std::env::var_os("ProgramFiles(x86)")
        .map(PathBuf::from)
        .ok_or("Program Files unavailable.")?;
    let vswhere = programs.join("Microsoft Visual Studio/Installer/vswhere.exe");
    let installation = control.run(
        "vswhere",
        vec![
            vswhere.into_os_string(),
            "-latest".into(),
            "-products".into(),
            "*".into(),
            "-requires".into(),
            "Microsoft.VisualStudio.Component.VC.Tools.ARM64".into(),
            "-property".into(),
            "installationPath".into(),
        ],
        30,
    )?;
    let installation = PathBuf::from(installation.trim());
    require(
        installation.is_dir(),
        "Native compiler installation absent.",
    )?;
    let tool_version = String::from_utf8(read(
        &installation.join("VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt"),
        1024,
    )?)
    .map_err(|e| e.to_string())?;
    let tool_version = tool_version.trim();
    require(
        version(tool_version, 3..=4).is_some(),
        "Invalid MSVC tools version.",
    )?;
    let tools = installation.join("VC/Tools/MSVC").join(tool_version);
    let binary = tools.join("bin/Hostarm64/arm64");
    let compiler = binary.join("cl.exe");
    let linker = binary.join("link.exe");
    arm64(&compiler)?;
    arm64(&linker)?;
    let sdk = programs.join("Windows Kits/10");
    let sdk_version = sdk_version(&sdk)?;
    let join = |paths: Vec<PathBuf>| {
        paths
            .iter()
            .map(|path| path.to_string_lossy())
            .collect::<Vec<_>>()
            .join(";")
    };
    control.environment.insert(
        "INCLUDE".into(),
        join(vec![
            tools.join("include"),
            sdk.join(format!("Include/{sdk_version}/ucrt")),
            sdk.join(format!("Include/{sdk_version}/shared")),
            sdk.join(format!("Include/{sdk_version}/um")),
        ]),
    );
    control.environment.insert(
        "LIB".into(),
        join(vec![
            tools.join("lib/arm64"),
            sdk.join(format!("Lib/{sdk_version}/ucrt/arm64")),
            sdk.join(format!("Lib/{sdk_version}/um/arm64")),
        ]),
    );
    control.environment.insert(
        "PATH".into(),
        format!(
            "{};{}",
            binary.display(),
            std::env::var("PATH").unwrap_or_default()
        ),
    );
    let selected = [
        compiler.clone(),
        linker,
        binary.join("c1.dll"),
        binary.join("c2.dll"),
    ]
    .iter()
    .map(|path| {
        let mut pin = identity(path)?;
        pin["path"] = json!(path);
        Ok(pin)
    })
    .collect::<Result<Vec<_>>>()?;
    report["compiler"] = json!({"selectedFiles":selected,"toolsVersion":tool_version,"windowsSdkVersion":sdk_version,"target":"aarch64-pc-windows-msvc","scope":"selected native compiler/linker components; not-complete-OS-closure"});
    Ok(compiler)
}
fn inputs(control: &Control, pin: &Value, report: &mut Value) -> Result<PathBuf> {
    let public = json_file(&control.source("public-qaic.lock.json"), 1_048_576)?;
    // Check the complete retained source closure before spending network or
    // compiler work. The portable policy fixtures exercise this same intake.
    report["retainedWireReplay"] = retained::verify(&control.root, &public)?;
    let inputs = control.output.join("inputs");
    let retained = control.review.join("inputs");
    directory(&inputs)?;
    directory(&retained)?;
    let mut entries = Vec::new();
    for input in list(pin, "publicInputs")? {
        entries.push(control.receive(input, &inputs, 1_048_576)?);
        let name = string(input, "name")?;
        copy(&inputs.join(name), &retained.join(format!("{name}.txt")))?;
    }
    let archive = control
        .root
        .join("docs/evidence/hexagon-public-qaic-41171859264");
    let notices = control.review.join("notices");
    directory(&notices)?;
    for entry in fs::read_dir(archive.join("inputs-and-notices")).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.file_type().map_err(|e| e.to_string())?.is_file() {
            copy(&entry.path(), &notices.join(entry.file_name()))?;
        }
    }
    copy(
        &archive.join("NOTICES.txt"),
        &notices.join("generation-NOTICES.txt"),
    )?;
    for name in [
        "windows_queue_abi.c",
        "windows_queue_abi.rs",
        "public_idl_control.rs",
        "public-qaic-wire.expected.json",
    ] {
        copy(&control.source(name), &retained.join(format!("{name}.txt")))?;
    }
    report["inputs"] = json!(entries);
    report["localSources"] = json!(
        fs::read_dir(&retained)
            .map_err(|e| e.to_string())?
            .map(|entry| identity(&entry.map_err(|e| e.to_string())?.path()))
            .collect::<Result<Vec<_>>>()?
    );
    Ok(inputs)
}
fn build(
    control: &mut Control,
    compiler: &Path,
    inputs: &Path,
    name: &str,
    sources: &[PathBuf],
) -> Result<PathBuf> {
    let executable = control.output.join(format!("{name}.exe"));
    let mut args: Vec<OsString> = vec![
        compiler.as_os_str().to_owned(),
        "/nologo".into(),
        "/std:c11".into(),
        "/TC".into(),
        "/Od".into(),
        "/GS".into(),
        "/guard:cf".into(),
        "/W4".into(),
        "/WX".into(),
        format!("/I{}", inputs.display()).into(),
        format!("/Fo{}\\", control.output.display()).into(),
        format!("/Fe{}", executable.display()).into(),
    ];
    args.extend(sources.iter().map(|source| source.as_os_str().to_owned()));
    args.extend([
        "/link".into(),
        "/machine:arm64".into(),
        format!(
            "/MAP:{}",
            control.review.join(format!("{name}.map")).display()
        )
        .into(),
    ]);
    control.run(&format!("{name}-compile"), args, 30)?;
    arm64(&executable)?;
    Ok(executable)
}
fn rust(
    control: &mut Control,
    rustc: &Path,
    compiler: &Path,
    name: &str,
    source: &str,
    tests: bool,
) -> Result<(PathBuf, Value)> {
    let source_identity = identity(&control.source(source))?;
    let executable = control.output.join(format!("{name}-rust.exe"));
    let args = vec![
        rustc.as_os_str().to_owned(),
        "--edition=2024".into(),
        "--target".into(),
        "aarch64-pc-windows-msvc".into(),
        "-D".into(),
        "warnings".into(),
        "-C".into(),
        format!("linker={}", compiler.with_file_name("link.exe").display()).into(),
        control.source(source).into_os_string(),
        "-o".into(),
        executable.clone().into_os_string(),
    ];
    if tests {
        let test_executable = control.output.join(format!("{name}-rust-tests.exe"));
        let mut test_args = args.clone();
        *test_args
            .last_mut()
            .ok_or("Rust output argument missing.")? = test_executable.clone().into_os_string();
        test_args.push("--test".into());
        control.run(&format!("{name}-rust-test-build"), test_args, 30)?;
        arm64(&test_executable)?;
        control.run(
            &format!("{name}-rust-tests"),
            vec![test_executable.into_os_string()],
            5,
        )?;
    }
    control.run(&format!("{name}-rust-build"), args, 30)?;
    arm64(&executable)?;
    let text = control.run(
        &format!("{name}-rust-capture"),
        vec![executable.clone().into_os_string()],
        5,
    )?;
    pinned(&control.source(source), &source_identity)?;
    Ok((executable, wire::parse(&text)?))
}
fn queue(
    control: &mut Control,
    compiler: &Path,
    inputs: &Path,
    rustc: &Path,
    report: &mut Value,
) -> Result<()> {
    let executable = build(
        control,
        compiler,
        inputs,
        "queue-abi",
        &[control.source("windows_queue_abi.c")],
    )?;
    let text = control.run(
        "queue-abi-capture",
        vec![executable.clone().into_os_string()],
        5,
    )?;
    let observed = wire::parse(&text)?;
    let (rust_executable, rust) = rust(
        control,
        rustc,
        compiler,
        "queue-abi",
        "windows_queue_abi.rs",
        false,
    )?;
    queue_observations(&observed, &rust)?;
    report["queueAbi"] = json!({"passed":true,"nativeWindowsConditionals":true,"completeAssertedLayoutsObserved":true,"observed":observed,"rustObserved":rust,"executable":identity(&executable)?,"rustExecutable":identity(&rust_executable)?});
    report["queueAbi"]["negativeControlsRejected"] = negatives(control, compiler, inputs)?;
    Ok(())
}

fn queue_expected() -> Value {
    json!({"schemaVersion":2,"kind":"windows-arm64-public-queue-abi-control",
        "queueBuffer":{"bytes":24,"alignment":8,"fdOffset":0,"sizeOffset":4,"offsetOffset":8,"flagsOffset":12,"addressOffset":16},
        "htp":{"tensorBytes":56,"bufferBytes":24,"operationBytes":232,"requestBytes":24,"responseBytes":88,"profileBytes":48},
        "layouts":{
            "result":{"bytes":4,"alignment":4},
            "queue":{"bytes":8,"alignment":8},
            "queueBuffer":{"bytes":24,"alignment":8,"fd":0,"size":4,"offset":8,"flags":12,"ptr":16,"address":16},
            "tensor":{"bytes":56,"alignment":8,"data":0,"size":8,"flags":12,"type":16,"bi":20,"ti":22,"ne":24,"nb":40},
            "buffer":{"bytes":24,"alignment":8,"base":0,"size":8,"flags":16,"fd":20},
            "operation":{"bytes":232,"alignment":4,"opcode":0,"flags":4,"params":8,"kernel_params":72,"src":200,"dst":220,"pad":228},
            "request":{"bytes":24,"alignment":8,"seq":0,"n_bufs":8,"n_tensors":12,"n_ops":16,"n_traces":20},
            "response":{"bytes":88,"alignment":8,"seq":0,"cycles_start":8,"cycles_stop":16,"status":24,"n_bufs":28,"n_tensors":32,"n_ops":36,"usecs":40,"n_traces":44},
            "profile":{"bytes":48,"alignment":4,"opcode":0,"usecs":4,"cycles_start":8,"cycles_stop":12,"pmu":16}},
        "constants":{"packetMessage":1,"packetBuffers":2,"reference":4,"dereference":8,"flushSender":16,"invalidateSender":32,"flushRecipient":64,"invalidateRecipient":128,"multiplyMatrix":4,"float32":0,"q2_0":42,"ok":1},
        "queueWrites":{"timed":{"arguments":7,"timeoutMicroseconds":100_000,"calls":1},"nonblocking":{"arguments":6,"calls":1}},
        "infiniteTimeout":4_294_967_295_u64,"mockCalls":6,"unexpectedCalls":0,"returnCode":0,
        "bufferHex":"98badcfe403020100403020194000000efcdab8967452381",
        "driverLoaded":false,"deviceExecuted":false,"profileEligible":false})
}

fn queue_observations(observed: &Value, rust: &Value) -> Result<()> {
    require(
        *observed == queue_expected(),
        "Public native queue/packet ABI differs from independent expectation.",
    )?;
    let shared = [
        "schemaVersion",
        "queueBuffer",
        "htp",
        "layouts",
        "constants",
        "infiniteTimeout",
        "bufferHex",
        "driverLoaded",
        "deviceExecuted",
        "profileEligible",
    ];
    require(
        rust.as_object()
            .is_some_and(|object| object.len() == shared.len() + 1)
            && rust["kind"] == "windows-arm64-Rust-queue-abi-control",
        "Native Rust queue receipt shape differs.",
    )?;
    fields_equal(rust, observed, &shared)
}
fn start_stop(
    control: &mut Control,
    compiler: &Path,
    rustc: &Path,
    report: &mut Value,
) -> Result<()> {
    let public = json_file(&control.source("public-qaic.lock.json"), 1_048_576)?;
    let source = pinned(
        &control.source("public_idl_control.rs"),
        &public["rustWireControl"]["source"],
    )?;
    let observed = &report["retainedWireReplay"]["observed"];
    let (rust_executable, rust) = rust(
        control,
        rustc,
        compiler,
        "start-stop",
        "public_idl_control.rs",
        true,
    )?;
    pinned(&control.source("public_idl_control.rs"), &source)?;
    wire::rust_receipt(&rust, observed)?;
    report["startStopWire"] = json!({"passed":true,"mode":"retained-independent-C-observation-and-fresh-Rust-encoding","originalProviderTarget":"x86_64-linux-gnu","freshRustTarget":"aarch64-pc-windows-msvc","freshCExecution":false,"wholeWindowsHeaderClosureQualified":false,"sdkEquivalenceQualified":false,"observed":observed,"rustObserved":rust,"rustExecutable":identity(&rust_executable)?});
    Ok(())
}
pub(super) fn run(request: &Request) -> Result<Value> {
    let mut control = Control::new(request, 600)?;
    let mut report = json!({"schemaVersion":1,"kind":"hosted-windows-arm64-public-queue-wire-control","passed":false,"sourceCommit":std::env::var("GITHUB_SHA").unwrap_or_default(),"runId":std::env::var("GITHUB_RUN_ID").unwrap_or_default(),"runAttempt":std::env::var("GITHUB_RUN_ATTEMPT").unwrap_or_default(),"controller":identity(&request.repository_directory.join("tools/bonsai/src/hexagon_tools/public_control/windows.rs"))?,"driverLoaded":false,"sdkUsed":false,"deviceExecuted":false,"profileEligible":false,"scope":"native public queue/packet ABI plus retained Linux C start/stop observations and fresh Rust encoding","wholeWindowsHeaderClosureQualified":false,"sdkEquivalenceQualified":false});
    control.provenance(&mut report)?;
    let result: Result<()> = (|| {
        let path = control.source("windows-queue-abi.lock.json");
        let pin = json_file(&path, 1_048_576)?;
        report["pin"] = identity(&path)?;
        let inputs = inputs(&control, &pin, &mut report)?;
        let compiler = compiler(&mut control, &mut report)?;
        arm64(&request.rustc)?;
        report["rustCompiler"] = identity(&request.rustc)?;
        let banner = control.run(
            "rustc-identity",
            vec![request.rustc.clone().into_os_string(), "-vV".into()],
            30,
        )?;
        require(
            banner
                .lines()
                .any(|line| line.trim_end() == "host: aarch64-pc-windows-msvc"),
            "Independent Rust compiler must be native Windows ARM64.",
        )?;
        queue(
            &mut control,
            &compiler,
            &inputs,
            &request.rustc,
            &mut report,
        )?;
        start_stop(&mut control, &compiler, &request.rustc, &mut report)?;
        for component in list(&report["compiler"], "selectedFiles")? {
            pinned(Path::new(string(component, "path")?), component)?;
        }
        pinned(&request.rustc, &report["rustCompiler"])?;
        report["compilerIdentitiesRechecked"] = json!(true);
        report["passed"] = json!(true);
        Ok(())
    })();
    report["commands"] = json!(control.commands);
    if let Err(error) = &result {
        report["failure"] = json!(error.chars().take(512).collect::<String>());
    }
    save(&control.review.join("provenance.json"), &report)?;
    result.map(|()| json!({"passed":true,"receipt":control.review.join("provenance.json")}))
}

fn sdk_version(sdk: &Path) -> Result<String> {
    let mut versions = Vec::new();
    for entry in fs::read_dir(sdk.join("Include")).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(parsed) = version(&name, 4..=4)
            && sdk
                .join(format!("Lib/{name}/ucrt/arm64/ucrt.lib"))
                .is_file()
            && sdk
                .join(format!("Lib/{name}/um/arm64/kernel32.lib"))
                .is_file()
        {
            versions.push((parsed, name));
        }
    }
    versions.sort();
    let sdk_version = &versions
        .last()
        .ok_or("Native ARM64 Windows SDK libraries required.")?
        .1;
    Ok(sdk_version.clone())
}

fn negatives(control: &mut Control, compiler: &Path, inputs: &Path) -> Result<Value> {
    let controls = [
        (
            "narrow-queue-pointer",
            "#include \"dspqueue.h\"\ntypedef int (*bad_close)(uint32_t);\nbad_close narrowed = dspqueue_close;\n",
        ),
        (
            "timed-write-without-timeout",
            "#include \"dspqueue.h\"\ntypedef AEEResult (*bad_write)(dspqueue_t, uint32_t, uint32_t, struct dspqueue_buffer *, uint32_t, const uint8_t *);\nbad_write missing_timeout = dspqueue_write;\n",
        ),
        (
            "noblock-write-with-timeout",
            "#include \"dspqueue.h\"\ntypedef AEEResult (*bad_noblock)(dspqueue_t, uint32_t, uint32_t, struct dspqueue_buffer *, uint32_t, const uint8_t *, uint32_t);\nbad_noblock extra_timeout = dspqueue_write_noblock;\n",
        ),
        (
            "wrong-buffer-layout",
            "#include \"dspqueue.h\"\n_Static_assert(sizeof(struct dspqueue_buffer) == 20, \"deliberate wrong buffer ABI\");\n",
        ),
        (
            "wrong-operation-layout",
            "#include <stdint.h>\n#include \"htp-ops.h\"\n_Static_assert(sizeof(struct htp_op_desc) == 224, \"deliberate wrong operation ABI\");\n",
        ),
    ];
    for (name, source) in controls {
        let path = control.output.join(format!("{name}.c"));
        fs::write(&path, source).map_err(|e| e.to_string())?;
        copy(&path, &control.review.join(format!("{name}.c.txt")))?;
        control.command(
            name,
            vec![
                compiler.as_os_str().to_owned(),
                "/nologo".into(),
                "/std:c11".into(),
                "/TC".into(),
                "/W4".into(),
                "/WX".into(),
                "/c".into(),
                format!("/I{}", inputs.display()).into(),
                format!(
                    "/Fo{}",
                    control.output.join(format!("{name}.obj")).display()
                )
                .into(),
                path.into_os_string(),
            ],
            &control.output.clone(),
            30,
            true,
        )?;
        let mut diagnostics = String::new();
        for suffix in ["log", "stderr.log"] {
            let path = control.review.join(format!("{name}.{suffix}"));
            if path.is_file() {
                diagnostics.push_str(&String::from_utf8_lossy(&read(&path, 4 * 1024 * 1024)?));
            }
        }
        require(
            if name.contains("layout") {
                diagnostics.contains(if name == "wrong-buffer-layout" {
                    "deliberate wrong buffer ABI"
                } else {
                    "deliberate wrong operation ABI"
                })
            } else {
                diagnostics
                    .split(|ch: char| !ch.is_ascii_alphanumeric())
                    .any(|word| ["C4113", "C4133", "C4047"].contains(&word))
            },
            "Negative ABI control failed for an unrelated reason.",
        )?;
    }
    Ok(json!(controls.map(|control| control.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rust_queue_observation() -> Result<Value> {
        let mut rust = queue_expected();
        let fields = rust.as_object_mut().ok_or("Literal queue object absent.")?;
        for field in ["queueWrites", "mockCalls", "unexpectedCalls", "returnCode"] {
            fields.remove(field);
        }
        rust["kind"] = json!("windows-arm64-Rust-queue-abi-control");
        Ok(rust)
    }

    #[test]
    fn windows_queue_complete_layout_observation_rejects_mutation_and_omission() -> Result<()> {
        let native = queue_expected();
        let rust = rust_queue_observation()?;
        queue_observations(&native, &rust)?;
        // Every retained assertion remains mandatory, including offsets that
        // the old receipt omitted and the union's pointer/address alias.
        for (layout, fields) in native["layouts"]
            .as_object()
            .ok_or("Literal layouts absent.")?
        {
            for field in fields.as_object().ok_or("Literal layout absent.")?.keys() {
                let mut missing = rust.clone();
                missing["layouts"][layout]
                    .as_object_mut()
                    .ok_or("Literal layout absent.")?
                    .remove(field);
                assert!(queue_observations(&native, &missing).is_err());
                let mut changed = native.clone();
                changed["layouts"][layout][field] = json!(999);
                let mut matching_changed_rust = rust.clone();
                matching_changed_rust["layouts"][layout][field] = json!(999);
                assert!(queue_observations(&changed, &matching_changed_rust).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn windows_queue_complete_constants_and_mock_calls_are_mandatory() -> Result<()> {
        let native = queue_expected();
        let rust = rust_queue_observation()?;
        for name in native["constants"]
            .as_object()
            .ok_or("Literal constants absent.")?
            .keys()
        {
            let mut changed = rust.clone();
            changed["constants"][name] = json!(999);
            assert!(queue_observations(&native, &changed).is_err());
        }
        let mut changed = native.clone();
        changed["queueWrites"]["timed"]["calls"] = json!(0);
        assert!(queue_observations(&changed, &rust).is_err());
        changed = native.clone();
        changed["queueWrites"]["nonblocking"]["calls"] = json!(0);
        assert!(queue_observations(&changed, &rust).is_err());
        changed = native;
        changed["schemaVersion"] = json!(1);
        assert!(queue_observations(&changed, &rust).is_err());
        Ok(())
    }

    #[test]
    fn reviewed_sources_require_complete_exact_archive() -> Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("Fixture workspace root is unavailable.")?;
        let public = json_file(&root.join("tools/hexagon/public-qaic.lock.json"), 1_048_576)?;
        let replay = retained::verify(root, &public)?;
        assert_eq!(replay["originalSourceInputsVerified"], true);
        assert!(retained::verify(&root.join("Cargo.toml"), &public).is_err());
        let mut changed = public;
        changed["wireCapture"]["headers"][0]["sha256"] = json!("0".repeat(64));
        assert!(retained::verify(root, &changed).is_err());
        Ok(())
    }
}
