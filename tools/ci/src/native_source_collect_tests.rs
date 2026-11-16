//! Literal original-artifact fixtures; no compiler or generator is executed.

use super::*;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::fs;

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

struct Scratch(crate::impact::workspace::ScratchDir);

impl Scratch {
    fn new() -> std::result::Result<Self, Box<dyn std::error::Error>> {
        Ok(Self(crate::impact::workspace::ScratchDir::new(
            "Bonsai-NativeCollect",
        )?))
    }

    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }

    fn fixture(&self) -> std::result::Result<Value, Box<dyn std::error::Error>> {
        // None of these miniature files is executable or a compiler capture.
        for (name, bytes) in [
            ("clang", "compiler identity"),
            ("generate", "generator identity"),
            ("value", "7"),
            ("intermediate", "7\n"),
            ("generated.h", "#define VALUE 7\n"),
            ("input.c", "int x = VALUE;\n"),
            ("input.i", "int x = 7;\n"),
            ("input.obj", "abc"),
        ] {
            fs::write(self.path(name), bytes)?;
        }
        let escape = |path: String| {
            path.strip_prefix("\\\\?\\")
                .unwrap_or(&path)
                .replace('\\', "/")
                .replace('$', "$$")
                .replace(' ', "\\ ")
        };
        fs::write(
            self.path("input.d"),
            format!(
                "{}: {} {}\n",
                escape(self.path("input.obj")),
                escape(self.path("input.c")),
                escape(self.path("generated.h"))
            ),
        )?;
        let arguments = json!([
            self.path("clang"),
            "-c",
            self.path("input.c"),
            "-MD",
            "-MF",
            self.path("input.d"),
            "-o",
            self.path("input.obj")
        ]);
        fs::write(
            self.path("compile_commands.json"),
            serde_json::to_vec(&json!([{
                "directory":self.0.path(), "file":self.path("input.c"), "arguments":arguments,
                "output":self.path("input.obj")
            }]))?,
        )?;
        let intermediate = json!({
            "path":self.path("intermediate"), "origin":"generated", "generation":{
                "generator":self.path("generate"),
                "arguments":[self.path("generate"), self.path("value")],
                "workingDirectory":self.0.path(), "exitCode":0,
                "inputs":[{"path":self.path("value"), "origin":"source"}]
            }
        });
        let generated = json!({
            "path":self.path("generated.h"), "origin":"generated", "generation":{
                "generator":self.path("generate"),
                "arguments":[self.path("generate"), self.path("intermediate")],
                "workingDirectory":self.0.path(), "exitCode":0, "inputs":[intermediate]
            }
        });
        let unit = json!({
            "source":self.path("input.c"), "workingDirectory":self.0.path(),
            "arguments":arguments, "preprocessingArguments":[self.path("clang"),
                "-E", self.path("input.c"), "-o", self.path("input.i")],
            "compilerExitCode":0, "preprocessorExitCode":0,
            "dependencies":self.path("input.d"), "preprocessed":self.path("input.i"),
            "object":self.path("input.obj"), "inputs":[
                {"path":self.path("input.c"), "origin":"source"}, generated
            ]
        });
        Ok(json!({
            "schemaVersion":1, "kind":"selected-clang-native-source-collection",
            "sourceRevision":"a".repeat(40), "target":"aarch64-pc-windows-msvc",
            "configuration":"literal collection control", "compiler":self.path("clang"),
            "compilationDatabase":self.path("compile_commands.json"), "units":[unit]
        }))
    }
}

fn collect(value: Value) -> Result<String> {
    assemble(serde_json::from_value(value).map_err(|error| error.to_string())?)
}

#[test]
fn native_source_collect_original_bytes_and_recursive_provenance() -> TestResult {
    let scratch = Scratch::new()?;
    let manifest = collect(scratch.fixture()?)?;
    let value: Value = serde_json::from_str(&manifest)?;
    assert_eq!(value["schemaVersion"], 3);
    assert_eq!(value["units"][0]["object"]["bytes"], 3);
    assert_eq!(
        value["units"][0]["object"]["sha256"],
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let expected = hex(&Sha256::digest(manifest.as_bytes()));
    let receipt: Value =
        serde_json::from_str(&verify_manifest_bytes(manifest.as_bytes(), &expected)?)?;
    assert_eq!(receipt["generatedInputs"], 2);
    assert_eq!(receipt["selectedBuildInventoryChecked"], true);
    assert_eq!(receipt["freshCompilerExecution"], false);
    assert_eq!(receipt["nativeRetirementQualified"], false);
    // Replaying the retained manifest detects later original-file mutation.
    fs::write(scratch.path("input.obj"), b"abd")?;
    assert!(verify_manifest_bytes(manifest.as_bytes(), &expected).is_err());
    Ok(())
}

#[test]
fn native_source_collect_clang_cl_retains_original_database_arguments() -> TestResult {
    let scratch = Scratch::new()?;
    let mut request = scratch.fixture()?;
    let compiler = scratch.path("clang-cl.exe");
    fs::write(&compiler, b"literal clang-cl identity; not executable")?;
    request["compiler"] = json!(compiler);
    let arguments = json!([
        compiler,
        "--target=aarch64-pc-windows-msvc",
        "/MD",
        "/c",
        format!("/Fo{}", scratch.path("input.obj")),
        "/clang:-MD",
        format!("/clang:-MF{}", scratch.path("input.d")),
        format!("/clang:-MT{}", scratch.path("input.obj")),
        scratch.path("input.c")
    ]);
    request["units"][0]["arguments"] = arguments.clone();
    request["units"][0]["preprocessingArguments"] = json!([
        compiler,
        "--target=aarch64-pc-windows-msvc",
        "/MD",
        "/P",
        format!("/Fi{}", scratch.path("input.i")),
        scratch.path("input.c")
    ]);
    fs::write(
        scratch.path("compile_commands.json"),
        serde_json::to_vec(&json!([{
            "directory":scratch.0.path(), "file":scratch.path("input.c"), "arguments":arguments,
            "output":scratch.path("input.obj")
        }]))?,
    )?;
    let manifest: Value = serde_json::from_str(&collect(request.clone())?)?;
    assert_eq!(manifest["units"][0]["arguments"], arguments);
    // Changing both the manifest command and preprocessing config cannot hide
    // a different original compilation-database command.
    request["units"][0]["arguments"][2] = json!("/MT");
    request["units"][0]["preprocessingArguments"][2] = json!("/MT");
    assert!(collect(request).is_err());
    Ok(())
}

#[test]
fn native_source_collect_rejects_failures_missing_inputs_and_inventory() -> TestResult {
    let scratch = Scratch::new()?;
    let original = scratch.fixture()?;
    for field in ["compilerExitCode", "preprocessorExitCode"] {
        let mut changed = original.clone();
        changed["units"][0][field] = json!(1);
        assert!(collect(changed).is_err());
    }
    let mut changed = original.clone();
    changed["units"][0]["inputs"][1]["generation"]["exitCode"] = json!(1);
    assert!(collect(changed).is_err());
    let mut changed = original.clone();
    changed["units"][0]["inputs"]
        .as_array_mut()
        .ok_or("Missing inputs")?
        .pop();
    assert!(collect(changed).is_err());
    fs::write(scratch.path("compile_commands.json"), b"[]")?;
    assert!(collect(original).is_err());
    Ok(())
}

#[test]
fn native_source_collect_requires_reviewed_request_and_strict_schema() -> TestResult {
    let scratch = Scratch::new()?;
    let original = scratch.fixture()?;
    let bytes = serde_json::to_vec(&original)?;
    let path = scratch.path("request.json");
    fs::write(&path, &bytes)?;
    let mut arguments = vec![
        "native-closure-collect".into(),
        "--request".into(),
        path.into(),
        "--expected-sha256".into(),
        OsString::from("0".repeat(64)),
    ];
    assert!(run(&arguments).is_err());
    arguments[4] = hex(&Sha256::digest(&bytes)).into();
    assert!(run(&arguments).is_ok());
    let mut changed = original.clone();
    changed["unknown"] = json!(true);
    assert!(collect(changed).is_err());
    let mut changed = original;
    changed["units"][0]["inputs"][0]["generation"] = json!({});
    assert!(collect(changed).is_err());
    Ok(())
}
