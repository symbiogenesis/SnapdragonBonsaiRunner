//! Assemble schema-3 captures from retained original build outputs, without execution.

use super::{
    Generation, GenerationInput, Identity, Input, MAX_DOCUMENT, MAX_FILE, MAX_TOTAL, Manifest,
    Origin, Unit, checkpoint, digest_until, hex, lower_hex, path_key, read_bounded,
    verify_manifest_bytes_until,
};
use crate::Result;
use bonsai_native_hash::hash_bytes;
use serde::Deserialize;
use std::{ffi::OsString, path::Path, time::Instant};

#[cfg(test)]
use super::verify_manifest_bytes;

const USAGE: &str =
    "Usage: bonsai-ci native-closure-collect --request PATH --expected-sha256 SHA256";

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CaptureGeneration {
    generator: String,
    arguments: Vec<String>,
    working_directory: String,
    exit_code: i32,
    inputs: Vec<CaptureInput>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CaptureInput {
    path: String,
    origin: Origin,
    generation: Option<CaptureGeneration>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CaptureUnit {
    source: String,
    working_directory: String,
    arguments: Vec<String>,
    preprocessing_arguments: Vec<String>,
    compiler_exit_code: i32,
    preprocessor_exit_code: i32,
    dependencies: String,
    preprocessed: String,
    object: String,
    inputs: Vec<CaptureInput>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    kind: String,
    source_revision: String,
    target: String,
    configuration: String,
    compiler: String,
    compilation_database: String,
    units: Vec<CaptureUnit>,
}

#[derive(Default)]
struct Intake {
    total: u64,
    records: usize,
    deadline: Option<Instant>,
}

impl Intake {
    fn identity(&mut self, path: String, limit: u64) -> Result<Identity> {
        checkpoint(self.deadline)?;
        path_key(&path)?;
        let (bytes, sha256) = digest_until(Path::new(&path), limit, self.deadline)?;
        self.total = self
            .total
            .checked_add(bytes)
            .filter(|total| *total <= MAX_TOTAL)
            .ok_or("Native closure collection exceeds its aggregate byte bound.")?;
        Ok(Identity {
            path,
            bytes,
            sha256,
        })
    }

    fn inputs(&mut self, inputs: Vec<CaptureInput>, depth: usize) -> Result<Vec<Input>> {
        checkpoint(self.deadline)?;
        if depth >= 64 || !(1..=8192).contains(&inputs.len()) {
            return Err("Native closure collection exceeds its input/depth bounds.".into());
        }
        inputs
            .into_iter()
            .map(|input| {
                self.records = self
                    .records
                    .checked_add(1)
                    .filter(|records| *records <= 100_000)
                    .ok_or("Native closure collection exceeds its record bound.")?;
                if (input.origin == Origin::Generated) != input.generation.is_some() {
                    return Err("Only generated collection inputs require provenance.".into());
                }
                let identity = self.identity(input.path, MAX_FILE)?;
                let generation = input
                    .generation
                    .map(|generation| -> Result<Generation> {
                        if generation.exit_code != 0 {
                            return Err("Cannot collect failed native generation.".into());
                        }
                        Ok(Generation {
                            generator: self.identity(generation.generator, MAX_FILE)?,
                            arguments: generation.arguments,
                            working_directory: Some(generation.working_directory),
                            exit_code: Some(generation.exit_code),
                            inputs: self
                                .inputs(generation.inputs, depth + 1)?
                                .into_iter()
                                .map(GenerationInput::Classified)
                                .collect(),
                        })
                    })
                    .transpose()?;
                Ok(Input {
                    identity,
                    origin: input.origin,
                    generation,
                })
            })
            .collect()
    }
}

fn assemble(request: Request) -> Result<String> {
    assemble_until(request, None)
}

fn assemble_until(request: Request, deadline: Option<Instant>) -> Result<String> {
    checkpoint(deadline)?;
    if request.schema_version != 1
        || request.kind != "selected-clang-native-source-collection"
        || request.target != "aarch64-pc-windows-msvc"
        || !lower_hex(&request.source_revision, 40)
        || !(1..=4096).contains(&request.units.len())
    {
        return Err("Unsupported native source collection identity or bounds.".into());
    }
    let mut intake = Intake {
        deadline,
        ..Intake::default()
    };
    let compiler = intake.identity(request.compiler, MAX_FILE)?;
    let compilation_database = Some(intake.identity(request.compilation_database, MAX_DOCUMENT)?);
    let units = request
        .units
        .into_iter()
        .map(|unit| {
            if unit.compiler_exit_code != 0 || unit.preprocessor_exit_code != 0 {
                return Err("Cannot collect failed native compilation/preprocessing.".into());
            }
            Ok(Unit {
                source: unit.source,
                working_directory: unit.working_directory,
                arguments: unit.arguments,
                preprocessing_arguments: unit.preprocessing_arguments,
                compiler_exit_code: unit.compiler_exit_code,
                preprocessor_exit_code: unit.preprocessor_exit_code,
                dependencies: intake.identity(unit.dependencies, MAX_DOCUMENT)?,
                preprocessed: intake.identity(unit.preprocessed, MAX_FILE)?,
                object: intake.identity(unit.object, MAX_FILE)?,
                inputs: intake.inputs(unit.inputs, 0)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let manifest = Manifest {
        schema_version: 3,
        kind: "selected-clang-native-source-closure".into(),
        source_revision: request.source_revision,
        target: request.target,
        configuration: request.configuration,
        compiler,
        compilation_database,
        units,
    };
    let bytes = serde_json::to_vec(&manifest).map_err(|error| error.to_string())?;
    if u64::try_from(bytes.len()).map_err(|error| error.to_string())? > MAX_DOCUMENT {
        return Err("Collected native source manifest exceeds its document bound.".into());
    }
    // Run the same full verifier before publishing any manifest. It rehashes
    // original inputs, rejects changed identities on its verification pass,
    // and checks the exact
    // compiler dependency set, complete database and recursive provenance.
    let manifest_digest = hex(&hash_bytes(&bytes).map_err(|error| error.to_string())?);
    verify_manifest_bytes_until(&bytes, &manifest_digest, deadline)?;
    let result = String::from_utf8(bytes).map_err(|error| error.to_string())?;
    checkpoint(deadline)?;
    Ok(result)
}

pub(super) fn run(arguments: &[OsString]) -> Result<String> {
    let mut cursor = crate::command::ArgCursor::new(arguments, USAGE);
    cursor.skip_command()?;
    let path = cursor.expect_value("--request")?;
    let expected = cursor.expect_hex("--expected-sha256", 64)?;
    cursor.finish()?;
    path_key(path)?;
    let bytes = read_bounded(Path::new(path), MAX_DOCUMENT)?;
    let request_digest = hash_bytes(&bytes).map_err(|error| error.to_string())?;
    if hex(&request_digest) != expected {
        return Err("Native source collection request differs from its reviewed SHA-256.".into());
    }
    let request: Request = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    assemble(request)
}

pub(super) fn capture(arguments: &[OsString]) -> Result<String> {
    execution::run(arguments)
}

#[path = "native_source_capture.rs"]
mod execution;

#[cfg(test)]
#[path = "native_source_collect_tests.rs"]
mod tests;
