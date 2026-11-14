//! Verified file intake and deadline handoff for one complete Rust replay.

use bonsai_artifacts::{ArtifactError, VerifiedFile, open_locked_file};
use bonsai_speech_asr::{
    replay::{Binding, Provider, Replay, Tolerance, Tolerances},
    rust_engine::{ARITHMETIC_IDENTITY, RustRecognizer},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{
    ffi::OsString,
    io::{self, Read as _, Write as _},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Identity {
    revision: String,
    compiler: String,
    target: String,
    executable_sha256: String,
    arithmetic: String,
}

impl Identity {
    fn provider(&self) -> io::Result<Provider<'_>> {
        Ok(Provider {
            revision: hexadecimal(&self.revision)?,
            compiler: &self.compiler,
            target: &self.target,
            executable: hexadecimal(&self.executable_sha256)?,
            arithmetic: &self.arithmetic,
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Artifact {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Rule {
    Bits,
    Finite { absolute: f32, relative: f32 },
}

impl Rule {
    const fn tolerance(&self) -> Tolerance {
        match *self {
            Self::Bits => Tolerance::Bits,
            Self::Finite { absolute, relative } => Tolerance::Finite { absolute, relative },
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Rules {
    frontend: Rule,
    encoder: Rule,
    logits: Rule,
    cache: Rule,
}

impl Rules {
    const fn tolerances(&self) -> Tolerances {
        Tolerances {
            frontend: self.frontend.tolerance(),
            encoder: self.encoder.tolerance(),
            logits: self.logits.tolerance(),
            cache: self.cache.tolerance(),
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    reference: Identity,
    candidate: Identity,
    model: PathBuf,
    pcm: Artifact,
    packet: Artifact,
    tolerances: Rules,
    owner_started_tick: u64,
    work_deadline_tick: u64,
    clock_frequency: u64,
    fixed_work: String,
    estimated_seconds: u64,
}

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid source-bound ASR replay request.",
    )
}

fn hexadecimal<const N: usize>(text: &str) -> io::Result<[u8; N]> {
    if text.len() != N * 2
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid());
    }
    let mut bytes = [0; N];
    for (output, pair) in bytes.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
        let nibble = |byte: u8| {
            if byte.is_ascii_digit() {
                byte - b'0'
            } else {
                byte - b'a' + 10
            }
        };
        *output = nibble(pair[0]) * 16 + nibble(pair[1]);
    }
    if bytes.iter().all(|byte| *byte == 0) {
        return Err(invalid());
    }
    Ok(bytes)
}

fn arguments(arguments: impl IntoIterator<Item = OsString>) -> io::Result<(PathBuf, String)> {
    let mut arguments = arguments.into_iter();
    if arguments.next().as_deref() != Some(std::ffi::OsStr::new("--request")) {
        return Err(invalid());
    }
    let path = PathBuf::from(arguments.next().ok_or_else(invalid)?);
    if arguments.next().as_deref() != Some(std::ffi::OsStr::new("--sha256")) {
        return Err(invalid());
    }
    let digest = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .ok_or_else(invalid)?;
    let _bytes: [u8; 32] = hexadecimal(&digest)?;
    if arguments.next().is_some() || !path.is_absolute() {
        return Err(invalid());
    }
    Ok((path, digest))
}

fn power() -> Value {
    bonsai_windows_process::power_status().map_or_else(
        |_| bonsai_cpu_policy::power::unknown_snapshot(),
        |status| {
            bonsai_cpu_policy::power::observed_snapshot(
                status.ac_online,
                status.battery_saver,
                status.battery_percent,
            )
        },
    )
}

impl Request {
    fn admit_clock(&self, now: u64, frequency: u64) -> io::Result<()> {
        let allowance = self
            .work_deadline_tick
            .checked_sub(self.owner_started_tick)
            .ok_or_else(invalid)?;
        let maximum = frequency.checked_mul(270).ok_or_else(invalid)?;
        let estimate = frequency
            .checked_mul(self.estimated_seconds)
            .ok_or_else(invalid)?;
        let remaining = self
            .work_deadline_tick
            .checked_sub(now)
            .ok_or_else(invalid)?;
        if self.schema_version != 1
            || frequency == 0
            || frequency != self.clock_frequency
            || self.owner_started_tick == 0
            || self.owner_started_tick > now
            || now >= self.work_deadline_tick
            || allowance > maximum
            || estimate > remaining
            || !(1..=240).contains(&self.estimated_seconds)
            || self.fixed_work.trim().is_empty()
            || self.fixed_work.len() > 1024
            || self.fixed_work.chars().any(char::is_control)
        {
            return Err(invalid());
        }
        Ok(())
    }

    fn expired(&self) -> bool {
        bonsai_windows_process::monotonic_ticks().map_or(true, |(tick, frequency)| {
            frequency != self.clock_frequency
                || tick < self.owner_started_tick
                || tick >= self.work_deadline_tick
        })
    }

    fn verify(&self, artifact: &Artifact) -> io::Result<VerifiedFile> {
        VerifiedFile::verify_with(&artifact.path, artifact.bytes, &artifact.sha256, || {
            if self.expired() {
                Err(ArtifactError::Cancelled)
            } else {
                Ok(())
            }
        })
        .map_err(io::Error::other)
    }
}

fn bounded_bytes(path: &Path, maximum: u64) -> io::Result<Vec<u8>> {
    let file = open_locked_file(path).map_err(io::Error::other)?;
    if file.metadata()?.len() > maximum {
        return Err(invalid());
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).map_err(|_| invalid())? > maximum {
        return Err(invalid());
    }
    Ok(bytes)
}

fn pcm(bytes: &[u8]) -> io::Result<Vec<f32>> {
    let (samples, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() || !(201..=480_000).contains(&samples.len()) {
        return Err(invalid());
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(samples.len())
        .map_err(io::Error::other)?;
    for sample in samples {
        let value = f32::from_le_bytes(*sample);
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return Err(invalid());
        }
        values.push(value);
    }
    Ok(values)
}

fn compare(request: &Request, report: &mut Value) -> io::Result<()> {
    bonsai_speech_service::launch::require_native_host()?;
    let (tick, frequency) = bonsai_windows_process::monotonic_ticks()?;
    request.admit_clock(tick, frequency)?;
    report["fixedWork"] = request.fixed_work.clone().into();
    report["estimatedSeconds"] = request.estimated_seconds.into();
    report["ownerStartedTick"] = request.owner_started_tick.into();
    report["workDeadlineTick"] = request.work_deadline_tick.into();
    report["clockFrequency"] = frequency.into();
    let reference = request.reference.provider()?;
    let candidate = request.candidate.provider()?;
    if candidate.arithmetic != ARITHMETIC_IDENTITY
        || reference.arithmetic == ARITHMETIC_IDENTITY
        || reference.executable == candidate.executable
        || candidate.target != "aarch64-pc-windows-msvc"
        || request.pcm.bytes < 201 * 4
        || request.pcm.bytes > 480_000 * 4
        || request.packet.bytes == 0
        || request.packet.bytes > 16 * 1024 * 1024 * 1024
    {
        return Err(invalid());
    }
    report["phase"] = "verify-candidate-image".into();
    let image = std::env::current_exe()?;
    let image_guard = request.verify(&Artifact {
        bytes: image.metadata()?.len(),
        path: image,
        sha256: request.candidate.executable_sha256.clone(),
    })?;
    let pin: Value =
        serde_json::from_str(include_str!("../../model.lock.json")).map_err(io::Error::other)?;
    let model = Artifact {
        path: request.model.clone(),
        bytes: pin["bytes"].as_u64().ok_or_else(invalid)?,
        sha256: pin["sha256"].as_str().ok_or_else(invalid)?.to_owned(),
    };
    report["phase"] = "verify-pinned-model".into();
    let model_guard = request.verify(&model)?;
    report["phase"] = "verify-original-pcm".into();
    let pcm_guard = request.verify(&request.pcm)?;
    report["phase"] = "verify-original-packet".into();
    let packet_guard = request.verify(&request.packet)?;
    let samples = pcm(&bounded_bytes(pcm_guard.path(), 480_000 * 4)?)?;
    let binding = Binding {
        reference,
        candidate,
        model: hexadecimal(&model.sha256)?,
        pcm: hexadecimal(&request.pcm.sha256)?,
    };
    let input = open_locked_file(packet_guard.path()).map_err(io::Error::other)?;
    let replay = Replay::new(input, &binding, request.tolerances.tolerances())?;
    report["phase"] = "load-rust-model".into();
    let mut recognizer =
        RustRecognizer::load_cancellable(model_guard.path(), &mut || request.expired())?;
    report["phase"] = "complete-independent-replay".into();
    let transcript = replay.transcribe(&mut recognizer, &samples, &mut || request.expired())?;
    drop(recognizer);
    drop((image_guard, model_guard, pcm_guard, packet_guard));
    if request.expired() {
        return Err(io::ErrorKind::Interrupted.into());
    }
    report["transcriptBytes"] = transcript.len().into();
    let mut digest = String::with_capacity(64);
    for byte in Sha256::digest(transcript.as_bytes()) {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        digest.push(char::from(HEX[usize::from(byte >> 4)]));
        digest.push(char::from(HEX[usize::from(byte & 15)]));
    }
    report["transcriptSha256"] = digest.into();
    report["completeReplayPassed"] = true.into();
    report["phase"] = "completed".into();
    Ok(())
}

#[expect(
    clippy::redundant_pub_crate,
    reason = "Keep this binary entry adapter private to its parent module."
)]
pub(super) fn run() -> io::Result<bool> {
    let before = power();
    let mut report = json!({"schemaVersion":1,"provider":"bonsai-asr-replay","passed":false,
        "completeReplayPassed":false,"productionQualified":false,"nativeRetirementQualified":false,
        "performanceComparisonEligible":false,"phase":"request-intake","powerBefore":before,
        "powerScope":"This consumer entry/exit only; owner aggregate setup, reference arm and retention observations remain required.",
        "cleanup":"Rust-owned tensors and verified-file guards drop on every result or unwind; external owner must enforce process lifetime."});
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> io::Result<()> {
        let (path, digest) = arguments(std::env::args_os().skip(1))?;
        let bytes = bounded_bytes(&path, 64 * 1024)?;
        if <[u8; 32]>::from(Sha256::digest(&bytes)) != hexadecimal::<32>(&digest)? {
            return Err(invalid());
        }
        report["requestSha256"] = digest.into();
        let request: Request = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        compare(&request, &mut report)
    }));
    let passed = matches!(result, Ok(Ok(())));
    report["failureKind"] = match &result {
        Ok(Ok(())) => Value::Null,
        Ok(Err(error)) => format!("{:?}", error.kind()).into(),
        Err(_) => "unwind".into(),
    };
    report["passed"] = passed.into();
    report["status"] = if passed {
        "complete-replay-passed"
    } else {
        "failed-or-incomplete"
    }
    .into();
    report["powerAfter"] = power();
    report["powerConditions"] =
        bonsai_cpu_policy::power::compare_endpoints(&report["powerBefore"], &report["powerAfter"]);
    serde_json::to_writer(io::stdout().lock(), &report)?;
    writeln!(io::stdout().lock())?;
    Ok(passed)
}

#[cfg(test)]
#[path = "consumer_tests.rs"]
mod tests;
