//! Immutable intake and inherited owner deadline for an ordered TTS replay.

use bonsai_artifacts::{ArtifactError, VerifiedFile, open_locked_file};
use bonsai_speech_tts::{
    Synthesizer,
    observation::ARITHMETIC_IDENTITY,
    replay::{Binding, Provider, Replay, Tolerance, Tolerances},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{
    ffi::{OsStr, OsString},
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
        if [&self.compiler, &self.target, &self.arithmetic]
            .iter()
            .any(|text| {
                text.trim().is_empty() || text.len() > 4096 || text.chars().any(char::is_control)
            })
        {
            return Err(invalid());
        }
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
    fn tolerance(&self) -> io::Result<Tolerance> {
        match *self {
            Self::Bits => Ok(Tolerance::Bits),
            Self::Finite { absolute, relative }
                if absolute.is_finite()
                    && relative.is_finite()
                    && absolute >= 0.0
                    && relative >= 0.0 =>
            {
                Ok(Tolerance::Finite { absolute, relative })
            }
            Self::Finite { .. } => Err(invalid()),
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Rules {
    prompt: Rule,
    hidden: Rule,
    logits: Rule,
    cache: Rule,
    speaker: Rule,
    pcm: Rule,
}

impl Rules {
    fn tolerances(&self) -> io::Result<Tolerances> {
        Ok(Tolerances {
            prompt: self.prompt.tolerance()?,
            hidden: self.hidden.tolerance()?,
            logits: self.logits.tolerance()?,
            cache: self.cache.tolerance()?,
            speaker: self.speaker.tolerance()?,
            pcm: self.pcm.tolerance()?,
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Case {
    text: Artifact,
    voice: Artifact,
    packet: Artifact,
    retained_voice: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Models {
    design: PathBuf,
    base: PathBuf,
    codec: PathBuf,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
enum Quantization {
    Q8,
    Q4,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    reference: Identity,
    candidate: Identity,
    quantization: Quantization,
    models: Models,
    workers: usize,
    cases: Vec<Case>,
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
        "Invalid source-bound TTS replay request.",
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
    if arguments.next().as_deref() != Some(OsStr::new("--request")) {
        return Err(invalid());
    }
    let path = PathBuf::from(arguments.next().ok_or_else(invalid)?);
    if arguments.next().as_deref() != Some(OsStr::new("--sha256")) {
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
        if self.expired() {
            return Err(io::ErrorKind::Interrupted.into());
        }
        VerifiedFile::verify_with(&artifact.path, artifact.bytes, &artifact.sha256, || {
            if self.expired() {
                Err(ArtifactError::Cancelled)
            } else {
                Ok(())
            }
        })
        .map_err(io::Error::other)
    }

    fn validate(&self) -> io::Result<Tolerances> {
        let reference = self.reference.provider()?;
        let candidate = self.candidate.provider()?;
        if candidate.arithmetic != ARITHMETIC_IDENTITY
            || reference.arithmetic == ARITHMETIC_IDENTITY
            || reference.executable == candidate.executable
            || candidate.target != "aarch64-pc-windows-msvc"
            || !(1..=256).contains(&self.workers)
            || !(1..=8).contains(&self.cases.len())
            || self.cases[0].retained_voice
            || self.cases.iter().any(|case| {
                !(1..=8192).contains(&case.text.bytes)
                    || !(1..=4096).contains(&case.voice.bytes)
                    || !(1..=16_777_216_000).contains(&case.packet.bytes)
            })
        {
            return Err(invalid());
        }
        self.tolerances.tolerances()
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

fn input(bytes: Vec<u8>, maximum: usize) -> io::Result<String> {
    let text = String::from_utf8(bytes).map_err(|_| invalid())?;
    if text.trim().is_empty()
        || text.contains('\0')
        || text.chars().take(maximum + 1).count() > maximum
    {
        return Err(invalid());
    }
    Ok(text)
}

fn request_digest(text: &str, voice: &str) -> io::Result<[u8; 32]> {
    let mut hash = Sha256::new();
    for value in [text, voice] {
        hash.update(
            u64::try_from(value.len())
                .map_err(|_| invalid())?
                .to_le_bytes(),
        );
        hash.update(value.as_bytes());
    }
    Ok(hash.finalize().into())
}

fn model_artifacts(request: &Request) -> io::Result<([Artifact; 3], [u8; 32])> {
    let bytes = match request.quantization {
        Quantization::Q8 => include_bytes!("../../model.lock.json").as_slice(),
        Quantization::Q4 => include_bytes!("../../model-q4.lock.json").as_slice(),
    };
    let pin: Value = serde_json::from_slice(bytes).map_err(io::Error::other)?;
    let artifact = |name: &str, path: &Path| -> io::Result<Artifact> {
        Ok(Artifact {
            path: path.to_owned(),
            bytes: pin["files"][name]["bytes"].as_u64().ok_or_else(invalid)?,
            sha256: pin["files"][name]["sha256"]
                .as_str()
                .ok_or_else(invalid)?
                .to_owned(),
        })
    };
    Ok((
        [
            artifact("talker", &request.models.design)?,
            artifact("voice_talker", &request.models.base)?,
            artifact("codec", &request.models.codec)?,
        ],
        Sha256::digest(bytes).into(),
    ))
}

struct PreparedCase {
    text: String,
    voice: String,
    replay: Replay<std::fs::File>,
    _guards: [VerifiedFile; 3],
}

fn prepare_case(
    request: &Request,
    case: &Case,
    models: &[Artifact; 3],
    model_lock: [u8; 32],
    tolerances: Tolerances,
) -> io::Result<PreparedCase> {
    let text_guard = request.verify(&case.text)?;
    let voice_guard = request.verify(&case.voice)?;
    let packet_guard = request.verify(&case.packet)?;
    let text = input(bounded_bytes(text_guard.path(), 8192)?, 2048)?;
    let voice = input(bounded_bytes(voice_guard.path(), 4096)?, 1024)?;
    let binding = Binding {
        reference: request.reference.provider()?,
        candidate: request.candidate.provider()?,
        model_lock,
        models: [
            hexadecimal(&models[0].sha256)?,
            hexadecimal(&models[1].sha256)?,
            hexadecimal(&models[2].sha256)?,
        ],
        request: request_digest(&text, &voice)?,
        workers: request.workers,
        retained_voice: case.retained_voice,
    };
    let packet = open_locked_file(packet_guard.path()).map_err(io::Error::other)?;
    let replay = Replay::new(packet, &binding, tolerances)?;
    if request.expired() {
        return Err(io::ErrorKind::Interrupted.into());
    }
    Ok(PreparedCase {
        text,
        voice,
        replay,
        _guards: [text_guard, voice_guard, packet_guard],
    })
}

fn retained_schedule(voices: impl IntoIterator<Item = (bool, impl AsRef<str>)>) -> io::Result<()> {
    let mut previous = None::<String>;
    for (retained, voice) in voices {
        if retained
            != previous
                .as_deref()
                .is_some_and(|previous| previous == voice.as_ref())
        {
            return Err(invalid());
        }
        previous = Some(voice.as_ref().to_owned());
    }
    Ok(())
}

fn compare(request: &Request, report: &mut Value) -> io::Result<()> {
    bonsai_speech_service::launch::require_native_host()?;
    let (tick, frequency) = bonsai_windows_process::monotonic_ticks()?;
    request.admit_clock(tick, frequency)?;
    let tolerances = request.validate()?;
    report["fixedWork"] = request.fixed_work.clone().into();
    report["estimatedSeconds"] = request.estimated_seconds.into();
    report["ownerStartedTick"] = request.owner_started_tick.into();
    report["workDeadlineTick"] = request.work_deadline_tick.into();
    report["clockFrequency"] = frequency.into();
    report["cases"] = Value::Array(request.cases.iter().enumerate().map(|(index, case)| json!({"index":index,
        "retainedVoice":case.retained_voice,"textSha256":case.text.sha256,"voiceSha256":case.voice.sha256,
        "packetSha256":case.packet.sha256,"status":"unattempted"})).collect());
    report["phase"] = "verify-candidate-image".into();
    let image = std::env::current_exe()?;
    let image_guard = request.verify(&Artifact {
        bytes: image.metadata()?.len(),
        path: image,
        sha256: request.candidate.executable_sha256.clone(),
    })?;
    let (models, model_lock) = model_artifacts(request)?;
    report["phase"] = "verify-original-inputs-and-packet-headers".into();
    // Authenticate every original byte and header before paying model-loading
    // cost; readers and immutable ancestor/file guards remain alive until use.
    let prepared = request
        .cases
        .iter()
        .map(|case| prepare_case(request, case, &models, model_lock, tolerances))
        .collect::<io::Result<Vec<_>>>()?;
    retained_schedule(
        request
            .cases
            .iter()
            .zip(&prepared)
            .map(|(request, case)| (request.retained_voice, &case.voice)),
    )?;
    report["phase"] = "verify-pinned-models".into();
    let model_guards = models
        .iter()
        .map(|model| request.verify(model))
        .collect::<io::Result<Vec<_>>>()?;
    if request.expired() {
        return Err(io::ErrorKind::Interrupted.into());
    }
    report["phase"] = "load-rust-models".into();
    // Blocking loading remains supervised by the external native owner under
    // this same absolute deadline; no compilation or renewed clock is allowed.
    let mut synthesizer = Synthesizer::load_with_threads(
        &models[0].path,
        &models[2].path,
        &models[1].path,
        Some(request.workers),
    )?;
    for (index, case) in prepared.into_iter().enumerate() {
        report["phase"] = "complete-independent-replay".into();
        report["caseIndex"] = index.into();
        let entry = &mut report["cases"][index];
        entry["powerBefore"] = power();
        entry["status"] = "in-progress".into();
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> io::Result<()> {
                case.replay
                    .synthesize(&mut synthesizer, &case.text, &case.voice, &mut || {
                        request.expired()
                    })?;
                if request.expired() {
                    return Err(io::ErrorKind::Interrupted.into());
                }
                Ok(())
            }));
        drop(case._guards);
        entry["powerAfter"] = power();
        entry["powerConditions"] = bonsai_cpu_policy::power::compare_endpoints(
            &entry["powerBefore"],
            &entry["powerAfter"],
        );
        entry["status"] = if matches!(result, Ok(Ok(()))) {
            "complete-replay-passed"
        } else {
            "failed-or-incomplete"
        }
        .into();
        match result {
            Ok(result) => result?,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
    drop(synthesizer);
    drop((image_guard, model_guards));
    if request.expired() {
        return Err(io::ErrorKind::Interrupted.into());
    }
    report["completeReplayPassed"] = true.into();
    report["phase"] = "completed".into();
    Ok(())
}

#[expect(
    clippy::redundant_pub_crate,
    reason = "Keep the executable entry adapter private to its parent module."
)]
pub(super) fn run() -> io::Result<bool> {
    let mut report = json!({"schemaVersion":1,"provider":"bonsai-tts-replay","passed":false,
        "completeReplayPassed":false,"productionQualified":false,"nativeRetirementQualified":false,
        "performanceComparisonEligible":false,"phase":"request-intake","powerBefore":power(),
        "powerScope":"Consumer entry/exit and each candidate case only; aggregate setup, reference-arm and retention observations remain required.",
        "cleanup":"Rust-owned tensors and verified-file guards drop on result or unwind; external native owner must enforce process lifetime."});
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
