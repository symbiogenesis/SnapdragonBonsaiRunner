//! Immutable intake and inherited owner deadline for an ordered TTS replay.

use crate::{
    Synthesizer,
    observation::ARITHMETIC_IDENTITY,
    replay::{Binding, Provider, Replay, Tolerance, Tolerances},
};
use bonsai_artifacts::{VerifiedFile, open_locked_file};
use bonsai_native_hash::{Hasher, hash_bytes};
#[cfg(test)]
use bonsai_speech_service::replay_harness::arguments;
use bonsai_speech_service::replay_harness::{
    Artifact, ClockSpec, Identity, ReplayRunConfig, Rule, bounded_bytes, hexadecimal, invalid,
    power_snapshot, run_replay, validate_provider_pair, verify_artifact, verify_candidate_image,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{io, path::PathBuf};

trait IdentityExt {
    fn provider(&self) -> io::Result<Provider<'_>>;
}

impl IdentityExt for Identity {
    fn provider(&self) -> io::Result<Provider<'_>> {
        let parsed = self.parse()?;
        Ok(Provider {
            revision: parsed.revision,
            compiler: parsed.compiler,
            target: parsed.target,
            executable: parsed.executable,
            arithmetic: parsed.arithmetic,
        })
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
        let map = |rule: Rule| {
            rule.map(Tolerance::Bits, |absolute, relative| Tolerance::Finite {
                absolute,
                relative,
            })
        };
        Ok(Tolerances {
            prompt: map(self.prompt)?,
            hidden: map(self.hidden)?,
            logits: map(self.logits)?,
            cache: map(self.cache)?,
            speaker: map(self.speaker)?,
            pcm: map(self.pcm)?,
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

fn power() -> Value {
    power_snapshot(
        bonsai_cpu_policy::power::observed_snapshot,
        bonsai_cpu_policy::power::unknown_snapshot,
    )
}

impl Request {
    fn clock_spec(&self) -> ClockSpec<'_> {
        ClockSpec {
            schema_version: self.schema_version,
            owner_started_tick: self.owner_started_tick,
            work_deadline_tick: self.work_deadline_tick,
            clock_frequency: self.clock_frequency,
            fixed_work: &self.fixed_work,
            estimated_seconds: self.estimated_seconds,
        }
    }

    #[cfg(test)]
    fn admit_clock(&self, now: u64, frequency: u64) -> io::Result<()> {
        self.clock_spec().admit(now, frequency)
    }

    fn expired(&self) -> bool {
        self.clock_spec().expired()
    }

    fn verify(&self, artifact: &Artifact) -> io::Result<VerifiedFile> {
        verify_artifact(artifact, &|| self.expired())
    }

    fn validate(&self) -> io::Result<Tolerances> {
        let reference = self.reference.parse()?;
        let candidate = self.candidate.parse()?;
        validate_provider_pair(&reference, &candidate, ARITHMETIC_IDENTITY)?;
        if !(1..=256).contains(&self.workers)
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
    let mut hash = Hasher::new()?;
    for value in [text, voice] {
        hash.update(
            &u64::try_from(value.len())
                .map_err(|_| invalid())?
                .to_le_bytes(),
        )?;
        hash.update(value.as_bytes())?;
    }
    hash.finish()
}

fn model_artifacts(request: &Request) -> io::Result<([Artifact; 3], [u8; 32])> {
    let bytes = match request.quantization {
        Quantization::Q8 => include_bytes!("../../model.lock.json").as_slice(),
        Quantization::Q4 => include_bytes!("../../model-q4.lock.json").as_slice(),
    };
    let pin: Value = serde_json::from_slice(bytes).map_err(io::Error::other)?;
    let artifact = |name: &str, path: &std::path::Path| -> io::Result<Artifact> {
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
        hash_bytes(bytes)?,
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
    request.clock_spec().admit_and_record(report)?;
    let tolerances = request.validate()?;
    report["cases"] = Value::Array(request.cases.iter().enumerate().map(|(index, case)| json!({"index":index,
        "retainedVoice":case.retained_voice,"textSha256":case.text.sha256,"voiceSha256":case.voice.sha256,
        "packetSha256":case.packet.sha256,"status":"unattempted"})).collect());
    let expired = || request.expired();
    let image_guard =
        verify_candidate_image(&request.candidate.executable_sha256, &expired, report)?;
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

/// Executes the source-bound TTS replay consumer entrypoint.
pub fn run() -> io::Result<bool> {
    let config = ReplayRunConfig {
        provider: "bonsai-tts-replay",
        power_scope: "Consumer entry/exit and each candidate case only; aggregate setup, reference-arm and retention observations remain required.",
        cleanup: "Rust-owned tensors and verified-file guards drop on result or unwind; external native owner must enforce process lifetime.",
        sha256: hash_bytes,
        power,
        compare_power: bonsai_cpu_policy::power::compare_endpoints,
    };
    run_replay(&config, compare)
}

#[cfg(test)]
#[path = "consumer_tests.rs"]
mod tests;
