//! Verified file intake and deadline handoff for one complete Rust replay.

use bonsai_artifacts::open_locked_file;
use bonsai_native_hash::hash_bytes;
use bonsai_speech_asr::{
    replay::{Binding, Provider, Replay, Tolerance, Tolerances},
    rust_engine::{ARITHMETIC_IDENTITY, RustRecognizer},
};
#[cfg(test)]
use bonsai_speech_service::replay_harness::arguments;
use bonsai_speech_service::replay_harness::{
    Artifact, ClockSpec, Identity as HarnessIdentity, ReplayRunConfig, Rule, bounded_bytes,
    hexadecimal, invalid, power_snapshot, run_replay, validate_provider_pair, verify_artifact,
    verify_candidate_image,
};
use serde::Deserialize;
use serde_json::Value;
use std::{io, path::PathBuf};

#[derive(Deserialize)]
#[serde(transparent)]
struct Identity(HarnessIdentity);

impl Identity {
    fn provider(&self) -> io::Result<Provider<'_>> {
        let parsed = self.0.parse()?;
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
    frontend: Rule,
    encoder: Rule,
    logits: Rule,
    cache: Rule,
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
            frontend: map(self.frontend)?,
            encoder: map(self.encoder)?,
            logits: map(self.logits)?,
            cache: map(self.cache)?,
        })
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
    request.clock_spec().admit_and_record(report)?;
    let reference = request.reference.0.parse()?;
    let candidate = request.candidate.0.parse()?;
    validate_provider_pair(&reference, &candidate, ARITHMETIC_IDENTITY)?;
    if request.pcm.bytes < 201 * 4
        || request.pcm.bytes > 480_000 * 4
        || request.packet.bytes == 0
        || request.packet.bytes > 16 * 1024 * 1024 * 1024
    {
        return Err(invalid());
    }
    let expired = || request.expired();
    let image_guard =
        verify_candidate_image(&request.candidate.0.executable_sha256, &expired, report)?;
    let pin: Value =
        serde_json::from_str(include_str!("../../model.lock.json")).map_err(io::Error::other)?;
    let model = Artifact {
        path: request.model.clone(),
        bytes: pin["bytes"].as_u64().ok_or_else(invalid)?,
        sha256: pin["sha256"].as_str().ok_or_else(invalid)?.to_owned(),
    };
    report["phase"] = "verify-pinned-model".into();
    let model_guard = verify_artifact(&model, &expired)?;
    report["phase"] = "verify-original-pcm".into();
    let pcm_guard = verify_artifact(&request.pcm, &expired)?;
    report["phase"] = "verify-original-packet".into();
    let packet_guard = verify_artifact(&request.packet, &expired)?;
    let samples = pcm(&bounded_bytes(pcm_guard.path(), 480_000 * 4)?)?;
    let binding = Binding {
        reference: request.reference.provider()?,
        candidate: request.candidate.provider()?,
        model: hexadecimal(&model.sha256)?,
        pcm: hexadecimal(&request.pcm.sha256)?,
    };
    let input = open_locked_file(packet_guard.path()).map_err(io::Error::other)?;
    let replay = Replay::new(input, &binding, request.tolerances.tolerances()?)?;
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
    report["transcriptSha256"] = hex::encode(hash_bytes(transcript.as_bytes())?).into();
    report["completeReplayPassed"] = true.into();
    report["phase"] = "completed".into();
    Ok(())
}

#[expect(
    clippy::redundant_pub_crate,
    reason = "Keep this binary entry adapter private to its parent module."
)]
pub(super) fn run() -> io::Result<bool> {
    let config = ReplayRunConfig {
        provider: "bonsai-asr-replay",
        power_scope: "This consumer entry/exit only; owner aggregate setup, reference arm and retention observations remain required.",
        cleanup: "Rust-owned tensors and verified-file guards drop on every result or unwind; external owner must enforce process lifetime.",
        sha256: &hash_bytes,
        power: &power,
        compare_power: &bonsai_cpu_policy::power::compare_endpoints,
    };
    run_replay(&config, compare)
}

#[cfg(test)]
#[path = "consumer_tests.rs"]
mod tests;
