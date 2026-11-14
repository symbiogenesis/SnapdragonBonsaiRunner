//! Complete hosted-only paired comparison, with fixed public listening fixtures.

use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;

#[cfg(test)]
mod tests;
#[cfg(windows)]
mod windows;

#[cfg(any(windows, test))]
const MODEL: &str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign";
#[cfg(any(windows, test))]
const VOICE: &str = "A calm, clear narrator with a natural speaking pace.";
#[cfg(any(windows, test))]
const MAX_PCM: usize = 5_760_000;

/// Fixed public packets keep listening captures reproducible without taking
/// arbitrary or private text into the hosted comparison.
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum SpeechPacket {
    #[default]
    Control,
    DailyNarrative,
    DailyInstructions,
}

#[cfg(any(windows, test))]
impl SpeechPacket {
    const fn name(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::DailyNarrative => "daily-narrative",
            Self::DailyInstructions => "daily-instructions",
        }
    }

    const fn text(self) -> &'static str {
        match self {
            Self::Control => "The quick brown fox crossed the quiet garden.",
            Self::DailyNarrative => {
                "At six, the station clock rang twice. \"Wait,\" said Lena. \"The last train is early.\" Outside, rain softened the chalk arrows on the platform."
            }
            Self::DailyInstructions => {
                "First, fold the blue card. Next, place it beside the cup. Pause here. Then read the final line quietly: the room is ready."
            }
        }
    }
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Format {
    #[default]
    Pcm,
    Wav,
}

#[cfg(any(windows, test))]
impl Format {
    const fn name(self) -> &'static str {
        match self {
            Self::Pcm => "pcm",
            Self::Wav => "wav",
        }
    }
    const fn media(self) -> &'static str {
        match self {
            Self::Pcm => "audio/pcm",
            Self::Wav => "audio/wav",
        }
    }
    const fn limit(self) -> usize {
        match self {
            Self::Pcm => MAX_PCM,
            Self::Wav => MAX_PCM + 44,
        }
    }
}

const fn repetitions() -> u8 {
    2
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[cfg_attr(
    not(windows),
    expect(
        dead_code,
        reason = "Decode the same evaluation contract on unsupported platforms before rejecting execution."
    )
)]
pub(super) struct Request {
    repository_root: PathBuf,
    data_directory: PathBuf,
    tts_path: Option<PathBuf>,
    evidence_directory: Option<PathBuf>,
    #[serde(default = "repetitions")]
    repetitions: u8,
    #[serde(default)]
    response_format: Format,
    #[serde(default)]
    speech_packet: SpeechPacket,
}

#[cfg(any(windows, test))]
fn synthesis_payload(request: &Request) -> crate::Result<Vec<u8>> {
    serde_json::to_vec(&serde_json::json!({"model": MODEL,
        "input": request.speech_packet.text(), "voice_description": VOICE,
        "language": "Auto", "response_format": request.response_format.name()}))
    .map_err(|error| error.to_string())
}

#[cfg(any(windows, test))]
fn clip_identity(bytes: &[u8]) -> crate::Result<Value> {
    let pcm = bonsai_audio::mono_pcm16_data(bytes, 24_000, 2_880_000)
        .ok_or("Evaluation output was not a complete bounded mono 24 kHz PCM16 WAV.")?;
    Ok(
        serde_json::json!({"ClipSha256": crate::identity::sha256(bytes),
        "ClipBytes": bytes.len(), "SampleRate": 24_000, "Channels": 1,
        "SampleCount": pcm.len() / 2}),
    )
}

pub(super) fn thread_grid(available: usize) -> crate::Result<Vec<usize>> {
    if !(1..=256).contains(&available) {
        return Err("Evaluation CPU count exceeds the supported bound.".into());
    }
    let threads = [
        (available / 2).max(1),
        available.saturating_sub(2).max(1),
        available,
    ];
    let mut unique = Vec::new();
    for count in threads {
        if !unique.contains(&count) {
            unique.push(count);
        }
    }
    Ok(unique)
}

#[cfg(any(windows, test))]
fn wave(format: Format, bytes: &[u8]) -> crate::Result<Vec<u8>> {
    let wave = match format {
        Format::Pcm => {
            if bytes.is_empty() || bytes.len() > MAX_PCM || !bytes.len().is_multiple_of(2) {
                return Err("TTS returned empty, oversized or unaligned PCM.".into());
            }
            let header = bonsai_audio::mono_pcm16_header(24_000, bytes.len() / 2)
                .ok_or("Invalid evaluation PCM extent.")?;
            let mut wave = Vec::with_capacity(bytes.len() + 44);
            wave.extend_from_slice(&header);
            wave.extend_from_slice(bytes);
            wave
        }
        Format::Wav => bytes.to_vec(),
    };
    bonsai_audio::mono_pcm16_data(&wave, 24_000, 2_880_000)
        .ok_or("Evaluation output was not a complete bounded mono 24 kHz PCM16 WAV.")?;
    Ok(wave)
}

#[cfg(any(windows, test))]
fn ready(value: &Value) -> bool {
    value["ready"].as_bool() == Some(true)
        && value["consistent_voice"].as_bool() == Some(true)
        && value["model"].as_str() == Some(MODEL)
}

#[cfg(any(windows, test))]
fn power_state(observation: &Value) -> crate::Result<(bool, bool)> {
    if observation["Status"] != "observed"
        || observation["Source"] != "GetSystemPowerStatus"
        || !(observation["BatteryPercent"].is_null()
            || observation["BatteryPercent"]
                .as_u64()
                .is_some_and(|value| value <= 100))
    {
        return Err("TTS power observations are unavailable or invalid; retain the unqualified raw session.".into());
    }
    let ac = observation["AcConnected"]
        .as_bool()
        .ok_or("TTS comparison requires an observed AC power source for every arm endpoint.")?;
    let saver = observation["BatterySaver"]
        .as_bool()
        .ok_or("TTS comparison requires an observed battery-saver state for every arm endpoint.")?;
    Ok((ac, saver))
}

#[cfg(any(windows, test))]
fn comparisons(
    sessions: &[Value],
    threads: &[usize],
    repetitions: u8,
) -> crate::Result<Vec<Value>> {
    let mut output = Vec::new();
    for count in threads {
        for repetition in 1..=repetitions {
            let pair: Vec<_> = sessions
                .iter()
                .filter(|session| {
                    session["Threads"] == *count && session["Repetition"] == repetition
                })
                .collect();
            if pair.len() != 2
                || pair.iter().any(|session| {
                    session["Status"] != "completed" || session["Memory"]["Status"] != "completed"
                })
            {
                return Err("Only complete paired TTS sessions can enter comparisons.".into());
            }
            let arm = |name: &str| {
                pair.iter()
                    .find(|session| session["Quantization"] == name)
                    .ok_or("Missing distinct paired TTS quantization.")
            };
            let q8 = arm("Q8_0")?;
            let q4 = arm("Q4_K_M")?;
            let baseline = power_state(&q8["PowerBefore"])?;
            for observation in [&q8["PowerAfter"], &q4["PowerBefore"], &q4["PowerAfter"]] {
                if power_state(observation)? != baseline {
                    return Err("TTS arm AC or battery-saver endpoints differ; performance comparison remains unqualified.".into());
                }
            }
            let difference = |field: &str| -> crate::Result<i64> {
                let counter = |session: &Value| {
                    session["Memory"][field]
                        .as_u64()
                        .and_then(|value| i64::try_from(value).ok())
                        .ok_or("Invalid TTS memory counter.")
                };
                counter(q8)?
                    .checked_sub(counter(q4)?)
                    .ok_or_else(|| "TTS counter difference overflowed.".into())
            };
            output.push(serde_json::json!({"Threads": count, "Repetition": repetition,
                "Q8MinusQ4PeakWorkingSetBytes": difference("PeakWorkingSetBytes")?,
                "Q8MinusQ4PeakJobMemoryBytes": difference("PeakJobMemoryBytes")?,
                "PowerEndpointsMatch": true, "PowerStability": "unknown-between-endpoints",
                "PerformanceQualified": false,
                "Definition": "Paired process-lifetime counter differences; positive means lower Q4 usage. Not isolated allocation or quality evidence. Matching power endpoints do not establish uninterrupted stability or qualify performance."}));
        }
    }
    Ok(output)
}

pub(super) fn run(request: &Request) -> crate::Result<Value> {
    #[cfg(windows)]
    {
        windows::run(request)
    }
    #[cfg(not(windows))]
    {
        let _ = request;
        Err("TTS evaluation requires native Windows ARM64 hosted main CI.".into())
    }
}
