//! Pure Rust egui desktop chat with isolated native process and speech boundaries.
// The unit-test harness needs a console for its summary and failure diagnostics.
#![cfg_attr(all(windows, not(test)), windows_subsystem = "windows")]
#![allow(
    clippy::multiple_crate_versions,
    reason = "The pinned native GUI, graphics and WinRT dependencies require incompatible major versions of shared implementation crates."
)]

#[cfg(windows)]
mod attachment_import;
#[cfg(windows)]
#[allow(
    unsafe_code,
    reason = "Audited native file picker, file clipboard and Windows audio conversion boundary."
)]
mod audio_import;
#[cfg(windows)]
mod bounded_file;
#[cfg(windows)]
mod bounded_json;
#[cfg(windows)]
mod canary;
#[cfg(windows)]
mod client;
#[cfg(windows)]
#[allow(
    unsafe_code,
    reason = "Audited WinRT/COM apartment, CoTaskMemFree, and modal IFileDialog boundary."
)]
mod com_dialog;
#[cfg(windows)]
mod context;
#[cfg(windows)]
mod context_cards;
#[cfg(windows)]
mod http;
#[cfg(windows)]
mod image_export;
#[cfg(windows)]
mod isolated_worker;
#[cfg(windows)]
mod markdown_service;
#[cfg(windows)]
mod native_speech;
#[cfg(windows)]
mod recovery;
#[cfg(windows)]
mod scrub;
#[cfg(windows)]
#[allow(
    unsafe_code,
    reason = "Audited event-driven WASAPI playback boundary using owned COM interfaces and staging buffers."
)]
mod speech;
#[cfg(windows)]
mod startup;
#[cfg(windows)]
mod tts;
#[cfg(windows)]
mod window;

#[cfg(windows)]
#[derive(Clone, Copy, Eq, PartialEq)]
enum HostFamily {
    SnapdragonXElite,
    SnapdragonX2EliteExtreme,
    Other,
}

#[cfg(windows)]
impl HostFamily {
    const fn default_native_row_workers(self) -> u8 {
        match self {
            Self::SnapdragonXElite | Self::SnapdragonX2EliteExtreme => 8,
            Self::Other => 4,
        }
    }
}

#[cfg(windows)]
fn host_family_from_processor_name(name: &str) -> HostFamily {
    if name.is_empty() || name.len() > 512 || name.chars().any(char::is_control) {
        return HostFamily::Other;
    }
    let mut words = name
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| {
            !word.is_empty() && !word.eq_ignore_ascii_case("r") && !word.eq_ignore_ascii_case("tm")
        });
    let Some(mut word) = words.next() else {
        return HostFamily::Other;
    };
    if word.eq_ignore_ascii_case("qualcomm") {
        let Some(next) = words.next() else {
            return HostFamily::Other;
        };
        word = next;
    }
    if !word.eq_ignore_ascii_case("snapdragon") {
        return HostFamily::Other;
    }
    let Some(generation) = words.next() else {
        return HostFamily::Other;
    };
    let Some(tier) = words.next() else {
        return HostFamily::Other;
    };
    let extreme = words
        .next()
        .is_some_and(|word| word.eq_ignore_ascii_case("extreme"));
    if !tier.eq_ignore_ascii_case("elite") {
        return HostFamily::Other;
    }
    if (generation.eq_ignore_ascii_case("x") || generation.eq_ignore_ascii_case("x1")) && !extreme {
        HostFamily::SnapdragonXElite
    } else if generation.eq_ignore_ascii_case("x2") && extreme {
        HostFamily::SnapdragonX2EliteExtreme
    } else {
        HostFamily::Other
    }
}

#[cfg(windows)]
fn detect_host_family() -> HostFamily {
    let enough_cores = std::thread::available_parallelism()
        .ok()
        .is_some_and(|cores| cores.get() >= 8);
    if !enough_cores {
        return HostFamily::Other;
    }
    let Ok(host) = bonsai_windows_process::native_build_host() else {
        return HostFamily::Other;
    };
    host_family_from_processor_name(&host.processor_name)
}

#[cfg(windows)]
fn configuration(
    args: impl IntoIterator<Item = std::ffi::OsString>,
) -> Result<client::Config, &'static str> {
    configuration_for_family(args, HostFamily::Other)
}

#[cfg(windows)]
fn next_str<T>(
    args: &mut impl Iterator<Item = std::ffi::OsString>,
    parse: impl FnOnce(&str) -> Option<T>,
    err: &'static str,
) -> Result<T, &'static str> {
    args.next()
        .as_deref()
        .and_then(std::ffi::OsStr::to_str)
        .and_then(parse)
        .ok_or(err)
}

#[cfg(windows)]
fn next_path(
    args: &mut impl Iterator<Item = std::ffi::OsString>,
    err: &'static str,
) -> Result<std::path::PathBuf, &'static str> {
    args.next().map(std::path::PathBuf::from).ok_or(err)
}

#[cfg(windows)]
fn next_port(
    args: &mut impl Iterator<Item = std::ffi::OsString>,
    err: &'static str,
) -> Result<u16, &'static str> {
    next_str(args, |text| text.parse::<u16>().ok(), err)
}

#[cfg(windows)]
fn next_strict_decimal(
    args: &mut impl Iterator<Item = std::ffi::OsString>,
    min: usize,
    max: usize,
    err: &'static str,
) -> Result<usize, &'static str> {
    args.next()
        .and_then(|value| value.into_string().ok())
        .filter(|text| {
            !text.is_empty()
                && !(text.len() > 1 && text.starts_with('0'))
                && text.bytes().all(|byte| byte.is_ascii_digit())
        })
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|count| (min..=max).contains(count))
        .ok_or(err)
}

#[cfg(windows)]
#[expect(
    clippy::too_many_lines,
    reason = "Parse the bounded native launcher contract without accepting arbitrary service arguments."
)]
fn configuration_for_family(
    args: impl IntoIterator<Item = std::ffi::OsString>,
    host_family: HostFamily,
) -> Result<client::Config, &'static str> {
    let mut args = args.into_iter();
    let (mut broker, mut data, mut evidence) = (None, None, None);
    let (mut frontend_port, mut backend_port) = (None, None);
    let (mut canary_port, mut canary_key_file) = (None, None);
    let (mut canary_executable, mut canary_model) = (None, None);
    let (mut tts_port, mut tts_key_file) = (None, None);
    let (mut tts_executable, mut tts_model, mut tts_codec) = (None, None, None);
    let mut tts_voice_model = None;
    let (mut tts_quantization, mut tts_threads) = (None, None);
    let mut require_broker_pipe = false;
    let mut no_abliterate_lora = false;
    let mut daisyrose_lora = None;
    let mut no_daisyrose_lora = false;
    let mut no_mmproj = false;
    let mut use_repacked_ptq1 = None;
    let mut bonsai25_variant = None;
    let mut native_engine = None;
    let mut native_row_workers = None;
    let mut prompt_reuse = None;
    let mut prompt_reuse_checkpoints = None;
    let mut prompt_reuse_checkpoint_bytes = None;
    let mut kv_cache = None;
    let mut cpu_prefill_tokens = None;
    let mut cpu_prefill_gdn = None;
    let mut spec_type = None;
    let mut draft_max = None;
    let mut cpu_family = None;
    let mut placement = None;
    while let Some(option) = args.next() {
        match option.to_str() {
            Some("--require-broker-pipe") if !require_broker_pipe => require_broker_pipe = true,
            Some("--no-abliterate-lora") if !no_abliterate_lora => no_abliterate_lora = true,
            Some("--daisyrose-lora") if daisyrose_lora.is_none() => {
                daisyrose_lora = Some(next_path(
                    &mut args,
                    "A DaisyRose multi-rank LoRA GGUF path is required.",
                )?);
            }
            Some("--no-daisyrose-lora") if !no_daisyrose_lora => no_daisyrose_lora = true,
            Some("--no-mmproj") if !no_mmproj => no_mmproj = true,
            Some("--use-repacked-ptq1") if use_repacked_ptq1.is_none() => {
                use_repacked_ptq1 = Some(true);
            }
            Some("--use-original-pq2") if use_repacked_ptq1.is_none() => {
                use_repacked_ptq1 = Some(false);
            }
            Some("--bonsai-2-5") if bonsai25_variant.is_none() => {
                bonsai25_variant = Some(next_str(
                    &mut args,
                    client::Bonsai25Variant::from_cli,
                    "Bonsai 2.5 variant must be pq2 or ptq1.",
                )?);
            }
            Some("--native-engine") if native_engine.is_none() => native_engine = Some(true),
            Some("--publisher-engine") if native_engine.is_none() => native_engine = Some(false),
            Some("--native-row-workers") if native_row_workers.is_none() => {
                native_row_workers = Some(next_str(
                    &mut args,
                    |text| match text {
                        "1" => Some(1),
                        "4" => Some(4),
                        "8" => Some(8),
                        "12" => Some(12),
                        _ => None,
                    },
                    "Native row workers must be 1, 4, 8 or 12.",
                )?);
            }
            Some("--prompt-reuse") if prompt_reuse.is_none() => {
                prompt_reuse = Some(next_str(
                    &mut args,
                    client::PromptReuse::from_cli,
                    "Prompt reuse must be disabled, latest or prompt-end-and-latest.",
                )?);
            }
            Some("--prompt-reuse-checkpoints") if prompt_reuse_checkpoints.is_none() => {
                let count = next_strict_decimal(
                    &mut args,
                    1,
                    8,
                    "Prompt reuse checkpoints must be between 1 and 8.",
                )?;
                prompt_reuse_checkpoints = u8::try_from(count).ok();
            }
            Some("--prompt-reuse-checkpoint-bytes") if prompt_reuse_checkpoint_bytes.is_none() => {
                prompt_reuse_checkpoint_bytes = Some(next_strict_decimal(
                    &mut args,
                    1,
                    usize::MAX,
                    "Prompt reuse checkpoint bytes must be a positive integer.",
                )?);
            }
            Some("--kv-cache") if kv_cache.is_none() => {
                kv_cache = Some(next_str(
                    &mut args,
                    client::KvCache::from_cli,
                    "KV cache must be f16, rsq4 or rsq4k3v.",
                )?);
            }
            Some("--cpu-prefill-tokens") if cpu_prefill_tokens.is_none() => {
                cpu_prefill_tokens = Some(next_str(
                    &mut args,
                    |text| match text {
                        "8" => Some(8_u8),
                        "16" => Some(16_u8),
                        "32" => Some(32_u8),
                        _ => None,
                    },
                    "CPU prefill tokens must be 8, 16 or 32.",
                )?);
            }
            Some("--cpu-prefill-gdn") if cpu_prefill_gdn.is_none() => {
                cpu_prefill_gdn = Some(next_str(
                    &mut args,
                    client::CpuPrefillGdn::from_cli,
                    "CPU prefill GDN mode must be sequential or wy-f32.",
                )?);
            }
            Some("--spec-type") if spec_type.is_none() => {
                spec_type = Some(next_str(
                    &mut args,
                    client::SpeculationType::from_cli,
                    "Speculation type must be none or suffix.",
                )?);
            }
            Some("--draft-max") if draft_max.is_none() => {
                let count =
                    next_strict_decimal(&mut args, 1, 8, "Draft max must be between 1 and 8.")?;
                draft_max = u8::try_from(count).ok();
            }
            Some("--cpu-family") if cpu_family.is_none() => {
                cpu_family = Some(next_str(
                    &mut args,
                    client::CpuFamily::from_cli,
                    "CPU family must be x1 or x2.",
                )?);
            }
            Some("--placement") if placement.is_none() => {
                placement = Some(next_str(
                    &mut args,
                    client::Placement::from_cli,
                    "Placement must be auto or cpu.",
                )?);
            }
            Some("--broker-path") if broker.is_none() => {
                broker = args.next().map(std::path::PathBuf::from);
            }
            Some("--data-directory") if data.is_none() => {
                data = args.next().map(std::path::PathBuf::from);
            }
            Some("--evidence-directory") if evidence.is_none() => {
                evidence = args.next().map(std::path::PathBuf::from);
            }
            Some("--frontend-port") if frontend_port.is_none() => {
                frontend_port = Some(next_port(&mut args, "Invalid frontend port.")?);
            }
            Some("--backend-port") if backend_port.is_none() => {
                backend_port = Some(next_port(&mut args, "Invalid backend port.")?);
            }
            Some("--canary-port") if canary_port.is_none() => {
                canary_port = Some(next_port(&mut args, "Invalid Qwen3-ASR port.")?);
            }
            Some("--canary-key-file") if canary_key_file.is_none() => {
                canary_key_file = Some(next_path(
                    &mut args,
                    "A Qwen3-ASR key-file path is required.",
                )?);
            }
            Some("--tts-port") if tts_port.is_none() => {
                tts_port = Some(next_port(&mut args, "Invalid Qwen TTS port.")?);
            }
            Some("--tts-key-file") if tts_key_file.is_none() => {
                tts_key_file = Some(next_path(
                    &mut args,
                    "A Qwen TTS key-file path is required.",
                )?);
            }
            Some("--canary-executable") if canary_executable.is_none() => {
                canary_executable = Some(next_path(
                    &mut args,
                    "A native Qwen3-ASR executable path is required.",
                )?);
            }
            Some("--canary-model") if canary_model.is_none() => {
                canary_model = Some(next_path(
                    &mut args,
                    "A Qwen3-ASR GGUF model path is required.",
                )?);
            }
            Some("--tts-executable") if tts_executable.is_none() => {
                tts_executable = Some(next_path(
                    &mut args,
                    "A native Qwen TTS executable path is required.",
                )?);
            }
            Some("--tts-model") if tts_model.is_none() => {
                tts_model = Some(next_path(
                    &mut args,
                    "A Qwen TTS GGUF model path is required.",
                )?);
            }
            Some("--tts-codec") if tts_codec.is_none() => {
                tts_codec = Some(next_path(
                    &mut args,
                    "A Qwen TTS codec GGUF path is required.",
                )?);
            }
            Some("--tts-voice-model") if tts_voice_model.is_none() => {
                tts_voice_model = Some(next_path(
                    &mut args,
                    "A Qwen TTS Base voice GGUF path is required.",
                )?);
            }
            Some("--tts-quantization") if tts_quantization.is_none() => {
                tts_quantization = Some(next_str(
                    &mut args,
                    |text| matches!(text, "Q8_0" | "Q4_K_M").then(|| text.to_owned()),
                    "TTS quantization must be Q8_0 or Q4_K_M.",
                )?);
            }
            Some("--tts-threads") if tts_threads.is_none() => {
                tts_threads = Some(next_str(
                    &mut args,
                    |text| {
                        text.parse::<usize>()
                            .ok()
                            .filter(|count| (1..=256).contains(count))
                    },
                    "TTS threads must be between 1 and 256.",
                )?);
            }
            _ => {
                return Err(
                    "Usage: bonsai-chat --broker-path ABSOLUTE_EXE --data-directory ABSOLUTE_DIRECTORY --evidence-directory ABSOLUTE_DIRECTORY [--frontend-port 5173] [--backend-port 8081] [--no-abliterate-lora] [--daisyrose-lora ABSOLUTE_GGUF] [--no-daisyrose-lora] [--no-mmproj] [--use-repacked-ptq1|--use-original-pq2|--bonsai-2-5 pq2|ptq1] [--native-engine|--publisher-engine] [--native-row-workers 1|4|8|12] [--prompt-reuse disabled|latest] [--kv-cache f16|rsq4|rsq4k3v] [--cpu-prefill-tokens 8|16|32] [--placement auto|cpu] [--canary-key-file ABSOLUTE_FILE --canary-port 8766 [--canary-executable ABSOLUTE_EXE --canary-model ABSOLUTE_GGUF]] [--tts-key-file ABSOLUTE_FILE --tts-port 8767 [--tts-executable ABSOLUTE_EXE --tts-model ABSOLUTE_GGUF --tts-codec ABSOLUTE_GGUF --tts-voice-model ABSOLUTE_GGUF [--tts-quantization Q8_0|Q4_K_M] [--tts-threads 10]]]",
                );
            }
        }
    }
    let native_engine = native_engine.unwrap_or(true);
    let native_row_workers = native_row_workers.unwrap_or(if native_engine {
        if let Some(family) = cpu_family {
            family.default_native_row_workers()
        } else {
            host_family.default_native_row_workers()
        }
    } else {
        1
    });
    let prompt_reuse = prompt_reuse.unwrap_or_default();
    if require_broker_pipe && !native_engine {
        return Err("Broker-pipe transport requires --native-engine.");
    }
    if prompt_reuse.enabled() && !native_engine {
        return Err("Prompt reuse requires --native-engine.");
    }
    if (prompt_reuse_checkpoints.is_some() || prompt_reuse_checkpoint_bytes.is_some())
        && (!native_engine || !prompt_reuse.enabled())
    {
        return Err("Prompt reuse checkpoints require --native-engine and enabled prompt reuse.");
    }
    if kv_cache.is_some() && !native_engine {
        return Err("KV cache selection requires --native-engine.");
    }
    if cpu_prefill_tokens.is_some() && !native_engine {
        return Err("CPU prefill tile selection requires --native-engine.");
    }
    if cpu_prefill_gdn.is_some() && (!native_engine || cpu_prefill_tokens.is_none()) {
        return Err("CPU prefill GDN selection requires --native-engine and --cpu-prefill-tokens.");
    }
    if cpu_prefill_gdn == Some(client::CpuPrefillGdn::WyF32)
        && kv_cache != Some(client::KvCache::F16)
    {
        return Err("CPU prefill GDN wy-f32 requires --kv-cache f16.");
    }
    if (spec_type.is_some() || draft_max.is_some()) && !native_engine {
        return Err("CPU speculative decoding requires --native-engine.");
    }
    if draft_max.is_some() && spec_type != Some(client::SpeculationType::Suffix) {
        return Err("Draft max requires --spec-type suffix.");
    }
    if spec_type == Some(client::SpeculationType::Suffix) && cpu_prefill_tokens.is_none() {
        return Err("Suffix speculation requires --cpu-prefill-tokens.");
    }
    if spec_type == Some(client::SpeculationType::Suffix)
        && cpu_prefill_gdn == Some(client::CpuPrefillGdn::WyF32)
    {
        return Err("CPU prefill GDN wy-f32 cannot be combined with suffix speculation.");
    }
    if cpu_family.is_some() && !native_engine {
        return Err("CPU family selection requires --native-engine.");
    }
    if placement.is_some() && !native_engine {
        return Err("Placement selection requires --native-engine.");
    }
    if no_daisyrose_lora && daisyrose_lora.is_some() {
        return Err("--no-daisyrose-lora cannot be combined with --daisyrose-lora.");
    }
    if daisyrose_lora.is_some() && !native_engine {
        return Err("DaisyRose multi-rank LoRA requires --native-engine.");
    }
    if bonsai25_variant.is_some() && use_repacked_ptq1.is_some() {
        return Err("Do not combine --bonsai-2-5 with --use-repacked-ptq1 or --use-original-pq2.");
    }
    if bonsai25_variant.is_some() && !native_engine {
        return Err("--bonsai-2-5 requires --native-engine.");
    }
    if native_row_workers != 1 && !native_engine {
        return Err("Parallel native row workers require --native-engine.");
    }
    let canary_launch = match (canary_executable, canary_model) {
        (None, None) => None,
        (Some(executable), Some(model)) if canary_key_file.is_some() => {
            Some(native_speech::Launch {
                executable,
                model,
                codec: None,
                voice_model: None,
                quantization: None,
                threads: None,
                evidence: evidence
                    .clone()
                    .ok_or("An external evidence directory is required.")?,
            })
        }
        _ => return Err("Configure both native Qwen3-ASR launch paths with --canary-key-file."),
    };
    let tts_launch = match (tts_executable, tts_model, tts_codec, tts_voice_model) {
        (None, None, None, None) => None,
        (Some(executable), Some(model), Some(codec), Some(voice_model))
            if tts_key_file.is_some() =>
        {
            Some(native_speech::Launch {
                executable,
                model,
                codec: Some(codec),
                voice_model: Some(voice_model),
                quantization: tts_quantization.clone(),
                threads: tts_threads,
                evidence: evidence
                    .clone()
                    .ok_or("An external evidence directory is required.")?,
            })
        }
        _ => return Err("Configure all four native Qwen TTS launch paths with --tts-key-file."),
    };
    if (tts_quantization.is_some() || tts_threads.is_some()) && tts_launch.is_none() {
        return Err("TTS quantization and thread settings require an owned native TTS service.");
    }
    let config = client::Config {
        broker: broker.ok_or("A broker executable is required.")?,
        data: data.ok_or("A provisioned data directory is required.")?,
        evidence: evidence.ok_or("An external evidence directory is required.")?,
        frontend_port: frontend_port.unwrap_or(5173),
        backend_port: backend_port.unwrap_or(8081),
        require_broker_pipe,
        use_abliterate_lora: !no_abliterate_lora,
        daisyrose_lora,
        no_daisyrose_lora,
        no_mmproj,
        use_repacked_ptq1: use_repacked_ptq1.unwrap_or(false),
        bonsai25_variant,
        native_engine,
        native_row_workers,
        prompt_reuse,
        prompt_reuse_checkpoints,
        prompt_reuse_checkpoint_bytes,
        kv_cache,
        cpu_prefill_tokens,
        cpu_prefill_gdn,
        spec_type,
        draft_max,
        cpu_family,
        placement,
        canary: canary_key_file.map(|key_file| canary::Config {
            port: canary_port.unwrap_or(8766),
            key_file,
            launch: canary_launch,
        }),
        qwen_tts: tts_key_file.map(|key_file| tts::Config {
            port: tts_port.unwrap_or(8767),
            key_file,
            launch: tts_launch,
        }),
    };
    validate_configuration(&config, canary_port.is_some(), tts_port.is_some())?;
    Ok(config)
}

#[cfg(windows)]
fn validate_speech_service(
    service: Option<(&std::path::Path, u16, Option<&native_speech::Launch>)>,
    has_port: bool,
    port_required_err: &'static str,
    key_err: &'static str,
    service_err: &'static str,
    used_ports: &[u16],
) -> Result<(), &'static str> {
    let Some((key_file, port, launch)) = service else {
        return if has_port {
            Err(port_required_err)
        } else {
            Ok(())
        };
    };
    bonsai_windows_credentials::validate_external_path(key_file).map_err(|_| key_err)?;
    if !key_file.is_file() || port < 1024 || used_ports.contains(&port) {
        return Err(service_err);
    }
    if let Some(launch) = launch {
        launch.validate()?;
    }
    Ok(())
}

#[cfg(windows)]
fn validate_configuration(
    config: &client::Config,
    has_canary_port: bool,
    has_tts_port: bool,
) -> Result<(), &'static str> {
    for path in [&config.broker, &config.data, &config.evidence] {
        bonsai_windows_credentials::validate_external_path(path)
            .map_err(|_| "Use existing expanded absolute local paths outside Git.")?;
    }
    if !config.broker.is_file() || !config.data.is_dir() || !config.evidence.is_dir() {
        return Err("The broker, model data and evidence directory must already exist.");
    }
    if let Some(daisyrose_path) = &config.daisyrose_lora {
        bonsai_windows_credentials::validate_external_path(daisyrose_path)
            .map_err(|_| "Use an existing expanded absolute DaisyRose LoRA GGUF path.")?;
        if daisyrose_path.to_str().is_none() || !daisyrose_path.is_file() {
            return Err("The DaisyRose multi-rank LoRA GGUF must be an existing Unicode file.");
        }
    }
    if config.frontend_port < 1024
        || config.backend_port < 1024
        || config.frontend_port == config.backend_port
    {
        return Err("Use distinct unprivileged frontend and backend ports.");
    }
    validate_speech_service(
        config
            .canary
            .as_ref()
            .map(|c| (c.key_file.as_path(), c.port, c.launch.as_ref())),
        has_canary_port,
        "Configure --canary-key-file with --canary-port.",
        "Use an expanded absolute external Qwen3-ASR key-file path.",
        "Qwen3-ASR needs an existing key file and a distinct unprivileged local port.",
        &[config.frontend_port, config.backend_port],
    )?;
    validate_speech_service(
        config
            .qwen_tts
            .as_ref()
            .map(|t| (t.key_file.as_path(), t.port, t.launch.as_ref())),
        has_tts_port,
        "Configure --tts-key-file with --tts-port.",
        "Use an expanded absolute external Qwen TTS key-file path.",
        "Qwen TTS needs an existing key file and a distinct unprivileged local port.",
        &[
            config.frontend_port,
            config.backend_port,
            config.canary.as_ref().map_or(0, |c| c.port),
        ],
    )?;
    Ok(())
}

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args
        .first()
        .is_some_and(|argument| argument == "--audio-import-worker")
    {
        return audio_import::run_worker(&args);
    }
    if args
        .first()
        .is_some_and(|argument| argument == "--attachment-import-worker")
    {
        return attachment_import::run_worker(&args);
    }
    let result = if args.len() == 1 && args[0] == "--preview" {
        window::run_preview().map(|()| false)
    } else if args
        .first()
        .is_some_and(|argument| argument == "--startup-tool")
    {
        startup::Request::parse(&args).and_then(window::run_startup)
    } else {
        configuration_for_family(args, detect_host_family())
            .map_err(std::io::Error::other)
            .and_then(window::run)
            .map(|()| false)
    };
    match result {
        Ok(true) => std::process::ExitCode::from(42),
        Ok(false) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            window::show_error(&error.to_string());
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::configuration;
    use std::{ffi::OsString, fs, io};

    #[test]
    fn native_speech_configuration_requires_complete_groups() -> io::Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("Bonsai-native-speech-config-{stamp}"));
        fs::create_dir(&directory)?;
        let broker = directory.join("broker.exe");
        let model = directory.join("model.gguf");
        let codec = directory.join("codec.gguf");
        let key = directory.join("speech.key");
        for path in [&broker, &model, &codec, &key] {
            fs::write(path, b"")?;
        }
        let base = vec![
            OsString::from("--broker-path"),
            broker.clone().into_os_string(),
            OsString::from("--data-directory"),
            directory.clone().into_os_string(),
            OsString::from("--evidence-directory"),
            directory.clone().into_os_string(),
        ];
        for (key_flag, launch_flags) in [
            (
                "--canary-key-file",
                vec![("--canary-executable", &broker), ("--canary-model", &model)],
            ),
            (
                "--tts-key-file",
                vec![
                    ("--tts-executable", &broker),
                    ("--tts-model", &model),
                    ("--tts-codec", &codec),
                    ("--tts-voice-model", &model),
                ],
            ),
        ] {
            let mut enabled = base.clone();
            for (flag, path) in &launch_flags {
                enabled.extend([OsString::from(flag), path.as_os_str().to_owned()]);
            }
            assert!(configuration(enabled.clone()).is_err());
            enabled.extend([OsString::from(key_flag), key.clone().into_os_string()]);
            for (missing, _) in &launch_flags {
                let incomplete = enabled
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .filter(|pair| pair[0] != *missing)
                    .flat_map(|pair| pair.iter().cloned())
                    .collect::<Vec<_>>();
                assert!(configuration(incomplete).is_err());
            }
            let config = configuration(enabled.clone()).map_err(io::Error::other)?;
            if key_flag == "--tts-key-file" {
                assert_tts_tuning(&enabled)?;
            }
            assert!(
                config
                    .canary
                    .as_ref()
                    .is_some_and(|config| config.launch.is_some())
                    || config
                        .qwen_tts
                        .as_ref()
                        .is_some_and(|config| config.launch.is_some())
            );
            let mut partial = base.clone();
            partial.extend([
                OsString::from(key_flag),
                key.clone().into_os_string(),
                OsString::from(launch_flags[0].0),
                broker.clone().into_os_string(),
            ]);
            assert!(configuration(partial).is_err());
            let mut repeated = enabled;
            repeated.extend([
                OsString::from(launch_flags[0].0),
                broker.clone().into_os_string(),
            ]);
            assert!(configuration(repeated).is_err());
        }
        for (flag, value) in [("--tts-threads", "10"), ("--tts-quantization", "Q4_K_M")] {
            let mut unowned = base.clone();
            unowned.extend([OsString::from(flag), OsString::from(value)]);
            assert!(configuration(unowned).is_err());
        }
        for path in [&broker, &model, &codec, &key] {
            fs::remove_file(path)?;
        }
        fs::remove_dir(directory)?;
        Ok(())
    }

    fn assert_tts_tuning(enabled: &[OsString]) -> io::Result<()> {
        let mut tuned = enabled.to_vec();
        tuned.extend([
            OsString::from("--tts-quantization"),
            OsString::from("Q4_K_M"),
            OsString::from("--tts-threads"),
            OsString::from("10"),
        ]);
        let tuned = configuration(tuned).map_err(io::Error::other)?;
        let launch = tuned
            .qwen_tts
            .and_then(|config| config.launch)
            .ok_or_else(|| io::Error::other("Missing owned TTS configuration"))?;
        assert_eq!(launch.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(launch.threads, Some(10));
        for (flag, value) in [
            ("--tts-threads", "0"),
            ("--tts-threads", "257"),
            ("--tts-threads", "-1"),
            ("--tts-quantization", "Q2_K"),
        ] {
            let mut invalid = enabled.to_vec();
            invalid.extend([OsString::from(flag), OsString::from(value)]);
            assert!(configuration(invalid).is_err());
        }
        Ok(())
    }
    #[test]
    fn canary_configuration_requires_explicit_key_and_distinct_port() -> io::Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("Bonsai-canary-config-{stamp}"));
        fs::create_dir(&directory)?;
        let broker = directory.join("broker.exe");
        let key = directory.join("canary.key");
        fs::write(&broker, b"")?;
        fs::write(&key, "a".repeat(64))?;
        let args = vec![
            OsString::from("--broker-path"),
            broker.clone().into_os_string(),
            OsString::from("--data-directory"),
            directory.clone().into_os_string(),
            OsString::from("--evidence-directory"),
            directory.clone().into_os_string(),
        ];
        assert!(
            configuration(args.clone())
                .map_err(io::Error::other)?
                .canary
                .is_none()
        );
        let mut port_only = args.clone();
        port_only.extend([OsString::from("--canary-port"), OsString::from("8766")]);
        assert!(configuration(port_only).is_err());
        let mut enabled = args;
        enabled.extend([
            OsString::from("--canary-key-file"),
            key.clone().into_os_string(),
        ]);
        let config = configuration(enabled.clone()).map_err(io::Error::other)?;
        assert_eq!(config.canary.as_ref().map(|canary| canary.port), Some(8766));
        let mut colliding = enabled.clone();
        colliding.extend([OsString::from("--canary-port"), OsString::from("8081")]);
        assert!(configuration(colliding).is_err());
        let mut duplicated = enabled;
        duplicated.extend([
            OsString::from("--canary-key-file"),
            key.clone().into_os_string(),
        ]);
        assert!(configuration(duplicated).is_err());
        fs::remove_file(key)?;
        fs::remove_file(broker)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn original_pq2_selection_rejects_conflicts_and_preserves_engine_and_adapter() -> io::Result<()>
    {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("Bonsai-pq2-config-{stamp}"));
        fs::create_dir(&directory)?;
        let broker = directory.join("broker.exe");
        fs::write(&broker, b"")?;
        let base = vec![
            OsString::from("--broker-path"),
            broker.clone().into_os_string(),
            OsString::from("--data-directory"),
            directory.clone().into_os_string(),
            OsString::from("--evidence-directory"),
            directory.clone().into_os_string(),
        ];
        for native in [false, true] {
            let mut args = base.clone();
            args.push(
                if native {
                    "--native-engine"
                } else {
                    "--publisher-engine"
                }
                .into(),
            );
            args.push("--use-original-pq2".into());
            let config = configuration(args).map_err(io::Error::other)?;
            assert!(!config.use_repacked_ptq1);
            assert!(config.use_abliterate_lora);
            assert_eq!(config.native_engine, native);
        }
        for first in ["--use-original-pq2", "--use-repacked-ptq1"] {
            for second in ["--use-original-pq2", "--use-repacked-ptq1"] {
                let mut rejected = base.clone();
                rejected.extend([first, second].map(OsString::from));
                assert!(configuration(rejected).is_err());
            }
        }
        for (flag, expected) in [
            ("pq2", crate::client::Bonsai25Variant::Pq2),
            ("ptq1", crate::client::Bonsai25Variant::Ptq1),
        ] {
            let mut valid = base.clone();
            valid.extend(["--native-engine", "--bonsai-2-5", flag].map(OsString::from));
            assert_eq!(
                configuration(valid)
                    .map_err(io::Error::other)?
                    .bonsai25_variant,
                Some(expected)
            );
        }
        for bad in [
            vec!["--bonsai-2-5"],
            vec!["--bonsai-2-5", "invalid"],
            vec!["--bonsai-2-5", "pq2", "--bonsai-2-5", "ptq1"],
            vec!["--bonsai-2-5", "pq2", "--use-repacked-ptq1"],
            vec!["--bonsai-2-5", "pq2", "--use-original-pq2"],
            vec!["--publisher-engine", "--bonsai-2-5", "pq2"],
        ] {
            let mut rejected = base.clone();
            rejected.extend(bad.into_iter().map(OsString::from));
            assert!(configuration(rejected).is_err());
        }
        fs::remove_file(broker)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    fn assert_runtime_defaults_and_rollback(args: &[OsString]) -> io::Result<()> {
        let default = configuration(args.to_vec()).map_err(io::Error::other)?;
        assert!(default.use_abliterate_lora);
        assert!(!default.use_repacked_ptq1);
        assert!(default.native_engine);
        assert!(!default.require_broker_pipe);
        assert_eq!(default.native_row_workers, 4);
        assert_eq!(default.prompt_reuse, crate::client::PromptReuse::Disabled);
        assert!(default.prompt_reuse_checkpoints.is_none());
        assert!(default.prompt_reuse_checkpoint_bytes.is_none());
        assert!(default.kv_cache.is_none());
        assert!(default.cpu_prefill_gdn.is_none());
        assert!(default.spec_type.is_none());
        assert!(default.draft_max.is_none());
        assert!(default.cpu_family.is_none());
        assert!(default.placement.is_none());
        for (name, placement) in [
            ("auto", crate::client::Placement::Auto),
            ("cpu", crate::client::Placement::Cpu),
        ] {
            let mut selected = args.to_vec();
            selected.extend(["--placement", name].map(OsString::from));
            assert_eq!(
                configuration(selected.clone())
                    .map_err(io::Error::other)?
                    .placement,
                Some(placement)
            );
            selected.extend(["--placement", name].map(OsString::from));
            assert!(configuration(selected).is_err());
        }
        for suffix in [
            vec!["--placement"],
            vec!["--placement", "CPU"],
            vec!["--placement", "npu"],
            vec!["--publisher-engine", "--placement", "auto"],
            vec!["--publisher-engine", "--placement", "cpu"],
        ] {
            let mut rejected = args.to_vec();
            rejected.extend(suffix.into_iter().map(OsString::from));
            assert!(configuration(rejected).is_err());
        }
        for (name, policy) in [
            ("f16", crate::client::KvCache::F16),
            ("rsq4", crate::client::KvCache::Rsq4),
            ("rsq4k3v", crate::client::KvCache::Rsq4K3v),
        ] {
            let mut selected = args.to_vec();
            selected.extend(["--kv-cache", name].map(OsString::from));
            assert_eq!(
                configuration(selected.clone())
                    .map_err(io::Error::other)?
                    .kv_cache,
                Some(policy)
            );
            selected.extend(["--kv-cache", name].map(OsString::from));
            assert!(configuration(selected).is_err());
        }
        for suffix in [
            vec!["--kv-cache"],
            vec!["--kv-cache", "F16"],
            vec!["--kv-cache", "rsq3"],
            vec!["--publisher-engine", "--kv-cache", "f16"],
        ] {
            let mut rejected = args.to_vec();
            rejected.extend(suffix.into_iter().map(OsString::from));
            assert!(configuration(rejected).is_err());
        }
        for (value, reuse) in [
            ("disabled", crate::client::PromptReuse::Disabled),
            ("latest", crate::client::PromptReuse::Latest),
            (
                "prompt-end-and-latest",
                crate::client::PromptReuse::PromptEndAndLatest,
            ),
        ] {
            let mut selected = args.to_vec();
            selected.extend(["--prompt-reuse", value].map(OsString::from));
            assert_eq!(
                configuration(selected.clone())
                    .map_err(io::Error::other)?
                    .prompt_reuse,
                reuse
            );
            selected.extend(["--prompt-reuse", value].map(OsString::from));
            assert!(configuration(selected).is_err());
        }
        for value in ["Latest", "true", "", "approximate"] {
            let mut rejected = args.to_vec();
            rejected.extend(["--prompt-reuse", value].map(OsString::from));
            assert!(configuration(rejected).is_err());
        }
        let mut branch_selected = args.to_vec();
        branch_selected.extend(
            [
                "--require-broker-pipe",
                "--prompt-reuse",
                "prompt-end-and-latest",
                "--prompt-reuse-checkpoints",
                "4",
                "--prompt-reuse-checkpoint-bytes",
                "134217728",
                "--cpu-prefill-tokens",
                "16",
                "--kv-cache",
                "f16",
                "--cpu-prefill-gdn",
                "wy-f32",
                "--cpu-family",
                "x1",
            ]
            .map(OsString::from),
        );
        let parsed_branch = configuration(branch_selected).map_err(io::Error::other)?;
        assert!(parsed_branch.require_broker_pipe);
        assert_eq!(
            parsed_branch.prompt_reuse,
            crate::client::PromptReuse::PromptEndAndLatest
        );
        assert_eq!(parsed_branch.prompt_reuse_checkpoints, Some(4));
        assert_eq!(
            parsed_branch.prompt_reuse_checkpoint_bytes,
            Some(134_217_728)
        );
        assert_eq!(
            parsed_branch.cpu_prefill_gdn,
            Some(crate::client::CpuPrefillGdn::WyF32)
        );
        assert_eq!(parsed_branch.cpu_family, Some(crate::client::CpuFamily::X1));
        assert_eq!(parsed_branch.native_row_workers, 8);

        let mut spec_selected = args.to_vec();
        spec_selected.extend(
            [
                "--cpu-prefill-tokens",
                "16",
                "--spec-type",
                "suffix",
                "--draft-max",
                "6",
                "--cpu-family",
                "x2",
            ]
            .map(OsString::from),
        );
        let parsed_spec = configuration(spec_selected).map_err(io::Error::other)?;
        assert_eq!(
            parsed_spec.spec_type,
            Some(crate::client::SpeculationType::Suffix)
        );
        assert_eq!(parsed_spec.draft_max, Some(6));
        assert_eq!(parsed_spec.cpu_family, Some(crate::client::CpuFamily::X2));
        assert_eq!(parsed_spec.native_row_workers, 8);

        for invalid in [
            vec!["--publisher-engine", "--require-broker-pipe"],
            vec!["--prompt-reuse-checkpoints", "2"],
            vec![
                "--prompt-reuse",
                "disabled",
                "--prompt-reuse-checkpoints",
                "2",
            ],
            vec![
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoints",
                "0",
            ],
            vec![
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoints",
                "9",
            ],
            vec![
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoints",
                "02",
            ],
            vec![
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoint-bytes",
                "0",
            ],
            vec!["--cpu-prefill-gdn", "wy-f32"],
            vec!["--cpu-prefill-tokens", "16", "--cpu-prefill-gdn", "wy-f32"],
            vec![
                "--cpu-prefill-tokens",
                "16",
                "--kv-cache",
                "rsq4",
                "--cpu-prefill-gdn",
                "wy-f32",
            ],
            vec!["--draft-max", "4"],
            vec!["--spec-type", "none", "--draft-max", "4"],
            vec!["--spec-type", "suffix"],
            vec![
                "--cpu-prefill-tokens",
                "16",
                "--kv-cache",
                "f16",
                "--cpu-prefill-gdn",
                "wy-f32",
                "--spec-type",
                "suffix",
            ],
            vec!["--cpu-family", "x3"],
            vec!["--publisher-engine", "--cpu-family", "x2"],
        ] {
            let mut rejected = args.to_vec();
            rejected.extend(invalid.into_iter().map(OsString::from));
            assert!(configuration(rejected).is_err());
        }
        let mut publisher = args.to_vec();
        publisher.push("--publisher-engine".into());
        let rollback = configuration(publisher).map_err(io::Error::other)?;
        assert!(!rollback.native_engine);
        assert_eq!(rollback.native_row_workers, 1);
        assert!(!rollback.use_repacked_ptq1);
        assert!(rollback.use_abliterate_lora);
        publisher = args.to_vec();
        publisher.extend(["--publisher-engine", "--prompt-reuse", "latest"].map(OsString::from));
        assert!(configuration(publisher).is_err());
        for first in ["--native-engine", "--publisher-engine"] {
            for second in ["--native-engine", "--publisher-engine"] {
                let mut rejected = args.to_vec();
                rejected.extend([first, second].map(OsString::from));
                assert!(configuration(rejected).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn placement_kv_cache_and_rust_defaults_preserve_explicit_runtime_weight_and_adapter_choices()
    -> io::Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("Bonsai-demo-config-{stamp}"));
        fs::create_dir(&directory)?;
        let broker = directory.join("broker.exe");
        fs::write(&broker, b"")?;
        let args = vec![
            OsString::from("--broker-path"),
            broker.clone().into_os_string(),
            OsString::from("--data-directory"),
            directory.clone().into_os_string(),
            OsString::from("--evidence-directory"),
            directory.clone().into_os_string(),
        ];
        assert_runtime_defaults_and_rollback(&args)?;
        let mut repacked = args.clone();
        repacked.push("--use-repacked-ptq1".into());
        let alternate = configuration(repacked.clone()).map_err(io::Error::other)?;
        assert!(alternate.use_repacked_ptq1);
        assert!(alternate.use_abliterate_lora);
        repacked.push("--use-repacked-ptq1".into());
        assert!(configuration(repacked).is_err());
        let mut native = args.clone();
        native.push("--native-engine".into());
        assert!(configuration(native.clone()).is_ok_and(|config| config.use_abliterate_lora));
        for value in ["0", "2", "3", "5", "16", "01", "+4", "4.0", " 4"] {
            let mut rejected = native.clone();
            rejected.extend(["--native-row-workers", value].map(OsString::from));
            assert!(configuration(rejected).is_err());
        }
        for value in ["1", "4", "8", "12"] {
            let mut configured = native.clone();
            configured.extend(["--native-row-workers", value].map(OsString::from));
            let selected = configuration(configured.clone()).map_err(io::Error::other)?;
            assert_eq!(selected.native_row_workers.to_string(), value);
            assert!(selected.use_abliterate_lora && !selected.use_repacked_ptq1);
            configured.extend(["--native-row-workers", value].map(OsString::from));
            assert!(configuration(configured).is_err());
        }
        for value in ["4", "8", "12"] {
            let mut publisher_rows = args.clone();
            publisher_rows
                .extend(["--publisher-engine", "--native-row-workers", value].map(OsString::from));
            assert!(configuration(publisher_rows).is_err());
        }
        let x1_default = super::configuration_for_family(
            args.clone(),
            super::host_family_from_processor_name(
                "Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU",
            ),
        )
        .map_err(io::Error::other)?;
        assert_eq!(x1_default.native_row_workers, 8);
        let x2_default = super::configuration_for_family(
            args.clone(),
            super::host_family_from_processor_name(
                "Snapdragon(R) X2 Elite Extreme - X2E-94-100 - Qualcomm(R) Oryon(TM) CPU",
            ),
        )
        .map_err(io::Error::other)?;
        assert_eq!(x2_default.native_row_workers, 8);
        for (value, expected) in [("8", 8_u8), ("16", 16), ("32", 32)] {
            let mut tiled = native.clone();
            tiled.extend(["--cpu-prefill-tokens", value].map(OsString::from));
            assert_eq!(
                configuration(tiled.clone())
                    .map_err(io::Error::other)?
                    .cpu_prefill_tokens,
                Some(expected)
            );
            tiled.extend(["--cpu-prefill-tokens", value].map(OsString::from));
            assert!(configuration(tiled).is_err());
        }
        for invalid in ["0", "4", "64", "eight"] {
            let mut rejected = native.clone();
            rejected.extend(["--cpu-prefill-tokens", invalid].map(OsString::from));
            assert!(configuration(rejected).is_err());
        }
        let mut publisher_prefill = args.clone();
        publisher_prefill
            .extend(["--publisher-engine", "--cpu-prefill-tokens", "8"].map(OsString::from));
        assert!(configuration(publisher_prefill).is_err());
        let mut missing_rows = native.clone();
        missing_rows.push("--native-row-workers".into());
        assert!(configuration(missing_rows).is_err());
        native.push("--no-abliterate-lora".into());
        let rust = configuration(native.clone()).map_err(io::Error::other)?;
        assert!(rust.native_engine);
        assert!(!rust.use_abliterate_lora);
        let mut duplicate_native = native.clone();
        duplicate_native.push("--native-engine".into());
        assert!(configuration(duplicate_native).is_err());
        let mut mmproj_disabled = native.clone();
        mmproj_disabled.push("--no-mmproj".into());
        let no_vision = configuration(mmproj_disabled.clone()).map_err(io::Error::other)?;
        assert!(no_vision.no_mmproj);
        mmproj_disabled.push("--no-mmproj".into());
        assert!(configuration(mmproj_disabled).is_err());
        let daisy_gguf = directory.join("daisyrose-vet.gguf");
        fs::write(&daisy_gguf, b"GGUF")?;
        let mut with_daisy = native.clone();
        with_daisy.extend([
            OsString::from("--daisyrose-lora"),
            daisy_gguf.as_os_str().to_owned(),
        ]);
        let parsed_daisy = configuration(with_daisy.clone()).map_err(io::Error::other)?;
        assert_eq!(parsed_daisy.daisyrose_lora.as_ref(), Some(&daisy_gguf));
        let mut conflict_daisy = with_daisy.clone();
        conflict_daisy.push("--no-daisyrose-lora".into());
        assert!(configuration(conflict_daisy).is_err());
        with_daisy.extend([
            OsString::from("--daisyrose-lora"),
            daisy_gguf.as_os_str().to_owned(),
        ]);
        assert!(configuration(with_daisy).is_err());

        let mut no_daisy = native.clone();
        no_daisy.push("--no-daisyrose-lora".into());
        let parsed_no_daisy = configuration(no_daisy.clone()).map_err(io::Error::other)?;
        assert!(parsed_no_daisy.daisyrose_lora.is_none());
        assert!(parsed_no_daisy.no_daisyrose_lora);
        assert!(!parsed_daisy.no_daisyrose_lora);
        no_daisy.push("--no-daisyrose-lora".into());
        assert!(configuration(no_daisy).is_err());

        let mut publisher_daisy = args.clone();
        publisher_daisy.extend([
            OsString::from("--publisher-engine"),
            OsString::from("--daisyrose-lora"),
            daisy_gguf.as_os_str().to_owned(),
        ]);
        assert!(configuration(publisher_daisy).is_err());

        native.push("--use-repacked-ptq1".into());
        assert!(configuration(native).is_ok_and(|config| config.use_repacked_ptq1));
        let mut disabled = args;
        disabled.push("--no-abliterate-lora".into());
        let opted_out = configuration(disabled.clone()).map_err(io::Error::other)?;
        assert!(!opted_out.use_abliterate_lora);
        assert!(!opted_out.no_mmproj);
        disabled.push("--no-abliterate-lora".into());
        assert!(configuration(disabled).is_err());
        // Neither mode requires adapter bytes at UI parsing time; the broker
        // verifies the selected artifacts before creating an inference process.
        assert!(!directory.join("adapters").exists());
        fs::remove_file(daisy_gguf)?;
        fs::remove_file(broker)?;
        fs::remove_dir(directory)?;
        Ok(())
    }
}

#[cfg(not(windows))]
fn main() -> std::process::ExitCode {
    std::process::ExitCode::FAILURE
}
