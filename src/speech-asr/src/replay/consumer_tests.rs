//! Request, native deadline and immutable audio-wire admission controls.

use super::{Request, arguments, hexadecimal, pcm};
use serde_json::{Value, json};
use std::io;

fn request() -> Value {
    let identity = json!({"revision":"01".repeat(20),"compiler":"reviewed compiler",
        "target":"aarch64-pc-windows-msvc","executableSha256":"02".repeat(32),
        "arithmetic":"independent-native-arithmetic"});
    let artifact = json!({"path":"C:\\retained\\input.bin","bytes":804,"sha256":"03".repeat(32)});
    json!({"schemaVersion":1,"reference":identity,"candidate":identity,"model":"C:\\models\\asr.gguf",
        "pcm":artifact,"packet":artifact,"tolerances":{"frontend":{"kind":"bits"},
        "encoder":{"kind":"finite","absolute":0.01,"relative":0.02},
        "logits":{"kind":"bits"},"cache":{"kind":"bits"}},
        "ownerStartedTick":1000,"workDeadlineTick":271000,"clockFrequency":1000,
        "fixedWork":"One original PCM complete independent replay with model loading and cleanup.","estimatedSeconds":240})
}

#[test]
fn rust_asr_consumer_requires_exact_request_shape_identities_and_fixed_cli() -> io::Result<()> {
    let parsed: Request = serde_json::from_value(request()).map_err(io::Error::other)?;
    assert_eq!(parsed.reference.provider()?.revision, [1; 20]);
    for unknown in ["nativeFallback", "skipCache", "allowPartial"] {
        let mut changed = request();
        changed[unknown] = true.into();
        assert!(serde_json::from_value::<Request>(changed).is_err());
    }
    let digest = "01".repeat(32);
    let args = [
        "--request",
        "C:\\retained\\request.json",
        "--sha256",
        &digest,
    ];
    assert!(arguments(args.map(Into::into)).is_ok());
    assert!(arguments(args.into_iter().chain(["--skip-cache"]).map(Into::into)).is_err());
    for value in [
        "",
        "00",
        &"00".repeat(32),
        &"AA".repeat(32),
        &"g0".repeat(32),
    ] {
        assert!(hexadecimal::<32>(value).is_err());
    }
    Ok(())
}

#[test]
fn rust_asr_consumer_keeps_original_deadline_and_cleanup_reserve() -> io::Result<()> {
    let parsed: Request = serde_json::from_value(request()).map_err(io::Error::other)?;
    assert!(parsed.admit_clock(1000, 1000).is_ok());
    assert!(parsed.admit_clock(31000, 1000).is_ok());
    for (tick, frequency) in [
        (999, 1000),
        (271000, 1000),
        (31001, 1000),
        (270999, 1000),
        (1001, 0),
        (1001, 2000),
    ] {
        assert!(parsed.admit_clock(tick, frequency).is_err());
    }
    for (field, value) in [
        ("workDeadlineTick", 271001),
        ("ownerStartedTick", 0),
        ("estimatedSeconds", 0),
        ("estimatedSeconds", 241),
    ] {
        let mut changed = request();
        changed[field] = value.into();
        let parsed: Request = serde_json::from_value(changed).map_err(io::Error::other)?;
        assert!(parsed.admit_clock(1001, 1000).is_err());
    }
    Ok(())
}

#[test]
fn rust_asr_consumer_pcm_retains_original_float_bits_and_rejects_partial_or_invalid_audio()
-> io::Result<()> {
    let mut bytes = vec![0; 201 * 4];
    bytes[0..4].copy_from_slice(&(-0.0_f32).to_le_bytes());
    bytes[4..8].copy_from_slice(&(-1.0_f32).to_le_bytes());
    bytes[8..12].copy_from_slice(&1.0_f32.to_le_bytes());
    let samples = pcm(&bytes)?;
    assert_eq!(samples[0].to_bits(), 0x8000_0000);
    assert_eq!(&samples[1..3], &[-1.0, 1.0]);
    for length in [0, 200 * 4, 201 * 4 - 1] {
        assert!(pcm(&bytes[..length]).is_err());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.001, -1.001] {
        bytes[0..4].copy_from_slice(&value.to_le_bytes());
        assert!(pcm(&bytes).is_err());
    }
    Ok(())
}
