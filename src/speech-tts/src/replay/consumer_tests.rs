//! Admission controls use literal model identities and budget boundaries; no model executes.

use super::{
    Request, arguments, hexadecimal, input, model_artifacts, request_digest, retained_schedule,
};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::io;

fn request() -> Value {
    let identity = json!({"revision":"01".repeat(20),"compiler":"reviewed compiler",
        "target":"aarch64-pc-windows-msvc","executableSha256":"02".repeat(32),
        "arithmetic":"independent-native-tts"});
    let mut candidate = identity.clone();
    candidate["arithmetic"] = crate::observation::ARITHMETIC_IDENTITY.into();
    candidate["executableSha256"] = "03".repeat(32).into();
    let artifact = json!({"path":"C:\\retained\\original","bytes":10,"sha256":"04".repeat(32)});
    json!({"schemaVersion":1,"reference":identity,"candidate":candidate,"quantization":"q8",
        "models":{"design":"C:\\models\\design.gguf","base":"C:\\models\\base.gguf","codec":"C:\\models\\codec.gguf"},
        "workers":2,"cases":[{"text":artifact,"voice":artifact,"packet":artifact,"retainedVoice":false}],
        "tolerances":{"prompt":{"kind":"bits"},"hidden":{"kind":"bits"},"logits":{"kind":"finite","absolute":0.01,"relative":0.02},
            "cache":{"kind":"bits"},"speaker":{"kind":"bits"},"pcm":{"kind":"bits"}},
        "ownerStartedTick":1000,"workDeadlineTick":271000,"clockFrequency":1000,
        "fixedWork":"One complete original TTS replay including hashing, loading and cleanup.","estimatedSeconds":240})
}

fn parse(value: Value) -> io::Result<Request> {
    serde_json::from_value(value).map_err(io::Error::other)
}

#[test]
fn rust_tts_consumer_rejects_self_comparison_missing_rules_and_invalid_schedules() -> io::Result<()>
{
    let parsed = parse(request())?;
    let _rules = parsed.validate()?;
    for (path, value) in [
        ("workers", json!(0)),
        ("workers", json!(257)),
        ("cases", json!([])),
        ("cases", json!(vec![request()["cases"][0].clone(); 9])),
    ] {
        let mut changed = request();
        changed[path] = value;
        assert!(parse(changed)?.validate().is_err());
    }
    for field in ["arithmetic", "executableSha256"] {
        let mut changed = request();
        changed["reference"][field] = changed["candidate"][field].clone();
        assert!(parse(changed)?.validate().is_err());
    }
    for unknown in ["nativeFallback", "allowPartial", "skipCache"] {
        let mut changed = request();
        changed[unknown] = true.into();
        assert!(parse(changed).is_err());
    }
    let mut changed = request();
    changed["cases"][0]["retainedVoice"] = true.into();
    assert!(parse(changed)?.validate().is_err());
    assert!(
        retained_schedule([
            (false, "A"),
            (true, "A"),
            (false, "B"),
            (true, "B"),
            (false, "A")
        ])
        .is_ok()
    );
    for voices in [
        vec![(true, "A")],
        vec![(false, "A"), (false, "A")],
        vec![(false, "A"), (true, "B")],
    ] {
        assert!(retained_schedule(voices).is_err());
    }
    let mut changed = request();
    changed["tolerances"]
        .as_object_mut()
        .ok_or_else(|| io::Error::other("Missing rules."))?
        .remove("pcm");
    assert!(parse(changed).is_err());
    let mut changed = request();
    changed["tolerances"]["logits"]["relative"] = (-0.1).into();
    assert!(parse(changed)?.validate().is_err());
    let digest = "01".repeat(32);
    let args = [
        "--request",
        "C:\\retained\\request.json",
        "--sha256",
        &digest,
    ];
    assert!(arguments(args.map(Into::into)).is_ok());
    assert!(arguments(args.into_iter().chain(["--allow-partial"]).map(Into::into)).is_err());
    for value in ["00".repeat(32), "AA".repeat(32), "g0".repeat(32)] {
        assert!(hexadecimal::<32>(&value).is_err());
    }
    Ok(())
}

#[test]
fn rust_tts_consumer_preserves_original_aggregate_deadline_and_cleanup_reserve() -> io::Result<()> {
    let parsed = parse(request())?;
    assert!(parsed.admit_clock(1000, 1000).is_ok());
    assert!(parsed.admit_clock(31000, 1000).is_ok());
    for (tick, frequency) in [
        (999, 1000),
        (31001, 1000),
        (271000, 1000),
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
        assert!(parse(changed)?.admit_clock(1001, 1000).is_err());
    }
    Ok(())
}

#[test]
fn rust_tts_consumer_binds_literal_q8_q4_models_and_lossless_private_input() -> io::Result<()> {
    let (models, lock) = model_artifacts(&parse(request())?)?;
    assert_eq!(
        [models[0].bytes, models[1].bytes, models[2].bytes],
        [2_042_833_824, 2_079_448_256, 291_150_624]
    );
    assert_eq!(
        models[0].sha256,
        "575610ab1ddcca4dca6bd9a64bcd859d93bbad8764f9cab24e1dbc0c51f62276"
    );
    assert_eq!(
        lock,
        <[u8; 32]>::from(Sha256::digest(include_bytes!("../../model.lock.json")))
    );
    let mut changed = request();
    changed["quantization"] = "q4".into();
    let (q4, q4_lock) = model_artifacts(&parse(changed)?)?;
    assert_eq!(
        [q4[0].bytes, q4[1].bytes, q4[2].bytes],
        [1_182_630_816, 1_219_245_248, 291_150_624]
    );
    assert_ne!(q4_lock, lock);
    assert_eq!(q4[2].sha256, models[2].sha256);
    let text = "  日本語\n🙂  ";
    assert_eq!(input(text.as_bytes().to_vec(), 2048)?, text);
    for bytes in [
        vec![],
        vec![0xff],
        b" \r\n ".to_vec(),
        b"hello\0voice".to_vec(),
    ] {
        assert!(input(bytes, 2048).is_err());
    }
    assert!(input("🙂".repeat(1025).into_bytes(), 1024).is_err());
    // Literal length-prefixed byte stream: u64(1), 'a', u64(2), 'bc'.
    let original = vec![
        1, 0, 0, 0, 0, 0, 0, 0, b'a', 2, 0, 0, 0, 0, 0, 0, 0, b'b', b'c',
    ];
    assert_eq!(
        request_digest("a", "bc")?,
        <[u8; 32]>::from(Sha256::digest(&original))
    );
    assert_ne!(request_digest("ab", "c")?, request_digest("a", "bc")?);
    Ok(())
}
