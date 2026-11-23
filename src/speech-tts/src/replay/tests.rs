//! Literal miniature observations exercise the wire contract, not native parity.

use std::io;

use super::{Binding, Provider, Replay, Tolerance, Tolerances, request_digest};
use crate::{
    engine::LayerKvCache,
    observation::{
        ARITHMETIC_IDENTITY, Observation, Phase, PredictorObservation, TalkerObservation,
    },
};

fn binding() -> Binding<'static> {
    Binding {
        reference: Provider {
            revision: [1; 20],
            compiler: "literal mathematical observations",
            target: "miniature-f32",
            executable: [2; 32],
            arithmetic: "literal-tts-miniature-v1",
        },
        candidate: Provider {
            revision: [3; 20],
            compiler: "test-compiler",
            target: "test-target",
            executable: [4; 32],
            arithmetic: ARITHMETIC_IDENTITY,
        },
        model_lock: [5; 32],
        models: [[6; 32], [7; 32], [8; 32]],
        request: [9; 32],
        workers: 2,
        retained_voice: false,
    }
}

fn number(bytes: &mut Vec<u8>, value: usize) -> io::Result<()> {
    bytes.extend(
        u64::try_from(value)
            .map_err(io::Error::other)?
            .to_le_bytes(),
    );
    Ok(())
}

fn text(bytes: &mut Vec<u8>, value: &str) -> io::Result<()> {
    number(bytes, value.len())?;
    bytes.extend(value.as_bytes());
    Ok(())
}

fn floats(bytes: &mut Vec<u8>, values: &[f32]) -> io::Result<()> {
    number(bytes, values.len())?;
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    Ok(())
}

fn header() -> io::Result<Vec<u8>> {
    let binding = binding();
    header_for(&binding)
}

fn header_for(binding: &Binding<'_>) -> io::Result<Vec<u8>> {
    let mut bytes = b"TTSREPL1".to_vec();
    for provider in [&binding.reference, &binding.candidate] {
        bytes.extend(provider.revision);
        text(&mut bytes, provider.compiler)?;
        text(&mut bytes, provider.target)?;
        bytes.extend(provider.executable);
        text(&mut bytes, provider.arithmetic)?;
    }
    bytes.extend(binding.model_lock);
    for model in binding.models {
        bytes.extend(model);
    }
    bytes.extend(binding.request);
    number(&mut bytes, binding.workers)?;
    bytes.push(u8::from(binding.retained_voice));
    bytes.extend([0; 6]);
    Ok(bytes)
}

fn cache_packet(bytes: &mut Vec<u8>, positions: usize) -> io::Result<()> {
    number(bytes, 2)?;
    for (index, keys, values) in [
        (0, [0.0_f32, -0.0], [1.0_f32, 2.0]),
        (1, [3.0, 4.0], [-5.0, 6.0]),
    ] {
        number(bytes, index)?;
        number(bytes, positions)?;
        floats(bytes, &keys.repeat(positions))?;
        floats(bytes, &values.repeat(positions))?;
    }
    Ok(())
}

fn caches(positions: usize) -> Vec<LayerKvCache> {
    vec![
        LayerKvCache {
            k: [0x0000_u16, 0x8000].repeat(positions),
            v: [0x3c00_u16, 0x4000].repeat(positions),
            seq_len: positions,
        },
        LayerKvCache {
            k: [0x4200_u16, 0x4400].repeat(positions),
            v: [0xc500_u16, 0x4600].repeat(positions),
            seq_len: positions,
        },
    ]
}

fn graph_packet(bytes: &mut Vec<u8>, phase: u8) -> io::Result<()> {
    bytes.extend([1, phase]);
    number(bytes, 2)?;
    floats(bytes, &[1.0, -0.0])?;
    floats(bytes, &[2.0, -3.0])?;
    number(bytes, 1)?;
    floats(bytes, &[-1.0, 0.0])?;
    number(bytes, 1)?;
    bytes.extend([2, phase]);
    number(bytes, 0)?;
    floats(bytes, &[3.0, -4.0])?;
    floats(bytes, &[0.25, -0.5])?;
    cache_packet(bytes, 2)?;
    bytes.extend([3, phase]);
    number(bytes, 0)?;
    bytes.extend(0_i32.to_le_bytes());
    bytes.push(0);
    bytes.extend([4, phase]);
    number(bytes, 0)?;
    cache_packet(bytes, 1)?;
    for codebook in 0..15 {
        bytes.extend([5, phase]);
        number(bytes, 0)?;
        number(bytes, codebook)?;
        floats(bytes, &[-2.0, 1.0])?;
        floats(bytes, &[9.0, -9.0])?;
        cache_packet(bytes, codebook + 2)?;
        bytes.extend([6, phase]);
        number(bytes, 0)?;
        number(bytes, codebook)?;
        bytes.extend(0_i32.to_le_bytes());
    }
    bytes.extend([7, phase]);
    number(bytes, 0)?;
    bytes.extend([0; 16 * 4]);
    if phase == 1 {
        bytes.extend([10, phase]);
        floats(bytes, &[0.5, -0.25])?;
    }
    bytes.extend([2, phase]);
    number(bytes, 1)?;
    floats(bytes, &[-3.0, 4.0])?;
    floats(bytes, &[-0.25, 0.5])?;
    cache_packet(bytes, 3)?;
    bytes.extend([3, phase]);
    number(bytes, 1)?;
    bytes.extend(1_i32.to_le_bytes());
    bytes.push(1);
    Ok(())
}

fn packet() -> io::Result<Vec<u8>> {
    let mut bytes = header()?;
    graph_packet(&mut bytes, 0)?;
    bytes.extend([10, 0]);
    floats(&mut bytes, &[0.125, -0.125, 0.25])?;
    bytes.push(8);
    floats(&mut bytes, &[2.0, -3.0])?;
    bytes.push(9);
    number(&mut bytes, 1)?;
    bytes.extend([0; 16 * 4]);
    graph_packet(&mut bytes, 1)?;
    bytes.push(11);
    number(&mut bytes, 2)?;
    bytes.extend(24_000_u32.to_le_bytes());
    bytes.push(255);
    number(&mut bytes, 79)?;
    Ok(bytes)
}

struct ReplayFixture {
    binding: Binding<'static>,
    prompt_rows: Vec<Vec<f32>>,
    prompt_trailing: Vec<Vec<f32>>,
    caches_by_pos: Vec<Vec<LayerKvCache>>,
}

impl ReplayFixture {
    fn new() -> Self {
        Self {
            binding: binding(),
            prompt_rows: vec![vec![1.0, -0.0], vec![2.0, -3.0]],
            prompt_trailing: vec![vec![-1.0, 0.0]],
            caches_by_pos: (0..=16).map(caches).collect(),
        }
    }
}

fn graph_events_with_fixture(
    fixture: &ReplayFixture,
    observer: &mut impl FnMut(Observation<'_>) -> io::Result<()>,
    phase: Phase,
) -> io::Result<()> {
    let mut observe =
        |observation: TalkerObservation<'_>| observer(Observation::Talker { phase, observation });
    observe(TalkerObservation::Prompt {
        rows: &fixture.prompt_rows,
        trailing: &fixture.prompt_trailing,
        cacheable_rows: 1,
    })?;
    observe(TalkerObservation::Logits {
        frame: 0,
        hidden: &[3.0, -4.0],
        logits: &[0.25, -0.5],
        cache: &fixture.caches_by_pos[2],
    })?;
    observe(TalkerObservation::Decision {
        frame: 0,
        token: 0,
        eos: false,
    })?;
    observe(TalkerObservation::Predictor {
        frame: 0,
        observation: PredictorObservation::Prefill {
            cache: &fixture.caches_by_pos[1],
        },
    })?;
    for codebook in 0..15 {
        observe(TalkerObservation::Predictor {
            frame: 0,
            observation: PredictorObservation::Logits {
                codebook,
                hidden: &[-2.0, 1.0],
                logits: &[9.0, -9.0],
                cache: &fixture.caches_by_pos[codebook + 2],
            },
        })?;
        observe(TalkerObservation::Predictor {
            frame: 0,
            observation: PredictorObservation::Decision { codebook, token: 0 },
        })?;
    }
    observe(TalkerObservation::Frame {
        frame: 0,
        codes: &[0; 16],
    })?;
    if phase == Phase::Base {
        observer(Observation::Pcm {
            phase,
            samples: &[0.5, -0.25],
        })?;
    }
    observer(Observation::Talker {
        phase,
        observation: TalkerObservation::Logits {
            frame: 1,
            hidden: &[-3.0, 4.0],
            logits: &[-0.25, 0.5],
            cache: &fixture.caches_by_pos[3],
        },
    })?;
    observer(Observation::Talker {
        phase,
        observation: TalkerObservation::Decision {
            frame: 1,
            token: 1,
            eos: true,
        },
    })
}

fn graph_events(
    observer: &mut impl FnMut(Observation<'_>) -> io::Result<()>,
    phase: Phase,
) -> io::Result<()> {
    let fixture = ReplayFixture::new();
    graph_events_with_fixture(&fixture, observer, phase)
}

fn drive_events_with_fixture(
    fixture: &ReplayFixture,
    observer: &mut impl FnMut(Observation<'_>) -> io::Result<()>,
) -> io::Result<()> {
    graph_events_with_fixture(fixture, observer, Phase::Design)?;
    observer(Observation::Pcm {
        phase: Phase::Design,
        samples: &[0.125, -0.125, 0.25],
    })?;
    observer(Observation::Speaker(&[2.0, -3.0]))?;
    observer(Observation::ReferenceCodes(&[[0; 16]]))?;
    graph_events_with_fixture(fixture, observer, Phase::Base)?;
    observer(Observation::Complete {
        samples: 2,
        sample_rate: 24_000,
    })
}

fn drive_events(observer: &mut impl FnMut(Observation<'_>) -> io::Result<()>) -> io::Result<()> {
    let fixture = ReplayFixture::new();
    drive_events_with_fixture(&fixture, observer)
}

fn drive_with_fixture<R: io::Read>(
    fixture: &ReplayFixture,
    replay: &mut Replay<R>,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<()> {
    drive_events_with_fixture(fixture, &mut |event| replay.observe(event, cancel))
}

fn drive<R: io::Read>(replay: &mut Replay<R>, cancel: &mut impl FnMut() -> bool) -> io::Result<()> {
    let fixture = ReplayFixture::new();
    drive_with_fixture(&fixture, replay, cancel)
}

#[test]
fn tts_replay_compares_every_literal_graph_cache_code_pcm_and_terminal() -> io::Result<()> {
    let bytes = packet()?;
    let mut replay = Replay::new(bytes.as_slice(), &binding(), Tolerances::BITS)?;
    drive(&mut replay, &mut || false)?;
    assert!(replay.complete && !replay.failed);
    assert_eq!(replay.events, 79);
    assert!(
        replay
            .observe(Observation::Speaker(&[]), &mut || false)
            .is_err()
    );
    Ok(())
}

#[test]
fn tts_replay_rejects_changed_identities_workers_voice_and_tolerances() -> io::Result<()> {
    let bytes = header()?;
    for changed in 0..12 {
        let mut binding = binding();
        match changed {
            0 => binding.reference.revision[0] ^= 1,
            1 => binding.reference.compiler = "changed",
            2 => binding.candidate.target = "changed",
            3 => binding.candidate.executable[0] ^= 1,
            4 => {
                binding.candidate.arithmetic = "rust-tts-packed-f32-activation-v1-f32-kv";
            }
            5 => binding.model_lock[0] ^= 1,
            6..=8 => binding.models[changed - 6][0] ^= 1,
            9 => binding.request[0] ^= 1,
            10 => binding.workers = 4,
            _ => binding.retained_voice = true,
        }
        assert!(Replay::new(bytes.as_slice(), &binding, Tolerances::BITS).is_err());
    }
    let mut rules = Tolerances::BITS;
    rules.cache = Tolerance::Finite {
        absolute: -1.0,
        relative: 0.0,
    };
    assert!(Replay::new(bytes.as_slice(), &binding(), rules).is_err());
    assert_ne!(request_digest("ab", "c")?, request_digest("a", "bc")?);
    assert_ne!(request_digest("é", "v")?, request_digest("e", "v")?);
    Ok(())
}

#[test]
fn tts_replay_rejects_truncation_mutation_reordering_and_trailing_data() -> io::Result<()> {
    let original = packet()?;
    let header_len = header()?.len();
    let fixture = ReplayFixture::new();
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .clamp(1, 16);
    let total = original.len() - header_len;
    let chunk_size = total.div_ceil(workers);
    std::thread::scope(|scope| -> io::Result<()> {
        let mut handles = Vec::with_capacity(workers);
        for worker_idx in 0..workers {
            let start = header_len + worker_idx * chunk_size;
            if start >= original.len() {
                break;
            }
            let stop = (start + chunk_size).min(original.len());
            let original_ref = &original;
            let fixture_ref = &fixture;
            handles.push(scope.spawn(move || -> io::Result<()> {
                let mut changed = original_ref.clone();
                for end in start..stop {
                    // Truncate at each byte: none of the original replay inputs can disappear.
                    let mut replay =
                        Replay::new(&original_ref[..end], &fixture_ref.binding, Tolerances::BITS)?;
                    assert!(
                        drive_with_fixture(fixture_ref, &mut replay, &mut || false).is_err(),
                        "length {end}"
                    );
                    assert!(replay.failed && !replay.complete);
                    assert!(
                        replay
                            .observe(Observation::Speaker(&[]), &mut || false)
                            .is_err()
                    );
                    // Sample mutations across original record bytes, including cache bits and IDs.
                    if (end - header_len).is_multiple_of(37) {
                        changed[end] ^= 1;
                        let mut mut_replay = Replay::new(
                            changed.as_slice(),
                            &fixture_ref.binding,
                            Tolerances::BITS,
                        )?;
                        assert!(
                            drive_with_fixture(fixture_ref, &mut mut_replay, &mut || false)
                                .is_err(),
                            "byte {end}"
                        );
                        changed[end] ^= 1;
                    }
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| io::Error::other("replay check worker panicked"))??;
        }
        Ok(())
    })?;
    let mut trailing = original.clone();
    trailing.push(0);
    let mut replay = Replay::new(trailing.as_slice(), &fixture.binding, Tolerances::BITS)?;
    assert!(drive_with_fixture(&fixture, &mut replay, &mut || false).is_err());
    let mut replay = Replay::new(original.as_slice(), &fixture.binding, Tolerances::BITS)?;
    assert!(
        replay
            .observe(Observation::Speaker(&[2.0, -3.0]), &mut || false)
            .is_err()
    );
    Ok(())
}

#[test]
fn tts_replay_float_rules_preserve_exceptional_bits_and_finite_bounds() {
    assert!(!Tolerance::Bits.accepts(0.0, -0.0));
    assert!(Tolerance::Bits.accepts(f32::from_bits(0x7fc0_0001), f32::from_bits(0x7fc0_0001)));
    assert!(!Tolerance::Bits.accepts(f32::from_bits(0x7fc0_0001), f32::from_bits(0x7fc0_0002)));
    let finite = Tolerance::Finite {
        absolute: 0.25,
        relative: 0.125,
    };
    assert!(finite.accepts(2.0, 2.5));
    assert!(!finite.accepts(2.0, 2.500_001));
    assert!(!finite.accepts(f32::INFINITY, f32::INFINITY));
    assert!(!finite.accepts(f32::NAN, 0.0));
    assert!(
        !Tolerance::Finite {
            absolute: f32::NAN,
            relative: 0.0
        }
        .valid()
    );
}

#[test]
fn tts_replay_cancellation_and_unwind_are_sticky() -> io::Result<()> {
    let bytes = packet()?;
    let fixture = ReplayFixture::new();
    let mut checks = 0;
    let mut complete = Replay::new(bytes.as_slice(), &fixture.binding, Tolerances::BITS)?;
    drive_with_fixture(&fixture, &mut complete, &mut || {
        checks += 1;
        false
    })?;
    for boundary in 1..=checks {
        let mut replay = Replay::new(bytes.as_slice(), &fixture.binding, Tolerances::BITS)?;
        let mut seen = 0;
        let error = drive_with_fixture(&fixture, &mut replay, &mut || {
            seen += 1;
            seen == boundary
        });
        assert_eq!(
            error.err().map(|e| e.kind()),
            Some(io::ErrorKind::Interrupted)
        );
        assert!(replay.failed && !replay.complete);
    }
    let mut replay = Replay::new(bytes.as_slice(), &fixture.binding, Tolerances::BITS)?;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drive_with_fixture(&fixture, &mut replay, &mut || {
            std::panic::resume_unwind(Box::new("cancel owner unwound"))
        })
    }));
    assert!(unwind.is_err());
    assert!(replay.failed && !replay.complete);
    assert!(
        replay
            .observe(Observation::Speaker(&[]), &mut || false)
            .is_err()
    );
    Ok(())
}

fn capture_layout() -> super::CaptureLayout {
    let decoder = super::DecoderLayout {
        hidden: 2,
        vocabulary: 2,
        layers: 2,
        kv_width: 2,
    };
    let graph = super::GraphLayout {
        talker: decoder,
        predictor: decoder,
        eos: 1,
    };
    super::CaptureLayout {
        design: graph,
        base: graph,
        speaker_width: 2,
        sample_rate: 24_000,
    }
}

#[test]
fn tts_capture_matches_independent_literal_packet_and_replay() -> io::Result<()> {
    let mut capture =
        super::CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
    drive_events(&mut |event| capture.observe(event.into(), &mut || false))?;
    let bytes = capture.finish(Ok(()), &mut || false)?;
    assert_eq!(bytes, packet()?);
    let mut replay = Replay::new(bytes.as_slice(), &binding(), Tolerances::BITS)?;
    drive(&mut replay, &mut || false)?;
    assert!(replay.complete);
    Ok(())
}

#[test]
fn tts_capture_borrows_original_contiguous_prompts_and_all_layer_views() -> io::Result<()> {
    use super::{Cache, CaptureEvent, LayerCache, Rows};
    let mut capture =
        super::CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
    drive_events(&mut |event| match CaptureEvent::from(event) {
        CaptureEvent::Prompt {
            phase,
            cacheable_rows,
            ..
        } => capture.observe(
            CaptureEvent::Prompt {
                phase,
                rows: Rows::Contiguous {
                    values: &[1.0, -0.0, 2.0, -3.0],
                    width: 2,
                },
                trailing: Rows::Contiguous {
                    values: &[-1.0, 0.0],
                    width: 2,
                },
                cacheable_rows,
            },
            &mut || false,
        ),
        CaptureEvent::Logits {
            phase,
            frame,
            codebook,
            hidden,
            logits,
            cache: Cache::Rust(layers),
        } => {
            let owned: Vec<(Vec<f32>, Vec<f32>, usize)> = layers
                .iter()
                .map(|layer| {
                    let mut k = vec![0.0_f32; layer.k.len()];
                    let mut v = vec![0.0_f32; layer.v.len()];
                    crate::engine::widen_f16_slice(&layer.k, &mut k);
                    crate::engine::widen_f16_slice(&layer.v, &mut v);
                    (k, v, layer.seq_len)
                })
                .collect();
            let layers: Vec<_> = owned
                .iter()
                .map(|(k, v, seq_len)| LayerCache {
                    keys: k,
                    values: v,
                    positions: *seq_len,
                })
                .collect();
            capture.observe(
                CaptureEvent::Logits {
                    phase,
                    frame,
                    codebook,
                    hidden,
                    logits,
                    cache: Cache::Original(&layers),
                },
                &mut || false,
            )
        }
        CaptureEvent::PredictorPrefill {
            phase,
            frame,
            cache: Cache::Rust(layers),
        } => {
            let owned: Vec<(Vec<f32>, Vec<f32>, usize)> = layers
                .iter()
                .map(|layer| {
                    let mut k = vec![0.0_f32; layer.k.len()];
                    let mut v = vec![0.0_f32; layer.v.len()];
                    crate::engine::widen_f16_slice(&layer.k, &mut k);
                    crate::engine::widen_f16_slice(&layer.v, &mut v);
                    (k, v, layer.seq_len)
                })
                .collect();
            let layers: Vec<_> = owned
                .iter()
                .map(|(k, v, seq_len)| LayerCache {
                    keys: k,
                    values: v,
                    positions: *seq_len,
                })
                .collect();
            capture.observe(
                CaptureEvent::PredictorPrefill {
                    phase,
                    frame,
                    cache: Cache::Original(&layers),
                },
                &mut || false,
            )
        }
        event => capture.observe(event, &mut || false),
    })?;
    assert_eq!(capture.finish(Ok(()), &mut || false)?, packet()?);
    Ok(())
}

#[test]
fn tts_capture_warm_voice_omits_design_and_retains_complete_base_conditioning() -> io::Result<()> {
    let mut binding = binding();
    binding.retained_voice = true;
    let mut expected = header_for(&binding)?;
    expected.push(8);
    floats(&mut expected, &[2.0, -3.0])?;
    expected.push(9);
    number(&mut expected, 1)?;
    expected.extend([0; 16 * 4]);
    graph_packet(&mut expected, 1)?;
    expected.push(11);
    number(&mut expected, 2)?;
    expected.extend(24_000_u32.to_le_bytes());
    expected.push(255);
    number(&mut expected, 41)?;
    let mut capture =
        super::CaptureWriter::new(Vec::new(), &binding, Tolerances::BITS, capture_layout())?;
    capture.observe(Observation::Speaker(&[2.0, -3.0]).into(), &mut || false)?;
    capture.observe(Observation::ReferenceCodes(&[[0; 16]]).into(), &mut || {
        false
    })?;
    graph_events(
        &mut |event| capture.observe(event.into(), &mut || false),
        Phase::Base,
    )?;
    capture.observe(
        Observation::Complete {
            samples: 2,
            sample_rate: 24_000,
        }
        .into(),
        &mut || false,
    )?;
    assert_eq!(capture.finish(Ok(()), &mut || false)?, expected);
    Ok(())
}

#[test]
fn tts_capture_rejects_incomplete_geometry_schedule_decisions_and_false_completion()
-> io::Result<()> {
    use super::{Cache, CaptureEvent, CaptureWriter, Rows};
    for changed in 0..11 {
        let mut capture =
            CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
        let mut modified = false;
        let result = drive_events(&mut |event| {
            let mut event = CaptureEvent::from(event);
            if !modified {
                match (changed, &mut event) {
                    (0, CaptureEvent::Prompt { rows, .. }) => {
                        *rows = Rows::Contiguous {
                            values: &[1.0, 2.0, 3.0],
                            width: 2,
                        };
                        modified = true;
                    }
                    (1, CaptureEvent::Logits { cache, .. }) => {
                        *cache = Cache::Original(&[]);
                        modified = true;
                    }
                    (2, CaptureEvent::Logits { frame, .. }) => {
                        *frame += 1;
                        modified = true;
                    }
                    (3, CaptureEvent::Decision { eos, .. }) => {
                        *eos = true;
                        modified = true;
                    }
                    (
                        4,
                        CaptureEvent::Decision {
                            token,
                            codebook: Some(_),
                            ..
                        },
                    ) => {
                        *token = 2;
                        modified = true;
                    }
                    (
                        5,
                        CaptureEvent::Logits {
                            codebook: Some(codebook),
                            ..
                        },
                    ) => {
                        *codebook += 1;
                        modified = true;
                    }
                    (6, CaptureEvent::Frame { codes, .. }) => {
                        *codes = &[1; 16];
                        modified = true;
                    }
                    (7, CaptureEvent::Pcm { samples, .. }) => {
                        *samples = &[f32::NAN];
                        modified = true;
                    }
                    (8, CaptureEvent::Speaker(values)) => {
                        *values = &[];
                        modified = true;
                    }
                    (9, CaptureEvent::Complete { samples, .. }) => {
                        *samples += 1;
                        modified = true;
                    }
                    (10, CaptureEvent::Complete { sample_rate, .. }) => {
                        *sample_rate = 48_000;
                        modified = true;
                    }
                    _ => {}
                }
            }
            capture.observe(event, &mut || false)
        });
        assert!(modified && result.is_err(), "mutation {changed}");
        assert!(capture.finish(Ok(()), &mut || false).is_err());
    }
    let capture = CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
    assert!(capture.finish(Ok(()), &mut || false).is_err());
    let mut capture =
        CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
    drive_events(&mut |event| capture.observe(event.into(), &mut || false))?;
    assert!(
        capture
            .finish(Err(io::ErrorKind::Other.into()), &mut || false)
            .is_err()
    );
    let mut capture =
        CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
    drive_events(&mut |event| capture.observe(event.into(), &mut || false))?;
    assert!(
        capture
            .observe(CaptureEvent::Speaker(&[2.0, -3.0]), &mut || false)
            .is_err()
    );
    assert!(capture.finish(Ok(()), &mut || false).is_err());
    Ok(())
}

struct FlushFailure;

impl io::Write for FlushFailure {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::ErrorKind::BrokenPipe.into())
    }
}

#[test]
fn tts_capture_cancellation_unwind_and_flush_failure_cannot_publish_success() -> io::Result<()> {
    use super::{CaptureEvent, CaptureWriter};
    let mut capture =
        CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
    let mut checks = 0;
    drive_events(&mut |event| {
        capture.observe(event.into(), &mut || {
            checks += 1;
            false
        })
    })?;
    capture.finish(Ok(()), &mut || {
        checks += 1;
        false
    })?;
    for boundary in 1..=checks {
        let mut capture =
            CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
        let mut seen = 0;
        let mut cancel = || {
            seen += 1;
            seen == boundary
        };
        let result = drive_events(&mut |event| capture.observe(event.into(), &mut cancel));
        let result = capture.finish(result, &mut cancel);
        assert_eq!(
            result.err().map(|error| error.kind()),
            Some(io::ErrorKind::Interrupted),
            "boundary {boundary}"
        );
    }
    let mut capture =
        CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        capture.observe(CaptureEvent::Speaker(&[]), &mut || {
            std::panic::resume_unwind(Box::new("capture cancellation unwound"))
        })
    }));
    assert!(unwound.is_err());
    assert!(capture.finish(Ok(()), &mut || false).is_err());
    let mut capture =
        CaptureWriter::new(FlushFailure, &binding(), Tolerances::BITS, capture_layout())?;
    drive_events(&mut |event| capture.observe(event.into(), &mut || false))?;
    assert_eq!(
        capture
            .finish(Ok(()), &mut || false)
            .err()
            .map(|error| error.kind()),
        Some(io::ErrorKind::BrokenPipe)
    );
    Ok(())
}
