//! Literal miniature-graph replay answers; no expected values come from capture.

use super::{
    Binding, CacheValues, CaptureLayout, CaptureWriter, DecoderSnapshot, LayerCache, Provider,
    Replay, Stage, Tolerance, Tolerances,
};
use crate::{
    protocol::PromptTokens,
    rust_engine::{
        ARITHMETIC_IDENTITY, Observation,
        tests::{assert_cleared, recognizer},
    },
};
use sha2::{Digest as _, Sha256};
use std::io;

fn binding() -> Binding<'static> {
    Binding {
        reference: Provider {
            revision: [1; 20],
            compiler: "independent mathematical fixture",
            target: "mathematical-f32",
            executable: [2; 32],
            arithmetic: "literal-asr-miniature-v1",
        },
        candidate: Provider {
            revision: [3; 20],
            compiler: "test-compiler",
            target: "test-target",
            executable: [4; 32],
            arithmetic: ARITHMETIC_IDENTITY,
        },
        model: [5; 32],
        pcm: Sha256::digest([0; 201 * 4]).into(),
    }
}

fn number(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend(value.to_le_bytes());
}
fn text(bytes: &mut Vec<u8>, value: &str) {
    number(bytes, u64::try_from(value.len()).unwrap_or_default());
    bytes.extend(value.as_bytes());
}
fn floats(bytes: &mut Vec<u8>, values: &[f32]) {
    number(bytes, u64::try_from(values.len()).unwrap_or_default());
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
}

struct Packet {
    bytes: Vec<u8>,
    frontend: usize,
    encoder: usize,
    prompt: usize,
    decoder: usize,
    cache: usize,
    terminal: usize,
}

fn packet() -> Packet {
    let binding = binding();
    let mut bytes = b"ASRREPL1".to_vec();
    for provider in [&binding.reference, &binding.candidate] {
        bytes.extend(provider.revision);
        text(&mut bytes, provider.compiler);
        text(&mut bytes, provider.target);
        bytes.extend(provider.executable);
        text(&mut bytes, provider.arithmetic);
    }
    bytes.extend(binding.model);
    bytes.extend(binding.pcm);
    bytes.extend([0; 4]);
    let frontend = bytes.len();
    bytes.push(1);
    number(&mut bytes, 1);
    floats(&mut bytes, &[-1.5; 128]);
    let encoder = bytes.len();
    bytes.push(2);
    number(&mut bytes, 1);
    number(&mut bytes, 2);
    floats(&mut bytes, &[2.0, -3.0]);
    let prompt = bytes.len();
    bytes.push(3);
    number(&mut bytes, 16);
    for id in [1_u32, 4, 3, 2, 3, 1, 5, 3, 8, 9, 10, 2, 3, 1, 6, 3] {
        bytes.extend(id.to_le_bytes());
    }
    number(&mut bytes, 9);
    number(&mut bytes, 10);
    let decoder = bytes.len();
    bytes.push(4);
    number(&mut bytes, 0);
    floats(&mut bytes, &[1.0 / 1.5_f32.sqrt(); 11]);
    bytes.extend(0_u32.to_le_bytes());
    number(&mut bytes, 16);
    number(&mut bytes, 2);
    let cache = bytes.len();
    for layer in 0..2 {
        number(&mut bytes, layer);
        // Independently known RoPE zero signs and F16-rounded unit/audio rows.
        let mut keys = [0.0_f32; 32];
        for index in [4, 6, 9, 16, 18, 21, 30] {
            keys[index] = -0.0;
        }
        floats(&mut bytes, &keys);
        let mut values = [0.0_f32; 32];
        if layer == 1 {
            for row in values.as_chunks_mut::<2>().0 {
                row[0] = 0.816_406_25;
            }
            values[18..20].copy_from_slice(&[0.730_468_75, -1.095_703_1]);
        }
        floats(&mut bytes, &values);
    }
    let terminal = bytes.len();
    bytes.push(5);
    number(&mut bytes, 1);
    number(&mut bytes, 0);
    Packet {
        bytes,
        frontend,
        encoder,
        prompt,
        decoder,
        cache,
        terminal,
    }
}

const fn capture_layout() -> CaptureLayout {
    CaptureLayout {
        encoder_width: 2,
        vocabulary: 11,
        decoder_layers: 2,
        kv_width: 2,
        eos_token: 0,
    }
}

fn capture_prefix<W: io::Write>(capture: &mut CaptureWriter<W>) -> io::Result<()> {
    capture.frontend(1, &[-1.5; 128], &mut || false)?;
    capture.encoder(1, &[2.0, -3.0], &mut || false)?;
    capture.prompt(
        &[1, 4, 3, 2, 3, 1, 5, 3, 8, 9, 10, 2, 3, 1, 6, 3],
        9..10,
        &mut || false,
    )
}

fn capture_answer<W: io::Write>(capture: &mut CaptureWriter<W>) -> io::Result<()> {
    // The same mathematical answers as packet(), without invoking either graph.
    let mut keys = [0_u16; 32];
    for index in [4, 6, 9, 16, 18, 21, 30] {
        keys[index] = 0x8000;
    }
    let mut values = [0_u16; 32];
    for row in values.as_chunks_mut::<2>().0 {
        row[0] = 0x3a88;
    }
    values[18..20].copy_from_slice(&[0x39d8, 0xbc62]);
    capture.decoder(
        &DecoderSnapshot {
            generation_index: 0,
            logits: &[1.0 / 1.5_f32.sqrt(); 11],
            token: 0,
            positions: 16,
            layers: &[
                LayerCache {
                    keys: CacheValues::F16(&keys),
                    values: CacheValues::F16(&[0; 32]),
                },
                LayerCache {
                    keys: CacheValues::F16(&keys),
                    values: CacheValues::F16(&values),
                },
            ],
        },
        &mut || false,
    )
}

#[test]
fn rust_asr_capture_writes_original_literal_packet_and_replays() -> io::Result<()> {
    let mut capture =
        CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
    capture_prefix(&mut capture)?;
    capture_answer(&mut capture)?;
    let bytes = capture.finish(Ok(""), &mut || false)?;
    assert_eq!(bytes, packet().bytes);
    let mut recognizer = recognizer(0)?;
    assert_eq!(
        Replay::new(bytes.as_slice(), &binding(), Tolerances::BITS)?.transcribe(
            &mut recognizer,
            &[0.0; 201],
            &mut || false
        )?,
        ""
    );
    assert_cleared(&recognizer);
    Ok(())
}

#[test]
fn rust_asr_capture_requires_complete_original_state_greedy_eos_and_success() -> io::Result<()> {
    for case in 0..6 {
        let mut capture =
            CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
        capture_prefix(&mut capture)?;
        let mut logits = [0.0; 11];
        if case == 5 {
            logits[0] = f32::NAN;
        }
        let layers = [
            LayerCache {
                keys: CacheValues::F32(&[0.0; 32]),
                values: CacheValues::F32(&[0.0; 32]),
            },
            LayerCache {
                keys: CacheValues::F32(&[0.0; 32]),
                values: CacheValues::F32(&[0.0; 32]),
            },
        ];
        assert!(
            capture
                .decoder(
                    &DecoderSnapshot {
                        generation_index: usize::from(case == 0),
                        logits: if case == 1 { &logits[..10] } else { &logits },
                        token: u32::from(case == 2),
                        positions: if case == 3 { 15 } else { 16 },
                        layers: if case == 4 { &layers[..1] } else { &layers },
                    },
                    &mut || false
                )
                .is_err()
        );
        assert!(capture_answer(&mut capture).is_err());
        assert!(capture.finish(Ok(""), &mut || false).is_err());
    }
    let mut layout = capture_layout();
    layout.eos_token = 1;
    let mut capture = CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, layout)?;
    capture_prefix(&mut capture)?;
    capture_answer(&mut capture)?;
    assert!(capture.finish(Ok(""), &mut || false).is_err());
    let mut capture =
        CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
    capture_prefix(&mut capture)?;
    capture_answer(&mut capture)?;
    assert!(
        capture
            .finish(Err(io::ErrorKind::Interrupted.into()), &mut || false)
            .is_err()
    );
    let mut capture =
        CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
    capture_prefix(&mut capture)?;
    capture_answer(&mut capture)?;
    assert!(capture_answer(&mut capture).is_err());
    assert!(capture.finish(Ok(""), &mut || false).is_err());
    Ok(())
}

#[test]
fn rust_asr_capture_rejects_changed_layout_and_sticky_order_cancel_unwind() -> io::Result<()> {
    let mut layout = capture_layout();
    layout.decoder_layers = 0;
    assert!(CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, layout).is_err());
    let mut identity = binding();
    identity.reference.executable = [0; 32];
    assert!(CaptureWriter::new(Vec::new(), &identity, Tolerances::BITS, capture_layout()).is_err());
    for case in 0..4 {
        let mut capture =
            CaptureWriter::new(Vec::new(), &binding(), Tolerances::BITS, capture_layout())?;
        if case == 0 {
            assert!(capture.encoder(1, &[2.0, -3.0], &mut || false).is_err());
        } else if case == 1 {
            assert!(capture.frontend(1, &[0.0; 127], &mut || false).is_err());
        } else if case == 2 {
            assert!(capture.frontend(1, &[0.0; 128], &mut || true).is_err());
        } else {
            #[expect(
                clippy::panic,
                reason = "Controlled callback unwind checks sticky capture refusal."
            )]
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                capture.frontend(1, &[0.0; 128], &mut || {
                    panic!("capture cancellation unwind")
                })
            }));
            assert!(result.is_err());
        }
        assert!(capture_prefix(&mut capture).is_err());
        assert!(capture.finish(Ok(""), &mut || false).is_err());
    }
    Ok(())
}

#[test]
fn rust_asr_capture_bounds_buffer_and_never_recovers_from_write_or_flush_failure() -> io::Result<()>
{
    struct Sink {
        fail_write: bool,
        fail_flush: bool,
    }
    impl io::Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            assert!(bytes.len() <= 16 * 1024);
            if self.fail_write {
                Err(io::ErrorKind::BrokenPipe.into())
            } else {
                Ok(bytes.len())
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.fail_flush {
                Err(io::ErrorKind::BrokenPipe.into())
            } else {
                Ok(())
            }
        }
    }
    let mut capture = CaptureWriter::new(
        Sink {
            fail_write: true,
            fail_flush: false,
        },
        &binding(),
        Tolerances::BITS,
        capture_layout(),
    )?;
    assert!(
        capture
            .frontend(33, &vec![0.0; 33 * 128], &mut || false)
            .is_err()
    );
    assert!(capture_prefix(&mut capture).is_err());
    assert!(capture.finish(Ok(""), &mut || false).is_err());
    let mut capture = CaptureWriter::new(
        Sink {
            fail_write: false,
            fail_flush: true,
        },
        &binding(),
        Tolerances::BITS,
        capture_layout(),
    )?;
    capture_prefix(&mut capture)?;
    capture_answer(&mut capture)?;
    assert!(capture.finish(Ok(""), &mut || false).is_err());
    Ok(())
}

#[test]
fn rust_asr_replay_literal_complete_packet_and_fresh_reset() -> io::Result<()> {
    let mut recognizer = recognizer(0)?;
    for _ in 0..2 {
        let packet = packet();
        let replay = Replay::new(packet.bytes.as_slice(), &binding(), Tolerances::BITS)?;
        assert_eq!(
            replay.transcribe(&mut recognizer, &[0.0; 201], &mut || false)?,
            ""
        );
        assert_cleared(&recognizer);
    }
    Ok(())
}

#[test]
fn rust_asr_replay_rejects_changed_identity_input_and_invalid_tolerances() -> io::Result<()> {
    let packet = packet();
    let mut changed_tolerance = packet.bytes.clone();
    changed_tolerance[packet.frontend - 1] = 1;
    assert!(Replay::new(changed_tolerance.as_slice(), &binding(), Tolerances::BITS).is_err());
    let mut identity = binding();
    identity.candidate.executable[0] ^= 1;
    assert!(Replay::new(packet.bytes.as_slice(), &identity, Tolerances::BITS).is_err());
    let mut identity = binding();
    identity.model[0] ^= 1;
    assert!(Replay::new(packet.bytes.as_slice(), &identity, Tolerances::BITS).is_err());
    let mut identity = binding();
    identity.reference.revision = [0; 20];
    assert!(Replay::new(packet.bytes.as_slice(), &identity, Tolerances::BITS).is_err());
    for invalid in [f32::NAN, f32::INFINITY, -1.0] {
        let tolerances = Tolerances {
            logits: Tolerance::Finite {
                absolute: invalid,
                relative: 0.0,
            },
            ..Tolerances::BITS
        };
        assert!(Replay::new(packet.bytes.as_slice(), &binding(), tolerances).is_err());
    }
    let mut recognizer = recognizer(0)?;
    let replay = Replay::new(packet.bytes.as_slice(), &binding(), Tolerances::BITS)?;
    assert!(
        replay
            .transcribe(&mut recognizer, &[0.1; 201], &mut || false)
            .is_err()
    );
    assert_cleared(&recognizer);
    Ok(())
}

#[test]
fn rust_asr_replay_rejects_every_boundary_truncation_cache_bits_and_extra_bytes() -> io::Result<()>
{
    let packet = packet();
    let mut recognizer = recognizer(0)?;
    for offset in [
        packet.frontend,
        packet.encoder,
        packet.prompt,
        packet.decoder,
        packet.cache,
        packet.terminal,
    ] {
        let replay = Replay::new(&packet.bytes[..offset], &binding(), Tolerances::BITS)?;
        assert!(
            replay
                .transcribe(&mut recognizer, &[0.0; 201], &mut || false)
                .is_err()
        );
        assert_cleared(&recognizer);
        let mut changed = packet.bytes.clone();
        changed[offset] ^= 1;
        let replay = Replay::new(changed.as_slice(), &binding(), Tolerances::BITS)?;
        assert!(
            replay
                .transcribe(&mut recognizer, &[0.0; 201], &mut || false)
                .is_err()
        );
        assert_cleared(&recognizer);
    }
    let mut changed = packet.bytes.clone();
    // Layer zero key four is negative zero; exact replay cannot erase its sign.
    changed[packet.cache + 8 + 8 + 4 * 4 + 3] ^= 0x80;
    let replay = Replay::new(changed.as_slice(), &binding(), Tolerances::BITS)?;
    assert!(
        replay
            .transcribe(&mut recognizer, &[0.0; 201], &mut || false)
            .is_err()
    );
    assert_cleared(&recognizer);
    let mut extra = packet.bytes;
    extra.push(0);
    let replay = Replay::new(extra.as_slice(), &binding(), Tolerances::BITS)?;
    assert!(
        replay
            .transcribe(&mut recognizer, &[0.0; 201], &mut || false)
            .is_err()
    );
    assert_cleared(&recognizer);
    Ok(())
}

#[test]
fn rust_asr_replay_tolerance_exception_bits_and_sticky_failed_schedule() -> io::Result<()> {
    let finite = Tolerance::Finite {
        absolute: 0.125,
        relative: 0.25,
    };
    assert!(finite.accepts(2.0, 2.625));
    assert!(!finite.accepts(2.0, 2.626));
    assert!(!finite.accepts(f32::INFINITY, f32::INFINITY));
    assert!(!finite.accepts(f32::NAN, f32::NAN));
    assert!(!Tolerance::Bits.accepts(0.0, -0.0));
    let nan = f32::from_bits(0x7fc0_0123);
    assert!(Tolerance::Bits.accepts(nan, nan));
    assert!(!Tolerance::Bits.accepts(nan, f32::from_bits(0x7fc0_0124)));
    let packet = packet();
    let mut replay = Replay::new(packet.bytes.as_slice(), &binding(), Tolerances::BITS)?;
    let prompt = PromptTokens::new([1, 2, 3, 4, 5, 6], [7, 8, 9], 10)?.automatic_prompt(1, 512)?;
    assert!(
        replay
            .observe(Observation::Prompt(&prompt), &mut || false)
            .is_err()
    );
    assert_eq!(replay.stage, Stage::Failed);
    let mel = crate::frontend::Frontend::from_checkpoint(None, None)?
        .compute(&[0.0; 201], &mut || false)?;
    assert!(
        replay
            .observe(Observation::Frontend(&mel), &mut || false)
            .is_err()
    );
    assert!(replay.finish("", &mut || false).is_err());
    Ok(())
}

#[test]
fn rust_asr_replay_cancelled_and_failed_engine_cannot_complete() -> io::Result<()> {
    let packet = packet();
    let mut cancelled = recognizer(0)?;
    let replay = Replay::new(packet.bytes.as_slice(), &binding(), Tolerances::BITS)?;
    assert_eq!(
        replay
            .transcribe(&mut cancelled, &[0.0; 201], &mut || true)
            .err()
            .map(|error| error.kind()),
        Some(io::ErrorKind::Interrupted)
    );
    assert_cleared(&cancelled);
    // Exact same first greedy decision, but token zero is not this model's EOS.
    // A complete-looking packet cannot turn the failing continuation into success.
    let mut failing = recognizer(1)?;
    let replay = Replay::new(packet.bytes.as_slice(), &binding(), Tolerances::BITS)?;
    assert!(
        replay
            .transcribe(&mut failing, &[0.0; 201], &mut || false)
            .is_err()
    );
    assert_cleared(&failing);
    assert_eq!(
        failing
            .transcribe_cancellable(&[0.0; 201], &mut || true)
            .err()
            .map(|error| error.kind()),
        Some(io::ErrorKind::Interrupted)
    );
    Ok(())
}

struct InterruptedInput<'a> {
    bytes: &'a [u8],
    offset: usize,
    boundary: usize,
    cancel: &'a std::cell::Cell<bool>,
    panic: bool,
}

impl io::Read for InterruptedInput<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if self.offset >= self.boundary {
            assert!(!self.panic, "injected replay source unwind");
            self.cancel.set(true);
        }
        let length = output.len().min(16).min(self.bytes.len() - self.offset);
        output[..length].copy_from_slice(&self.bytes[self.offset..self.offset + length]);
        self.offset += length;
        Ok(length)
    }
}

#[test]
fn rust_asr_replay_stream_cancellation_and_unwind_reset_active_decoder() -> io::Result<()> {
    let packet = packet();
    for panic in [false, true] {
        let cancelled = std::cell::Cell::new(false);
        let input = InterruptedInput {
            bytes: &packet.bytes,
            offset: 0,
            boundary: packet.cache,
            cancel: &cancelled,
            panic,
        };
        let replay = Replay::new(input, &binding(), Tolerances::BITS)?;
        let mut recognizer = recognizer(0)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            replay.transcribe(&mut recognizer, &[0.0; 201], &mut || cancelled.get())
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert_eq!(
                result
                    .map_err(|_| io::Error::other("unexpected unwind"))?
                    .err()
                    .map(|error| error.kind()),
                Some(io::ErrorKind::Interrupted)
            );
        }
        assert_cleared(&recognizer);
        assert_eq!(recognizer.transcribe(&[0.0; 201])?, "");
    }
    Ok(())
}

#[test]
fn rust_asr_replay_terminal_and_eof_cancellation_cannot_complete() -> io::Result<()> {
    let packet = packet();
    for boundary in [packet.terminal, packet.bytes.len()] {
        let cancelled = std::cell::Cell::new(false);
        let input = InterruptedInput {
            bytes: &packet.bytes,
            offset: 0,
            boundary,
            cancel: &cancelled,
            panic: false,
        };
        let replay = Replay::new(input, &binding(), Tolerances::BITS)?;
        let mut recognizer = recognizer(0)?;
        let result = replay.transcribe(&mut recognizer, &[0.0; 201], &mut || cancelled.get());
        assert_eq!(
            result.err().map(|error| error.kind()),
            Some(io::ErrorKind::Interrupted)
        );
        assert!(cancelled.get());
        assert_cleared(&recognizer);
        assert_eq!(recognizer.transcribe(&[0.0; 201])?, "");
    }
    Ok(())
}
