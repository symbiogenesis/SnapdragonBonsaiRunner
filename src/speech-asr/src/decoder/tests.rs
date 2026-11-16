//! Hand-calculated matrix, GQA, cache-rounding and state-publication controls.

use super::{Block, CachePrecision, Config, LayerState, Model, Projection, attend, cache_value};
use crate::protocol::PromptTokens;
use std::{io, sync::Arc};

fn projection(values: &[f32]) -> io::Result<Projection> {
    Projection::f32(2, 2, values.to_vec())
}

fn model() -> io::Result<Arc<Model>> {
    let identity = [1.0, 0.0, 0.0, 1.0];
    let zero = [0.0; 4];
    Ok(Arc::new(Model::new(
        config(),
        projection(&identity)?,
        vec![Block {
            attention_norm: vec![1.0; 2],
            ffn_norm: vec![1.0; 2],
            query: projection(&zero)?,
            key: projection(&zero)?,
            value: projection(&identity)?,
            attention_out: projection(&identity)?,
            query_norm: vec![1.0; 2],
            key_norm: vec![1.0; 2],
            gate: projection(&zero)?,
            up: projection(&identity)?,
            down: projection(&identity)?,
        }],
        vec![1.0; 2],
    )?))
}

const fn config() -> Config {
    Config {
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        context: 64,
        epsilon: 3.0,
        rope_theta: 1_000_000.0,
        cache_precision: CachePrecision::F16,
    }
}

#[test]
fn rust_asr_decoder_f32_q8_signed_projection_answers() -> io::Result<()> {
    assert_eq!(
        projection(&[1.0, -2.0, 3.0, 4.0])?.multiply(&[5.0, 6.0], &mut || false)?,
        [-7.0, 39.0]
    );
    // Half scale 0x3800=0.5. Signed quants sum to -1.
    let mut block = vec![0; 34];
    block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
    block[2..6].copy_from_slice(&[1, 255, 127, 128]);
    let projection = Projection::q8(32, 1, block)?;
    assert_eq!(projection.multiply(&[1.0; 32], &mut || false)?, [-0.5]);
    assert!(Projection::q8(32, 1, vec![0; 33]).is_err());
    let mut bad_scale = vec![0; 34];
    bad_scale[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(Projection::q8(32, 1, bad_scale).is_err());
    Ok(())
}

#[test]
fn rust_asr_decoder_binary16_ties_subnormals_and_overflow() -> io::Result<()> {
    let precision = CachePrecision::F16;
    assert_eq!(
        cache_value(1.0 + 1.0 / 2048.0, precision)?.to_bits(),
        1.0_f32.to_bits()
    );
    assert_eq!(
        cache_value(1.0 + 3.0 / 2048.0, precision)?.to_bits(),
        (1.0_f32 + 1.0 / 512.0).to_bits()
    );
    assert_eq!(
        cache_value(1.0 / 33_554_432.0, precision)?.to_bits(),
        0.0_f32.to_bits()
    );
    assert_eq!(
        cache_value(3.0 / 33_554_432.0, precision)?.to_bits(),
        (1.0_f32 / 8_388_608.0).to_bits()
    );
    assert_eq!(
        cache_value(-0.0, precision)?.to_bits(),
        (-0.0_f32).to_bits()
    );
    assert!(cache_value(65_520.0, precision).is_err());
    assert!(cache_value(-65_520.0, precision).is_err());
    assert_eq!(cache_value(65_504.0, precision)?, 65_504.0);
    assert_eq!(cache_value(-65_504.0, precision)?, -65_504.0);
    assert!(cache_value(f32::NAN, precision).is_err());
    assert_eq!(
        cache_value(1.0 + 1.0 / 2048.0, CachePrecision::F32)?,
        1.0 + 1.0 / 2048.0
    );
    Ok(())
}

#[test]
fn rust_asr_decoder_uniform_attention_and_grouped_head_mapping() -> io::Result<()> {
    let c = Config {
        heads: 4,
        kv_heads: 2,
        ..config()
    };
    // Zero Q produces uniform probabilities. Heads 0/1 share KV0; 2/3
    // share KV1, while each has a distinct output range.
    let old = LayerState::from_f32(vec![0.0; 4], vec![1.0, 3.0, 10.0, 30.0]);
    let output = attend(
        &[0.0; 8],
        &[0.0; 4],
        &[3.0, 5.0, 30.0, 50.0],
        &old,
        c,
        1,
        &mut || false,
    )?;
    assert_eq!(output, [2.0, 4.0, 2.0, 4.0, 20.0, 40.0, 20.0, 40.0]);
    Ok(())
}

fn cache_vecs(cache: Option<crate::replay::LayerCache<'_>>) -> Option<(Vec<f32>, Vec<f32>)> {
    cache.map(|c| (c.keys.to_f32_vec(), c.values.to_f32_vec()))
}

fn cache_f16_ptrs(
    cache: Option<crate::replay::LayerCache<'_>>,
) -> Option<(*const u16, *const u16)> {
    let c = cache?;
    match (c.keys, c.values) {
        (crate::replay::CacheValues::F16(k), crate::replay::CacheValues::F16(v)) => {
            Some((k.as_ptr(), v.as_ptr()))
        }
        _ => None,
    }
}

#[test]
fn rust_asr_decoder_full_block_logits_complete_cache_and_reset() -> io::Result<()> {
    let model = model()?;
    let mut session = model.session();
    let logits = session.step_audio(&[1.0, 1.0], &mut || false)?;
    let expected = 1.5 / 5.25_f32.sqrt();
    assert_eq!(logits, [expected, expected]);
    assert_eq!(session.position(), 1);
    assert_eq!(
        cache_vecs(session.cache(0)),
        Some((vec![0.0, 0.0], vec![0.5, 0.5]))
    );
    assert_eq!(
        session.step_audio(&[-1.0, -1.0], &mut || false)?,
        [-0.5, -0.5]
    );
    assert_eq!(
        cache_vecs(session.cache(0)),
        Some((vec![0.0; 4], vec![0.5, 0.5, -0.5, -0.5]))
    );
    session.reset();
    assert_eq!(session.position(), 0);
    assert_eq!(cache_vecs(session.cache(0)), Some((Vec::new(), Vec::new())));
    assert_eq!(session.step_audio(&[1.0, 1.0], &mut || false)?, logits);
    // A separate session cannot inherit the first session's positions or KV.
    assert_eq!(model.session().position(), 0);
    Ok(())
}

#[test]
fn rust_asr_decoder_failures_cancellation_and_unwind_preserve_committed_bits() -> io::Result<()> {
    let mut session = model()?.session();
    let _ = session.step_audio(&[1.0, 1.0], &mut || false)?;
    let original =
        cache_vecs(session.cache(0)).ok_or_else(|| io::Error::other("missing test layer"))?;
    assert!(
        session
            .step_audio(&[f32::MAX, f32::MAX], &mut || false)
            .is_err()
    );
    assert!(session.step_audio(&[f32::NAN, 0.0], &mut || false).is_err());
    let mut calls = 0;
    let result = session.step_audio(&[1.0, 1.0], &mut || {
        calls += 1;
        calls == 7
    });
    assert_eq!(
        result.err().map(|error| error.kind()),
        Some(io::ErrorKind::Interrupted)
    );
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut calls = 0;
        let _ = session.step_audio(&[1.0, 1.0], &mut || {
            calls += 1;
            assert!(calls < 7, "injected decoder cancellation callback panic");
            false
        });
    }));
    assert!(unwind.is_err());
    assert_eq!(session.position(), 1);
    assert_eq!(cache_vecs(session.cache(0)), Some(original));
    Ok(())
}

#[test]
fn rust_asr_decoder_prompt_audio_injection_and_failed_prefill_reset() -> io::Result<()> {
    let model = model()?;
    let prompt = PromptTokens::new([0; 6], [0; 3], 2)?.automatic_prompt(1, 512)?;
    let audio = [2.0, -2.0];
    let mut prefilled = model.session();
    let logits = prefilled.prefill(&prompt, &audio, &mut || false)?;
    let mut independent = model.session();
    let mut expected = Vec::new();
    for position in 0..prompt.ids().len() {
        expected = if prompt.audio_rows().contains(&position) {
            independent.step_audio(&audio, &mut || false)?
        } else {
            independent.step(0, &mut || false)?
        };
    }
    assert_eq!(logits, expected);
    assert_eq!(
        cache_vecs(prefilled.cache(0)),
        cache_vecs(independent.cache(0))
    );
    let mut calls = 0;
    assert!(
        prefilled
            .prefill(&prompt, &audio, &mut || {
                calls += 1;
                calls > 80
            })
            .is_err()
    );
    assert_eq!(prefilled.position(), 0);
    assert_eq!(
        cache_vecs(prefilled.cache(0)),
        Some((Vec::new(), Vec::new()))
    );
    Ok(())
}

#[test]
fn rust_asr_decoder_prefill_omits_unused_heads_and_keeps_final_failure_atomic() -> io::Result<()> {
    let mut model = Arc::try_unwrap(model()?).map_err(|_| io::Error::other("shared test model"))?;
    // Prompt token 0 is zero. An unselected vocabulary row [2,2], together
    // with the large final norm, overflows ONLY if an audio row's unused head
    // is evaluated. The final zero prompt row has literal all-zero logits.
    model.embedding = Projection::f32(2, 2, vec![0.0, 0.0, 2.0, 2.0])?;
    model.output_norm = vec![f32::MAX; 2];
    model.blocks[0].attention_out = projection(&[0.0; 4])?;
    let model = Arc::new(model);
    let prompt = PromptTokens::new([0; 6], [0; 3], 2)?.automatic_prompt(1, 512)?;
    let mut session = model.session();
    assert_eq!(
        session.prefill(&prompt, &[1.0, 1.0], &mut || false)?,
        [0.0; 2]
    );
    assert_eq!(session.position(), 16);
    let (_, values) =
        cache_vecs(session.cache(0)).ok_or_else(|| io::Error::other("missing test layer"))?;
    let mut expected_values = [0.0; 32];
    expected_values[18..20].copy_from_slice(&[0.5; 2]);
    assert_eq!(values, expected_values);

    // A requested step still evaluates that head, fails before append and
    // leaves every previously committed K/V bit unchanged.
    let original =
        cache_vecs(session.cache(0)).ok_or_else(|| io::Error::other("missing test layer"))?;
    assert!(session.step_audio(&[1.0, 1.0], &mut || false).is_err());
    assert_eq!(session.position(), 16);
    assert_eq!(cache_vecs(session.cache(0)), Some(original));

    // Selecting that nonzero token at the final prompt position must still
    // evaluate the failing head and reset all earlier successful prompt rows.
    let failing = PromptTokens::new([0, 0, 1, 0, 0, 0], [0; 3], 2)?.automatic_prompt(1, 512)?;
    assert!(
        session
            .prefill(&failing, &[1.0, 1.0], &mut || false)
            .is_err()
    );
    assert_eq!(session.position(), 0);
    assert_eq!(cache_vecs(session.cache(0)), Some((Vec::new(), Vec::new())));
    assert_eq!(
        session.prefill(&prompt, &[1.0, 1.0], &mut || false)?,
        [0.0; 2]
    );
    Ok(())
}

#[test]
fn rust_asr_decoder_batched_prefill_and_parallel_decode_match_step_by_step_bits() -> io::Result<()>
{
    fn q8_projection(input: usize, output: usize, seed: u32) -> io::Result<Projection> {
        let blocks_per_row = input / 32;
        let mut bytes = Vec::with_capacity(output * blocks_per_row * 34);
        let scales = [0x3800_u16, 0x3400, 0x3C00, 0xB800, 0x3600];
        for row in 0..output {
            for blk in 0..blocks_per_row {
                let scale = scales[(row + blk + usize::try_from(seed).unwrap_or(0)) % scales.len()];
                bytes.extend_from_slice(&scale.to_le_bytes());
                for elem in 0..32_u8 {
                    let raw = ((row * 17
                        + blk * 31
                        + usize::from(elem) * 13
                        + usize::try_from(seed).unwrap_or(0))
                        % 19)
                        .cast_signed()
                        - 9;
                    let q = i8::try_from(raw).unwrap_or(0);
                    bytes.push(q.cast_unsigned());
                }
            }
        }
        Projection::q8(input, output, bytes)
    }

    fn f32_projection(input: usize, output: usize, seed: u32) -> io::Result<Projection> {
        let mut data = Vec::with_capacity(input * output);
        for row in 0..output {
            for col in 0..input {
                let n = ((row * 19 + col * 29 + usize::try_from(seed).unwrap_or(0)) % 15)
                    .cast_signed()
                    - 7;
                let denom = f32::from(u16::try_from(input).unwrap_or(1));
                data.push(f32::from(i8::try_from(n).unwrap_or(0)) / (4.0 * denom));
            }
        }
        Projection::f32(input, output, data)
    }

    let c = Config {
        hidden: 32,
        heads: 4,
        kv_heads: 2,
        head_dim: 4,
        intermediate: 64,
        context: 512,
        epsilon: 1.0e-5,
        rope_theta: 1_000_000.0,
        cache_precision: CachePrecision::F16,
    };
    let vocab = 12;
    let mut emb = vec![0.0_f32; vocab * c.hidden];
    for col in 0..c.hidden {
        emb[col] = if col % 2 == 0 { -0.0 } else { 0.0 };
    }
    for tok in 1..vocab {
        for col in 0..c.hidden {
            let v = ((tok * 11 + col * 7) % 17).cast_signed() - 8;
            emb[tok * c.hidden + col] = f32::from(i8::try_from(v).unwrap_or(0)) * 0.0625;
        }
    }
    let build_model = |decode_threads: usize, batch_threads: usize| -> io::Result<Arc<Model>> {
        let mut blocks = Vec::with_capacity(2);
        for layer in 0..2_u32 {
            blocks.push(Block {
                attention_norm: (0..c.hidden)
                    .map(|i| 0.8 + f32::from(u8::try_from(i % 7).unwrap_or(0)) * 0.05)
                    .collect(),
                ffn_norm: (0..c.hidden)
                    .map(|i| 0.9 + f32::from(u8::try_from(i % 5).unwrap_or(0)) * 0.04)
                    .collect(),
                query: q8_projection(c.hidden, c.heads * c.head_dim, 10 + layer)?,
                key: f32_projection(c.hidden, c.kv_heads * c.head_dim, 20 + layer)?,
                value: q8_projection(c.hidden, c.kv_heads * c.head_dim, 30 + layer)?,
                attention_out: f32_projection(c.heads * c.head_dim, c.hidden, 40 + layer)?,
                query_norm: vec![1.1, 0.95, 1.05, 0.9],
                key_norm: vec![0.9, 1.05, 1.15, 0.85],
                gate: q8_projection(c.hidden, c.intermediate, 50 + layer)?,
                up: f32_projection(c.hidden, c.intermediate, 60 + layer)?,
                down: q8_projection(c.intermediate, c.hidden, 70 + layer)?,
            });
        }
        Ok(Arc::new(
            Model::new(
                c,
                Projection::f32(c.hidden, vocab, emb.clone())?,
                blocks,
                vec![1.05; c.hidden],
            )?
            .with_threads(decode_threads, batch_threads)?,
        ))
    };

    let serial_model = build_model(1, 1)?;
    let parallel_model = build_model(4, 6)?;
    let prompt = PromptTokens::new([0, 1, 2, 3, 4, 5], [6, 7, 8], 9)?.automatic_prompt(4, 512)?;
    assert_eq!(prompt.ids().len(), 19);
    let mut audio = vec![0.0_f32; 4 * c.hidden];
    for (idx, sample) in audio.iter_mut().enumerate() {
        let v = ((idx * 13 + 3) % 19).cast_signed() - 9;
        *sample = f32::from(i8::try_from(v).unwrap_or(0)) * 0.05;
    }

    let mut serial_sess = serial_model.session();
    let mut expected_logits = Vec::new();
    for (pos, &id) in prompt.ids().iter().enumerate() {
        expected_logits = if prompt.audio_rows().contains(&pos) {
            let start = (pos - prompt.audio_rows().start) * c.hidden;
            serial_sess.step_audio(&audio[start..start + c.hidden], &mut || false)?
        } else {
            serial_sess.step(id, &mut || false)?
        };
    }

    let mut parallel_sess = parallel_model.session();
    let parallel_logits = parallel_sess.prefill(&prompt, &audio, &mut || false)?;
    parallel_sess.reserve_decode(3)?;
    let reserved_ptrs: Vec<_> = (0..2)
        .map(|layer| {
            cache_f16_ptrs(parallel_sess.cache(layer))
                .ok_or_else(|| io::Error::other("missing parallel layer"))
        })
        .collect::<io::Result<Vec<_>>>()?;
    assert_eq!(parallel_sess.position(), serial_sess.position());
    for (got, want) in parallel_logits.iter().zip(&expected_logits) {
        assert_eq!(got.to_bits(), want.to_bits(), "prefill logit mismatch");
    }
    for layer in 0..2 {
        let (pk, pv) = cache_vecs(parallel_sess.cache(layer))
            .ok_or_else(|| io::Error::other("missing parallel layer"))?;
        let (sk, sv) = cache_vecs(serial_sess.cache(layer))
            .ok_or_else(|| io::Error::other("missing serial layer"))?;
        for (i, (got, want)) in pk.iter().zip(&sk).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "prefill K bit mismatch layer={layer} i={i}"
            );
        }
        for (i, (got, want)) in pv.iter().zip(&sv).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "prefill V bit mismatch layer={layer} i={i}"
            );
        }
    }

    for next_tok in [10_u32, 11, 4] {
        let want_step = serial_sess.step(next_tok, &mut || false)?;
        let got_step = parallel_sess.step(next_tok, &mut || false)?;
        for (got, want) in got_step.iter().zip(&want_step) {
            assert_eq!(got.to_bits(), want.to_bits(), "decode step logit mismatch");
        }
        for (layer, &(k_ptr, v_ptr)) in reserved_ptrs.iter().enumerate() {
            assert_eq!(
                cache_f16_ptrs(parallel_sess.cache(layer)),
                Some((k_ptr, v_ptr))
            );
            let (pk, pv) = cache_vecs(parallel_sess.cache(layer))
                .ok_or_else(|| io::Error::other("missing parallel layer"))?;
            let (sk, sv) = cache_vecs(serial_sess.cache(layer))
                .ok_or_else(|| io::Error::other("missing serial layer"))?;
            for (i, (got, want)) in pk.iter().zip(&sk).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "decode K mismatch layer={layer} i={i}"
                );
            }
            for (i, (got, want)) in pv.iter().zip(&sv).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "decode V mismatch layer={layer} i={i}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn rust_asr_session_reset_retains_normal_capacity_and_shrinks_bursts() -> io::Result<()> {
    let c = Config {
        hidden: 16,
        heads: 2,
        kv_heads: 1,
        head_dim: 8,
        intermediate: 32,
        context: 1024,
        epsilon: 1e-6,
        rope_theta: 10_000.0,
        cache_precision: CachePrecision::F32,
    };
    let make_f32 = |input, output, salt| -> io::Result<Projection> {
        let values: Vec<f32> = (0..input * output)
            .map(|i| (f32::from(u16::try_from((i + salt) % 23).unwrap_or(0)) - 11.0) * 0.03125)
            .collect();
        Projection::f32(input, output, values)
    };
    let mut blocks = Vec::new();
    for layer in 0..2_usize {
        blocks.push(Block {
            attention_norm: vec![1.0; c.hidden],
            ffn_norm: vec![1.0; c.hidden],
            query: make_f32(c.hidden, c.heads * c.head_dim, 1 + layer)?,
            key: make_f32(c.hidden, c.kv_heads * c.head_dim, 2 + layer)?,
            value: make_f32(c.hidden, c.kv_heads * c.head_dim, 3 + layer)?,
            attention_out: make_f32(c.heads * c.head_dim, c.hidden, 4 + layer)?,
            query_norm: vec![1.0; c.head_dim],
            key_norm: vec![1.0; c.head_dim],
            gate: make_f32(c.hidden, c.intermediate, 5 + layer)?,
            up: make_f32(c.hidden, c.intermediate, 6 + layer)?,
            down: make_f32(c.intermediate, c.hidden, 7 + layer)?,
        });
    }
    let model = Arc::new(Model::new(
        c,
        make_f32(c.hidden, 16, 10)?,
        blocks,
        vec![1.0; c.hidden],
    )?);
    let tokens = PromptTokens::new([0, 1, 2, 3, 4, 5], [6, 7, 8], 9)?;

    // 1. Normal utterance (<= 128 tokens) retains capacities unchanged across reset.
    let mut session = model.session();
    let normal_prompt = tokens.automatic_prompt(8, c.context)?;
    let normal_audio: Vec<f32> = (0..8 * c.hidden)
        .map(|i| (f32::from(u16::try_from(i % 19).unwrap_or(0)) - 9.0) * 0.05)
        .collect();
    let baseline_logits = session.prefill(&normal_prompt, &normal_audio, &mut || false)?;
    let normal_kv_cap = session.layers[0].key_capacity();
    let normal_gate_cap = session.workspace.batch_gate.capacity();
    let normal_quads_cap = session.workspace.prepared.quads_capacity();
    session.reset();
    assert_eq!(session.layers[0].key_capacity(), normal_kv_cap);
    assert_eq!(session.workspace.batch_gate.capacity(), normal_gate_cap);
    assert_eq!(
        session.workspace.prepared.quads_capacity(),
        normal_quads_cap
    );

    // 2. High-water burst (> 2 * 128 tokens) grows capacities, then shrinks on reset.
    let burst_rows = 300_usize;
    let burst_prompt = tokens.automatic_prompt(burst_rows, c.context)?;
    let burst_audio = vec![0.025_f32; burst_rows * c.hidden];
    let _ = session.prefill(&burst_prompt, &burst_audio, &mut || false)?;
    let burst_kv_cap = session.layers[0].key_capacity();
    let burst_gate_cap = session.workspace.batch_gate.capacity();
    let burst_quads_cap = session.workspace.prepared.quads_capacity();
    assert!(burst_kv_cap > normal_kv_cap * 2);
    assert!(burst_gate_cap > super::RETAIN_POSITIONS * c.intermediate * 2);
    session.reset();
    assert!(session.layers[0].key_capacity() < burst_kv_cap);
    assert!(session.layers[0].value_capacity() < burst_kv_cap);
    assert!(session.workspace.batch_gate.capacity() < burst_gate_cap);
    assert!(session.workspace.batch_up.capacity() < burst_gate_cap);
    assert!(session.workspace.prepared.quads_capacity() < burst_quads_cap);

    // 3. Post-shrink inference matches baseline bits identically.
    let after_logits = session.prefill(&normal_prompt, &normal_audio, &mut || false)?;
    assert_eq!(
        after_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        baseline_logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn rust_asr_decoder_incremental_prefill_matches_full_prefill_and_rolls_back() -> io::Result<()> {
    let model = model()?;
    let prompt = PromptTokens::new([0; 6], [0; 3], 2)?.automatic_prompt(3, 512)?;
    let audio = [1.0, -0.5, 0.25, 0.75, -1.25, 0.5];

    let mut full = model.session();
    let full_logits = full.prefill(&prompt, &audio, &mut || false)?;

    let mut inc = model.session();
    assert!(
        inc.prefill_tokens(&model_prefix(), false, &mut || false)?
            .is_none()
    );
    assert_eq!(inc.position(), 9);
    inc.prefill_audio_chunk(&audio[..4], 2, &mut || false)?;
    assert_eq!(inc.position(), 11);
    inc.prefill_audio_chunk(&audio[4..], 1, &mut || false)?;
    assert_eq!(inc.position(), 12);

    // Inject late error after partial commit attempt: must roll back to position 12.
    let saved_cache = cache_vecs(inc.cache(0));
    assert!(
        inc.prefill_audio_chunk(&[f32::NAN; 2], 1, &mut || false)
            .is_err()
    );
    assert!(inc.prefill_tokens(&[0, 1], false, &mut || true).is_err());
    assert_eq!(inc.position(), 12);
    assert_eq!(cache_vecs(inc.cache(0)), saved_cache);

    let inc_logits = inc
        .prefill_tokens(&model_suffix(), true, &mut || false)?
        .ok_or_else(|| io::Error::other("missing suffix logits"))?;
    assert_eq!(inc.position(), full.position());
    assert_eq!(
        inc_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        full_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    for layer in 0..full.layer_count() {
        let (ik, iv) =
            cache_vecs(inc.cache(layer)).ok_or_else(|| io::Error::other("missing inc layer"))?;
        let (fk, fv) =
            cache_vecs(full.cache(layer)).ok_or_else(|| io::Error::other("missing full layer"))?;
        assert_eq!(
            ik.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fk.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            iv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fv.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    return Ok(());

    fn model_prefix() -> [u32; 9] {
        PromptTokens::new([0; 6], [0; 3], 2)
            .map(PromptTokens::prefix_ids)
            .unwrap_or([0; 9])
    }

    fn model_suffix() -> [u32; 6] {
        PromptTokens::new([0; 6], [0; 3], 2)
            .map(PromptTokens::suffix_ids)
            .unwrap_or([0; 6])
    }
}
