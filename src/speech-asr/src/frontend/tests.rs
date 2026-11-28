//! Mathematical controls; these are not captured native inference observations.

use super::*;

fn frontend() -> io::Result<Frontend> {
    static CACHED: std::sync::OnceLock<Frontend> = std::sync::OnceLock::new();
    if let Some(frontend) = CACHED.get() {
        return Ok(frontend.clone());
    }
    let frontend = Frontend::from_checkpoint(None, None)?;
    Ok(CACHED.get_or_init(|| frontend).clone())
}

#[test]
fn rust_asr_frontend_silence_has_exact_whisper_floor_and_geometry() -> io::Result<()> {
    let frontend = frontend()?;
    for samples in [201, 319, 320, 479, 480] {
        let features = frontend.compute(&vec![0.0; samples], || false)?;
        assert_eq!(features.frames(), samples / 160);
        assert_eq!(features.values().len(), 128 * (samples / 160));
        // power=0 -> log10(1e-10)=-10 -> (-10+4)/4=-1.5.
        assert!(
            features
                .values()
                .iter()
                .all(|value| value.to_bits() == (-1.5_f32).to_bits())
        );
    }
    Ok(())
}

#[test]
fn rust_asr_frontend_rejects_invalid_pcm_tables_and_cancellation() -> io::Result<()> {
    let frontend = frontend()?;
    for samples in [0, 1, 160, 200, MAX_SAMPLES + 1] {
        assert!(frontend.compute(&vec![0.0; samples], || false).is_err());
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.01, -1.01] {
        let mut pcm = vec![0.0; 201];
        pcm[100] = invalid;
        assert!(frontend.compute(&pcm, || false).is_err());
    }
    let interrupted = frontend
        .compute(&[0.0; 201], || true)
        .err()
        .map(|error| error.kind());
    assert_eq!(interrupted, Some(io::ErrorKind::Interrupted));
    // Cancellation after the first frame cannot publish partial features or
    // contaminate a later call on the immutable frontend.
    let mut checks = 0;
    let interrupted = frontend
        .compute(&[0.0; 640], || {
            checks += 1;
            checks >= 5
        })
        .err()
        .map(|error| error.kind());
    assert_eq!(interrupted, Some(io::ErrorKind::Interrupted));
    assert!(frontend.compute(&[0.0; 201], || false).is_ok());
    for supplied in [
        vec![0.0; FFT_SIZE],
        vec![0.5; FFT_SIZE - 1],
        vec![f32::NAN; FFT_SIZE],
        vec![2.0; FFT_SIZE],
    ] {
        assert!(Frontend::from_checkpoint(Some(&supplied), None).is_err());
    }
    for supplied in [
        vec![0.0; FILTER_ELEMENTS],
        vec![0.5; FILTER_ELEMENTS - 1],
        vec![f32::INFINITY; FILTER_ELEMENTS],
        vec![-1.0; FILTER_ELEMENTS],
    ] {
        assert!(Frontend::from_checkpoint(None, Some(&supplied)).is_err());
    }
    Ok(())
}

#[test]
fn rust_asr_frontend_reflection_excludes_edge_samples() {
    let samples: Vec<_> = (0_u16..201).map(f32::from).collect();
    let mut padded = vec![0.0; samples.len() + 2 * PAD];
    padded[PAD..PAD + samples.len()].copy_from_slice(&samples);
    reflect_edges(&mut padded, samples.len());
    assert_eq!(&padded[..4], &[200.0, 199.0, 198.0, 197.0]);
    assert_eq!(&padded[PAD - 2..PAD + 3], &[2.0, 1.0, 0.0, 1.0, 2.0]);
    assert_eq!(
        &padded[PAD + samples.len() - 2..PAD + samples.len() + 3],
        &[199.0, 200.0, 199.0, 198.0, 197.0]
    );
    assert_eq!(padded.last().copied(), Some(0.0));
}

#[test]
fn rust_asr_frontend_tables_preserve_periodic_hann_and_checkpoint_bits() -> io::Result<()> {
    let generated = frontend()?;
    // N=400: w[0]=0, w[100]=1/2, w[200]=1, w[300]=1/2;
    // the last sample is nonzero, distinguishing periodic from symmetric Hann.
    assert_eq!(generated.window[0], 0.0);
    for (index, expected) in [(100, 0.5), (200, 1.0), (300, 0.5)] {
        assert!((generated.window[index] - expected).abs() < 1.0e-6);
    }
    assert!(generated.window[399] > 0.0);
    assert!((hz_to_mel(1000.0) - 15.0).abs() < 1.0e-12);
    assert!((mel_to_hz(42.0) - 6400.0).abs() < 1.0e-9);
    assert!(
        generated
            .filterbank
            .as_chunks::<FREQUENCIES>()
            .0
            .iter()
            .all(|row| row[0] == 0.0 && row[FREQUENCIES - 1].abs() < 1.0e-12)
    );
    let window = [0.25; FFT_SIZE];
    let bank = vec![0.5; FILTER_ELEMENTS];
    let provided = Frontend::from_checkpoint(Some(&window), Some(&bank))?;
    assert_eq!(provided.window, window);
    assert_eq!(provided.filterbank, bank);
    Ok(())
}

#[test]
fn rust_asr_frontend_fft_matches_independent_direct_dft_and_impulse() -> io::Result<()> {
    let frontend = frontend()?;
    let mut input = [0.0; 2 * FFT_SIZE];
    let mut output = [0.0; 8 * FFT_SIZE];
    input[0] = 1.0;
    mixed_fft(
        &mut input,
        FFT_SIZE,
        &frontend.cos,
        &frontend.sin,
        &mut output,
    );
    for complex in output[..2 * FFT_SIZE].as_chunks::<2>().0 {
        assert_eq!(complex[0], 1.0);
        assert_eq!(complex[1], 0.0);
    }
    // Sparse off-origin impulses exercise every mixed-radix split and odd leaf.
    // The reference evaluates the DFT definition in f64 with fresh trig values,
    // independently of the implementation's twiddle table and decomposition.
    let spikes = [(1, 0.75_f32), (47, -0.5), (200, 0.25), (399, -0.125)];
    input.fill(0.0);
    for (index, value) in spikes {
        input[index] = value;
    }
    mixed_fft(
        &mut input,
        FFT_SIZE,
        &frontend.cos,
        &frontend.sin,
        &mut output,
    );
    for bin in 0..FFT_SIZE {
        let mut real = 0.0;
        let mut imaginary = 0.0;
        for (index, value) in spikes {
            let phase =
                2.0 * std::f64::consts::PI * small_integer(bin) * small_integer(index) / 400.0;
            real += f64::from(value) * phase.cos();
            imaginary -= f64::from(value) * phase.sin();
        }
        assert!((f64::from(output[2 * bin]) - real).abs() < 1.0e-6);
        assert!((f64::from(output[2 * bin + 1]) - imaginary).abs() < 1.0e-6);
    }
    Ok(())
}

#[test]
fn rust_asr_frontend_normalization_uses_global_emitted_maximum() -> io::Result<()> {
    // Maximum 2 gives floor -6 across every row/frame. Frame omission occurs
    // before this pass; there is no extra trailing value in the max's scope.
    let mut values = [-10.0, -6.0, -2.0, 2.0, 0.0];
    normalize(&mut values, &mut || false)?;
    assert_eq!(values, [-0.5, -0.5, 0.5, 1.5, 1.0]);
    assert!(normalize(&mut [f32::NAN], &mut || false).is_err());
    Ok(())
}

#[test]
fn rust_asr_frontend_checkpoint_tables_drive_full_mel_major_pipeline() -> io::Result<()> {
    // A constant window and waveform yield only DC: FFT[0]=400*0.5=200,
    // power=40,000. Row zero's DC weight is one and every other row's
    // weight is 100. log10 powers are 4.60205999 and 6.60205999, hence
    // normalized values 2.150515 and 2.650515. Both are above max-8.
    let window = [0.5; FFT_SIZE];
    let mut bank = vec![0.0; FILTER_ELEMENTS];
    for (mel, row) in bank.as_chunks_mut::<FREQUENCIES>().0.iter_mut().enumerate() {
        row[0] = if mel == 0 { 1.0 } else { 100.0 };
    }
    let frontend = Frontend::from_checkpoint(Some(&window), Some(&bank))?;
    let features = frontend.compute(&[1.0; 480], || false)?;
    assert_eq!(features.frames(), 3);
    assert_eq!(features.values().len(), 384);
    assert!(
        features.values()[..3]
            .iter()
            .all(|value| (*value - 2.150_515).abs() < 5.0e-7)
    );
    assert!(
        features.values()[3..]
            .iter()
            .all(|value| (*value - 2.650_515).abs() < 5.0e-7)
    );
    Ok(())
}

#[test]
fn rust_asr_frontend_four_frame_simd_and_parallel_match_scalar_bits() -> io::Result<()> {
    let frontend = frontend()?;
    for samples in [201, 640, 800, 1440, 2720] {
        let mut pcm = Vec::with_capacity(samples);
        for idx in 0..samples {
            let t = f64::from(u16::try_from(idx).map_err(|_| invalid())?);
            let sample = (0.45 * (t * 0.071).sin() + 0.35 * (t * 0.293).cos()).clamp(-1.0, 1.0);
            pcm.push(rounded(sample));
        }
        let mut padded = zeroes(pcm.len() + 2 * PAD)?;
        padded[PAD..PAD + pcm.len()].copy_from_slice(&pcm);
        reflect_edges(&mut padded, pcm.len());
        let frames = pcm.len() / HOP_LENGTH;
        let mut scalar_values = zeroes(MEL_BINS * frames)?;
        let mut s_input = [0.0_f32; 2 * FFT_SIZE];
        let mut s_output = [0.0_f32; 8 * FFT_SIZE];
        let mut s_power = [0.0_f32; FREQUENCIES];
        let mut s_frame = [0.0_f32; MEL_BINS];
        for frame in 0..frames {
            compute_single_frame(
                &frontend,
                &padded,
                frame,
                &mut s_input,
                &mut s_output,
                &mut s_power,
                &mut s_frame,
            );
            for mel in 0..MEL_BINS {
                scalar_values[mel * frames + frame] = s_frame[mel];
            }
        }
        normalize(&mut scalar_values, &mut || false)?;

        for workers in [8, 4, 2, 1] {
            let actual = frontend.compute_with_threads(&pcm, workers, || false)?;
            assert_eq!(actual.frames(), frames);
            assert_eq!(actual.values().len(), scalar_values.len());
            for (index, (got, want)) in actual.values().iter().zip(&scalar_values).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "frontend bit mismatch for samples={samples}, workers={workers}, element={index}"
                );
            }
        }
    }
    assert_eq!(
        frontend
            .compute_with_threads(&[0.0; 201], 0, || false)
            .err()
            .map(|error| error.kind()),
        Some(io::ErrorKind::InvalidInput)
    );
    Ok(())
}
