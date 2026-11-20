//! Source-bound CNN/encoder and exact supported frontend metadata intake.

use super::{Index, decoder::Loader, invalid};
use crate::{
    encoder::{Block, Config, Encoder, Linear, Norm, Weights},
    frontend::Frontend,
    protocol::EncoderLayout,
};
use std::io::{self, Read, Seek};

#[expect(
    clippy::float_cmp,
    reason = "Admit only exact fixed frontend metadata values, never approximate a different model frontend."
)]
fn frontend_metadata(index: &Index) -> io::Result<()> {
    for (key, expected) in [
        ("type", "mel"),
        ("window", "hann_periodic"),
        ("normalize", "per_utterance"),
    ] {
        if index.text(&format!("stt.frontend.{key}"))? != expected {
            return Err(invalid());
        }
    }
    for (key, expected) in [
        ("sample_rate", 16_000),
        ("num_mels", 128),
        ("n_fft", 400),
        ("win_length", 400),
        ("hop_length", 160),
    ] {
        if index.integer(&format!("stt.frontend.{key}"))? != expected {
            return Err(invalid());
        }
    }
    for (key, expected) in [
        ("dither", 0.0),
        ("pre_emphasis", 0.0),
        ("f_min", 0.0),
        ("f_max", 8000.0),
    ] {
        if index.float(&format!("stt.frontend.{key}"))? != expected {
            return Err(invalid());
        }
    }
    for (key, expected) in [("pad_mode", "reflect"), ("mel_norm", "slaney")] {
        let full = format!("stt.frontend.{key}");
        if index.metadata.contains_key(&full) && index.text(&full)? != expected {
            return Err(invalid());
        }
    }
    if !index.boolean("stt.frontend.center", true)? {
        return Err(invalid());
    }
    Ok(())
}

fn linear<R: Read + Seek>(
    loader: &mut Loader<'_, R>,
    prefix: &str,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Linear> {
    Ok(Linear {
        weights: loader.projection(&format!("{prefix}.weight"), cancel)?,
        bias: loader.vector(&format!("{prefix}.bias"), cancel)?,
    })
}
fn norm<R: Read + Seek>(
    loader: &mut Loader<'_, R>,
    prefix: &str,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Norm> {
    Ok(Norm {
        weight: loader.vector(&format!("{prefix}.weight"), cancel)?,
        bias: loader.vector(&format!("{prefix}.bias"), cancel)?,
    })
}

pub(super) fn load<R: Read + Seek>(
    index: &Index,
    source: &mut R,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<(Encoder, Frontend)> {
    let mut loader = Loader::new(index, source);
    let (c, layers, frontend, convolutions) = load_encoder_prefix(index, &mut loader, cancel)?;
    let mut blocks = Vec::new();
    blocks.try_reserve_exact(layers).map_err(io::Error::other)?;
    for layer in 0..layers {
        let prefix = format!("enc.blocks.{layer}.");
        blocks.push(Block {
            attention_norm: norm(&mut loader, &(prefix.clone() + "norm_attn"), cancel)?,
            ffn_norm: norm(&mut loader, &(prefix.clone() + "norm_ffn"), cancel)?,
            query: linear(&mut loader, &(prefix.clone() + "attn.q"), cancel)?,
            key: linear(&mut loader, &(prefix.clone() + "attn.k"), cancel)?,
            value: linear(&mut loader, &(prefix.clone() + "attn.v"), cancel)?,
            attention_out: linear(&mut loader, &(prefix.clone() + "attn.out"), cancel)?,
            ffn_in: linear(&mut loader, &(prefix.clone() + "ffn.fc1"), cancel)?,
            ffn_out: linear(&mut loader, &(prefix + "ffn.fc2"), cancel)?,
        });
    }
    let weights = Weights {
        convolutions,
        convolution_out: loader.projection("enc.conv_out.weight", cancel)?,
        blocks,
        output_norm: norm(&mut loader, "enc.ln_post", cancel)?,
        output_in: linear(&mut loader, "enc.proj1", cancel)?,
        output_out: linear(&mut loader, "enc.proj2", cancel)?,
    };
    Ok((Encoder::new(c, weights)?, frontend))
}

#[cfg(all(windows, feature = "rust-qualification"))]
pub(super) fn load_mapped(
    index: &Index,
    mapping: &std::sync::Arc<bonsai_windows_process::MappedFile>,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<(Encoder, Frontend)> {
    use super::decoder::MappedSliceLoader;
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut cursor = io::Cursor::new(mapping.as_slice());
    let mut loader = Loader::new_mapped(index, &mut cursor, mapping);
    let (c, layers, frontend, convolutions) = load_encoder_prefix(index, &mut loader, cancel)?;
    let slice_loader = MappedSliceLoader::new(index, mapping);
    let mut remaining = loader.remaining;
    for layer in 0..layers {
        let prefix = format!("enc.blocks.{layer}.");
        for suffix in [
            "norm_attn.weight",
            "norm_attn.bias",
            "norm_ffn.weight",
            "norm_ffn.bias",
            "attn.q.weight",
            "attn.q.bias",
            "attn.k.weight",
            "attn.k.bias",
            "attn.v.weight",
            "attn.v.bias",
            "attn.out.weight",
            "attn.out.bias",
            "ffn.fc1.weight",
            "ffn.fc1.bias",
            "ffn.fc2.weight",
            "ffn.fc2.bias",
        ] {
            slice_loader.reserve_tensor(&mut remaining, &(prefix.clone() + suffix))?;
        }
    }
    loader.remaining = remaining;
    if cancel() {
        return Err(io::Error::from(io::ErrorKind::Interrupted));
    }
    let linear_mapped = |prefix: &str| -> io::Result<Linear> {
        Ok(Linear {
            weights: slice_loader.projection(&format!("{prefix}.weight"))?,
            bias: slice_loader.vector(&format!("{prefix}.bias"))?,
        })
    };
    let norm_mapped = |prefix: &str| -> io::Result<Norm> {
        Ok(Norm {
            weight: slice_loader.vector(&format!("{prefix}.weight"))?,
            bias: slice_loader.vector(&format!("{prefix}.bias"))?,
        })
    };
    let abort = AtomicBool::new(false);
    let blocks = (0..layers)
        .into_par_iter()
        .map(|layer| {
            if abort.load(Ordering::Acquire) {
                return Err(invalid());
            }
            let prefix = format!("enc.blocks.{layer}.");
            let load_block = || -> io::Result<Block> {
                Ok(Block {
                    attention_norm: norm_mapped(&(prefix.clone() + "norm_attn"))?,
                    ffn_norm: norm_mapped(&(prefix.clone() + "norm_ffn"))?,
                    query: linear_mapped(&(prefix.clone() + "attn.q"))?,
                    key: linear_mapped(&(prefix.clone() + "attn.k"))?,
                    value: linear_mapped(&(prefix.clone() + "attn.v"))?,
                    attention_out: linear_mapped(&(prefix.clone() + "attn.out"))?,
                    ffn_in: linear_mapped(&(prefix.clone() + "ffn.fc1"))?,
                    ffn_out: linear_mapped(&(prefix + "ffn.fc2"))?,
                })
            };
            let res = load_block();
            if res.is_err() {
                abort.store(true, Ordering::Release);
            }
            res
        })
        .collect::<io::Result<Vec<_>>>()?;
    if cancel() {
        return Err(io::Error::from(io::ErrorKind::Interrupted));
    }
    let weights = Weights {
        convolutions,
        convolution_out: loader.projection("enc.conv_out.weight", cancel)?,
        blocks,
        output_norm: norm(&mut loader, "enc.ln_post", cancel)?,
        output_in: linear(&mut loader, "enc.proj1", cancel)?,
        output_out: linear(&mut loader, "enc.proj2", cancel)?,
    };
    Ok((Encoder::new(c, weights)?, frontend))
}

fn load_encoder_prefix<R: Read + Seek>(
    index: &Index,
    loader: &mut Loader<'_, R>,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<(Config, usize, Frontend, [Linear; 3])> {
    frontend_metadata(index)?;
    let count = |suffix| index.count("stt.qwen3_asr.encoder", suffix);
    let c = Config {
        mel_bins: count("num_mel_bins")?,
        channels: count("downsample_hidden")?,
        hidden: count("d_model")?,
        heads: count("n_heads")?,
        intermediate: count("ffn_dim")?,
        output: count("output_dim")?,
        window: count("n_window")?,
        max_positions: count("max_source_positions")?,
    };
    let layers = count("n_layers")?;
    // Reject weight-sized hostile allocations before tensor reads.
    if c.mel_bins != 128
        || !(1..=1024).contains(&c.channels)
        || !(4..=8192).contains(&c.hidden)
        || !c.hidden.is_multiple_of(2)
        || c.heads == 0
        || c.heads > 128
        || !c.hidden.is_multiple_of(c.heads)
        || !(1..=65_536).contains(&c.intermediate)
        || !(1..=8192).contains(&c.output)
        || !(1..=256).contains(&layers)
        || EncoderLayout::new(1, c.window)?.rows_per_chunk > c.max_positions
    {
        return Err(invalid());
    }
    let window = loader.optional_vector("frontend.window", cancel)?;
    let bank = loader.optional_vector("frontend.mel_filterbank", cancel)?;
    let frontend = Frontend::from_checkpoint(window.as_deref(), bank.as_deref())?;
    let mut convolutions = Vec::new();
    for layer in 0..3 {
        convolutions.push(Linear {
            weights: loader.convolution(&format!("enc.conv.{layer}.weight"), cancel)?,
            bias: loader.vector(&format!("enc.conv.{layer}.bias"), cancel)?,
        });
    }
    Ok((
        c,
        layers,
        frontend,
        convolutions.try_into().map_err(|_| invalid())?,
    ))
}
