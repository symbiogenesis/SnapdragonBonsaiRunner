//! Source-bound CNN/encoder and exact supported frontend metadata intake.

use super::{
    Index,
    decoder::{BlockLoader, CancellableLoader, Loader},
    invalid,
};
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

fn linear(loader: &mut impl BlockLoader, prefix: &str) -> io::Result<Linear> {
    Ok(Linear {
        weights: loader.projection(&format!("{prefix}.weight"))?,
        bias: loader.vector(&format!("{prefix}.bias"))?,
    })
}

fn norm(loader: &mut impl BlockLoader, prefix: &str) -> io::Result<Norm> {
    Ok(Norm {
        weight: loader.vector(&format!("{prefix}.weight"))?,
        bias: loader.vector(&format!("{prefix}.bias"))?,
    })
}

fn load_encoder_block(loader: &mut impl BlockLoader, layer: usize) -> io::Result<Block> {
    let prefix = format!("enc.blocks.{layer}.");
    Ok(Block {
        attention_norm: norm(loader, &(prefix.clone() + "norm_attn"))?,
        ffn_norm: norm(loader, &(prefix.clone() + "norm_ffn"))?,
        query: linear(loader, &(prefix.clone() + "attn.q"))?,
        key: linear(loader, &(prefix.clone() + "attn.k"))?,
        value: linear(loader, &(prefix.clone() + "attn.v"))?,
        attention_out: linear(loader, &(prefix.clone() + "attn.out"))?,
        ffn_in: linear(loader, &(prefix.clone() + "ffn.fc1"))?,
        ffn_out: linear(loader, &(prefix + "ffn.fc2"))?,
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
    let mut block_loader = CancellableLoader::new(&mut loader, cancel);
    for layer in 0..layers {
        blocks.push(load_encoder_block(&mut block_loader, layer)?);
    }
    let convolution_out = block_loader.projection("enc.conv_out.weight")?;
    let weights = Weights {
        convolutions,
        convolution_out,
        blocks,
        output_norm: norm(&mut block_loader, "enc.ln_post")?,
        output_in: linear(&mut block_loader, "enc.proj1")?,
        output_out: linear(&mut block_loader, "enc.proj2")?,
    };
    Ok((Encoder::new(c, weights)?, frontend))
}

#[cfg(all(windows, feature = "rust-qualification"))]
pub(super) fn load_mapped(
    index: &Index,
    mapping: &std::sync::Arc<bonsai_windows_process::MappedFile>,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<(Encoder, Frontend)> {
    use super::decoder::{MappedSliceLoader, load_mapped_blocks};

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
    let blocks = load_mapped_blocks(layers, cancel, |layer| {
        let mut s = &slice_loader;
        load_encoder_block(&mut s, layer)
    })?;
    let mut tail_loader = CancellableLoader::new(&mut loader, cancel);
    let weights = Weights {
        convolutions,
        convolution_out: tail_loader.projection("enc.conv_out.weight")?,
        blocks,
        output_norm: norm(&mut tail_loader, "enc.ln_post")?,
        output_in: linear(&mut tail_loader, "enc.proj1")?,
        output_out: linear(&mut tail_loader, "enc.proj2")?,
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
