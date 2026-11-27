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
    let mut block = Block {
        attention_norm: norm(loader, &(prefix.clone() + "norm_attn"))?,
        ffn_norm: norm(loader, &(prefix.clone() + "norm_ffn"))?,
        query: linear(loader, &(prefix.clone() + "attn.q"))?,
        key: linear(loader, &(prefix.clone() + "attn.k"))?,
        value: linear(loader, &(prefix.clone() + "attn.v"))?,
        attention_out: linear(loader, &(prefix.clone() + "attn.out"))?,
        ffn_in: linear(loader, &(prefix.clone() + "ffn.fc1"))?,
        ffn_out: linear(loader, &(prefix + "ffn.fc2"))?,
    };
    block.repack_q4_k_if_aligned();
    Ok(block)
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
    use crate::decoder::Projection;

    if cancel() {
        return Err(io::Error::from(io::ErrorKind::Interrupted));
    }
    let (c, layers) = validate_encoder_geometry(index)?;
    let slice_loader = MappedSliceLoader::new(index, mapping);
    let mut remaining: u64 = 8 * 1024 * 1024 * 1024;
    for opt in ["frontend.window", "frontend.mel_filterbank"] {
        if index.tensors.contains_key(opt) {
            slice_loader.reserve_tensor(&mut remaining, opt)?;
        }
    }
    for conv_idx in 0..3 {
        slice_loader.reserve_tensor(&mut remaining, &format!("enc.conv.{conv_idx}.weight"))?;
        slice_loader.reserve_tensor(&mut remaining, &format!("enc.conv.{conv_idx}.bias"))?;
    }
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
    for tail_name in [
        "enc.conv_out.weight",
        "enc.ln_post.weight",
        "enc.ln_post.bias",
        "enc.proj1.weight",
        "enc.proj1.bias",
        "enc.proj2.weight",
        "enc.proj2.bias",
    ] {
        slice_loader.reserve_tensor(&mut remaining, tail_name)?;
    }

    let load_conv = |conv_idx: usize| -> io::Result<Linear> {
        Ok(Linear {
            weights: slice_loader.convolution(&format!("enc.conv.{conv_idx}.weight"))?,
            bias: slice_loader.vector(&format!("enc.conv.{conv_idx}.bias"))?,
        })
    };
    let load_frontend = || -> io::Result<Frontend> {
        let window = slice_loader.optional_vector("frontend.window")?;
        let bank = slice_loader.optional_vector("frontend.mel_filterbank")?;
        Frontend::from_checkpoint(window.as_deref(), bank.as_deref())
    };
    let load_convolutions = || -> io::Result<[Linear; 3]> {
        let (c0_res, c1_res, c2_res) = std::thread::scope(|scope| {
            let h1 = scope.spawn(|| load_conv(1));
            let h2 = scope.spawn(|| load_conv(2));
            let c0_res = load_conv(0);
            let c1_res = h1.join().unwrap_or_else(|_| {
                Err(io::Error::other("ASR encoder conv intake thread panicked."))
            });
            let c2_res = h2.join().unwrap_or_else(|_| {
                Err(io::Error::other("ASR encoder conv intake thread panicked."))
            });
            (c0_res, c1_res, c2_res)
        });
        Ok([c0_res?, c1_res?, c2_res?])
    };
    let load_tail = || -> io::Result<(Projection, Norm, Linear, Linear)> {
        let (conv_out_res, ln_post_res, proj1_res, proj2_res) = std::thread::scope(|scope| {
            let conv_handle = scope.spawn(|| slice_loader.projection("enc.conv_out.weight"));
            let ln_handle = scope.spawn(|| {
                let mut n_loader = &slice_loader;
                norm(&mut n_loader, "enc.ln_post")
            });
            let p1_handle = scope.spawn(|| {
                let mut p1_loader = &slice_loader;
                linear(&mut p1_loader, "enc.proj1")
            });
            let mut p2_loader = &slice_loader;
            let proj2_res = linear(&mut p2_loader, "enc.proj2");
            let conv_out_res = conv_handle.join().unwrap_or_else(|_| {
                Err(io::Error::other(
                    "ASR encoder conv_out intake thread panicked.",
                ))
            });
            let ln_post_res = ln_handle.join().unwrap_or_else(|_| {
                Err(io::Error::other(
                    "ASR encoder ln_post intake thread panicked.",
                ))
            });
            let proj1_res = p1_handle.join().unwrap_or_else(|_| {
                Err(io::Error::other(
                    "ASR encoder proj1 intake thread panicked.",
                ))
            });
            (conv_out_res, ln_post_res, proj1_res, proj2_res)
        });
        Ok((conv_out_res?, ln_post_res?, proj1_res?, proj2_res?))
    };

    let (frontend_res, conv_res, blocks_res, tail_res) = std::thread::scope(|scope| {
        let frontend_handle = scope.spawn(load_frontend);
        let conv_handle = scope.spawn(load_convolutions);
        let tail_handle = scope.spawn(load_tail);
        let blocks_res = load_mapped_blocks(layers, &mut || false, |layer| {
            let mut s = &slice_loader;
            load_encoder_block(&mut s, layer)
        });
        let frontend_res = frontend_handle.join().unwrap_or_else(|_| {
            Err(io::Error::other(
                "ASR encoder frontend intake thread panicked.",
            ))
        });
        let conv_res = conv_handle.join().unwrap_or_else(|_| {
            Err(io::Error::other(
                "ASR encoder convolution intake thread panicked.",
            ))
        });
        let tail_res = tail_handle
            .join()
            .unwrap_or_else(|_| Err(io::Error::other("ASR encoder tail intake thread panicked.")));
        (frontend_res, conv_res, blocks_res, tail_res)
    });

    if cancel() {
        return Err(io::Error::from(io::ErrorKind::Interrupted));
    }
    let frontend = frontend_res?;
    let convolutions = conv_res?;
    let blocks = blocks_res?;
    let (convolution_out, output_norm, output_in, output_out) = tail_res?;
    let weights = Weights {
        convolutions,
        convolution_out,
        blocks,
        output_norm,
        output_in,
        output_out,
    };
    Ok((Encoder::new(c, weights)?, frontend))
}

fn validate_encoder_geometry(index: &Index) -> io::Result<(Config, usize)> {
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
    Ok((c, layers))
}

fn load_encoder_prefix<R: Read + Seek>(
    index: &Index,
    loader: &mut Loader<'_, R>,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<(Config, usize, Frontend, [Linear; 3])> {
    let (c, layers) = validate_encoder_geometry(index)?;
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
