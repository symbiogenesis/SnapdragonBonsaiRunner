//! Decode admitted source-bound tensor ranges into the independent Rust decoder.

use super::{Index, invalid};
use crate::decoder::{Block, CachePrecision, Config, Model, Projection};
use std::io::{self, Read, Seek, SeekFrom};

pub(super) struct Loader<'a, R> {
    pub(super) index: &'a Index,
    pub(super) source: &'a mut R,
    pub(super) remaining: u64,
}

impl<'a, R: Read + Seek> Loader<'a, R> {
    pub(super) const fn new(index: &'a Index, source: &'a mut R) -> Self {
        Self {
            index,
            source,
            remaining: 8 * 1024 * 1024 * 1024,
        }
    }

    pub(super) fn optional_vector(
        &mut self,
        name: &str,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Option<Vec<f32>>> {
        if self.index.tensors.contains_key(name) {
            self.vector(name, cancel).map(Some)
        } else {
            Ok(None)
        }
    }

    fn count(&self, suffix: &str) -> io::Result<usize> {
        self.index.count("stt.qwen3_asr.decoder", suffix)
    }

    fn scalar(&self, suffix: &str) -> io::Result<f32> {
        self.index.scalar("stt.qwen3_asr.decoder", suffix)
    }

    pub(super) fn bytes(
        &mut self,
        name: &str,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<(u32, Vec<u8>)> {
        let tensor = self.index.tensors.get(name).ok_or_else(invalid)?;
        self.remaining = self
            .remaining
            .checked_sub(tensor.bytes)
            .ok_or_else(invalid)?;
        let count = usize::try_from(tensor.bytes).map_err(|_| invalid())?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(count).map_err(io::Error::other)?;
        self.source.seek(SeekFrom::Start(tensor.offset))?;
        while bytes.len() < count {
            if cancel() {
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            let start = bytes.len();
            let end = count.min(start + 64 * 1024);
            bytes.resize(end, 0);
            self.source.read_exact(&mut bytes[start..end])?;
        }
        Ok((tensor.kind, bytes))
    }

    fn stream_tensor_values(
        &mut self,
        name: &str,
        expected_kind: u32,
        elem_bytes: usize,
        cancel: &mut impl FnMut() -> bool,
        mut decode_chunk: impl FnMut(&[u8], &mut Vec<f32>) -> io::Result<()>,
    ) -> io::Result<Vec<f32>> {
        let tensor = self.index.tensors.get(name).ok_or_else(invalid)?;
        let elem_bytes_u64 = u64::try_from(elem_bytes).map_err(|_| invalid())?;
        if tensor.kind != expected_kind || !tensor.bytes.is_multiple_of(elem_bytes_u64) {
            return Err(invalid());
        }
        self.remaining = self
            .remaining
            .checked_sub(tensor.bytes)
            .ok_or_else(invalid)?;
        let byte_count = usize::try_from(tensor.bytes).map_err(|_| invalid())?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(byte_count / elem_bytes)
            .map_err(io::Error::other)?;
        self.source.seek(SeekFrom::Start(tensor.offset))?;
        let mut scratch = [0_u8; 64 * 1024];
        let max_step = (scratch.len() / elem_bytes) * elem_bytes;
        let mut read_bytes = 0_usize;
        while read_bytes < byte_count {
            if cancel() {
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            let step = (byte_count - read_bytes).min(max_step);
            self.source.read_exact(&mut scratch[..step])?;
            decode_chunk(&scratch[..step], &mut values)?;
            read_bytes += step;
        }
        Ok(values)
    }

    pub(super) fn vector(
        &mut self,
        name: &str,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Vec<f32>> {
        self.stream_tensor_values(name, 0, 4, cancel, |chunk, values| {
            for &raw in chunk.as_chunks::<4>().0 {
                values.push(f32::from_le_bytes(raw));
            }
            Ok(())
        })
    }

    fn f16_vector(
        &mut self,
        name: &str,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Vec<f32>> {
        self.stream_tensor_values(name, 1, 2, cancel, |chunk, values| {
            for &raw in chunk.as_chunks::<2>().0 {
                values.push(decode_f16(u16::from_le_bytes(raw))?);
            }
            Ok(())
        })
    }

    pub(super) fn projection(
        &mut self,
        name: &str,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Projection> {
        let tensor = self.index.tensors.get(name).ok_or_else(invalid)?;
        let input = usize::try_from(tensor.shape[0]).map_err(|_| invalid())?;
        // GGML fills omitted trailing dimensions with one. The common model
        // catalog admits that convention, so loading must preserve it too.
        let output =
            usize::try_from(tensor.shape.get(1).copied().unwrap_or(1)).map_err(|_| invalid())?;
        if tensor.shape.iter().skip(2).any(|axis| *axis != 1) {
            return Err(invalid());
        }
        match tensor.kind {
            0 => Projection::f32(input, output, self.vector(name, cancel)?),
            1 => Projection::f32(input, output, self.f16_vector(name, cancel)?),
            8 => Projection::q8(input, output, self.bytes(name, cancel)?.1),
            _ => Err(invalid()),
        }
    }

    pub(super) fn convolution(
        &mut self,
        name: &str,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Projection> {
        let tensor = self.index.tensors.get(name).ok_or_else(invalid)?;
        if tensor.shape.first() != Some(&3)
            || tensor.shape.get(1) != Some(&3)
            || tensor.shape.iter().skip(4).any(|axis| *axis != 1)
        {
            return Err(invalid());
        }
        let input = usize::try_from(
            9_u64
                .checked_mul(tensor.shape.get(2).copied().unwrap_or(1))
                .ok_or_else(invalid)?,
        )
        .map_err(|_| invalid())?;
        let output =
            usize::try_from(tensor.shape.get(3).copied().unwrap_or(1)).map_err(|_| invalid())?;
        match tensor.kind {
            0 => Projection::f32(input, output, self.vector(name, cancel)?),
            1 => Projection::f32(input, output, self.f16_vector(name, cancel)?),
            _ => Err(invalid()),
        }
    }
}

fn decode_f16(bits: u16) -> io::Result<f32> {
    let value = half::f16::from_bits(bits);
    if !value.is_finite() {
        return Err(invalid());
    }
    Ok(value.to_f32())
}

pub(super) fn load<R: Read + Seek>(
    index: &Index,
    source: &mut R,
    precision: CachePrecision,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Model> {
    let mut loader = Loader::new(index, source);
    let config = Config {
        hidden: loader.count("hidden_size")?,
        heads: loader.count("n_heads")?,
        kv_heads: loader.count("n_kv_heads")?,
        head_dim: loader.count("head_dim")?,
        intermediate: loader.count("intermediate_size")?,
        context: loader.count("max_position_embeddings")?.min(8192),
        epsilon: loader.scalar("rms_norm_eps")?,
        rope_theta: loader.scalar("rope_theta")?,
        cache_precision: precision,
    };
    let count = loader.count("n_layers")?;
    // Validate geometry before any weight-sized allocation; Model::new checks
    // each admitted layer again after the complete tensor set is loaded.
    if !(1..=128).contains(&count)
        || !(1..=8192).contains(&config.hidden)
        || !(1..=128).contains(&config.heads)
        || config.kv_heads == 0
        || !config.heads.is_multiple_of(config.kv_heads)
        || config.head_dim == 0
        || config.head_dim > 256
        || !config.head_dim.is_multiple_of(2)
        || !(1..=65_536).contains(&config.intermediate)
        || config.context == 0
        || !config.epsilon.is_finite()
        || config.epsilon <= 0.0
        || !config.rope_theta.is_finite()
        || config.rope_theta <= 0.0
    {
        return Err(invalid());
    }
    let embedding = loader.projection("dec.token_embd.weight", cancel)?;
    let output_norm = loader.vector("dec.output_norm.weight", cancel)?;
    let mut blocks = Vec::new();
    blocks.try_reserve_exact(count).map_err(io::Error::other)?;
    for layer in 0..count {
        let prefix = format!("dec.blocks.{layer}.");
        blocks.push(Block {
            attention_norm: loader.vector(&(prefix.clone() + "norm_attn.weight"), cancel)?,
            ffn_norm: loader.vector(&(prefix.clone() + "norm_ffn.weight"), cancel)?,
            query: loader.projection(&(prefix.clone() + "attn.q.weight"), cancel)?,
            key: loader.projection(&(prefix.clone() + "attn.k.weight"), cancel)?,
            value: loader.projection(&(prefix.clone() + "attn.v.weight"), cancel)?,
            attention_out: loader.projection(&(prefix.clone() + "attn.o.weight"), cancel)?,
            query_norm: loader.vector(&(prefix.clone() + "attn.q_norm.weight"), cancel)?,
            key_norm: loader.vector(&(prefix.clone() + "attn.k_norm.weight"), cancel)?,
            gate: loader.projection(&(prefix.clone() + "ffn.gate.weight"), cancel)?,
            up: loader.projection(&(prefix.clone() + "ffn.up.weight"), cancel)?,
            down: loader.projection(&(prefix + "ffn.down.weight"), cancel)?,
        });
    }
    Model::new(config, embedding, blocks, output_norm)
}

#[cfg(test)]
mod tests {
    use super::{Index, Loader, Projection};
    use crate::intake::Tensor;
    use std::{
        collections::BTreeMap,
        io::{self, Cursor},
    };

    #[test]
    fn rust_asr_decoder_wire_weight_ranges_and_storage_types() -> io::Result<()> {
        let mut bytes = vec![0; 128];
        bytes[8..24].copy_from_slice(
            &[1.0_f32, -2.0, 3.0, 4.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        bytes[32..40].copy_from_slice(
            &[0x3c00_u16, 0xc000, 0x4200, 0x4400]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        bytes[64..66].copy_from_slice(&0x3800_u16.to_le_bytes());
        bytes[66..98].fill(2);
        let mut tensors = BTreeMap::new();
        for (name, kind, offset, count, shape) in [
            ("float", 0, 8, 16, vec![2, 2]),
            ("half", 1, 32, 8, vec![2, 2]),
            ("quant", 8, 64, 34, vec![32, 1]),
        ] {
            tensors.insert(
                name.into(),
                Tensor {
                    shape,
                    kind,
                    offset,
                    bytes: count,
                },
            );
        }
        let mut source = Cursor::new(bytes);
        let index = Index {
            metadata: BTreeMap::new(),
            tensors,
        };
        let mut loader = Loader {
            index: &index,
            source: &mut source,
            remaining: 128,
        };
        for name in ["float", "half"] {
            let projection = loader.projection(name, &mut || false)?;
            assert_eq!(
                projection.multiply(&[5.0, 6.0], &mut || false)?,
                [-7.0, 39.0]
            );
        }
        assert_eq!(
            loader
                .projection("quant", &mut || false)?
                .multiply(&[1.0; 32], &mut || false)?,
            [32.0]
        );
        assert!(loader.projection("missing", &mut || false).is_err());
        assert!(loader.projection("quant", &mut || true).is_err());
        assert!(Projection::f16(1, 1, &[0]).is_err());
        Ok(())
    }

    fn one_tensor(shape: &[u64], kind: u32, bytes: u64) -> Index {
        Index {
            metadata: BTreeMap::new(),
            tensors: BTreeMap::from([(
                "weight".into(),
                Tensor {
                    shape: shape.to_vec(),
                    kind,
                    offset: 0,
                    bytes,
                },
            )]),
        }
    }

    #[test]
    fn rust_asr_decoder_wire_projection_preserves_omitted_unit_axes() -> io::Result<()> {
        for shape in [&[2][..], &[2, 1], &[2, 1, 1], &[2, 1, 1, 1]] {
            for (kind, bytes) in [
                (
                    0,
                    [1.0_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat(),
                ),
                (
                    1,
                    [0x3c00_u16.to_le_bytes(), 0xc000_u16.to_le_bytes()].concat(),
                ),
            ] {
                let length = u64::try_from(bytes.len()).map_err(|_| super::invalid())?;
                let mut source = Cursor::new(bytes);
                let index = one_tensor(shape, kind, length);
                let mut loader = Loader {
                    index: &index,
                    source: &mut source,
                    remaining: length,
                };
                let projection = loader.projection("weight", &mut || false)?;
                assert_eq!(projection.dimensions(), (2, 1));
                assert_eq!(projection.multiply(&[5.0, 6.0], &mut || false)?, [-7.0]);
            }
        }
        for shape in [&[32][..], &[32, 1], &[32, 1, 1, 1]] {
            // Half scale .5 and signed coefficients 1,-1,127,-128 sum to -1.
            let mut bytes = vec![0; 34];
            bytes[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
            bytes[2..6].copy_from_slice(&[1, 255, 127, 128]);
            let mut source = Cursor::new(bytes);
            let index = one_tensor(shape, 8, 34);
            let mut loader = Loader {
                index: &index,
                source: &mut source,
                remaining: 34,
            };
            assert_eq!(
                loader
                    .projection("weight", &mut || false)?
                    .multiply(&[1.0; 32], &mut || false)?,
                [-0.5]
            );
        }
        // A genuinely nonunit third axis is not a matrix and must fail before I/O.
        let mut source = Cursor::new(Vec::<u8>::new());
        let index = one_tensor(&[2, 1, 2], 0, 16);
        let mut loader = Loader {
            index: &index,
            source: &mut source,
            remaining: 16,
        };
        assert!(loader.projection("weight", &mut || false).is_err());
        assert_eq!(source.position(), 0);
        Ok(())
    }

    #[test]
    fn rust_asr_decoder_wire_convolution_preserves_omitted_unit_axes() -> io::Result<()> {
        for shape in [&[3, 3][..], &[3, 3, 1], &[3, 3, 1, 1]] {
            for (kind, bytes) in [
                (
                    0,
                    [1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                ),
                (
                    1,
                    [
                        0x3c00_u16, 0xc000, 0x4200, 0xc400, 0x4500, 0xc600, 0x4700, 0xc800, 0x4880,
                    ]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
                ),
            ] {
                let length = u64::try_from(bytes.len()).map_err(|_| super::invalid())?;
                let mut source = Cursor::new(bytes);
                let index = one_tensor(shape, kind, length);
                let mut loader = Loader {
                    index: &index,
                    source: &mut source,
                    remaining: length,
                };
                let projection = loader.convolution("weight", &mut || false)?;
                assert_eq!(projection.dimensions(), (9, 1));
                assert_eq!(projection.multiply(&[1.0; 9], &mut || false)?, [5.0]);
            }
        }
        let mut source = Cursor::new(Vec::<u8>::new());
        let index = one_tensor(&[3], 0, 12);
        let mut loader = Loader {
            index: &index,
            source: &mut source,
            remaining: 12,
        };
        assert!(loader.convolution("weight", &mut || false).is_err());
        assert_eq!(source.position(), 0);
        Ok(())
    }
}
