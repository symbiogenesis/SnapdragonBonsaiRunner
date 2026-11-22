//! Decode admitted source-bound tensor ranges into the independent Rust decoder.

use super::{Index, invalid};
use crate::decoder::{Block, CachePrecision, Config, Model, Projection, WeightBytes};
use std::io::{self, Read, Seek, SeekFrom};

pub(super) trait BlockLoader {
    fn vector(&mut self, name: &str) -> io::Result<Vec<f32>>;
    fn projection(&mut self, name: &str) -> io::Result<Projection>;
}

pub(super) struct CancellableLoader<'a, 'b, R, C> {
    loader: &'a mut Loader<'b, R>,
    cancel: &'a mut C,
}

impl<'a, 'b, R: Read + Seek, C: FnMut() -> bool> CancellableLoader<'a, 'b, R, C> {
    pub(super) const fn new(loader: &'a mut Loader<'b, R>, cancel: &'a mut C) -> Self {
        Self { loader, cancel }
    }
}

impl<R: Read + Seek, C: FnMut() -> bool> BlockLoader for CancellableLoader<'_, '_, R, C> {
    fn vector(&mut self, name: &str) -> io::Result<Vec<f32>> {
        self.loader.vector(name, self.cancel)
    }

    fn projection(&mut self, name: &str) -> io::Result<Projection> {
        self.loader.projection(name, self.cancel)
    }
}

pub(super) fn decode_f16_bytes_into(bytes: &[u8], values: &mut Vec<f32>) -> io::Result<()> {
    let (chunks8, tail) = bytes.as_chunks::<16>();
    for raw_chunk in chunks8 {
        let pairs = raw_chunk.as_chunks::<2>().0;
        let bits: [u16; 8] = std::array::from_fn(|idx| u16::from_le_bytes(pairs[idx]));
        let [lo, hi] = bonsai_neon_dot::widen_f16_eight(&bits);
        for value in lo.into_iter().chain(hi) {
            if !value.is_finite() {
                return Err(invalid());
            }
            values.push(value);
        }
    }
    for &raw in tail.as_chunks::<2>().0 {
        values.push(decode_f16(u16::from_le_bytes(raw))?);
    }
    Ok(())
}

fn projection_dims(index: &Index, name: &str) -> io::Result<(usize, usize, u32)> {
    let tensor = index.tensors.get(name).ok_or_else(invalid)?;
    let input = usize::try_from(tensor.shape[0]).map_err(|_| invalid())?;
    // GGML fills omitted trailing dimensions with one. The common model
    // catalog admits that convention, so loading must preserve it too.
    let output =
        usize::try_from(tensor.shape.get(1).copied().unwrap_or(1)).map_err(|_| invalid())?;
    if tensor.shape.iter().skip(2).any(|axis| *axis != 1) {
        return Err(invalid());
    }
    Ok((input, output, tensor.kind))
}

#[cfg(all(windows, feature = "rust-qualification"))]
fn mapped_tensor_range(tensor: &super::Tensor, mapping_len: usize) -> io::Result<(usize, usize)> {
    let offset = usize::try_from(tensor.offset).map_err(|_| invalid())?;
    let len = usize::try_from(tensor.bytes).map_err(|_| invalid())?;
    let end = offset.checked_add(len).ok_or_else(invalid)?;
    if end > mapping_len {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
    }
    Ok((offset, len))
}

#[cfg(all(windows, feature = "rust-qualification"))]
pub(super) fn load_mapped_blocks<T: Send>(
    count: usize,
    cancel: &mut impl FnMut() -> bool,
    load_layer: impl Fn(usize) -> io::Result<T> + Sync,
) -> io::Result<Vec<T>> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    if cancel() {
        return Err(io::Error::from(io::ErrorKind::Interrupted));
    }
    let abort = AtomicBool::new(false);
    let blocks = (0..count)
        .into_par_iter()
        .map(|layer| {
            if abort.load(Ordering::Acquire) {
                return Err(invalid());
            }
            let res = load_layer(layer);
            if res.is_err() {
                abort.store(true, Ordering::Release);
            }
            res
        })
        .collect::<io::Result<Vec<_>>>()?;
    if cancel() {
        return Err(io::Error::from(io::ErrorKind::Interrupted));
    }
    Ok(blocks)
}

pub(super) struct Loader<'a, R> {
    pub(super) index: &'a Index,
    pub(super) source: &'a mut R,
    pub(super) remaining: u64,
    #[cfg(all(windows, feature = "rust-qualification"))]
    pub(super) mapping: Option<&'a std::sync::Arc<bonsai_windows_process::MappedFile>>,
}

impl<'a, R: Read + Seek> Loader<'a, R> {
    pub(super) const fn new(index: &'a Index, source: &'a mut R) -> Self {
        Self::with_limit(index, source, 8 * 1024 * 1024 * 1024)
    }

    const fn with_limit(index: &'a Index, source: &'a mut R, remaining: u64) -> Self {
        Self {
            index,
            source,
            remaining,
            #[cfg(all(windows, feature = "rust-qualification"))]
            mapping: None,
        }
    }

    #[cfg(all(windows, feature = "rust-qualification"))]
    pub(super) const fn new_mapped(
        index: &'a Index,
        source: &'a mut R,
        mapping: &'a std::sync::Arc<bonsai_windows_process::MappedFile>,
    ) -> Self {
        Self {
            index,
            source,
            remaining: 8 * 1024 * 1024 * 1024,
            mapping: Some(mapping),
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

    fn projection_bytes(
        &mut self,
        name: &str,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<(u32, WeightBytes)> {
        #[cfg(all(windows, feature = "rust-qualification"))]
        if let Some(mapping) = self.mapping {
            let tensor = self.index.tensors.get(name).ok_or_else(invalid)?;
            self.remaining = self
                .remaining
                .checked_sub(tensor.bytes)
                .ok_or_else(invalid)?;
            if cancel() {
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            let (offset, len) = mapped_tensor_range(tensor, mapping.len())?;
            return Ok((
                tensor.kind,
                WeightBytes::Mapped {
                    _mapping: std::sync::Arc::clone(mapping),
                    offset,
                    len,
                },
            ));
        }
        let (kind, raw) = self.bytes(name, cancel)?;
        Ok((kind, WeightBytes::Owned(raw)))
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
        self.stream_tensor_values(name, 1, 2, cancel, decode_f16_bytes_into)
    }

    pub(super) fn projection(
        &mut self,
        name: &str,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Projection> {
        let (input, output, kind) = projection_dims(self.index, name)?;
        match kind {
            0 => Projection::f32(input, output, self.vector(name, cancel)?),
            1 => Projection::f32(input, output, self.f16_vector(name, cancel)?),
            8 => Projection::q8_bytes(input, output, self.projection_bytes(name, cancel)?.1),
            12 => Projection::q4_k_bytes(input, output, self.projection_bytes(name, cancel)?.1),
            14 => Projection::q6_k_bytes(input, output, self.projection_bytes(name, cancel)?.1),
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
    let value = bonsai_ternary::half_to_float(bits);
    if !value.is_finite() {
        return Err(invalid());
    }
    Ok(value)
}

#[cfg(all(windows, feature = "rust-qualification"))]
pub(super) struct MappedSliceLoader<'a> {
    index: &'a Index,
    mapping: &'a std::sync::Arc<bonsai_windows_process::MappedFile>,
}

#[cfg(all(windows, feature = "rust-qualification"))]
impl<'a> MappedSliceLoader<'a> {
    pub(super) const fn new(
        index: &'a Index,
        mapping: &'a std::sync::Arc<bonsai_windows_process::MappedFile>,
    ) -> Self {
        Self { index, mapping }
    }

    pub(super) fn reserve_tensor(&self, remaining: &mut u64, name: &str) -> io::Result<()> {
        let tensor = self.index.tensors.get(name).ok_or_else(invalid)?;
        *remaining = remaining.checked_sub(tensor.bytes).ok_or_else(invalid)?;
        Ok(())
    }

    fn tensor_slice(
        &self,
        name: &str,
        expected_kind: u32,
        elem_bytes: u64,
    ) -> io::Result<&'a [u8]> {
        let tensor = self.index.tensors.get(name).ok_or_else(invalid)?;
        if tensor.kind != expected_kind || !tensor.bytes.is_multiple_of(elem_bytes) {
            return Err(invalid());
        }
        let offset = usize::try_from(tensor.offset).map_err(|_| invalid())?;
        let len = usize::try_from(tensor.bytes).map_err(|_| invalid())?;
        let end = offset.checked_add(len).ok_or_else(invalid)?;
        self.mapping
            .as_slice()
            .get(offset..end)
            .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))
    }

    pub(super) fn vector(&self, name: &str) -> io::Result<Vec<f32>> {
        let slice = self.tensor_slice(name, 0, 4)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(slice.len() / 4)
            .map_err(io::Error::other)?;
        for &raw in slice.as_chunks::<4>().0 {
            values.push(f32::from_le_bytes(raw));
        }
        Ok(values)
    }

    pub(super) fn f16_vector(&self, name: &str) -> io::Result<Vec<f32>> {
        let slice = self.tensor_slice(name, 1, 2)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(slice.len() / 2)
            .map_err(io::Error::other)?;
        decode_f16_bytes_into(slice, &mut values)?;
        Ok(values)
    }

    pub(super) fn projection(&self, name: &str) -> io::Result<Projection> {
        let (input, output, kind) = projection_dims(self.index, name)?;
        let tensor = self.index.tensors.get(name).ok_or_else(invalid)?;
        let (offset, len) = mapped_tensor_range(tensor, self.mapping.len())?;
        let mapped_bytes = || WeightBytes::Mapped {
            _mapping: std::sync::Arc::clone(self.mapping),
            offset,
            len,
        };
        match kind {
            0 => Projection::f32(input, output, self.vector(name)?),
            1 => Projection::f32(input, output, self.f16_vector(name)?),
            8 => Projection::q8_bytes(input, output, mapped_bytes()),
            12 => Projection::q4_k_bytes(input, output, mapped_bytes()),
            14 => Projection::q6_k_bytes(input, output, mapped_bytes()),
            _ => Err(invalid()),
        }
    }
}

#[cfg(all(windows, feature = "rust-qualification"))]
impl BlockLoader for &MappedSliceLoader<'_> {
    fn vector(&mut self, name: &str) -> io::Result<Vec<f32>> {
        MappedSliceLoader::vector(self, name)
    }

    fn projection(&mut self, name: &str) -> io::Result<Projection> {
        MappedSliceLoader::projection(self, name)
    }
}

fn load_decoder_block(loader: &mut impl BlockLoader, layer: usize) -> io::Result<Block> {
    let prefix = format!("dec.blocks.{layer}.");
    Ok(Block {
        attention_norm: loader.vector(&(prefix.clone() + "norm_attn.weight"))?,
        ffn_norm: loader.vector(&(prefix.clone() + "norm_ffn.weight"))?,
        query: loader.projection(&(prefix.clone() + "attn.q.weight"))?,
        key: loader.projection(&(prefix.clone() + "attn.k.weight"))?,
        value: loader.projection(&(prefix.clone() + "attn.v.weight"))?,
        attention_out: loader.projection(&(prefix.clone() + "attn.o.weight"))?,
        query_norm: loader.vector(&(prefix.clone() + "attn.q_norm.weight"))?,
        key_norm: loader.vector(&(prefix.clone() + "attn.k_norm.weight"))?,
        gate: loader.projection(&(prefix.clone() + "ffn.gate.weight"))?,
        up: loader.projection(&(prefix.clone() + "ffn.up.weight"))?,
        down: loader.projection(&(prefix + "ffn.down.weight"))?,
    })
}

pub(super) fn load<R: Read + Seek>(
    index: &Index,
    source: &mut R,
    precision: CachePrecision,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Model> {
    let mut loader = Loader::new(index, source);
    let (config, count) = validate_decoder_geometry(&loader, precision)?;
    let embedding = loader.projection("dec.token_embd.weight", cancel)?;
    let output_norm = loader.vector("dec.output_norm.weight", cancel)?;
    let mut blocks = Vec::new();
    blocks.try_reserve_exact(count).map_err(io::Error::other)?;
    let mut block_loader = CancellableLoader::new(&mut loader, cancel);
    for layer in 0..count {
        blocks.push(load_decoder_block(&mut block_loader, layer)?);
    }
    Model::new(config, embedding, blocks, output_norm)
}

#[cfg(all(windows, feature = "rust-qualification"))]
pub(super) fn load_mapped(
    index: &Index,
    mapping: &std::sync::Arc<bonsai_windows_process::MappedFile>,
    precision: CachePrecision,
    cancel: &mut impl FnMut() -> bool,
) -> io::Result<Model> {
    let mut cursor = io::Cursor::new(mapping.as_slice());
    let mut loader = Loader::new_mapped(index, &mut cursor, mapping);
    let (config, count) = validate_decoder_geometry(&loader, precision)?;
    let embedding = loader.projection("dec.token_embd.weight", cancel)?;
    let output_norm = loader.vector("dec.output_norm.weight", cancel)?;
    let slice_loader = MappedSliceLoader::new(index, mapping);
    let mut remaining = loader.remaining;
    for layer in 0..count {
        let prefix = format!("dec.blocks.{layer}.");
        for suffix in [
            "norm_attn.weight",
            "norm_ffn.weight",
            "attn.q.weight",
            "attn.k.weight",
            "attn.v.weight",
            "attn.o.weight",
            "attn.q_norm.weight",
            "attn.k_norm.weight",
            "ffn.gate.weight",
            "ffn.up.weight",
            "ffn.down.weight",
        ] {
            slice_loader.reserve_tensor(&mut remaining, &(prefix.clone() + suffix))?;
        }
    }
    let blocks = load_mapped_blocks(count, cancel, |layer| {
        let mut s = &slice_loader;
        load_decoder_block(&mut s, layer)
    })?;
    Model::new(config, embedding, blocks, output_norm)
}

fn validate_decoder_geometry<R: Read + Seek>(
    loader: &Loader<'_, R>,
    precision: CachePrecision,
) -> io::Result<(Config, usize)> {
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
    Ok((config, count))
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
        let mut loader = Loader::with_limit(&index, &mut source, 128);
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

    #[test]
    fn rust_asr_decoder_wire_kquant_projections() -> io::Result<()> {
        let mut q4 = vec![0_u8; 144];
        q4[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes()); // d = 1.0
        q4[4] = 1; // group 0 scale = 1
        q4[16..48].fill(0x01); // elements 0..32 low nibble = 1
        let mut q6 = vec![0_u8; 210];
        q6[0..16].fill(0x01);
        q6[128..144].fill(0x02);
        q6[192] = 1; // scale[0] = 1 (elements 0..16), remaining scales = 0
        q6[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes()); // d = 1.0

        let mut source_bytes = Vec::new();
        source_bytes.extend_from_slice(&q4);
        source_bytes.extend_from_slice(&q6);
        let tensors = BTreeMap::from([
            (
                "q4k".into(),
                Tensor {
                    shape: vec![256, 1],
                    kind: 12,
                    offset: 0,
                    bytes: 144,
                },
            ),
            (
                "q6k".into(),
                Tensor {
                    shape: vec![256, 1],
                    kind: 14,
                    offset: 144,
                    bytes: 210,
                },
            ),
        ]);
        let index = Index {
            metadata: BTreeMap::new(),
            tensors,
        };
        let mut cursor = Cursor::new(source_bytes);
        let mut loader = Loader::new(&index, &mut cursor);
        assert_eq!(
            loader
                .projection("q4k", &mut || false)?
                .multiply(&[1.0; 256], &mut || false)?,
            [32.0]
        );
        assert_eq!(
            loader
                .projection("q6k", &mut || false)?
                .multiply(&[1.0; 256], &mut || false)?,
            [16.0]
        );
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
                let mut loader = Loader::with_limit(&index, &mut source, length);
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
            let mut loader = Loader::with_limit(&index, &mut source, 34);
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
        let mut loader = Loader::with_limit(&index, &mut source, 16);
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
                let mut loader = Loader::with_limit(&index, &mut source, length);
                let projection = loader.convolution("weight", &mut || false)?;
                assert_eq!(projection.dimensions(), (9, 1));
                assert_eq!(projection.multiply(&[1.0; 9], &mut || false)?, [5.0]);
            }
        }
        let mut source = Cursor::new(Vec::<u8>::new());
        let index = one_tensor(&[3], 0, 12);
        let mut loader = Loader::with_limit(&index, &mut source, 12);
        assert!(loader.convolution("weight", &mut || false).is_err());
        assert_eq!(source.position(), 0);
        Ok(())
    }
}
