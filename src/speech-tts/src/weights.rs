//! Bounded Rust GGUF tensor loading and packed GGML weight decoding.
//!
//! `Q4_K/Q6_K` layouts follow the retained MIT-licensed GGML revision recorded in
//! model.lock.json and LICENSE.ggml. No native library participates in loading
//! or decoding. The caller owns the artifact's complete digest verification.
#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "The speech modules share this private weight reader."
)]

use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use crate::engine::{LinearProjection, WeightQuantization};

const MAX_ITEMS: u64 = 131_072;
const MAX_METADATA_BYTES: usize = 128 * 1024 * 1024;
const MAX_RETAINED_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

pub(crate) fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid or unsupported speech GGUF weights.",
    )
}

#[derive(Debug)]
enum Metadata {
    U32(u32),
    I32(i32),
    F32(f32),
    String(String),
    U32Array(Vec<u32>),
    StringArray(Vec<String>),
    Other,
}

#[derive(Clone, Debug)]
struct Tensor {
    shape: Vec<usize>,
    kind: u32,
    offset: u64,
    bytes: usize,
}

/// Read-only retained memory-mapped tensor view sharing an `Arc<MappedFile>` without an open handle.
#[cfg(windows)]
pub(crate) struct MappedWeights {
    mapped: std::sync::Arc<bonsai_windows_process::MappedFile>,
    metadata: BTreeMap<String, Metadata>,
    tensors: BTreeMap<String, Tensor>,
}

#[cfg(windows)]
impl Metadata {
    fn clone_without_string_arrays(&self) -> Option<Self> {
        match self {
            Self::U32(value) => Some(Self::U32(*value)),
            Self::I32(value) => Some(Self::I32(*value)),
            Self::F32(value) => Some(Self::F32(*value)),
            Self::String(value) => Some(Self::String(value.clone())),
            Self::U32Array(values) => Some(Self::U32Array(values.clone())),
            Self::StringArray(_) | Self::Other => None,
        }
    }
}

fn metadata_u32(metadata: &BTreeMap<String, Metadata>, key: &str) -> io::Result<u32> {
    match metadata.get(key) {
        Some(Metadata::U32(value)) => Ok(*value),
        Some(Metadata::I32(value)) => u32::try_from(*value).map_err(|_| invalid()),
        _ => Err(invalid()),
    }
}

fn metadata_f32(metadata: &BTreeMap<String, Metadata>, key: &str) -> io::Result<f32> {
    match metadata.get(key) {
        Some(Metadata::F32(value)) => Ok(*value),
        _ => Err(invalid()),
    }
}

fn metadata_u32_array(metadata: &BTreeMap<String, Metadata>, key: &str) -> io::Result<Vec<u32>> {
    match metadata.get(key) {
        Some(Metadata::U32Array(values)) => Ok(values.clone()),
        _ => Err(invalid()),
    }
}

/// Validated tensor index retaining a non-writable model file handle.
pub(crate) struct Weights {
    path: PathBuf,
    file: File,
    #[cfg(windows)]
    mapped: Option<std::sync::Arc<bonsai_windows_process::MappedFile>>,
    metadata: BTreeMap<String, Metadata>,
    tensors: BTreeMap<String, Tensor>,
    stream_buf: Vec<u8>,
}

const ALLOWED_TTS_TENSORS: &[u32] = &[0, 1, 8, 12, 14];

fn retain_tts_strings(key: &str) -> bool {
    matches!(key, "tokenizer.ggml.tokens" | "tokenizer.ggml.merges")
}

const fn retain_tts_scalar_arrays(_key: &str, element_type_id: u32) -> bool {
    matches!(element_type_id, 4 | 5)
}

fn tts_index_policy(
    allow_zero_tensors: bool,
    metadata_budget: usize,
) -> bonsai_gguf::IndexPolicy<'static> {
    bonsai_gguf::IndexPolicy {
        allow_v2: true,
        allow_zero_tensors,
        allow_zero_metadata: true,
        max_file_bytes: MAX_FILE_BYTES,
        max_tensors: MAX_ITEMS,
        max_metadata: MAX_ITEMS,
        metadata_budget,
        max_key_bytes: 65_536,
        max_string_bytes: 16 * 1024 * 1024,
        max_array_string_bytes: 16 * 1024 * 1024,
        max_retained_string_array_bytes: Some(MAX_RETAINED_STRING_BYTES),
        max_array_items: 1_000_000,
        max_alignment: 1_048_576,
        allow_positive_i32_alignment: true,
        reject_nonfinite_floats: true,
        reject_interior_nul_keys: true,
        reject_interior_nul_strings: true,
        kv_overhead_bytes: 0,
        string_overhead_bytes: 0,
        tensor_overhead_bytes: 0,
        charge_string_array_slots: false,
        charge_skipped_scalar_arrays: false,
        prism_dialect: false,
        allowed_tensor_types: Some(ALLOWED_TTS_TENSORS),
        retain_string_array: retain_tts_strings,
        retain_scalar_array: retain_tts_scalar_arrays,
    }
}

fn map_tts_metadata(entry: bonsai_gguf::IndexValue) -> Metadata {
    use bonsai_gguf::{IndexValue, MetadataArray, MetadataValue};
    match entry {
        IndexValue::Metadata(meta) => match meta {
            MetadataValue::U32(value) => Metadata::U32(value),
            MetadataValue::I32(value) => Metadata::I32(value),
            MetadataValue::F32(value) => Metadata::F32(value),
            MetadataValue::String(value) => Metadata::String(value),
            MetadataValue::Array(MetadataArray::U32(values)) => Metadata::U32Array(values),
            MetadataValue::Array(MetadataArray::I32(values)) => values
                .into_iter()
                .map(|value| u32::try_from(value).ok())
                .collect::<Option<Vec<u32>>>()
                .map_or(Metadata::Other, Metadata::U32Array),
            MetadataValue::Array(MetadataArray::String(values)) => Metadata::StringArray(values),
            _ => Metadata::Other,
        },
        IndexValue::SkippedArray { .. } => Metadata::Other,
    }
}

struct ParsedWeightsIndex {
    metadata: BTreeMap<String, Metadata>,
    tensors: BTreeMap<String, Tensor>,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retained for GGUF budget accounting assertions in unit tests."
        )
    )]
    remaining_budget: usize,
}

fn parse_index(
    source: &mut (impl Read + Seek),
    length: u64,
    allow_zero_tensors: bool,
    budget: usize,
) -> io::Result<ParsedWeightsIndex> {
    let parsed = bonsai_gguf::read_index(
        source,
        length,
        &tts_index_policy(allow_zero_tensors, budget),
    )
    .map_err(|err| {
        if err.kind() == io::ErrorKind::InvalidData {
            invalid()
        } else {
            err
        }
    })?;
    let metadata = parsed
        .metadata
        .into_iter()
        .map(|(key, value)| (key, map_tts_metadata(value)))
        .collect();
    let mut tensors = BTreeMap::new();
    for (name, tensor) in parsed.tensors {
        let mut shape = Vec::with_capacity(tensor.dimensions.len());
        for dim in tensor.dimensions {
            shape.push(usize::try_from(dim).map_err(|_| invalid())?);
        }
        let bytes = usize::try_from(tensor.bytes).map_err(|_| invalid())?;
        tensors.insert(
            name,
            Tensor {
                shape,
                kind: tensor.type_id,
                offset: tensor.offset,
                bytes,
            },
        );
    }
    Ok(ParsedWeightsIndex {
        metadata,
        tensors,
        remaining_budget: parsed.remaining_budget,
    })
}

impl Weights {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        Self::open_with_policy(path, false)
    }

    #[cfg(test)]
    pub(crate) fn open_metadata(path: &Path) -> io::Result<Self> {
        Self::open_with_policy(path, true)
    }

    fn open_with_policy(path: &Path, allow_zero_tensors: bool) -> io::Result<Self> {
        if !path.is_absolute() {
            return Err(invalid());
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt as _;
            // Read sharing only: replacements and writes cannot race loading.
            // Combine FILE_FLAG_OPEN_REPARSE_POINT with FILE_FLAG_SEQUENTIAL_SCAN.
            options
                .share_mode(1)
                .custom_flags(0x0020_0000 | 0x0800_0000);
        }
        let file = options.open(path)?;
        let properties = file.metadata()?;
        if !properties.is_file() || properties.len() > MAX_FILE_BYTES {
            return Err(invalid());
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt as _;
            if properties.file_attributes() & 0x400 != 0 {
                return Err(invalid());
            }
        }
        Self::read(
            path.to_path_buf(),
            file,
            properties.len(),
            allow_zero_tensors,
        )
    }

    fn read(
        path: PathBuf,
        mut file: File,
        length: u64,
        allow_zero_tensors: bool,
    ) -> io::Result<Self> {
        #[cfg(windows)]
        let mapped = if length > 0 {
            Some(std::sync::Arc::new(
                bonsai_windows_process::MappedFile::from_file_without_prefetch(&file)?,
            ))
        } else {
            None
        };
        let ParsedWeightsIndex {
            metadata, tensors, ..
        } = {
            #[cfg(windows)]
            if let Some(view) = &mapped {
                let mut cursor = io::Cursor::new(view.as_slice());
                parse_index(&mut cursor, length, allow_zero_tensors, MAX_METADATA_BYTES)?
            } else {
                parse_index(&mut file, length, allow_zero_tensors, MAX_METADATA_BYTES)?
            }
            #[cfg(not(windows))]
            parse_index(&mut file, length, allow_zero_tensors, MAX_METADATA_BYTES)?
        };
        #[cfg(windows)]
        let mapped = if tensors.is_empty() { None } else { mapped };
        Ok(Self {
            path,
            file,
            #[cfg(windows)]
            mapped,
            metadata,
            tensors,
            stream_buf: Vec::new(),
        })
    }

    pub(crate) const fn is_mapped(&self) -> bool {
        #[cfg(windows)]
        {
            self.mapped.is_some()
        }
        #[cfg(not(windows))]
        {
            false
        }
    }

    #[cfg(windows)]
    pub(crate) fn retain_mapped_view(&self) -> Option<MappedWeights> {
        self.mapped.as_ref().map(|mapped| MappedWeights {
            mapped: std::sync::Arc::clone(mapped),
            metadata: self
                .metadata
                .iter()
                .filter_map(|(k, v)| v.clone_without_string_arrays().map(|cloned| (k.clone(), cloned)))
                .collect(),
            tensors: self.tensors.clone(),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn metadata_contains(&self, key: &str) -> bool {
        self.metadata.contains_key(key)
    }

    pub(crate) fn string_array(&self, key: &str) -> io::Result<&[String]> {
        match self.metadata.get(key) {
            Some(Metadata::StringArray(values)) => Ok(values.as_slice()),
            _ => Err(invalid()),
        }
    }

    pub(crate) fn u32(&self, key: &str) -> io::Result<u32> {
        metadata_u32(&self.metadata, key)
    }

    pub(crate) fn f32_metadata(&self, key: &str) -> io::Result<f32> {
        metadata_f32(&self.metadata, key)
    }

    pub(crate) fn string(&self, key: &str) -> io::Result<&str> {
        match self.metadata.get(key) {
            Some(Metadata::String(value)) => Ok(value),
            _ => Err(invalid()),
        }
    }

    pub(crate) fn u32_array(&self, key: &str) -> io::Result<Vec<u32>> {
        metadata_u32_array(&self.metadata, key)
    }

    pub(crate) fn tensor_shape(&self, name: &str) -> io::Result<&[usize]> {
        self.tensors
            .get(name)
            .map(|tensor| tensor.shape.as_slice())
            .ok_or_else(invalid)
    }

    fn tensor_bytes(&mut self, name: &str, shape: &[usize]) -> io::Result<(u32, Vec<u8>)> {
        let tensor = self.tensors.get(name).ok_or_else(invalid)?;
        if tensor.shape != shape {
            return Err(invalid());
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(tensor.bytes)
            .map_err(|_| invalid())?;
        bytes.resize(tensor.bytes, 0);
        self.file.seek(SeekFrom::Start(tensor.offset))?;
        self.file.read_exact(&mut bytes)?;
        // Each consumer validates its final representation: projection() uses
        // LinearProjection::new's packed validation, and tensor_f32() checks all
        // decoded values. Avoid scanning the entire model a second time here.
        Ok((tensor.kind, bytes))
    }

    pub(crate) fn tensor_f32(&mut self, name: &str, shape: &[usize]) -> io::Result<Vec<f32>> {
        #[cfg(windows)]
        if self.mapped.is_some() {
            return self.tensor_f32_mapped_with_threads(name, shape, 1);
        }
        let tensor = self.tensors.get(name).ok_or_else(invalid)?;
        if tensor.shape != shape {
            return Err(invalid());
        }
        let kind = tensor.kind;
        let offset = tensor.offset;
        let tensor_bytes = tensor.bytes;
        let (block, size) = packed_layout(kind)?;
        if !tensor_bytes.is_multiple_of(size) {
            return Err(invalid());
        }
        let total_blocks = tensor_bytes / size;
        let count = total_blocks.checked_mul(block).ok_or_else(invalid)?;
        let mut output = Vec::new();
        output.try_reserve_exact(count).map_err(|_| invalid())?;
        output.resize(count, 0.0);
        self.file.seek(SeekFrom::Start(offset))?;
        let max_chunk_blocks = (64 * 1024 / size).max(1);
        let chunk_bytes = total_blocks.min(max_chunk_blocks) * size;
        if self.stream_buf.len() < chunk_bytes {
            self.stream_buf
                .try_reserve_exact(chunk_bytes - self.stream_buf.len())
                .map_err(|_| invalid())?;
            self.stream_buf.resize(chunk_bytes, 0);
        }
        let mut out_offset = 0;
        let mut blocks_left = total_blocks;
        while blocks_left > 0 {
            let step_blocks = blocks_left.min(max_chunk_blocks);
            let step_bytes = step_blocks * size;
            let buf = &mut self.stream_buf[..step_bytes];
            self.file.read_exact(buf)?;
            let out_step = step_blocks * block;
            let dest_slice = &mut output[out_offset..out_offset + out_step];
            decode_slice_f32(kind, buf, dest_slice)?;
            out_offset += out_step;
            blocks_left -= step_blocks;
        }
        Ok(output)
    }

    pub(crate) fn tensor_f32_mapped(&self, name: &str, shape: &[usize]) -> io::Result<Vec<f32>> {
        self.tensor_f32_mapped_with_threads(name, shape, 1)
    }

    pub(crate) fn tensor_f32_mapped_with_threads(
        &self,
        name: &str,
        shape: &[usize],
        threads: usize,
    ) -> io::Result<Vec<f32>> {
        #[cfg(windows)]
        {
            let mapped = self.mapped.as_deref().ok_or_else(invalid)?;
            return mapped_tensor_f32(&self.tensors, mapped, name, shape, threads);
        }
        #[cfg(not(windows))]
        {
            let _ = (name, shape, threads);
            Err(invalid())
        }
    }

    pub(crate) fn projection(
        &mut self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> io::Result<LinearProjection> {
        #[cfg(windows)]
        if self.mapped.is_some() {
            return self.projection_mapped(name, in_dim, out_dim, 1);
        }
        let (ggml_type, bytes) = self.tensor_bytes(name, &[in_dim, out_dim])?;
        LinearProjection::new(
            name,
            in_dim,
            out_dim,
            WeightQuantization::Packed { ggml_type, bytes },
            None,
        )
    }

    pub(crate) fn projection_mapped(
        &self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
        validate_threads: usize,
    ) -> io::Result<LinearProjection> {
        #[cfg(windows)]
        {
            let mapped = self.mapped.as_ref().ok_or_else(invalid)?;
            return mapped_projection(&self.tensors, mapped, name, in_dim, out_dim, validate_threads);
        }
        #[cfg(not(windows))]
        {
            let _ = (name, in_dim, out_dim, validate_threads);
            Err(invalid())
        }
    }
}

#[cfg(windows)]
impl MappedWeights {
    pub(crate) fn u32(&self, key: &str) -> io::Result<u32> {
        metadata_u32(&self.metadata, key)
    }

    pub(crate) fn f32_metadata(&self, key: &str) -> io::Result<f32> {
        metadata_f32(&self.metadata, key)
    }

    pub(crate) fn u32_array(&self, key: &str) -> io::Result<Vec<u32>> {
        metadata_u32_array(&self.metadata, key)
    }

    pub(crate) fn tensor_shape(&self, name: &str) -> io::Result<&[usize]> {
        self.tensors
            .get(name)
            .map(|tensor| tensor.shape.as_slice())
            .ok_or_else(invalid)
    }

    pub(crate) fn tensor_f32_mapped(&self, name: &str, shape: &[usize]) -> io::Result<Vec<f32>> {
        mapped_tensor_f32(&self.tensors, &self.mapped, name, shape, 1)
    }

    pub(crate) fn projection_mapped(
        &self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
        validate_threads: usize,
    ) -> io::Result<LinearProjection> {
        mapped_projection(
            &self.tensors,
            &self.mapped,
            name,
            in_dim,
            out_dim,
            validate_threads,
        )
    }
}

#[cfg(windows)]
fn mapped_tensor_f32(
    tensors: &BTreeMap<String, Tensor>,
    mapped: &bonsai_windows_process::MappedFile,
    name: &str,
    shape: &[usize],
    threads: usize,
) -> io::Result<Vec<f32>> {
    let tensor = tensors.get(name).ok_or_else(invalid)?;
    if tensor.shape != shape {
        return Err(invalid());
    }
    let kind = tensor.kind;
    let tensor_bytes = tensor.bytes;
    let (block, size) = packed_layout(kind)?;
    if !tensor_bytes.is_multiple_of(size) {
        return Err(invalid());
    }
    let start = usize::try_from(tensor.offset).map_err(|_| invalid())?;
    let end = start.checked_add(tensor_bytes).ok_or_else(invalid)?;
    let bytes = mapped.as_slice().get(start..end).ok_or_else(invalid)?;
    let total_blocks = tensor_bytes / size;
    let count = total_blocks.checked_mul(block).ok_or_else(invalid)?;
    let mut output = Vec::new();
    output.try_reserve_exact(count).map_err(|_| invalid())?;
    output.resize(count, 0.0);
    decode_slice_f32_parallel(kind, bytes, &mut output, threads)?;
    Ok(output)
}

#[cfg(windows)]
fn mapped_projection(
    tensors: &BTreeMap<String, Tensor>,
    mapped: &std::sync::Arc<bonsai_windows_process::MappedFile>,
    name: &str,
    in_dim: usize,
    out_dim: usize,
    validate_threads: usize,
) -> io::Result<LinearProjection> {
    let tensor = tensors.get(name).ok_or_else(invalid)?;
    if tensor.shape != [in_dim, out_dim] {
        return Err(invalid());
    }
    let start = usize::try_from(tensor.offset).map_err(|_| invalid())?;
    let end = start.checked_add(tensor.bytes).ok_or_else(invalid)?;
    let slice = crate::engine::MappedSlice::new(std::sync::Arc::clone(mapped), start, end)?;
    LinearProjection::new_with_threads(
        name,
        in_dim,
        out_dim,
        WeightQuantization::MappedPacked {
            ggml_type: tensor.kind,
            bytes: slice,
        },
        None,
        validate_threads,
    )
}

pub(crate) fn packed_layout(kind: u32) -> io::Result<(usize, usize)> {
    match kind {
        0 => Ok((1, 4)),
        1 => Ok((1, 2)),
        8 => Ok((32, 34)),
        12 => Ok((256, 144)),
        14 => Ok((256, 210)),
        _ => Err(invalid()),
    }
}

pub(crate) fn half(bytes: &[u8]) -> f32 {
    bonsai_ternary::half_to_float(u16::from_le_bytes([bytes[0], bytes[1]]))
}

#[inline]
pub(crate) const fn unpack_q4_k_scale_min(group: usize, scales: &[u8]) -> (u8, u8) {
    bonsai_ternary::unpack_q4_k_scale_min(group, scales)
}

#[inline]
const fn f16_bytes_finite(lo: u8, hi: u8) -> bool {
    (u16::from_le_bytes([lo, hi]) & 0x7C00) != 0x7C00
}

#[inline]
const fn f32_bytes_finite(raw: [u8; 4]) -> bool {
    (u32::from_le_bytes(raw) & 0x7F80_0000) != 0x7F80_0000
}

pub(crate) fn validate_packed(kind: u32, bytes: &[u8]) -> io::Result<()> {
    let (_, size) = packed_layout(kind)?;
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid());
    }
    let valid = match kind {
        0 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|raw| f32_bytes_finite(*raw)),
        1 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|raw| f16_bytes_finite(raw[0], raw[1])),
        8 => bytes
            .chunks_exact(34)
            .all(|block| f16_bytes_finite(block[0], block[1])),
        12 => bytes.chunks_exact(144).all(|block| {
            f16_bytes_finite(block[0], block[1]) && f16_bytes_finite(block[2], block[3])
        }),
        14 => bytes
            .chunks_exact(210)
            .all(|block| f16_bytes_finite(block[208], block[209])),
        _ => false,
    };
    if valid { Ok(()) } else { Err(invalid()) }
}

pub(crate) fn validate_packed_parallel(kind: u32, bytes: &[u8], threads: usize) -> io::Result<()> {
    const PARALLEL_THRESHOLD_BYTES: usize = 4 * 1024 * 1024;
    let (_, size) = packed_layout(kind)?;
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid());
    }
    if threads <= 1 || bytes.len() < PARALLEL_THRESHOLD_BYTES {
        return validate_packed(kind, bytes);
    }
    let total_blocks = bytes.len() / size;
    let workers = threads.min(total_blocks).min(32);
    if workers <= 1 {
        return validate_packed(kind, bytes);
    }
    let chunk_blocks = total_blocks.div_ceil(workers);
    let chunk_bytes = chunk_blocks * size;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers - 1);
        let mut chunks = bytes.chunks(chunk_bytes);
        let first = chunks.next().unwrap_or(&[]);
        for chunk in chunks {
            handles.push(scope.spawn(move || validate_packed(kind, chunk)));
        }
        let mut result = validate_packed(kind, first);
        for handle in handles {
            let joined = handle.join().unwrap_or_else(|_| Err(invalid()));
            if result.is_ok() {
                result = joined;
            }
        }
        result
    })
}

fn decode_slice_f32(kind: u32, bytes: &[u8], output: &mut [f32]) -> io::Result<()> {
    match kind {
        0 => {
            for (raw, dst) in bytes.as_chunks::<4>().0.iter().zip(output.iter_mut()) {
                if !f32_bytes_finite(*raw) {
                    return Err(invalid());
                }
                *dst = f32::from_le_bytes(*raw);
            }
            Ok(())
        }
        1 => {
            let (in8, in_tail) = bytes.as_chunks::<16>();
            let (out8, out_tail) = output.as_chunks_mut::<8>();
            for (src, dst) in in8.iter().zip(out8.iter_mut()) {
                let pairs = src.as_chunks::<2>().0;
                let bits: [u16; 8] = std::array::from_fn(|idx| u16::from_le_bytes(pairs[idx]));
                if bits.iter().any(|&raw| (raw & 0x7C00) == 0x7C00) {
                    return Err(invalid());
                }
                let [lo, hi] = bonsai_neon_dot::widen_f16_eight(&bits);
                dst[..4].copy_from_slice(&lo);
                dst[4..].copy_from_slice(&hi);
            }
            for (src, dst) in in_tail.as_chunks::<2>().0.iter().zip(out_tail.iter_mut()) {
                if !f16_bytes_finite(src[0], src[1]) {
                    return Err(invalid());
                }
                *dst = half(src);
            }
            Ok(())
        }
        8 => {
            for (source, destination) in bytes.chunks_exact(34).zip(output.chunks_exact_mut(32)) {
                if !f16_bytes_finite(source[0], source[1]) {
                    return Err(invalid());
                }
                let scale = half(source);
                for (value, &byte) in destination.iter_mut().zip(&source[2..]) {
                    *value = scale * f32::from(byte.cast_signed());
                }
            }
            Ok(())
        }
        12 | 14 => {
            validate_packed(kind, bytes)?;
            let (block, size) = packed_layout(kind)?;
            for (source, destination) in bytes.chunks_exact(size).zip(output.chunks_exact_mut(block))
            {
                decode_block(kind, source, destination);
            }
            if output.iter().any(|value| !value.is_finite()) {
                return Err(invalid());
            }
            Ok(())
        }
        _ => Err(invalid()),
    }
}

#[cfg(windows)]
fn decode_slice_f32_parallel(
    kind: u32,
    bytes: &[u8],
    output: &mut [f32],
    threads: usize,
) -> io::Result<()> {
    const PARALLEL_DECODE_THRESHOLD_BYTES: usize = 2 * 1024 * 1024;
    if threads <= 1 || bytes.len() < PARALLEL_DECODE_THRESHOLD_BYTES {
        return decode_slice_f32(kind, bytes, output);
    }
    let (block, size) = packed_layout(kind)?;
    let total_blocks = bytes.len() / size;
    let workers = threads.min(total_blocks).min(32);
    if workers <= 1 {
        return decode_slice_f32(kind, bytes, output);
    }
    let chunk_blocks = total_blocks.div_ceil(workers);
    let in_chunk = chunk_blocks * size;
    let out_chunk = chunk_blocks * block;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers - 1);
        let mut in_chunks = bytes.chunks(in_chunk);
        let mut out_chunks = output.chunks_mut(out_chunk);
        let first_in = in_chunks.next().unwrap_or(&[]);
        let first_out = out_chunks.next().unwrap_or(&mut []);
        for (src, dst) in in_chunks.zip(out_chunks) {
            handles.push(scope.spawn(move || decode_slice_f32(kind, src, dst)));
        }
        let mut result = decode_slice_f32(kind, first_in, first_out);
        for handle in handles {
            let joined = handle.join().unwrap_or_else(|_| Err(invalid()));
            if result.is_ok() {
                result = joined;
            }
        }
        result
    })
}

// The caller validates the exact kind, source block size and destination width.
#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve GGML's separately rounded Q4_K scale and minimum operations."
)]
pub(crate) fn decode_block(kind: u32, source: &[u8], destination: &mut [f32]) {
    match kind {
        0 => {
            if let Some(raw) = source.first_chunk::<4>() {
                destination[0] = f32::from_le_bytes(*raw);
            }
        }
        1 => destination[0] = half(source),
        8 => {
            if let Some((scale_bytes, q_bytes)) = source.split_first_chunk::<2>() {
                let scale = half(scale_bytes);
                for (value, byte) in destination.iter_mut().zip(q_bytes) {
                    *value = scale * f32::from(i8::from_le_bytes([*byte]));
                }
            }
        }
        12 => {
            let scale = half(source);
            let minimum = half(&source[2..]);
            let scales = &source[4..16];
            for group in 0..8 {
                let (s, m) = unpack_q4_k_scale_min(group, scales);
                for index in 0..32 {
                    let byte = source[16 + (group / 2) * 32 + index];
                    let quant = if group % 2 == 0 { byte & 15 } else { byte >> 4 };
                    destination[group * 32 + index] =
                        scale * f32::from(s) * f32::from(quant) - minimum * f32::from(m);
                }
            }
        }
        14 => {
            let scale = half(&source[208..]);
            for half_index in 0..2 {
                for index in 0..32 {
                    let high = source[128 + half_index * 32 + index];
                    for quarter in 0..4 {
                        let low = source[half_index * 64 + (quarter % 2) * 32 + index];
                        let nibble = if quarter < 2 { low & 15 } else { low >> 4 };
                        let quant = i16::from(nibble | (((high >> (quarter * 2)) & 3) << 4)) - 32;
                        let s = i8::from_le_bytes([
                            source[192 + half_index * 8 + quarter * 2 + index / 16]
                        ]);
                        destination[half_index * 128 + quarter * 32 + index] =
                            scale * f32::from(s) * f32::from(quant);
                    }
                }
            }
        }
        _ => destination.fill(f32::NAN),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn fixture(tensors: &[(&str, &[u64], u32, u64)], payload: &[u8]) -> io::Result<Fixture> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(
            u64::try_from(tensors.len())
                .map_err(|_| invalid())?
                .to_le_bytes(),
        );
        bytes.extend(0_u64.to_le_bytes());
        for &(name, shape, kind, offset) in tensors {
            bytes.extend(
                u64::try_from(name.len())
                    .map_err(|_| invalid())?
                    .to_le_bytes(),
            );
            bytes.extend(name.bytes());
            bytes.extend(
                u32::try_from(shape.len())
                    .map_err(|_| invalid())?
                    .to_le_bytes(),
            );
            for value in shape {
                bytes.extend(value.to_le_bytes());
            }
            bytes.extend(kind.to_le_bytes());
            bytes.extend(offset.to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.extend(payload);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| invalid())?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "Bonsai-RustWeights-{}-{nonce}.gguf",
            std::process::id()
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(&bytes)?;
        Ok(Fixture(path))
    }

    #[test]
    fn gguf_tensor_loading_preserves_shape_and_rejects_invalid_index_ranges() -> io::Result<()> {
        let valid = fixture(&[("weight", &[2, 1], 0, 0)], &[0, 0, 128, 63, 0, 0, 0, 64])?;
        let mut loaded = Weights::open(&valid.0)?;
        assert_eq!(loaded.tensor_shape("weight")?, [2, 1]);
        assert_eq!(loaded.tensor_f32("weight", &[2, 1])?, [1.0, 2.0]);
        assert!(loaded.tensor_f32("weight", &[1, 2]).is_err());
        assert!(loaded.tensor_f32("missing", &[2, 1]).is_err());
        let projection = loaded.projection("weight", 2, 1)?;
        assert_eq!(projection.row(0)?, [1.0, 2.0]);
        assert!(projection.row(1).is_err());
        drop(loaded);
        for tensors in [
            vec![
                ("duplicate", &[1_u64][..], 0, 0),
                ("duplicate", &[1][..], 0, 32),
            ],
            vec![("first", &[1][..], 0, 0), ("second", &[1][..], 0, 0)],
            vec![("unaligned", &[1][..], 0, 1)],
            vec![("truncated", &[100][..], 0, 0)],
            vec![("quant-row", &[1][..], 8, 0)],
            vec![("unsupported", &[1][..], 99, 0)],
            vec![("empty", &[0][..], 0, 0)],
            vec![("overflow", &[u64::MAX, 2][..], 0, 0)],
        ] {
            let invalid_file = fixture(&tensors, &[0; 64])?;
            assert!(Weights::open(&invalid_file.0).is_err());
        }
        Ok(())
    }

    #[test]
    fn gguf_integer_metadata_accepts_nonnegative_signed_rates_and_rejects_negative_rates()
    -> io::Result<()> {
        let mut bytes = gguf_header(1);
        let _ = push_kv_str(&mut bytes, b"k");
        let value_offset = bytes.len();
        bytes.extend(9_u32.to_le_bytes());
        bytes.extend(5_u32.to_le_bytes());
        bytes.extend(4_u64.to_le_bytes());
        for value in [8_i32, 5, 4, 3] {
            bytes.extend(value.to_le_bytes());
        }
        let length = u64::try_from(bytes.len()).map_err(|_| invalid())?;
        let parsed = parse_index(
            &mut io::Cursor::new(&bytes),
            length,
            true,
            MAX_METADATA_BYTES,
        )?;
        assert!(matches!(
            parsed.metadata.get("k"),
            Some(Metadata::U32Array(values)) if values == &[8, 5, 4, 3]
        ));
        bytes[value_offset + 16..value_offset + 20].copy_from_slice(&(-1_i32).to_le_bytes());
        let parsed = parse_index(
            &mut io::Cursor::new(bytes),
            length,
            true,
            MAX_METADATA_BYTES,
        )?;
        assert!(matches!(parsed.metadata.get("k"), Some(Metadata::Other)));
        Ok(())
    }

    #[test]
    fn discarded_tokenizer_arrays_preserve_utf8_budget_and_truncation_checks() -> io::Result<()> {
        let mut bytes = gguf_header(1);
        let _ = push_kv_str(&mut bytes, b"k");
        let value_offset = bytes.len();
        bytes.extend(9_u32.to_le_bytes());
        bytes.extend(8_u32.to_le_bytes());
        bytes.extend(3_u64.to_le_bytes());
        let strings = ["a long initial spelling", "é", ""];
        for spelling in strings {
            bytes.extend(
                u64::try_from(spelling.len())
                    .map_err(|_| invalid())?
                    .to_le_bytes(),
            );
            bytes.extend(spelling.bytes());
        }
        let length = u64::try_from(bytes.len()).map_err(|_| invalid())?;
        let required: usize = 1 + strings.iter().map(|spelling| spelling.len()).sum::<usize>();
        let parsed = parse_index(&mut io::Cursor::new(&bytes), length, true, required)?;
        assert!(matches!(parsed.metadata.get("k"), Some(Metadata::Other)));
        assert_eq!(parsed.remaining_budget, 0);
        assert!(parse_index(&mut io::Cursor::new(&bytes), length, true, required - 1).is_err());
        assert!(
            parse_index(
                &mut io::Cursor::new(&bytes[..bytes.len() - 1]),
                length - 1,
                true,
                required,
            )
            .is_err()
        );
        // Corrupt the multibyte spelling while leaving lengths and bounds valid.
        let unicode = value_offset + 16 + 8 + strings[0].len() + 8;
        bytes[unicode] = 0xff;
        assert!(parse_index(&mut io::Cursor::new(bytes), length, true, required).is_err());
        Ok(())
    }

    #[test]
    fn tensor_consumers_reject_nonfinite_payloads_after_loading() -> io::Result<()> {
        for kind in [0, 1, 8, 12, 14] {
            let (width, size) = packed_layout(kind)?;
            let mut bytes = vec![0; size];
            match kind {
                0 => bytes.copy_from_slice(&f32::INFINITY.to_le_bytes()),
                1 | 8 => bytes[..2].copy_from_slice(&0x7c00_u16.to_le_bytes()),
                12 => bytes[2..4].copy_from_slice(&0x7c00_u16.to_le_bytes()),
                14 => bytes[208..].copy_from_slice(&0x7c00_u16.to_le_bytes()),
                _ => return Err(invalid()),
            }
            let shape = [u64::try_from(width).map_err(|_| invalid())?, 1];
            let file = fixture(&[("weight", &shape, kind, 0)], &bytes)?;
            let mut loaded = Weights::open(&file.0)?;
            assert!(loaded.projection("weight", width, 1).is_err());
            assert!(loaded.tensor_f32("weight", &[width, 1]).is_err());
        }
        Ok(())
    }

    #[test]
    fn packed_q4_k_scales_minima_and_q6_k_signed_planes_decode_independently() -> io::Result<()> {
        let mut q4 = vec![0; 144];
        q4[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        q4[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
        q4[4..8].fill(1);
        q4[8..12].fill(2);
        q4[12..16].fill(0x21);
        q4[16..].fill(0x73);
        let mut output = [0.0; 256];
        validate_packed(12, &q4)?;
        decode_block(12, &q4, &mut output);
        for (group, values) in output.as_chunks::<32>().0.iter().enumerate() {
            assert!(
                values.iter().all(|value| value.to_bits()
                    == if group % 2 == 0 { 2.0_f32 } else { 6.0_f32 }.to_bits())
            );
        }
        let mut q6 = vec![0; 210];
        q6[..128].fill(0x21);
        q6[128..192].fill(0b11_10_01_00);
        for (index, value) in q6[192..208].iter_mut().enumerate() {
            *value = i8::try_from(index).map_err(|_| invalid())?.to_le_bytes()[0];
        }
        q6[192] = (-2_i8).to_le_bytes()[0];
        q6[208..].copy_from_slice(&0x3c00_u16.to_le_bytes());
        validate_packed(14, &q6)?;
        decode_block(14, &q6, &mut output);
        assert_eq!(output[0], 62.0);
        assert_eq!(output[16], -31.0);
        assert_eq!(output[32], -30.0);
        assert_eq!(output[64], 8.0);
        assert_eq!(output[96], 108.0);
        q6[208..].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert!(validate_packed(14, &q6).is_err());
        Ok(())
    }

    #[test]
    fn packed_projection_computes_without_expanding_or_accepting_partial_rows() -> io::Result<()> {
        for kind in [0, 1, 8, 12, 14] {
            let (width, size) = packed_layout(kind)?;
            let mut bytes = vec![0; size];
            match kind {
                0 => bytes.copy_from_slice(&1.0_f32.to_le_bytes()),
                1 => bytes.copy_from_slice(&0x3c00_u16.to_le_bytes()),
                8 => {
                    bytes[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                    bytes[2..].fill(1);
                }
                12 => {
                    bytes[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                    bytes[4..8].fill(1);
                    bytes[12..16].fill(1);
                    bytes[16..].fill(0x11);
                }
                14 => {
                    bytes[..128].fill(0x11);
                    bytes[128..192].fill(0b10_10_10_10);
                    bytes[192..208].fill(1);
                    bytes[208..].copy_from_slice(&0x3c00_u16.to_le_bytes());
                }
                _ => return Err(invalid()),
            }
            let projection = LinearProjection::new(
                "fixture",
                width,
                1,
                WeightQuantization::Packed {
                    ggml_type: kind,
                    bytes: bytes.clone(),
                },
                Some(vec![2.0]),
            )?;
            assert_eq!(projection.row(0)?, vec![1.0; width]);
            let expected = f32::from(u16::try_from(width).map_err(|_| invalid())?) + 2.0;
            assert_eq!(projection.forward_vec(&vec![1.0; width], None)?, [expected]);
            bytes.pop();
            assert!(
                LinearProjection::new(
                    "truncated",
                    width,
                    1,
                    WeightQuantization::Packed {
                        ggml_type: kind,
                        bytes
                    },
                    None
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn packed_half_and_q8_keep_signed_zero_subnormals_and_signed_quants() -> io::Result<()> {
        assert_eq!(
            half(&0x8000_u16.to_le_bytes()).to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(half(&1_u16.to_le_bytes()), 1.0 / 16_777_216.0);
        assert_eq!(half(&0x3c00_u16.to_le_bytes()), 1.0);
        let mut bytes = [0; 34];
        bytes[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        bytes[2] = (-128_i8).to_le_bytes()[0];
        bytes[3] = 127;
        let mut decoded = [0.0; 32];
        validate_packed(8, &bytes)?;
        decode_block(8, &bytes, &mut decoded);
        assert_eq!(decoded[..2], [-64.0, 63.5]);
        Ok(())
    }

    fn push_kv_str(out: &mut Vec<u8>, bytes: &[u8]) -> io::Result<()> {
        out.extend(
            u64::try_from(bytes.len())
                .map_err(|_| invalid())?
                .to_le_bytes(),
        );
        out.extend(bytes);
        Ok(())
    }

    fn write_raw_gguf(bytes: &[u8]) -> io::Result<Fixture> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "Bonsai-RustWeightsHardened-{}-{id}.gguf",
            std::process::id()
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(bytes)?;
        Ok(Fixture(path))
    }

    fn single_tensor_gguf(
        kv: &[(&[u8], u32, &[u8])],
        alignment: usize,
        tensor_offset: u64,
    ) -> io::Result<Fixture> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(1_u64.to_le_bytes());
        bytes.extend(
            u64::try_from(kv.len())
                .map_err(|_| invalid())?
                .to_le_bytes(),
        );
        for &(key, kind, val) in kv {
            push_kv_str(&mut bytes, key)?;
            bytes.extend(kind.to_le_bytes());
            bytes.extend(val);
        }
        push_kv_str(&mut bytes, b"w")?;
        bytes.extend(1_u32.to_le_bytes());
        bytes.extend(1_u64.to_le_bytes());
        bytes.extend(0_u32.to_le_bytes());
        bytes.extend(tensor_offset.to_le_bytes());
        let padded = bytes.len().next_multiple_of(alignment);
        bytes.resize(
            padded + usize::try_from(tensor_offset).map_err(|_| invalid())? + 4,
            0,
        );
        write_raw_gguf(&bytes)
    }

    #[test]
    fn hardened_gguf_metadata_and_alignment_checks() -> io::Result<()> {
        // Bool scalar: 0/1 valid, 2 invalid.
        let ok_bool = single_tensor_gguf(&[(b"b", 7, &[1])], 32, 0)?;
        assert!(Weights::open(&ok_bool.0).is_ok());
        let bad_bool = single_tensor_gguf(&[(b"b", 7, &[2])], 32, 0)?;
        assert!(Weights::open(&bad_bool.0).is_err());

        // Bool array: 2 rejected.
        let mut bad_bool_arr = Vec::new();
        bad_bool_arr.extend(7_u32.to_le_bytes());
        bad_bool_arr.extend(2_u64.to_le_bytes());
        bad_bool_arr.extend([0_u8, 2]);
        let bad_bool_arr_file = single_tensor_gguf(&[(b"arr", 9, &bad_bool_arr)], 32, 0)?;
        assert!(Weights::open(&bad_bool_arr_file.0).is_err());

        // Interior NUL in string metadata rejected.
        let mut nul_str = Vec::new();
        push_kv_str(&mut nul_str, b"hi\0there")?;
        let nul_str_file = single_tensor_gguf(&[(b"s", 8, &nul_str)], 32, 0)?;
        assert!(Weights::open(&nul_str_file.0).is_err());

        // Positive power-of-two INT32 general.alignment accepted; non-positive, non-power-of-two,
        // and > 1_048_576 rejected.
        let positive_i32_align =
            single_tensor_gguf(&[(b"general.alignment", 5, &32_i32.to_le_bytes())], 32, 0)?;
        assert!(Weights::open(&positive_i32_align.0).is_ok());
        for bad_i32 in [0_i32, -32, 3, 2_097_152] {
            let file =
                single_tensor_gguf(&[(b"general.alignment", 5, &bad_i32.to_le_bytes())], 32, 0)?;
            assert!(Weights::open(&file.0).is_err());
        }

        // Non-power-of-two and > 1_048_576 general.alignment rejected; 1_048_576 accepted.
        for bad_align in [0_u32, 3, 2_097_152] {
            let file = single_tensor_gguf(
                &[(b"general.alignment", 4, &bad_align.to_le_bytes())],
                32,
                0,
            )?;
            assert!(Weights::open(&file.0).is_err());
        }
        let max_align = single_tensor_gguf(
            &[(b"general.alignment", 4, &1_048_576_u32.to_le_bytes())],
            1_048_576,
            0,
        )?;
        assert!(Weights::open(&max_align.0).is_ok());

        // Duplicate metadata key rejected.
        let dup_key = single_tensor_gguf(
            &[
                (b"dup", 4, &1_u32.to_le_bytes()),
                (b"dup", 4, &2_u32.to_le_bytes()),
            ],
            32,
            0,
        )?;
        assert!(Weights::open(&dup_key.0).is_err());
        Ok(())
    }

    #[test]
    fn retains_tokenizer_string_arrays_and_requires_tensors_for_weights_open() -> io::Result<()> {
        let mut str_arr = Vec::new();
        str_arr.extend(8_u32.to_le_bytes());
        str_arr.extend(2_u64.to_le_bytes());
        push_kv_str(&mut str_arr, b"alpha")?;
        push_kv_str(&mut str_arr, b"beta")?;

        let mut other_arr = Vec::new();
        other_arr.extend(8_u32.to_le_bytes());
        other_arr.extend(1_u64.to_le_bytes());
        push_kv_str(&mut other_arr, b"ignored")?;

        let file = single_tensor_gguf(
            &[
                (b"tokenizer.ggml.tokens", 9, &str_arr),
                (b"tokenizer.ggml.merges", 9, &str_arr),
                (b"other.strings", 9, &other_arr),
            ],
            32,
            0,
        )?;
        let weights = Weights::open(&file.0)?;
        assert_eq!(weights.string_array("tokenizer.ggml.tokens")?, ["alpha", "beta"]);
        assert_eq!(weights.string_array("tokenizer.ggml.merges")?, ["alpha", "beta"]);
        assert!(weights.string_array("other.strings").is_err());

        let zero_tensor = {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3_u32.to_le_bytes());
            bytes.extend(0_u64.to_le_bytes());
            bytes.extend(0_u64.to_le_bytes());
            write_raw_gguf(&bytes)?
        };
        assert!(Weights::open(&zero_tensor.0).is_err());
        assert!(Weights::open_metadata(&zero_tensor.0).is_ok());
        Ok(())
    }

    fn gguf_header(kv_count: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(b"GGUF");
        out.extend(3_u32.to_le_bytes());
        out.extend(0_u64.to_le_bytes());
        out.extend(kv_count.to_le_bytes());
        out
    }

    #[test]
    fn rejects_invalid_bool_scalar_and_bool_array() {
        let mut valid = gguf_header(2);
        let _ = push_kv_str(&mut valid, b"flag");
        valid.extend(7_u32.to_le_bytes());
        valid.push(1);
        let _ = push_kv_str(&mut valid, b"flags");
        valid.extend(9_u32.to_le_bytes());
        valid.extend(7_u32.to_le_bytes());
        valid.extend(2_u64.to_le_bytes());
        valid.extend([0_u8, 1_u8]);
        let len = u64::try_from(valid.len()).unwrap_or(u64::MAX);
        assert!(parse_index(&mut io::Cursor::new(&valid), len, true, MAX_METADATA_BYTES).is_ok());

        let mut bad_scalar = gguf_header(1);
        let _ = push_kv_str(&mut bad_scalar, b"flag");
        bad_scalar.extend(7_u32.to_le_bytes());
        bad_scalar.push(2);
        let len = u64::try_from(bad_scalar.len()).unwrap_or(u64::MAX);
        assert!(
            parse_index(
                &mut io::Cursor::new(&bad_scalar),
                len,
                true,
                MAX_METADATA_BYTES
            )
            .is_err()
        );

        let mut bad_array = gguf_header(1);
        let _ = push_kv_str(&mut bad_array, b"flags");
        bad_array.extend(9_u32.to_le_bytes());
        bad_array.extend(7_u32.to_le_bytes());
        bad_array.extend(2_u64.to_le_bytes());
        bad_array.extend([0_u8, 255_u8]);
        let len = u64::try_from(bad_array.len()).unwrap_or(u64::MAX);
        assert!(
            parse_index(
                &mut io::Cursor::new(&bad_array),
                len,
                true,
                MAX_METADATA_BYTES
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_non_utf8_embedded_nul_strings_and_duplicate_keys() {
        let mut dup = gguf_header(2);
        for val in [1_u32, 2_u32] {
            let _ = push_kv_str(&mut dup, b"k");
            dup.extend(4_u32.to_le_bytes());
            dup.extend(val.to_le_bytes());
        }
        let len = u64::try_from(dup.len()).unwrap_or(u64::MAX);
        assert!(parse_index(&mut io::Cursor::new(&dup), len, true, MAX_METADATA_BYTES).is_err());

        for bad_key in [b"\xff".as_slice(), b"a\0b".as_slice()] {
            let mut data = gguf_header(1);
            let _ = push_kv_str(&mut data, bad_key);
            data.extend(4_u32.to_le_bytes());
            data.extend(1_u32.to_le_bytes());
            let len = u64::try_from(data.len()).unwrap_or(u64::MAX);
            assert!(
                parse_index(&mut io::Cursor::new(&data), len, true, MAX_METADATA_BYTES).is_err()
            );
        }

        for bad_val in [b"\xfe\xff".as_slice(), b"ok\0tail".as_slice()] {
            let mut data = gguf_header(1);
            let _ = push_kv_str(&mut data, b"s");
            data.extend(8_u32.to_le_bytes());
            let _ = push_kv_str(&mut data, bad_val);
            let len = u64::try_from(data.len()).unwrap_or(u64::MAX);
            assert!(
                parse_index(&mut io::Cursor::new(&data), len, true, MAX_METADATA_BYTES).is_err()
            );
        }

        for bad_elem in [b"\x80".as_slice(), b"tok\0suffix".as_slice()] {
            let mut data = gguf_header(1);
            let _ = push_kv_str(&mut data, b"tokens");
            data.extend(9_u32.to_le_bytes());
            data.extend(8_u32.to_le_bytes());
            data.extend(1_u64.to_le_bytes());
            let _ = push_kv_str(&mut data, bad_elem);
            let len = u64::try_from(data.len()).unwrap_or(u64::MAX);
            assert!(
                parse_index(&mut io::Cursor::new(&data), len, true, MAX_METADATA_BYTES).is_err()
            );
        }
    }

    #[test]
    fn mapped_and_streamed_tensor_f32_match_bits_across_multi_chunk_reads() -> io::Result<()> {
        let count = 20_000_usize; // 80_000 bytes (> 64 KiB chunk boundary)
        let mut payload = Vec::with_capacity(count * 4);
        let mut expected = Vec::with_capacity(count);
        for index in 0..count {
            let value = f32::from(u16::try_from(index % 251).unwrap_or(0)) * 0.25 - 31.0;
            expected.push(value);
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let file = fixture(
            &[("big", &[u64::try_from(count).map_err(|_| invalid())?], 0, 0)],
            &payload,
        )?;
        let mut loaded = Weights::open(&file.0)?;
        let mapped_out = loaded.tensor_f32("big", &[count])?;
        assert_eq!(
            mapped_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        #[cfg(windows)]
        {
            loaded.mapped = None;
        }
        let streamed_out = loaded.tensor_f32("big", &[count])?;
        assert_eq!(
            streamed_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(loaded.stream_buf.len() >= 64 * 1024);
        Ok(())
    }

    #[test]
    fn bitmask_f16_finiteness_and_neon_widening_match_half_to_float_exhaustively() {
        for raw in 0..=u16::MAX {
            let [lo, hi] = raw.to_le_bytes();
            let expected = bonsai_ternary::half_to_float(raw);
            assert_eq!(f16_bytes_finite(lo, hi), expected.is_finite());
        }
        let finite_bits: Vec<u16> = (0..=u16::MAX)
            .filter(|&bits| (bits & 0x7C00) != 0x7C00)
            .collect();
        for chunk in finite_bits.chunks(8) {
            let mut input = [0_u16; 8];
            input[..chunk.len()].copy_from_slice(chunk);
            let [lo, hi] = bonsai_neon_dot::widen_f16_eight(&input);
            let widened: [f32; 8] = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
            for (idx, &bits) in chunk.iter().enumerate() {
                assert_eq!(
                    widened[idx].to_bits(),
                    bonsai_ternary::half_to_float(bits).to_bits()
                );
            }
        }
    }

    #[test]
    fn parallel_packed_validation_and_retained_mapped_view_preserve_exact_semantics()
    -> io::Result<()> {
        let blocks = 130_000_usize; // > 4 MiB Q8_0 payload
        let mut payload = vec![0_u8; blocks * 34];
        for (i, block) in payload.chunks_exact_mut(34).enumerate() {
            block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
            block[2..].fill(i8::try_from(i % 17).unwrap_or(1).to_le_bytes()[0]);
        }
        validate_packed_parallel(8, &payload, 4)?;
        let bad_offset = (blocks - 1) * 34;
        payload[bad_offset..bad_offset + 2].copy_from_slice(&0x7e00_u16.to_le_bytes());
        assert!(validate_packed_parallel(8, &payload, 4).is_err());

        // Verify retained mapped view matches direct Weights reads.
        let mut f16_raw = Vec::with_capacity(18 * 2);
        for i in 0_u16..18 {
            f16_raw.extend_from_slice(&(0x3c00 + i).to_le_bytes());
        }
        let file = fixture(&[("half_vec", &[18], 1, 0)], &f16_raw)?;
        let mut weights = Weights::open(&file.0)?;
        let expected = weights.tensor_f32("half_vec", &[18])?;
        #[cfg(windows)]
        {
            let retained = weights
                .retain_mapped_view()
                .ok_or_else(|| io::Error::other("expected mapped view"))?;
            assert_eq!(
                retained.tensor_f32_mapped("half_vec", &[18])?,
                expected
            );
        }
        Ok(())
    }
}
