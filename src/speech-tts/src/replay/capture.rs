//! Bounded serialization of original, independently supplied TTS observations.

use std::io::{self, BufWriter, Write};

use super::{Binding, Tolerance, Tolerances, WireSink, check_cancel, invalid, valid_header_inputs};
use crate::{
    engine::{LayerKvCache, TOTAL_CODE_GROUPS},
    observation::{Observation, Phase, PredictorObservation, TalkerObservation},
};

/// Independently reviewed decoder geometry; all layers and vocabulary entries are mandatory.
#[derive(Clone, Copy)]
pub struct DecoderLayout {
    /// Width of every normalized hidden row.
    pub hidden: usize,
    /// Complete output vocabulary size.
    pub vocabulary: usize,
    /// Complete transformer layer count.
    pub layers: usize,
    /// Key/value scalars per committed position.
    pub kv_width: usize,
}

impl DecoderLayout {
    fn valid(self) -> bool {
        (1..=4096).contains(&self.hidden)
            && (1..=8192).contains(&self.vocabulary)
            && (1..=64).contains(&self.layers)
            && (1..=16_384).contains(&self.kv_width)
    }
}

/// Source/model-bound dimensions for both Talker and residual decoder graphs.
#[derive(Clone, Copy)]
pub struct GraphLayout {
    /// Semantic decoder geometry.
    pub talker: DecoderLayout,
    /// Residual acoustic decoder geometry, shared by all fifteen codebooks.
    pub predictor: DecoderLayout,
    /// Model-authoritative semantic end marker.
    pub eos: i32,
}

impl GraphLayout {
    fn valid(self) -> bool {
        self.talker.valid()
            && self.predictor.valid()
            && usize::try_from(self.eos).is_ok_and(|id| id < self.talker.vocabulary)
    }
}

/// Reviewed complete capture geometry, supplied before executing either provider.
#[derive(Clone, Copy)]
pub struct CaptureLayout {
    /// Cold reference-generation graph.
    pub design: GraphLayout,
    /// Utterance graph, including warm retained-voice requests.
    pub base: GraphLayout,
    /// Complete Base speaker embedding width.
    pub speaker_width: usize,
    /// Actual codec output rate.
    pub sample_rate: u32,
}

/// Borrowed complete prompt rows, without copying an original provider's tensor.
#[derive(Clone, Copy)]
pub enum Rows<'a> {
    /// Original contiguous time-major tensor and its reviewed row width.
    Contiguous {
        /// Every original row in time-major order.
        values: &'a [f32],
        /// Scalars per row; zero or a partial final row rejects.
        width: usize,
    },
    /// Existing Rust serving rows, borrowed without allocating another tensor.
    Rust(&'a [Vec<f32>]),
}

impl Rows<'_> {
    fn count(self, width: usize) -> io::Result<usize> {
        match self {
            Self::Contiguous {
                values,
                width: actual,
            } if actual == width && actual != 0 && values.len().is_multiple_of(actual) => {
                Ok(values.len() / actual)
            }
            Self::Rust(rows) if rows.len() <= 4096 && rows.iter().all(|row| row.len() == width) => {
                Ok(rows.len())
            }
            _ => Err(invalid()),
        }
    }
}

/// One original complete committed cache layer in position/head/channel order.
#[derive(Clone, Copy)]
pub struct LayerCache<'a> {
    /// Original committed keys.
    pub keys: &'a [f32],
    /// Original committed values.
    pub values: &'a [f32],
    /// Complete position count, including the current position.
    pub positions: usize,
}

/// Complete borrowed cache, independent of an inference engine's owned container.
#[derive(Clone, Copy)]
pub enum Cache<'a> {
    /// Views supplied by a reviewed original observation provider.
    Original(&'a [LayerCache<'a>]),
    /// Existing Rust serving storage, without copying any complete cache.
    Rust(&'a [LayerKvCache]),
}

#[derive(Clone, Copy)]
enum LayerView<'a> {
    F32(LayerCache<'a>),
    F16 {
        keys: &'a [u16],
        values: &'a [u16],
        positions: usize,
    },
}

impl<'a> Cache<'a> {
    const fn len(self) -> usize {
        match self {
            Self::Original(layers) => layers.len(),
            Self::Rust(layers) => layers.len(),
        }
    }

    fn view(self, index: usize) -> LayerView<'a> {
        match self {
            Self::Original(layers) => LayerView::F32(layers[index]),
            Self::Rust(layers) => LayerView::F16 {
                keys: &layers[index].k,
                values: &layers[index].v,
                positions: layers[index].seq_len,
            },
        }
    }
}

/// One complete original numerical event, without native/Rust engine ownership.
#[derive(Clone, Copy)]
pub enum CaptureEvent<'a> {
    /// Complete prompt and trailing text rows before prefill.
    Prompt {
        /// Model role.
        phase: Phase,
        /// Complete projected prompt rows.
        rows: Rows<'a>,
        /// Complete trailing overlays.
        trailing: Rows<'a>,
        /// Exact-prefix candidate count before the byte bound is applied.
        cacheable_rows: usize,
    },
    /// Raw semantic or acoustic output, before sampler mutation.
    Logits {
        /// Model role.
        phase: Phase,
        /// Contiguous acoustic frame index, including the final EOS frame.
        frame: usize,
        /// None for semantic output, otherwise residual codebook 0 through 14.
        codebook: Option<usize>,
        /// Complete final normalized hidden row.
        hidden: &'a [f32],
        /// Every original vocabulary logit.
        logits: &'a [f32],
        /// Every layer's complete committed keys and values.
        cache: Cache<'a>,
    },
    /// Original seeded semantic or acoustic selection.
    Decision {
        /// Model role.
        phase: Phase,
        /// Owning acoustic frame index.
        frame: usize,
        /// None for semantic output, otherwise residual codebook 0 through 14.
        codebook: Option<usize>,
        /// Original selected token.
        token: i32,
        /// True only for the final semantic end marker.
        eos: bool,
    },
    /// Complete acoustic decoder state after its initial hidden-row prefill.
    PredictorPrefill {
        /// Model role.
        phase: Phase,
        /// Owning acoustic frame index.
        frame: usize,
        /// Every committed layer at position zero.
        cache: Cache<'a>,
    },
    /// Complete semantic plus fifteen acoustic codes before codec delivery.
    Frame {
        /// Model role.
        phase: Phase,
        /// Contiguous frame index.
        frame: usize,
        /// Original sampled code groups in order.
        codes: &'a [i32; TOTAL_CODE_GROUPS],
    },
    /// Complete selected Base speaker embedding.
    Speaker(&'a [f32]),
    /// Complete retained reference conditioning in codebook order.
    ReferenceCodes(&'a [[i32; TOTAL_CODE_GROUPS]]),
    /// Validated original mono PCM before delivery or Design teardown.
    Pcm {
        /// Model role.
        phase: Phase,
        /// Complete waveform or Base streaming chunk.
        samples: &'a [f32],
    },
    /// Provisional completion; successful provider return is separately required.
    Complete {
        /// Complete delivered Base sample count.
        samples: usize,
        /// Actual codec output rate.
        sample_rate: u32,
    },
}

impl<'a> From<Observation<'a>> for CaptureEvent<'a> {
    fn from(event: Observation<'a>) -> Self {
        match event {
            Observation::Talker { phase, observation } => match observation {
                TalkerObservation::Prompt {
                    rows,
                    trailing,
                    cacheable_rows,
                } => Self::Prompt {
                    phase,
                    rows: Rows::Rust(rows),
                    trailing: Rows::Rust(trailing),
                    cacheable_rows,
                },
                TalkerObservation::Logits {
                    frame,
                    hidden,
                    logits,
                    cache,
                } => Self::Logits {
                    phase,
                    frame,
                    codebook: None,
                    hidden,
                    logits,
                    cache: Cache::Rust(cache),
                },
                TalkerObservation::Decision { frame, token, eos } => Self::Decision {
                    phase,
                    frame,
                    codebook: None,
                    token,
                    eos,
                },
                TalkerObservation::Predictor { frame, observation } => match observation {
                    PredictorObservation::Prefill { cache } => Self::PredictorPrefill {
                        phase,
                        frame,
                        cache: Cache::Rust(cache),
                    },
                    PredictorObservation::Logits {
                        codebook,
                        hidden,
                        logits,
                        cache,
                    } => Self::Logits {
                        phase,
                        frame,
                        codebook: Some(codebook),
                        hidden,
                        logits,
                        cache: Cache::Rust(cache),
                    },
                    PredictorObservation::Decision { codebook, token } => Self::Decision {
                        phase,
                        frame,
                        codebook: Some(codebook),
                        token,
                        eos: false,
                    },
                },
                TalkerObservation::Frame { frame, codes } => Self::Frame {
                    phase,
                    frame,
                    codes,
                },
            },
            Observation::Speaker(values) => Self::Speaker(values),
            Observation::ReferenceCodes(frames) => Self::ReferenceCodes(frames),
            Observation::Pcm { phase, samples } => Self::Pcm { phase, samples },
            Observation::Complete {
                samples,
                sample_rate,
            } => Self::Complete {
                samples,
                sample_rate,
            },
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Prompt,
    SemanticLogits,
    SemanticDecision,
    PredictorPrefill,
    PredictorLogits,
    PredictorDecision,
    Frame,
    Pcm,
    Speaker,
    ReferenceCodes,
    Complete,
    Terminal,
    Failed,
}

/// Single-use bounded writer of complete original `TTSREPL1` observations.
///
/// This writer executes no inference and authenticates no provider. Borrowed
/// views support independently supplied original tensors without copying full
/// prompts/caches/audio. It retains one 16 KiB output buffer and sixteen code IDs.
/// The caller reviews bindings/geometry, retains original artifact/input guards
/// and bounds output I/O within the original absolute workload allowance.
/// Cooperative cancellation cannot interrupt a blocked `Write`.
///
/// Only [`Self::finish`] may emit Terminal after actual provider success. Earlier
/// failure, refusal, cancellation or unwinding is sticky. Retained partial bytes
/// must never be admitted as complete, even if a Complete record was observed.
#[must_use]
pub struct CaptureWriter<W: Write> {
    output: BufWriter<W>,
    layout: CaptureLayout,
    phase: Phase,
    stage: Stage,
    frame: usize,
    prompt_positions: usize,
    codebook: usize,
    codes: [i32; TOTAL_CODE_GROUPS],
    samples: usize,
    events: usize,
}

impl<W: Write> WireSink for CaptureWriter<W> {
    fn bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.output.write_all(bytes)
    }

    fn float(&mut self, value: f32, _tolerance: Tolerance) -> io::Result<()> {
        self.output.write_all(&value.to_le_bytes())
    }
}

impl<W: Write> CaptureWriter<W> {
    /// Write the reviewed provider/model/request/tolerance-bound packet header.
    ///
    /// # Errors
    /// Rejects invalid identities, arithmetic, geometry, rules or output I/O.
    pub fn new(
        output: W,
        binding: &Binding<'_>,
        rules: Tolerances,
        layout: CaptureLayout,
    ) -> io::Result<Self> {
        if !layout.design.valid()
            || !layout.base.valid()
            || !(1..=4096).contains(&layout.speaker_width)
            || !crate::is_supported_sample_rate(layout.sample_rate)
            || !valid_header_inputs(binding, rules)
        {
            return Err(invalid());
        }
        let mut capture = Self {
            output: BufWriter::with_capacity(16 * 1024, output),
            layout,
            phase: if binding.retained_voice {
                Phase::Base
            } else {
                Phase::Design
            },
            stage: if binding.retained_voice {
                Stage::Speaker
            } else {
                Stage::Prompt
            },
            frame: 0,
            prompt_positions: 0,
            codebook: 0,
            codes: [0; TOTAL_CODE_GROUPS],
            samples: 0,
            events: 0,
        };
        capture.header(binding, rules)?;
        Ok(capture)
    }

    /// Serialize one complete original observation, preserving all original F32 bits.
    ///
    /// Use `CaptureEvent::from(event)` for borrowed Rust observations; doing so
    /// captures Rust execution only, never an independent native reference.
    ///
    /// # Errors
    /// Rejects incomplete/reordered geometry or graph/phase/frame/codebook state,
    /// invalid decisions/PCM, cancellation and I/O. Failure/unwind is sticky.
    pub fn observe(
        &mut self,
        event: CaptureEvent<'_>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        if matches!(self.stage, Stage::Failed | Stage::Terminal) {
            self.stage = Stage::Failed;
            return Err(invalid());
        }
        let stage = self.stage;
        self.stage = Stage::Failed;
        check_cancel(cancel)?;
        let next = self.event(stage, event, cancel)?;
        check_cancel(cancel)?;
        self.events = self.events.checked_add(1).ok_or_else(invalid)?;
        self.stage = next;
        Ok(())
    }

    /// Publish Terminal only after the actual observed provider returned successfully.
    ///
    /// Pass its real result after provider cleanup, rather than a constructed
    /// success. The writer cannot authenticate that result or the claimed source.
    /// Flush and final cancellation checks must pass before retaining the packet.
    ///
    /// # Errors
    /// Propagates provider failure, missing Complete/EOS, earlier refusal,
    /// cancellation, terminal-write and flush errors.
    pub fn finish(
        mut self,
        result: io::Result<()>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<W> {
        result?;
        if self.stage != Stage::Terminal {
            return Err(invalid());
        }
        self.stage = Stage::Failed;
        check_cancel(cancel)?;
        self.output.write_all(&[255])?;
        self.number(self.events)?;
        self.output.flush()?;
        check_cancel(cancel)?;
        self.output
            .into_inner()
            .map_err(io::IntoInnerError::into_error)
    }

    fn graph(&self) -> GraphLayout {
        if self.phase == Phase::Design {
            self.layout.design
        } else {
            self.layout.base
        }
    }

    fn record(&mut self, tag: u8, phase: Phase) -> io::Result<()> {
        if phase != self.phase {
            return Err(invalid());
        }
        self.record_tag(tag, phase)
    }

    fn event(
        &mut self,
        stage: Stage,
        event: CaptureEvent<'_>,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<Stage> {
        let graph = self.graph();
        match event {
            CaptureEvent::Prompt {
                phase,
                rows,
                trailing,
                cacheable_rows,
            } if stage == Stage::Prompt => {
                let count = rows.count(graph.talker.hidden)?;
                if !(1..=4096).contains(&count)
                    || cacheable_rows > count
                    || trailing.count(graph.talker.hidden)? > 4096
                {
                    return Err(invalid());
                }
                self.record(1, phase)?;
                self.rows(rows, graph.talker.hidden, cancel)?;
                self.rows(trailing, graph.talker.hidden, cancel)?;
                self.number(cacheable_rows)?;
                self.prompt_positions = count;
                Ok(Stage::SemanticLogits)
            }
            CaptureEvent::Logits {
                phase,
                frame,
                codebook,
                hidden,
                logits,
                cache,
            } if frame == self.frame
                && ((stage == Stage::SemanticLogits && codebook.is_none())
                    || (stage == Stage::PredictorLogits && codebook == Some(self.codebook))) =>
            {
                let decoder = if codebook.is_some() {
                    graph.predictor
                } else {
                    graph.talker
                };
                let positions = if codebook.is_some() {
                    self.codebook + 2
                } else {
                    self.prompt_positions + self.frame
                };
                if positions > 4096
                    || hidden.len() != decoder.hidden
                    || logits.len() != decoder.vocabulary
                {
                    return Err(invalid());
                }
                self.record(if codebook.is_some() { 5 } else { 2 }, phase)?;
                self.number(frame)?;
                if let Some(codebook) = codebook {
                    self.number(codebook)?;
                }
                self.floats(hidden, cancel)?;
                self.floats(logits, cancel)?;
                self.cache(cache, decoder, positions, cancel)?;
                Ok(if codebook.is_some() {
                    Stage::PredictorDecision
                } else {
                    Stage::SemanticDecision
                })
            }
            CaptureEvent::Decision {
                phase,
                frame,
                codebook,
                token,
                eos,
            } if frame == self.frame
                && ((stage == Stage::SemanticDecision && codebook.is_none())
                    || (stage == Stage::PredictorDecision && codebook == Some(self.codebook))) =>
            {
                let id = usize::try_from(token).map_err(|_| invalid())?;
                if codebook.is_some() {
                    if eos || id >= graph.predictor.vocabulary {
                        return Err(invalid());
                    }
                    self.record(6, phase)?;
                    self.number(frame)?;
                    self.number(self.codebook)?;
                    self.output.write_all(&token.to_le_bytes())?;
                    self.codes[self.codebook + 1] = token;
                    self.codebook += 1;
                    Ok(if self.codebook == TOTAL_CODE_GROUPS - 1 {
                        Stage::Frame
                    } else {
                        Stage::PredictorLogits
                    })
                } else {
                    if id >= graph.talker.vocabulary
                        || eos != (token == graph.eos)
                        || (!eos && id >= graph.predictor.vocabulary)
                        || (eos && self.frame == 0)
                    {
                        return Err(invalid());
                    }
                    self.record(3, phase)?;
                    self.number(frame)?;
                    self.output.write_all(&token.to_le_bytes())?;
                    self.output.write_all(&[u8::from(eos)])?;
                    self.codes[0] = token;
                    self.codebook = 0;
                    Ok(if eos {
                        if self.phase == Phase::Design {
                            Stage::Pcm
                        } else {
                            Stage::Complete
                        }
                    } else {
                        Stage::PredictorPrefill
                    })
                }
            }
            CaptureEvent::PredictorPrefill {
                phase,
                frame,
                cache,
            } if stage == Stage::PredictorPrefill && frame == self.frame => {
                self.record(4, phase)?;
                self.number(frame)?;
                self.cache(cache, graph.predictor, 1, cancel)?;
                Ok(Stage::PredictorLogits)
            }
            CaptureEvent::Frame {
                phase,
                frame,
                codes,
            } if stage == Stage::Frame && frame == self.frame && codes == &self.codes => {
                self.record(7, phase)?;
                self.number(frame)?;
                self.codes(codes)?;
                self.frame += 1;
                if self.frame >= 1500 {
                    return Err(invalid());
                }
                Ok(if self.phase == Phase::Design {
                    Stage::SemanticLogits
                } else {
                    Stage::Pcm
                })
            }
            CaptureEvent::Pcm { phase, samples } if stage == Stage::Pcm => {
                let rate = usize::try_from(self.layout.sample_rate).map_err(|_| invalid())?;
                let total = self
                    .samples
                    .checked_add(samples.len())
                    .ok_or_else(invalid)?;
                let maximum = if self.phase == Phase::Design {
                    rate * 30
                } else {
                    let hop = rate / 12;
                    (rate * 120).min((1500 - 2) * hop - 1)
                };
                if samples.is_empty() || total > maximum {
                    return Err(invalid());
                }
                for chunk in samples.chunks(4096) {
                    check_cancel(cancel)?;
                    if chunk.iter().any(|value| !value.is_finite()) {
                        return Err(invalid());
                    }
                }
                self.record(10, phase)?;
                self.floats(samples, cancel)?;
                if self.phase == Phase::Design {
                    self.phase = Phase::Base;
                    self.frame = 0;
                    Ok(Stage::Speaker)
                } else {
                    self.samples = total;
                    Ok(Stage::SemanticLogits)
                }
            }
            CaptureEvent::Speaker(values)
                if stage == Stage::Speaker && values.len() == self.layout.speaker_width =>
            {
                self.output.write_all(&[8])?;
                self.floats(values, cancel)?;
                Ok(Stage::ReferenceCodes)
            }
            CaptureEvent::ReferenceCodes(frames) if stage == Stage::ReferenceCodes => {
                if frames.is_empty()
                    || frames.len() > 375
                    || frames.iter().flatten().any(|id| !(0..2048).contains(id))
                {
                    return Err(invalid());
                }
                self.reference_codes(frames, cancel)?;
                Ok(Stage::Prompt)
            }
            CaptureEvent::Complete {
                samples,
                sample_rate,
            } if stage == Stage::Complete
                && samples == self.samples
                && samples != 0
                && sample_rate == self.layout.sample_rate =>
            {
                self.output.write_all(&[11])?;
                self.number(samples)?;
                self.output.write_all(&sample_rate.to_le_bytes())?;
                Ok(Stage::Terminal)
            }
            _ => Err(invalid()),
        }
    }

    fn cache(
        &mut self,
        cache: Cache<'_>,
        layout: DecoderLayout,
        positions: usize,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        if cache.len() != layout.layers {
            return Err(invalid());
        }
        let values = positions.checked_mul(layout.kv_width).ok_or_else(invalid)?;
        self.number(cache.len())?;
        for index in 0..cache.len() {
            check_cancel(cancel)?;
            self.number(index)?;
            self.number(positions)?;
            match cache.view(index) {
                LayerView::F32(layer) => {
                    if layer.positions != positions
                        || layer.keys.len() != values
                        || layer.values.len() != values
                    {
                        return Err(invalid());
                    }
                    WireSink::floats(self, layer.keys, Tolerance::Bits, cancel)?;
                    WireSink::floats(self, layer.values, Tolerance::Bits, cancel)?;
                }
                LayerView::F16 {
                    keys,
                    values: v_bits,
                    positions: layer_positions,
                } => {
                    if layer_positions != positions
                        || keys.len() != values
                        || v_bits.len() != values
                    {
                        return Err(invalid());
                    }
                    self.floats_f16(keys, Tolerance::Bits, cancel)?;
                    self.floats_f16(v_bits, Tolerance::Bits, cancel)?;
                }
            }
        }
        Ok(())
    }

    fn floats(&mut self, values: &[f32], cancel: &mut impl FnMut() -> bool) -> io::Result<()> {
        WireSink::floats(self, values, Tolerance::Bits, cancel)
    }

    fn rows(
        &mut self,
        rows: Rows<'_>,
        width: usize,
        cancel: &mut impl FnMut() -> bool,
    ) -> io::Result<()> {
        self.number(rows.count(width)?)?;
        match rows {
            Rows::Contiguous { values, .. } => {
                for row in values.chunks(width) {
                    check_cancel(cancel)?;
                    self.floats(row, cancel)?;
                }
            }
            Rows::Rust(rows) => {
                for row in rows {
                    check_cancel(cancel)?;
                    self.floats(row, cancel)?;
                }
            }
        }
        Ok(())
    }
}
