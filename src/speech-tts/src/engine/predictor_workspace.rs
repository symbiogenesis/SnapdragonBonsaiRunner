//! Request-owned acoustic decoder capacity; no logical frame state is reused.
#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "Talker shares this private engine workspace without exposing it to library callers."
)]

use std::io;

use super::{LayerKvCache, decoder_workspace::DecoderWorkspace};
use crate::sampling::SamplerScratch;

#[derive(Default)]
pub(crate) struct PredictorWorkspace {
    layers: Vec<LayerKvCache>,
    sampler: SamplerScratch,
    decoder: DecoderWorkspace,
    next_raw: Vec<f32>,
    prefill_in: Vec<f32>,
    prefill_cache: Vec<LayerKvCache>,
    h_a: Vec<f32>,
    h_b: Vec<f32>,
    normed: Vec<f32>,
    logits: Vec<f32>,
}

impl PredictorWorkspace {
    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.k.clear();
            layer.v.clear();
            layer.seq_len = 0;
        }
        for layer in &mut self.prefill_cache {
            layer.k.clear();
            layer.v.clear();
            layer.seq_len = 0;
        }
        self.sampler.clear();
        self.decoder.clear();
        self.next_raw.clear();
        self.prefill_in.clear();
        self.h_a.clear();
        self.h_b.clear();
        self.normed.clear();
        self.logits.clear();
    }

    pub(crate) fn decoder(&mut self) -> &mut DecoderWorkspace {
        &mut self.decoder
    }

    pub(super) fn frame(&mut self, layers: usize) -> io::Result<Frame<'_>> {
        self.reset();
        if self.layers.len() != layers {
            self.layers.clear();
            self.layers.try_reserve(layers).map_err(io::Error::other)?;
            self.layers.resize_with(layers, LayerKvCache::default);
        }
        Ok(Frame(self))
    }
}

pub(super) struct Frame<'a>(&'a mut PredictorWorkspace);

impl Frame<'_> {
    #[cfg(test)]
    pub(super) fn buffers(
        &mut self,
    ) -> (
        &mut [LayerKvCache],
        &mut SamplerScratch,
        &mut DecoderWorkspace,
        &mut Vec<f32>,
        &mut Vec<f32>,
        &mut Vec<f32>,
        &mut Vec<f32>,
        &mut Vec<f32>,
    ) {
        (
            &mut self.0.layers,
            &mut self.0.sampler,
            &mut self.0.decoder,
            &mut self.0.next_raw,
            &mut self.0.h_a,
            &mut self.0.h_b,
            &mut self.0.normed,
            &mut self.0.logits,
        )
    }

    #[expect(
        clippy::type_complexity,
        reason = "Returns disjoint mutable borrows into request-owned frame buffers without wrapper overhead."
    )]
    pub(super) fn all_buffers(
        &mut self,
    ) -> (
        &mut [LayerKvCache],
        &mut SamplerScratch,
        &mut DecoderWorkspace,
        &mut Vec<f32>,
        &mut Vec<f32>,
        &mut Vec<LayerKvCache>,
        &mut Vec<f32>,
        &mut Vec<f32>,
        &mut Vec<f32>,
        &mut Vec<f32>,
    ) {
        (
            &mut self.0.layers,
            &mut self.0.sampler,
            &mut self.0.decoder,
            &mut self.0.next_raw,
            &mut self.0.prefill_in,
            &mut self.0.prefill_cache,
            &mut self.0.h_a,
            &mut self.0.h_b,
            &mut self.0.normed,
            &mut self.0.logits,
        )
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        self.0.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::PredictorWorkspace;
    use std::{io, panic::AssertUnwindSafe};

    #[test]
    fn predictor_workspace_clears_every_frame_on_return_and_unwind_without_reallocating()
    -> io::Result<()> {
        let mut workspace = PredictorWorkspace::default();
        let (allocations, scratch_allocations) = {
            let mut frame = workspace.frame(2)?;
            let (layers, sampler, decoder, next_raw, h_a, h_b, normed, logits) = frame.buffers();
            decoder.norm.extend_from_slice(&[f32::NAN; 16]);
            next_raw.extend_from_slice(&[1.0; 12]);
            h_a.extend_from_slice(&[2.0; 8]);
            h_b.extend_from_slice(&[3.0; 8]);
            normed.extend_from_slice(&[4.0; 8]);
            logits.extend_from_slice(&[5.0; 16]);
            for (index, layer) in layers.iter_mut().enumerate() {
                layer.k.extend_from_slice(&[0x8000, 0x3400, 0x3800, 0x3a00]);
                layer.v.extend_from_slice(&[0x3c00, 0x3e00, 0x3f00, 0x3fc0]);
                layer.seq_len = index + 1;
            }
            sampler.seen_mut().extend_from_slice(&[1; 8]);
            let layer_allocs = layers
                .iter()
                .map(|layer| {
                    (
                        layer.k.as_ptr(),
                        layer.v.as_ptr(),
                        layer.k.capacity(),
                        layer.v.capacity(),
                    )
                })
                .collect::<Vec<_>>();
            let scratch_allocs = [
                (next_raw.as_ptr(), next_raw.capacity()),
                (h_a.as_ptr(), h_a.capacity()),
                (h_b.as_ptr(), h_b.capacity()),
                (normed.as_ptr(), normed.capacity()),
                (logits.as_ptr(), logits.capacity()),
            ];
            (layer_allocs, scratch_allocs)
        };
        assert!(
            workspace
                .layers
                .iter()
                .all(|layer| layer.k.is_empty() && layer.v.is_empty() && layer.seq_len == 0)
        );
        assert!(workspace.sampler.is_empty());
        assert_eq!(workspace.decoder.norm, Vec::<f32>::new());
        assert_eq!(workspace.decoder.norm.capacity(), 16);
        for (vec_ref, &(ptr, cap)) in [
            (&workspace.next_raw, &scratch_allocations[0]),
            (&workspace.h_a, &scratch_allocations[1]),
            (&workspace.h_b, &scratch_allocations[2]),
            (&workspace.normed, &scratch_allocations[3]),
            (&workspace.logits, &scratch_allocations[4]),
        ] {
            assert!(vec_ref.is_empty());
            assert_eq!((vec_ref.as_ptr(), vec_ref.capacity()), (ptr, cap));
        }
        let interrupted = std::panic::catch_unwind(AssertUnwindSafe(|| -> io::Result<()> {
            let mut frame = workspace.frame(2)?;
            let (layers, sampler, decoder, next_raw, h_a, h_b, normed, logits) = frame.buffers();
            assert_eq!(decoder.norm, Vec::<f32>::new());
            decoder.norm.extend_from_slice(&[f32::NAN; 16]);
            assert_eq!(
                [
                    (next_raw.as_ptr(), next_raw.capacity()),
                    (h_a.as_ptr(), h_a.capacity()),
                    (h_b.as_ptr(), h_b.capacity()),
                    (normed.as_ptr(), normed.capacity()),
                    (logits.as_ptr(), logits.capacity()),
                ],
                scratch_allocations
            );
            next_raw.extend_from_slice(&[f32::NAN; 12]);
            h_a.extend_from_slice(&[f32::NAN; 8]);
            h_b.extend_from_slice(&[f32::NAN; 8]);
            normed.extend_from_slice(&[f32::NAN; 8]);
            logits.extend_from_slice(&[f32::NAN; 16]);
            for (layer, allocation) in layers.iter_mut().zip(&allocations) {
                assert_eq!(
                    (
                        layer.k.as_ptr(),
                        layer.v.as_ptr(),
                        layer.k.capacity(),
                        layer.v.capacity()
                    ),
                    *allocation
                );
                layer.k.extend_from_slice(&[0x4000; 4]);
                layer.v.extend_from_slice(&[0x4200; 4]);
                layer.seq_len = 2;
            }
            sampler.seen_mut().extend_from_slice(&[1; 8]);
            std::panic::resume_unwind(Box::new("injected predictor fault"));
        }));
        assert!(interrupted.is_err());
        assert!(
            workspace
                .layers
                .iter()
                .all(|layer| layer.k.is_empty() && layer.v.is_empty() && layer.seq_len == 0)
        );
        assert!(workspace.sampler.is_empty());
        assert_eq!(workspace.decoder.norm, Vec::<f32>::new());
        assert_eq!(workspace.decoder.norm.capacity(), 16);
        for (vec_ref, &(ptr, cap)) in [
            (&workspace.next_raw, &scratch_allocations[0]),
            (&workspace.h_a, &scratch_allocations[1]),
            (&workspace.h_b, &scratch_allocations[2]),
            (&workspace.normed, &scratch_allocations[3]),
            (&workspace.logits, &scratch_allocations[4]),
        ] {
            assert!(vec_ref.is_empty());
            assert_eq!((vec_ref.as_ptr(), vec_ref.capacity()), (ptr, cap));
        }
        drop(workspace.frame(1)?);
        assert_eq!(workspace.layers.len(), 1);
        assert!(workspace.layers[0].k.is_empty() && workspace.layers[0].v.is_empty());
        Ok(())
    }
}
