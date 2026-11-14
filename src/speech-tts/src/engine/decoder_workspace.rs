//! Utterance-owned temporary decoder storage shared by layers and acoustic frames.
#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "Talker and the private predictor workspace share this crate-private storage."
)]

use std::io;

/// Temporary values are overwritten before use; no KV or sampling state lives here.
#[derive(Default)]
pub(crate) struct DecoderWorkspace {
    pub(super) norm: Vec<f32>,
    pub(super) q: Vec<f32>,
    pub(super) k: Vec<f32>,
    pub(super) v: Vec<f32>,
    pub(super) attention: Vec<f32>,
    pub(super) scores: Vec<f32>,
    pub(super) projected: Vec<f32>,
    pub(super) gate: Vec<f32>,
    pub(super) up: Vec<f32>,
}

impl DecoderWorkspace {
    pub(super) fn clear(&mut self) {
        self.norm.clear();
        self.q.clear();
        self.k.clear();
        self.v.clear();
        self.attention.clear();
        self.scores.clear();
        self.projected.clear();
        self.gate.clear();
        self.up.clear();
    }
}

pub(super) fn resize(values: &mut Vec<f32>, length: usize) -> io::Result<()> {
    if length > values.len() {
        values
            .try_reserve_exact(length - values.len())
            .map_err(io::Error::other)?;
    }
    values.resize(length, 0.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::DecoderWorkspace;
    use crate::engine::{LayerKvCache, LinearProjection, QwenDecoderLayer, WeightQuantization};
    use std::io;

    fn buffers(workspace: &mut DecoderWorkspace) -> [&mut Vec<f32>; 9] {
        [
            &mut workspace.norm,
            &mut workspace.q,
            &mut workspace.k,
            &mut workspace.v,
            &mut workspace.attention,
            &mut workspace.scores,
            &mut workspace.projected,
            &mut workspace.gate,
            &mut workspace.up,
        ]
    }

    fn layer(hidden: usize) -> io::Result<QwenDecoderLayer> {
        let projection = |input, output, bias| {
            LinearProjection::new(
                "fixture",
                input,
                output,
                WeightQuantization::F32(vec![0.0; input * output]),
                bias,
            )
        };
        let mut output = projection(2, hidden, None)?;
        if let WeightQuantization::F32(values) = &mut output.weight {
            values[0] = 1.0;
            values[3] = 1.0;
        }
        let mut down_bias = vec![0.0; hidden];
        down_bias[..2].copy_from_slice(&[0.5, -0.25]);
        Ok(QwenDecoderLayer {
            attn_norm_w: vec![1.0; hidden],
            q_proj: projection(hidden, 2, None)?,
            k_proj: projection(hidden, 2, None)?,
            v_proj: projection(hidden, 2, Some(vec![0.25, -0.75]))?,
            o_proj: output,
            q_norm_w: vec![1.0; 2],
            k_norm_w: vec![1.0; 2],
            ffn_norm_w: vec![1.0; hidden],
            gate_proj: projection(hidden, hidden * 2, Some(vec![0.0; hidden * 2]))?,
            up_proj: projection(hidden, hidden * 2, Some(vec![1.0; hidden * 2]))?,
            down_proj: projection(hidden * 2, hidden, Some(down_bias))?,
        })
    }

    fn cache() -> LayerKvCache {
        LayerKvCache {
            k: vec![0.0, 0.0],
            v: vec![-0.5, 0.25],
            seq_len: 1,
        }
    }

    #[test]
    fn decoder_workspace_overwrites_poison_and_reuses_storage_across_widths() -> io::Result<()> {
        let mut workspace = DecoderWorkspace::default();
        for values in buffers(&mut workspace) {
            values.resize(32, f32::NAN);
        }
        let allocations =
            buffers(&mut workspace).map(|values| (values.as_ptr(), values.capacity()));
        for hidden in [4, 2, 4] {
            let layer = layer(hidden)?;
            for workers in [1, 3] {
                let mut input = vec![0.0; hidden];
                input[..2].copy_from_slice(&[0.25, -0.5]);
                let mut cache = cache();
                let output = layer.forward_step_reusing(
                    &input,
                    1,
                    1,
                    2,
                    1_000_000.0,
                    1e-6,
                    &mut cache,
                    None,
                    workers,
                    &mut workspace,
                )?;
                // Zero queries/keys give exactly 1/2 per row. Identity O then
                // adds [-1/8, -1/4]; zero gate and the down bias add [1/2, -1/4].
                assert_eq!(
                    output[..2].iter().map(|n| n.to_bits()).collect::<Vec<_>>(),
                    [0x3f20_0000, 0xbf80_0000]
                );
                assert!(output[2..].iter().all(|value| *value == 0.0));
                assert_eq!(cache.seq_len, 2);
                assert_eq!(cache.k, [0.0; 4]);
                assert_eq!(cache.v, [-0.5, 0.25, 0.25, -0.75]);
                assert_eq!(
                    buffers(&mut workspace).map(|values| (values.as_ptr(), values.capacity())),
                    allocations
                );
                for values in buffers(&mut workspace) {
                    values.fill(f32::NAN);
                }
            }
        }
        workspace.clear();
        assert!(
            buffers(&mut workspace)
                .iter()
                .all(|values| values.is_empty())
        );
        assert_eq!(
            buffers(&mut workspace).map(|values| (values.as_ptr(), values.capacity())),
            allocations
        );
        Ok(())
    }

    #[test]
    fn decoder_workspace_late_failure_preserves_kv_and_recovers_from_poison() -> io::Result<()> {
        let mut workspace = DecoderWorkspace::default();
        let mut layer = layer(2)?;
        let mut cache = cache();
        layer.down_proj.bias = Some(vec![f32::NAN; 2]);
        assert!(
            layer
                .forward_step_reusing(
                    &[0.25, -0.5],
                    1,
                    1,
                    2,
                    1_000_000.0,
                    1e-6,
                    &mut cache,
                    None,
                    1,
                    &mut workspace,
                )
                .is_err()
        );
        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.k, [0.0; 2]);
        assert_eq!(cache.v, [-0.5, 0.25]);
        layer.down_proj.bias = Some(vec![0.5, -0.25]);
        let output = layer.forward_step_reusing(
            &[0.25, -0.5],
            1,
            1,
            2,
            1_000_000.0,
            1e-6,
            &mut cache,
            None,
            1,
            &mut workspace,
        )?;
        assert_eq!(
            output.iter().map(|n| n.to_bits()).collect::<Vec<_>>(),
            [0x3f20_0000, 0xbf80_0000]
        );
        Ok(())
    }
}
