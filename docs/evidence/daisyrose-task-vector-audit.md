# DaisyRose-27B BF16 Task-Vector Audit (Gate 1 No-Go)

On November 19, 2026, Gate 1 (`DR1`) of the conditional `DaisyRose-27B` task-vector LoRA investigation returned an empirical **No-Go (`no-go-high-rank`)** verdict under [PLAN.md](../../PLAN.md#daisyrose-27b-composition-boundary). Across all 24 FFN and full-attention linear matrices in the 4-layer probe (`blk.0`, `blk.3`, `blk.31`, `blk.63`), energy-weighted cumulative Frobenius energy captured at rank 32 was **14.26%** (`0.142637`), well below the **50%** hard stop threshold (`E(64) = 18.38%`), and only **31.87%** of block linear delta energy sat in `ffn_down + attn_output`. Per the gated plan, `DR2`–`DR4` stopped before any adapter export, `prism-inference`, broker, or UI changes.

## Verdict and Root-Cause Analysis

- **Gate 1 result**: `passed: false`, `verdict: "no-go-high-rank"`.
- **Non-matrix parameters did not dominate**: Pure non-matrix parameters (`norm`, `ssm_conv1d`, `ssm_a`, `ssm_dt_bias`, `ssm_norm`) accounted for **0.0031%** of non-vocab delta energy (`meanTensorRelativeFrobeniusDelta = 4.51e-5`); including small GDN matrices (`ssm_alpha`, `ssm_beta`), non-linear/small parameters held only **0.0360%** of non-vocab energy (`ssm_a`, `ssm_dt.bias`, `ssm_norm`, `attn_q_norm`, and `attn_k_norm` were exact zero delta).
- **Linear projections hold 99.964% of non-vocab energy, with a genuinely flat/high-rank spectrum**: Pure 18-day full-parameter RL updated **23.5%–38.9%** of coefficients across **100% of rows** (`5120/5120` or `17408/17408`) in every block projection.
- **BF16 1-ULP quantization floor and Marchenko-Pastur bulk**: Across all 2,771,442,400 audited elements, every nonzero weight difference had `maxAbsDelta <= 2^{-13} = 0.0001220703125` (exact 1 BF16 ULP for `|w| in [0.015625, 0.03125)`), `relativeFrobeniusDelta` of **0.08%–0.19%**, and zero elements with `|w| >= 0.03125` shifted by even 1 BF16 ULP (`2^{-12}`). Roughly 70–85% of raw BF16 diff Frobenius energy sits in an isotropic 1-ULP rounding bulk (tail singular values `sigma[56..63] ~ 0.0028–0.0058`), while genuine coherent spikes rise 5x–16x above that bulk (`blk.63.ffn_down` `sigma_1 = 0.0608`, `blk.63.attn_output` `sigma_1 = 0.0325`, `blk.3.attn_output` `sigma_1 = 0.0279`) and distribute across all FFN and attention projections (`65.95%` in `ffn_gate` / `ffn_up` / `attn_qkv` / `attn_q` / `attn_k` / `attn_v` / `attn_gate`, outside Candidate A's NPU-compatible `ffn_down + attn_output`).
- **Why SVD LoRA extraction does not transfer RL capability**: Extracting a rank-32 LoRA from raw BF16 differences would discard ~85.7% of block linear task-vector energy (and ~81.6% at rank 64) while penalizing CPU/NPU execution and RAM across targeted layers.

## Subset Aggregates and Per-Family Measurements

| Aggregate | Tensors | Elements | Delta Energy | Agg Rel Delta | Share of Linear | E(16) | E(32) | E(64) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `allAudited` | 49 | 2,771,442,400 | 0.522628 | `1.060e-3` | 107.19% | 13.40% | 18.04% | 23.19% |
| `nonVocabAll` | 48 | 1,500,044,000 | 0.487730 | `1.371e-3` | 100.04% | 10.71% | 14.15% | 18.25% |
| `linearMatrices` | 27 | 1,499,463,680 | 0.487554 | `1.400e-3` | 100.00% | 10.70% | 14.13% | 18.22% |
| `ffnAndFullAttnLinearMatrices` | 24 | 1,384,120,320 | 0.452423 | `1.435e-3` | 92.79% | 10.80% | **14.26%** | 18.38% |
| `candidateA_ffnDownOnly` | 4 | 356,515,840 | 0.127671 | `1.695e-3` | 26.19% | 11.49% | 14.66% | 18.44% |
| `candidateA_ffnDownAndAttnOutput` | 7 | 450,887,680 | 0.155395 | `1.487e-3` | 31.87% | 12.79% | 16.16% | 20.08% |
| `candidateB_gateUpQkvOnly` | 19 | 1,017,118,720 | 0.321525 | `1.373e-3` | 65.95% | 9.71% | 13.19% | 17.38% |
| `gdnLinearMatrices` | 3 | 115,343,360 | 0.035131 | `1.103e-3` | 7.21% | 9.36% | 12.41% | 16.20% |
| `nonMatrixAndSmallGdn` | 21 | 580,320 | `1.758e-4` | `1.282e-4` | 0.036% | 47.36% | 77.72% | 100.00% |
| `pureNonMatrix` | 19 | 88,800 | `1.513e-5` | `3.782e-5` | 0.003% | — | — | — |
| `vocabMatrices` (`token_embd`) | 1 | 1,271,398,400 | 0.034898 | `4.114e-4` | 7.16% | 51.04% | 72.46% | 92.24% |

| Family | Category | Count | Agg Rel Delta | Nonzero Fraction | Share of Linear (Non-Vocab) | E(16) | E(32) | E(64) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_down` | `linear-matrix` | 4 | `1.695e-3` | 36.86% | 26.19% | 11.49% | 14.66% | 18.44% |
| `ffn_up` | `linear-matrix` | 4 | `1.614e-3` | 35.60% | 24.74% | 10.14% | 13.70% | 17.94% |
| `ffn_gate` | `linear-matrix` | 4 | `1.579e-3` | 35.17% | 24.10% | 9.64% | 13.05% | 17.09% |
| `attn_q` | `linear-matrix` | 3 | `1.039e-3` | 27.17% | 10.11% | 8.25% | 11.49% | 15.46% |
| `attn_output` | `linear-matrix` | 3 | `1.035e-3` | 27.72% | 5.69% | 18.79% | 23.05% | 27.65% |
| `attn_qkv` | `linear-matrix` | 1 | `1.066e-3` | 27.54% | 3.01% | 10.37% | 13.62% | 17.60% |
| `ssm_out` | `linear-matrix` | 1 | `1.169e-3` | 30.45% | 2.18% | 9.95% | 12.80% | 16.42% |
| `attn_gate` | `linear-matrix` | 1 | `1.097e-3` | 27.71% | 2.02% | 7.21% | 10.18% | 13.89% |
| `attn_k` | `linear-matrix` | 3 | `1.211e-3` | 31.38% | 1.08% | 12.33% | 18.16% | 25.60% |
| `attn_v` | `linear-matrix` | 3 | `8.019e-4` | 24.52% | 0.89% | 16.67% | 21.96% | 28.74% |
| `ssm_beta` | `gdn-small-matrix` | 1 | `2.035e-3` | 42.02% | 0.021% | 45.28% | 76.44% | 100.00% |
| `ssm_alpha` | `gdn-small-matrix` | 1 | `7.805e-4` | 22.80% | 0.012% | 50.99% | 79.95% | 100.00% |
| `ssm_conv1d` | `non-matrix` | 1 | `5.489e-4` | 42.47% | 0.003% | — | — | — |
| `norm` | `non-matrix` | 15 | `1.338e-5` | 1.79% | 0.0002% | — | — | — |
| `ssm_a` / `ssm_dt_bias` / `ssm_norm` | `non-matrix` | 3 | `0.0` | 0.00% | 0.00% | — | — | — |
| `token_embd` | `vocab-matrix` | 1 | `4.114e-4` | 6.37% | 6.68% of audited | 51.04% | 72.46% | 92.24% |

Notes on `token_embd.weight` and small GDN matrices: `token_embd.weight` had `170,732 / 248,320` nonzero rows and `sketchEnergyRatio = 0.97197` under the single-pass two-sided sketch (`k=80, l=160`); `ssm_alpha` and `ssm_beta` have row dimension `48`, so exact Jacobi Gram SVD captures `100%` of their variance by rank 48.

## Provenance, Allowance, Power, and Cleanup

- **Upstream pins**: `RadixArk/DaisyRose-27B` revision `64a4ec665091bb0c1931babce4e97e686eef3c5c` vs `Qwen/Qwen3.8-27B` revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Both `model.safetensors.index.json` manifests verified identical SHA-256 `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.
- **Tooling provenance**: Runner base commit `3f9e7f5f597f8a78c165bcac009a68a80c1760cd` with local `bonsai-tools lora-diff-audit`, compiler `nightly-2026-11-12`, prebuilt Release `bonsai-tools.exe` (`12,237,312` bytes, SHA-256 `1da4d6164eef31d616d15a79fa62f7a6076ac5ab509a043e66b931501cb5bc7e`), host Snapdragon X2 Elite Extreme `X2E94100` (`18` logical processors, `47.6 GiB` RAM, Windows ARM64). Hosted CI acceptance for `bonsai-tools lora-diff-audit` remains pending.
- **Execution budget and scope**: Predeclared 12-minute (720 s) allowance (`timeBudgetSeconds: 660`, `workers: 16`, `httpConcurrency: 4`). Executed once from `2026-11-19T22:49:58.2049681Z` to `2026-11-19T23:00:58.2910083Z` (`elapsedMilliseconds: 660004`, exit code `0`, no retry). Streamed `14,869,386,784` bytes total (`7,430,499,088` candidate + `7,438,887,696` base) via bounded HTTP 206 Range requests without writing model shards to disk. `49` of `50` selected tensors completed; `output.weight` (`lm_head.weight`) was skipped cleanly when the 660 s budget elapsed (`completed: false`, `budgetExceeded: true`). All 24/24 FFN and full-attention tensors in `layers: [0, 3, 31, 63]` completed before the cutoff.
- **Power (`GetSystemPowerStatus`)**: Before: `ac_online: true`, `battery_saver: false`, `battery_percent: 100`. After: `ac_online: true`, `battery_saver: false`, `battery_percent: 99`. Conditions: `matched-known-endpoints` (`performanceComparisonEligible: true`, `continuousPowerStabilityProven: false`).
- **Evidence selection and cleanup**: Full scratch JSON receipt (`dr1-probe.json`, `142,396` bytes, SHA-256 `51f7f8876ec7cfbd2563e10a209541ba224cc9f6dc5823937191cd4e8cb92d95`) was digest-verified, summarized here, and deleted with its temporary scratch directory under [the evidence-retention policy](../validation.md#evidence-retention) because `DR2`–`DR4` will not run.
