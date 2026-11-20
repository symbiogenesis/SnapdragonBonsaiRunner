# DaisyRose-27B Spectral-Denoising Audit (`DR-A1` Empirical Outcome)

On November 19, 2026, Track A Gate `DR-A1` executed a Rademacher sign-flip null
bulk calibration + Gavish–Donoho (2017) optimal-shrinkage spectral audit on the
16 `npu-friendly` linear projections across layers `[0, 3, 31, 63]` comparing
`RadixArk/DaisyRose-27B` against `Qwen/Qwen3.8-27B` (`qwen35`). The audit
returned **`verdict: "saturated-high-rank-tail"` (`anyTailSaturated: true`)**,
closing standalone zero-shot BF16 weight-transfer export (`DR-A2`) as **no-go**
and confirming **Track B Dual-Anchor On-Policy Distillation (`DR-B1`)** as the
primary composition path in [PLAN.md](../../PLAN.md#daisyrose-27b--swift-bonsai-2-hybrid-composition).

## Findings and Interpretation

- **Empirical bulk separation confirmed**: Across all 16 `npu-friendly` matrices,
  the spike-deflated Rademacher-null / Marchenko–Pastur bulk accounts for
  **81.98%** (`estimatedBulkEnergyFraction = 0.819801`) of raw BF16 diff
  Frobenius energy (`0.214295`), isolating **15.71%** (`denoisedSignalEnergy = 0.033657`,
  `denoisedShareOfRawEnergy = 0.157060`) in shrunk above-bulk singular modes.
- **Genuine coherent spikes exist in all 16/16 matrices**: Every audited matrix
  cleared the calibrated bulk edge (`tensorsWithSpikes = 16/16`), with mean
  spike-to-edge ratio **6.75x** (`6.749750`) and peak **14.92x** (`14.915443`
  on `blk.63.ffn_down.weight`, `sigma_1 = 0.06082` vs `effectiveBulkEdge = 0.00408`).
- **Why standalone zero-shot weight transfer (`DR-A2`) closes no-go**:
  1. **Power-law tail past rank 64 (`saturated-high-rank-tail`)**: In **13 of 16**
     audited matrices, `sigma_64` remained strictly above `effectiveBulkEdge`
     (`aboveBulkRank = 64`, `meanAboveBulkRank = 57.6875`, `tailSaturated = true`).
     Only 3 matrices decayed below the bulk edge within rank 64 (`blk.63.attn_v`
     rank `7`, `blk.3.attn_v` rank `22`, `blk.63.attn_output` rank `62`).
  2. **QAT manifold gap remains**: Even though **69.21%** of the top-64 shrunk
     denoised signal concentrates by rank 16 and **85.27%** by rank 32 (up to
     `81.08%` in `blk.63.ffn_down` and `86.91%` in `blk.63.attn_output` at `r=16`),
     a `Qwen3.8-BF16` tangent delta (`relDelta ~ 0.0005–0.0018`) still cannot
     compensate for the ~35–45% QAT ternary weight gap on `Swift-Bonsai-2` or
     guarantee Swift anti-looping preservation without on-policy adaptation.
- **Actionable value for Track B (`DR-B1`)**: `export-denoised` remains
  available only as an optional rank-16/32 warm-start initializer for Track B
  Dual-Anchor OPD (`DR-B1`), where `Swift-Bonsai-2` activations and dual Reverse-KL
  + Swift anti-looping anchors adapt the full `npu-friendly` subspace directly.

## Per-Tensor Measurements (`npu-friendly`, Layers `0, 3, 31, 63`)

| Tensor (`blk.N.*`) | Shape | Rel Delta | Raw E(32) | `sigma_1` | `sigma_64` | Null Edge | MP Edge | Eff Edge | Above Rank | Tail Sat | Ratio | Bulk Frac | Den E(16) | Den E(32) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `blk.0.ffn_down` | `5120x17408` | 0.001790 | 13.57% | 0.02301 | 0.00550 | 0.00330 | 0.00354 | 0.00354 | 64 | `true` | 6.49x | 83.72% | 66.03% | 83.33% |
| `blk.3.ffn_down` | `5120x17408` | 0.001769 | 10.01% | 0.01832 | 0.00546 | 0.00340 | 0.00369 | 0.00369 | 64 | `true` | 4.97x | 87.39% | 57.96% | 78.78% |
| `blk.3.attn_q` | `12288x5120` | 0.000843 | 11.08% | 0.01323 | 0.00363 | 0.00238 | 0.00246 | 0.00246 | 64 | `true` | 5.39x | 85.81% | 54.95% | 78.11% |
| `blk.3.attn_k` | `1024x5120` | 0.001110 | 18.16% | 0.00468 | 0.00179 | 0.00156 | 0.00162 | 0.00162 | 64 | `true` | 2.88x | 78.82% | 60.61% | 83.14% |
| `blk.3.attn_v` | `1024x5120` | 0.001066 | 23.98% | 0.01045 | 0.00164 | 0.00233 | 0.00166 | 0.00233 | 22 | `false` | 4.49x | 80.70% | 98.49% | 100.00% |
| `blk.3.attn_output` | `5120x6144` | 0.001078 | 21.38% | 0.02787 | 0.00282 | 0.00229 | 0.00224 | 0.00229 | 64 | `true` | 12.19x | 75.79% | 78.72% | 90.79% |
| `blk.31.ffn_down` | `5120x17408` | 0.001682 | 8.34% | 0.02150 | 0.00524 | 0.00343 | 0.00369 | 0.00369 | 64 | `true` | 5.83x | 89.37% | 56.55% | 77.54% |
| `blk.31.attn_q` | `12288x5120` | 0.001172 | 9.52% | 0.01678 | 0.00431 | 0.00280 | 0.00300 | 0.00300 | 64 | `true` | 5.58x | 87.70% | 55.85% | 77.29% |
| `blk.31.attn_k` | `1024x5120` | 0.001191 | 15.59% | 0.00497 | 0.00178 | 0.00167 | 0.00175 | 0.00175 | 64 | `true` | 2.84x | 82.66% | 67.28% | 87.81% |
| `blk.31.attn_v` | `1024x5120` | 0.001008 | 20.08% | 0.00796 | 0.00173 | 0.00171 | 0.00162 | 0.00171 | 64 | `true` | 4.65x | 77.52% | 71.63% | 90.12% |
| `blk.31.attn_output` | `5120x6144` | 0.001159 | 22.27% | 0.02226 | 0.00359 | 0.00225 | 0.00233 | 0.00233 | 64 | `true` | 9.53x | 72.99% | 65.33% | 83.79% |
| `blk.63.ffn_down` | `5120x17408` | 0.001552 | 28.38% | 0.06082 | 0.00524 | 0.00408 | 0.00303 | 0.00408 | 64 | `true` | 14.92x | 68.16% | 81.08% | 91.68% |
| `blk.63.attn_q` | `12288x5120` | 0.001136 | 14.13% | 0.02586 | 0.00392 | 0.00264 | 0.00269 | 0.00269 | 64 | `true` | 9.60x | 82.99% | 66.99% | 83.70% |
| `blk.63.attn_k` | `1024x5120` | 0.001360 | 20.73% | 0.00624 | 0.00182 | 0.00163 | 0.00169 | 0.00169 | 64 | `true` | 3.70x | 76.56% | 67.18% | 87.14% |
| `blk.63.attn_v` | `1024x5120` | 0.000528 | 21.76% | 0.00815 | 0.00130 | 0.00244 | 0.00135 | 0.00244 | 7 | `false` | 3.34x | 87.39% | 100.00% | 100.00% |
| `blk.63.attn_output` | `5120x6144` | 0.000890 | 26.04% | 0.03247 | 0.00277 | 0.00280 | 0.00201 | 0.00280 | 62 | `false` | 11.61x | 70.94% | 86.91% | 95.72% |

Aggregate `npuFriendly` denoised cumulative energy (weighted by shrunk signal energy): `E(1) = 25.77%`, `E(4) = 41.97%`, `E(8) = 54.29%`, `E(16) = 69.21%`, `E(32) = 85.27%` (raw un-denoised `E(16) = 11.81%`, `E(32) = 15.25%`).

## Provenance, Allowance, Power, and Cleanup

- **Upstream pins**: `RadixArk/DaisyRose-27B` revision `64a4ec665091bb0c1931babce4e97e686eef3c5c` vs `Qwen/Qwen3.8-27B` revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Both `model.safetensors.index.json` manifests verified identical SHA-256 `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.
- **Tooling provenance**: Runner base commit `0a60be366be0db6f97d714692cfa3ff8c5b9a8d5` + `DR-A1` spectral-denoising/GGUF extensions, compiler `nightly-2026-11-12`, prebuilt Release `bonsai-tools.exe` (`12,465,152` bytes, SHA-256 `32ac09b365046d31315ed501b8acc1705366529cb9620929b6ba4165d08d702f`), host Snapdragon X2 Elite Extreme `X2E94100` (Windows ARM64). Hosted CI acceptance remains pending.
- **Execution budget and scope**: Predeclared 5-minute (300 s) owner-device allowance (`timeBudgetSeconds: 270`, `workers: 16`, `httpConcurrency: 4`, fixed work: 16 `npu-friendly` tensors across layers `[0, 3, 31, 63]`). Executed once (`completed: true`, `budgetExceeded: false`, `elapsedMilliseconds: 115360`, exit code `0`, `16/16` tensors completed, no retry). Streamed `2,684,685,600` bytes total (`1,342,342,800` candidate + `1,342,342,800` base) via bounded HTTP 206 Range requests without writing model shards to disk.
- **Power (`GetSystemPowerStatus`)**: Before: `ac_online: false`, `battery_saver: false`, `battery_percent: 87`. After: `ac_online: false`, `battery_saver: false`, `battery_percent: 86`. Conditions: `matched-known-endpoints` (`performanceComparisonEligible: true`, `continuousPowerStabilityProven: false`).
- **Evidence selection and cleanup**: Full scratch JSON receipt (`dra1-probe.json`, `118,616` bytes, SHA-256 `a9011db0b752e5092b964ddcb5fbe7a6446657cb26d14f4dd81aabce0dd8bfa2`) was digest-verified, summarized here, and deleted with its temporary request file and scratch directory under [the evidence-retention policy](../validation.md#evidence-retention).
