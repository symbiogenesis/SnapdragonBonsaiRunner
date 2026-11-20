# DaisyRose-27B + Swift-Bonsai-2 Hybrid Composition

This contract governs the hybrid composition of **DaisyRose-27B** reasoning,
coding, and instruction-following capability on top of **Swift-Bonsai-2**
(`PQ2_0` / `PTQ1_0`), preserving the benefits of all three methods without
reopening raw BF16 SVD task-vector LoRA extraction (`DR1`) or requiring
high-RAM FP16 execution:

1. **DaisyRose-27B** (`RadixArk/DaisyRose-27B`, revision
   `64a4ec665091bb0c1931babce4e97e686eef3c5c`, Apache-2.0): pure-RL
   post-trained derivative of `Qwen/Qwen3.8-27B` (`qwen35`) bringing major
   coding, tool-use, long-horizon reasoning, and instruction-following gains.
2. **Bonsai** (`PQ2_0` / `PTQ1_0` immutable packed 1.58-bit ternary base,
   `~5.95–7.21 GiB`): preserves compact laptop footprint, Hadamard-folded
   ternary kernels, and Hexagon NPU fused `external_ffn_swiglu` /
   `external_ffn_chain` execution.
3. **Swift** (`ukisai/Swift-Bonsai-2-GGUF`, revision
   `a3bdac086bbb6b04d87d0108d18943b5f63868f7`): preserves anti-overthinking /
   cognitive-loop suppression (`~39.8%` fewer thinking tokens) and orthogonal
   stacking with the pinned Orca abliteration rank-1 adapter (`scale 1.3`).

## Closed no-go routes and physical constraints

- **Direct ternary weight merging is a hard no-go**: `DR1` established that
  `ΔW_DaisyRose` has `maxAbsDelta <= 2^-13` (`~1.22e-4`, relative Frobenius
  delta `0.08%–0.19%`), whereas `PQ2_0` group scales are `s_g ~ 0.015–0.030`
  (`~1,000x` larger than the RL weight delta). Adding `ΔW` in Hadamard space
  and re-rounding trits either flips zero trits or injects `~1,000x` noise.
- **Raw BF16 SVD task-vector LoRA (`DR1`) is closed no-go**: As documented in the
  [Gate 1 audit summary](evidence/daisyrose-task-vector-audit.md), raw BF16
  rank-32 SVD captured only `E(32) = 14.26%` (`< 50%`) because `23.5%–38.9%`
  of coefficients shifted by 1 BF16 ULP, burying coherent spikes inside an
  isotropic Marchenko–Pastur bulk (`~70–85%` of raw diff Frobenius energy).
- **Three-model logit ensembling is out of scope**: Running full-precision
  `DaisyRose-27B` alongside `Swift-Bonsai-2` violates the laptop memory budget;
  quantized `DaisyRose-27B` stays separate under the high-RAM reference boundary
  in [model selection](model-selection.md#selection-boundary).

## Staged NPU-friendly target envelopes

All candidate LoRA targets exclude the 48 `ssm_out.weight` tensors until `P1`
resolves Swift's `prism.hadamard.gdn_v_grouped=true` channel order in
[Swift LoRA](swift-lora.md#structural-compatibility-and-unresolved-basis), and
omit `token_embd.weight` unless explicitly qualified later:

| Envelope | Target sites | Count | F16 MiB / rank | `r = 16` F16 | `r = 32` F16 | Hexagon NPU FFN chains |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `residual-writers` (Stage 1A) | `64× ffn_down` + `16× attn_output` | 80 | `3.09375` | `49.50 MiB` | `99.00 MiB` | All 64 intact |
| `npu-friendly` (Stage 1B, default) | `64× ffn_down` + `16× {attn_q, attn_k, attn_v, attn_output}` | 128 | `4.00000` | `64.00 MiB` | `128.00 MiB` | All 64 intact |
| `full-linear` (Stage 2, fallback) | `64× {ffn_gate, ffn_up, ffn_down}` + `16× {attn_q, attn_k, attn_v, attn_output}` | 256 | `9.50000` | `152.00 MiB` | `304.00 MiB` | Fused NPU FFN off when enabled |

Notes:
- `ffn_down` has shape `[17408, 5120]` (`64` layers), `attn_output` is `[6144, 5120]`,
  `attn_q` is `[12288, 5120]`, and `attn_k`/`attn_v` are `[1024, 5120]` (`16`
  full-attention layers at `(layer + 1) % 4 == 0`), matching `qwen35` metadata in
  [swift.metadata.json](../config/swift.metadata.json).
- Leaving `ffn_gate` and `ffn_up` untouched under `residual-writers` and
  `npu-friendly` preserves Prism's fused NPU `external_ffn_swiglu` execution
  across all 64 Swift layers (`adapter_affects_any`).
- Full-attention `attn_q`/`attn_k`/`attn_v` run on CPU/attention paths and
  enable steering what full-attention heads attend to, at only `+0.90625 MiB/rank`
  above `residual-writers`.

## Hybrid staged strategy

### Track A (`DR-A1`, `DR-A2`): Spectral-denoised fast falsification

`DR-A1` tests whether a genuine low-rank coherent signal core survives above the
BF16 1-ULP quantization bulk when calibrated empirically against a
Rademacher sign-flip null and denoised via **Gavish–Donoho (2017) Frobenius-optimal
singular value shrinkage**:

1. **Rademacher sign-flip null bulk calibration**: For each `ΔW` matrix, apply
   deterministic random ±1 entrywise signs without allocating a second matrix
   and estimate the top singular value `σ_null` of the sign-flipped matrix. Sign
   flipping destroys coherent rank-1..r structure while preserving entrywise
   heteroscedastic BF16 ULP noise variance row/column-wise.
2. **Spike-deflated effective bulk edge**: Iterate 3 fixed-point passes separating
   spike energy `sum_{i < r} σ_i^2` from residual bulk variance, scaling
   `σ_null` by `sqrt(bulk_energy / total_energy)` and taking the max against the
   isotropic Marchenko–Pastur upper edge `σ_+ = σ_bulk * (sqrt(m) + sqrt(n))`.
3. **Gavish–Donoho optimal shrinkage**: Map `y_i = (σ_i / bulk_edge) * (1 + sqrt(β))`
   with `β = min(m, n) / max(m, n)`, hard-threshold `y_i <= 1 + sqrt(β)`, and
   shrink surviving spikes by
   `η_F*(y_i; β) = sqrt((y_i^2 - β - 1)^2 - 4β) / (y_i * (1 + sqrt(β))) * bulk_edge`.
4. **Tail saturation guard**: If `σ_k > effective_bulk_edge` at the maximum
   sketch rank (`tailSaturated = true`), the spectrum has a continuous power-law
   tail exceeding sketch capacity rather than a clean low-rank spike set.
5. **QAT manifold risk and empirical `DR-A1` outcome**: The November 19, 2026
   `DR-A1` probe ([spectral-denoising audit](evidence/daisyrose-spectral-denoising-audit.md))
   measured **81.98%** BF16 bulk energy and genuine **6.75x** mean /
   **14.92x** peak coherent spikes (`E_denoised(16) = 69.21%`, `E_denoised(32) = 85.27%`),
   but returned **`saturated-high-rank-tail`** (`13/16` `npu-friendly` matrices
   saturated at `aboveBulkRank = 64`), closing standalone zero-shot BF16 weight
   transfer (`DR-A2`) as **no-go**. `export-denoised` remains available only as
   an optional rank-16/32 warm-start initializer for Track B OPD (`DR-B1`).

### Track B (`DR-B1`, `DR-B2`): Dual-Anchor On-Policy Distillation (OPD)

**Track B is the primary high-ceiling path** because it trains the adapter
directly on `Swift-Bonsai-2`'s packed/unfolded ternary manifold instead of
assuming tangent linearity from `Qwen3.8-BF16`:

- **Why on-policy OPD works**: `DaisyRose-27B` and `Swift-Bonsai-2` share the
  exact `248,320`-token vocabulary and `qwen35` chat template. Generating
  rollouts directly from `Swift-Bonsai-2` + trainable `npu-friendly` LoRA (`r = 16`
  or `r = 32`, optionally warm-started from Track A denoised spikes) exposes the
  adapter to Swift's true ternary hidden states.
- **Dual-Anchor objective**:
  1. **Capability Reverse-KL**: Token-level `KL(π_θ || π_DaisyRose)` over
     top-64 teacher logprobs across curated coding, tool/agent, math/reasoning,
     and instruction-following trajectories (disjoint from held-out eval).
  2. **Swift Anti-Looping Anchor**: `λ_swift * KL(π_θ || π_Swift)` applied on
     overthinking / reasoning-transition steps plus explicit reasoning-budget
     regularization targeting median thinking tokens `<= 1.10x` unmodified
     `Swift-Bonsai-2`, preventing DaisyRose's 1M-token RL horizon from
     reintroducing pathological cognitive loops.
- **Repository language boundary**: Any GPU OPD training run executes
  externally outside this repository (honoring [AGENTS.md](../AGENTS.md#tooling-languages))
  and yields an immutable Apache-2.0 GGUF LoRA file consumed and independently
  verified in pure Rust here via `bonsai-tools lora-diff-audit` (`verify-adapter`).

## GGUF multi-rank adapter contract

To remain compatible with both `--publisher-engine` (`llama.cpp`) and future
native `prism-inference` `MultiRankAdapter`:

- Container: GGUF v3, little-endian, `general.alignment = 32`, tensor alignment `32`.
- Required metadata:
  - `general.architecture = "qwen35"` (string)
  - `general.type = "adapter"` (string)
  - `adapter.type = "lora"` (string)
  - `adapter.lora.alpha` = `F32` finite positive scalar (`llama.cpp` `LLM_KV_ADAPTER_LORA_ALPHA`)
  - `adapter.lora_alpha` = `F32` bit-identical scalar (`prism-inference` `lora.rs`)
- Required factor layout (GGUF column-major / fastest dimension first):
  - `<base_tensor>.lora_a`: dtype `F16` or `F32`, dimensions `[in_features, r]`
  - `<base_tensor>.lora_b`: dtype `F16` or `F32`, dimensions `[r, out_features]`
  - All values finite; when `F16`, magnitudes strictly below `65520.0`.

## Shared acceptance gates (`DR-G2`, `DR-C1`, `DR-C2`)

No runtime (`prism-inference`), broker, or UI changes happen until **`DR-G2`**
passes on `--publisher-engine` with `--lora-scaled` stacking Orca (`1.3`) +
candidate (`scale in {0.0, 0.25, 0.5, 0.75, 1.0}`):

1. **DaisyRose capability gain**: Measurable Thinking-ON gain on held-out
   coding, reasoning, and instruction-following prompts over `scale = 0.0`.
2. **Swift anti-looping preservation**: Median thinking token count `<= 1.10x`
   `scale = 0.0` Swift baseline, with zero regression in `4,096`-token
   truncation or pathological repetition/loop rates.
3. **Creative & Orca non-regression**: Thinking-OFF creative prose quality,
   conciseness, and Orca refusal behavior match or exceed `scale = 0.0`.
4. **Native runtime (`DR-C1`) & Runner integration (`DR-C2`)**: Stackable opt-in
   `MultiRankAdapter` runtime wiring (`DR-C1` in `prism-inference`) and Runner
   artifact, `bonsai-cpu-policy::context` accounting, canonical broker validation,
   default-off Settings controls, and CLI/prefetch forwarding (`DR-C2`, unpinned)
   are implemented and smoke-vetted locally on a 4-layer `npu-friendly` F16 export
   at `scale = 0.0` (see
   [multi-rank wiring and vet](evidence/daisyrose-multirank-wiring-vet.md)).
   Promoting any DaisyRose adapter to production still requires `DR-B1` / `DR-G2`
   quality approval, Prism release repin in `config/prism-inference.lock.json`,
   pinning `config/daisyrose-lora.lock.json`, live nonzero rollback and matched-power
   X1/X2 device qualification under [validation](validation.md) and
   [Swift LoRA qualification](swift-lora.md#standard-profile-qualification), and
   owning CI (`source-checks`, `foundation`).
