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
   default-off Settings controls, CLI/prefetch forwarding, gated-off download
   provenance/provisioning/`chat_launch::setup`/`--no-daisyrose-lora` plumbing,
   promoted default-on broker/UI plumbing (promoted-pin auto-resolution,
   lock-digest verification, tight-RAM shedding, default-scale request policy
   and readiness reporting, and fresh/new-conversation UI default initialization
   preserving saved workspaces; `DR-C2`, with `pinned_daisyrose_lora_artifact`
   returning `Ok(None)` and `MULTI_RANK_LORA_RELEASE = None`), and pure-Rust
   offline readiness tooling (`bonsai-tools lora-diff-audit` operations
   `generate-test-adapter`, `generate-lock`, `dr-g2-plan`, and `dr-g2-score`)
   are implemented. Runtime wiring is vetted locally on both a 4-layer
   `npu-friendly` F16 export at `scale = 0.0` (battery) and an offline AC-powered
   4-layer `r = 16` synthetic fixture covering `:0.0` bit-exact parity, `:0.5`
   nonzero logits perturbation, matched-AC 4-layer overhead, and `:0.5`
   full-state `rollback` (`status: "passed"` in `33.875 s`; see
   [multi-rank wiring and vet](evidence/daisyrose-multirank-wiring-vet.md)).
   Promoting DaisyRose to default-on still requires `DR-B1` (external Track B
   training), `DR-G2` + held-out human review, Prism `v0.1.33+` publication and
   repin in `config/prism-inference.lock.json` (`MULTI_RANK_LORA_RELEASE`),
   pinning `config/daisyrose-lora.lock.json`, matched-power full-adapter X1/X2
   device qualification under [validation](validation.md) and
   [Swift LoRA qualification](swift-lora.md#standard-profile-qualification), and
   owning CI (`source-checks`, `foundation`).

## External Track B (`DR-B1`) training handoff and promotion checklist

### Pinned base, teacher, and stacked adapter identities

- **Student base (`Swift-Bonsai-2`)**: `ukisai/Swift-Bonsai-2-GGUF`, revision
  `a3bdac086bbb6b04d87d0108d18943b5f63868f7`, `Swift-Bonsai-2-PQ2_0.gguf`
  (`7,206,168,928` bytes, SHA-256
  `5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5`,
  `248,320`-token vocabulary, `qwen35` chat template in
  [swift.metadata.json](../config/swift.metadata.json)).
- **Teacher (`DaisyRose-27B`)**: `RadixArk/DaisyRose-27B`, revision
  `64a4ec665091bb0c1931babce4e97e686eef3c5c` (`Apache-2.0`).
- **Stacked rank-1 Orca adapter**: `Continuum-AI-Corp/OrcaBonsai-27B-Uncensored`,
  revision `703028bb7fba03496e796b144839b161cf0f5409`,
  `gguf/bonsai-abliterate-lora.gguf` (`9,682,464` bytes, SHA-256
  `f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42`, served at
  `scale = 1.3`).

### Exact 128-target `npu-friendly` HF-to-GGUF mapping

| Hugging Face tensor pattern | GGUF base name | Sites | GGUF `lora_a` | GGUF `lora_b` | Scope |
| --- | --- | ---: | --- | --- | --- |
| `model.language_model.layers.{L}.mlp.down_proj.weight` | `blk.{L}.ffn_down.weight` | 64 | `[17408, r]` | `[r, 5120]` | `L = 0..63` |
| `model.language_model.layers.{L}.self_attn.q_proj.weight` | `blk.{L}.attn_q.weight` | 16 | `[5120, r]` | `[r, 12288]` | `(L + 1) % 4 == 0` |
| `model.language_model.layers.{L}.self_attn.k_proj.weight` | `blk.{L}.attn_k.weight` | 16 | `[5120, r]` | `[r, 1024]` | `(L + 1) % 4 == 0` |
| `model.language_model.layers.{L}.self_attn.v_proj.weight` | `blk.{L}.attn_v.weight` | 16 | `[5120, r]` | `[r, 1024]` | `(L + 1) % 4 == 0` |
| `model.language_model.layers.{L}.self_attn.o_proj.weight` | `blk.{L}.attn_output.weight` | 16 | `[6144, r]` | `[r, 5120]` | `(L + 1) % 4 == 0` |

Forbidden targets excluded by `verify-adapter` (`requireNpuFriendly: true`,
`requireNoGroupedSsm: true`): `ffn_gate`, `ffn_up` (would disable Hexagon NPU
`external_ffn_swiglu`), `ssm_out` (48 layers blocked until `P1` resolves
`prism.hadamard.gdn_v_grouped=true`), `token_embd`, `output`, `ssm_alpha`,
`ssm_beta`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, and all norms.

### External Track B recipe and split-disjointness discipline

1. Optionally generate a rank-16/32 warm-start `.gguf` with
   `bonsai-tools lora-diff-audit` `operation: "export-denoised"` (never promoted
   standalone per `DR-A1`).
2. Outside this repository on GPU, sample on-policy rollouts directly from
   `Swift-Bonsai-2` + trainable `npu-friendly` `F16` LoRA (`r = 16` or `32`),
   optimize teacher top-64 Reverse-KL against `RadixArk/DaisyRose-27B` anchored
   by `Swift-Bonsai-2` anti-looping KL and reasoning-budget regularization
   targeting `<= 1.10x` Swift median thinking tokens.
3. Export GGUF v3 (`general.alignment = 32`, `F16`, finite scalars with
   `|v| < 65520.0`, equal `adapter.lora.alpha` and `adapter.lora_alpha`) under an
   immutable `Apache-2.0` commit/revision, and record the SHA-256 digest of every
   training prompt (`trainingPromptSha256s`) to enforce disjointness against
   `DR-G2` held-out prompts.

### Pure-Rust candidate intake, `DR-G2` screen, and lock generation

```powershell
# 1. Verify GGUF v3 layout, finiteness, dual alpha equality, and NPU/SSM bounds
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"verify-adapter","adapterPath":"C:/External/daisyrose-r16.gguf","verifyPolicy":{"requireNpuFriendly":true,"requireNoGroupedSsm":true,"maxAllowedRank":32}}'

# 2. Emit canonical DR-G2 contract + evidenceTemplate (then capture --publisher-engine three-way + human review)
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"dr-g2-plan","screenConfig":{"candidateAdapterSha256":"<CANDIDATE_SHA256>","candidateMaxRank":16,"candidateTargetCount":128}}'

# 3. Score completed held-out evidence + human reviews fail-closed
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"dr-g2-score","evidencePath":"C:/External/dr-g2-evidence.json"}'

# 4. After DR-G2 passes, write schema-validated candidate lock JSON (refuses synthetic fixtures when defaultEnabled=true)
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"generate-lock","adapterPath":"C:/External/daisyrose-r16.gguf","outputPath":"C:/External/daisyrose-lora.lock.json","lockSource":{"host":"huggingface","repository":"<OWNER>/<REPO>","revision":"<40_HEX>","path":"<POSIX_PATH>","filename":"<FILE>.gguf","url":"https://huggingface.co/<OWNER>/<REPO>/resolve/<40_HEX>/<POSIX_PATH>","license":"Apache-2.0"},"defaultEnabled":true,"defaultScaleTenths":10}'
```

### Default-on promotion flip checklist

1. Publish a `prism-inference` `v0.1.33+` bundle containing `MultiRankAdapter`
   through Foundation, verify and pin it in
   [prism-inference.lock.json](../config/prism-inference.lock.json) via
   `bonsai-tools inference-lock`, and set `MULTI_RANK_LORA_RELEASE` in
   [inference.rs](../src/broker/src/inference.rs) to that exact
   `(version, commit)` tuple.
2. Commit the `generate-lock` output as `config/daisyrose-lora.lock.json` and
   wire `include_str!("../../../config/daisyrose-lora.lock.json")` into
   `pinned_daisyrose_lora_artifact` in
   [daisyrose.rs](../src/cpu-policy/src/daisyrose.rs).
3. Add `config/daisyrose-lora.lock.json` to `package_inputs` (`bonsai-cpu-policy`
   and `bonsai-tools`) and `groups.release.paths` in
   [impact.json](../.github/ci/impact.json), and add the Apache-2.0 artifact
   notice in [licensing.md](licensing.md) and [NOTICE](../NOTICE).
4. Pass owning CI (`source-checks`, `foundation`), matched-power full-adapter
   X1/X2 throughput/RAM qualification, and independent held-out human review
   before shipping default-on.
