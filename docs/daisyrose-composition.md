# DaisyRose-27B + Swift-Bonsai-2 Hybrid Composition

This contract governs the hybrid composition of **DaisyRose-27B** reasoning,
coding, and instruction-following capability on top of **Swift-Bonsai-2**
(`PQ2_0` / `PTQ1_0`) with **zero retraining**:

1. Extract the pure-Rust Rademacher-null + Gavish–Donoho spectral-denoised
   multi-rank F16 LoRA (`export-denoised`) and stack ("plop") it directly on top
   of immutable `Swift-Bonsai-2` + pinned Orca `1.3`.
2. Screen that unedited stacked adapter empirically via `DR-G2`. If it passes
   all four gates as-is, skip weight editing and advance straight to `DR-C1` /
   `DR-C2`.
3. Only if the direct stack fails `DR-G2`, resolve diagnosed failures via
   pure-Rust **direct weight editing** (`DR-W1`) on the floating-point
   multi-rank LoRA `(A, B)` factors without GPU/gradient retraining or
   modifying the packed ternary base.

Pinned components:

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
- **Full-on retraining / external GPU OPD (`Track B` / `DR-B1`) is retired / out of scope**:
  No external GPU training, rollout distillation, or gradient fine-tuning will
  be run for DaisyRose composition.

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

## Zero-retraining composition strategy

### Stage 1 (`DR-E1`): Zero-retraining spectral-denoised extraction & direct stacking

`DR-A1` isolated genuine low-rank coherent spikes above the BF16 1-ULP
quantization bulk by calibrating empirically against a Rademacher sign-flip null
and denoising via **Gavish–Donoho (2017) Frobenius-optimal singular value
shrinkage**:

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
   sketch rank (`tailSaturated = true`), the spectrum has a power-law tail
   extending beyond the truncated rank rather than an isolated finite-rank spike
   set.
5. **Empirical `DR-A1` signal and policy update**: The November 19, 2026
   `DR-A1` probe ([spectral-denoising audit](evidence/daisyrose-spectral-denoising-audit.md))
   measured **81.98%** BF16 bulk energy and genuine **6.75x** mean /
   **14.92x** peak coherent spikes (`E_denoised(16) = 69.21%`,
   `E_denoised(32) = 85.27%`), with `13/16` audited matrices above the bulk
   edge at rank 64. The owner superseded the earlier `DR-A1` policy stop on
   standalone zero-retraining export in favor of empirically testing the full
   64-layer (`128`-pair `npu-friendly`) denoised F16 LoRA (`r = 16`, `64 MiB`;
   optional `r = 32`, `128 MiB`) stacked directly on top of `Swift-Bonsai-2` +
   pinned Orca `1.3`.
6. **Operational extraction bound**: Streaming all `128` `npu-friendly` BF16
   tensors across both models transfers `29,527,900,160` bytes (`27.50 GiB`,
   `13.75 GiB` per model). Keep offline extraction (`local-directories` over
   pre-fetched safetensors shards, a dedicated `export-denoised` pass decoupled
   from inference timers, or layer-chunked extraction merged into one GGUF in
   pure Rust if live Hugging Face Range I/O exceeds
   `controls.timeBudgetSeconds <= 900`) strictly separate from owner-device
   inference qualification clocks.

### Stage 2 (`DR-G2`): Direct-stack empirical screen ("plop and test")

1. Verify the extracted full 64-layer GGUF via `bonsai-tools lora-diff-audit`
   `operation: "verify-adapter"`.
2. Stack the unedited LoRA directly on `Swift-Bonsai-2-PQ2_0.gguf` + pinned Orca
   `1.3` via `--publisher-engine` `--lora-scaled` across a scale grid starting
   at `0.0` (for example `[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`; `dr-g2-plan`
   and `dr-g2-score` accept any strictly increasing grid in `[0.0, 2.0]`, and
   scales `> 1.0` are directly relevant given `relDelta ~ 0.0005–0.0018` and
   Gavish–Donoho shrinkage).
3. Score fail-closed with `dr-g2-score` (`trainingPromptSha256s: []` for the
   unedited extraction).
4. **Branching decision**:
   - If the unedited direct stack passes all four `DR-G2` gates, **skip `DR-W1`**
     and advance directly to `DR-C1` / `DR-C2`.
   - If it fails, record exact per-category and per-prompt failure modes (e.g.,
     weak/muted signal, overthinking/cognitive looping, creative/Orca
     interference, or layer/numerical instability) and enter `DR-W1`.

### Stage 3 (`DR-W1`): Direct weight editing remediation (zero retraining, pure Rust)

Triggered **only** if the unedited `DR-E1` direct stack fails `DR-G2`. All edits
operate directly in pure Rust on the floating-point `F16`/`F32` multi-rank GGUF
LoRA `(A, B)` factors while keeping `Swift-Bonsai-2`'s packed ternary base
immutable and staying inside the `npu-friendly` target envelope (no Python, no
GPU/gradient retraining). Ordered levers mapped to diagnosed `DR-G2` failures
(following [weight-space programming](weight-space-programming.md)):

1. **Algebraic factor & spectral edits (zero activation capture)**:
   - Adjust rank (`r = 16 -> 32`), Gavish–Donoho shrinkage / `minSpikeRatio` /
     unshrunk-spike retention, per-family multipliers (`ffn_down` vs
     `attn_q`/`attn_k`/`attn_v`/`attn_output`), and layer-depth masks/tapers
     (for example, attenuating noisy early layers while emphasizing high-SNR
     late reasoning layers such as `blk.63.ffn_down` `14.92x` and
     `blk.63.attn_output` `11.61x`).
   - Surgically drop or re-weight singular modes `(u_k, v_k)`, cap per-site
     norms, and prune small coefficients inside `(A, B)` (generalizing
     `bonsai-tools adapter-edit` to multi-rank GGUF).
   - Orthogonalize DaisyRose factors against pinned Orca rank-1 factors at
     overlapping `ffn_down`/`attn_*` sites (projecting DaisyRose `B` or `A`
     columns orthogonal to Orca so stacked adapters do not cancel or conflict at
     Orca `scale = 1.3`).
2. **Closed-form activation-guided weight-space edits (pure-Rust linear algebra
   / convex LP on bounded `Swift-Bonsai-2` activations — zero gradient
   retraining)**:
   - Apply equal-byte per-site KL-gap and Softmax-Fisher variable-rank
     reallocation (following
     [LookAtMyQuant sensitivity allocation](research-register.md#small-edits-around-cheaper-execution)):
     instead of uniform `r = 16` across all 128 `npu-friendly` sites
     (`64 MiB` F16), score each site on disjoint diagnostic splits by isolated /
     leave-one-out KL gap (`KL(all) - KL(site = 0)`) and token-weighted output
     Softmax-Fisher (`E[p_v(1 - p_v)]`), separating Thinking-ON coding/reasoning
     gain from Thinking-OFF creative/Orca drift. Reallocate variable per-site
     rank (`r = 0..32`, bounded to `<= 64 MiB` total F16 payload inside
     `npu-friendly` and `MultiRankAdapter`) toward high-SNR late reasoning
     layers (such as `blk.63.ffn_down` `14.92x` and `blk.63.attn_output`
     `11.61x`) while zeroing or trimming noisy or prose-interfering early sites.
     Record all diagnostic prompt digests in `trainingPromptSha256s` so final
     held-out `DR-G2` splits remain untouched.
   - Apply targeted behavioral remediation via `bonsai-tools weight-patch` (PDHG
     sparse LP `u * (v^T * h)` on original-basis activations, compared against a
     constant residual steering baseline `x + β d` per
     [practical investigation order](weight-space-programming.md#practical-investigation-order)),
     folded or appended directly into the multi-rank DaisyRose GGUF at selected
     `ffn_down` sites to eliminate diagnosed cognitive looping, unwanted
     refusals, or Thinking-OFF prose drift.
   - Apply closed-form ridge / least-squares `B` or `A` manifold realignment
     (reusing the algebraic dual-ridge solver pattern from
     `bonsai-tools ffn-surrogate`) on small bounded `Swift-Bonsai-2`
     hidden-state captures (`min ||B (A^T X_swift) - Y_target||_F^2 + λ ||Δ||_F^2`)
     if the QAT ternary manifold gap misaligns or attenuates BF16-extracted
     factors.

**Split discipline and stop criteria**: Any prompts used to diagnose failures or
fit LP/ridge edits must remain strictly disjoint from final held-out `DR-G2`
evaluation prompts (record diagnostic/fit prompt SHA-256 digests in
`trainingPromptSha256s`, and freeze a fresh untouched held-out split before final
scoring if editing after inspecting prior `DR-G2` outputs, per the
[direct-edit evaluation contract](weight-space-programming.md#direct-edit-evaluation-contract)).
Freeze every edited GGUF, verify with `verify-adapter`, and re-score fail-closed
with `dr-g2-score`. If bounded direct weight editing still fails `DR-G2` on
untouched holdouts, record the losing/no-go outcome honestly under
[evidence retention](validation.md#evidence-retention) and keep
`config/daisyrose-lora.lock.json` unpinned (`Ok(None)`).

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
candidate (`scale` grid starting at `0.0`, e.g. `[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`):

1. **DaisyRose capability gain**: Measurable Thinking-ON gain on held-out
   coding, reasoning, and instruction-following prompts over `scale = 0.0`
   without per-category regression.
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
   Promoting DaisyRose to default-on still requires full 64-layer
   zero-retraining extraction (`DR-E1`), `DR-G2` + held-out human review (plus
   conditional `DR-W1` direct weight editing only if `DR-G2` fails on the
   unedited adapter), Prism `v0.1.33+` publication and repin in
   `config/prism-inference.lock.json` (`MULTI_RANK_LORA_RELEASE`), pinning
   `config/daisyrose-lora.lock.json`, matched-power full-adapter X1/X2 device
   qualification under [validation](validation.md) and
   [Swift LoRA qualification](swift-lora.md#standard-profile-qualification), and
   owning CI (`source-checks`, `foundation`).

## Zero-retraining workflow, direct-edit fallback, and promotion checklist

### Pinned base, source RL model, and stacked adapter identities

- **Student base (`Swift-Bonsai-2`)**: `ukisai/Swift-Bonsai-2-GGUF`, revision
  `a3bdac086bbb6b04d87d0108d18943b5f63868f7`, `Swift-Bonsai-2-PQ2_0.gguf`
  (`7,206,168,928` bytes, SHA-256
  `5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5`,
  `248,320`-token vocabulary, `qwen35` chat template in
  [swift.metadata.json](../config/swift.metadata.json)).
- **Source RL model (`DaisyRose-27B`)**: `RadixArk/DaisyRose-27B`, revision
  `64a4ec665091bb0c1931babce4e97e686eef3c5c` (`Apache-2.0`).
- **Source base model (`Qwen3.8-27B`)**: `Qwen/Qwen3.8-27B`, revision
  `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
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

### Zero-retraining extraction, direct-edit remediation, and split-disjointness discipline

1. **Extract full 64-layer spectral-denoised GGUF (`DR-E1`)**: Run pure-Rust
   `bonsai-tools lora-diff-audit` `operation: "export-denoised"` across all 64
   layers (`128` `npu-friendly` pairs, `F16`, `r = 16` first, optional `r = 32`)
   against pinned `RadixArk/DaisyRose-27B` vs `Qwen/Qwen3.8-27B` (using local
   prefetched shards or a dedicated extraction pass decoupled from inference
   timers). Verify GGUF v3 layout, finiteness (`|v| < 65520.0`), dual alpha
   equality, and NPU/SSM bounds with `verify-adapter`.
2. **Screen direct stack (`DR-G2`)**: Stack the unedited `DR-E1` GGUF directly
   on `Swift-Bonsai-2-PQ2_0.gguf` + Orca `1.3` via `--publisher-engine`
   `--lora-scaled` across `scaleGrid` (`[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`)
   and score fail-closed with `dr-g2-plan` / `dr-g2-score`
   (`trainingPromptSha256s: []`).
3. **If `DR-G2` passes, stop editing and promote (`DR-C1` / `DR-C2`)**: Do not
   run `DR-W1` if the unedited `DR-E1` direct stack already satisfies all four
   `DR-G2` gates.
4. **If `DR-G2` fails, apply pure-Rust direct weight editing (`DR-W1`) and
   re-screen**: Diagnose exact failing categories/prompts, apply bounded
   algebraic or closed-form activation-guided edits to the floating-point `(A, B)`
   factors within `npu-friendly` (zero retraining), record any diagnostic/fit
   prompt SHA-256 digests in `trainingPromptSha256s` so final held-out splits
   stay strictly disjoint, re-verify with `verify-adapter`, and re-score on an
   untouched held-out `DR-G2` split.

### Pure-Rust candidate intake, `DR-G2` screen, and lock generation

`DR-W1` direct weight-edited candidates follow the same `verify-adapter` ->
`dr-g2-plan` -> `dr-g2-score` -> `generate-lock` pipeline after editing:

```powershell
# 1. Extract full 64-layer npu-friendly denoised F16 LoRA (use local-directories or dedicated pass decoupled from inference clocks)
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"export-denoised","source":{"kind":"pinned-daisyrose"},"scope":{"layers":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63],"targetEnvelope":"npu-friendly","exportDtype":"f16","maxRank":16},"outputPath":"C:/External/daisyrose-r16.gguf"}'

# 2. Verify GGUF v3 layout, finiteness, dual alpha equality, and NPU/SSM bounds
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"verify-adapter","adapterPath":"C:/External/daisyrose-r16.gguf","verifyPolicy":{"requireNpuFriendly":true,"requireNoGroupedSsm":true,"maxAllowedRank":32}}'

# 3. Emit canonical DR-G2 contract + evidenceTemplate (then capture --publisher-engine three-way + human review)
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"dr-g2-plan","screenConfig":{"candidateAdapterSha256":"<CANDIDATE_SHA256>","candidateMaxRank":16,"candidateTargetCount":128,"scaleGrid":[0.0,0.25,0.5,0.75,1.0,1.5,2.0]}}'

# 4. Score completed held-out evidence + human reviews fail-closed
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"dr-g2-score","evidencePath":"C:/External/dr-g2-evidence.json"}'

# 5. After DR-G2 passes, write schema-validated candidate lock JSON (refuses synthetic fixtures when defaultEnabled=true; supports host "huggingface", "github", or "github-release")
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"generate-lock","adapterPath":"C:/External/daisyrose-r16.gguf","outputPath":"C:/External/daisyrose-lora.lock.json","lockSource":{"host":"github-release","repository":"<OWNER>/<REPO>","release":"<TAG>","revision":"<40_HEX>","path":"<FILE>.gguf","filename":"<FILE>.gguf","url":"https://github.com/<OWNER>/<REPO>/releases/download/<TAG>/<FILE>.gguf","license":"Apache-2.0"},"defaultEnabled":true,"defaultScaleTenths":10}'
#    For Hugging Face or raw GitHub tree files (no release tag), set host to "huggingface" (https://huggingface.co/<OWNER>/<REPO>/resolve/<40_HEX>/<POSIX_PATH>) or "github" (https://raw.githubusercontent.com/<OWNER>/<REPO>/<40_HEX>/<POSIX_PATH>).
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
