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
  above `residual-writers`. Meanwhile [Who Asks, Who Answers](research-register.md#hybrid-gated-deltanet-full-attention-and-multimodal-efficiency)
  (`arXiv:2611.01909`) shows hybrid GDN blocks construct retrieval queries for downstream softmax
  attention, elevating the value of resolving `P1` to unlock selective `ssm_out` LoRA sites;
  for multimodal `mmproj` runs, apply [Selective-RL](research-register.md#qwen38-27b-adapters-surgical-abliteration-weight-editing-and-low-bit-recovery)
  (`arXiv:2610.08659`) visual-preservation checks across `DR-G2`/`DR-W1`.

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
   For `export-denoised` gain sensitivity testing, `scope.shrinkageMode` selects
   `"gavish-donoho"` (default), `"unshrunk"`, or `"blended"` (interpolating
   `(1 - alpha) * sigma + alpha * shrunk` via `scope.shrinkageBlend` in
   `0.0..=1.0`, recorded in GGUF metadata `bonsai.composition.shrinkage_mode`).
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
   from inference timers, or layer-chunked extraction joined via pure-Rust
   `merge-adapters` if live Hugging Face Range I/O exceeds
   `controls.timeBudgetSeconds <= 900`) strictly separate from owner-device
   inference qualification clocks.

### Stage 2 (`DR-G2`): Direct-stack empirical screen ("plop and test")

1. Verify the extracted full 64-layer GGUF via `bonsai-tools lora-diff-audit`
   `operation: "verify-adapter"`.
2. Stack the unedited LoRA directly on `Swift-Bonsai-2-PQ2_0.gguf` + pinned Orca
   `1.3` (or opt-in `NariQ` without Orca) across a scale grid starting at
   `0.0` (for example `[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`; `dr-g2-plan`,
   `dr-g2-capture`, and `dr-g2-score` accept any strictly increasing grid in
   `[0.0, 2.0]`, and scales `> 1.0` are directly relevant given
   `relDelta ~ 0.0005–0.0018` and Gavish–Donoho shrinkage):
   - **Native sleep-inhibited capture (`dr-g2-capture`):** Spawns two owned
     `prism-inference-server` child Jobs (`Server A` baseline vs `Server B`
     with `--daisyrose-lora`, `--daisyrose-lora-bytes`, and
     `--daisyrose-lora-sha256`) on numeric loopback while acquiring
     `bonsai_windows_process::SystemAwakeGuard` (a `!Send + !Sync` thread-local
     RAII wrapper around `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`
     restored to `ES_CONTINUOUS` on drop) so idle Windows suspend cannot abort
     multi-category captures (`sleepInhibitionActive: true`), recording overall
     and per-arm `GetSystemPowerStatus` snapshots (`armPowerObservations`),
     `thinkingTokens`, `hit4096Truncation`, `detect_repetition_loop`
     (`loopDetected`), `DrG2Evidence` (`outputPath`), and optional
     `responsesOutputPath`.
   - **Blinded owner review (`dr-g2-blind-packet` / `dr-g2-apply-reviews`):**
     `dr-g2-blind-packet` (`blindConfig`: `responsesPath`, `blindKeyPath`,
     optional `blindSalt`) hashes `dr-g2-blind-v1` + `contractSha256` +
     `promptId` + `blindSalt` to randomize `armA`/`armB` placement alongside an
     integrity-hashed unblinding key (`integritySha256`); `dr-g2-apply-reviews`
     (`applyConfig`: `reviewPath`, `blindKeyPath`) verifies text and key
     digests, populates `humanReviewed: true` back into `DrG2Evidence`, and runs
     `dr-g2-score` fail-closed.
   - **Base-profile and `Selective-RL` multimodal options:** `dr-g2-plan`
     (`screenConfig`) and `dr-g2-score` support `baseModel`
     (`BaseModelProfile`: `"swift-pq2"` default, `"nariq-2.5bpw"`,
     `"nariq-3.0bpw"`, `"nariq-2.0bpw"` — where NariQ profiles validate against
     `NariqVariant::sha256()` and require `orcaAdapterSha256: null` and
     `orcaScale: 0.0`, plus `"bonsai-2.5-pq2"` and `"bonsai-2.5-ptq1"` — which
     default to `orcaAdapterSha256: null, orcaScale: 0.0` via
     `includeOrca: false` or stack Orca via `includeOrca: true`, also
     supported by `dr-g2-capture` and `generate-lock`), plus
     `includeMultimodalChecks: true` / `multimodalPrompts`
     (`MultimodalPromptPair` carrying `imageSha256` and
     `visualGroundingPreserved` under a `>= 3.0` median/floor
     `multimodalPreservation` gate).
3. Score fail-closed with `dr-g2-score` (`trainingPromptSha256s: []` for the
   unedited extraction).
4. **Branching decision**:
   - If the unedited direct stack passes all four `DR-G2` gates (plus
     `multimodalPreservation` when enabled), **skip `DR-W1`** and advance
     directly to `DR-C1` / `DR-C2`.
   - If it fails, record exact per-category and per-prompt failure modes (e.g.,
     weak/muted signal, overthinking/cognitive looping, creative/Orca
     interference, or layer/numerical instability) and enter `DR-W1`.

### Ternary Bonsai 2.5 27B + DaisyRose-27B composition

**Ternary Bonsai 2.5 27B (`prism-ml/Ternary-Bonsai-2.5-27B-GGUF`) does not obviate
DaisyRose-27B; the two compose directly.** Bonsai 2.5 recovers the ternary
quantization loss relative to FP16 `Qwen/Qwen3.8-27B` and fixes Swift-Bonsai-2's
overthinking/looping natively in the base weights, while `DaisyRose-27B` adds pure-RL
post-training gains above FP16 `Qwen3.8-27B` on agentic coding, terminal tool use,
and strict instruction following (`IFBench`, `Terminal-Bench`, `SWE-bench Pro`).
Because Bonsai 2.5 preserves exact `qwen35` geometry, all 128 `npu-friendly` matrix
shapes, identical Hadamard signs, identical tokenizer/template (`c3cf9e34...`),
and bit-for-bit identical `mmproj-Q8_0.gguf`, `MultiRankAdapter` (`DaisyRose`)
GGUFs stack directly onto `--bonsai-2-5 pq2|ptq1` — both **Orca-free** (default
for `bonsai-2.5-*` `DR-G2` plans, omitting `--abliterate-lora`) and
**Orca-stacked** (`--use-abliterate-lora`, `scale 1.3`) — while preserving Hexagon
NPU FFN chains, Adreno Vulkan, Composite placement, and Vision.

### Stage 3 (`DR-W1`): Direct weight editing remediation (zero retraining, pure Rust)

Triggered **only** if the unedited `DR-E1` direct stack fails `DR-G2`. All edits
operate directly in pure Rust on the floating-point `F16`/`F32` multi-rank GGUF
LoRA `(A, B)` factors while keeping `Swift-Bonsai-2`'s packed ternary base
immutable and staying inside the `npu-friendly` target envelope (no Python, no
GPU/gradient retraining). Ordered levers mapped to diagnosed `DR-G2` failures
(following [weight-space programming](weight-space-programming.md)):

1. **Algebraic factor & spectral edits (zero activation capture)**:
   - Adjust rank (`r = 16 -> 32`) and shrinkage (`scope.shrinkageMode`:
     `"gavish-donoho"`, `"unshrunk"`, `"blended"` with `scope.shrinkageBlend`,
     plus `scope.minSpikeRatio`) via `export-denoised`.
   - Apply pure-Rust algebraic factor edits via `bonsai-tools lora-diff-audit`
     `operation: "edit-adapter"` (`editConfig` -> `EditAdapterConfig`, which
     renormalizes dual alpha keys `adapter.lora.alpha` and `adapter.lora_alpha`
     to `max_new_rank`, writes `bonsai.composition.edit_parent_sha256` and
     `bonsai.composition.edit_method = "algebraic-multi-rank-edit-v1"`, and
     self-verifies the GGUF via `verify_published_adapter`):
     - **Rank & mode controls:** `targetMaxRank` (`1..=64`), `perSiteRanks`
       (`0..=64`, `0` drops the site), `dropModes` (`0..64`), and `modeWeights`
       (`[0.0, 100.0]`).
     - **Gains, layer masks & `Attention Residuals` depth steering:**
       `globalGain` (`(0.0, 100.0]`), `familyScales` (`ffn_down` vs
       `attn_q`/`attn_k`/`attn_v`/`attn_output`), `layerMin`/`layerMax` (`0..64`),
       sparse per-layer `layerGains` (`0..64` -> `[0.0, 100.0]`), and
       `depthGainStart`/`depthGainEnd` with `depthGainCurve` (`"linear"`,
       `"cosine"`, or `"power"` with `depthGainExponent` in `0.25..=8.0`) to
       attenuate noisy early layers while emphasizing high-SNR late reasoning
       layers such as `blk.63.ffn_down` (`14.92x`) and `blk.63.attn_output`
       (`11.61x`).
     - **Magnitude pruning, norm bounds & `LS-LoRA` site selection:**
       `pruneKeepNumerator`/`pruneKeepDenominator` (`1 <= num <= denom <= 1024`)
       with `pruneTarget` (`"a"`, `"b"`, `"both"`), `minCoeffAbs`,
       `minSiteFrobeniusNorm`/`maxSiteFrobeniusNorm`, and `lsLoraReferenceAdapterPath`
       + `lsLoraMinCosine` (`-1.0..=1.0`) for `LS-LoRA` cosine-similarity filtering.
     - **`CANS` neuron scaling & Orca / `AlphaEdit` subspace orthogonalization:**
       `cansNeuronScales` (per-site input-neuron scales matching `in_features` on
       `A` rows), `orthogonalizeAgainstRankOnePath` (e.g., pinned Orca rank-1 GGUF)
       with `orthogonalizeSide` (`"b"`, `"a"`, `"both"`),
       `orthogonalizeAgainstAdapterPath` (multi-rank reference GGUF) with
       `orthogonalizeAdapterSide` (`"b"`, `"a"`, `"both"`), and
       `alphaEditNullSpaceRidge` (`>= 0.0`) for `AlphaEdit` Tikhonov null-space
       projection; `exportDtype` selects `"f16"` or `"f32"`.
2. **Closed-form activation-guided weight-space edits (pure-Rust linear algebra
   / convex LP on bounded `Swift-Bonsai-2` activations — zero gradient
   retraining)**:
   - Apply equal-byte per-site KL-gap and Softmax-Fisher variable-rank
     reallocation via `operation: "reallocate-ranks"` (`reallocateConfig` ->
     `ReallocateRanksConfig`, following
     [LookAtMyQuant sensitivity allocation](research-register.md#small-edits-around-cheaper-execution)):
     instead of uniform `r = 16` across all 128 `npu-friendly` sites
     (`64 MiB` F16), score each site on disjoint diagnostic splits by isolated /
     leave-one-out KL gap (`KL(all) - KL(site = 0)`) and token-weighted output
     Softmax-Fisher (`E[p_v(1 - p_v)]`), separating Thinking-ON coding/reasoning
     gain from Thinking-OFF creative/Orca drift. Exact multiple-choice knapsack
     selection (`maxPayloadBytes`, `minRank`/`maxRank`, `candidateRanks`,
     `thinkingOffPenaltyWeight`, `siteSensitivities`) reallocates variable per-site
     rank (`r = 0..32`, bounded to `<= 64 MiB` total F16 payload inside
     `npu-friendly` and `MultiRankAdapter`) toward high-SNR late reasoning
     layers while zeroing or trimming noisy or prose-interfering early sites.
     Record all diagnostic prompt digests in `diagnosticPromptSha256s` and
     `trainingPromptSha256s` so final held-out `DR-G2` splits remain untouched.
   - Apply targeted behavioral remediation via `bonsai-tools weight-patch` (PDHG
     sparse LP `u * (v^T * h)` on original-basis activations, compared against a
     constant residual steering baseline `x + β d` per
     [practical investigation order](weight-space-programming.md#practical-investigation-order))
     and fold/append those patches directly into the multi-rank DaisyRose GGUF via
     `operation: "fold-patch"` (`foldPatchConfig` -> `FoldPatchConfig`): accepts
     optional `adapterPath`, `diagnosticPromptSha256s`, `exportDtype`,
     `targetMaxRank` (`1..=64`), `mode` (`"append"` -> `"activation-fold-append-v1"`
     or exact thin-QR + Jacobi-SVD compression `"fold-svd"` ->
     `"activation-fold-svd-v1"`), and `1..=64` `patches` entries (`"weight-patch-file"`
     referencing schema-1 `"experimental-ffn-rank-one-patch"` JSON with `patchPath` /
     optional `patchScale`, or `"inline-rank-one"` with `target`, `a`, `b`, `scale`),
     recording `bonsai.composition.fold_parent_sha256` and
     `bonsai.composition.fold_method`.
   - Apply closed-form Tikhonov ridge manifold realignment via
     `operation: "realign-factors"` (`realignConfig` -> `RealignFactorsConfig`) on
     bounded `Swift-Bonsai-2` hidden-state captures
     (`min ||B (A^T X_swift) - Y_target||_F^2 + λ ||Δ||_F^2`) if the QAT ternary
     manifold gap misaligns or attenuates BF16-extracted factors: solves exact
     Cholesky/Jacobi updates across `1..=16` `sites` (`target` + inline `rows` or
     `rowsPath` up to `48 MiB`, `3..=256` rows with `>= 2` `"fit"` and `>= 1`
     `"development"` rows and disjoint `sampleId`/`groupId`/inputs) using
     `mode` in `"realign-b"` (`"activation-ridge-realign-b-v1"`), `"realign-a"`
     (`"activation-ridge-realign-a-v1"`), or `"alternating-ab"`
     (`"activation-ridge-alternating-ab-v1"` with `alternatingSteps` `1..=6`),
     `ridge` (`1e-8..=1e6`), `timeBudgetMilliseconds` (`1..=30000`),
     `diagnosticPromptSha256s`, `exportDtype`, and fail-closed
     `allowDevelopmentRegression` (default `false` — rejects publication as
     `"development-regression-rejected"` if target-dtype quantization regresses
     development MSE), recording `bonsai.composition.realign_parent_sha256` and
     `bonsai.composition.realign_method`.

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

### Alternative donor classes and low-bit residual adapters

Beyond `DaisyRose-27B` BF16 extraction, three related donor/recovery routes tracked in the
[research register](research-register.md#qwen38-27b-adapters-surgical-abliteration-weight-editing-and-low-bit-recovery)
match `MultiRankAdapter` without retraining Swift's ternary base:

- **Native Thinking-OFF RL LoRAs (`harness-hub`):** Prebuilt rank-32 LoRAs
  (`harness-hub/Qwen3.8-27B-pi-adapted`, `opencode-adapted`, `hermes-adapted`, merged rank-96
  `Qwen3.8-27B-super-adapter`) carry native clean LoRA deltas without BF16 1-ULP bulk noise,
  providing an alternative `npu-friendly` donor source for low-token Thinking-OFF improvements
  subject to revision/license audit and `DR-G2`.
  - **Pure-Rust PEFT importer (`import-peft-lora`):** `peftConfig` (`ImportPeftConfig`)
    imports native PEFT `safetensors` checkpoints directly from
    `peftConfig.source` (`"local-directory"` with `directory`, `license`,
    optional `repository`/`revision`; or `"pinned-huggingface"` with
    `repository`, `revision`, `license`, optional `configFilename` /
    `weightsFilename`). It parses `adapter_config.json` (`peft_type = "LORA"`,
    `r` in `1..=256`, `lora_alpha`, `use_rslora`, rejecting DoRA/`fan_in_fan_out`/bias),
    drops vision/`mmproj`, `ssm_out`, and out-of-envelope tensors
    (`targetEnvelope` in `"residual-writers"`, `"npu-friendly"` default, or
    `"full-linear"`, recording `droppedCounts`), optionally projects out
    rank-1 (`orthogonalizeAgainstRankOnePath`) or multi-rank
    (`orthogonalizeAgainstAdapterPath`) subspaces, compresses sites with
    `r_src > maxRank` (`1..=64`) via exact thin QR + one-sided Hestenes–Jacobi
    SVD (`jacobi_svd_f64`), enforces `maxPayloadBytes`, records
    `bonsai.composition.method = "peft-safetensors-import-v1"` plus donor
    metadata, and self-verifies the published GGUF v3 output.
- **Creative-prose and frontier reasoning distillations:** Community distill donors such as
  `barozp/Qwen3.8-27B-Opus-Distill` and `r0b0tlab/qwen3.8-max-distillation-50k` target
  creative writing and reasoning more directly than pure code-agent RL runs, subject to license
  review, extraction/denoising and `DR-G2` held-out scoring.
  - **Generic Hugging Face diff auditing (`"pinned-huggingface"`):** `plan`,
    `audit`, and `export-denoised` accept `source.kind = "pinned-huggingface"`
    (`candidateRepository`, `candidateRevision`, `candidateLicense`,
    `baseRepository`, `baseRevision`), embedding
    `bonsai.composition.candidate_repo`, `candidate_revision`,
    `candidate_license`, `base_repo`, and `base_revision` in GGUF provenance.
- **Quantization-residual LoRAs (`SVDQuant`, `Recover-LoRA`, `OnQ` / `OnlineQAT`):** Low-rank
  FP16 residual branches trained on student rollouts can target ternary approximation loss rather
  than RL task deltas. **Critical safeguard:** never extract `W_Qwen3.8 - dequant(W_Swift)`
  naively onto `Swift-Bonsai-2`, because that difference partially cancels Swift's merged
  anti-overthinking edit; fit residual adapters against Swift's own rollouts (`OnQ`/`Recover-LoRA`)
  or project updates orthogonal to Swift's anti-overthinking subspace.

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

No promoted default lock ships until **`DR-G2`** passes on `--publisher-engine`
(or verified native `prism-inference-server` via `dr-g2-capture`) stacking Orca
(`1.3`, omitted on `NariQ`) + candidate (`scale` grid starting at `0.0`, e.g.
`[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`):

1. **DaisyRose capability gain**: Measurable Thinking-ON gain on held-out
   coding, reasoning, and instruction-following prompts over `scale = 0.0`
   without per-category regression.
2. **Swift anti-looping preservation**: Median thinking token count `<= 1.10x`
   `scale = 0.0` Swift baseline, with zero regression in `4,096`-token
   truncation or pathological repetition/loop rates.
3. **Creative & Orca non-regression**: Thinking-OFF creative prose quality,
   conciseness, and Orca refusal behavior match or exceed `scale = 0.0` (plus
   `multimodalPreservation` `>= 3.0` median/floor visual grounding when
   `includeMultimodalChecks` / `multimodalPrompts` are enabled).
4. **Native runtime (`DR-C1`) & Runner integration (`DR-C2`)**: Stackable opt-in
   `MultiRankAdapter` runtime wiring (`DR-C1` in `prism-inference`) and Runner
   artifact/policy/UI integration (`DR-C2`) are implemented locally:
   - **Pin status:** In [inference.rs](../src/broker/src/inference.rs),
     `REVIEWED_0_1_35`
     (`("0.1.35", "3bc76f135bf23c538b3bb789aef382143ff36839")`) now enables
     `MULTI_RANK_LORA_RELEASE`, `RSQ4KVF_RELEASE`, `NARIQ_TRELLIS_RELEASE`, and
     `BONSAI_2_5_RELEASE` (`K4V3_CONTEXT_RELEASE` remains `None`), while
     `pinned_daisyrose_lora_artifact` in [daisyrose.rs](../src/cpu-policy/src/daisyrose.rs)
     continues to return `Ok(None)` pending a qualified `config/daisyrose-lora.lock.json`
     and owner `DR-G2` review.
   - **Trials #2 & #3 + delegated blinded review ([bonsai-2-5-x2-vet.md](evidence/bonsai-2-5-x2-vet.md)):**
     On `bonsai-2.5-pq2` Orca-free + `daisyrose-npu-r16-drw1.gguf` (`scale 1.0`,
     `reasoningEffort: "medium"`, `reasoningBudget: 448`), Trial #2 (`886.959 s`,
     `900 s` AC allowance) sealed zero-scale parity (`0` mismatches) and `3/3`
     Thinking-ON pairs (`4b221d19...`) while skipping `creative-b25-001` on the
     budget guard; Trial #3 (`138.826 s`, `300 s` AC allowance) captured
     `creative-b25-001` (`thinkingOff`) via pure-Rust `supplementEvidencePath`
     and sealed the complete `4/4` blind packet (`6dc82c06...`). Owner-delegated
     blinded scoring (`reviewerId: "agent-delegated-by-owner"`, pre-unblinding
     scores applied via `dr-g2-apply-reviews` and scored via `dr-g2-score`,
     receipt `9d8e9a71...`) returned **`drG2Passed: false` (No-Go for default flip today)**:
     `3/4` gates passed (`capabilityGain` `+0.8333` with `coding` `+1.5`, `reasoning`
     `+1.0`, `instruction-following` `0.0`; `antiLoopingPreservation` `474 / 472 = 1.0042x`,
     `0` loops/truncations; `zeroScaleIdentity` `0/1` mismatches), while Gate 3
     (`creativeAndOrcaNonRegression`) failed strictly (`3.2 < 3.5` on `n = 1`
     `creative-b25-001`, though clearing the `>= 3.0` floor with `0` refusals).
     Default remains `Swift-Bonsai-2-PQ2_0` + Orca.
   - **Newly landed `DR-C1` work in `..\prism-inference`:**
     - `examples/swift_model_check.rs`, `examples/swift_model_check/request_timing.rs`,
       and `examples/swift_model_check/ffn_timing.rs` support `--nariq`
       (`<2.5bpw|3.0bpw|2.0bpw>`) and `--multi-rank-adapter`
       (`<absolute-gguf>[:<scale-0..2>]`) across `trajectory`, `request-timing`,
       and `ffn-timing` modes (rejecting `--nariq` + `--rank-one-adapter` and
       recording `multiRankAdapterSha256`, `multiRankAdapterMaxRank`, and
       `multiRankAdapterScaleBits`).
     - `tools/adreno-vulkan/src/graph.rs` (`prepare_adapters_upload` /
       `PreparedAdapters` / `PreparedLinearLoraFactor`, verified by
       `sparse_heterogeneous_multi_rank_adapter_prepares_per_site_ranks_and_scratch_layout`)
       preserves per-site heterogeneous `MultiRankAdapter` ranks and per-site
       scaling while sizing scratch by `max_rank`.
     - `src/graph.rs`, `src/graph/prefill_ffn_tests.rs`, and
       `src/graph/prefill_folded_tests.rs` add parity tests verifying sparse
       heterogeneous `MultiRankAdapter` stacked with `RankOneAdapter` (`Orca`),
       Hexagon NPU FFN fusion preservation
       (`sparse_heterogeneous_multi_rank_keeps_omitted_chain_lora_and_matches_zero_ulp_decode_and_tiled`),
       `rsq4kvf` and NariQ trellis graphs
       (`qwen35_four_layer_trellis_heterogeneous_multi_rank_and_rsq4kvf_parity`),
       and folded speculative verification
       (`folded_speculative_verification_with_stacked_orca_sparse_heterogeneous_multi_rank_and_rsq4kvf`).
   - **Newly landed `DR-C2` work in Runner:**
     - [daisyrose.rs](../src/cpu-policy/src/daisyrose.rs):
       `daisyrose_lora_artifact_for_base_sha256` and
       `DaisyRoseLoraArtifact::for_nariq_lock` (offline lock resolution/validation
       against `NariqVariant::sha256()`).
     - [context.rs](../src/cpu-policy/src/context.rs):
       `MultiRankAdapterBudget::for_npu_friendly_rank` (charging `4_194_304`
       bytes/rank `F16` or `8_388_608` bytes/rank `F32` plus
       `DAISYROSE_SCRATCH_HEADROOM_BYTES`), alongside `from_verified_bytes` and
       `for_calibrated_execution_with_adapter_reserve`.
     - [launcher.rs](../src/broker/src/launcher.rs), [main.rs](../src/broker/src/main.rs),
       and `launcher/identity.rs`: `Placement::Composite` AppContainer fix in
       `sandbox_readonly_paths` granting read-only access to
       `hexagon_runtime_readonly_paths`, NariQ auto-attach guard in
       `resolve_daisyrose_selection_with` (ensuring compiled PQ2 locks never
       auto-attach onto NariQ base weights), `swift_context_plan_with_daisyrose`
       calibrated CoW checks, `LaunchedService::daisyrose_max_rank()` and
       `daisyrose_sha256()` surfaced on readiness, and `launcher/identity.rs`
       tests for DaisyRose stacking with suffix speculation and `Vulkan` /
       `Composite` placements.
     - [context_capacity.rs](../src/chat-ui/src/window/context_capacity.rs),
       [response.rs](../src/chat-ui/src/window/settings/response.rs),
       `client.rs`, and `window.rs`: readiness parsing and Settings summary
       (`Loaded: rank ≤<rank> · <prefix>` via `loaded_daisyrose_summary`), plus
       `npu_friendly_daisyrose_adapter_bytes`,
       `PlannerRuntimeShape.daisyrose_adapter_bytes`, and
       `format_footprint_diagnostic` reporting `+ <N> MiB DaisyRose` in context
       capacity diagnostics.
     - Pure-Rust `bonsai-tools lora-diff-audit` operations (`plan`, `audit`,
       `export-denoised`, `verify-adapter`, `merge-adapters`, `edit-adapter`,
       `reallocate-ranks`, `fold-patch`, `realign-factors`,
       `generate-test-adapter`, `generate-lock`, `dr-g2-plan`, `dr-g2-score`,
       `dr-g2-blind-packet`, `dr-g2-apply-reviews`, `dr-g2-capture`,
       `import-peft-lora`) and `bonsai_windows_process::SystemAwakeGuard`.
   - **Local vetting & open gates:** Runtime wiring is vetted locally on a
     4-layer `npu-friendly` F16 export at `scale = 0.0` (battery), an offline
     AC-powered 4-layer `r = 16` synthetic fixture (`status: "passed"` in
     `33.875 s`; see
     [multi-rank wiring and vet](evidence/daisyrose-multirank-wiring-vet.md)),
     and full 64-layer (`128`-pair) parity, rollback, and matched-AC overhead
     vets (`3.16%` mean / `3.56%` median overhead for `daisyrose-npu-r16.gguf @ 1.0`
     on `91298a4c`; `-1.9%` mean / `5.01%` median overhead for
     `daisyrose-npu-r32.gguf @ 2.0` on `51a5a84`; and `6.66%` mean / `8.27%`
     median overhead, `+64.28 MiB` WS, `8/8` bit-exact `0.0`, `6/6` rollback for
     pure-Rust `DR-W1` `daisyrose-npu-r16-drw1.gguf @ 1.0` on `51a5a84`; see
     [DR-E1/DR-G2 screen summary](evidence/daisyrose-dr-e1-dr-g2-screen.md)).
     An awake AC untouched held-out `4/4`-category `DR-G2` capture (`*-008`,
     `664.15 s`, `zeroScaleMismatches = 0`, `1.0149x` median thinking token
     ratio, `0` loops/truncations) completed on `daisyrose-npu-r16-drw1.gguf @ 1.0`
     and is sealed as
     `packetSha256 = 399d08ed0fa685b8bccc3f1cddfdaeb59134f142d24c94ecf796f45535e9558a`
     awaiting owner blinded review (`humanReviewed = true`) and fail-closed
     `dr-g2-score` to decide default-on promotion vs categorical no-go, while the
     `bonsai-2.5-pq2` `6dc82c06...` packet completed delegated blinded scoring
     (`9d8e9a71...`, **`drG2Passed: false`**: `3/4` gates passed with `+0.8333`
     Thinking-ON gain, failed Gate 3 `3.2 < 3.5` on `n = 1` Thinking-OFF). Prism
     `v0.1.35` (`3bc76f135bf23c538b3bb789aef382143ff36839`) is now pinned with
     `MULTI_RANK_LORA_RELEASE`, `RSQ4KVF_RELEASE`, `NARIQ_TRELLIS_RELEASE`, and
     `BONSAI_2_5_RELEASE` enabled; before any default flip, a passing disjoint
     Thinking-OFF screen or `DR-W1` pass + independent human review, immutable remote
     hosting + `config/daisyrose-lora.lock.json` (`pinned_daisyrose_lora_artifact`),
     matched-power full-adapter X1/X2 device qualification under
     [validation](validation.md) and
     [Swift LoRA qualification](swift-lora.md#standard-profile-qualification),
     and owning CI (`source-checks`, `foundation`) remain open.

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
   layers (`128` `npu-friendly` pairs, `F16`, `r = 16` first, optional `r = 32`,
   optionally chunked and combined via `merge-adapters`) against pinned
   `RadixArk/DaisyRose-27B` vs `Qwen/Qwen3.8-27B` (using local prefetched
   shards or a dedicated extraction pass decoupled from inference timers).
   Verify GGUF v3 layout, finiteness (`|v| < 65520.0`), dual alpha equality, and
   NPU/SSM bounds with `verify-adapter`.
2. **Screen direct stack (`DR-G2`)**: Stack the unedited `DR-E1` GGUF directly
   on `Swift-Bonsai-2-PQ2_0.gguf` + Orca `1.3` (or `NariQ` without Orca) via
   `dr-g2-capture` (sleep-inhibited via `SystemAwakeGuard`) or
   `--publisher-engine` `--lora-scaled` across `scaleGrid`
   (`[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`), blind via `dr-g2-blind-packet`,
   and unblind/score fail-closed with `dr-g2-apply-reviews` / `dr-g2-score`
   (`trainingPromptSha256s: []`).
3. **If `DR-G2` passes, stop editing and promote (`DR-C1` / `DR-C2`)**: Do not
   run `DR-W1` if the unedited `DR-E1` direct stack already satisfies all four
   `DR-G2` gates.
4. **If `DR-G2` fails, apply pure-Rust direct weight editing (`DR-W1`) and
   re-screen**: Diagnose exact failing categories/prompts, apply bounded
   algebraic (`edit-adapter` / `export-denoised` `shrinkageMode`) or
   closed-form activation-guided (`reallocate-ranks`, `fold-patch`,
   `realign-factors`) edits to the floating-point `(A, B)` factors within
   `npu-friendly` (zero retraining), record any diagnostic/fit prompt SHA-256
   digests in `diagnosticPromptSha256s` and `trainingPromptSha256s` so final
   held-out splits stay strictly disjoint, re-verify with `verify-adapter`, and
   re-score on an untouched held-out `DR-G2` split.

### Pure-Rust candidate intake, `DR-G2` screen, and lock generation

`DR-W1` direct weight-edited and `DR-Alt` imported candidates follow the same
`verify-adapter` -> `dr-g2-plan` -> `dr-g2-capture` -> `dr-g2-blind-packet` ->
`dr-g2-apply-reviews` -> `dr-g2-score` -> `generate-lock` pipeline after
editing:

```powershell
# 1. Extract full 64-layer npu-friendly denoised F16 LoRA (use local-directories or dedicated pass decoupled from inference clocks)
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"export-denoised","source":{"kind":"pinned-daisyrose"},"scope":{"layers":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63],"targetEnvelope":"npu-friendly","exportDtype":"f16","shrinkageMode":"gavish-donoho","maxRank":16},"outputPath":"C:/External/daisyrose-r16.gguf"}'

# 2. Verify GGUF v3 layout, finiteness, dual alpha equality, and NPU/SSM bounds
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"verify-adapter","adapterPath":"C:/External/daisyrose-r16.gguf","verifyPolicy":{"requireNpuFriendly":true,"requireNoGroupedSsm":true,"maxAllowedRank":32}}'

# 3. Emit canonical DR-G2 contract + evidenceTemplate (or capture directly with SystemAwakeGuard via dr-g2-capture)
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"dr-g2-plan","screenConfig":{"baseModel":"swift-pq2","includeMultimodalChecks":false,"candidateAdapterSha256":"<CANDIDATE_SHA256>","candidateMaxRank":16,"candidateTargetCount":128,"scaleGrid":[0.0,0.25,0.5,0.75,1.0,1.5,2.0]}}'

# 3b. Sleep-inhibited two-server loopback capture, then generate blinded owner review packet + unblind/score
bonsai-tools lora-diff-audit --request C:/External/dr-g2-capture-request.json
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"dr-g2-blind-packet","evidencePath":"C:/External/dr-g2-evidence.json","outputPath":"C:/External/dr-g2-blind-packet.json","blindConfig":{"responsesPath":"C:/External/dr-g2-responses.json","blindKeyPath":"C:/External/dr-g2-blind-key.json"}}'
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"dr-g2-apply-reviews","evidencePath":"C:/External/dr-g2-evidence.json","outputPath":"C:/External/dr-g2-evidence-reviewed.json","applyConfig":{"reviewPath":"C:/External/dr-g2-blind-packet-reviewed.json","blindKeyPath":"C:/External/dr-g2-blind-key.json"}}'

# 4. Score completed held-out evidence + human reviews fail-closed
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"dr-g2-score","evidencePath":"C:/External/dr-g2-evidence-reviewed.json"}'

# 4b. Conditional DR-W1 / DR-Alt operations: algebraic edit, variable-rank reallocation, patch folding, ridge realignment, or PEFT import
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"edit-adapter","adapterPath":"C:/External/daisyrose-r32.gguf","outputPath":"C:/External/daisyrose-edited.gguf","editConfig":{"targetMaxRank":16,"depthGainStart":0.5,"depthGainEnd":1.5,"depthGainCurve":"cosine","orthogonalizeAgainstRankOnePath":"C:/External/orca.gguf","orthogonalizeSide":"b","alphaEditNullSpaceRidge":0.01}}'
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"reallocate-ranks","adapterPath":"C:/External/daisyrose-r32.gguf","outputPath":"C:/External/daisyrose-reallocated.gguf","reallocateConfig":{"maxPayloadBytes":67108864,"maxRank":32,"diagnosticPromptSha256s":["<DIAGNOSTIC_PROMPT_SHA256>"],"siteSensitivities":[{"target":"blk.63.ffn_down.weight","reasoningKlGap":2.4,"thinkingOffKlPenalty":0.0}]}}'
bonsai-tools lora-diff-audit --request C:/External/fold-patch-request.json
bonsai-tools lora-diff-audit --request C:/External/realign-factors-request.json
bonsai-tools lora-diff-audit --request C:/External/import-peft-request.json

# 5. After DR-G2 passes, write schema-validated candidate lock JSON (refuses synthetic fixtures when defaultEnabled=true; supports host "huggingface", "github", or "github-release")
bonsai-tools lora-diff-audit --json '{"schemaVersion":1,"operation":"generate-lock","adapterPath":"C:/External/daisyrose-r16.gguf","outputPath":"C:/External/daisyrose-lora.lock.json","lockSource":{"host":"github-release","repository":"<OWNER>/<REPO>","release":"<TAG>","revision":"<40_HEX>","path":"<FILE>.gguf","filename":"<FILE>.gguf","url":"https://github.com/<OWNER>/<REPO>/releases/download/<TAG>/<FILE>.gguf","license":"Apache-2.0"},"defaultEnabled":true,"defaultScaleTenths":10}'
#    For Hugging Face or raw GitHub tree files (no release tag), set host to "huggingface" (https://huggingface.co/<OWNER>/<REPO>/resolve/<40_HEX>/<POSIX_PATH>) or "github" (https://raw.githubusercontent.com/<OWNER>/<REPO>/<40_HEX>/<POSIX_PATH>).
```

### Default-on promotion flip checklist

1. `MULTI_RANK_LORA_RELEASE`, `RSQ4KVF_RELEASE`, `NARIQ_TRELLIS_RELEASE`, and
   `BONSAI_2_5_RELEASE` in [inference.rs](../src/broker/src/inference.rs) are
   now pinned to `REVIEWED_0_1_35`
   (`0.1.35`, `3bc76f135bf23c538b3bb789aef382143ff36839`) in
   [prism-inference.lock.json](../config/prism-inference.lock.json).
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
