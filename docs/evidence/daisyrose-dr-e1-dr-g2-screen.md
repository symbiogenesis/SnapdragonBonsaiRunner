# DaisyRose `DR-E1` Extraction, Full 64-Layer `DR-C1` Vet, and Battery CPU `DR-G2` Screen

Concise evidence record for full 64-layer zero-retraining spectral-denoised DaisyRose extraction (`DR-E1`), full 64-layer numerical parity and rollback vetting (`DR-C1`), and the bounded battery CPU empirical `DR-G2` direct-stack screen (closed incomplete at `1/8` prompt pairs within its 900 s ceiling).

## Tooling and source revisions

- Pure-Rust resumable multi-chunk GGUF merge (`merge-adapters`) and algebraic weight editing (`edit-adapter`) checked into local `main` at commit `937b27c0` (hosted CI acceptance pending; not pushed).
- Prebuilt `bonsai-tools.exe`: `12,886,016` bytes, SHA-256 `867d16038a7decfcd8230cf9800223f561b62f6d845a16547592af1561a10ee2`.
- Prebuilt `..\prism-inference` binaries:
  - `swift_model_check.exe` (source commit `f2efd9c19d6fead6a9b838f52bdc5ecfea3134c4`): `3,704,320` bytes, SHA-256 `eefa53205fee4b795d52db3ba52ca6f999e39b0d5f3e5ebc850a779b5859164c`.
  - `prism-inference-server.exe` (with `supports_daisyrose_scale` and parallel CPU row workers).
- Local base weights and rank-1 adapter:
  - `Swift-Bonsai-2-PQ2_0`: `5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5`.
  - Orca rank-1 adapter (scale `1.3`): `f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42`.

## `DR-E1` full 64-layer extraction and verification (local pass)

- **Source pins:** `RadixArk/DaisyRose-27B` revision `64a4ec665091bb0c1931babce4e97e686eef3c5c` vs `Qwen/Qwen3.8-27B` revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` (shared `model.safetensors.index.json` SHA-256 `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`).
- **Execution and power (`GetSystemPowerStatus`):** 8 resumable 8-layer `export-denoised` chunks (`0..63`, `npu-friendly`, `f16`, `maxRank = 32`, `oversampling = 16`, `powerIterations = 2`, `minSpikeRatio = 1.0`, `timeBudgetSeconds = 300`, `workers = 16`, `httpConcurrency = 4`) plus `merge-adapters` (`r = 32`), offline `edit-adapter` (`targetMaxRank = 16`), and `verify-adapter`. Elapsed `807.971 s` (after reusing verified `chunk-00-07`); battery `91% -> 87%`, `acOnline = false`, `batterySaver = false`.
- **Merged `r = 32` artifact:** `133,326,912` bytes (`133,308,416` payload bytes / `127.1328 MiB`), `128` pairs (`256` tensors), `minRank = 7`, `maxRank = 32`, `alpha = 32.0`, uniform `F16`, SHA-256 `8820b56ae09544096711b5a5c97c7ce90f42a5b1dd961edb5e23322be1224705`, `verify-adapter`: `verified = true`, `detectedEnvelope = "npu-friendly"`, `preservesHexagonFfnChains = true`, `excludesGroupedSsmOut = true`, dual alpha compatible.
- **Offline-derived `r = 16` artifact (`edit-adapter`, `targetMaxRank = 16`):** `66,930,944` bytes (`66,912,256` payload bytes / `63.8125 MiB`), `128` pairs (`256` tensors), `minRank = 7`, `maxRank = 16`, `alpha = 16.0`, uniform `F16`, SHA-256 `1b850494b8759e24c511aac39621db3e88a4d708a075012b47dca2682001c569`, `verify-adapter`: `verified = true`, `detectedEnvelope = "npu-friendly"`, `preservesHexagonFfnChains = true`, `excludesGroupedSsmOut = true`, dual alpha compatible.

## Full 64-layer `DR-C1` numerical parity and rollback vet (local pass)

- **Budget, scope, and cleanup:** Predeclared `300 s` ceiling, `zeroCompilationInsideTimer = true`, `zeroNetworkInsideTimer = true`, elapsed **`138.61 s`**, `scratchExistsAfter = false`.
- **Power (`GetSystemPowerStatus`):** `acOnline = false`, `batterySaver = false`, `batteryFlag = 1` across `powerBefore` (`87%`), all per-arm snapshots (`arm1`–`arm3`: `87%`; `arm4`: `87% -> 86%`), and `powerAfter` (`86%`). Equal endpoint percentages on `arm1`–`arm3` do not prove uninterrupted stability, and no performance qualification is claimed on battery.
- **Arm results (`swift_model_check.exe --row-workers 8`, `auto`, tokens `1,2,3` on `daisyrose-npu-r16.gguf`):**
  - **Arm 1 (`Orca 1.3` baseline `forward`):** exit `0`, `wall = 32,188.938 ms`, `eval = 3,469.514 ms`, `greedy = 198 / 16 / 2`, `checkpoint_bytes = 156970616 / 157036152 / 157167224`, `parallel_row_operations = 497 / 994 / 1491`, `logits-0..2.f32le` SHA-256s `e82ce1b7a03c98f10060dcd673436324aee1e8eaa64105d3dde917cba0b760de`, `585e889861a121bd27b8c522588337f3a7a22015873f2f9a2cded04627de2b53`, `b2c9c6861da34afd49a6f957ea48bba45b10f133ba01decd47e1b0f8b5692357`.
  - **Arm 2 (`Orca 1.3 + r16:0.0` `forward`):** exit `0`, `wall = 29,256.726 ms`, `eval = 2,337.674 ms`, `multi_rank_scale_bits = "00000000"`. Matched Arm 1 **bit-exact** across all 3 tokens on `greedy_token`, `checkpoint_bytes`, `parallel_row_operations`, and `logits-0..2.f32le` SHA-256s, with `:multi-rank-lora-v1:20716906c96d29581a48c8bba54faded836dcc7fa610edbcbca4e6eb890c2ebf:00000000` in `identity`.
  - **Arm 3 (`Orca 1.3 + r16:1.0` `forward`):** exit `0`, `wall = 28,025.444 ms`, `eval = 3,563.561 ms`, `multi_rank_scale_bits = "3f800000"`, `identity` suffix `:3f800000`. All 3 steps produced finite perturbed logits (`7d6c9d3bc8b69a1560e1787073769d4cee2890865a7a3faef2916229592fd1dc`, `aeda064e70a5941e4a7b229e6e5774184e9a5005e2a8a46c8d4c01e4964666dd`, `d19d7e889b98de0d22633ce06bdf381bf273fe72a04571fa772df4b673baaab3`), preserving `parallel_row_operations = 497 / 994 / 1491`.
  - **Arm 4 (`Orca 1.3 + r16:1.0` `rollback`):** exit `0`, `wall = 48,153.974 ms`, `status = "passed"`, all 6 `completedSteps` passed (`cancelledProvisionalToken = 1`, `continuationToken = 2`, `snapshotPayloadBytesWritten = 633,380,864`, `multiRankAdapterMaxRank = 16`, `multiRankAdapterScaleBits = "3f800000"`).

## `DR-G2` direct-stack empirical screen (closed incomplete at `1/8` pairs on battery CPU)

- **Budget, cleanup, and prior wrapper aborts:** Predeclared `900 s` aggregate ceiling. Two prior pre-inference wrapper setup aborts completed `0` generations (`28 s` on `.data[0]` readiness indexing and `39 s` on PowerShell unwrapping a 1-element `lora` array). The follow-up run declared `budgetSeconds = 820` (`67 s + 810.45 s = 877.45 s <= 900 s`), completed cleanup cleanly (`lingeringServerCount = 0`, `credDirExistsAfter = false`), and closed without restarting or retrying.
- **Power (`GetSystemPowerStatus`):** `acOnline = false`, `batterySaver = false`, `batteryFlag = 1` throughout: `powerTrialBefore = 85%` -> `serverA = 85%` -> `zeroCtrl = 85%` -> `creative-001` baseline (`85% -> 85%`) -> `creative-001` candidate (`85% -> 82%`) -> `powerTrialAfter = 73%`. No performance or throughput qualification is claimed on battery.
- **Plan contract & server readiness:** `dr-g2-plan` `contractSha256 = 963b26d9cb7d2e5dcba0cf4e3b5290706594f7e89dac7f0cfbb8a6661dc39413`, `scaleGrid = [0.0, 0.5, 1.0, 1.5, 2.0]`, `selectedScale = 1.0`. Both Server A and Server B ran `--cpu-family x2 --row-workers 8 --kv-cache rsq4 --seed 42` with `startup_projection.effective = "cpu"` (no local Hexagon HTP package installed; Vulkan GPU placement not selected). Server B reported `supports_daisyrose_scale = true`, `daisyrose_adapter_max_rank = 16`, and `daisyrose_adapter_sha256 = 1b850494b8759e24c511aac39621db3e88a4d708a075012b47dca2682001c569`.
- **Completed live HTTP checks and empirical pairs (`humanReviewed = false`):**
  - **`zero-ctrl-001` (`promptSha256 = 7eceafe93ef72a2c6ab2eec0609f89f5e82d6221ce7be76d7b7c14efe89346ee`, Thinking-ON, cap `64`):** **bit-exact parity passed.** Both Server A (no `--daisyrose-lora`, `orca = 1.3`) and Server B (`--daisyrose-lora`, `scale = 0.0`) emitted `51` completion tokens (`67` prompt tokens, `finish_reason = "stop"`, answer `391`), identical `responseSha256 = cb08876d335e8c62d6d8278be979f0804a97eb23743b2f451fac9a73a0f0854f` (`16,236.72 ms` vs `16,750.93 ms`).
  - **`creative-001` (`promptSha256 = 59f8fe51591185c778ad862c343a3c2da5ddbea793140044184e09ad110afa59`, `creative-prose`, Thinking-OFF, cap `1024`):** non-degenerate delta (`equalSha = false`), zero unwanted refusals:
    - Baseline (`scale = 0.0`): `427` completion tokens, `71,817.17 ms` (`168.19 ms/tok` diagnostic on battery CPU), `finish_reason = "stop"`, `responseSha256 = 77c7ed99a63e01802e1a59401a99353044723ee11c300e0a02c601f9bf68f9eb`.
    - Candidate (`scale = 1.0`): `334` completion tokens, `97,259.84 ms` (`291.20 ms/tok` diagnostic on battery CPU; charge dropped `85% -> 82%` during this arm), `finish_reason = "stop"`, `responseSha256 = f85deceae3c0a6877b4c0e73e6c372d93216145574095ef152e55755fa905a8f`.
  - **`if-001` (`instruction-following`, Thinking-ON, cap `4096`):** canceled cleanly by the remaining-budget deadline at `810.45 s` due to `~3.4–5.9 tok/s` battery CPU decode speed without accelerator offload. The `instruction-following`, `coding`, and `reasoning` pairs (`7/8` held-out prompts) did not complete inside this closed budget.

## Remaining gates before default-on promotion

1. **Complete `DR-G2` and human review:** Run all 4 required `DR-G2` categories on AC power using hardware acceleration (Hexagon NPU residency, `--placement vulkan`, or `--publisher-engine`), perform blinded side-by-side human review (`humanReviewed = true`), and pass `dr-g2-score` (or trigger `DR-W1` via `edit-adapter` on a disjoint fit split if needed).
2. **Remote hosting and candidate lock:** Upload the verified final `.gguf` to an immutable remote host (`huggingface` or `github` raw URL + 40-hex commit) and generate `config/daisyrose-lora.lock.json` via `bonsai-tools lora-diff-audit` (`generate-lock`).
3. **Prism release bundle and feature enablement:** Publish a `prism-inference` `v0.1.33+` GitHub release bundle (`foundation-bundle.yml`), repin `config/prism-inference.lock.json`, flip `MULTI_RANK_LORA_RELEASE` and `pinned_daisyrose_lora_artifact` (+ `.github/ci/impact.json`, `docs/licensing.md`, `NOTICE`).
4. **Device performance and owning CI:** Run matched-AC X1/X2 full-adapter throughput and RAM qualification (`<= 10%` overhead gate), push commits/tags when requested, and pass owning CI (`source-checks`, `foundation`).
