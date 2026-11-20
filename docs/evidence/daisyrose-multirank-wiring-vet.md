# DaisyRose Multi-Rank Wiring and Bounded Owner-Device Vet

Concise evidence record for opt-in stackable `MultiRankAdapter` (`F16`/`F32`, `NpuFriendly`) wiring across `..\prism-inference` (`DR-C1`) and `SwiftBonsai2Runner` (`DR-C2`), together with a zero-retraining bounded owner-device vet.

## Source revisions

- `..\prism-inference` `main` commits implementing multi-rank support:
  - `39ae592` (multi-rank F16/F32 kernels, `MultiRankAdapter`, `ActiveAdapters` / `Qwen35`)
  - `4c49f6f` (`Runtime` and `SwiftSession` secondary adapter loading and scale transitions)
  - `5aab2f4` (`prism-inference-server` stackable `MultiRankAdapter` + Orca support)
  - `7d63f8a` (`examples/swift_model_check`: `--multi-rank-adapter <path>[:<scale>]` and rollback receipt fields)
  - Source-bound release `swift_model_check.exe` binary built from `..\prism-inference` `HEAD` `f2efd9c`.
- `SwiftBonsai2Runner` `main` commits implementing Runner opt-in wiring:
  - `647dc42e` (`bonsai-artifacts`: `#![forbid(unsafe_code)]`, `MultiRankEnvelope`, `VerifiedMultiRankAdapter`, `verify_multi_rank_lora_gguf`)
  - `fa982da5` (`bonsai-cpu-policy`: `DaisyRoseLoraLock`, `MultiRankAdapterBudget`, `SwiftExecutionShape` / `SwiftContextPlan` multi-rank accounting)
  - `82e100f5` (`bonsai-broker`: inference, launch policy, launcher, and service validation for optional DaisyRose adapter + scale)
  - `fb5af9b5` (`bonsai-chat`: opt-in default-off DaisyRose Settings control, client wiring, and workspace persistence)
  - `f67f6b7a` (`tools(bonsai)`: `--daisyrose-lora` CLI/launch forwarding and 1-slot HTTP Range prefetch pipelining in `lora-diff-audit`) — also the `HEAD` that built `target\release\bonsai-tools.exe`.

## Clippy, policy, and test checks

### `..\prism-inference` (release profile, `--locked --features parallel-rows --all-targets -- -D warnings`)

- `cargo +nightly-2026-11-09 clippy --manifest-path ..\prism-inference\Cargo.toml -p prism-inference -p prism-inference-server --locked --release --features parallel-rows --all-targets -- -D warnings`: exit `0`.
- `cargo +nightly-2026-11-09 test --manifest-path ..\prism-inference\Cargo.toml -p prism-inference -p prism-inference-server --locked --release --features parallel-rows`:
  - `prism-inference` lib: `521 passed; 0 failed`.
  - `examples/swift_model_check.rs`: `16 passed; 0 failed`.
  - `prism-inference-server` lib: `71 passed; 0 failed`.
  - `tests/native_response.rs`: `4 passed; 0 failed`.
  - `examples/private_service_check.rs`: `5 passed; 0 failed`.

### `SwiftBonsai2Runner`

- `bonsai-artifacts` (`+nightly-2026-11-12`):
  - Clippy (`-p bonsai-artifacts --all-targets -- -D warnings`): exit `0`.
  - `cargo test -p bonsai-artifacts`: `15 passed; 0 failed` (`lib`), `2 passed; 2 ignored; 0 failed` (`tests/contained.rs`), `8 passed; 0 failed` (`tests/lifetime.rs`), `10 passed; 0 failed` (`tests/provisioning.rs`).
- `bonsai-cpu-policy` (`+nightly-2026-11-12`):
  - Clippy (`-p bonsai-cpu-policy --all-targets -- -D warnings`): exit `0`.
  - `cargo test -p bonsai-cpu-policy`: `52 passed; 0 failed`.
- `bonsai-broker` (`+nightly-2026-11-12`):
  - Clippy (`-p bonsai-broker` default and `--features json-candidate`, `--all-targets -- -D warnings`): exit `0`.
  - `cargo test -p bonsai-broker`: `0 failed` across all test targets (`136 passed; 2 ignored` `lib`, `23` `backend_http`, `3` `control_tokens`, `6` `fragmentation`, `15` `http_reception`, `20` `http_response`, `5` `launch_policy`, `15` `local_service`, `1 passed; 1 ignored` `markdown_worker`, `81 passed; 14 ignored` `native_model`, `6` `policy_boundaries`, `29` `request_policy`, `5` `runtime`, `6` `scheduler`, `3` `scheduler_simulation`, `13` `supervisor`).
- `bonsai-chat` (`+nightly-2026-11-12`):
  - Clippy (`-p bonsai-chat --all-targets -- -D warnings`): exit `0`.
  - Parallel `cargo test -p bonsai-chat`: `698 passed; 2 failed; 1 ignored` (all `main::tests`, `client::tests`, `client::effort::tests`, `window::interaction_tests`, and `window::persistence::tests`, including `daisyrose_settings_toggle_and_slider_default_off_and_reach_next_turn`, passed). Unrelated failures:
    - `recovery::tests::recovery_lock_rejects_a_leaf_redirect_at_open` (Windows symlink privilege `os error 1314`).
    - `image_export::tests::png_export_encodes_unpremultiplied_rgba_and_decodes_cleanly` (failed under parallel suite run; re-run standalone: `1 passed; 0 failed`).
- `bonsai-tools` (`+nightly-2026-11-12`):
  - Clippy (`-p bonsai-tools -p bonsai-artifacts -p bonsai-cpu-policy -p bonsai-broker -p bonsai-chat -p bonsai-ci --all-targets -- -D warnings`): exit `0` (full `--workspace` clippy failed earlier without CMake on default `PATH` while building unrelated `transcribe-cpp-sys`).
  - Filtered `bonsai-tools` tests after final edit:
    - `chat_launch`: `43 passed; 0 failed` (`lib`).
    - `cli`: `10 passed; 0 failed` (`lib`), `1 passed; 0 failed` (`tests/provisioning.rs`).
    - `lora_diff_audit`: `8 passed; 0 failed` (`lib`), `1 passed; 0 failed` (`tests/cli.rs`).
  - Full `cargo test -p bonsai-tools` stopped after `--lib` at `641 passed; 4 failed; 4 ignored` due to unrelated pre-existing/environment failures (`cpu_evaluation::source_comparison::tests` x3 and `startup_process::tests::compiler_overlay_preserves_inherited_profile_and_unicode` `os error 5`).
- `pwsh -NoLogo -NoProfile -File tools/ci/check_owned_languages.ps1`: exit `0`.

## Bounded owner-device vet (zero retraining)

### Predeclared budget and power observations

- Predeclared ceiling: `300 s` aggregate budget using prebuilt release binaries only (`target\release\bonsai-tools.exe` and `..\prism-inference\target\release\examples\swift_model_check.exe`), zero compilation inside the timer, zero retraining.
- Initial wrapper invocation exited in `1.15 s` at CLI JSON parsing (`syntax error at line 1, column 1`) before starting download or inference because Windows PowerShell 5.1 `Set-Content -Encoding UTF8` wrote a UTF-8 BOM. The BOM-less follow-up run declared `remainingBudgetSeconds = 280` inside the `300 s` aggregate ceiling and completed cleanup in `66.83 s` (`67.98 s` aggregate across both).
- Power (`GetSystemPowerStatus`):
  - `AC=Battery`, `BatteryFlag=0`, `BatterySaver=Off` throughout every snapshot.
  - Charge observed: `46%` before Step 4A -> `43%` after Step 4A -> `43%` before/after Arm 1 -> `43%` before/after Arm 2 -> `43%` after scratch cleanup.
  - Equal endpoint snapshots do not rule out intervening changes, and no performance or throughput qualification is claimed on battery.

### Step 4A: `bonsai-tools.exe lora-diff-audit` (`export-denoised` + `verify-adapter`)

- Scope: `source.kind = "pinned-daisyrose"`, `layers = [0, 3, 31, 63]`, `targetEnvelope = "npu-friendly"`, `exportDtype = "f16"`, `maxRank = 16`, `oversampling = 16`, `powerIterations = 2`, `minSpikeRatio = 1.0`, `controls = { timeBudgetSeconds: 120, workers: 16, httpConcurrency: 4 }`.
- `export-denoised`: exit `0` at `51.73 s` (`elapsedMilliseconds: 51571`), `transferredBytes = 2,684,685,600` (`1,342,342,800` base + `1,342,342,800` candidate), output size `6,708,288` bytes (`16` target pairs / `32` tensors, `minRank = 7`, `maxRank = 16`, `uniformRank = false`, `alpha = 16.0`, SHA-256 `58a1eb245ec9b2564e4232de0919bf7f2cc90c51dbef1fcfc79ac3a021b61fe6`).
- `verify-adapter` (`requireNpuFriendly = true`, `requireNoGroupedSsm = true`): exit `0` at `51.84 s`, `verified = true`, `detectedEnvelope = "npu-friendly"`, `preservesHexagonFfnChains = true`, `excludesGroupedSsmOut = true`, `prismInferenceAlphaCompatible = true`, `publisherEngineAlphaCompatible = true`.

### Step 4B: `swift_model_check.exe` on `Swift-Bonsai-2-PQ2_0.gguf` + Orca `1.3` (`--row-workers 8`, `auto`, tokens `1,2,3`)

- **Arm 1 (`--rank-one-adapter <Orca> 1.3 forward`)**: exit `0` at `59.24 s`.
  - Step 0 (`token 1`): `greedy_token = 198`, `checkpoint_bytes = 156970616`, `parallel_row_operations = 497`, `logits-0.f32le` SHA-256 `e82ce1b7a03c98f10060dcd673436324aee1e8eaa64105d3dde917cba0b760de`.
  - Step 1 (`token 2`): `greedy_token = 16`, `checkpoint_bytes = 157036152`, `parallel_row_operations = 994`, `logits-1.f32le` SHA-256 `585e889861a121bd27b8c522588337f3a7a22015873f2f9a2cded04627de2b53`.
  - Step 2 (`token 3`): `greedy_token = 2`, `checkpoint_bytes = 157167224`, `parallel_row_operations = 1491`, `logits-2.f32le` SHA-256 `b2c9c6861da34afd49a6f957ea48bba45b10f133ba01decd47e1b0f8b5692357`.
- **Arm 2 (`--rank-one-adapter <Orca> 1.3 --multi-rank-adapter <Daisy>:0.0 forward`)**: exit `0` at `66.53 s`.
  - Loaded receipt: `multi_rank_adapter_sha256 = 58a1eb245ec9b2564e4232de0919bf7f2cc90c51dbef1fcfc79ac3a021b61fe6`, `multi_rank_max_rank = 16`, `multi_rank_scale = 0.0`, `multi_rank_scale_bits = "00000000"`, `multi_rank_target_pairs = 16`.
  - All three steps matched Arm 1 **bit-exact** on `greedy_token`, `checkpoint_bytes`, `parallel_row_operations`, and `logits-0..2.f32le` SHA-256 (`e82ce1b7a03c98f10060dcd673436324aee1e8eaa64105d3dde917cba0b760de`, `585e889861a121bd27b8c522588337f3a7a22015873f2f9a2cded04627de2b53`, `b2c9c6861da34afd49a6f957ea48bba45b10f133ba01decd47e1b0f8b5692357`), while appending `:multi-rank-lora-v1:f5f3bd8f53ef98880d3410ac9d160cff94f2478d2929fe07a41b3309b02e90a3:00000000` to `identity`.
- **Arm 3 (`--multi-rank-adapter <Daisy>:0.5 rollback`)**: **skipped** because the PowerShell 5.1 wrapper threw `MethodNotFound` on `[IO.Path]::GetRelativePath` right after Arm 2 output and entered `finally` cleanup (`scratchExistsAfter = False` at `66.83 s`). Per `AGENTS.md`, the trial was not restarted or retried; live 27B `:0.5` rollback on real weights remains open alongside Track B.

## Scope limits and remaining open gates

- Zero retraining was performed; the vet exercised exported denoised spikes (`layers: [0,3,31,63]`), not a qualified Track B (`DR-B1`) adapter.
- `config/prism-inference.lock.json` remains pinned at `v0.1.32` (not repinned), and no `config/daisyrose-lora.lock.json` was created.
- `DR-B1` (external OPD training), `DR-B2`/`DR-G2` `--publisher-engine` three-way quality screen, live 27B nonzero rollback and matched-power X1/X2 performance/RAM qualification, and owning CI (`source-checks`, `foundation`) remain open.

## Follow-up AC-powered vet attempt (closed — HTTP export budget exceeded)

A bounded AC-powered follow-up vet was launched to attempt the skipped `:0.5` rollback check and a matched-AC overhead arm without compilation inside the timer, and closed without completing any model arm:

- **Predeclared budget and prebuilt inputs (`budgetSeconds: 300`, zero compilation inside timer):**
  - `target\release\bonsai-tools.exe`: `12,482,560` bytes, SHA-256 `7c44c7e332a0e8f6b1a5943ac8a0edd77a9b6ff8ad83e3d8294507daf9b42331`.
  - `..\prism-inference\target\release\examples\swift_model_check.exe`: `3,704,320` bytes, SHA-256 `eefa53205fee4b795d52db3ba52ca6f999e39b0d5f3e5ebc850a779b5859164c`.
  - Existing local weights inspected at intake: `Swift-Bonsai-2-PQ2_0.gguf` (`7,206,168,928` bytes) and Orca rank-1 adapter (`9,682,464` bytes).
- **Power (`GetSystemPowerStatus`):**
  - Before: `ac_online: true` (`raw_ac_line_status: 1`), `battery_saver: false`, `battery_flag: 8`, `battery_percent: 65`.
  - After: `ac_online: true` (`raw_ac_line_status: 1`), `battery_saver: false`, `battery_flag: 8`, `battery_percent: 66`.
  - Endpoint AC snapshots alone do not rule out intervening changes, and no inference arm ran.
- **Outcome:** `status: "failed"` at `totalElapsedSeconds: 120.629`. `bonsai-tools lora-diff-audit` (`export-denoised`) exceeded `timeBudgetSeconds: 120` during live Hugging Face HTTP Range streaming and exited with `export-denoised did not publish/complete: budget-exceeded-not-published`. All four model arms (`arm1`–`arm4`) and `comparisons` remained `null`, and temporary scratch cleanup completed (`scratchExistsAfter: false`).
- **Disposition:** Closed without restart or retry per `AGENTS.md`. Future AC-powered 27B rollback and matched-AC performance trials should consume an already-local or pinned adapter rather than coupling live Hugging Face HTTP Range export into the inference trial clock.

## Offline AC-Powered 27B `:0.5` Rollback and Overhead Vet

Using the new offline `generate-test-adapter` subcommand in `bonsai-tools lora-diff-audit`, a separate single-run AC-powered vet exercised `:0.0` bit-exact parity, `:0.5` nonzero logits perturbation, matched-AC 4-layer forward timing, and `:0.5` full-state `rollback` on `Swift-Bonsai-2-PQ2_0.gguf` + Orca `1.3` with zero network I/O and zero compilation inside the trial timer.

### Predeclared budget, binaries, power, and cleanup

- **Predeclared budget and scope:** `budgetSeconds: 300`, `zeroCompilationInsideTimer: true`, `zeroNetworkInsideTimer: true`, total elapsed **`33.875 s`**, `scratchExistsAfter: false`.
- **Prebuilt binaries:**
  - `target\release\bonsai-tools.exe` (built from Runner `574b961adbc8278fb2d883fe749027afff90ee65`): `12,755,968` bytes, SHA-256 `e3df9d4843085b0552d52ab792e36036c1ca3654f31d913439f5148937f02649`.
  - `..\prism-inference\target\release\examples\swift_model_check.exe` (built from `f2efd9c19d6fead6a9b838f52bdc5ecfea3134c4`): `3,704,320` bytes, SHA-256 `eefa53205fee4b795d52db3ba52ca6f999e39b0d5f3e5ebc850a779b5859164c`.
  - Local weights: `Swift-Bonsai-2-PQ2_0.gguf` (`7,206,168,928` bytes), Orca rank-1 adapter (`9,682,464` bytes).
- **Power (`GetSystemPowerStatus`):**
  - `powerBefore`, all per-arm `powerBefore`/`powerAfter` snapshots (`arm1`–`arm4`), and post-cleanup `powerAfter` all observed `ac_online: true` (`raw_ac_line_status: 1`), `battery_flag: 1`, `battery_saver: false`, `battery_percent: 100`. Equal endpoint snapshots alone do not establish uninterrupted stability.

### Step results (`--row-workers 8`, `auto`, tokens `1,2,3`)

- **Test fixture (`generate-test-adapter`, `layers: [0, 3, 31, 63]`, `npu-friendly`, `f16`, `maxRank: 16`):** `78.206 ms`, `6,818,400` bytes, SHA-256 `fab91c769b74d7935b56edb6489d7987889b1c3e0b710c2a39356864aa729e8a`, `16` pairs (`32` tensors), `alpha = 16.0`, `preservesHexagonFfnChains: true`, `excludesGroupedSsmOut: true`.
- **Arm 1 (`Orca 1.3` baseline `forward`):** exit `0`, `wallMilliseconds = 7745.974`, `evalTotalMilliseconds = 485.267` (`188.991`, `151.641`, `144.634`), `greedy_token = 198 / 16 / 2`, `parallel_row_operations = 497 / 994 / 1491`, `logits-0..2.f32le` SHA-256 `e82ce1b7a03c98f10060dcd673436324aee1e8eaa64105d3dde917cba0b760de`, `585e889861a121bd27b8c522588337f3a7a22015873f2f9a2cded04627de2b53`, `b2c9c6861da34afd49a6f957ea48bba45b10f133ba01decd47e1b0f8b5692357`.
- **Arm 2 (`Orca 1.3 + synthetic:0.0` `forward`):** exit `0`, `wallMilliseconds = 6533.245`, `evalTotalMilliseconds = 445.507` (`161.936`, `134.13`, `149.441`), `multi_rank_scale_bits = "00000000"`. Matched Arm 1 **bit-exact** on all 3 steps (`greedy_token`, `checkpoint_bytes`, `parallel_row_operations`, and `logits-0..2.f32le` SHA-256s), with `:multi-rank-lora-v1:a575b4c7e7d6f9e1ccc574afbb4b729e5c064b06809be1d69bfaf49e1f1fe1b7:00000000` in `identity`.
- **Arm 3 (`Orca 1.3 + synthetic:0.5` `forward`):** exit `0`, `wallMilliseconds = 6511.256`, `evalTotalMilliseconds = 450.261` (`161.372`, `150.477`, `138.413`), `multi_rank_scale_bits = "3f000000"`, `identity` suffix `:multi-rank-lora-v1:a575b4c7e7d6f9e1ccc574afbb4b729e5c064b06809be1d69bfaf49e1f1fe1b7:3f000000`. All `3/3` steps emitted finite perturbed logits (`ff15d8048e9cfa187aa3b51cc0c2eb2e98cee1101199c40055a0e3367564d68e`, `0bcbadb1de55b836ad982840e7c6b90003bd8649beaa90be15f0691a571a0250`, `61870b3d392f46ee604fcb5a0dd9361ca9ddbc3ea999bee38d98443505fd098b`) while preserving `greedy_token = 198 / 16 / 2` and `parallel_row_operations = 497 / 994 / 1491`. Matched-AC 4-layer ratios vs cold-first Arm 1 were `evalRatioZeroOverBase = 0.9181`, `evalRatioHalfOverBase = 0.9279`, `wallRatioZeroOverBase = 0.8434`, `wallRatioHalfOverBase = 0.8406` (warm Arm 3 vs Arm 2: `1.0107` eval, `0.9966` wall).
- **Arm 4 (`Orca 1.3 + synthetic:0.5` `rollback`):** exit `0`, `wallMilliseconds = 12432.531`, `status = "passed"`, `multiRankAdapterSha256 = fab91c769b74d7935b56edb6489d7987889b1c3e0b710c2a39356864aa729e8a`, `multiRankAdapterScaleBits = "3f000000"`, `multiRankAdapterMaxRank = 16`, `cancelledProvisionalToken = 1`, `continuationToken = 2`, `snapshotPayloadBytesWritten = 633380864`, all 6 `completedSteps` passed (`exact-model-and-profile-loaded`, `complete-pre-cancellation-state-and-logits-captured`, `extended-prompt-and-one-evaluated-token-cancelled`, `complete-state-tokens-and-logits-restored-before-any-reset`, `post-rollback-one-token-continuation-captured`, `fresh-recompute-continuation-token-full-state-and-logits-match`).
- **Scope limits:** Synthetic 4-layer (`16` of `128` `npu-friendly` targets) fixture validates runtime multi-rank numerical parity, nonzero logits application, full-state rollback, and 4-layer wiring overhead only; it does not establish trained Track B (`DR-B1`) quality, full 64-layer (`128`-pair) production throughput/RAM, or owning CI acceptance.
