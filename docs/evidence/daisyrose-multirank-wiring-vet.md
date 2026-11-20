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
