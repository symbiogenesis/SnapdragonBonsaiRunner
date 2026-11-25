# Ternary Bonsai 2.5 27B X2 vetting trials

This note records bounded AC-logged owner-device trials on Snapdragon X2 for `prism-ml/Ternary-Bonsai-2.5-27B-GGUF` (`28a7093cc25bc151cc2b284c5570f12ce343fdcf`), executed without compilation or network downloads inside trial allowances.

## Provisioned artifacts verified by SHA-256

- `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf`: `5,993,834,848` bytes, SHA-256 `929560ec9d37cf3cadd4b00cc87b51c570082a628dd830b4692dc0b2ed18dbbe`
- `Ternary-Bonsai-2.5-27B-PQ2_0.gguf`: `7,253,354,848` bytes, SHA-256 `103a11c94324029d8bae3f8152f30bba414730064cfb77438f8589db72e1aa85`
- Shared vision projector `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`: `629,246,976` bytes, SHA-256 `6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903`
- Candidate DaisyRose LoRA `daisyrose-npu-r16-drw1.gguf`: `66,930,944` bytes, SHA-256 `ba307a5f60372c69c2f4b773cc04771933a5db16b09aa5bc754b05743a7bc5d0`

## Trial #1 — Bounded 7-arm `swift_model_check.exe` verification (`<= 300 s` budget)

- **Timing & power:** `106.113 s` total elapsed (`2026-11-25T04:21:24.764Z` to `04:23:10.877Z`, budget `300 s`, prebuilt `swift_model_check.exe`, `--row-workers 12`, `auto` CPU/NEON, tokens `1,2,3,4,5,6,7,8` for forward and `1,2,3` for rollback). Every arm recorded `GetSystemPowerStatus` before/after as `acOnline: true`, `batteryLifePercent: 99`, `batterySaverOn: false` (endpoint observations only; continuous mid-run power stability is unproven). `lingeringProcessCount: 0`; temp evidence cleaned up.

| Arm | Configuration | Wall (ms) | Peak WS (B) | Cold tok 0 (ms) | Warm 1..7 avg (ms) | Outcome / key check |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `arm1` | `PTQ1_0` base forward | `13,088.219` | `6,341,107,712` | `266.960` | `226.317` | Greedy `[6253, 220, 4, 5, 9, 24218, 11, 198]`; logits `f6202f13...` |
| `arm2` | `PQ2_0` base forward | `14,747.296` | `7,600,562,176` | `161.168` | `177.778` | Identical 8/8 greedy tokens; logits `44b55481...` (differs from `PTQ1_0`) |
| `arm3` | `PTQ1_0` + Orca `1.3` forward | `13,295.870` | `6,357,127,168` | `258.084` | `258.080` | Execution digest matched `e8dce51125fb57607b007789d4837cfc0856715e8cab6846010c1c6844ad127f` |
| `arm4` | `PQ2_0` + Orca `1.3` forward | `14,685.316` | `7,606,657,024` | `192.418` | `164.617` | Execution digest matched `d6b597fe8e1d1d559a41cbba127756be42401fbf5a44866cd4527471770f94ca` |
| `arm5` | `PTQ1_0` + DaisyRose `:0.0` forward | `13,400.044` | `6,398,349,312` | `283.687` | `227.839` | All 8 logit SHA-256 digests bit-identical to `arm1` |
| `arm6` | `PTQ1_0` + DaisyRose `:1.0` forward | `13,582.725` | `6,402,031,616` | `276.594` | `236.056` | `+4.30%` warm decode over `arm1`; same 8/8 greedy tokens |
| `arm7` | `PTQ1_0` + DaisyRose `:1.0` rollback | `16,951.840` | `6,560,858,112` | — | — | `status: "passed"`, `6/6` steps (`cancelledProvisionalToken: 4`, `continuationToken: 4`) |

- **Trial #1 comparison outcome:** `PTQ1_0` reduced Peak WS by `-1,259,454,464` B (`-1.173 GiB`) and matched `PQ2_0` greedy tokens `8/8`, while `PQ2_0` ran faster on warmed CPU decode (`177.778 ms/tok` vs `226.317 ms/tok`, speedup ratio `0.7855`). Per selection criteria, `bonsai-2.5-pq2` was selected for Trial #2.

## Trial #2 — AC-logged `dr-g2-capture` on published `v0.1.35` (`<= 900 s` budget)

- **Preflight abort note:** A prior launch attempt aborted at preflight before creating any directory or starting any workload because `ACLineStatus` was `0`; Trial #2 was executed cleanly once AC power was restored.
- **Configuration:** Published `prism-inference-server.exe` (`v0.1.35`, `3bc76f135bf23c538b3bb789aef382143ff36839`, SHA-256 `8cf39507...`), `baseModel: "bonsai-2.5-pq2"` Orca-free, `daisyrose-npu-r16-drw1.gguf` `selectedScale: 1.0`, `rowWorkers: 12`, `kvCache: "rsq4"`, `contextTokens: 4864`, `cpuPrefillTokens: 16`, `cpuPrefillGdn: "wy-f32"`, `placement: "cpu"`, `reasoningEffort: "medium"`, `reasoningBudget: 448`, `20` excluded digests (`15` prior training hashes + `5` from `*-heldout-008`), fresh disjoint `4/4` prompt split + zero-scale control.
- **Timing, power & cleanup:** `886.959 s` total wrapper elapsed (`885,005 ms` inside `dr-g2-capture`, `2026-11-25T05:33:02.992Z` to `05:47:49.951Z`), AC online before/after and at all recorded arm endpoints (`ac_online: true`, `battery_percent: 84`, `battery_saver: false`; continuous mid-run power stability remains unproven), `sleepInhibitionActive: true`, `lingeringProcessCount: 0`, `credDirExistsAfter: false`.
- **Captured outcomes:**
  - `zero-ctrl-b25-001` (`zero-no-adapter` vs `zero-scale`): `completedZeroControls: 1`, `zeroScaleMismatches: 0` (`58` completion / `54` thinking tokens, `13,352 ms` vs `13,339 ms`, `finishReason: "stop"`).
  - `if-b25-001` (`instruction-following`): baseline `287 + 524` tokens (`441` thinking, `111,535 ms`), candidate `287 + 524` tokens (`440` thinking, `111,242 ms`); both `finishReason: "stop"`, `hit4096Truncation: false`, `loopDetected: false`.
  - `coding-b25-001` (`coding`): baseline `161 + 713` tokens (`472` thinking, `140,899 ms`), candidate `161 + 682` tokens (`474` thinking, `134,932 ms`); both `finishReason: "stop"`, `hit4096Truncation: false`, `loopDetected: false`.
  - `reasoning-b25-001` (`reasoning`): baseline `155 + 594` tokens (`496` thinking, `115,233 ms`), candidate `155 + 834` tokens (`506` thinking, `214,929 ms`); both `finishReason: "stop"`, `hit4096Truncation: false`, `loopDetected: false`.
  - `creative-b25-001` (`thinkingOff`): skipped cleanly before exhaustion (`completedThinkingOff: 0`, `budgetExceeded: true`, `status: "budget-exceeded-incomplete"`).
- **Sealed blind review packet (key and responses remain unopened):**
  - Directory: `%LOCALAPPDATA%\SwiftBonsai2\staging\daisyrose-dr-e1\dr-g2-awake-bonsai25-pq2-3bc76f1-r16-drw1-s10-heldout-001\`
  - `dr-g2-blind-packet.json` (`itemCount: 3`, SHA-256 `4b221d19b38c1108070fe562b4fe3eadc1af1a098b4d34f36346e70f0d0d134b`)
  - `dr-g2-evidence.json` (`evidenceSha256: 69a2fae92954f7d3c39c93f59fc1f33d6a29850be0de0a633e6b4f5788ed1564`, `contractSha256: e4f77b6f9a05feb47132329752d60a7326f9f4045af03e1d6461e209020937af`)
  - `responsesSha256: bd94f19cc5f0218d8c65f89e8ed6e9a68f0d996dfb02ef0d0cd4491bff4782c2` (`dr-g2-responses.json` unopened)
  - `blindKeySha256: 8ecc38316ec7156b8daa3d5171dba6d60a3d4d45c6db1a7ad2e52974a3648c9b` (`dr-g2-blind-key.json` unopened)

## Trial #3 — AC-logged Thinking-OFF supplement capture (`<= 300 s` budget)

- **Configuration & binaries:** Prebuilt `bonsai-tools.exe` (`14,538,240` bytes, SHA-256 `6c1dc5c5b9717819a61a5af4b2fcdc1768b050eac1db43d9b9a916d98b19d92e`, commit `b83fbea3`) and published `prism-inference-server.exe` (`v0.1.35`, `3bc76f13...`, SHA-256 `8cf39507...`) run with `supplementEvidencePath` / `supplementResponsesPath` pointing to sealed Trial #2 (`69a2fae9...`) under the identical `bonsai-2.5-pq2` Orca-free + `daisyrose-npu-r16-drw1.gguf` (`selectedScale: 1.0`, `contractSha256: e4f77b6f...`) contract (`innerTimeBudgetSeconds: 285`, zero compilation/network inside timer). Trial #2's directory remains untouched.
- **Timing, power & cleanup:** `138.826 s` total wrapper elapsed (`136,307 ms` inside `dr-g2-capture`, `2026-11-25T16:16:00.609Z` to `16:18:19.435Z`), AC online at preflight, `powerBefore`, `powerAfter`, and both `creative-b25-001` arm endpoints (`ac_online: true`, `battery_percent: 99`, `battery_saver: false`, `performanceComparisonEligible: true`; continuous mid-run power stability remains unproven), `sleepInhibitionActive: true`, `lingeringProcessCount: 0`, `credDirExistsAfter: false`.
- **Captured outcome (`status: "capture-completed"`, `completed: true`, `budgetExceeded: false`):**
  - `creative-b25-001` (`creative-prose`, `enableThinking: false`, cap `384`): baseline `77 + 242` tokens (`64,151 ms`, `finishReason: "stop"`), candidate `77 + 271` tokens (`63,573 ms`, `finishReason: "stop"`); `hit4096Truncation: false`, `loopDetected: false`, `unwantedRefusal: false`, distinct `responseSha256` (`8a3e04ab...` vs `b7c68f36...`).
- **Complete `4/4` sealed blind review packet (before delegated scoring):**
  - Directory: `%LOCALAPPDATA%\SwiftBonsai2\staging\daisyrose-dr-e1\dr-g2-awake-bonsai25-pq2-3bc76f1-r16-drw1-s10-heldout-001-complete\`
  - `dr-g2-blind-packet.json` (`itemCount: 4`, SHA-256 `6dc82c06739de52c3274b1d1704ca99c9c9d3ad603a448797fd43ccacc5b0cc1`)
  - `dr-g2-evidence.json` (`evidenceSha256: bda85f10ff332fa765fbd76840b691d02f912ed2aab90cd0b48edbf601dbc208`, `contractSha256: e4f77b6f9a05feb47132329752d60a7326f9f4045af03e1d6461e209020937af`, `completedZeroControls: 1`, `completedThinkingOn: 3`, `completedThinkingOff: 1`, `zeroScaleMismatches: 0`)
  - `responsesSha256: 99380cbfe46a24222c310d18089df44ba47344a665413b86838db7aabc1bfdd5` (`dr-g2-responses.json` remained unopened during scoring)
  - `blindKeySha256: b6f40f7a3f1c10bfe401fa474f5dce1a4dedca96756bf2c35a303a7e2775a562` (`dr-g2-blind-key.json` remained unopened until `dr-g2-apply-reviews`)

## Delegated blinded `DR-G2` review and fail-closed `dr-g2-score` verdict (`drG2Passed: false` — No-Go for default flip)

- **Reviewer provenance & tooling:** Owner-delegated blinded assessment (`reviewerId: "agent-delegated-by-owner"`, not an independent external human reviewer). All four `Arm A` / `Arm B` ratings were locked blind against `dr-g2-blind-packet.json` (`6dc82c06...`) before `dr-g2-blind-key.json` (`b6f40f7a...`) was unblinded via prebuilt `bonsai-tools.exe` `dr-g2-apply-reviews` and independently re-scored via `dr-g2-score`.
- **Retained review & score artifacts (same Trial #3 directory):**
  - `dr-g2-blind-packet-reviewed.json`: SHA-256 `19c40b843a14ca92dcedbf188305fc3d5f3dad33d97c9071d640b98cde0a1393`
  - `dr-g2-evidence-reviewed.json`: SHA-256 `a9110b2f498ec50e781d50081de54b470be76566aa29372f4181869efd57218b`
  - `dr-g2-apply-receipt.json`: SHA-256 `1722da9b4d1c5c6d9cd6467b5d1d314a47814e4ebfec202cffc66777461bee48`
  - `dr-g2-score-receipt.json`: SHA-256 `9d8e9a71dacdd672867168473bc2b617431fa08e2087994c15cd1a7582e93ff9`

| Prompt ID | Category | Blind placement (`Arm A` / `Arm B`) | Baseline (`0.0`) | Candidate (`1.0`) | Delta | Rationale summary |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `if-b25-001` | `instruction-following` (Thinking-ON) | `baseline` / `candidate` | `4.0` | `4.0` | `0.0` | Both produced byte-identical exact minified JSON matching schema, dependency filtering, sorting, and sums. |
| `coding-b25-001` | `coding` (Thinking-ON) | `baseline` / `candidate` | `1.5` | `3.0` | `+1.5` | Baseline used broken `used <= limit.saturating_sub(w)` (admitting `w > limit` at `used == 0`), lacked `else { break; }`, and omitted bullet list formatting; candidate used bullets and `else { break; }`, missing only `checked_add` near `u64::MAX`. |
| `reasoning-b25-001` | `reasoning` (Thinking-ON) | `candidate` / `baseline` | `1.5` | `2.5` | `+1.0` | Baseline computed wrong second-draw probability `1/8` and emitted invalid fraction `635/531 > 1`; candidate derived exact unreduced numerator/denominator `94754/219303` (`1606/3717`), erring only in final GCD reduction to `1504/3481`. |
| `creative-b25-001` | `creative-prose` (Thinking-OFF) | `candidate` / `baseline` | `3.5` | `3.2` | `-0.3` | Both cleared the `>= 3.0` floor with `0` unwanted refusals; baseline had cleaner physical restraint despite dawn-timing drift, while candidate kept night-before timing but introduced "chalk was old ... previous night's darkness" and "turned the dial" phrasing slips. |

- **Fail-closed `dr-g2-score` summary (`drG2Passed: false`, `status: "failed"`):**
  - **Gate 1 — `capabilityGain` (`passed: true`):** Overall Thinking-ON mean `3.1667` vs `2.3333` (`+0.8333`), zero per-category regressions (`coding` `+1.5`, `reasoning` `+1.0`, `instruction-following` `0.0`).
  - **Gate 2 — `antiLoopingPreservation` (`passed: true`):** Median Thinking-ON tokens `474.0` vs `472.0` (`ratio = 1.0042 <= 1.10`), `0` loops (`baselineLoops = 0`, `candidateLoops = 0`), `0` `4,096`-token truncations.
  - **Gate 3 — `creativeAndOrcaNonRegression` (`passed: false`):** `candidateMedianScore = 3.2 < baselineMedianScore = 3.5` (`n = 1`, `candidateMinScore = 3.2 >= 3.0`, `0` unwanted refusals) — triggers `failures: ["Thinking-OFF creative median score regressed (3.2000 < 3.5000)."]`.
  - **Gate 4 — `zeroScaleIdentity` (`passed: true`):** `controlCount = 1`, `mismatchCount = 0`.
- **No-Go / Hold decision for default-on promotion:** Fail-closed `DR-G2` Gate 3 failed on `creative-b25-001` (`3.2 < 3.5`, `n = 1`), `daisyrose-npu-r16-drw1.gguf` (`ba307a5f...`) currently has no immutable remote HTTPS URL (`huggingface` / `github` / `github-release`) required by `config/daisyrose-lora.lock.json` (`defaultEnabled: true`), and `Ternary-Bonsai-2.5-27B-PQ2_0` without DaisyRose scored `1.5/4.0` on coding and reasoning under `reasoningBudget: 448` while lacking head-to-head qualification against `Swift-Bonsai-2-PQ2_0`. Keep `Swift-Bonsai-2-PQ2_0` + Orca unchanged as default and `config/daisyrose-lora.lock.json` unpinned.

## Swifting Bonsai 2.5 exploration (No-Go)

- **Outcome:** **No-Go (stripped).** Neither `Swift-Bonsai-2` PQ2 diff extraction nor BF16 `ukisai/Swift-1.5-Qwen3.8-27b` minus `Qwen/Qwen3.8-27B` LoRA extraction (`npu-friendly` or full-linear `AnyLinear`) transferred useful anti-overthinking behavior onto `Ternary-Bonsai-2.5-27B-PQ2_0`. All Swifting source code, temporary Prism probe worktree/branch/binaries, and staged Swift adapters/receipts were removed completely; the `Ternary-Bonsai-2.5-27B` + `DaisyRose` path remains 100% Apache-2.0.
- **Arm 0 — PQ2 weight-diff audit (`75 s` AC, `battery_percent: 96`, `battery_saver: false`):** Comparing `Swift-Bonsai-2-PQ2_0` vs `Ternary-Bonsai-2-27B-PQ2_0` showed `0` flipped trits across all `npu-friendly` tensors (FP16 scale-only update, `relDelta = 0.00050–0.00095`, rank-32 SVD capturing only `1.5%–6.7%` of dequantized diff energy). Comparing `Ternary-Bonsai-2-27B-PQ2_0` vs `Ternary-Bonsai-2.5-27B-PQ2_0` showed `4.58%–12.08%` flipped trits (`relW = 0.26–0.42`), ruling out direct scale-delta transfer.
- **Arm 1 — `npu-friendly` BF16 `Swift-1.5-Qwen3.8-27b` LoRA (`128` sites, `32.0%` of full-linear factor Frobenius energy):**
  - Staged (now deleted) `swift15-npu-r32.gguf` (`134,236,448` B, SHA-256 `d144af4c499d0d9f12414c5189788fa7b09ce8c3ec3d35a5c45ebf71e58e6a3a`) and composed `swift15-daisyrose-npu-r32.gguf` (`134,039,808` B, SHA-256 `1413c89c2bb4190923411ba20be02cb5b4be38f37b90c21f295bf2b0075da2d8`; near-orthogonal to `daisyrose-npu-r16-drw1.gguf`, `meanCosine = +0.000148`, `|cos| <= 0.00805`).
  - `dr-g2-capture` on `swift15-daisyrose-npu-r32.gguf` (`885 s / 900 s` AC `97% -> 99%`, `reasoningBudget: 1024`): `if-swdr-001` (`e9681f11...`) produced `962` (`scale = 0.0`) vs `965` (`scale = 1.0`) thinking tokens (both hit the 1024 budget cap); remaining prompts skipped on budget guard.
  - Unclipped `scale-screen` #1 (`585.0 s / 600 s` AC `99% -> 99%`, `bonsai-2.5-pq2`, `reasoningEffort: "medium"`, `reasoningBudget: null`): on `diag-sw-math-001` (`2fc56c28...`), thinking tokens were `320` (`scale = 0.0`, `567` comp, `109.5 s`, `doubleCheck=1`), `324` (`scale = 1.0`, `540` comp, `108.6 s`, `doubleCheck=1`), and `212` (`scale = 2.0`, `-33.8%`, `424` comp, `84.8 s`, `markers=0`), all exact; `diag-sw-logic-001` (`d74d07e5...`) completed `scale = 0.0` (`717` think / `1,046` comp / `215.5 s`) before `1.0`/`2.0` skipped at budget.
  - Unclipped `scale-screen` #2 (`705.0 s / 720 s` AC `99% -> 100%`, `bonsai-2.5-pq2`): on `diag-sw-logic-001` (`d74d07e5...`), `scale = 0.0` produced `717` think / `1,046` comp / `231.3 s` (`221.14 ms/tok`, `markers=0`) vs `scale = 2.0` producing `717` think (`0.0%` reduction) / `1,069` comp / `233.0 s` (`217.99 ms/tok`, `verify=1`), both exact; `diag-sw-code-001` skipped at budget.
- **Arm 2 — Full-linear (`AnyLinear`) BF16 `Swift-1.5-Qwen3.8-27b` LoRA (`256` sites = `ffn_gate + ffn_up + npu-friendly`, `r = 16`, breaks Hexagon NPU fused FFN chains):**
  - Staged (now deleted) `swift15-full-r16.gguf` (`159,419,424` B, SHA-256 `35aa5710ad0242260a36696742a800c93450ccb5a61752549b1031ca0d6f7f57`; `ffn_gate + ffn_up` = `405.28 / 595.91` = `68.0%` of factor Frobenius energy, `preservesHexagonFfnChains: false`), evaluated with a temporary isolated `prism-inference` probe (`probe/anylinear-multi-rank`, `d8317103`, now deleted).
  - Unclipped `scale-screen` (`885.0 s / 900 s` AC `99% -> 99%`, `bonsai-2.5-pq2`, `reasoningEffort: "medium"`, `reasoningBudget: null`):
    - `diag-sw-logic-001` (`d74d07e5...`): `scale = 0.0` produced `717` think / `1,046` comp / `176.0 s` (`168.25 ms/tok`, `markers=0`); `scale = 1.0` increased thinking to `763` (`+6.4%`) / `1,130` comp (`+8.0%`) / `215.2 s` (`190.47 ms/tok`, `+13.2%`, `verify=3`); `scale = 1.5` increased thinking to `781` (`+8.9%`) / `1,092` comp (`+4.4%`) / `214.7 s` (`196.65 ms/tok`, `+16.9%`, `verify=1`), all exact.
    - `diag-sw-math-001` (`2fc56c28...`): `scale = 0.0` produced `320` think / `567` comp / `104.8 s` (`184.88 ms/tok`, `doubleCheck=1`) vs `scale = 1.0` producing `210` think (`-34.4%`) / `416` comp / `84.2 s` (`202.34 ms/tok`, `+9.4%`, `markers=0`), both exact; `scale = 1.5` skipped at budget.
- **Power, limits & cleanup:** All trials ran on Snapdragon X2 CPU/NEON (`--row-workers 12`, `rsq4`, `contextTokens: 4864`) with matched AC endpoint snapshots via `GetSystemPowerStatus` (`ac_online: true`, `battery_saver: false`; continuous mid-run power stability unproven), `sleepInhibitionActive: true`, `lingeringServerCount: 0`, and `credDirExistsAfter: false`. Because even full-linear `AnyLinear` increased multi-step logic thinking tokens (`+6.4%` to `+8.9%`) and decode latency (`+9.4%` to `+16.9%`), broke Hexagon NPU fused FFN chains, and added `Swift-Open-License-1.0` licensing overhead, Swifting Bonsai 2.5 via LoRA diff closed no-go and all Swifting code, probe worktree/branch/builds, and staged artifacts were stripped.

## Remaining qualification limits

- Delegated blinded `DR-G2` review on the complete `4/4` merged packet (`6dc82c06...`) closed **No-Go (`drG2Passed: false`, `9d8e9a71...`)** for immediate default flip (`3/4` gates passed with `+0.8333` Thinking-ON gain, failed Gate 3 `3.2 < 3.5` on `n = 1` Thinking-OFF `creative-b25-001`). Disjoint multi-prompt Thinking-OFF follow-up (or `DR-W1` refinement), independent human review, and immutable remote hosting (`huggingface`/`github-release`) for `daisyrose-npu-r16-drw1.gguf` remain open before `config/daisyrose-lora.lock.json` can be pinned.
- Standalone `Ternary-Bonsai-2.5-27B-PQ2_0` (without DaisyRose) scored `1.5/4.0` on `coding-b25-001` and `reasoning-b25-001` (`2.3333` Thinking-ON mean under `reasoningBudget: 448`) and still lacks a head-to-head comparison against `Swift-Bonsai-2-PQ2_0`.
- Snapdragon X1 testing, physical X1/X2 Hexagon NPU / Adreno Vulkan / Composite device-resident multi-rank LoRA trials, owning CI (`source-checks`, `foundation`), and `*-mtp` (`nextn_predict_layers = 1`, planned for `v0.1.36+`) remain open.
- `Swift-Bonsai-2-PQ2_0` + Orca remains the default profile and `config/daisyrose-lora.lock.json` remains unpinned (`Ok(None)`).
