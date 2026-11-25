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

## Remaining qualification limits

- Full `dr-g2-score` requires at least one Thinking-OFF (`creative-prose`) prompt pair (`readiness.rs` fails closed without it), so a separate AC-budgeted Thinking-OFF capture plus owner blinded review (`dr-g2-apply-reviews` -> `dr-g2-score`) remain open.
- Snapdragon X1 testing, physical X1/X2 Hexagon NPU / Adreno Vulkan / Composite device-resident multi-rank LoRA trials, and `*-mtp` (`nextn_predict_layers = 1`, planned for `v0.1.36+`) remain open.
- `Swift-Bonsai-2-PQ2_0` + Orca remains the default profile until blinded `DR-G2` review and full acceptance qualification close.
