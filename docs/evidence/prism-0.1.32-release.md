# Prism 0.1.32 release delivery

Runner's [runtime lock](../../config/prism-inference.lock.json) selects the
[0.1.32 prerelease](https://github.com/symbiogenesis/prism-inference/releases/tag/v0.1.32).
This record establishes source-bound binary delivery, integrity and bounded X2
brokered launch checks, not full Foundation CI, X1 hardware, multi-context
M1–M3, quality or sustained performance acceptance.

| Identity | Value |
| --- | --- |
| Source/tag commit | `23e24e444a821d4426f65d56c6cf874cf568b0d9` |
| Target/compiler | `aarch64-pc-windows-msvc`, `nightly-2026-11-09-aarch64-pc-windows-msvc` |
| Build | [Foundation bundle 43976153537, attempt 1](https://github.com/symbiogenesis/prism-inference/actions/runs/43976153537) |
| Operation | `release-0-1-32` |
| Runtime ZIP | 6,907,962 bytes; SHA-256 `aad9879b647f03f60a784f55bf7781d49d80b6f351de6f1030b1a443063c1b6a` |
| Exact source ZIP | 5,568,843 bytes; SHA-256 `6d6e318081a73bfac0e545666ded65b8a29f57727c5a478a54dff6ae69e3743c` |
| Original bundle receipt | 1,979 bytes; SHA-256 `e3a39da5cb165a5fb378cb8e2d79daaeb7f312f8e294637b2375f70a5e4248d8` |
| Local integrity verifier | `bonsai-tools.exe`, SHA-256 `dae840a9bc17f8ce7085cffc9a5c2849be81c627a2554a8ed42cf54ba0901b61` |

On `23e24e444a821d4426f65d56c6cf874cf568b0d9`, all three hosted
`symbiogenesis/prism-inference` workflows succeeded:
`Foundation bundle` (`43976153537`), `Source checks` (`43976102733`), and
`Foundation` (`43976102730`), using GitHub's hosted execution cap without local
timeout or cancellation.

The native `inference-lock` verifier checked the selected run, revision,
attempt, receipt, archive identity, installed executable and license entries,
lengths, full SHA-256 digests and ARM64 headers. The tag was independently
checked against the selected commit, and the published ZIP, receipt and exact
source ZIP were downloaded from GitHub and verified byte-for-byte by SHA-256.

The runtime and original receipt remain release assets; the exact source URL and
every installed-file identity are in the lock. Those selected bytes plus the
receipt are the replay inputs for the
[lock-update procedure](../development-guide.md#prepare-a-runtime-lock-update).
Successful build chatter and duplicate ZIP/source copies are not repository
evidence. Preserve any required independent replay inputs under
[retention policy](../validation.md#evidence-retention).

Prism `0.1.32` incorporates `0.1.31` plus `23e24e444a821d4426f65d56c6cf874cf568b0d9`
runtime additions (`--broker-pipe`, `KIND_VERIFIED_SECTION` 52-byte LE intake
via `--verified-sections`, `--admission-policy calibrated-cow`,
`--prompt-reuse prompt-end-and-latest`, `--prompt-reuse-checkpoints 1..=8`,
`--prompt-reuse-checkpoint-bytes`, `--cpu-prefill-gdn sequential|wy-f32`,
`--spec-type none|suffix`, `--draft-max 1..=8`, `--cpu-family x1|x2`,
mixed-phase Hexagon handoff, `[token][head]` attention probabilities, Hexagon
`ffn_down` LoRA side-branch fusion, `"SB2HQ2C1"` `NativeQ2ArenaCache`,
`Qwen35::release_executor_resident_host_weights`, and
`GrammarConstrainedSampler`). Runner binds the `0.1.32` capability tuples
(`CPU_PLACEMENT_RELEASE`, `MEMORY_CONTEXT_RELEASE`, `K4V4_FULL_CONTEXT_RELEASE`,
`RESET_ON_CANCEL_RELEASE`, `VISION_RELEASE`, `CPU_PREFILL_RELEASE`,
`BROKER_PIPE_RELEASE`, `CALIBRATED_COW_ADMISSION_RELEASE`,
`BRANCH_PROMPT_REUSE_RELEASE`, `CPU_PREFILL_GDN_RELEASE`, and
`CPU_SPECULATION_RELEASE`) to `("0.1.32", "23e24e444a821d4426f65d56c6cf874cf568b0d9")`,
while leaving `K4V3_CONTEXT_RELEASE` disabled (`None`). Runner also wires
end-to-end `KIND_VERIFIED_SECTION` section inheritance, Hexagon runtime
AppContainer read-only grants, default `--admission-policy calibrated-cow` with
split Job private-commit vs working-set limits, live UI footprint diagnostics,
and full CLI/launcher exposure for the `0.1.32` session and hardware options.

## Registry delivery

`prism-inference 0.1.32` was packaged and published to `crates.io` from commit
`23e24e444a821d4426f65d56c6cf874cf568b0d9` while retaining unchanged helper
crate pins from [the 0.1.31 record](prism-0.1.31-release.md#registry-delivery):

| Package | Archive bytes | Registry SHA-256 |
| --- | ---: | --- |
| [prism-inference 0.1.32](https://crates.io/crates/prism-inference/0.1.32) | 4,207,366 | `5d56008b6442cabcb00447df98701d09f2f1b50f5db232edb8efb437b21153ec` |

Verified by downloading `prism-inference-0.1.32.crate` from `static.crates.io`
and checking exact byte length and SHA-256 digest before removing the scratch
copy.

## Bounded X2 owner-device qualification (`X2E94100`)

Provisioned the verified `0.1.32` runtime into `%LOCALAPPDATA%\SwiftBonsai2` via
`bonsai-tools.exe provision` (`operation: "inference"`) against pre-built
release binaries (`bonsai-broker.exe` and `bonsai-tools.exe`, no compilation inside
trial budgets) and existing local `Swift-Bonsai-2-PQ2_0.gguf` / adapter / `mmproj`
artifacts. Both runs recorded `GetSystemPowerStatus` before Arm 1, between arms,
and after teardown (`ACLineStatus=1`, Battery 98%, BatterySaver=0 throughout):

1. **Initial startup/containment run (24.38 s of predeclared 480.0 s AC allowance):**
   Both Arm 1 (`--require-native-sandbox --require-broker-pipe --reset-on-cancel`,
   ready in 10.99 s, `Context tokens: 262144`, `Input modalities: text, image`,
   `WorkingSet=7,686.9 MiB`, `PrivateCommit=216.8 MiB`) and Arm 2
   (`--cpu-prefill-tokens 16 --spec-type suffix --draft-max 4`, ready in 10.86 s,
   `Context tokens: 262144`, `Input modalities: text` confirming automatic
   `mmproj` suppression, `WorkingSet=7,094.1 MiB`, `PrivateCommit=234.3 MiB`)
   started inside AppContainer with `frontend.key` present and `backend.key`
   absent on disk, and tore down with `BrokerExitCode=0`, `CleanupConfirmed=true`,
   and `0` lingering `prism-inference-server` processes. Completion HTTP
   requests in this run rejected `400 Bad Request` (`invalid_chat_request`)
   because the test client sent `"model": "Swift-Bonsai-2-PQ2_0"` instead of the
   broker's fixed public `MODEL_ALIAS` (`"bonsai"`).
2. **Fixed-alias (`"model": "bonsai"`) inference, reset and speculation run (34.92 s of predeclared 180.0 s AC allowance):**
   - **Arm 1 (`--require-native-sandbox --require-broker-pipe --reset-on-cancel --use-abliterate-lora`):**
     Ready in 11.06 s (`Context tokens: 262144`, `Input modalities: text, image`,
     `frontend.key` present, `backend.key` absent; idle `WorkingSet=7,707.4 MiB`,
     `PrivateCommit=237.3 MiB`). Non-streaming completion returned `200 OK` in
     2.91 s (`"X2_OK"`, 3 completion tokens). An active streaming request was
     aborted client-side after 2 SSE chunks (`"1"`), triggering in-place
     `--reset-on-cancel` `KIND_RESET`, and a follow-up completion on the same
     contained worker succeeded with `200 OK` in 2.65 s (`"RESET_OK"`,
     `WorkingSet=7,840.4 MiB`, `PrivateCommit=369.0 MiB`). Clean exit (`code 0`,
     `CleanupConfirmed=true`, `0` lingering processes).
   - **Arm 2 (`--require-native-sandbox --require-broker-pipe --cpu-prefill-tokens 16 --spec-type suffix --draft-max 4 --use-abliterate-lora`):**
     Ready in 10.19 s (`Context tokens: 262144`, `Input modalities: text`,
     `frontend.key` present, `backend.key` absent; idle `WorkingSet=7,094.1 MiB`,
     `PrivateCommit=234.4 MiB`), completed short request `200 OK` in 2.12 s
     (`"X2_OK"`, post-run `WorkingSet=7,386.5 MiB`, `PrivateCommit=521.1 MiB`),
     and exited cleanly (`code 0`, `CleanupConfirmed=true`, `0` lingering
     processes).

Physical Snapdragon X1 hardware runs, multi-context M1–M3 Job-commit
calibration sweeps, sustained throughput/energy benchmarks, hosted Runner CI
acceptance, and independent human qualitative/listening reviews (`Q1`, `P5`,
speech) remain open in [PLAN.md](../../PLAN.md). Prior `0.1.31`, `0.1.30` and
`0.1.28` receipts retain their original identities and verdicts.
