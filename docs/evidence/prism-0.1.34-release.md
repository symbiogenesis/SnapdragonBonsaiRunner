# Prism 0.1.34 release delivery

Runner's [runtime lock](../../config/prism-inference.lock.json) selects the
[0.1.34 prerelease](https://github.com/symbiogenesis/prism-inference/releases/tag/v0.1.34).
This record establishes source-bound binary delivery, integrity verification
and registry crate delivery, not full Prism `Foundation` CI, X1/X2 device
runs, multi-context M1–M3, quality or sustained performance acceptance.

| Identity | Value |
| --- | --- |
| Source/tag commit | `c64c9648f85ac4c99ff8d57f709871b31e785370` |
| Target/compiler | `aarch64-pc-windows-msvc`, `nightly-2026-11-09-aarch64-pc-windows-msvc` |
| Build | [Foundation bundle 45113319918, attempt 1](https://github.com/symbiogenesis/prism-inference/actions/runs/45113319918) |
| Operation | `release-0-1-34` |
| Runtime ZIP | 8,750,549 bytes; SHA-256 `5085478dad9fe2eff405fbbb810d6ccb3986e32414493404401d8fee4fab6d7b` |
| Exact source ZIP | 6,292,053 bytes; SHA-256 `20147e72c4458bb9d3269311be50872e9ce0bbdc64e3bd078c41760c0e859970` |
| Original bundle receipt | 1,979 bytes; SHA-256 `4824cff7f1a25a7f54e57ec8b6a9fdee69d733149426d18336818607d6ad839d` |
| Local integrity verifier | `bonsai-tools.exe`, SHA-256 `82cebb83387b23032c118801d5ece620089c22bd9b04a16b4b6d876115467b51` |

On `c64c9648f85ac4c99ff8d57f709871b31e785370`, `Foundation bundle` (`45113319918`,
attempt 1) succeeded (`ReceiptControlsPassed=9`, `PeImportControlsPassed=5`,
`StaticRuntimeUnwindControlsPassed=2`, `ScopedContextControlsPassed=6`,
`FinalSourceChecksPassed=true`, `ProductionLinkingPassed=true`,
`ExternalVcRuntimeImportsAbsent=true`, `FullFoundationQualified=false`) and
`Source checks` (`45111096691`) succeeded, using GitHub's hosted execution cap
without local timeout or cancellation. Full `Foundation` (`45111096706`) failed
on `bonsai-adreno-vulkan` private intra-doc links (`MAX_STAGE_BINDINGS`),
`x86_64` unused `PathBuf` import / unfulfilled lint expectation, and an
outdated assertion in `tests/graph/verification.rs:198`; full Prism Foundation
qualification therefore remains open.

The native `bonsai-tools.exe inference-lock --json` verifier checked the
selected run, revision, attempt, receipt, archive identity, all 9 installed
files and license entries, lengths, full SHA-256 digests and ARM64 headers
(`Verified: true`). Tag `v0.1.34` was independently checked against
`c64c9648f85ac4c99ff8d57f709871b31e785370`, and both published release assets
plus the exact source ZIP were downloaded from GitHub and verified byte-for-byte
by SHA-256.

The runtime and original receipt remain release assets; the exact source URL and
every installed-file identity are in the lock. Those selected bytes plus the
receipt are the replay inputs for the
[lock-update procedure](../development-guide.md#prepare-a-runtime-lock-update).
Successful build chatter and duplicate ZIP/source copies are not repository
evidence. Preserve any required independent replay inputs under
[retention policy](../validation.md#evidence-retention).

Prism `0.1.34` incorporates `0.1.32` plus post-`0.1.32` runtime additions
(`bonsai-adreno-vulkan` and `--placement vulkan` / `composite` readiness, opt-in
`--hexagon-arena-cache <dir>`, `wy-f32` rotated-KV support, multi-rank LoRA
`--adapter`, `PTQ1_0` on `supports_f32` executors, `#18`
`release_executor_resident_host_weights` forwarding, `#41` dynamic VTCM
placement, and NEON / `NativeRowPool` / `SDOT` / `I8MM` / `GQA` / `GDN` kernel
improvements). Runner binds all 15 capability tuples (`CPU_PLACEMENT_RELEASE`,
`MEMORY_CONTEXT_RELEASE`, `K4V4_FULL_CONTEXT_RELEASE`, `RESET_ON_CANCEL_RELEASE`,
`VISION_RELEASE`, `CPU_PREFILL_RELEASE`, `BROKER_PIPE_RELEASE`,
`CALIBRATED_COW_ADMISSION_RELEASE`, `BRANCH_PROMPT_REUSE_RELEASE`,
`CPU_PREFILL_GDN_RELEASE`, `CPU_SPECULATION_RELEASE`, `VULKAN_PLACEMENT_RELEASE`,
`HEXAGON_ARENA_CACHE_RELEASE`, `WY_F32_ROTATED_KV_RELEASE`, and
`MULTI_RANK_LORA_RELEASE`) to `("0.1.34", "c64c9648f85ac4c99ff8d57f709871b31e785370")`,
while keeping `K4V3_CONTEXT_RELEASE` disabled (`None`), and extends
`identity.rs` startup projection verification for `"adreno-vulkan"`
(`Placement::Vulkan`) and `"composite-hexagon-vulkan"` (`Placement::Composite`).

## Registry delivery

All eight public workspace crates in the `c64c9648f85ac4c99ff8d57f709871b31e785370`
release were packaged and published to `crates.io`:

| Package | Archive bytes | Registry SHA-256 |
| --- | ---: | --- |
| [bonsai-ternary 0.1.5](https://crates.io/crates/bonsai-ternary/0.1.5) | 44,514 | `f0934aa0d3e74deaf52b12154cfdcf6ee37332d4690b8bb5e17538962973e76b` |
| [bonsai-native-hash 0.1.6](https://crates.io/crates/bonsai-native-hash/0.1.6) | 18,347 | `935893c6124862f846bf1465e0d22bbfa249911e0c667b4595d2de29b5e1f086` |
| [bonsai-prism 0.1.8](https://crates.io/crates/bonsai-prism/0.1.8) | 121,015 | `070da4f88692eb6b2e46167817019864251ab4fbc8d68f6ca9a6d16d7e5e9584` |
| [bonsai-neon-dot 0.1.8](https://crates.io/crates/bonsai-neon-dot/0.1.8) | 97,307 | `e9ebcb94052b4446a8675e604590060ae7cbd106d88c44457a9d32f2d3e20738` |
| [bonsai-gguf 0.1.9](https://crates.io/crates/bonsai-gguf/0.1.9) | 47,503 | `67708c1fdc5bd3cb0c8e06b6d6f748920e04ad9baf218b4c6a527589fdd1aac6` |
| [bonsai-windows-credentials 0.1.7](https://crates.io/crates/bonsai-windows-credentials/0.1.7) | 30,986 | `909360a39cbb2fe33aaaca2a0c1cc660c93843bcfad3c9ac690f17d0f2075350` |
| [bonsai-windows-process 0.1.5](https://crates.io/crates/bonsai-windows-process/0.1.5) | 156,183 | `202abcc49060118f59102cfb427267786be19340515faa42dcf28c8f0756bb33` |
| [prism-inference 0.1.34](https://crates.io/crates/prism-inference/0.1.34) | 4,742,736 | `7ed0ee1b9674f76f90471cccedcdedc964679922656cf8bcf3ffdeb6e2d44a75` |

Verified by downloading each `.crate` archive from `static.crates.io` and
checking exact byte lengths and SHA-256 digests before removing the scratch
copies.

X1/X2 device trials on `0.1.34`, multi-context M1–M3 calibration, sustained
throughput/energy benchmarks, hosted Runner CI acceptance, full Prism
`Foundation` repair, and human qualitative/listening reviews (`Q1`, `P5`,
speech) remain open in [PLAN.md](../../PLAN.md). Prior
[0.1.32](prism-0.1.32-release.md), `0.1.31`, `0.1.30`, and `0.1.28` records
retain their original identities and verdicts.
