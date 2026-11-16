# Prism 0.1.31 release delivery

Runner's [runtime lock](../../config/prism-inference.lock.json) selects the
[0.1.31 prerelease](https://github.com/symbiogenesis/prism-inference/releases/tag/v0.1.31).
This record establishes source-bound binary delivery and integrity, not full
Foundation, actual-model, quality, resource, isolation or performance acceptance.

| Identity | Value |
| --- | --- |
| Source/tag commit | `577966d98cf8b36021df52da194b1a5540429a6b` |
| Target/compiler | `aarch64-pc-windows-msvc`, `nightly-2026-11-09-aarch64-pc-windows-msvc` |
| Build | [Foundation bundle 43900505724, attempt 1](https://github.com/symbiogenesis/prism-inference/actions/runs/43900505724) |
| Operation | `release-0-1-31` |
| Runtime ZIP | 6,834,124 bytes; SHA-256 `ef3e950d0922d1805ebfd19eec0e332c22840fd4126e54e8b4313e88aa9d9f26` |
| Exact source ZIP | 5,518,785 bytes; SHA-256 `21145070c8375e72b93d72acb3c2ae0f948bb2f292203b2c227a8a3b27c930e0` |
| Original bundle receipt | 1,979 bytes; SHA-256 `eb899501f16370dea15f15ec2cb05f40391f5b07631c2c9b96349cda45c7a427` |
| Local integrity verifier | `bonsai-tools.exe`, SHA-256 `96635946dddb1a7524584ffef297337299311e3ab85f862c14ce4fcfdb240e01` |

The dedicated `Foundation bundle` lane completed successfully (`43900505724`):
final source checks, parallel-rows Clippy, receipt/version controls, PE-import
controls, static-runtime unwind/drain controls, scoped-context controls and
production linking. It confirmed external VC-runtime imports were absent and used
GitHub's hosted execution cap without local timeout or cancellation. On the same
commit `577966d98cf8b36021df52da194b1a5540429a6b`, `Source checks`
(`43900458598`) succeeded, while push `Foundation` (`43900458316`) passed
`source-package` and `cross-native-arm64` and failed `correctness-3` on
`tests/runtime.rs::oversized_optional_checkpoint_preserves_an_older_prefix`.
Complete Foundation suite acceptance remains pending.

The native `inference-lock` verifier checked the selected run, revision,
attempt, receipt, archive identity, installed executable and license entries,
lengths, full SHA-256 digests and ARM64 headers. The tag was independently
checked against the selected commit, and the published ZIP, receipt and exact
source ZIP were downloaded from GitHub and compared byte-for-byte by SHA-256.
No runtime executable or model was run during this local artifact verification.

The runtime and original receipt remain release assets; the exact source URL and
every installed-file identity are in the lock. Those selected bytes plus the
receipt are the replay inputs for the
[lock-update procedure](../development-guide.md#prepare-a-runtime-lock-update).
Successful build chatter and duplicate ZIP/source copies are not repository
evidence. Preserve any required independent replay inputs under
[retention policy](../validation.md#evidence-retention).

Prism `0.1.31` delivers zero-copy `MappedSlice` GGUF mapping and 8-worker
payload validation, the zero-allocation spin-then-park `NativeRowPool`,
bit-exact `G in {2, 3, 6}` grouped GQA packed-KV decode/tile attention,
per-layer `adapter_affects_any` `external_ffn_chain`/`external_ffn_swiglu` LoRA
unlocking, `bonsai-gguf` bounded cancellable `Read + Seek` stream index parsing,
shared `bonsai-neon-dot` `kquants` SIMD kernels, `bonsai-ternary`
`unpack_q4_k_scale_min`, `hex` crate consolidation, and repinned
`bonsai-windows-credentials 0.1.5` / `bonsai-windows-process 0.1.3`. Runner
binds the `0.1.31` capability tuples (`CPU_PLACEMENT_RELEASE`,
`MEMORY_CONTEXT_RELEASE`, `K4V4_FULL_CONTEXT_RELEASE`, `RESET_ON_CANCEL_RELEASE`,
`VISION_RELEASE`, `CPU_PREFILL_RELEASE`) to
`("0.1.31", "577966d98cf8b36021df52da194b1a5540429a6b")`, leaving
`K4V3_CONTEXT_RELEASE` and `BROKER_PIPE_RELEASE` disabled (`None`).

Final-source Runner compiler/analyzer and named regression acceptance, actual
long-prefill/Stop/Retry, full-state/logits/adapters, signed-device placement,
resources, multimodal model evaluation, independent creative/speech review and
clean-machine acceptance remain pending in [PLAN.md](../../PLAN.md). Prior
`0.1.30` and `0.1.28` receipts retain their original identities and verdicts. For
a rollback to `0.1.30` or `0.1.28`, recover the matching lock, capability
guards and lock-embedding consumers; changing a version string alone is
insufficient.

## Registry delivery

The published crates below correspond to `prism-inference` helper commit
`1a9ff0a` (`bonsai-ternary 0.1.4`, `bonsai-native-hash 0.1.4`, `bonsai-prism 0.1.7`,
`bonsai-neon-dot 0.1.6`, `bonsai-gguf 0.1.7`), `SwiftBonsai2Runner` commit
`d2e98a00` (`bonsai-windows-credentials 0.1.5`, `bonsai-windows-process 0.1.3`),
and `prism-inference` release commit `577966d98cf8b36021df52da194b1a5540429a6b`
(`prism-inference 0.1.31`). Private `prism-inference-server` and Hexagon
packages remain `publish = false`.

| Package | Archive bytes | Registry SHA-256 |
| --- | ---: | --- |
| [bonsai-ternary 0.1.4](https://crates.io/crates/bonsai-ternary/0.1.4) | 44,521 | `160abdf86b4de49b1707c464f24ba352c23ffaf755300709cb71c840a9428bba` |
| [bonsai-native-hash 0.1.4](https://crates.io/crates/bonsai-native-hash/0.1.4) | 16,310 | `d8d2e05287d77ea0fe409474ac76dc9fe02783b32786ec77ac7467b75d6c3ed2` |
| [bonsai-prism 0.1.7](https://crates.io/crates/bonsai-prism/0.1.7) | 200,040 | `9355df807b7cdad514b8492e8106a6a10809924347f60a7166b210fff694d8ef` |
| [bonsai-neon-dot 0.1.6](https://crates.io/crates/bonsai-neon-dot/0.1.6) | 64,940 | `8e0a0891e6efade20e21ab528058aa9ebf851699dbc37f2a735f66e7efe3f644` |
| [bonsai-gguf 0.1.7](https://crates.io/crates/bonsai-gguf/0.1.7) | 47,330 | `111c6148f4ef061e33fc30e4b4dae4fef60ba53586bca61ad5c775fb77311a7c` |
| [bonsai-windows-credentials 0.1.5](https://crates.io/crates/bonsai-windows-credentials/0.1.5) | 26,768 | `94da447b6f49bde8fcbcb4918c8a894f7f9c7d4be7d42e433a5c30a0ead4e2ff` |
| [bonsai-windows-process 0.1.3](https://crates.io/crates/bonsai-windows-process/0.1.3) | 144,168 | `1b3f69dfd8ff188ae1559467795523ebbbe83f3fef984c5d29515195c21aa4b4` |
| [prism-inference 0.1.31](https://crates.io/crates/prism-inference/0.1.31) | 4,176,998 | `bda5316c898fd1d32821b277193e51cb164776f187d1b9ca02e83ae8b8669da2` |

All eight `.crate` archives were published in dependency order and verified by
downloading from `static.crates.io` and checking exact byte sizes and SHA-256
digests. Scratch downloads were removed after verification. Local publication
and release-lock verification ran under a predeclared fifteen-minute device
allowance on continuous AC power (`AC=Online`). Full Foundation suite runs,
actual-model trials and X1/X2 device benchmarks remain separate gates.
