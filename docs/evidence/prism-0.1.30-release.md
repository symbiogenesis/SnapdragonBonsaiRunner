# Prism 0.1.30 release delivery

Runner's [runtime lock](../../config/prism-inference.lock.json) selects the
[0.1.30 prerelease](https://github.com/symbiogenesis/prism-inference/releases/tag/v0.1.30).
This record establishes source-bound binary delivery and integrity, not full
Foundation, actual-model, quality, resource, isolation or performance acceptance.

| Identity | Value |
| --- | --- |
| Source/tag commit | `89df9aaff9e580d66679f2befe94d471bcf4fc7f` |
| Target/compiler | `aarch64-pc-windows-msvc`, `nightly-2026-11-09-aarch64-pc-windows-msvc` |
| Build | [Foundation bundle 43745027467, attempt 1](https://github.com/symbiogenesis/prism-inference/actions/runs/43745027467) |
| Operation | `release-0-1-30` |
| Runtime ZIP | 6,880,056 bytes; SHA-256 `7f2b09b6872530279d7c5607945fc5c36e118b83e496832984fef37c06df6747` |
| Exact source ZIP | 5,380,762 bytes; SHA-256 `83e4077e4abb058dd103ada4e3fdd3812debf7a49aa9ccf745f0cff61acc11a6` |
| Original bundle receipt | 1,979 bytes; SHA-256 `8b7d290e7a1cf23be9cc3661d290262a83f2906f0dc9955ceaa531627d7f86e5` |
| Local integrity verifier | `bonsai-tools.exe`, SHA-256 `5efd77dc9a90b9435ee415295e4b904e4c68f037d0fd9183f7432eb1ea76bd2a` |

The dedicated hosted lane completed successfully: final source checks,
nine receipt/version controls, five PE-import controls, two static-runtime
unwind/drain controls, six scoped-context controls and production linking. It
confirmed external VC-runtime imports were absent. It explicitly left full
Foundation, model acceptance and performance unqualified. The run used GitHub's
hosted execution cap; no local operator or compiler timeout was imposed and no
build was cancelled for elapsed wall time.

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

Prism `0.1.30` delivers transient-checkpoint separation, GDN precision and state
guards, `SwiftExecutionMemoryLedger`, cooperative worker cancellation, bounded
attention score scratch growth (`508edc3a`), independent Rust replacements for
retired native providers, and Qwen3-VL `mmproj` vision projector loading
(`--mmproj`, `--image-max-tokens`, and readiness `mmproj_sha256` and
`image_max_tokens` reporting). Runner binds the `0.1.30` capability tuples and
enables `VISION_RELEASE` for
`("0.1.30", "89df9aaff9e580d66679f2befe94d471bcf4fc7f")`. K4/V3 and fitted-value
precision remain outside Runner admission.

Final-source Runner compiler/analyzer and named regression acceptance, actual
long-prefill/Stop/Retry, full-state/logits/adapters, signed-device placement,
resources, multimodal model evaluation, independent creative/speech review and
clean-machine acceptance remain pending in [PLAN.md](../../PLAN.md). Prior
`0.1.28` receipts retain their original identities and verdicts. For a source
rollback to `0.1.28`, recover its complete lock from Runner commit `33e1eaff`,
restore its matching capability guards and rebuild all three lock-embedding
consumers; changing a version string alone is insufficient.

## Registry delivery

The published registry packages below use clean Prism commit
`89df9aaff9e580d66679f2befe94d471bcf4fc7f` (with `bonsai-native-hash` and
`bonsai-ternary` remaining at `0.1.3` because neither crate had source changes
since `0.1.28`). The private server and Hexagon packages remain
`publish = false`.

| Package | Archive bytes | Registry SHA-256 |
| --- | ---: | --- |
| [bonsai-native-hash 0.1.3](https://crates.io/crates/bonsai-native-hash/0.1.3) | 7,132 | `315284b1f3fc7e98f2cb0f749e63690853f1c441d2b3a3eec2ad68abe45180a2` |
| [bonsai-ternary 0.1.3](https://crates.io/crates/bonsai-ternary/0.1.3) | 43,593 | `f6ea8b95000a131bb48e27453b776e9e42096194648a3a8ebde782d5cfe9f604` |
| [bonsai-neon-dot 0.1.5](https://crates.io/crates/bonsai-neon-dot/0.1.5) | 45,381 | `eb88f7086bc2946a4a8059643105cdc92449d004fbb55892cbb3999e090843d5` |
| [bonsai-gguf 0.1.6](https://crates.io/crates/bonsai-gguf/0.1.6) | 37,397 | `602f5fa19943d86a33dee8f7a32a9f98abd79021dcc8841cca46505f5ef4f5fe` |
| [bonsai-prism 0.1.6](https://crates.io/crates/bonsai-prism/0.1.6) | 211,216 | `1854e80d14997431f9c850b10c9c4e40af22b555d8217aa14547c08926b0009e` |
| [prism-inference 0.1.30](https://crates.io/crates/prism-inference/0.1.30) | 4,086,098 | `b4bc8aec98c8f39982706bdafdd5d8d18f563751f45eb27b779a4eefaa975094` |

The four changed publishable crates (`bonsai-neon-dot 0.1.5`,
`bonsai-gguf 0.1.6`, `bonsai-prism 0.1.6` and `prism-inference 0.1.30`) were
published in dependency order and verified by downloading all six `.crate`
archives from `static.crates.io` and checking exact byte sizes and SHA-256
hashes.

This dedicated publication operation declared a five-minute allowance and
finished in 39.94 seconds (`00:00:39.9438182`). Aggregate Windows power
endpoints before and after reported AC connected (`AC=Online`, flags `Charging`),
charge 45% to 46% and battery saver `unknown`; these endpoint snapshots do not
prove uninterrupted stability or establish performance. Temporary scratch files
were removed after verification. No tests, full suite, model, benchmark,
fallback-target or performance/device qualification ran in this operation.
