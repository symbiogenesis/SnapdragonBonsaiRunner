# Prism 0.1.35 delivery evidence

This note records the `prism-inference` `v0.1.35` prerelease built from fixed upstream CI blockers and pinned in `SwiftBonsai2Runner`.

## Summary

- **Released commit:** `3bc76f135bf23c538b3bb789aef382143ff36839` on `symbiogenesis/prism-inference` (`main`, tag `v0.1.35`), preceded by CI blocker fixes in `a5630e2` (`tools/adreno-vulkan/src/device.rs` `pub const MAX_STAGE_BINDINGS`, `server/src/native_backend/worker.rs` `#[cfg]` + `allow(dead_code, ...)`, and `tests/graph/verification.rs` `ReleasedWeightRefusingExecutor`).
- **Hosted release bundle:** `foundation-bundle.yml` run [`45546690940`](https://github.com/symbiogenesis/prism-inference/actions/runs/45546690940) (attempt `1`, job `138977912061`, `12m9s`, `success`). Complete `Foundation` and `Source checks` workflows remain owner-run and separate.
- **Lock verification (`bonsai-tools inference-lock`):** `Verified: true` against downloaded bundle artifact `prism-inference-0.1.35-aarch64-pc-windows-msvc.zip` and source archive `3bc76f135bf23c538b3bb789aef382143ff36839.zip`.
- **Independent release re-verification:** Re-downloading the published [`v0.1.35` ZIP](https://github.com/symbiogenesis/prism-inference/releases/download/v0.1.35/prism-inference-0.1.35-aarch64-pc-windows-msvc.zip) from the GitHub prerelease matched byte length `9,030,681` and SHA-256 `8f0279a02fe34e4a46f3a22414778a351a33a8652e9f3987ff26ba75a0ffb52b`.
- **Local runtime provisioning (`bonsai-tools provision`, `operation: "inference"`):** Installed and verified into `%LOCALAPPDATA%\SwiftBonsai2\inference\3bc76f135bf23c538b3bb789aef382143ff36839\aarch64-pc-windows-msvc\` (`Verified: true`).

## Locked identities ([prism-inference.lock.json](../../config/prism-inference.lock.json))

- **Source ZIP:** `6,358,775` bytes, SHA-256 `37d90309db3ead3b3339a766916bfaecb032badafbde6abbd22d6cd152520801` (mirrored in [pins.json](../../tools/native-reference/pins.json)).
- **Release bundle ZIP:** `prism-inference-0.1.35-aarch64-pc-windows-msvc.zip`, `9,030,681` bytes, SHA-256 `8f0279a02fe34e4a46f3a22414778a351a33a8652e9f3987ff26ba75a0ffb52b`.
- **Installed ARM64 Windows executables & license files:**
  - `prism-inference-server.exe`: `6,860,800` bytes, SHA-256 `8cf39507c80e5f75d100723940db5f2b4345b3cb34835e71a07e2ed858a3ea83`
  - `prism-inference.exe`: `3,374,080` bytes, SHA-256 `7791981836ede97c84ad67e53ef19b262b577bbeeafbb1053326410e90df9cc9`
  - `bonsai-swift-worker.exe`: `4,360,192` bytes, SHA-256 `d725d403902bf60b28396317c0173bb031550b556b5ecdd3c851e31e9bd8ee98`
  - `bonsai-token-worker.exe`: `692,224` bytes, SHA-256 `7fa85552b845df6295044790d2714b271fbf5331b8f4eeb8dc2be8c3c548cbf5`
  - `bonsai-gguf.exe`: `550,400` bytes, SHA-256 `c5c568fed1806c206c1345764dcdda4530d3a3082a1e66cde4ecf2b2dc416ba3`
  - `LICENSE`: `11,560` bytes (`3ddf9be5...`), `NOTICE`: `2,369` bytes (`e3b6278b...`), `PRISM-LICENSE.txt`: `1,099` bytes (`bcd8ec74...`), `CAMELID-LICENSE.txt`: `1,087` bytes (`cfd0bfa3...`).

## Broker gate state ([inference.rs](../../src/broker/src/inference.rs))

- Promoted `REVIEWED_0_1_35 = Some(("0.1.35", "3bc76f135bf23c538b3bb789aef382143ff36839"))` across all existing `0.1.34` gates plus `RSQ4KVF_RELEASE`, `NARIQ_TRELLIS_RELEASE`, and `BONSAI_2_5_RELEASE`.
- Kept `K4V3_CONTEXT_RELEASE = None` pending dedicated K4/V3 capacity/quality qualification.
- Kept default model selection at `Swift-Bonsai-2-PQ2_0` + Orca pending owner blinded `DR-G2` review and full acceptance qualification. Reserved `*-mtp` (`866`-tensor `nextn_predict_layers = 1`) Bonsai 2.5 checkpoints remain deferred to `v0.1.36+`.
