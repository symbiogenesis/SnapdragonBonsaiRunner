# Licensing and dependency terms

Original Swift Bonsai 2 - Runner code, documentation and configuration use
[Apache-2.0](../LICENSE), copyright 2026 Edward Miller and contributors.
Third-party material retains its own terms. [NOTICE](../NOTICE) preserves
Prism/ggml attribution and the MIT permission notice for inherited portions.

## Project license

Apache-2.0 permits commercial use, modification and redistribution, with an
express contributor patent grant and patent-litigation termination. That grant
covers qualifying contributor claims, not unrelated third-party patents.
See [license sections 2, 3 and 5](https://www.apache.org/licenses/LICENSE-2.0).
No current dependency requires the original application to adopt GPL, LGPL
or MPL; permissive dependencies remain subject to their own notices.

## Reviewed dependencies

The [repository research catalog](research-register.md#research-catalog) lists
unadopted sources separately from the dependencies below. Before any reuse, pin
the exact revision and review the selected files, test assets and transitive
dependencies for permission and required notices. A repository-level license
label alone is insufficient. In particular, ciot has no verified reuse permission
from the survey, and SillyTavern is a design reference rather than approved code
for the Apache-2.0 application. Cataloging a source adds no dependency or NOTICE
entry; actual incorporation requires its own review.

The review covers workspace manifests/lockfile, resolved metadata and cached
licenses, Prism notices and pinned artifacts. Dependency terms apply independently.

| Material | Declared terms | Distribution requirements |
| --- | --- | --- |
| `http`, `httparse`, `serde`, `serde_json`, `sha2`, `windows-sys`, `tempfile` | MIT OR Apache-2.0 | Retain the applicable license and attribution. |
| `tokio` 1.54.1 | MIT | Retain copyright and permission notice. |
| `mxc-sdk` 1.0.1 | MIT | Retain Microsoft copyright and MIT permission notice; the native PSEC integration is confined to the Windows process boundary. |
| `flatbuffers` 25.12.19 | Apache-2.0 | Retain the selected license and notices for the fixed PSEC policy encoder. |
| `fearless_simd` 1.2.0 (Runner), 1.1.0 (external runtime) | Apache-2.0 OR MIT | Retain the applicable license/attribution for each independently pinned selection. |
| `rayon` 1.12.0, `rayon-core` 1.13.0 and Crossbeam worker queues | MIT OR Apache-2.0 | Retain the selected licenses/attribution for the Rust ASR worker pool and its queue dependencies. |
| `subtle` 2.6.1 | BSD-3-Clause | Retain copyright, conditions and disclaimer; respect non-endorsement. |
| `zip` 9.5.1 | MIT | Retain the notice; enabled deflate dependencies also have permissive options. |
| `zlib-rs` | Zlib | Retain source notices and identify altered versions. |
| `unicode-ident` | (MIT OR Apache-2.0) AND Unicode-3.0 | Include Unicode-3.0 notices as well as the selected code license. |
| `r-efi` | MIT OR Apache-2.0 OR LGPL-2.1-or-later | A permissive alternative is available; LGPL is not required. |
| `eframe`, `egui`, `epaint`, `egui-wgpu`, `egui-winit`, `egui_system_font_provider` 0.37.0 | MIT OR Apache-2.0 | Retain the selected code license and attribution; embedded font terms are separate. |
| `wgpu` and its core/HAL/type crates 30.0.1 | MIT OR Apache-2.0 | Retain the selected license and attribution. |
| `winit` 0.30.13, `accesskit_winit` 0.34.2 | Apache-2.0 | Retain the license and applicable notices; identify modified files. |
| `accesskit` 0.25.2, `accesskit_consumer` 0.39.2, `accesskit_windows` 0.35.2 | MIT OR Apache-2.0 | Retain the selected license and attribution. |
| `fontique` 0.11.1, `skrifa` 0.44.0, `read-fonts` 0.41.0, `font-types` 0.12.7 | MIT OR Apache-2.0 | Retain code licenses/notices independently of the fonts they load. |
| `harfrust` 0.12.0 | MIT | Retain the HarfBuzz developers' and Yevhenii Reizner's copyright and permission notice. |
| `arboard` 3.7.0 / `clipboard-win` 5.4.3 | MIT OR Apache-2.0 / BSL-1.0 | Retain applicable clipboard code licenses/notices. |
| `epaint_default_fonts` 0.37.0 | (MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0 | Preserve the individual font notices described below; the crate's code license does not replace them. |
| Prism/ggml codec, loader and native reference portions | MIT | Retain upstream MIT notices; Apache-2.0 does not replace them. |

All reviewed code packages declare licenses with permissive options; fonts have
separate terms.

### Native GUI fonts

The GUI enables `accesskit`, `clipboard`, `default_fonts`, `system_fonts` and
`wgpu_no_default_features`, with explicit wgpu DX12, Vulkan and GLES features;
inspection, persistence and browser-link features are disabled. The
downloaded 0.37.0 font crate embeds Hack Regular, Ubuntu Light and the small
`egui-icons.ttf` subset with this selection. Its optional monochrome Noto Emoji
and full emoji-icon fonts are not enabled. See the pinned
[font manifest](https://github.com/emilk/egui/blob/0.37.0/crates/epaint_default_fonts/Cargo.toml)
and [embedded assets](https://github.com/emilk/egui/blob/0.37.0/crates/epaint_default_fonts/src/lib.rs).

Binary distributions must retain accessible font copyright/license material
even though the fonts are compiled into the executable. Copy the applicable
files from the exact `epaint_default_fonts` package's `fonts/` directory:

- `Hack-Regular.txt`: Source Foundry MIT notice, DejaVu public-domain statement
  and Bitstream Vera license, including its font naming and standalone-sale
  restrictions.
- `UFL.txt`: Ubuntu Font Licence 1.0 for Ubuntu Light; also retain the font's
  metadata copyright, `Copyright 2011 Canonical Ltd.`, and follow its naming
  rules if modifying the font.
- `emoji-icon-font-mit-license.txt` and `egui-icons.txt`: John Slegers' MIT
  notice and the provenance of the always-embedded icon subset.
- `OFL.txt` and corresponding font copyrights if shipping the optional Noto
  Emoji asset, a source distribution containing it, or another OFL font. Keep
  OFL font software under OFL and respect reserved names and standalone-sale
  restrictions.

`system_fonts` uses `fontique` to locate and memory-map installed fonts; it does
not bundle them or grant redistribution rights. Review each font's terms before
shipping it. Font terms do not relicense the application code.

## External engines

The [native Rust reference](https://github.com/symbiogenesis/prism-inference/blob/main/README.md)
uses independently published GGUF/ternary/Prism crates. Its graph/scalar numerical
code follows pinned Prism sources; preserve
[PRISM-LICENSE.txt](https://github.com/symbiogenesis/prism-inference/blob/main/PRISM-LICENSE.txt)
and root NOTICE. [Cargo.lock](../Cargo.lock) owns Runner's `fearless_simd` pin and
checksum; the external runtime has an independent lock. See the
[dependency choices](dependency-audit.md#direct-dependency-decisions) for its
current consumers and the dependency table above for license options.

The reviewed OxiBonsai v0.2.3 source is revision
`1d10c48e14754ff6999158f8fa13c7c92b40b0dc`. Its selected GGUF/NEON sources
declare Apache-2.0, compatible with Runner's original license. Preserve COOLJAPAN
OU (Team Kitasan) attribution and identify modifications if importing source.
Oxi crates are not dependencies; their optional backends remain unapproved.

| Artifact | Pinned source | Terms |
| --- | --- | --- |
| Reference llama.cpp | `36e36e63a54fc8f85b95e9c499a022b597e478dd` | [MIT](https://github.com/ggml-org/llama.cpp/blob/36e36e63a54fc8f85b95e9c499a022b597e478dd/LICENSE) |
| Swift assessment Prism | `73bfe6cc485554822f80c473a3c0c4f25e80f41b` | [MIT](https://github.com/PrismML-Eng/llama.cpp/blob/73bfe6cc485554822f80c473a3c0c4f25e80f41b/LICENSE) |
| Prism overlay source | `b87ca7225456b06b0975b8239c66b5aecf07e37c` | [MIT](https://github.com/PrismML-Eng/llama.cpp/blob/b87ca7225456b06b0975b8239c66b5aecf07e37c/LICENSE) |

The review-only [headless argv patch](runtime-candidates/README.md#headless-ucrt-argv-candidate)
modifies that exact Swift assessment source under its MIT license. Its manifest
retains the source LICENSE digest; preserve the upstream copyright/license with
any source or binary integration. No UCRT implementation source is distributed.

ASR includes the MIT-licensed transcribe.cpp native core. Rust TTS derives from
reviewed Qwen/GGML source without linking those C++ engines. Retain source-specific
notices from the exact [speech pins](../config/native-speech.lock.json),
[ASR lock](../src/speech-asr/model.lock.json) and `src/speech-tts/LICENSE.*` files.
These code terms are separate from the speech model terms below.

Additional surveyed upstream speech and ARM64 CPU kernel implementations with
permissive Apache-2.0-compatible terms (requiring exact revision pinning, notice
retention, and change attribution on incorporation):

| Source | Surveyed revision | Declared terms | Notes |
| --- | --- | --- | --- |
| `ServeurpersoCom/qwentts.cpp` | `5916bf7c79dbb4e66fc059cdc507a79a23a28bb4`..`f1a6924789883bd92eeb59176793a94af55730b3` | MIT | TTS/codec/speaker C++ reference (`src/speech-tts/LICENSE.qwen-core`) |
| `ServeurpersoCom/ggml` | `cc5df49a6b878db2b68b9a7b45976e279594627d`..`0d403add5cb6329935290f02462e1072045aacd2` | MIT | ARM64 `SDOT`/`I8MM` quantized kernels (`LICENSE.ggml`) |
| `handy-computer/transcribe.cpp` | `4d7c1ccefd39e5a9ea36cca5f8fefc6dcc14b99b`..`adfc50210a3f026a655a3e19937d90ef2bf0728c` | MIT | Qwen3-ASR engine, Win32 mmap, `gate+up` alias (`src/speech-asr/LICENSE.transcribe`) |
| `QwenLM/Qwen3-TTS` | `022e286b98fbec7e1e916cb940cdf532cd9f488e` | Apache-2.0 | Upstream TTS architecture & tokenization reference |
| `QwenLM/Qwen3-ASR` | `7c6daf77a2421100f5fb066495372c00129d39ff` | Apache-2.0 | Upstream ASR encoder windowing & prompting reference |
| `gpillon/ignis` (`crates/speech`) | `19b23fc45cd5f765943e325ebc437a214c5b3c55` | Apache-2.0 | Pure-Rust Qwen3-ASR `SDOT`/`I8MM`, `BlockQ4Kx8`, spin-then-park pool, streaming mic (`vad.rs` adapted from `snakers4/silero-vad`, MIT) |
| `kyutai-labs/moshi` (`rust/moshi-core`) | `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` | Apache-2.0 | Pure-Rust streaming Mimi conv/ring-buffer codec |
| `ARM-software/kleidiai` | `0dfee30a1879035957cde46db7b95521722cc2e0` | Apache-2.0 | ARM64 microkernel packing & tile interfaces (`LICENSES/Apache-2.0.txt`) |
| `robertknight/rten` | `b9407f41a71f1930d4f5dde3f721e808a9f08369` | Apache-2.0 | Pure-Rust ARM64 `rten-gemm` packing/tiling reference |

Excluded speech repositories (do **not** copy source into Apache-2.0 deliverables):
`Dicklesworthstone/franken_tts` (`267d84add52ef1f48644f8f63119b261b58f311f` uses
"MIT License (with OpenAI/Anthropic Rider)", a non-OSI party restriction incompatible
with Apache-2.0 redistribution) and `second-state/qwen3_tts_rs` / `TrevorS/qwen3-tts-rs`
(no declared repository license).

Retain each exact distribution's bundled licenses/notices. A top-level MIT
license does not replace terms for additional included components.

## Models and adapters

Weights and adapters retain their publishers' terms.

| Artifact | Pinned revision | Published terms |
| --- | --- | --- |
| Swift Bonsai 2 GGUF | `a3bdac086bbb6b04d87d0108d18943b5f63868f7` | [Apache-2.0](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF/blob/a3bdac086bbb6b04d87d0108d18943b5f63868f7/LICENSE) |
| Qwen3.8-27B GGUF reference | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | [Apache-2.0 declaration](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/README.md) |
| Approved OrcaBonsai LoRA | `703028bb7fba03496e796b144839b161cf0f5409` | [Apache-2.0 declaration](https://github.com/Continuum-AI-Corp/OrcaBonsai-27B-Uncensored/blob/703028bb7fba03496e796b144839b161cf0f5409/README.md#license) |
| Qwen3.8-27B-NariQ-GGUF | `17886395feb8a752b5ca5f02ad0c25543950915e` | [Apache-2.0](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF/tree/17886395feb8a752b5ca5f02ad0c25543950915e) (`2.0bpw`, `2.5bpw`, `3.0bpw`, and reserved `mmproj-BF16` in [the NariQ lock](../config/nariq-qwen38.lock.json)) |
| Ternary Bonsai 2.5 27B GGUF | `28a7093cc25bc151cc2b284c5570f12ce343fdcf` | [Apache-2.0](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF/tree/28a7093cc25bc151cc2b284c5570f12ce343fdcf) (`PQ2_0` native default, `PTQ1_0`, bit-identical `mmproj-Q8_0`, and reserved `*-mtp` in [the Bonsai 2.5 lock](../config/bonsai-2-5-27b.lock.json)) |
| DaisyRose-27B (source RL) & `daisyrose-npu-r16-drw1.gguf` | Source `64a4ec665091bb0c1931babce4e97e686eef3c5c`; artifact SHA-256 `ba307a5f60372c69c2f4b773cc04771933a5db16b09aa5bc754b05743a7bc5d0` | [Apache-2.0](https://huggingface.co/RadixArk/DaisyRose-27B/tree/64a4ec665091bb0c1931babce4e97e686eef3c5c) (`RadixArk/DaisyRose-27B` minus `Qwen/Qwen3.8-27B`, extracted/edited in pure Rust via `DR-E1`/`DR-W1` and pinned in [the DaisyRose LoRA lock](../config/daisyrose-lora.lock.json)) |
| Qwen3-ASR 1.7B Q8_0 recognition | Conversion `0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f`; source `7278e1e70fe206f11671096ffdd38061171dd6e5` | Apache-2.0, recorded in the [native lock](../src/speech-asr/model.lock.json); retain model attribution and identify conversion changes. |
| Qwen3-TTS VoiceDesign/Base and codec | Conversion `b7ee2e8c7459c3bea99da23e3d178125a7d1713c` | Apache-2.0; [speech artifact documentation](../tools/qwen-tts/README.md#artifacts) identifies the Q8_0 and optional Q4_K_M selections. |

Swift's [NOTICE](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF/blob/a3bdac086bbb6b04d87d0108d18943b5f63868f7/NOTICE.txt)
attributes Bonsai to Prism ML and the Qwen base to Alibaba Cloud. Preserve it
and the model license when redistributing weights, including derived PTQ1;
identify changes to derived artifacts. Future models, accelerator SDKs and agent
tools require their own review if adopted.

## Redistribution

Include project LICENSE/NOTICE and every bundled dependency's applicable
copyright/license/notice material, including Unicode notices. Mark modified
Apache-licensed files. Include separate upstream notices/licenses when shipping
models/adapters. Review separately any additional build tools redistributed with a release.

Assemble notices from the exact shipped packages, runtime archives and artifacts.
Repeat review when pins, features or bundled assets change. The portable executable
embeds its notices; models and runtimes retain separate distribution terms.
