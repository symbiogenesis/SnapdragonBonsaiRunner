# Dependency policy and choices

[Cargo.toml](../Cargo.toml), package manifests, [Cargo.lock](../Cargo.lock) and the
[SDK reader lockfile](../tools/hexagon/sdk-reader/Cargo.lock) own the exact dependency
selection. [PLAN.md](../PLAN.md#deferred-dependency-experiments)
owns pending alternatives. This guide explains current responsibilities and acceptance
requirements; prior audit snapshots and prototype results belong in Git history.

## Dependency policy

Artifact verification belongs in `bonsai-artifacts`; the broker has no direct `sha2` dependency.
Shared artifact, maintenance, replay, and CI hashing now delegates through `bonsai-native-hash`
(Windows CNG streaming/native dispatch on Windows, portable SHA-256 elsewhere), while
`bonsai-windows-credentials` retains stateless `BCryptHash` for one-shot slice hashing and tests retain
`sha2` as an independent reference.
ZIP disables defaults and enables only `deflate-flate2-zlib-rs`. Bounded custom
extraction remains necessary; upstream hardening does not replace Runner policy.

[The source-checks audit job](../.github/workflows/source-checks.yml) runs for relevant
main pushes, weekly and manually. Pinned cargo-audit checks both lockfiles against
fresh RustSec advisories; vulnerabilities/yanks fail, while maintenance notices
remain visible without blanket ignores.

## Dependency footprint and duplication

Assess normal/build/dev dependencies and enabled features for each shipped executable
and target. All-target lockfile counts include foreign platforms; package counts do
not measure linked code, binary size, build cost or runtime speed. Cargo feature
unification can broaden a dependency even when Runner disables its defaults.

The required MXC ProcessContainer integration broadens the shared process boundary.
Its lifecycle and validation policy cannot be removed to reduce dependency counts.
A separate supervisor or archive crate split needs demonstrated build/size benefit,
reviewed ownership/IPC and unchanged containment. Native speech remains in separate
executables with independently qualified model, cancellation and resource contracts.

### Renderer features

`wgpu_no_default_features` plus the direct wgpu edge selects Windows DX12, Vulkan and
GLES support. Native chat retains upstream adapter selection and `WGPU_BACKEND`;
source launches can explicitly select a backend with `--graphics-backend`.
Removing a Windows backend requires a separate support and qualification decision.
Font parsers, color-font rendering and accessibility serve the native UI; they are
not evidence of a browser frontend.

## Direct dependency decisions

| Dependency | Decision |
| --- | --- |
| `serde`, `serde_json`, `tokio` | Retain: [JSON](#json-selection)/[runtime](#runtime-screening) screens did not qualify replacements. Tokio explicitly enables io-util/net/rt/signal/sync/time; test-util is dev-only, while MXC broadens features. |
| `http`, `httparse` | Complementary typed policy and bounded wire parsing; no general web framework needed. |
| `subtle` | Constant-time bearer comparison; preserve disabled release debug assertions. |
| `sha2`, `zip` | Artifact verification and minimal deflate extraction at ZIP 9.5.1; custom extraction policy remains necessary. |
| `windows-sys` | Selected process/credential/audio/accessibility APIs. Broker pre-bind options use Tokio's `TcpSocket` and the native exclusive-address helper; `socket2` is transitive through Tokio. |
| `windows`, `windows-future` | Compatible WinRT microphone permission/capture; distinct from windows-sys. Retain future 0.3.2 with windows 0.62.2; 0.100 is incompatible. |
| `eframe`, `wgpu` | Native UI, accessibility, clipboard, fallback/system fonts and all three explicit Windows backends. |
| `markdown-syntax` | Exact `=0.7.2`, default features disabled: dependency-free Markdown AST parsing for native response rendering. No HTML renderer or web frontend. See [Markdown parsing](#native-markdown-parsing) for the reviewed scope and limits. |
| `resvg` | Exact `=0.50.0`, defaults disabled, `text` feature only (no `svgz`, `raster-images`, `system-fonts` or `memmap-fonts`): dual MIT/Apache-2.0 pure Rust SVG rasterization inside the offline `--markdown-worker` container, using bundled `epaint_default_fonts` with external/data `href` resolvers disabled. |
| `png` | Exact `=0.18.1`, `default-features = false`: dual MIT/Apache-2.0 pure Rust PNG encoder already in `Cargo.lock`, used in `bonsai-chat` for user-initiated `IFileSaveDialog` PNG export of rasterized generated SVGs. |
| `egui_kittest`, `egui_system_font_provider` | Test harness/font checks; capture-only splitting offers little while production needs graphics. |
| `toml` | CI-only manifest parsing with parse/serde/std features. |
| `ureq` | Synchronous HTTPS for native provisioning, updates and CI. Reuse the workspace pin with defaults, automatic compression and cookies disabled; Windows selects Schannel and system trust through native-tls-no-default and PlatformVerifier, portable checks select Rustls. No new async runtime is required. |
| `transcribe-cpp` | Static CPU ASR, defaults disabled; reviewed CMake script downloads no source and explicitly enables MSVC optimization. |
| `simd-json` | Exact optional qualification-only crates.io pin; ordinary requests still use serde_json. |
| `fearless_simd` | Pinned safe SIMD (`=1.2.0` in Runner speech and the active external runtime, with independent locks). ASR uses it in `validate_pcm` and the Rust `frontend`/`encoder`/`decoder`/`packed` SIMD kernels; TTS uses it in the serving Rust `engine`/`codec` modules. |
| `bonsai-gguf` | Exact published `crates.io` pin (`=0.1.10`, `default-features = false`): dependency-free bounded cancellable `Read + Seek` GGUF v2/v3 stream index reader (`read_index`, `read_index_cancellable`, `stock_tensor_layout`, `scalar_width`) consumed by `bonsai-speech-asr` and `bonsai-speech-tts`. |
| `bonsai-native-hash` | Exact published `crates.io` pin (`=0.1.7`): shared Windows CNG / portable SHA-256 streaming boundary consumed directly by `bonsai-windows-credentials`, `bonsai-artifacts`, `bonsai-tools`, `bonsai-speech-asr`, `bonsai-speech-tts`, and `bonsai-ci`, avoiding duplicated CNG FFI handles and reducing direct `sha2` edges to independent dev-test verification. |
| `bonsai-neon-dot` | Exact published `crates.io` pin (`=0.1.10`): dependency-free shared ARM64 `SDOT`/`I8MM`/`FCVTL`/`FCVTN` capability and `kquants` (`quantize_row_q8_0`, `quantize_row_q8_k`, row/4-row/repacked dot and GEMM) boundary consumed by `bonsai-windows-process`, `bonsai-speech-tts`, and `bonsai-speech-asr` (`widen_f16_four`, `widen_f16_eight`, `widen_f16_slice`, `narrow_f32_to_f16_slice`, `dot_q8_0_block`, `smmla_q8_0_4x2`, `has_dotprod`, `has_i8mm`). Retires direct workspace `half` usage alongside `bonsai-ternary` (`half` remains transitive-only in `Cargo.lock` via graphics/image crates). |
| `bonsai-ternary` | Exact published `crates.io` pin (`=0.1.6`): dependency-free shared binary16 (`half_to_float`, `float_to_half`, `checked_float_to_half`) and quantized block primitives (`unpack_q4_k_scale_min`) consumed by `bonsai-windows-process`, `bonsai-speech-tts`, `bonsai-speech-asr`, and `bonsai-tools`. |
| `rayon` | TTS and ASR pin `=1.12.0` on non-Windows targets for safe borrowed projection, attention, and frontend work on a reusable, explicitly bounded team owned by its calling thread; Windows builds use `bonsai-windows-process::RowWorkerPool`. No global pool or whole-weight copies; one-worker and failed-team construction run serially. |
| `memchr` | Direct (`default-features = false`) in `bonsai-broker`, `bonsai-chat`, and `bonsai-speech-service` for NEON-vectorized HTTP header, SSE line, multipart boundary, and ASCII folded Find scanning. |
| Windows CNG | `bonsai-native-hash` owns the shared streaming CNG SHA-256 handle and `bonsai-windows-credentials` retains stateless `BCryptHash` (`digest_bytes`); CPU dispatch belongs to the Windows provider. These paths alone establish no device speedup. |
| `mxc-sdk`, `flatbuffers` | Required security integration; pursue upstream modularity. |
| Internal crates | Retain shared process/credential/audio/policy/artifact boundaries; no unused direct path edge found. |

ASR/TTS have separate executables and ownership. ASR pins CPU-only CMake;
RustSec does not fully cover its bundled C/C++. TTS serves in Rust on CPU under
its [source contract](../src/speech-tts/README.md). Speech uses only `VerifiedFile`
from `bonsai-artifacts`; splitting its unconditional ZIP APIs needs demonstrated
build/size benefit after Cargo unification and linker elimination.

### Native HTTPS downloads

The shared `bonsai-tools` HTTPS client replaces the external curl executable and
its version-admission subprocess. Rust's standard library supplies sockets, not
HTTP or TLS; ureq supplies HTTP framing and the selected TLS provider. Windows
uses the operating system's certificate trust and TLS updates through Schannel.
The pinned Rust dependencies make builds reproducible; they do not freeze the
Windows trust store or qualify every proxy and TLS configuration.

Runner retains URL, redirect, byte, deadline, cancellation and artifact identity
policy above the client. A small connector guard validates credential-free target
URLs, including redirected targets, before transport creation. The guard uses
ureq's pinned unversioned connector API and needs review when that pin changes.
Artifact transfers use fixed-buffer streaming and
validated range responses; updates retain independent compressed/expanded body
limits and conditional-cache/backoff policy. Cancellation invokes `Agent::shutdown()`
and joins the scoped HTTP worker; bounded DNS and TCP connection phases
remain independently necessary. Removing curl removes an installed
executable requirement, but no binary-size, latency or throughput improvement is
claimed without a matching measurement. [PLAN.md](../PLAN.md) owns live-network
and device qualification.

### Native Markdown parsing

The pure Rust `bonsai-markdown` library pins `markdown-syntax = "=0.7.2"` with
`default-features = false`. It has no dependency on `egui`, `eframe`, `winit`,
`wgpu`, or native presentation crates, emitting pure-Rust `BMD1`/`BMD2` wire packets. Production
parsing runs in a separate offline invocation of the configured headless
`bonsai-broker` image; the private entry point precedes model, credential and
network setup. The UI independently validates its output before rendering.
The reviewed crates.io package has no normal, build or development dependencies;
its manifest sets `build = false`, and both `default` and the unused `html` feature
are empty. Its Rust source contains no `unsafe` token. The package is dual licensed
MIT/Apache-2.0; the registry checksum in [Cargo.lock](../Cargo.lock) binds the exact
reviewed bytes. Repeat the source and dependency review before changing this pin.

The [offline MXC parser worker](sandboxing.md#markdown-parser-worker) consumes the
owned AST, absolute UTF-8 spans and validated top-level fenced `svg` blocks. The
trusted UI validates bounded `BMD1`/`BMD2` render descriptions and `BMG1`
rasterized RGBA8 packets before native egui layout and GPU texture upload,
without generating HTML. MXC reuses the application's existing process boundary;
the broker image hosts this separate worker without adding another packaged
executable or initializing its ordinary service.
Unavailable or failed worker parsing leaves plain source visible; there is no
in-process parser fallback. Headings, emphasis/strong/strikethrough, code,
lists/task lists, quotes, text-column tables and thematic breaks form the supported
subset. Link labels and image alt text are inert; raw HTML and unsupported
extensions stay literal. Source-to-display mappings keep Find tied to original
response bytes; matches without an exact displayed mapping show the message's source. Message Copy,
export, recovery and model history keep the original text. Parsing and mapping
retain the [transcript bounds](streaming-resources.md#native-chat-and-transcript)
and existing layout-cache reuse.

This is a young library, not an established reputation or independent performance
qualification. The reviewed source caps block and inline nesting at 32 and leaves
deeper constructs literal. Its upstream claims linear parsing and bounded stack,
but Runner makes no measured latency or throughput claim. The exact-source
[conformance harness](https://github.com/plimeor/markdown-syntax/blob/20c7d290545773044446f6c4421fa79d9fa067f0/tests/html_conformance.rs)
compares against vendored CommonMark/GFM cases, excludes nine CommonMark cases
that its extra syntax changes, and reports results without asserting a pass-rate
threshold. That harness and empty named defect lists do not establish complete
CommonMark conformance. Runner's supported-subset regressions and native device
acceptance remain separate obligations in [PLAN.md](../PLAN.md).

### Offline SVG parsing and rasterization

`bonsai-markdown` pins `resvg = "=0.50.0"` (`default-features = false`,
`features = ["text"]`), resolving `usvg = "=0.50.0"`, `roxmltree = "=0.21.2"`,
`simplecss = "=0.2.2"`, `svgtypes = "=0.16.2"`, `fontdb = "=0.24.0"` and
`tiny-skia = "=0.12.0"` alongside in-memory `epaint_default_fonts = "=0.37.0"`.
`bonsai-chat` has no normal dependency on `bonsai-markdown` or `resvg`; production
SVG inspection and CPU rasterization execute only inside the offline MXC/PSEC
`--markdown-worker` process, with no network stack in `bonsai-markdown`'s crate
graph and kernel-enforced `SandboxNetwork::Offline` denial.

Source review of the pinned closure establishes that the renderer provides **no
scripting, event, animation, DOM, network or host-font capabilities**:

- `resvg`, `usvg`, `roxmltree`, `simplecss` and `bonsai-markdown` set
  `#![forbid(unsafe_code)]` (or contain zero `unsafe` tokens). Transitive `unsafe`
  inside the untrusted worker is confined to SIMD and zero-copy data primitives
  (`tiny-skia`, `bytemuck`, `yoke`, `stable_deref_trait`, `base64`, `crc32fast`,
  `zlib-rs`) and never enters the trusted UI process.
- `usvg` disables `svgz`, `system-fonts`, `memmap-fonts` and `raster-images`; its
  closed element/attribute tables omit `<script>`, `<foreignObject>`, SMIL
  `<animate*>`/`<set>`, HTML/media elements and all `on*` event attributes.
- `fontdb` resolves with empty features (`fs`, `memmap` and `fontconfig` disabled),
  compiling out filesystem and system-font discovery; only bundled `UBUNTU_LIGHT`
  and `HACK_REGULAR` bytes are loaded into a shared read-only database.
- `simplecss` skips all CSS `@`-rules (`@import`, `@font-face`) without fetching
  URLs, and `roxmltree` runs with `entity_resolver: None`.

Because both model/user SVG inputs and the sandboxed worker are treated as
untrusted, `src/markdown/src/svg.rs` and `src/markdown/src/svg/policy.rs` enforce
strict pre- and post-conversion gates rather than relying on `usvg` silently
ignoring active markup:

- Parses XML via `roxmltree::Document::parse_with_options` with `allow_dtd: false`
  and `nodes_limit: 4_096` instead of `usvg::Tree::from_data` (which enables DTDs
  upstream), blocking `<!DOCTYPE ...>` and entity-expansion payloads at the
  tokenizer boundary, and rejects all XML processing instructions (`<?...?>`).
- Enforces closed static SVG element (`51` tags) and attribute allowlists and
  closed namespaces (`svg`, `xlink:href` only, and `xml:space`/`xml:lang` only,
  blocking `xml:base`, XHTML/MathML/XSLT, `<a>`, `<image>`, `<feImage>` and
  `<tref>`).
- Requires every `href`/`xlink:href` and CSS/attribute `url(...)` to be a
  same-document `#<id>` fragment, rejects `javascript:`/`vbscript:`/`data:`/
  `http:`/`https:`/`file:`/`blob:` tokens, CSS `@`-rules and CSS backslash escapes,
  bounds nesting depth (`32`), `feConvolveMatrix` kernels (`8x8`), cycle-free
  `href` chains and `<use>` DAG expansion (`8,192` visits), and verifies resolved
  `usvg::Tree` caps (`16,384` nodes, `262,144` path segments, zero `Node::Image`)
  before `resvg::render`.

Rejected fenced `svg` blocks remain visible as inert `BMD1` code blocks in chat,
and both `.png` and `.svg` file exports require a validated `BMG1` raster response
from the offline MXC worker before writing to disk. Repeat this source and feature
audit before changing any `resvg`/`usvg` pin.

## JSON selection

Production uses `serde_json`; the optional parser candidate cannot change ordinary
requests before qualification. The gate requires at least 5% lower equal-weight
geometric-mean time with no case more than 5% slower in both cold and reused screens.
Six workloads cover plain/escaped small inputs, 8/32/128-message histories and near-1
MiB bodies. Timing includes immutable-input preparation, schema/history validation,
canonical serialization, allocations/copies and destruction. Discard one warmup round;
rotate order over nine measured rounds. Unicode, duplicate/unknown fields, shapes,
bounds and canonical bytes must pass before interpreting timings.

### Current candidate and pending gate

The optional `json-candidate` uses the exact crates.io pin and registry checksum
in [Cargo.lock](../Cargo.lock). `Deserializer::from_tape` recovers/resets tape
allocation while preserving borrowing and validation.

`padded-tape-serde-reuse` reuses padded input, parser buffers and tape across requests:

| Screen | Included work |
| --- | --- |
| Cold | Scratch setup/destruction plus all per-request work below. |
| Reused | Immutable-body copy, validation, canonical serialization, output destruction and decimal/error fallback; excludes scratch setup/destruction. |

Explicit adapter scales and SIMD parse failures fall back to production parsing:
correctly rounded SIMD conversion can differ from serde_json's default without
either being wrong. Supplemental scale cases include fallback cost. The retired main-only
`json-candidate.yml` workflow ran correctness before the original six-case and supplemental
5%/5% gates; any case over 5% slower rejects qualification.
**Both cold/reused screens must pass; qualification remains pending.**

## Runtime screening

Tokio remains the broker runtime. A replacement must preserve bounded HTTP parsing,
queues, one-active-request admission, disconnect cancellation and absolute deadlines.
Include thread creation/stacks and lifecycle work when comparing blocking and async
designs; a loopback echo microbenchmark cannot qualify the complete broker or model.

## Reviewed pins and tooling

Use the exact [Rust toolchain](../rust-toolchain.toml), workspace/package manifests,
model/source locks and action pins. A pin gives reproducibility, not continuing
advisory coverage. Review security, source, supported-target and native-linking
obligations independently whenever an identity changes. Follow the
[runtime lock-update procedure](development-guide.md#prepare-a-runtime-lock-update)
for engine releases; speech source locks have their own build and model acceptance.

## Validation and evidence

The owner operates dependency-audit and Foundation acceptance under
[AGENTS.md](../AGENTS.md). Local format, Release analyzers and focused checks establish
only their tested source/target properties. Full suites and benchmarks remain CI-only.
Keep audit notices visible, without blanket ignores or removing accessibility merely
to suppress dependencies. Audits do not exhaustively inspect bundled native sources,
downloaded tools, drivers, models or every transitive build script. Qualification
requires reviewed results bound to the current revision and selected artifacts.
