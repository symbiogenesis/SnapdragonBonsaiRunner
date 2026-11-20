# C++ retirement contract

The milestone removes C++ from Runner and Prism, including controls, generated
source, bundled/downloaded libraries and compiler/build dependencies.
[PLAN.md](../PLAN.md) owns remaining work. Preserve the [native UI](ui-architecture.md),
[local API](broker-policy.md), source identities, licenses/notices and independent
replay inputs. Rust bindings and process isolation do not retire a C++ dependency.
Vendor OS/driver ABIs require an explicit boundary decision.

## Replacement and deletion rules

Before retiring each file or inseparable family:

1. Bind behavior to immutable source/patch, license, target/toolchain, inputs,
   configuration, assertions and failure/lifecycle cases.
2. Preserve independent controls as immutable input/output fixtures, mathematical
   known answers or separately implemented Rust references. Identify captured
   observations as captures, without claiming fresh execution of the old engine.
3. Pass focused replacement checks and retain every approved assertion, tolerance,
   numerical order, exceptional bit, ISA boundary, green ID and exclusion. Leave
   owning hosted consumer acceptance pending when only local checks ran.
4. Remove operative includes/generators, compiler/link/download/cache paths,
   dependency/feature edges, workflow dispatch and obsolete documentation.
   Retain notices and required evidence as data.
5. Qualify actual-model/device behavior, quality, resources, isolation and
   distribution independently under [validation](validation.md). Synthetic replay
   and source deletion do not satisfy those gates.

The [conformance boundary](prism-upstream-tests.md) and
[native runtime](native-rust-engine.md) define acceptance. Provider replacement
must preserve the complete upstream catalog/schedule and all required green IDs;
missing captures/assertions remain pending or rejected.

## Runner checked-in native files

`tools/hexagon/windows_queue_abi.c` remains the hosted C control for pinned Windows
ARM64 public declarations, packet layout and queue-call signatures. It links no
inference library, SDK runtime or driver. Retire compilation only after independent
retained observations and Rust assertions satisfy its owning contract.

Queue observation schema 2 records every asserted size, alignment, member offset
and constant, together with separate timed/nonblocking mock results. The verifier
rejects omitted or changed observations; the Rust producer exposes the same
complete layout selection. This closes the former partial-output gap. Original
Windows C observations and owning hosted capture remain required before deletion.

The public QAIC start/stop harness and its generator/Haskell/Cabal/GCC route are
retired. The [public wire contract](hexagon-source-audit.md#public-qaic-generation-and-wire-control)
now authenticates the original successful provider, complete retained selection,
generated/public inputs and original C/Rust outputs before comparing fresh Rust
encoding and literal assertions. Original harness/provider source remains inert
replay data with exact historical identities. Retired generation requests fail
before intake. This closes the fixed Linux LP64 vector's source/build dependency;
it supplies no fresh Windows C ABI observation, SDK equivalence or device verdict.
Owning hosted acceptance of the replay migration remains pending.

Speech's licensed Philox, sampling, tokenizer and lifetime observations live in
`src/speech-tts/tests/fixtures/`. Keep their provenance, replay inputs and independent
assertions. [Speech documentation](../src/speech-tts/README.md) owns serving behavior.
Philox, sampler, graph-lifetime and tokenizer replay use safe Rust directly;
all obsolete C ABI adapters and their unsafe allowance are removed. The tokenizer
shares serving's bounded transactional metadata loader, preserves all 1,810
immutable native requests and checks candidate cleanup, failed replacement and
owned result reuse. Its historical pointer/holder assertions have explicit Rust
ownership dispositions in the retained provenance. Historical graph-cache
fixtures exercise a test-only cache and do not qualify current serving graphs.

## Prism checked-in native files

Use Prism's [migration inventory](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/upstream/migration.json)
for the file-level provider list. The native boundary under
`tools/prism/upstream/common/` includes `bridge.cpp`, its 34 C++ headers and coupled
`public_c_abi.h`/`public_c_abi.c`. Retire each route only after its numerical,
text/API, model/session or concurrency assertions have independent replacements;
remove the final dispatcher after every provider has migrated.

Two model-free reference families now use separate Rust implementations in the
owning reference arm. MTP derives its six original predicates from nonnegative
capacity minus reserved anchors. Reasoning budget uses active delimiter-prefix
positions and accepted-event ordinals, preserving all 71 source semantic
conditions, null-vocabulary lifecycle, clone ownership, exceptional candidate bits
and UTF-8 boundary behavior. The reasoning C++ header and both families' native
include/dispatch/build edges are removed. Original source IDs, required reference
green IDs, catalog and ordered geometry schedules remain intact; reports disclose
the independent providers and do not claim native replay. Owning hosted acceptance
and the remaining provider families stay separate gates.

The concurrent logger reference now uses independent Rust bounded-channel
producers and an owned consumer, separately from production's shared writer.
It preserves the original eight-worker Windows random schedule, 8,000 messages,
format toggles, joins, flush/free and required `test-log.cpp` green ID; the
source has no output-content assertions. Its C++ header and native include,
dispatch and build edges are removed. Three focused controls and strict native
ARM64 Release analysis passed. Remaining native bridge compilation and owning
hosted acceptance stay pending; the catalog and both geometry schedules are
unchanged.

Independent Rust MTMD tiling/grouping, chunk wire/lifetime and grammar-parser
providers additionally preserve all 29, 46 and 274 original assertions respectively.
They use separate integer-lattice/run grouping, bounded wire decoding and
lexer/AST/lowering implementations, without calling production Rust or looking
up expected answers. Their three C++ headers and native include/dispatch/build
edges are removed; source identities, required green IDs, catalog and ordered
geometry schedules remain intact. Focused controls and strict native ARM64
Release analysis passed. Native bridge compilation and complete owning acceptance
remain pending. Separate independent Rust providers for `threadpool.hpp` (`test-barrier`),
`batch_allocator.hpp` (`test-batch-alloc`), `allocation.hpp` (`test-alloc`),
`quantization.hpp` (`test-quant-type-selection`), top-level `unicode-*` dispatch in
`bridge.cpp` (`test-unicode`), `arena_graph.hpp` (`test-moe-weighted-reduce`),
`tensor_graph.hpp` (`test-col2im-1d`, `test-dfly-fusion`, `test-rope`),
`gguf.hpp` (`test-gguf`), `sampling.hpp` (`test-sampling`), `grammar.hpp`
(`test-llama-grammar`, `test-gbnf-validator`, `test-grammar-integration`), and
`json_schema.hpp` (`test-json-schema-to-grammar`, `test-grammar-integration`)
additionally pass their focused `reference_*` unit tests and Release ARM64
Clippy, removing those ten C++ headers and leaving 25 checked-in C++ files (plus
2 C ABI files = 27 total native files; 28 cross-repo including Runner's 1 C
file). Independent clean-room Rust providers for `public_c_abi.c`/`.h`
(`test-c` via `reference_public_c_abi.rs`), `dspark.hpp` (`test-dspark-logsnr-meta`
via `reference_dspark_logsnr.rs`), `arg_parser.hpp` (`test-arg-parser` via
`reference_arg_parser.rs`), `optimizer.hpp` (`test-opt` via
`reference_optimizer.rs`), `codec.hpp` (`test-quantize-fns` and
`test-ptq1_0-cuda-dot` via `reference_codec.rs`), and `jinja.hpp` (`test-jinja`
via `reference_jinja.rs`) additionally pass their focused `reference_*` and
assertion tests, `rustfmt --check`, and Release Clippy on integrated Prism
`main` (`15e3e59fc`), removing those 5 C++ headers and 2 C ABI files (with
helper functions inlined into `codec_buffers.hpp`) and leaving 20 checked-in C++
files and 0 C files in Prism (20 Prism native files; 21 cross-repo including
Runner's 1 C file), with all 19 `RequiredReferencePassedIds` in `baseline.json`
off checked-in C++; complete Prism suite execution, native bridge compilation and
owning hosted acceptance remain pending.

The [checked-in boundary verifier](../tools/ci/README.md#checked-in-native-retirement-boundary)
now binds Runner's remaining C file and the reviewed Prism snapshot's 20 C++
files plus migrated Rust assertion/provider edges.
Exact per-file families and declared
compiler/provider/contracts identities preserve the green-ID baseline and both
original ordered geometry schedules. Four focused controls, strict Release
analysis/docs and real snapshot verification passed. The guard rejects added,
deleted or changed files/declared edges and every retirement-qualified status;
inventory acceptance cannot authorize deletion. It verifies original bytes,
not Git revision provenance, external archives or compiled/link closure.

## External engines, generators and build dependencies

| Boundary | Retirement requirement |
| --- | --- |
| TTS | Serving executes GGUF intake, Talker/CodePredictor, codec and speaker graphs in Rust on CPU. Preserve immutable controls and qualify Q8/Q4 model parity, listening, state/lifecycle and resources; native-linked results do not qualify Rust serving. |
| ASR | `transcribe-cpp` / `transcribe-cpp-sys` bundles Qwen3-ASR/GGML. A replacement must preserve verification, audio/token contracts, cancellation/state, output quality and device resources. |
| Publisher runtime and PTQ1 collector | The collector uses pinned native inference DLLs. A Rust driver does not replace that engine. Retire publisher locks, builders, capture routes and patches only after replacement/fallback and distribution gates pass. |
| Hexagon | Pure Rust model/codec execution and signed `hexagon-native-rpc` replace GGML host engines. Review vendor FastRPC separately: DriverStore `libcdsprpc.dll` (`qcnspmcdm`), `remote_handle64_*`, `rpcmem_*`, `dspqueue_*` and signed `libggml-htp-v73.so` / `libggml-htp-v81.so` DSP skels. |

Expand external families to exact pinned source files before porting. Prism's
remaining boundary includes CMake projects/adapters, native loader/LoRA-export
linkage, LLGuidance and codec downloads, packaging, caches and provider dispatch.
Preserve assertions when removing their build paths. TTS's Rust build and scalar,
tokenizer and graph-lifetime fixtures have no live source-overlay generator or
native Qwen/GGML registration dependency.

The obsolete `speech-build stage-source` command and its Qwen/GGML ZIP extraction,
source-copy verification and cache-publication machinery are removed. It has no
serving, build or notice consumer after the Rust TTS port. Retired requests fail
before intake. Native ASR's compiler provisioning and LLVM archive-name guard
remain operative; this deletion grants no ASR or whole-application retirement.

ASR's [registry-package inventory](../src/speech-asr/native-sources.json) binds
all 1,718 files in `transcribe-cpp-sys` 0.4.1 to byte lengths and SHA-256 identities,
including GGML, platform sources, generators, build inputs and notices. It retains
file identities rather than copied source. The Rust
[`native-source-verify` command](../tools/ci/README.md#native-source-inventory)
checks the complete extracted file set, original archive checksum, Cargo lock and
upstream revision without compiling or executing the package. Only Cargo's local
`.cargo-ok` extraction marker is excluded. This package inventory includes unused
backend families; it does not establish the selected preprocessed/compiled source
closure, inference parity or permission to delete native files.

The Rust [`native-closure-verify` command](../tools/ci/README.md#selected-compiler-source-closure)
separately checks a reviewed selection of compiled translation units against
original Clang dependency reports, preprocessed/object identities and generated
input provenance. Schema-2 captures recursively classify and verify generated
intermediates, including each original successful invocation and working directory;
schema-1 replay retains its direct-input scope. Schema 3 additionally binds the
original complete argument-array compilation database, rejecting omitted, extra,
duplicate or changed invocations and preserving distinct configurations of one
source file. Five focused controls and strict Release analysis passed. Owning
builds must still establish complete selection, effective configuration and
linked-library closure; a selected capture does not qualify native retirement.

The separate Rust [`native-link-verify` command](../tools/bonsai/README.md#selected-native-link-and-package-capture)
binds the independently reviewed schema-3 source-verifier manifest/receipt to
every selected object, an explicit successful ARM64 lld-link invocation, original
linker/object/library bytes, actual normal/delay PE imports and exact flat package
contents. Packaged images must match linked outputs byte for byte. Response files,
library search and implicit default-library inputs are outside its supported
subset. It neither reexecutes the source verifier/linker nor establishes static
library source ancestry, system dependency closure or absence of embedded C++.
Actual owning capture and hosted no-project-C++ builds remain separate gates.

The source verifier's [`native-closure-collect` assembler](../tools/ci/README.md#selected-compiler-source-closure)
hashes a reviewed selection of retained original artifacts into schema 3 and
runs the existing verifier before emitting the manifest. Three new collector
controls and the existing source controls passed (20 selected checks), with
strict native ARM64 Release all-target analysis and docs. It executes no compiler
or generator, authenticates no supplied exit code and establishes no effective
compiler target. Its separate `clang-cl` subset preserves original argument
arrays, matched ordered configuration, explicit outputs and complete forwarded
dependency generation; CRT `-MD` cannot substitute for dependency `-MD`. Three
new literal controls and two named compatibility controls, strict ARM64 Release
all-target analysis, docs and production tool linking passed. Hidden PCH/modules,
unknown options, extra sources, shell-command databases and response files reject.
Actual owning-build instrumentation and original captures remain required;
historical shell commands or relative outputs cannot be rewritten into evidence.
Rehashing is not an atomic snapshot or continuous file-stability proof.

The explicit [`native-closure-capture`](../tools/ci/README.md#explicit-live-compiler-capture)
route can execute a newly reviewed supported Clang/clang-cl selection, retaining
the original request, argument arrays, actual completed statuses and required
source artifacts before collector/verifier handoff. Its inherited absolute clock
covers supervised compiler processes and deadline-aware intake, hashing and
verification; failures leave later phases explicitly unattempted. Source/compiler
hashes are endpoint observations, and effective source/configuration and complete
owning-build selection still require independent review. This route does not
instrument CMake or supply historical missing captures. Actual owning captures,
original generated-input provenance, link closure and hosted acceptance remain
pending; synchronous OS I/O and retention require the external owner's supervision.

The Rust [`native-archive-inventory` command](../tools/ci/README.md#immutable-external-source-archives)
also expands pinned external source ZIPs into complete sorted member identities,
including build inputs and notices, without extraction or source execution. It
binds original archive/pin hashes and rejects unsafe or aliased names, unsupported
members, expansion violations and file/directory collisions. Its four focused
controls, native strict Release analysis and tool link passed. Actual source
intake is not qualified: the pinned publisher URL returned HTTP 404, stopping
the fixed two-archive workload before Prism intake without retry. The
[retained no-go](evidence/native-source-archives/summary.json) records exact
identities, missing pre-setup power, intake/cleanup power, allowance and cleanup.
Submodule contents and selected compiler/link closure remain separate inventories.

Source-location review found commit
[`73bfe6cc485554822f80c473a3c0c4f25e80f41b`](https://github.com/PrismML-Eng/llama.cpp/commit/73bfe6cc485554822f80c473a3c0c4f25e80f41b)
publicly available in PrismML's `llama.cpp` repository. Git history traced the
different `prism-engine` URL/size/digest to commit `cb00d288`, an unrelated Rust
release repin. The [`native-reference` pin](../tools/native-reference/pins.json)
restores the exact native tuple from its parent: 39,423,279 bytes/SHA-256
`5ee8738706229e84e85db7051d7a40797c91e22a7ffa8f4bfb8fea4bf2cb0293`, also retained
in the earlier [headless source record](runtime-candidates/headless-ucrt-argv.json).
Native Git commit/header identities and the current Rust source pin are unchanged.
The historical 404 receipt retains its actual failed URL and pin; its workload
stays closed. No archive intake was repeated. Fresh original-byte verification
and file inventory under the restored pin remain owning-CI gates; a corrected
URL or equal Git commit text supplies neither verdict.

Rust ASR exposes borrowed mel, encoder rows, prompt IDs, each full logits/greedy
decision and all-layer committed KV through EOS, before its state resets.
Observer refusal, cancellation and unwind clear every layer without publishing
a partial transcript. Six focused controls, strict Rust-only Release analysis,
docs and the qualification executable link passed. This supplies the Rust half
of a comparison: callers still bind exact source/model/PCM and independent
expected observations. It neither captures the native engine nor qualifies S2.

Rust prompt construction retains the admitted model's authoritative audio token
IDs, including duplicate vocabulary spellings. Prefill projects the tied vocabulary
head only for the final prompt position, matching the native selection; earlier
positions retain their block, final-normalization, cancellation and transactional
cache checks. Literal prompt, unused-head overflow/final-failure, complete cache,
observer and miniature EOS/replay controls passed with strict Rust-only Release
analysis. These corrections supply no measured model latency or S2 verdict.

The ASR CNN also preserves GGML's binary16 im2col/kernel operand boundary, including
nearest-even ties, signed zero and subnormals. Kernels narrow once at intake and
each stage narrows post-GELU activations before its scalar F32 dot. Finite overflow
rejects transactionally. The candidate arithmetic identity is v2; old v1 candidate
packets reject without reinterpreting retained observations. Independent dyadic
and staged-convolution controls supply no native reduction-order or model verdict.

Rust GGUF intake now polls the caller's original cancellation/deadline callback
through metadata scalars, buffered strings and tensor-range finalization, before
initial source I/O and before publishing loaded graph/tokenizer state. The focused
pre-cancelled and buffered-metadata controls preserve interruption and fresh intake;
the omitted-unit-axis projection/convolution controls remain passing. Rust-only
strict native Release analysis covers the qualification binaries. Blocking OS I/O
still needs the external owner's original deadline and process supervision; these
checks supply no actual-model cancellation or S2 acceptance.

The [ASR replay comparator](../src/speech-asr/README.md#streaming-independent-replay)
now binds reviewed reference/candidate source/compiler/target/executable,
arithmetic, verified model, actual PCM digest and explicit tolerances. It reads
original expected values with bounded scratch and compares every stage, ordered
generation decision and all-layer KV, including exceptional bits when requested.
Engine success, the exact terminal transcript and EOF are mandatory; cancellation
during terminal/EOF intake cannot grant success. Seven focused controls and strict
Rust-only Release analysis/docs passed. Arbitrary readers still need caller-bound
I/O and the original absolute deadline. No independent native observations were
manufactured or captured, and production ASR remains on its existing native engine.

The explicit [`bonsai-asr-replay` consumer](../src/speech-asr/README.md#prebuilt-independent-replay-consumer)
verifies the reviewed request, executing candidate image, pinned model, original
PCM and complete packet identities before running Rust replay. Its inherited
native counter deadline covers intake, hashing, loading, comparison and cleanup;
the owner retains the aggregate clock, reference-arm power observations and
external process supervision. Endpoint power observations and a successful
complete replay remain separate from performance and production qualification.
The consumer supplies no missing native packet or upstream observation ABI.

The provider-neutral [`CaptureWriter`](../src/speech-asr/README.md#original-observation-packet-writer)
serializes that complete replay from borrowed original observations, with reviewed
geometry/EOS, contiguous full logits/KV and sticky write/order/cancellation/unwind
failures. It widens original F16 cache words scalar by scalar through a fixed
16 KiB buffer; terminal publication requires successful provider completion and
flush. Five writer controls and the existing literal replay control passed with
strict Rust-only Release analysis/docs. The pinned Rust/C API exposes no numerical
observer, and internal device argmax/debug output cannot supply the missing
observations. A reviewed upstream observation ABI and actual independent captures
remain required; serialization alone supplies neither native execution nor parity.

The Rust loader preserves GGML's omitted trailing unit dimensions for F32/F16/Q8
projections and F32/F16 convolutions, with three selected wire-loader controls and
strict Rust-only Release analysis/docs passing. Complete native replay remains
unavailable: native debug output exposes selected intermediates/prefill logits,
but not every generated token's raw logits and all-layer committed KV. A native
observation provider must supply those values before the replay can qualify S2.

[Rust TTS observations](../src/speech-tts/README.md#independent-graph-observations)
likewise lend complete prompts/overlays, raw semantic and acoustic logits before
sampling, all committed Talker/predictor KV, seeded decisions through EOS, codes,
speaker/reference conditioning and validated Design/Base PCM. Callback refusal
and unwinding invalidate reuse after input admission; ordinary service installs
a no-op observer. Six focused controls, strict Release analysis/docs and service
linking passed. These observations execute Rust only. Independent pinned-model
codec/speaker controls, complete parity, listening and resource gates remain open.

The [TTS replay comparator](../src/speech-tts/README.md#streaming-independent-replay)
binds both providers, all model/lock digests, actual text/voice bytes, worker count
and retained-voice state. It compares every ordered prompt, hidden/logit/cache,
decision/code/conditioning and PCM event through a fixed 16 KiB buffer. Complete
and terminal EOF are checked inside synthesis cleanup ownership; only successful
engine return grants success. Literal miniature controls and independent review
cover the comparator, without supplying original native observations.

The explicit [prebuilt TTS replay consumer](../src/speech-tts/README.md#prebuilt-consumer)
verifies its executing image, compiled Q8/Q4 model pins, original UTF-8 inputs and
complete packet identities/headers before loading. Ordered cold, repeated, edited
and changed-voice cases share one synthesizer and inherited absolute owner deadline.
Failures leave later cases unattempted, and candidate-case/entry/exit power stays
separate from reference-arm and aggregate observations. It opens no audio device
and supplies no missing independent packets or S1 acceptance. Ordinary Rust TTS
builds no longer invoke an empty build script.

TTS's provider-neutral `CaptureWriter` writes the same bounded replay format
from borrowed original prompts, logits, all-layer caches, decisions, codes,
conditioning and PCM. Reviewed geometry, complete cold/warm ordering and EOS
are mandatory; only successful provider return, terminal flush and final
cancellation acceptance permit retaining a complete packet. Refusal and unwind
are sticky. Adapting Rust observation events records Rust execution only; the
writer authenticates neither a provider nor independent pinned-model answers.

Eight `speaker_reference_` mathematical controls additionally exercise production
Slaney linear/logarithmic anchors, eight-point Hann/DFT and channel-major mel,
signed TDNN/reflection/dilation, SE gates, Res2Net carry/residual and channel-specific
attentive moments, plus a complete miniature waveform-to-vector graph with
closed-form embedding answers. The speaker frontend now preserves pinned
`audio-mel.h`'s double-precision angles, narrowing before Hann subtraction and
multiply/divide/add mel interpolation. The candidate arithmetic identity is v2;
replay rejects v1 candidate identities without reinterpreting old observations.
These controls establish synthetic graph behavior, not complete pinned-weight
x-vectors, native parity, voice identity or listening acceptance.

Codec transformer appends are provisional until all layers, final normalization
and output projection succeed with finite output. Sliding-window eviction occurs
only at commit; errors and unwinds restore the original cache bits, extents, layer
count and position without copying complete caches. Four focused transaction
controls and the existing worker/state control passed with strict Release analysis.
Whole-codec/PCM, pinned-model cancellation and recovery remain independent gates.

Reference RVQ encoding rejects invalid geometry, nonfinite/overflowing
centroid norms, projections, candidate scores and residuals. Failure and unwind
restore the caller's original code prefix; successful finite-input FMA order and
first-index ties remain unchanged. Four new mathematical/failure controls and the
existing nearest-centroid control passed with native ARM64 Release analysis/docs.
These synthetic controls do not supply independent pinned-codec observations.

Three `dac_reference_` controls now exercise the complete four-block 8/5/4/3
waveform graph, nonzero overlap and every dilated residual branch against an
independently factored impulse polynomial, chunk schedules and flushed state.
Nonlinear final-Snake and finite saturation controls complement the exact dyadic
PCM answers. Final DAC output rejects nonfinite values before clipping, so overflow
cannot become apparently valid full-scale PCM. Strict Release analysis/docs and
one existing streaming regression also passed. These synthetic weights do not
supply independent pinned-model codec/PCM observations or listening acceptance.

Four further codec controls use closed-form miniature Mimi/decoder layers,
nonuniform grouped/window attention and state-preserving normalization refusal.
Mimi LayerNorm now rejects finite-input reduction or output overflow before
apparently valid bias-only results can commit, preserving valid arithmetic order.
Nine selected regressions, strict native ARM64 Release analysis and docs passed.
These mathematical controls supply no missing pinned-model/native observations
or S1 acceptance.

The [native Q8 projection adapter](../tools/hexagon/native_rpc/README.md#q8-matrix-preparation-and-immutable-placement)
connects immutable host dispatch to signed-queue ABI calls, exact FD mapping,
correlated completions and mandatory independent output review. Failed or
uncertain work, including caught callback unwinds, remains terminal and retains
owners through supervised exit. Twelve focused model-free controls and native
warning-denied analysis/docs passed. Actual identity admission remains the
supervisor's obligation; simulated ABI calls authenticate no source/model/package/
catalog/driver/VTCM/loaded-DSP tuple. Contained HTP0 execution and TTS integration
remain pending. TTS's [signed-dispatch prerequisites](../src/speech-tts/README.md#optional-hexagon-runtime)
still require independent wire/Windows ABI controls; neither these simulations
nor fixed F32/guarded-Q2 controls qualify Q8 arithmetic, shared KV or speech state.

The separate `Q8ExactOracle` reviews original row-major signed dots for a strict
diagnostic subset: integral activations with magnitude-127 anchors, dyadic weight
scales and exactly representable ordered accumulation prefixes. Unsupported
quantization rejects before submission. Four literal controls passed; they are
mathematical answers rather than DSP observations. Hosted native receipts now
bind every Q8 dispatch, native, projection and oracle source/test input; omitted
or changed identities reject. Signed execution, general speech activations and
device source/package/driver admission remain pending.

`Q8ReferenceOracle` separately evaluates original row-major signed weights and
ordinary finite F32 activations in ordered scalar F64, including finite half
subnormal/signed scales. Every output must meet explicitly predeclared absolute
and relative error limits before publication. Five new controls and the four
exact-oracle controls passed with optimized warning-denied native compilation and
docs; Clippy passed with documented preexisting exceptions. This is ideal
projection-error review, without emulating HVX half quantization, QF16 or kernel
reduction order. `Q8HvxQuantizationOracle` in `tools/hexagon/native_rpc/q8_oracle.rs`
further models exact binary16 (`0x2008`) activation scaling/inverse, signed RNE codes
and QF16 tiled-block accumulation (verified with focused `q8_` unit controls under
warning-denied `rustc --test -O`). Session-bound loaded-DSP bytes and usable VTCM,
predeclared speech error tolerances, actual signed-path device error qualification,
cancellation/deadline review and speech acceptance remain independent open gates.

The missing Q8 admission provider must independently bind the actual loaded
signed module bytes and effective usable VTCM to this session and its guarded
source/package/catalog/driver identities. Current URI and DSP capacity queries
do not establish those facts; the [source ledger](../tools/hexagon/native_rpc/sources.json)
keeps loaded-DSP binding pending. The exact diagnostic oracle also restricts
activation/scaling arithmetic; it cannot qualify ordinary speech activations or
authorize serving integration. A CLI route that reused fixed F32/Q2 admission
would lose these requirements.

Both Q8 oracles provide controlled preparation, review and submission methods
that poll the caller's original cancellation and absolute work deadline through
the final output review. Bounded owner work must use these methods and preserve
cleanup reserves; the convenience methods impose no time bound. Focused controls
exercise refusal during intake/arithmetic and late review without publication.
They establish neither native cancellation latency nor signed-session admission.

Final acceptance requires source/generated-source and dependency inventories,
binary-linkage assessment, clean hosted consumer builds without a project C++
toolchain, preserved qualification contracts and no operative retired controls.
Keep historical fixture metadata and legally required notices as data.

## Speech acceptance

S1 requires the pinned Q8 and Q4 model paths, complete independent expected
observations and the original tokenizer, Philox/MSVC tie and graph/cache/compute-lease
controls. Exercise cold, repeated, edited, voice-changed and failed utterances
through packed projections/worker teams, codec transformer/output tiles,
convolution/ConvNeXt and exact Base-prefix reuse. Include single-pass tensor
validation, metadata scratch reuse and short-frame convolution channel tiling.
Qualify Design teardown before Base extraction and the retained reference's voice
identity; primitive checks or synthetic PCM cannot supply these model verdicts.

Predeclare feasible matched-power workloads for cold loading, first audible
output and sustained readout/listening, including 400 ms playback pre-roll,
queued-PCM underrun recovery, slow supply and chat overlap. Diagnose the observed
latency regression before claiming benefit. Cancellation, recovery, resources,
full-model numerical/audio parity and human listening remain independent gates.
Source checks, literal replay controls and a source/cache-path deletion do not
close them. S2 independently retains its native numerical/transcript controls and
microphone-to-Swift, quality, lifecycle and resource requirements.

## MSVC startup and panic runtime

[Static CRT linkage](https://doc.rust-lang.org/reference/linkage.html#static-and-dynamic-c-runtimes)
removes separately installed Visual C++ DLL requirements by embedding Microsoft
runtime support; it does not replace those bytes with Rust. Inspect every shipped
image's imports. Apply [target flags](https://doc.rust-lang.org/cargo/reference/config.html#buildrustflags)
to the explicit application target, excluding host build scripts/procedural macros.

Rust's [MSVC SEH implementation](https://github.com/rust-lang/rust/blob/main/library/panic_unwind/src/seh.rs)
uses `_CxxThrowException` and `__CxxFrameHandler3`. Both workspaces retain unwinding:
Runner cancels/joins preparation peers after a panic, Prism drains active CPU
workers, and scoped cancellation/completion guards depend on destructors. Changing
`panic = "abort"` would alter recovery; ordinary passing tests do not qualify
[abort-mode releases](https://doc.rust-lang.org/cargo/reference/profiles.html#panic).
Neither workspace has a qualified pure Rust ARM64 MSVC CRT/SEH provider.

Schema 2 of the retained [link/package verifier](../tools/bonsai/README.md#selected-native-link-and-package-capture)
additionally inspects actual ARM64 COFF objects and MSVC archive members for a
bounded known C++ exception-symbol family, including short-import targets.
Schema 3 also observes named normal/delay imports in each exact linked/package
image and reports unresolved ordinal imports separately. Schema 4 additionally
resolves original named/ordinal imports through digest-bound retained package or
dependency images, including bounded named/ordinal forwarder chains and all export
aliases. Missing, cyclic, malformed and API-set routes reject; no loader search or
host DLL substitution supplies a missing provider. Historical replay is preserved.
These observations do not establish which archive members reached the
shipped image, complete runtime/source ancestry, or absence of embedded C++.
Schema 5 additionally traverses each used retained dependency's normal/delay
imports, including forwarder-only providers and zero-symbol descriptors, under
the same aggregate limits. Missing, API-set and malformed routes reject;
mutual DLL imports are visited once. Four focused controls and strict native
ARM64 Release analysis passed. Historical schemas retain their earlier scope.
`cppRuntimeAbsenceQualified` and `nativeRetirementQualified` remain false;
compiler/sysroot provider replacement and recovery acceptance are still required.
Literal link/parser controls cover the supported subset; no actual
toolchain/runtime artifact was inspected. Actual recursive dependency captures,
API-set routing, OS loader selection and provider implementation language remain
unqualified.
