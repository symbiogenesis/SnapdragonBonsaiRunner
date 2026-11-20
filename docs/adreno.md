# Adreno GPU contracts and OpenCL feasibility

Native Windows ARM64 Adreno GPU inference is unqualified on X1/X2. The current
Rust runtime uses CPU inference with the separate [X2 Hexagon development
default](hexagon.md); that NPU path establishes no GPU support.
[PLAN.md](../PLAN.md#adreno-vulkan) owns outstanding work. The shared
[profile policy](hardware-capabilities.md#automatic-profile-selection) governs selection,
admission, state transitions and promotion for both Vulkan and OpenCL.

## Backend and device boundaries

The publisher C++ Vulkan backend is a reference for independent Rust GPU work.
PQ2 and [current-weight PTQ1](swift-memory.md#current-weight-ptq1-option) qualify
separately. The [backend audit][prism-audit] identifies packed scalar, coopmat1
and integer-dot paths, without a direct coopmat2 decoder. Source support does
not establish Adreno execution.

| Repository fact | Execution boundary |
| --- | --- |
| [Rust release lock](../config/prism-inference.lock.json): Prism 0.1.32, source `23e24e444a821d4426f65d56c6cf874cf568b0d9` | CPU inference and the separate X2 Hexagon policy; no admitted Adreno inference |
| [Assessment lock](../config/swift-assessment.lock.json): 6.71 GiB PQ2, publisher Prism `73bfe6cc485554822f80c473a3c0c4f25e80f41b`, `win-cpu-arm64` | Explicit publisher rollback remains a CPU reference |
| [Publisher baseline build](../tools/bonsai/src/build_tools/engine.rs) and [overlay build][overlay] disable Vulkan/OpenCL | These publisher CPU recipes do not produce a GPU runtime |
| Publisher [completion](../src/cpu-policy/src/lib.rs), [server](../tools/bonsai/src/server.rs) and [broker](../src/broker/src/launch_policy.rs) use `--gpu-layers 0` / `--device none` | Drivers do not activate publisher GPU offload; its server policy also disables KV/operation offload |
| Rust [Doctor](hardware-capabilities.md#host-inventory) reports GPU identity/driver | Inventory does not establish loader/ICD, API features, kernels or budgets |

[OpenCL access and source coverage](#opencl-access-and-source-audit) remain unqualified for Swift;
X2 prebuilt-kernel claims do not qualify X1-85. [DirectML][directml]/Windows ML needs faithful
model/runtime integration rather than a GGUF switch. Conversion/custom DX12 is a separate route
requiring a measured limitation; Hexagon discovery is independent of Vulkan.

The recorded X1 device is Adreno X1-85 in X1E-78-100. X2 variants, including X2-90, have independent
capability/driver and graph evidence. Family names and advertised APIs do not prove subgroup,
integer-dot, cooperative-matrix or allocation support. Native ARM64 executable/loader/ICD/dependency
identities, enabled extensions, queues, selected device and shader build tools are separate facts;
software adapters and emulated inference are outside this native-device contract.

Feature evidence includes buffer/allocation/workgroup limits, heaps/types, subgroup properties,
FP16/storage, integer-dot support **and** acceleration, cooperative-matrix shapes and memory-budget
availability. See [capability queries][vulkan-query], [subgroup properties][subgroups], [integer-dot
properties][integer-dot] and [memory budgets][memory-budget]. Missing budget telemetry requires
conservative admission. Shared-memory/UMA reports and 128 MB dedicated memory are not total-capacity
or allocation entitlements.

## Swift arithmetic and placement

Use the [placement/state contract](hexagon.md#packed-arithmetic-and-graph-state)
for complete graph ownership, transfers and synchronization. Fix token IDs,
settings, artifact, CPU oracle and numerical tolerances before comparison.
CPU PQ2/Q8_K, PTQ1/Q8_0 and odd-width PQ2/Q8_0 differ from the Vulkan integer-dot [Q8_1 shader
family][prism-matvec] ([PTQ1 shader][prism-ptq1]). Exact weights do not imply bit-identical logits
across activation arithmetic. Awkward shapes, supported strides, extremes/zeros and explicit
fallback/rejection belong to packed-kernel acceptance. Whole-model scope includes prefill/decode
logits and KV/GDN state, tokenizer/template/sampler, prefixes, edits, reset, cancellation and
recovery; no-LoRA and approved-adapter basis/forward execution are separate lanes.

The pinned [Qwen35 graph][prism-graph] and [Vulkan guards][prism-vulkan] disable row-indexed
recurrent reads and raw-gate fusion. The gathered-state/activated-gate route is the numerical
control. Swift's `[5120, 248320]` embedding has about 322.07 MiB PQ2 payload or 265.23 MiB PTQ1;
heap space alone does not prove descriptor limits or indexing can address it. Forced-limit overrides
are outside the observed capability contract. [LoRA sharing][lora-pool] admits CPU buffers only; its
2.5 MiB storage saving does not establish device allocations, basis or adapter execution.

Scalar nonflash attention rounds globally normalized probabilities to FP16 before multiplying V.
Ordinary online softmax changes that intermediate and therefore has a distinct numerical contract.
[Attention/locality and chunkwise GDN](inference-algorithms.md) also retain state/logit, adapter and
lifecycle requirements after any scheduling or reduction change. Approximation and speculation have
separate qualification scopes.

## Decomposed attention candidate

The ggml 1.0 [Adreno decomposition][ggml-adreno-decomposition] is a resource-pressure
reference for native Rust GPU work (`..\prism-inference\tools\adreno-vulkan`,
`bonsai-adreno-vulkan`), contrasting a bounded single-dispatch fused attention path
against three explicit stages: `QK` scores, globally normalized probabilities with
explicit binary16 rounding, and `probs * V` accumulation, while keeping Runner's pinned
release lock (`backend_ready: false`) unqualified for application serving pending hosted
CI, X1-85 evidence, and complete-model comparison.

### Resolved X2 probe access and local SPIR-V 1.6 pilot (`bonsai-adreno-vulkan`)

On Snapdragon X2 Elite Extreme (`X2E94100`), Runner's headless probe originally hit
`VK_ERROR_INCOMPATIBLE_DRIVER` and GDI32 lookup error `126` because `HEADLESS_MITIGATIONS`
disabled Win32k system calls (`1 << 28`), blocking WDDM/DXCore ICD initialization, and
`qcvkarm64xum.dll` (`32,157,248` B) exceeded the earlier `16 MiB` per-file hash cap.
`MitigationProfile::GraphicsProbe` (`OwnedProcess::start_graphics_probe`, allowlisted for
`host-gpu-query`, `host-opencl-query`, and `opencl-projection-worker`), dynamic `user32.dll`
loading in `error_dialog`, and raised `host_gpu` hash limits (`64 MiB`/`192 MiB`/`1500 ms`)
unblocked live X2-90 discovery via `bonsai-tools doctor --json '{"nativeGraphics":true}'`:
- Vulkan loader `1.3.301`, `Adreno(TM) X2-90 GPU` device API `1.4.327`, ICD
  `qcvkarm64xum.dll` (`32,157,248` B, SHA-256 `eab059ad149a0370e4097b98e027abc2161b426aa094c75b5f94ab546b332c59`),
  `max_compute_shared_memory_size = 32,768` B, `max_compute_work_group_invocations = 1,024`,
  `max_storage_buffer_range = 2,147,483,647` B, `integer_dot_4x8_packed_signed_accelerated = true`,
  `VK_KHR_cooperative_matrix` shapes + `VK_QCOM_cooperative_matrix_conversion`, and
  Qualcomm OpenCL `3.0 QUALCOMM build: 863.0` (`25,563,551,744` B global / `32,768` B local).

In the sibling `..\prism-inference` repository, `bonsai-adreno-vulkan` emits SPIR-V 1.6
(`0x0001_0600`, complete global `OpEntryPoint` interfaces) on Vulkan 1.3+ (core integer
dot product `OpSDot`) for `#39` (`VulkanPq2Executor` + `Qwen35` +
`CompositeProjectionExecutor` / `OwnedCompositeProjectionExecutor`) and `#40`
(`VulkanAttentionExecutor` with fused `total_seq <= 512` vs 3-stage decomposed /
single-CB pipelined execution), expanded in post-lock local candidate commits `6c0a3b5`,
`0489989`, `1bfdb25`, `7697d3c`, `af6274d`, `e8c96be`, and `3fafc84` with persistent host
buffer mappings, `256 MiB` `WeightArena` allocations, `vk::PipelineCache`, a resettable
transient descriptor pool, coalesced 64-row `PQ2_0`/`PTQ1_0` GEMV & 4-token GEMM, GPU
4-plane `Q8_K` quantization using Veltkamp-Dekker `exact_fdiv_dekker` (eliminating the
`1 ULP` Adreno `OpFDiv` residual; `OpLogicalOr = 166` opcode fix), fused GPU FFN chain
(`gate/up + SwiGLU + Hadamard + down + residual` + rank-1 LoRA side-branch), zero-scale
`f16` block handling, full `Pq2ProjectionExecutor` FFN/tile forwarding on composite
executors, tiled verification admission for `supports_tile_tokens()` executors, and
explicit server `--placement vulkan|composite` with strict readiness and `8 GiB` weight
prewarm:
- **Live X2-90 parity (`5/5` unit tests + `9/9` non-ignored live tests)**: `PQ2_0 x Q8_K`
  (`fused=false`, batched `tokens=4`, zero-scale blocks, and `5120x17408`), `PQ2_0 x Q8_0`,
  `PTQ1_0 x Q8_0` (`pinned=false/true`), `PTQ1_0 x Q8_K`, GPU 4-plane `Q8_K` quantization,
  direct `F32` projections (`0 ULP` bit-identical to CPU), fused `F32` FFN chain + Hadamard +
  LoRA (`<= 1e-4`), 8-token tiled `Qwen35` decode/checkpoint/rollback, and
  `CompositeProjectionExecutor` overflow routing all pass against CPU.
- **Empirical compiler/precision findings on X2-90**: `fused=true` emits `GLSL.std.450`
  `Fma` (`NoContraction`), which Adreno's shader compiler lowers to hardware `mad.f32`
  (2-rounding, `<= 1 ULP` difference vs CPU `f32::mul_add`, `OpSDot == Portable` at `0 ULP`;
  gated separately from `fused=false`, which forced replacing Markstein FMA division with
  non-FMA Veltkamp-Dekker splitting in `exact_fdiv_dekker`); attention `round_f16` uses an
  explicit bitwise IEEE-754 binary16 shift/add + `GLSLstd450UnpackHalf2x16` (`62`) rather
  than `packHalf2x16` + `unpackHalf2x16` because Adreno QGLC peephole-folded the round-trip
  back to unrounded `f32`.
- **Bounded AC-logged X2E94100 trial (`3fafc84`; predeclared `180 s` ceiling overrun at
  `282.29 s` total because `bench_swift_arm_b_vulkan_seq` alone took `265.48 s`, stopped early
  before Arms C/D without retrying; every before/after `Win32_Battery` snapshot recorded
  `BatteryStatus = 2` AC, charge `43% -> 45%`, `EnergySaverStatus = Disabled`)**:
  - **Kernel micro-timings (`adreno_live`, `5.55 s` across all 4 arms)**:
    - `ARM1` (`PQ2_0 x Q8_K`, `5120x17408`): `OpSDot` `8.325 ms` median (`8.005 ms` min)
      vs portable `8.993 ms` (single-thread CPU scalar oracle `48.846 ms`).
    - `ARM2` (`PTQ1_0 x Q8_0` pinned, `5120x17408`): `OpSDot` `15.551 ms` median
      (`8.014 ms` min) vs portable `8.549 ms` (`1.525 ms` min; single-thread CPU scalar
      oracle `62.175 ms`).
    - `ARM3` (Attention `Q=1, H=24, KV=4, D=256`): `seq=64` fused `8.393 ms` /
      decomposed `8.529 ms`; `seq=512` fused `8.940 ms` / decomposed `8.702 ms`;
      `seq=2048` auto decomposed `15.275 ms`.
    - `ARM4` (coalesced `5120x17408` + 3-matrix `5120->17408->5120` FFN chain, `3` tensors =
      `71,024,640` B prewarmed in `24.676 ms`): `gemv1` `12.328 ms` (`2.748 ms` min),
      `gemm4` `8.199 ms` (`2.243 ms` min), `ffn_chain1` `9.419 ms` (`1.388 ms` min),
      `ffn_chain4` `12.375 ms` (`8.526 ms` min).
  - **Full-model `Swift-Bonsai-2-PQ2_0.gguf` losing/no-go verdict (`context=2048`, FP16 KV,
    12 row workers, seed `42`, thinking off, `72` prompt tokens, `16` generated tokens)**:
    - **CPU Arm A** (8-token tiled NEON `I8MM`/`SDOT` prefill): `load 4,808.1 ms`, prefill
      `3,384.46 ms` (`21.27 tok/s`), decode `1,469.09 ms` (`10.21 tok/s`, `15` evals), total
      `5,010.70 ms`, text `"1. **Memory Bandwidth**: Coalesced GPU weight layouts (transposed"`.
    - **Vulkan Arm B** (sequential, `401` resident tensors = `6,799,564,800` B prewarmed in
      `1,878.5 ms`): `load 4,945.0 ms`, prefill `214,684.18 ms` (`0.34 tok/s`), decode
      `40,097.79 ms` (`0.37 tok/s`, `15` evals), total `257,293.49 ms`, text
      `"1. **Memory Bandwidth**: Coalesced GPU weight layouts (swizz"` (matching tokens 1–13,
      with a near-tie argmax flip at token 14 within FP32 `SwiGLU`/`Hadamard` tolerance).
  - **Root cause and gating verdict (do not promote; keep `backend_ready: false`)**: Even
    with persistent scratch buffers and `1.388 ms` minimum GPU FFN-chain execution,
    per-projection synchronous `vkQueueSubmit` + `vkWaitForFences` incurs `~8–10 ms`
    WDDM/Vulkan submission round-trip latency across `~256` submissions/token (`64` layers),
    making sequential end-to-end Vulkan `~28x` slower on decode and `~63x` slower on prefill
    than 12-worker Oryon CPU. Moving norms, RoPE, attention/GDN, and projections into
    whole-layer / whole-graph single-submission command buffers is required before Adreno
    Vulkan can compete with multi-worker Oryon CPU.

For the first eligible Swift shape, compare a bounded fused candidate with three
explicit stages: QK scores, globally normalized probabilities, and probability/V
accumulation. Keep Prism's existing score and value reduction order, causal masks,
grouped-head ownership, current-token F32 K/V and selected stored-cache precision.
The normalization stage must round probabilities to FP16 before V. A tiled online
softmax that changes that boundary is a separate numerical profile, even if its
real-arithmetic formula is equivalent.

Declare token/head tiles and checked byte counts before allocation. Charge score
storage, rounded probabilities, reduction scratch, descriptors, cache conversion
and transfers as well as weights/KV. Device capacity alone cannot admit them.
Select the smallest justified tile from measured kernel resource limits; a phone
or another Adreno's tile size is not a Windows X1/X2 entitlement. Compare total
dispatch, synchronization, scratch and request cost, including cold compilation.

Each stage consumes only completed predecessors. Queue events must own the
producer/consumer dependency; incidental graph splits are not synchronization.
Retain all asynchronously accessed buffers until terminal completion or confirmed
worker exit, including cancelled/failed stages. Cache keys must bind complete
shape, stride, precision, kernel and device/driver identity.

The first controls must identify the path actually executed, compare each stage
and complete logits/state to independent CPU arithmetic, and exercise odd tiles,
causal boundaries, long sums, changed shapes and cancellation between stages.
Moving packet boundaries or adding unused work must not silently change arithmetic
or upload unread inputs. Adapters, native policy access and complete-request
benefit retain their separate gates. Runner's pinned release lock introduces no Adreno GPU
serving path yet; local Prism prototype work in `bonsai-adreno-vulkan` awaits hosted CI,
physical X1-85 verification, lock promotion and complete-request benefit in PLAN.

## Resources, failure and acceptance

Follow [profile eligibility](hardware-capabilities.md#automatic-profile-selection),
[accelerator storage/lifetimes](hardware-capabilities.md#accelerator-storage-and-lifetime)
and [measurement rules](performance-expectations.md#measurement-and-comparison).
Apply cache identity/invalidation to pipelines, descriptors and allocations.
Compare CPU-only, partial and maximum offload for both phases, charging cold
compilation/repack, submission/waits, transfers, state migration and duplicates.

GPU attribution separates pipeline first use, queue/kernel timestamps,
submission/waits and critical-path traces, including faults, evictions and
sustained failures. Overlapping durations cannot be summed; utilization is not
energy. [GPUView][gpuview] requires native ARM64 tooling. Native diagnostics use
absolute system-temp lanes and inherited Windows system/profile variables.
Moving source links are leads, not build pins. Serving admission, async storage and
DCVS/power-scaling gates from the August–November 2026 screening (`#39`–`#44` in the
[screening ledger](research-register.md#august-to-november-2026-screening-ledger)) remain
behind hosted CI, physical X1-85 evidence, pinned lock promotion, and matched host/device
resource accounting even though X2 probe access and the local `bonsai-adreno-vulkan`
SPIR-V 1.6 pilot are live-verified on X2-90.

## Cross-device research limits

Attribute CPU preparation, GPU service, shared-DRAM traffic and polling/wake-ups separately: faster
kernels may leave host gaps unchanged. Residency can increase register pressure and rendering/audio
contention; include frame delivery and speech underruns even when TPS is stable. Phone telemetry
and power controls provide no Windows authority, saving or launch prerequisite.

## OpenCL access and source audit

The pinned public OpenCL backend has **no reusable Swift PQ2/PTQ1 packed projection**: neither
format appears in `GGML_OP_MUL_MAT` eligibility, and PQ2/PTQ1/Hadamard implementations are absent.
Generic Q1/Q4/Q8 or an installed compiler cannot supply them. Public kernels remain possible;
this information-query/source audit selects no profile and is independent of Vulkan qualification.

The ten-second query
uses a 512 MiB one-process Job and 1 MiB per-output-stream bounds, creating no context, queue,
program, kernel or device allocation.

### Reproducible native query

`bonsai-tools host-opencl --request <absolute-empty-object.json>` (or `doctor --json '{"nativeGraphics":true}'`)
launches the ARM64 child with `MitigationProfile::GraphicsProbe` (`graphics-probe-worker`,
retaining all non-Win32k headless mitigations while permitting WDDM/DXCore ICD queries). It
loads only the absolute Windows system `OpenCL.dll`, restricting initial dependency search
to that directory and System32. `OCL_*`, `OPENCL_*` and `CL_*` environment overrides reject. Loader/ICD calls still execute installed native code and remain an
operator trust boundary. The internal `host-opencl-query` command has no intrinsic driver timeout;
callers must use the owning wrapper.

The query bounds eight platforms, sixteen total GPU devices, 16 KiB per native string and 512
extension names per device. Malformed UTF-8, missing terminators, invalid native sizes, non-GPU
handles and excessive counts reject. CPU devices are not requested. A successful empty enumeration
is observed, while load, driver, framing or deadline failure returns unknown and retains bounded
stderr. There is no retry with a weaker process policy.

The separate `runner-native-opencl-observations` report records platform/device, driver and OpenCL C
versions, compiler/device availability, address width, extensions and capacity limits. The loader
size/SHA-256 is a bounded file observation after querying, not proof that exactly those bytes or
every dependency were executed. DLL signatures, ICD dependencies and provenance remain unverified.
Global/local memory and maximum-allocation capacities are not live free-memory budgets and cannot
change admission. `backend_ready` stays false.

`bonsai-tools doctor --json '{"nativeGraphics":true}'` adds this report as
`NativeOpenClObservations`.
It neither builds tools nor changes the existing CPU compatibility digest or launch policy. The
native query follows the Khronos
[platform](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetPlatformIDs.html),
[GPU
enumeration](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetDeviceIDs.html)
and
[device-information](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetDeviceInfo.html)
contracts.

### Immutable public source audit

The audit uses Runner's existing C++ runtime revision `73bfe6cc485554822f80c473a3c0c4f25e80f41b` in
`PrismML-Eng/llama.cpp`. The immutable links below identify the reviewed files; this reference code
is not an application dependency.

| Checked source | Result |
| --- | --- |
| [OpenCL dispatch](https://github.com/PrismML-Eng/llama.cpp/blob/73bfe6cc485554822f80c473a3c0c4f25e80f41b/ggml/src/ggml-opencl/ggml-opencl.cpp#L7734) | Packed PQ2/PTQ1 projections are ineligible; the limited FP32 GDN and convolution branches do not establish complete Swift coverage. |
| [Build recipe](https://github.com/PrismML-Eng/llama.cpp/blob/73bfe6cc485554822f80c473a3c0c4f25e80f41b/ggml/src/ggml-opencl/CMakeLists.txt) | The external upstream recipe embeds kernels using an OpenCL SDK and a Python generator. Runner must replace any adopted generation step with Rust; this recipe is not an approved project build path. No accelerated Runner binary is built in this screen. |
| [Backend documentation](https://github.com/PrismML-Eng/llama.cpp/blob/73bfe6cc485554822f80c473a3c0c4f25e80f41b/docs/backend/OPENCL.md) | Windows ARM64 and X2-90 are listed, but optional Qualcomm prebuilt kernels are a separate manually provisioned library for selected other formats. They are excluded from this public-source comparison. |

The publisher rollback remains the CPU archive in `config/swift-assessment.lock.json`; the
application default uses the separate Rust release lock.

Local Prism `bonsai-hexagon-runtime` also provides `plan_three_processor_placement`
(`HeterogeneousPlacementPlan`, `AdrenoQ2KernelKind::WgslSubgroupShuffle` /
`OpenClSubgroupBlockRead`) and `plan_x1_v73_three_processor_placement`
(`X1_V73_MAX_NPU_LAYERS = 27`), partitioning `blk.0..=blk.26` (`168` `PQ2_0`
projections, `2,888,663,040` bytes) onto Hexagon NPU `native-q2` and
`blk.27..=blk.63` (`232` `PQ2_0` projections, `3,967,856,640` bytes) onto Adreno
X2-90 / X1-85 `Q2_0+FWHT` while keeping GDN/attention state and `lm_head` on CPU,
alongside `CompositeProjectionExecutor`, `CompositeTelemetry`, and
`HybridPrefillCoordinator`. In the Runner repository,
`src/windows-process/src/native/opencl/projection.rs` adds
`AdrenoFamily::{X1_85, X2_90}` and pure OpenCL C 1.2 integer `OpenClKernelKind`
variants (`BlockDots`, `Gemv4Row`, `Gemv8Row`, `Prefill4Col`, `FwhtLocal1024`)
via `opencl_pq2_kernel_source_for` and `opencl_pq2_projection_with_kernel`.
These are source-level placements, executors, and kernel sources, not physical
X1/X2 Adreno hardware parity or qualified application serving admission.

The opt-in [standalone original-PQ2 OpenCL pilot](adreno-pilot.md) implements one
real Swift FFN-gate shape with exact per-128 integer dots and CPU ordered-FMA
reduction, an independent scalar control and unchanged-policy child supervision.
It has not established device execution or a qualified backend.

The [Prism pilot contract](../PLAN.md#adreno-vulkan) requires one real Swift packed-projection shape
against scalar CPU arithmetic and proven unchanged-policy worker access before execution. New
kernels require reviewed builds, arithmetic/device and partial-placement transfer/state/fallback
evidence; changed formats and whole-model expansion are outside this screen. Neither query result
resolves Vulkan access or establishes speed/quality. Model/kernel comparisons and full suites remain
[CI-only](validation.md).

[prism-audit]: https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md
[directml]: https://learn.microsoft.com/en-us/windows/ai/directml/dml
[vulkan-query]: https://docs.vulkan.org/guide/latest/querying_extensions_features.html
[subgroups]: https://docs.vulkan.org/guide/latest/subgroups.html
[integer-dot]: https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderIntegerDotProductProperties.html
[memory-budget]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_memory_budget.html
[gpuview]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/using-gpuview
[prism-graph]: https://github.com/PrismML-Eng/llama.cpp/blob/73bfe6cc485554822f80c473a3c0c4f25e80f41b/src/models/qwen35.cpp
[prism-vulkan]: https://github.com/PrismML-Eng/llama.cpp/blob/73bfe6cc485554822f80c473a3c0c4f25e80f41b/ggml/src/ggml-vulkan/ggml-vulkan.cpp
[prism-matvec]: https://github.com/PrismML-Eng/llama.cpp/blob/73bfe6cc485554822f80c473a3c0c4f25e80f41b/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp
[prism-ptq1]: https://github.com/PrismML-Eng/llama.cpp/blob/73bfe6cc485554822f80c473a3c0c4f25e80f41b/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_ptq1_0.comp
[overlay]: https://github.com/symbiogenesis/prism-inference/blob/main/scripts/Build-PrismEngine.ps1
[lora-pool]: https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/src/pool.rs
[ggml-adreno-decomposition]: https://github.com/ggml-org/llama.cpp/commit/e1f7593c32e8dd782fc7addb908d0911260511ae
