# Qwen reference model

[runtime.lock.json](../config/runtime.lock.json) pins Qwen as a reproducible,
development-only reference. [Swift](swift-memory.md) remains the implementation
priority; neither profile has daily-service acceptance.

The reference is **Qwen3.8-27B UD-Q3_K_XL**, revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`, with **Gemma 4 12B IT QAT Q4_0** as a
literature-nominated fallback. Published evidence gives moderate confidence in the
base-model creative preference, but low confidence in this quantization and laptop
comfort. There is no local candidate benchmark. Generic model provisioning without
a lock override selects Qwen; native chat separately selects Swift and requires its
automatic [in-app setup](development-guide.md#native-chat-app). The Gemma
fallback is not selected.

## Selection boundary

Published creative-writing scores motivated the Qwen reference, but they evaluate
different serving representations and often use LLM judges. They establish no
creative-quality ranking for the exact GGUF packs on this laptop. Swift's
reasoning benchmarks likewise do not establish prose quality. The Gemma fallback
is unselected; there is no local exact-artifact comparison or comfort acceptance.

Compare complete held-out tasks with exact model, template, sampler, thinking,
quantization and termination identities. Keep human prose quality, instruction
following, memory, useful-answer latency and thermal comfort separate. Missing
benchmark entries mean missing evidence, not zero scores.

## Source-only compatibility and memory sanity check

The [verified metadata](../config/model.metadata.json) describes 866 tensors across 14
types, architecture `qwen35`, 65 blocks including one MTP block, four KV heads and
256-dimensional K/V. The main graph has 16 full-attention and 48 recurrent blocks. Full
size/SHA-256, dimensions and embedded-template identity are verified; actual CPU loading
remains pending.

The reference runtime is llama.cpp **b11840**, commit
`36e36e63a54fc8f85b95e9c499a022b597e478dd`. This daily release is an upstream
prerelease. Its [Qwen
graph](https://github.com/ggml-org/llama.cpp/blob/36e36e63a54fc8f85b95e9c499a022b597e478dd/src/models/qwen35.cpp)
handles MTP separately; its [CPU
backend](https://github.com/ggml-org/llama.cpp/tree/36e36e63a54fc8f85b95e9c499a022b597e478dd/ggml/src/ggml-cpu)
supports the artifact's tensor types, SSM convolution and Gated Delta Net. The
[release](https://github.com/ggml-org/llama.cpp/releases/tag/b11840) includes a native
Windows ARM64 CPU archive. This establishes source compatibility only.

For one sequence, 8,192 context tokens, FP16 K/V, speculation off, no prompt cache and
no retained checkpoints:

```text
weights                            13,146,393,504 bytes = 12.2435 GiB
KV = 8192 × 16 × 4 × 256 × 2 × 2       536,870,912 bytes = 0.5000 GiB
recurrent state (F32, including conv)   156,893,184 bytes = 0.1461 GiB
subtotal                           13,840,157,600 bytes = 12.8897 GiB
remaining under 18 GiB               5,487,195,232 bytes = 5.1103 GiB
```

Batch 256 / microbatch 128 also needs scratch, graph buffers and allocation overhead;
the subtotal is not peak residency. Reuse/checkpoints stay disabled because server
defaults exceed this estimate. The immutable file retains its MTP block, but speculation
and projector/vision allocations are not admitted. The guarded CPU check requires 22 GiB
available and refuses launch when that admission is unavailable; the profile remains
development-only and chat-launch-disabled.

Preserve and verify the embedded template. The [pinned quantizer
instructions](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/README.md)
document thinking control. A future non-thinking creative candidate uses temperature
0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5 and repetition penalty 1; that
does not establish that a lower-reasoning switch works.

## Conditional future Bonsai MoE design

Revisit this design only after Qwen 4 is released and a corresponding Bonsai
upgrade is available. The expected MoE architecture remains an assumption until
the published weights, tokenizer, quantization and adapter identities establish
the actual graph. This design authorizes no current implementation or device
trial; [PLAN](../PLAN.md#future-qwen-moe-migration) owns the conditional tasks.

The pinned [Colibri source](https://github.com/JustVugg/colibri/tree/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6)
supplies design references, not Snapdragon qualification:

- [Resident/expert capacity accounting](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/resource_plan.py)
  reserves dense/shared and router weights, context/recurrent state, workspace,
  metadata, staging and host headroom before sizing expert caches. Unequal experts
  use each layer's maximum resident size. Count retained host/device copies and
  Snapdragon's shared physical CPU/GPU/NPU RAM once, including any duplicated
  allocations within that pool.
  Keep arithmetic checked and unknown capacity explicit. Capacity coverage does
  not establish cache hit rate or admission.
- Hot/warm/cold placement may keep frequently used experts on an accelerator,
  warm experts in RAM and cold experts in verified local storage. Preserve every
  weight, precision and router decision. Admit only tiers justified by a measured
  batch-one bottleneck, including integrated-device copies, transfer and full
  restoration costs. Bound metadata, staging, I/O and cancellation/teardown;
  persistent usage traces require an explicit storage/privacy policy.
- [Download-space preflight](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/setup_catalog.py)
  checks the actual destination's additional locked bytes after a valid partial
  download, then rechecks the full requirement before an ignored-Range restart
  can truncate that partial. Retain resumable data on failure, explicit unknown
  space and immutable verification, with separate staging/extraction accounting.
  A free-space snapshot reserves no storage.

Any selected upgrade needs native Rust support, complete state/resource admission,
quality, cancellation, Stop/Retry, cleanup, native application acceptance and
measured benefit before profile selection. Preserve explicit Swift rollback and
the [native UI boundary](ui-architecture.md). Colibri CUDA/Vulkan results establish
neither Adreno/Hexagon behavior nor local disk latency; use the existing
[qualification budgets and power observations](validation.md).

## Opt-in Qwen3.8-27B NariQ trellis representations

[nariq-qwen38.lock.json](../config/nariq-qwen38.lock.json) pins the Apache-2.0
[`nari-labs/Qwen3.8-27B-NariQ-GGUF`](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF)
repository at revision `17886395feb8a752b5ca5f02ad0c25543950915e` as an opt-in
native CPU representation (`--nariq 2.5bpw|3.0bpw|2.0bpw`), keeping
`Swift-Bonsai-2-PQ2_0` + Orca (+ DaisyRose) unchanged as the default:

| Variant | Filename | Bytes | SHA-256 | Tensors | Notes |
| --- | --- | --- | --- | --- | --- |
| `2.5bpw` | `Qwen3.8-27B-NariQ-2.5bpw.gguf` | `8,443,119,712` | `2445372e6de20620b940fbfab53aa718af9ba741f202da30cd2f02b5df78d05c` | `2,710` | `lm_head_bands = 16`, unbanded `token_embd.weight`, `9 GiB` ceiling |
| `3.0bpw` | `Qwen3.8-27B-NariQ-3.0bpw.gguf` | `10,123,626,592` | `1eaad934aa05579b8eba47ed9b26450392f9875fc2354efe8cfa8b29be8f4419` | `2,710` | `lm_head_bands = 16`, unbanded `token_embd.weight`, `11 GiB` ceiling |
| `2.0bpw` | `Qwen3.8-27B-NariQ-2.0bpw.gguf` | `6,753,954,304` | `3a9e4955329dba8ba0be5509a9552220e5ccdae2f66bcc72c21c4580cd49bbb2` | `2,771` | `lm_head_bands = 16`, `embed_bands = 16` (`token_embd.vocab_gather`), `8 GiB` ceiling |

The lock also records `Qwen3.8-27B-NariQ-mmproj-BF16.gguf` (`931,145,760` bytes,
SHA-256 `8832425f593068d3278e08eda4c1591072c7a2d8dc90f21e4f86394e4b2eb002`) as a
reserved text-excluded projector (runtime vision remains disabled for NariQ).
Native CPU NariQ stacks with `MultiRankAdapter` (DaisyRose) and Prism's
`K4/V4` (`rsq4`) KV cache, excludes Orca `abliterate_lora`, Vulkan/Hexagon
offload, and uncalibrated fallback sizing, and remains broker-gated on
`NARIQ_TRELLIS_RELEASE = None` until a compatible Prism release bundle and
matched X1/X2 qualification complete.

## Stop and reopen conditions

Reject file/hash/header mismatches and unsupported graphs or types. Limit the first
compatibility investigation to one working day and one narrow, upstream-supported
repair. That investigation window does not extend the aggregate owner-device
allowances in [validation](validation.md). Custom quantization and backend kernels
are outside this reference bring-up.

Reopen the reference choice if that route cannot produce correct bounded output, exceeds
18 GiB at 8K with reuse off, or later fails creative, interactive-wait or owner-accepted
heat/fan requirements after bounded tuning. Record the failure and a fresh literature
decision before provisioning the fallback. Keep candidate benchmarks on CI.

See [PLAN.md](../PLAN.md) for qualification, the [development
guide](development-guide.md) for commands and [licensing](licensing.md) for
redistribution.
