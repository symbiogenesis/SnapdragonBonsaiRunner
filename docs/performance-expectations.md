# Performance measurement

The current development profile is defined by the [runtime lock](../config/prism-inference.lock.json),
[inference defaults](inference-defaults.md) and [memory policy](swift-memory.md#memory-selected-serving-context).
There is no qualified sustained-performance figure for this profile.
[PLAN.md](../PLAN.md) tracks acceptance. Older runtimes' short-request observations
cannot predict its speed, memory use, creative quality or effective accelerator placement.

## Current Swift evidence

Loading, short decode and synthetic controls do not establish a useful completed
answer, full-context behavior or repeated performance benefit. A release or repin
also supplies no model timing. Keep model-free, actual-model, resource, quality
and device acceptance separate under [validation](validation.md).

Publisher measurements for base Bonsai, Qwen or other hardware have different
weights, arithmetic, workloads and observation boundaries. They are not Runner
or Swift measurements and cannot serve as a Snapdragon forecast.

## Physical ceilings and remaining optimization headroom

Because sustained profiles remain unqualified until owner-operated CI and
matched-power device captures complete, remaining headroom is tracked by
separating **hard physical and information-theoretic limits** from **optimizations
currently disabled or awaiting qualification in Runner's default serving policy**.
Roofline ratios, byte ledgers ([swift-memory.md](swift-memory.md),
[hexagon-swift-traffic-ledger.json](evidence/hexagon-swift-traffic-ledger.json))
and power-ineligible smoke captures ([lora-weight-edit.md](evidence/lora-weight-edit.md),
where AC disconnected mid-trial) bound regimes and prioritize [PLAN.md](../PLAN.md)
work; they are not qualified baselines or performance promises.

### Hard physical and security floors

- **Batch-1 decode DRAM wall ([T2](research-register.md#t2-traffic-arithmetic-and-the-whole-request-ceiling)):**
  Dense single-token decode streams at least `~7.02 GB/token` for `PQ2_0` (`~6.80 GB`
  packed projections + `0.06 GB` F32/BF16 + `~0.16 GB` FP32 GDN state) or
  `~5.75 GB/token` for `PTQ1_0` (`-18.1%` traffic). Against unified LPDDR5x peak
  bandwidth (`~135 GB/s` on X1, `~228 GB/s` on X2), wire-speed batch-1 ceilings are
  `19.2–23.5 tok/s` on X1 and `32.5–39.6 tok/s` on X2 (`~15–19 tok/s` and
  `~26–32 tok/s` at ~78%–85% sustainable SoC memory-controller efficiency).
  Exceeding those ceilings requires amortizing weight reads across multiple tokens
  via [speculative verification](inference-algorithms.md#hybrid-cache-reuse-and-speculative-state)
  or omitting weight groups under [W8](weight-space-programming.md#selected-w8-investigation-recover-omitted-ffn-computation).
- **Lossless independent-block weight packing ([T1](research-register.md#t1-packing-limits-and-the-difference-between-sparse-and-small)):**
  128 independent trits require `ceil(128 log2(3)) = 203` bits (`26` code bytes)
  plus the 2-byte scale = `28` bytes. `PTQ1_0` (`5.95 GB`, `-17.5%` vs `7.21 GB`
  `PQ2_0`) achieves this bound exactly, and Swift's `32.76154%` zero rate falls
  below BITCOS's `37.5%` break-even: `0%` lossless independent-block headroom
  remains past `PTQ1_0`.
- **Startup byte-verification floor ([Mapping and startup](swift-memory.md#mapping-and-startup)):**
  Runner forbids trusting cached digests or timestamps without hashing file bytes on
  launch, setting lossless floors at `1×` cold NVMe sequential throughput and `1×`
  warm multicore hardware `SHA256` RAM bandwidth.

### Serving baseline versus remaining levers

| Dimension | Current default serving policy (Runner `main` + `0.1.30`, plus post-`0.1.30` Prism `main` source pending repin) | Remaining lossless / exact-contract headroom (in source or scoped in [PLAN.md](../PLAN.md)) | Speculative / approximate / profile-gated ceiling |
| --- | --- | --- | --- |
| **Decode (`tok/s`)** | `PQ2_0`, **8 CPU row workers on X1 and X2** (Runner `main`; `4` when unknown), batch-1, LoRA disabling fused NPU FFN under `0.1.30`. Post-`0.1.30` Prism `main` (pending repin) adds spin-then-park `NativeRowPool`, bit-exact `6:1` grouped GQA packed-KV decode/tile attention, and `adapter_affects_any` `external_ffn_swiglu` LoRA unlocking across all 64 layers | Repinning past `0.1.30`, full `ffn_down` LoRA side-branch VTCM fusion ([hexagon.md](hexagon.md#resident-packets-and-prefill)), tuned 6–12 AC workers + `SDOT` + 16-row tiles (`#23`) + `ciot` T2P + fused GDN (`#27`), `PTQ1_0` (`P2`–`P5`, `-18%` traffic), composite Hexagon `native-q2` + Adreno (`NPU-1`–`NPU-8`) approaching the LPDDR5x ceiling | Tiled verification (`#30`–`#34` `SuffixDrafter` on structured/code text; `#35` `DFlash2` learned ternary drafter) + `W8` 3-group FFN omission breaking the batch-1 DRAM wall |
| **Uncached `TTFT`** | Sequential `N = 1` prompt ingestion by default; explicit `--cpu-prefill-tokens 8\|16\|32` + `--cpu-prefill-scratch-bytes` (`:cpu-tiled-ffn-v1:{tokens}`) wired end-to-end in Runner `main` (omitted by default; skips Hexagon) | Qualifying/defaulting `--cpu-prefill-tokens`, exposing `gdn-wy-f32-v1` chunkwise GDN, decoupling CPU tiled prefill from Hexagon decode (`mixed-phase handoff`), and `I8MM` (`SMMLA`) tiles (`16×–32×` fewer weight reads) | Signed `N > 1` Hexagon HMX matrix solver + pipelined DMA/chunkwise GDN (`#41`, `#42`) + long-context `TokenGate` (`W8.3`) |
| **Multi-turn `TTFT`** | `--prompt-reuse disabled` (recomputes full chat history every turn) | Qualifying/enabling `--prompt-reuse latest` for linear chat prefixes, plus bounded sparse checkpoints (`<= 8` slots) across edited/switched branches | Combined prefix checkpoint hit + tiled/HMX tail ingestion |
| **Model preparation** | Runner `main` uses mapped `PrefetchVirtualMemory` (`16 MiB`) + cancellable `4 MiB` CNG SHA-256; post-`0.1.30` Prism `main` adds `MappedReadOnlyFile`/`MappedSlice` zero-copy GGUF + 8-worker validation (still `2 × 7.21 GB` total across Runner + Prism + per-launch NPU repack) | Post-`0.1.30` repin + single-pass verified-handle/section inheritance across `--broker-pipe` (`2× -> 1×`), lockfile-pinned parallel chunk `SHA256`, `PTQ1_0` (`-17.5%` bytes), digest-bound cached Hexagon arenas, `#16`/`#17` staging/direct I/O | Sub-second warm mapping could go lower only by weakening per-launch byte hashing (disallowed by security policy) |
| **Base RAM & admission** | `11.0 GiB` fixed admission (`+4 GiB` reserve), private CPU heap copies (`0.1.30`; zero-copy `MappedSlice` in post-`0.1.30` Prism `main`), `+2.69 GiB` duplicate host/NPU `blk.0..=26` weights | Post-`0.1.30` repin + `PTQ1_0` (`-1.173 GiB`), `#14` packing-aware `FIXED_RUNTIME_BYTES` calibration (`M1`–`M3`), `#18` selective tensor-range mapping / Job-commit credit (`~5.5–6.7 GiB` moved from private commit) + releasing CPU `blk.0..=26` spans after NPU arena seal | `W8.2` selective unloading of omitted Hadamard FFN groups (`~590–725 MiB` net after rank-16 recovery factors) |
| **Context (`tokens / GiB`)** | Dev `K4/V4` (`34,016` B/tok, charging `2×` full KV copies; `~4.0×` denser than `FP16`); `physical_cow_footprint_bytes` telemetry implemented in Runner `main` | Qualifying `K1` for production, `K4/V3` (`29,920` B/tok), and append-only CoW sealed-page KV admission (`live_kv_copies: 2 -> 1 + mutable tail`, up to `~4.1 GiB` physical saving at 256K) | Heterogeneous layer precision (`U3`), calibrated K-mean centering (`#21`), MLA (16 GQA layers) or `TokenGate` (`W8.3`) |
| **Speech (`TTS` / `ASR`)** | `Q8` CPU baseline (`S3` `SDOT`/`I8MM`, `F16` KV, `MappedFile`, spin-park pool, & dyadic `1->2->4->8` codec ramp in source) | Qualifying `S1`–`S3` Tier-2 CPU/streaming paths, warm `Voice` retention and Hexagon native `Q8` transport | `Q4` Talker weights and conditional ternary Talker profile |

## Measurement and comparison

Use the same observation boundary for both arms. Record request start, first output,
first visible/useful answer, last token, terminal framing and teardown separately.
Reasoning, buffered delivery and speculative bursts can separate those events.

For `N >= 2` output tokens over a positive observed interval, define
`r = (N - 1) / (t_last - t_first)`. Then
`last-token latency = TTFT + (N - 1) / r` is an identity at that boundary.
For one token, last-token latency equals TTFT. An arithmetic average of rates need
not preserve the identity. SSE events or buffered bursts are not token counts;
client-observed delivery rate is not isolated kernel decode throughput.

| Comparison requirement | Record |
| --- | --- |
| Paired repeated trials | Balance/reverse order; identify cold/warm/prefix-cache state. Hold model, adapter, template, sampler, context, output and numerical contracts fixed. Retain complete output or label truncation. |
| Actual placement | Bind effective CPU/GPU/NPU execution, device/SKU, driver, source/build and immutable artifacts. Requested acceleration or readiness alone does not prove offload. |
| Complete task | Charge initialization, loading, repack/compile, migration, prefill, decode, retries and revisions to the workload that pays them. Separate profiled attribution from unprofiled timing. |
| Memory | Separate persistent payload, temporary peak, private/Job commit, file-backed residency, device allocations and machine headroom. Charge scales, scratch, snapshots and retained high-precision copies. Never add overlapping counters. |
| Power and comfort | Record aggregate/per-arm AC, saver and charge observations; missing/mismatched conditions disqualify performance comparisons. Matching endpoints do not prove continuous stability or energy use. Measure useful-answer energy only with valid telemetry. Retain responsiveness, heat/fan and idle observations. |

Two short reversed pairs can supply a smoke comparison, not statistical or sustained
qualification. Stable TPS can coexist with worse desktop responsiveness under
shared memory/power contention. Include uncertainty and losing/no-go results;
use predeclared acceptance limits and [promotion criteria](validation.md#promotion-criteria).

Benchmarks and complete suites remain CI-only. Owner-device trials share one
predeclared allowance across setup, intake, loading, every arm, power, cleanup and
retention under [AGENTS.md](../AGENTS.md#owner-device-execution-budgets).
Do not retry, reopen a closed clock or split an infeasible matrix across repeated runs.

## Creative review packets

The [daily-use review packet](daily-use-review.md#creative-writing-packet) supplies
fixed original writing tasks, objective checks and independent prose/continuity
rubrics. Its [speech packet](daily-use-review.md#speech-listening-packet) defines
complete synthetic listening texts and sample/reviewer identities. These are
capture specifications; missing captures or human reviews remain pending.

The [Longform Writing Bench rubric organization](https://github.com/EQ-bench/longform-writing-bench/tree/main/data)
is a candidate source for independent creative review. Separate scene/chapter prose
quality from whole-narrative coherence: character consistency, chronology, world
rules and unresolved plot promises need the complete relevant story. Declare the
criteria, weights, negative criteria and review scope before comparing complete
held-out answers, including Thinking and Extra effort settings. Retain independent
human review and distinguish prose quality from token count or completion speed.

Adapted short packets are not leaderboard-equivalent. Its Python harness and
automatic external judging are outside this proposal; private prompts and prose
stay local unless the owner separately authorizes an external review. Source and
reuse limits live in the [speech and creative review references](research-register.md#speech-and-creative-review-references).
Existing trial budgets, per-arm power eligibility and evidence retention still
govern any selected capture.

## X1/X2 configuration evidence

[Hardware policy](hardware-capabilities.md#automatic-profile-selection) owns
selection, admission, phase handoff and qualified fallback/refusal. Family/core
counts, NPU TOPS and bandwidth bounds predict neither TPS nor energy: decode,
prefill and concurrent processors pay different bandwidth/preparation costs.

Compare each backend and phase combination at exact device/driver/runtime
identities. Shared CPU changes may follow the [cross-generation applicability rule](cpu-kernels.md#cross-generation-cpu-promotion),
without relabelling X2 measurements as X1 results. Family-specific tuned defaults,
accelerator rankings and complete device profiles retain independent acceptance.

## Q4 speech resource comparison

Q4 talker weights remain a separate experiment from the default Q8 speech profile.
Compare exact model/codec, voice reference, seed and thread settings with fresh
processes and complete clips. VoiceDesign and Base run sequentially; their active
weight reductions cannot be counted twice. Smaller files or lower short-fixture
working sets do not establish the cause, total-system memory savings or Snapdragon
benefit. Keep Job commit and mapped residency distinct and retain the physical reserve.

Measure voice bootstrap, warm talker prefill, residual-code steps, codec chunks
and playback. First generated PCM and first audible sound are separate endpoints.
Include chunk supply gaps/underruns, long-read memory and simultaneous chat/speech.
Average faster-than-real-time synthesis can still stall playback. Framing/ownership
checks cannot replace real-device audio or independent listening for timbre,
continuity, timing and prosody. See [speech controls](../tools/qwen-tts/README.md).

The Rust [ebur128 analyzer](https://github.com/sdroege/ebur128) is a candidate
review helper for integrated/short-term loudness, loudness range and true peak.
If loudness matching is used for listening, apply the declared gain only to
separately identified listening copies and record their relationship to the
original samples. Original generated PCM remains the authority for numerical,
streaming and continuity comparisons. Record unavailable measurements explicitly;
this proposal sets no loudness threshold and changes no serving PCM. Matched
loudness controls a listening confound, but establishes neither numerical equality
nor intelligibility, voice identity or audible quality. Follow the
[source and adoption limits](research-register.md#speech-and-creative-review-references)
before introducing the helper.
