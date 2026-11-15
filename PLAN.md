# Swift Bonsai 2 Runner TODO

This is the single backlog. [Technical docs](docs/README.md) own behavior,
contracts, commands and required evidence. Delete completed items and their
completion notes instead of marking them done or applying strikethrough.

Target native Windows ARM64 on **Snapdragon X Elite** (X1) and
**Snapdragon X2 Elite Extreme** (X2). Every hardware, kernel and deferred research
item below must serve those profiles. Qualify scalar and feature-disabled
fallbacks on ARM64; other CPU/GPU families and their platform-specific kernels
are outside this backlog. Prior-art references may supply general principles,
without adding another hardware target or qualification obligation.

Prioritize native daily use, then correctness, isolation, creative quality and
comfort, then measured optimization. Task IDs identify related work across docs.
Follow [delivery ownership](docs/development-guide.md#implementation-delivery),
[execution budgets](docs/validation.md) and
[promotion criteria](docs/validation.md#promotion-criteria). Local checks leave
hosted, device and human acceptance pending; push authorization and CI ownership
follow [AGENTS.md](AGENTS.md).

The [repository research catalog](docs/research-register.md#research-catalog)
records candidate sources, overlap and reuse limits. Its references are not
installed dependencies or qualified improvements; the scoped tasks below own
adoption decisions without displacing native daily-use acceptance.

## Publication and qualification prerequisites

- [ ] **L1 — Qualify the Rust/PowerShell tooling migration.** Have the owner
  qualify published source in Source checks/Foundation, migrated fixture
  inventories, exact native helper/cache handoffs and separate SDK/ABI controls.
  Include native CPU packet/evaluator and retained scalar replay identities;
  SDK, wire, Windows ABI and source-ledger controls; Rust icon/source-builder,
  release/publication and notice tools; native report/fixture admission; and
  prebuilt sandbox, acceptance and X2 owner clock/power/teardown handoffs.
  Preserve missing pre-bootstrap power as unknown and qualify exact cache hits,
  original aggregate deadlines, losing/no-go retention and independent replay.
  Include the expanded language guard, native PE closure and direct headless-build
  handoff; local checks do not establish hosted or actual-system acceptance.
  Public reconstruction and replay retain P1's schema-2 scalar-v2 identity.
- [ ] **C1 — Qualify current CI source.** Require owner-operated Source checks,
  Foundation and dependency-audit acceptance for the published startup,
  staged-index, ASR/UI fixture, workflow-consolidation and dependency corrections,
  syntax/selector guards, portable and planner compiler-cache recovery, portable
  environment fixtures and separate hosted/device scheduling. Qualify
  producer/check overlap, TTS-only routing, complete lane cache-input closure,
  separate build/check snapshots, coordinator-only fast bootstrap and Foundation's
  opt-in incremental Release compilation. Compare cold, unchanged and edited
  runs on matching hosted images, including cache transfer/storage costs; preserve
  every selected analyzer, test, reference and integration verdict. Include fresh
  advisory/yank checks for both workspace lockfiles and the SDK reader's patched
  archive parser. Verify
  cold/hit/input-change paths, failed-regression recovery and independent push handoffs.
  Verify per-test timing reports, shared UI font discovery, native process-tree
  fixtures and concurrent/virtual-time broker controls in their owning lanes.
  Include first-failure termination and explicit pending selections for local
  focused checks while preserving hosted ordinary-verdict coverage.
  Follow the [startup preflight](tools/ci/README.md#action-dependency-pins) before
  workflow changes and verify supported self-hosted runner versions.
  Include the [failure audit and latest ten-run follow-up](docs/evidence/ci-failure-audit.md), exact-source
  Release prerequisite/rerun admission, production-only lanes and same-run tool
  reuse; unrestricted main-push verdicts with unchanged scoped selection;
  immutable outgoing-tip push checks; public-input writer/identity handoff; and
  narrow projection dependency builds. Local focused checks leave hosted gates
  pending, including the historical linker-child identity that was not retained.
  Qualify the final snapshot-config regression and complete push adapter, preparation
  receipt/invalidation, packaged-shell and historical-tip/output-bound controls
  in owning CI; local focused controls do not establish complete acceptance. Keep
  helper preparation independent of redirected standard input and supervised in
  Rust within its declared allowance; explicit preparation retains external
  supervision. Qualify one-attempt full automatic preparation through an
  independently provisioned prebuilt maintenance executable when the recovery
  supervisor is unavailable or invalid: prerequisite/capability admission,
  archived committed preparation, both receipt checks/invalidation, native
  timeout and original-ref continuation within the original shared push allowance.
  Qualify one-attempt automatic checker recovery through
  the independently source/digest-bound native supervisor: missing/stale/corrupt
  checker receipts, array-valued receipt-field rejection, unchanged fast path,
  recovery-helper rejection, nonce-driven corrupted linked-output regeneration,
  target-specific recompilation with dependency/library reuse, legacy-cache
  compatibility without directory deletion or cache marking, offline incremental reuse,
  failed-build receipt invalidation, process-tree teardown
  and immutable-tip continuation within the original shared push allowance.
  Qualify trusted candidate-root adapter controls for both cache actions and
  source selection, including non-HEAD mutations and poisoned candidate fixtures;
  assertions and support remain bound to committed checking policy. Qualify
  failed-preparation receipt invalidation. Verify explicit target/output preparation
  against inherited Cargo target overrides before claiming prepared-helper admission.
  Include policy-only bootstrap closure, batched script scope/rejection controls
  and source-language scan newline diagnostics in owning CI acceptance.
  Qualify parallel formatter root/configuration/failure parity, managed AST
  traversal and pinned linter archive cache-hit/poisoning controls.
  Include portable maintenance consumer selection and budget/verdict fixtures,
  complete pinned public-ABI source intake before downloads, formatter-config
  and native-boundary Git-attribute push selection, and behavior-focused Clippy
  policy.
- [ ] Establish remote enforcement of basic sanity before updating `main`.
  Stage a trusted exact-commit candidate check before activating required-check
  rules, preserve reviewed direct-main promotion without routine pull requests,
  and restrict session administration/check-issuer credentials. Local hooks alone
  cannot enforce remote acceptance; the current repository has no active rules.
  See the [concrete rule/process requirements](docs/evidence/ci-failure-audit.md#process-corrections-and-remaining-limits).
- [ ] Qualify Runner's checked-tools producer, shared regression runner and
  native-acceptance caches: cold build, exact hit and relevant-input invalidation,
  fresh same-run packages, source/digest verification and every selected check.
- [ ] Complete owning-lane compiler/analyzer and focused regression acceptance
  for centralized UI bounded reads, atomic settings publication and transcript
  insertion and popup shortcut ownership; speech synthesis object-only intake;
  complete-event UTF-8 validation, HTTP/MIME whitespace, Unicode mandatory speech
  breaks, pinned ASR ASCII/decode paths and lossless startup-cache identities;
  native chat's test-only console attachment and captured Unicode speech/SSE,
  editor and Find test summaries;
  TTS numerical/GGUF primitives, checked projection layouts, packed-half F32
  operand/range/long-sum controls and invalid sampler distribution rejection;
  and tools bounded intake, selected nonblank startup
  diagnostics, PE probing, release/Hexagon validation, target-family evidence
  and startup admission, and Qwen-only reference-snapshot selection;
  native startup exit/descendant fixtures, transcript reflow/selection/anchors,
  and cancellable HTTPS raw reads with CONNECT tunnel passthrough.
  The Hexagon inventory alias correction and pinned-linker
  process-tree regression still need compiled owning-lane verification, including
  cold Cargo host/target links; the local direct diagnostic compile/self-test and
  Windows process-identity control passed.
- [ ] Complete owning-lane `source-checks` / `foundation` compiler, analyzer and
  focused regression acceptance for the August–November 2026 screening controls
  (`#8`, `#14`, `#45`–`#51` in the
  [screening ledger](docs/research-register.md#august-to-november-2026-screening-ledger)):
  broker queue-wait closure/`poll_peek` abandonment and 864-interleaving scheduler
  simulation, `json-candidate` single-pass error classification, UI `50 ms` TCP
  connect slices, `error_ui` frame-zero-clone copy action, Markdown prefix-layout
  retention without in-process UI parsing, incremental `stream.rs` cursor,
  cooperative `speech-tts` `CancelHook` entrypoints, `sampling.rs` `f64`/finite/greedy-argmax
  fixes, hardened GGUF validation and scalar-boundary multibyte whitespace tokenization,
  `bonsai-cpu-policy::context` itemized `SwiftExecutionShape` / `SwiftMemoryBreakdown`
  and `Rotated4FittedValues` (`16,896` B/tok), and `tools/bonsai` `calibration_metrics`,
  path-free `fitInputSha256` & omitted receipt paths, top-8 violation caps, single-pass
  startup path/64-hex redaction, `<= 256`-byte JSON parse errors and `evidence-archive`
  path guards. Audit `#1` (`ClampedL2`) is closed; pinned runtime/Runner source controls
  (`#2`–`#7`, `#9`–`#13`, `#15`, `#19`–`#20`, `#22`, `#26`, `#28`, `#30`–`#34`, `#36`)
  retain their compiled CI, actual-model and X1/X2 device gates below.
- [ ] Qualify compiler-cache recovery after failed analyzers/tests, immutable
  exact hits, cancelled/setup-failure skips and nonblocking optional saves.
  Qualify same-run publication recovery from partial drafts and ambiguous API
  responses, plus exact completed-release verification without writes; reject
  changed source, tag, metadata or asset identities.
- [ ] Qualify Prism's Foundation scheduling and source-package/Hexagon caches
  on matching hosted images, preserving complete correctness, mutation, packaging
  and independent ABI/SDK obligations.
- [ ] **R1 — Qualify the Runner release candidate with pinned Prism 0.1.30.**
  Complete actual-model, state, resource, quality and device acceptance using
  the source-bound package identities in the [distribution guide](docs/releases.md).
  Model-free release checks leave these development-runtime gates pending.
- [ ] Qualify the single-executable builder from a checkout and on a clean
  Windows ARM64 installation: double-click launch, visible native startup errors,
  stored/deflated package compatibility, malformed compression and expansion limits.
  Verify no separately installed VC runtime or TTS inference DLL is required.
- [ ] **D1 — Qualify portable setup, repair and updates.** Cover fresh/custom data folders,
  automatic model/adapter/runtime setup, interruption/resume, corruption repair,
  pause/retry, disk/network/permission failures and independent speech retry.
  Include the shared deadline through verified cache hits, hashing, recovery,
  publication and cleanup; the follow-up local check stopped before compilation.
  Qualify portable replacement's shared copy/hash/publication deadline and
  cancellation, preserving original bytes and removing unpublished sibling files;
  synchronous native I/O remains nonpreemptible.
- [ ] Qualify the shared native Rust HTTPS replacement in its owning compiler,
  analyzer and regression lanes, then exercise Windows Schannel/system trust,
  proxies, CI baseline lookup with Windows system trust, patched-client address
  fallback, HTTPS-only credential-free redirects, interrupted and ignored-Range
  transfers, strict length/range bounds, cancellation and aggregate deadlines
  through actual provisioning and updates. Qualify prompt stalled-body
  cancellation and worker joining with real Schannel and proxy connections;
  cached owner isolation coverage does not qualify these paths. Local checks
  establish no measured binary-size, latency or throughput improvement.
- [ ] Qualify automatic startup/15-minute release discovery, conditional metadata
  requests, compression, server-directed backoff and manual checks. Cover paginated
  numeric/prerelease selection, incomplete/no-release/network failures, changed
  publication and tampered staging/original bytes. Verify compact Settings with
  one action per update state, background download and a nonmodal restart nudge
  that preserves typing and generation. Qualify one shared confirmation only after
  initiating restart, cancellation without approval, Later and the Settings badge,
  deferred replacement on close without reopening, explicit replacement/restart,
  close during download or generation, renamed executables, original options,
  read-only targets, sharing exhaustion, restart failure, data preservation and
  rollback; source and preview
  launches must stay silent. See [update behavior](docs/releases.md#update-and-remove).

## Application and daily service

### Actual-model acceptance

- [ ] **A1 — Qualify the Rust/PQ2 daily-use path.** Prepare the exact-source
  [acceptance package](docs/validation.md#actual-model-application-acceptance)
  after C1, with strict Release analyzers, production linking and fresh typed
  workflow receipts. Qualify explicit DX12 startup, authenticated four-worker
  readiness, Stop/Retry, recovery and explicit publisher rollback.
- [ ] Qualify acknowledged model/broker shutdown, descendant teardown, empty
  Jobs, released listeners and fresh-owner replay. Include startup timeout,
  failed-cleanup receipt retention and one shared absolute deadline. Qualify
  inherited owner admission against the larger UTC/monotonic elapsed cost,
  including fractional expiry and a shorter inherited allowance.
- [ ] Qualify native startup descendant-cancellation and bounded failure/output
  fixtures, visible child readiness diagnostics, exact root exit/capture summaries
  and pre-cleanup status/teardown receipts. Preserve a single cleanup allowance
  through explicit preparation stop and contained-process destruction, including
  failure and unattempted-drop controls,
  and keep raw compiler captures out of presentation. Close the focused compiled
  controls and owning-lane checks; the [local workload](docs/evidence/daily-use-readiness.md)
  supplied no passing Rust-test verdict.
- [ ] Diagnose publisher PQ2/PTQ1 exits before HTTP readiness, including missing
  backend exit codes and bounded diagnostics, before selecting more timing work.
- [ ] Qualify source-launch first frame/readiness, staged text/speech setup,
  bounded parallel artifact preparation, runtime/broker overlap, peer cancellation
  and descendant cleanup. Measure cold and unchanged launches separately, including
  X1 sequential hashing and speech-build contention.
- [ ] Qualify all native UI flows at normal/minimum window sizes, display scaling,
  high contrast, keyboard/IME input and with assistive technologies under the
  [application guide](docs/development-guide.md#native-chat-app). Extend the
  [reference-guided controls](docs/validation.md#candidate-reference-guided-controls)
  with process-bound Windows UI Automation checks and representative cosmic-text
  multilingual edit cases; retain manual Narrator, IME and bidirectional review.
- [ ] Qualify shared error selection, full-message Copy error, details/startup
  dialogs and Retry with simultaneous setup/model/speech/audio/update failures,
  preserving the composer and modal shortcut ownership.
- [ ] Qualify pre-reply failure recovery with actual complex prompts: distinguish
  canonical request-size/schema rejections from counter/generation stage failures,
  preserve drafts/history and require confirmed cleanup before model retry.
- [ ] Qualify the pinned Prism ordinary-request deadline correction with actual
  long-prefill completion, stage/cause diagnostics, explicit diagnostic deadlines,
  Stop/cancellation and bounded intake/writes. Measure the supplied modest critique prompt
  before selecting prefill optimizations; keep its private text out of evidence.
- [ ] Qualify compact message headers, thinking expansion, speech/iteration
  controls and native Markdown responses: headings, nested emphasis/lists/quotes,
  code, task lists and text-column tables; Unicode selection, original-source
  Copy/export/recovery, rendered Find mapping and source display for hidden-syntax
  matches, malformed/streaming input and bounded plain-source fallback. Keep
  native accessibility, X1/X2 responsiveness and complete CommonMark claims
  separate from focused supported-subset checks.
  Include native regular/bold reading faces, all six heading sizes, line/paragraph
  spacing, nested list/quote prefixes and full-width code surfaces in Light, Dark
  and Windows high contrast, including width/DPI/reading-size reflow and Find
  highlights over code. Focused source controls do not establish device acceptance.
- [ ] Qualify the persistent offline Markdown worker's MXC/PSEC admission and
  denial controls, one-process/memory limits, bounded binary IPC and trusted-parent
  validation. Cover malformed/stale results, queue backpressure, worker exits and
  timeouts, failed cleanup and window-close teardown; require plain-source
  availability after failure without in-process parsing. Qualify actual native
  network/loopback/file/device/desktop/child denial and boundary overhead separately
  from portable parser/DTO regressions. Complete final-source strict native
  analysis and named regressions, including minimal child-environment contents.
  Include the configured headless broker image's production handshake/Unicode
  regression, private-entry direct-invocation refusal, pure `bonsai-markdown`
  parsing without native UI/graphics imports, and the six-key minimal environment
  with private `LOCALAPPDATA` in owning native acceptance. Verify recovered
  responses format before model preparation and after model setup failures using
  the launcher's prepared image, including explicit nonadjacent broker paths,
  initial bottom scrolling, offscreen backlogs and the under-one-second rendering
  target. Require the first formatted viewport before Swift/TTS work and
  unchanged-source asynchronous layout invalidation and repaint after request
  admission in an idle window. Include the source launch's
  early broker build and preserved source-lock provenance; explicit parser warmup
  without a source request, readiness before Swift/TTS preparation in staged and
  direct configured launches, responsive browsing/drafting during the handshake,
  and terminal-failure release of setup without parser retry. Older launchers
  without an image handoff retain deferred formatting. Parser entry must precede
  model, credential and network setup. Retain actual child exit state before cleanup
  when diagnosing failed startup or pipe exchange.
- [ ] Qualify Settings scope, compact/wrapped navigation, persistence, generation
  locks, adapter disclosure, diagnostics, scrolling and Close/Escape focus return.
- [ ] Qualify responsive sidebar/drawer/hide/reopen behavior, stable manual width,
  branding/footer contrast, narrow composer controls, reading size/width and
  appearance persistence across restart and conversation-saving opt-out.
- [ ] Complete final-source native UI analysis and named ergonomics, appearance,
  profile, reader-size, Settings, deletion, context/Thinking and iteration controls.
  Optional capture recipes live in [validation](docs/validation.md#focused-local-checks).
- [ ] Qualify Find/sidebar query synchronization, Unicode/long/title-only results,
  Clear/Close, focus and preserved local Find edits/navigation across live changes.
- [ ] Qualify more than 16 conversations, branching/saving/reopening, virtualized
  live search and bounded encrypted storage while preserving drafts, settings,
  history, selection and keyboard focus.
- [ ] Qualify edited/regenerated/interrupted reply branches, Continue and exact
  committed-prefix submission through Stop, recovery, background ownership,
  completion races, eviction and capacity limits. Include visible Retry response,
  same-card unanswered retries, preserved composer/cursor, interrupted-answer
  branches and explicit Reload model without automatic resend.
- [ ] Qualify composer context estimates/rings and pending states through draft,
  reasoning, branch and recovery changes; preserve Extra effort first-pass scope,
  reply settings, Start fresh and authoritative broker rejection.
- [ ] Qualify conversation switching/creation/deletion during startup, generation
  and failure: preserve drafts/cursors, route hidden responses/rejections to their
  owner, show vertical response activity only on the generating owner's sidebar
  row and keep speech readiness independent of text readiness.
- [ ] Qualify encrypted conversation, window and voice-profile persistence,
  opt-out, unreadable-byte preservation, monitor/DPI changes and no automatic
  request replay, microphone capture or playback after restore.
- [ ] Qualify Qwen3-ASR microphone permission, explicit/25-second completion,
  autodetection, cancellation, draft insertion and explicit Swift submission. Cover
  unavailable-model drafts, late results, ASR/TTS exclusion and recorder cleanup.
- [ ] Qualify microphone model preparation after permission and successful
  recording startup: loading/capture overlap, empty/invalid final clips,
  denied/revoked permission, Stop/read-aloud/switch/close during loading,
  generation ownership and no premature upload, transcript or idle lease.
  Measure cold and repeated microphone completion latency with matched power.
- [ ] Qualify the 30-second successful-input ASR lease: state reset, authenticated
  readiness/listener ownership, monitoring and cancellation/failure/expiry release.
  Compare repeated-input latency/resources with cold loading under matched power.
- [ ] Qualify source-built ASR DOTPROD/I8MM selection against ARMv8-A: independent
  feature discovery, unknown-feature fallback, cache separation, complete
  transcript parity and matched microphone/file latency.
- [ ] Qualify audio-file picker/paste, supported Windows decoders, rate/channel
  conversion, long files, unknown/changing-memory admission, full transcripts,
  draft insertion, exactly-once ordinary submission and preserved drafts without
  deferred Send under the
  [audio contract](docs/streaming-resources.md).
- [ ] Qualify persistent sequential Media Foundation decoding, one-segment
  lookahead, exact EOF/tail coverage, pause-aware cuts, failure/cancellation/flush
  cleanup and transcript-viewer behavior. Measure total import latency and
  boundary-word accuracy before making benefit claims.
- [ ] Run the named pause-boundary, segment-conservation, sequential-decoder and
  sequential-stream controls and final native analysis on the settled source.
- [ ] Qualify speech after X1/X2 data-directory moves, CPU-feature or Windows
  changes, retaining pinned-model loading/tokenization/sampling, workspace/lease,
  listening and lifecycle gates.
- [ ] Qualify Swift `mmproj` provisioning/verification
  (`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`), `768 MiB` memory-planner reserve
  and `--no-mmproj` opt-out across X1/X2 memory configurations,
  `--publisher-engine` multimodal image and video-keyframe turns
  (`Input modalities: text, image`), future native Rust `mmproj` enablement once
  supported by a pinned Prism release, composer `+` menu ASR loading state and
  draft-only attachment staging while Swift prepares, and DPAPI attachment
  persistence.
- [ ] Qualify full selected context/output, long histories, speech, offline
  restart, memory pressure and recovery. Independently exercise requested
  `bounded_history` and `stalled_consumer` workloads under the acceptance contract.
- [ ] Extend [transcript/Find](docs/streaming-resources.md#native-chat-and-transcript)
  device acceptance through appends, replacements, switches, eviction, Unicode
  wrapping, saturated results, selection and queued navigation. Qualify bounded
  variable-height reflow and stable reading/Find anchors through width, DPI and
  font changes, including one oversized selectable message and deferred layouts.
- [ ] Complete signed-distribution acceptance: binary provenance, dependency and
  license notices, clean-machine prerequisites, installation, upgrades and rollback.

### Native application improvements

- [ ] After long-draft resource attribution, evaluate Ferrite-style operation
  undo against the current full-draft snapshots under the
  [native candidate contract](docs/ui-architecture.md#candidate-native-improvements).
  Bound retained edit bytes and preserve atomic oversized-edit rejection,
  IME composition, selection/cursor restoration and complete undo/redo recovery.
- [ ] Add bounded native comparison of alternate replies/drafts using the
  [similar reference](docs/research-register.md#native-application-and-writing-references).
  Keep Unicode-aware diff work cancellable and off the event loop; preserve
  immutable branch identities and require explicit acceptance of any replacement.

### Isolation, transport and recovery

- [ ] Qualify the [input capability boundary](docs/sandboxing.md#input-capabilities):
  exact selected-file/ancestor lifetime, offline same-executable MXC decoder,
  native Media Foundation under Win32k denial, denied unrelated files/devices/
  networking, malformed/flooded/truncated pipes, Stop/close and confirmed teardown
  or bounded quarantine. Cover portable and installed app identities and Windows
  microphone consent/revocation independently; no in-process decoder fallback.
- [ ] Qualify AppContainer `--attachment-import-worker` WIC image
  normalization and Media Foundation sequential video keyframe extraction
  under Win32k, network and unrelated-file denial, including multi-file COM
  picker bounds, bounded pipe framing, cancellation/deadlines and confirmed Job
  teardown.
- [ ] Qualify [negligible boundary overhead](docs/validation.md#boundary-overhead-acceptance)
  for existing microphone/file, process, artifact/credential and authenticated
  transport paths as well as new isolation. Measure cold setup separately from
  representative warm operations, with the 1% warm-operation target, identical
  inputs, independent correctness and eligible power conditions. Source work/
  allocation controls and removed duplicate probes alone provide no timing verdict.
- [ ] Establish [restricted inference authority](docs/sandboxing.md#worker-authority):
  approved read-only artifacts, bounded private cache/temp, denied unrelated
  files/credentials and egress, broker-only communication and no same-user bypass.
  Offline `--broker-pipe` transport (`SB2PIPE1`/`SB2OKAY1`, `KIND_SERVER_READY`,
  `KIND_RESET`), `PrivateCredentials::create_frontend_only`,
  `OwnedProcess::start_sandboxed_broker_pipe` (`SandboxNetwork::Offline`, 6-entry
  minimal environment, pre-resume kernel `TokenIsAppContainer`/`GetAppContainerFolderPath`
  verification, `64 MiB`/`1_024`-entry private storage ceiling), the fail-closed
  `BROKER_PIPE_RELEASE` gate, and `inference-authority-v1` fixture/tooling
  support exist in source; qualifying `inference-authority-v1` in sandbox CI
  (after a closed local Case 1 fixture failure repaired without retry), pinning a
  reviewed `--broker-pipe` release, and qualifying actual-model MXC loading, atomic
  Job limits and exact identity without weaker fallback remain open.
- [ ] Qualify trusted-host artifact ancestor sharing and nested rename/DELETE
  regressions through native/AppContainer integration; host guards must outlive
  contained readers and preserve their narrower rights. Compare Codex's
  [no-reparse handle controls](docs/research-register.md#reliability-and-distribution-references)
  as an independent review source for staging/ancestor mutation races.
- [ ] Qualify parent-drained stdout/stderr's hard 16 MiB per-stream cap,
  read/write/panic failure, complete tail retention and shared-deadline cleanup.
  Qualify the separate source-bound `log-capture-v1` workload for stdout/stderr
  flood and normal-exit tail/file release, preserving the original six-case
  authority workload and each original deadline. Finish its workflow input/plumbing
  integration after mandatory startup/syntax preflight; the stopped local bootstrap
  ran none of those checks. Independently bound scratch and
  PSEC-private temp storage; qualify exhaustion, diagnostic/credential retention
  and owner-loss cleanup.
- [ ] Qualify the [broker contract](docs/broker-policy.md): authentication, numeric
  loopback, Host/Origin/routes, rejected tools/loading/overrides, queue/byte/token
  bounds, rate limits, absolute operation deadlines and disconnect/stalled-consumer
  cancellation. Cover complete 429 responses, constant-space draining and explicit
  resend without replay; ordinary generation retains no elapsed-time cap.
- [ ] Extend selected cancellation/ownership/shutdown models with
  [Shuttle schedule replay](docs/validation.md#candidate-reference-guided-controls)
  where existing scheduler histories leave a concrete concurrency gap. Keep
  models small and independently asserted; replay controls do not qualify real
  Windows process, Job, handle or network enforcement. Include owning CI and
  actual-model acceptance for screening items `#8`, `#49` and `#50` (`AcceptedChat::serve`
  queue-wait cancellation/`poll_peek`, deterministic 864-interleaving simulation without
  adding `shuttle` to workspace manifests, `json-candidate` single-pass malformed-JSON
  rejection, UI `50 ms` TCP connect slices, `error_ui` frame-zero-clone copy action,
  streaming Markdown prefix-layout retention without UI parsing and incremental TTS
  segmentation).
- [ ] Qualify actual-model [streaming/audio resources](docs/streaming-resources.md):
  ready prefixes, chat/speech overlap, PCM/WASAPI reuse, final-storage intake,
  fragmented framing, caps/tails/truncation, failed writes, device invalidation,
  Stop/recovery, task latency and peak memory.
- [ ] Qualify all 33 Swift control-token spellings in real template/tokenizer
  sessions, including escaped/split content, history/reasoning and lookalikes;
  require fixed-400 recovery and identical count/generation handling. Qualify the
  reference vocabulary separately under the
  [token contract](docs/prism-upstream-tests.md#chat-parser-and-tokenizer-contracts).
- [ ] Qualify load/allocation/backend-loss recovery, bounded restore attempts and
  unchanged workspace on failure, with no restart loops, silent profile changes,
  corruption or automatic cloud fallback.

### Durable qualification evidence

- [ ] Deploy native `evidence-archive` selection/seal/restore to durable owner
  storage, finish CLI/analyzer/guard acceptance, capture selected required bytes
  before artifact expiry and rehearse digest-verified retrieval without the
  original runner. Follow [evidence retention](docs/validation.md#evidence-retention)
  and retain independent reviews and losing/no-go outcomes.

## Runtime and correctness

- [ ] **R2 — Qualify settled Prism source.** Complete pinned-compiler Foundation
  checks for all 12 crates: strict Release Clippy/rustdoc, tests, builds and
  public-reference intake/lifetime controls under [Prism conformance](docs/prism-upstream-tests.md).
- [ ] Qualify released private HTTP and internal-worker counting, adapter
  readiness/scales, active-graph cancellation, reset/rollback and stale-state
  rejection, with independently retained Prism-owned receipts.
- [ ] Run actual-PQ2 `swift_model_check rollback`: compare every token/logit and
  complete KV/GDN/convolution bits before reset, then continuation against fresh
  computation under the [runtime contract](docs/native-rust-engine.md).
- [ ] Qualify full-context/output scalar/NEON behavior, rank-one zero/nonzero
  adapter application and writer basis, sampling, termination and full-state
  lifecycle at the declared numerical tolerance.
- [ ] **R3 — Share complete-state/fault tooling.** Map each prefix, prefill,
  speculation and accelerator path to its own independent logits, KV/GDN/conv,
  sampler, restore, adapter and resource assertions under
  [qualification reuse](docs/validation.md#reusing-qualification-controls).
- [ ] Use batch-one attribution to select allocation/copy work. Qualify existing
  reset checkpoints, top-20/finite-logit checks, rank-one/attention changes,
  immutable logits and BPE/special parsing under the
  [engine contract](docs/native-rust-engine.md) and
  [Hexagon execution contract](docs/hexagon.md).
- [ ] Qualify 8/16/32-token CPU prefill, distinct `gdn-wy-f32-v1` arithmetic,
  selected-cache service consumers and independent packed-KV head workers:
  scratch/admission, complete state, adapters, cancellation and whole-request benefit.
- [ ] Qualify six-head/paired-token score/value reuse. Extend score/value or
  online-softmax work only if selected-context attention is a measured bottleneck.
- [ ] Qualify explicit [latest-prefix reuse](docs/native-rust-engine.md#latest-prefix-reuse) across service,
  broker and UI: edited/switched histories, scale changes, cancellation/faults,
  fresh-state parity, bounded eviction/resources and repeated-history benefit.
  Include the [vLLM checkpoint review cases](docs/inference-algorithms.md#hybrid-cache-reuse-and-speculative-state):
  allocated-but-unwritten checkpoints, complete recurrent/convolution history,
  and shared attention prefixes without a usable matching recurrent checkpoint.
- [ ] Qualify the batched target verifier and integrated Swift sampled suffix
  consumer: per-position logits, actual sampler/RNG progression, checkpoint
  admission, accepted-prefix restoration, EOS/Stop/failure and complete-cycle benefit.
- [ ] After a useful verifier, screen short suffix chains, then a compatible
  published Bonsai DFlash2 drafter, then bounded adaptive length. Charge target
  features, shared embeddings/head, adapters/tokenizers, both models and rejected
  work under [speculation contracts](docs/inference-algorithms.md#hybrid-cache-reuse-and-speculative-state).
- [ ] Apply reviewed [cache/batch/transfer planning](docs/inference-algorithms.md#cache-batch-and-transfer-planning)
  to selected consumers: complete page identities, one cache owner, bounded
  transfer queues, terminal completion and cancellation before publication/reuse.
  Qualify fault paths, resources and whole-request benefit.
- [ ] Consider FP32 factor buffers or SpecLA/TreeWY/RecoverSSM recovery only for
  measured state/checkpoint cost; preserve complete checkpoints and numerical scope.
  Deferred screening items `#35` (shared drafter embeddings/head), `#37` (checkpointable
  delayed recurrent updates / `ReplaySSM`) and `#38` (transactional grammar speculation)
  likewise require explicit weight/feature/mask identity, atomic KV/GDN/convolution/sampler
  rollback and complete-cycle benefit before integration.

## Complete C++ retirement

- [ ] **S1 — Qualify Rust TTS.** Compare pinned Q8/Q4 models against independent
  complete graph/cache/code/PCM observations using the
  [speech acceptance workloads](docs/cpp-retirement.md#speech-acceptance).
  Close tokenizer/sampler, voice identity, Design/Base lifetime, cancellation,
  recovery, listening and resources; diagnose the latency regression with matched
  power. Mathematical controls and replay tooling do not close model acceptance.
  Include owning compiled CI and actual-model validation of `#8` (`synthesize_*_cancellable`)
  and `#45`–`#48` (`sampling.rs` finite/`f64`/`Option<usize>` greedy and repetition-penalty
  ordering, GGUF `TYPE_BOOL`/UTF-8/`INT32` `general.alignment`/bounds, and scalar-boundary
  multibyte Unicode whitespace scanning).
- [ ] Qualify the [native Q8 transport](tools/hexagon/native_rpc/README.md#q8-matrix-preparation-and-immutable-placement)
  under contained HTP0 execution and owning hosted compilation. Bind Q8
  tensor/solver mapping and exact source/model/package/catalog/driver/VTCM/
  loaded-DSP admission; qualify signed arithmetic, correlated completions,
  buffers, cancellation, teardown and TTS/shared-KV integration. Simulated native
  transport, fixed F32 and guarded-Q2 controls do not qualify speech state.
  First supply a reviewed session-bound provider for exact loaded-DSP bytes and
  effective usable VTCM; package/URI selection and capacity attributes cannot
  supply those missing observations. `Q8HvxQuantizationOracle` covers independent
  host HVX (`0x2008` binary16/QF16) quantization arithmetic controls; still review
  predeclared scalar projection-error tolerances for ordinary speech activations,
  then qualify actual signed-path error on device independently of the exact diagnostic
  subset. Verify cancellation and the original absolute work deadline through
  controlled oracle preparation and final review, preserving native cleanup reserves.
- [ ] **S2 — Qualify Rust ASR for production.** Compare the complete Rust graph and
  explicit Rust-only service against pinned Qwen3-ASR 1.7B controls. Complete
  owning hosted build/link and native-control regressions; preserve independent
  frontend/numerical/tokenizer observations. Qualify transcript/state,
  microphone-to-Swift submission, quality, cancellation, lifecycle and resources
  before replacing production or removing native dependencies.
  Supply a reviewed upstream numerical observation ABI and capture every generation
  step's raw logits and complete all-layer committed KV through the bounded packet
  writer; current debug output and device argmax cannot populate the complete
  independent replay.
  Qualify the prebuilt source-bound replay consumer with actual independent
  packets, matching provider identities and the original owner clock/power scope.
- [ ] Retire the remaining Windows queue ABI, ASR, Prism reference and publisher
  native/DLL families
  [file by file](docs/cpp-retirement.md): replace source assertions, geometry IDs,
  codec/ABI, state and fallback obligations with independent controls, then remove
  compiler/provisioning/build paths. Prism needs complete independent observation
  provider coverage for its 25 remaining C++ files; retain every approved
  green ID, all 22,817 ordered backend geometries and separate performance work.
  Retain required replay data and qualify owning hosted builds without a project
  C++ toolchain. Capture original selected compiled/preprocessed/generated inputs
  and the complete schema-3 compilation database beyond ASR's registry inventory;
  run source and [link/package verification](tools/bonsai/README.md#selected-native-link-and-package-capture)
  on those actual captures in owning CI.
  Qualify the retained-artifact collector and explicit live compiler-capture route
  on actual owning-build selections; retain original supported Clang/clang-cl
  argument arrays at execution, with explicit outputs and complete dependencies.
  Do not rewrite shell command strings or historical relative-output databases
  into admitted evidence.
  Inventory the restored original publisher archive identity in owning CI after
  fresh exact-byte intake. Retain the failed `prism-engine` HTTP 404 outcome and
  its original pin; do not substitute archive bytes or reopen that completed
  intake workload. The source-lock correction alone supplies no archive inventory.
- [ ] Replace the [MSVC startup/panic runtime boundary](docs/cpp-retirement.md#msvc-startup-and-panic-runtime)
  with a qualified Rust provider while preserving preparation-peer recovery,
  worker draining and scoped cleanup. Static CRT linkage alone does not retire
  the embedded C++ exception runtime; a blanket abort profile changes recovery.
  Qualify schema-5 COFF, exact-image PE runtime-symbol, retained ordinal/forwarder
  and recursive dependency-import observations on actual link/package inputs.
  Keep provider replacement, selected library members, API-set routing and source
  ancestry separate from those observations.

## Creative quality, adapters and memory

### Quality and speech

- [ ] **Q1 — Complete independent creative/speech review.** Use the
  [fixed daily-use packets](docs/daily-use-review.md), complete answers/audio
  and declared writing/art-direction rubrics, including Thinking,
  output settings and Extra effort. Reuse applicable retained captures without
  transferring their verdicts to changed source. Adapt the
  [Longform Writing Bench rubric split](docs/performance-expectations.md#creative-review-packets)
  between local prose and complete-story continuity/coherence; retain independent
  human review and held-out tasks without importing its Python/cloud-judge harness
  or claiming leaderboard equivalence for a shortened packet.
- [ ] Qualify [standard LoRA](docs/swift-lora.md#standard-profile-qualification)
  writer basis, loaded-zero/no-adapter identity, nonzero effect and application
  at scale 1.3; retain scale-specific throughput, complete-answer quality and RAM.
  Run the full 28-sample LoRA/14-sample PTQ1 matrix only in its owning CI, or select
  one independently useful bounded capture.
- [ ] Qualify Q8/Q4 speech and thread choices, punctuation/live read-aloud,
  sampling/accent/reference changes and boundary smoothing. Distinguish retained
  PCM continuity from audible continuity under [speech measurement](docs/performance-expectations.md).
  Include clause-aware hard cuts, preserved live paragraph separators and 5 ms
  segment-edge fades, with first/last and exact-packet endings, under real-device
  listening and synthesis-gap acceptance.
- [ ] Add Rust ebur128 loudness/true-peak observations and reproducible
  loudness-matched listening copies to selected speech review packets under
  [speech measurement](docs/performance-expectations.md#q4-speech-resource-comparison).
  Preserve original PCM and sample identities as numerical evidence; loudness
  matching alone establishes neither voice quality nor intelligibility.
- [ ] Qualify chat, warm TTS and overlap: idle/active/ASR monitoring, 30-second
  leases, Stop/wake, shared threads, memory, heat and comfort.
- [ ] Qualify shared utterance-owned Talker/CodePredictor decoder scratch:
  owning-lane strict Release analysis/linking; literal outputs, poisoned storage,
  width changes, allocation identity and late-failure recovery; packed Q8/Q4,
  adapter/model parity, cancellation and unwind. Measure retained cross-model
  capacities and codec-callback overlap with complete X1/X2 request latency and
  process resources under the [speech storage contract](docs/streaming-resources.md#native-speech-generation-and-streaming).
  Local `cargo +nightly-2026-11-12 clippy --locked --offline --release -p bonsai-speech-tts --all-targets --all-features -- -D warnings`
  and the three named workspace unit tests passed locally on Windows ARM64;
  complete workspace/CI acceptance remains pending.
- [ ] After ordinary Q4 acceptance, evaluate a verified 48 kHz decoder only with
  compatible locked weights, license, reference-audio/PCM/UI changes and listening
  acceptance.
- [ ] After Q4 acceptance and compatible runtime/Hadamard artifacts, evaluate
  ternary Talker separately; do not infer compatibility from repacked Q8/Q4 or
  extend it to codec/head/predictor without independent work.

### Writing context

- [ ] After creative-chat acceptance, implement a small native story-fact and
  scene-context packet system using [Spindle and SillyTavern design references](docs/research-register.md#native-application-and-writing-references).
  Use explicit approved facts, deterministic activation/priority, inclusion and
  exclusion reasons, and a bounded token budget with authoritative broker
  counting. Require reviewed canon changes; preserve branch ownership, encrypted
  persistence/opt-out and draft recovery. Start without embeddings or an MCP/web
  frontend, and evaluate complete-scene continuity and task cost before expansion.

### Direct LoRA weight editing

Follow the [evaluation contract](docs/weight-space-programming.md#direct-edit-evaluation-contract).
Within this research area, W8 is primary. The existing adapter-compression,
refitting and qualification work below is secondary; reuse its applicable
evaluation controls without making the pruned candidate a prerequisite for W8.

#### Primary: cheaper model execution with small weight edits

- [ ] **W8 — Use small editable adapters to preserve quality under substantially
  cheaper execution.** Investigate long-context attention/KV approximations and
  selected FFN/layer/projection skipping or learned gating, with the packed base
  weights immutable; select one measured bottleneck/profile first. Scope native
  graph/loading/state changes and additional adapter targets/rank needed to
  compensate; existing rank-one output adapters
  need not suffice. Bound calibration/distillation cost before selecting work.
  Compare the original graph, the cheaper graph without new compensation and the
  same cheaper graph with frozen adapter edits; retain existing adapter behavior
  at surviving sites and explicitly scope skipped targets. Bind execution/state
  identities and qualify rollback per profile. Require held-out answer quality,
  material useful-answer latency and/or actual-process memory benefit, with
  predeclared tradeoffs and the existing budget/power/release gates. Distinguish
  skipped computation from unloaded weights and reduced KV state; an FFN-down
  adapter cannot replace a whole skipped FFN while still requiring its hidden
  activations. Consider expert-routing edits only for a separately authenticated
  compatible MoE model; current dense Swift has no expert router.
- [ ] **W8.1 — Screen structured FFN omission with small recovery adapters.**
  Prioritize retaining most nonlinear computation over replacing a complete FFN
  with a linear map. Qualify source-only paired full/retained capture, the native
  `ffn-surrogate` fitter and frozen-compensation replay on the same normalized
  input, with owning-lane Release/analyzer acceptance and source-bound prebuilt
  helpers before actual-model intake. Bind complete 1,024-coordinate groups,
  original Hadamard signs/stride, exact source-order F32 factor application,
  retained standard-adapter semantics, original-prefix teacher drift,
  failed-token discard, disjoint groups, factor/source/scale admission and fresh
  graph/checkpoint identities. Qualify both exported-factor F64 and actual F32
  branch-error screens; teacher-forced one-site replay cannot establish
  autoregressive quality or multi-site drift recovery.
  Follow the [selected investigation](docs/weight-space-programming.md#selected-w8-investigation-recover-omitted-ffn-computation).
  Qualify the source-only native `ffn-timing` checker and use a bounded full-token
  CPU attribution to nominate one actual bottleneck; retain all layer observations,
  first-token scope, original deadline and power. It does not qualify sampled
  answers, narrowed/tiled prefill or Hexagon. Predeclare one feasible site/capture
  before its device work; stop at the development-error gate before building a serving
  profile. Reject compensation that loses to the uncompensated branch or fails
  the fixed development-error/generalization gates. Keep whole-FFN replacement
  as a separate capacity experiment. Scalar reconstruction does not qualify
  production NEON or complete-answer quality. Include owning compiled CI verification of
  `#51` (`calibration_metrics`, path-free `fitInputSha256`, omitted raw receipt paths,
  top-8 `weight_patch` violation caps, single-pass `startup_diagnostics` redaction,
  `<= 256`-byte scalar-free JSON parse errors and `evidence_archive` path guards) before
  running device calibration captures.
- [ ] **W8.2 — Turn a viable FFN candidate into removed work and allocations.**
  Add a distinct opt-in graph/loading identity with selective packed-payload
  admission, surviving adapter semantics and complete state/reset/rollback
  isolation. Compare original, omitted-without-recovery and frozen-recovery
  profiles. Preserve original sign slices and down-column strides when compacting
  complete Hadamard groups. Size the exact mixed omission/rank policy with native
  `ffn-plan` accounting against net factor/payload cost; its 512 MiB envelope is
  geometry only. Two groups at all 64 sites cannot meet W8.4's RAM target even
  before adding compensation. Qualify selected sites independently and their
  changed-prefix composition. Measure real process resources; a bypass retaining
  tensors saves no RAM.
- [ ] **W8.3 — Evaluate learned prompt retention only for a long-prefill hotspot.**
  Use the [TokenGate hybrid reference](docs/research-register.md#small-edits-around-cheaper-execution)
  to scope retained positions, gate bias, pre-/post-gate GDN history, attention
  Q/V compensation and cache/checkpoint identities. Bound external calibration
  before implementation; keep its profile and held-out retrieval/continuation
  acceptance separate from FFN omission and short-context decode.
- [ ] **W8.4 — Pursue the three-way quality, latency and RAM target.** Freeze
  representative untouched task groups and the selected investigation's material
  targets before owner-operated CI. Distinguish recovered baseline quality from
  improved reasoning/instructions/creative quality. Investigate a diagnosed
  behavioral correction separately, then compare its composition against the
  same compensated graph without it. Include an equal-resource comparison that
  spends saved work on useful reasoning while retaining absolute useful-answer
  latency and memory gates; shorter answers alone cannot establish ability.
  Retain all no-go outcomes, per-category
  review, source-bound resources and matched power; promote only after existing
  release/application gates. No simultaneous Swift benefit is established yet.

#### Secondary: adapter compression, refitting and qualification

W1-W3 assess the frozen pruning candidate if its expected benefit justifies more
work; W4 remains conditional, W5 qualifies selected candidates and W6 gates app
use. W7 concerns the original Swift correction's source and blocks neither W8
nor editing the available Abliteration factors.

- [ ] **W1 — Freeze the accuracy, speed and memory decision contract.** Define
  representative calibration/development/held-out answer groups covering facts,
  reasoning, instructions, creative output and intended adapter behavior, with
  independent review. Predeclare quality nonregression margins, useful-answer
  latency and actual-process memory targets, allowed tradeoffs and stop criteria.
  Keep the four familiar facts as smoke checks, outside tuning and held-out scores.
- [ ] **W2 — Prepare a native evaluator that separates weight and kernel effects.**
  Add explicit research loading/helper support for the same edited GGUF with
  dense execution, then compare original-dense, edited-dense and edited-sparse
  at identical scale/settings. Bind exact build/artifact/derivation identities,
  final factors and execution profile. Reuse native completion supervision;
  enforce aggregate budgets, arm order/cache policy, power eligibility, scoring
  and minimal replay evidence in Rust before selecting device work.
- [ ] **W3 — Characterize the frozen half-retained candidate using W1/W2.**
  Establish independent per-site correction/logit tolerances for the same edited
  weights under dense/sparse execution, and complete-state reset, rollback,
  cancellation, zero/nonzero scale and checkpoint isolation for the edited profile.
  Assess complete answers and
  attribute factor capacity, steady/peak process memory and adapter work before
  selecting further optimization. Keep P1/writer-basis and other backend gates
  open; historical pilot clocks stay closed and authorize no repeat.
- [ ] **W4 — If W3 identifies useful headroom, implement direct A-factor refitting.**
  Capture bounded original-basis FFN activations with source/profile provenance
  and disjoint fitting groups; do not transfer scalar/unfused probe verdicts to
  production NEON. Optimize final A-factor support and activation-weighted
  correction error under measured storage/work limits; fold aligned LP edits
  into existing A with exact direction/alpha/scale handling. Constrain final
  factor sparsity, not just delta sparsity, and certify exported F32 factors.
  Gate SSM refitting on P1/native writer-basis evidence; keep B, embeddings,
  target inventory and packed base fixed unless separately scoped.
- [ ] **W5 — Qualify frozen candidates on untouched held-out tasks.** After W1/W2
  and applicable W3/W4 checks, use owner-operated CI for the full quality and
  repeated/counterbalanced performance matrix. Admit only independently useful,
  predeclared bounded device work; retain per-arm/aggregate power, losing/no-go
  results and cleanup. Reject speed comparisons with changed/unknown power;
  separate factor payload, actual RAM and unchanged dense GGUF size. Claim
  accuracy gains only from independent held-out evidence; retain Pareto tradeoffs
  and stop candidates that miss quality or material system-benefit criteria.
- [ ] **W6 — Integrate only an accepted candidate through a reviewed opt-in.**
  After W3/W5 and standard-profile gates, publish a source-bound Prism release
  through the owner workflow, then add explicit Runner artifact/profile handling.
  Bind derivation and sparse arithmetic, invalidate state/prefixes on changes,
  retain approved-adapter rollback and qualify selected CPU/backend, broker,
  packaging and application paths. Gate extra kernels, factor precision or disk
  formats on measured benefit; research success alone changes no default.
- [ ] **W7 — Obtain original Swift LoRA factors before editing that correction.**
  Authenticate floating-point factors, base revision, target/basis mapping,
  merge/quantization recipe and license if they become available. Do not recover
  or substitute them by differencing quantized checkpoints or selecting a
  different model's adapter. Define a separate merge/requantization and complete
  acceptance pipeline before any packed-base change.
- [ ] For a separate behavioral correction, complete [fault assessment](docs/weight-space-programming.md#practical-investigation-order)
  and ordinary controls before selecting a direction/LP fit. Compare constant
  steering at the same site, preserve the original activation basis and qualify
  held-out complete answers, composition, state and resources before serving.

### Memory and profiles

- [ ] **P1 — Run public adapter reconstruction.** Execute the declared all-48
  A-factor [public-pack recipe](docs/adapter-public-reference.md), then independently
  replay retained comparisons with the schema-2 scalar auditor. Separately obtain
  authenticated original checkpoint/conversion/invocation/toolchain inputs and
  verify native writer mappings through [authoring intake](docs/adapter-authoring-intake.md).
- [ ] **P2 — Qualify PTQ1 numerical execution.** Bind one released-source
  candidate, selected worker/backend/cache/placement and adapter policy. Qualify
  scalar/explicit-NEON logits and complete live
  state for the selected worker policy, including loaded-zero identity, nonzero
  effect, scale/history changes and prefix reuse. Keep the CPU/FP16 reference
  policy, one-token smoke and development K4/V4 acceptance distinct; P1's
  authenticated authoring/basis requirements remain independent. Run owning
  strict Release analyzers and the artifact-pin, retained-greedy and promotion
  receipt regressions; local validation stopped before Rust compilation.
- [ ] **P3 — Qualify PTQ1 application service and rollback.** Prepare a newly
  source-bound PTQ1 service/rollback packet for the selected
  runtime. Qualify scale/count/context rejection, active interruption, independent
  fresh-owner replay, original-PQ2 rollback and final file verification under
  [application acceptance](docs/validation.md#actual-model-application-acceptance).
  Execute the CPU/cache/reuse, power, incomplete-response and malformed-transport
  cleanup regressions before actual-model capture; source review is not execution.
- [ ] **P4 — Measure the PTQ1 deployment benefit.** After correctness, run matched
  complete-answer CPU PQ2/PTQ1 comparisons with the same cache, backend, workers
  and standard adapter. Verify explicit CPU readiness; retain startup, prefill,
  decode, end-to-end latency, separate process-memory domains and losing outcomes.
  Separately compare against automatic PQ2 placement on X2. Qualify repeated
  benefit and per-prompt adapter gates in owning CI with eligible aggregate/per-arm
  power observations. Serialized savings and short tree-name pairs establish no
  actual usable RAM, energy, quality or repeated performance benefit.
- [ ] **P5 — Complete PTQ1 profile acceptance.** Close independent human
  quality/comfort, selected-cache/prefix/full context/output, resources, isolation,
  recovery and clean installation before reconsidering
  [PTQ1 default promotion](docs/swift-memory.md#release-decision). Follow the
  [vetting sequence](docs/swift-memory.md#vetting-the-cpu-option), keep X1/X2
  evidence separate and require all nineteen reference-policy gates plus the
  selected production-profile acceptance; missing gates remain pending.
- [ ] Qualify source request-timing attribution after owning hosted checks; use
  actual operation measurements before selecting hotspots.
- [ ] Qualify native adapter allocation reuse: omit unused serialized
  linear A copies and preserve exact decoded B sharing. Verify the calculated
  8.25 MiB approved-adapter payload reduction against process resources and
  unchanged request performance, rounding/order, dynamic scales and
  load-without-apply under [allocation semantics](docs/swift-lora.md#allocation-semantics).
  Keep publisher reservation/allocator attribution separate.
- [ ] Qualify the development Qwen reference service only when its 18 GiB runtime
  budget plus 4 GiB host reserve is admissible; retain explicit model selection.
- [ ] Qualify creative/daily/long-context/battery profiles with complete-request
  memory, energy, latency and comfort measurements under [profile policy](docs/hardware-capabilities.md).

## Native context expansion roadmap

- [ ] **M1 — Qualify memory-selected context.** Test paired context/memory child
  arguments, unknown/insufficient/changing RAM, prelaunch admission, Job pressure
  teardown, Retry and UI capacity reporting under the
  [memory contract](docs/swift-memory.md#memory-selected-serving-context).
- [ ] **M2 — Qualify cache-aware accounting and staged 64K/128K windows.** Cover
  selected-cache process/state/checkpoint bounds, native sequential scratch,
  two-cache rollback, full-state restore, adapters, cancellation, quality,
  resources and complete-request latency.
- [ ] **M3 — Qualify the scoped 262,144-token K4/V4 profile.** Test complete
  checkpoints, edited/divergent/reused histories, adapter changes, Stop/Retry,
  resource peaks, ingestion latency and held-out long-range quality. Keep generic
  allocation and FP16 bounds separate.
- [ ] Synchronize runtime/tokenizer, broker admission, capability/readiness and
  UI reporting through exact-source releases; count prompt, reply and template
  reserve without silent trimming. Preflight workload feasibility before selecting
  larger-window captures.
- [ ] Deliver K4/V3 only after a compatible reviewed release opens its separate
  exact-source gate; qualify held-out retrieval, instructions and creative
  continuation at the selected capacities, including adapter effects.
- [ ] Qualify capacity-first CPU placement with the selected cache/reuse path;
  measure distinct usable headroom from omitted NPU payloads and the latency
  tradeoff before reducing fixed memory charges.
- [ ] If allocation measurements identify avoidable retained workspace or overlap,
  remove it and qualify actual usable savings. Credit PTQ1 resident savings and
  unique-page sharing only when measured, without double-counting physical bytes.
- [ ] Qualify packed-attention score reuse across serial heads within
  each worker chunk: exact per-head/state/failure behavior, retained capacities,
  adapters, cancellation and unchanged complete-request performance. The
  [native allocation candidate](docs/swift-memory.md#native-allocation-candidates)
  projects 20 MiB less scratch at 262,144 tokens/four workers; retain conservative
  admission charges until actual usable savings are measured.
- [ ] Qualify native long drafts, whole-edit rejection, undo/redo, recovery,
  source insertion and complete rejected/interrupted-draft restoration. Before
  raising serialized request limits, coordinate Prism server/message/tokenizer/
  worker limits with Runner request/recovery bounds and qualify Unicode/escaped
  byte limits as well as exact token limits.
- [ ] Qualify `#14`'s itemized worst-case execution-shape ledger (`SwiftExecutionShape`,
  `SwiftMemoryBreakdown`, `SwiftKvCacheFormat::Rotated4FittedValues` at `16,896` B/tok
  and monotonic `SwiftContextPlan::for_execution_shape`) in compiled CI and M1–M3 model
  checks. Deferred screening leads `#16` (aligned staging reuse), `#17` (optional Windows
  unbuffered direct I/O) and `#18` (owned tensor range mapping) remain gated behind measured
  cold-start/I/O/mapping bottlenecks and retained-handle/alignment/Job-commit contracts.

## Bounded calibration and optional precision

Use the [Unsloth principle review](docs/research-register.md#unsloth-calibration-sensitivity-and-evaluation-principles)
to refine W1/W8, Q1 and K1 without switching models or importing an upstream-Qwen
quantization recipe. Keep Swift's packed base and existing defaults fixed; its
unchanged current graph supplies the teacher for execution approximations.
The tasks below do not displace native daily-use acceptance or reopen past trials.

- [ ] **U1 — Prepare representative, source-bound Swift calibration.** Reuse W1/W2
  and W8 capture intake to define a bounded native manifest with canonical rendered
  token/template identities, adapter/scale, effective Thinking/effort, activation
  basis, graph/profile, token positions and content-group ownership. Cover dialogue,
  creative prose, instructions, Unicode and selected history lengths; keep fit,
  development and untouched complete-answer evaluation groups separate. Reject
  exact duplicate inputs crossing splits regardless of declared IDs, and review
  semantic overlap separately. Retain finite second moments/counts, coverage,
  outliers and explicit missing observations only when a selected consumer needs
  them; choose and budget one capture before device work. Do not use upstream
  Qwen's imatrix as Swift calibration or retain private prompt text unnecessarily.
- [ ] **U2 — Allocate approximation and recovery by measured sensitivity.** Extend
  W8.1 with a bounded one-site-at-a-time screen using U1 and the same normalized
  inputs: activation-weighted branch error, local output/logit effects and
  changed-prefix drift. Keep comparative site/rank/policy screens in owner-operated
  CI; device capture selects one predeclared site within its original aggregate
  allowance. Review FFN/projection roles, embeddings/output and
  attention/recurrent boundaries separately. Compare magnitude/uniform choices
  against measured choices before allocating omission groups or recovery rank
  under a fixed net RAM/work budget. Freeze the selection on development data,
  then qualify multi-site composition and complete held-out answers through W8.4;
  stop at a failed local gate. No wider decode of ternary weights can recover
  unavailable original precision, and local sensitivity is no quality guarantee.
- [ ] **U3 — Screen heterogeneous execution precision only after K1 attribution.**
  For a measured cache/activation bottleneck, compare one explicit static
  per-layer/per-role policy with uniform K4/V4 and FP16 controls; reuse K1's
  fitted-scale candidate where applicable. Protect F32 recurrent/convolution
  state and existing norm/accumulation contracts initially. Preserve packed base
  weights; charge scales, exception paths, retained copies, scratch and recovery
  factors against net resources. Add native kernels only for an admitted format;
  bind the complete policy to discovery, admission, prefix/checkpoint identity,
  reset/rollback and release gates. Consider runtime adaptation only after a
  static policy establishes useful benefit and its state-transition costs fit.
- [ ] **U4 — Add multi-token diagnostics without replacing complete-task review.**
  Extend W2/Q1/K1 with bounded teacher-forced top-1 agreement and distribution
  error on common current-Swift prefixes, plus separately identified free-running
  divergence and complete sampled continuations. Bind prompts, rendered tokens,
  numerical profile, sampler/seed, reasoning-inclusive caps and termination.
  Keep per-domain/site failures and long-prefix accumulation visible; short
  greedy agreement or a lower mean KLD cannot qualify creativity, instructions
  or a changed sampler. Freeze candidates before held-out review and use the
  owning CI for repeated/counterbalanced scoring, preserving truncated/no-go
  outcomes and source-bound useful-answer latency/resources.
- [ ] **U5 — Validate evaluation machinery independently of candidates.** Extend
  W2/Q1's native evaluator with known-pass, known-fail, malformed, truncated,
  reasoning-only and infrastructure-failure controls for each selected scoring
  path. Bind extraction/tokenization, reference answers/rubrics and scorer source;
  retain original responses when required for independent replay. Keep model
  failures, incomplete outputs and invalid infrastructure distinct, report fixed
  planned/completed/scored denominators and missing outcomes, and cross-check
  selected frozen outputs independently before trusting a score change. Do not
  silently turn parser/timeouts or host contention into answer failures, discard
  losing candidates, or retry a closed owner-device trial.
- [ ] **U6 — Evaluate explicit reasoning/sampling profiles on unchanged Swift.**
  Use Q1's held-out tasks to compare enabled Thinking at `low`/`medium` against
  current `xhigh`, and assess publisher non-thinking sampler guidance separately
  from effort. Qualify any exposed control through native Settings, persistence,
  broker canonical count/generation bodies, exact context reservation and runtime
  effective-setting confirmation; preserve current defaults. Review reasoning
  replay policy as a separate context/quality tradeoff rather than dropping it
  silently. Count complete answers, truncation, useful-answer latency and RAM;
  shorter reasoning or fewer tokens alone cannot establish a win. No automatic
  penalty, forced-thinking, early-exit or model-switch heuristic is selected.
- [ ] **K1 — Qualify K4/V4 independently on X1/X2.** Complete the
  [three-task paired packet](docs/quantized-kv.md#first-practical-promotion-packet)
  for held-out retrieval, instructions and creative continuation: at least 65%
  lower measured KV storage, 128 MiB lower near-full peak private bytes and no
  more than 10% latency regression where timing is qualified. Retain explicit
  fresh-session FP16 recovery; qualify larger windows separately under M1–M3.
- [ ] Close selected-profile packed-attention, scalar/NEON, exact quantized
  checkpoint, prefix/adapter/cancellation and discovery-inventory gates. Charge
  scales, rotations, scratch and retained copies; preserve FP32 GDN/conv state.
- [ ] Qualify Prism's explicit `rsq4vf` fitted-value scale candidate against
  unchanged `rsq4`: bounded native encoding/whole-request cost and held-out
  quality under K1's source, power and budget controls. Its guarded rotated
  reconstruction improvement and unchanged record size do not qualify answer
  quality or a default change. See the [candidate contract](docs/quantized-kv.md#fitted-value-scale-candidate).
  Screening audit `#1` (`ClampedL2` reference identity) is closed; deferred item `#21`
  (calibrated K-cache mean centering) remains gated behind K1 baseline attribution,
  separate center-vector accounting and held-out answer parity.
- [ ] Only for material recurrent-state traffic/storage, compare FP16 recurrence
  with FP32, then at most one DAMP-inspired split; qualify trajectory error,
  convolution, rounding/RNG, restore and total benefit before extra factors.
- [ ] Only for a demonstrated weight/codec bottleneck, evaluate exact entropy
  coding, random access and decode cost against PTQ1; new codebooks/rotations/
  scales need bounded native layerwise cost and held-out quality preflight under
  [compression constraints](docs/inference-algorithms.md#compression-and-approximation-constraints).
- [ ] Investigate multi-head latent attention (MLA) only for a measured
  full-attention/cache bottleneck and with compatible trained/converted Swift
  weights, quantization/adapter identities and reviewed native Rust kernels.
  Scope conversion to the 16 GQA full-attention layers; preserve the 48
  GDN/convolution layers. Use a separate model/graph identity and explicit rollback.
  Compare against current GQA/K4/V4 for held-out complete-answer and long-context
  quality, prefill/decode latency and total memory, including workspace,
  checkpoints and retained copies, under [source, power and budget controls](docs/validation.md).
  Require quality acceptance and measured whole-request benefit before adoption.

## Hardware selection

- [ ] Extend checked hardware inventory and model-free plans with truthful
  device/driver/kernel, capacity, backend and effective-placement identities
  under [hardware policy](docs/hardware-capabilities.md).
- [ ] Qualify the bounded X2 thread screen and exact-candidate model smoke.
- [ ] Diagnose legacy headless startup; integrate/release the UCRT candidate
  upstream, qualify exact binaries with Unicode model paths, thinking,
  continuation, adapters and cancellation, then repin under the
  [candidate contract](docs/runtime-candidates/README.md).
- [ ] Qualify scalar/NEON, detected DOTPROD/I8MM, thread topology, wait and power
  choices, with independent fallback and source-bound code generation.
- [ ] Measure event waits versus polling and valid energy telemetry before
  selecting idle/active power policy or claiming energy benefit.
- [ ] Qualify the bounded deterministic profile catalog and live-resource
  admission against independently verified common evidence. Populate reviewed
  profiles and connect serving only after exact-source backend acceptance;
  the current catalog is an offline policy, not an execution permit.
- [ ] Connect truthful requested/effective diagnostics and native Auto controls
  only to qualified profiles and explicit supported subsets.
- [ ] Qualify request-boundary invalidation, complete state migration, driver/
  memory changes, device loss and strict-backend failure without silent replay.
- [ ] Qualify X1/X2 automatic operation, clean installation, pressure, upgrades
  and rollback independently; CPU fallback must retain its own acceptance.

## CPU kernels

- [ ] Qualify [bounded attention score growth](docs/cpu-kernels.md#bounded-attention-score-growth)
  in the later Prism source candidate: strict native Release analysis/linking,
  poisoned ordinary/tiled scores, reservation refusal, capacity ceilings and
  short tails; complete-model state/logits/adapters/cancellation and matched
  CPU-only request benefit. Account for larger early-context capacity within
  unchanged admission; source zero-store counts establish no process-RAM savings.
  The shared local run stopped before Prism compilation/tests. Release and repin
  only through the existing source/bundle workflow; preserve 0.1.28 rollback.
- [ ] Qualify the [ggml-inspired execution controls](docs/cpu-kernels.md#fusion-witnesses-precision-admission-and-few-token-tiles):
  actual completed fusion counts versus separate arithmetic, adapter/unsupported
  capability rejection before callbacks, pinned BF16/F32 precision boundaries,
  portable Q4/Q8 shared weight decoding and 2/3/7/9/17/24-token tails/recovery.
  Require native/feature-disabled strict Release analysis and focused controls,
  ARM64 scalar fallback, full-model state/logits/adapters and matched request
  benefit before promotion of the selected execution path.
- [ ] Qualify final-source exact/failure controls, strict native ARM64 Release
  analysis/linking, disabled-feature fallback, ARM64 codegen/resources, complete
  model state/logits/adapters/cancellation and matched CPU-only request benefit
  under [CPU acceptance](docs/cpu-kernels.md#numerical-and-resource-acceptance).
  Preserve matching helper/runtime source identities; apply [cross-generation promotion](docs/cpu-kernels.md#cross-generation-cpu-promotion)
  only within its reviewed shared-path scope.
- [ ] Publish and run the prepared [native CPU bundle comparison](tools/cpu-bundle/README.md):
  retain its fixed 0.1.23 baseline versus registry 0.1.25, FP16 cache, four workers
  and adapter 1.3. Keep thread screening and one-token attribution separate.
- [ ] Qualify CPU/prefill changes already consumed by the comparison baseline
  without requiring another repin; release newer candidates with exact source locks.
- [ ] Finish the checked current-source CPU comparison producer and native owner
  collection path for Prism's separate `cpu_candidate_check` collector and
  Runner's [offline admission](docs/cpu-kernels.md#current-source-comparison-admission).
  Freeze exact source/compiler/helper/executable locks, matching cache/context/
  tile/scratch and approved adapter, fixed complete workload, native resource/
  cleanup supervision and aggregate/per-arm power. Qualify the evaluator's
  positive/rejection controls and producer provenance; keep schema-2 receipts
  separate from the frozen 0.1.23/0.1.25 evaluator. Qualify short requests and
  long prefill independently; collector output alone is no packet.
- [ ] Qualify [scan-owned values and joined prompt projections](docs/cpu-kernels.md#scan-owned-packed-values-and-joined-prompt-projections).
- [ ] Qualify [joined BF16 recurrent projections and four-bit KV lookup](docs/cpu-kernels.md#joined-bf16-recurrent-projections-and-four-bit-kv-lookup):
  native ARM64 strict Release analysis, exact arithmetic/failure and disabled-feature
  controls, production linking, ARM64 codegen/resources, complete-model
  state/logit/adapter/cancellation parity and matched CPU-only request benefit.
  Retain matching helper/runtime identities and a verified runtime rollback.
- [ ] Qualify [ordered packed-KV code selection](docs/cpu-kernels.md#ordered-packed-kv-code-selection):
  exhaustive-search record identity, rounded first-code ties, atomic rejection,
  native/portable and feature-disabled controls, complete-model state/adapters/
  cancellation and matched CPU-only complete-request benefit before promotion.
- [ ] Qualify [joined BF16 prompt tiles and fused packed-KV scans](docs/cpu-kernels.md#joined-bf16-prompt-tiles-and-fused-packed-kv-scans):
  final-source native/portable and feature-disabled arithmetic, adapter, failure
  and recovery controls, strict Release analysis/linking, codegen/resources and
  complete-model parity. Use a matching-candidate complete-request packet for
  X1 benefit; the frozen X2 bundle lane and source work counts supply no verdict.
- [ ] Qualify [head-scoped GDN decode and packed value result checks](docs/cpu-kernels.md#head-scoped-gdn-decode-and-packed-value-result-checks):
  compiled native/portable arithmetic and failure controls, feature-disabled
  fallback, strict ARM64 Release analysis/linking, codegen/resources,
  complete-model parity and matched X2 CPU-only request benefit. Preserve the
  earlier failed Cargo diagnostic separately from source-bound tool-Job checks;
  retain the pinned runtime as rollback.
- [ ] Qualify [provisional GDN and prepared current-token KV](docs/cpu-kernels.md#provisional-gdn-and-prepared-current-token-kv):
  native/portable exact arithmetic, late-failure rollback, stale-record rejection,
  selected-cache and FP16 state/adapter/cancellation parity, codegen/resources and
  matched CPU-only complete-request benefit. Retain the pinned runtime as rollback.
- [ ] Qualify [prepared packed-KV queries and CPU preparation reuse](docs/cpu-kernels.md#prepared-packed-kv-queries-and-cpu-preparation-reuse).
- [ ] Qualify [parallel packed-cache prompt attention](docs/cpu-kernels.md#parallel-packed-cache-prompt-attention):
  serial/tiled causal logits and complete-state parity, sealed-page boundaries,
  raw-row identity, cancellation/late-failure rollback, strict Release analysis,
  worker/score-scratch bounds and matched X2 complete-request benefit. Preserve
  the explicit tiled-prefill opt-in and published runtime rollback.
- [ ] Qualify [zero-weight packed values, temporal GDN heads and joined Q8 tiles](docs/cpu-kernels.md#zero-weight-packed-values-temporal-gdn-heads-and-joined-q8-tiles):
  exact native/portable and feature-disabled controls, strict ARM64 Release
  analysis/linking, codegen/stack and resource review, complete-model
  logits/state/adapters/cancellation and matched CPU-only complete-request benefit.
  Keep ordinary K4/V4 decode, explicit tiled prefill and PTQ1 gates separate;
  retain a verified rollback while qualifying the selected execution path.
- [ ] Qualify [compact KV rotations and joined Q8 decode siblings](docs/cpu-kernels.md#compact-kv-rotations-and-joined-q8-decode-siblings):
  native ARM64 strict Release analysis, helper/runtime arithmetic and failure
  controls, feature-disabled execution, production collector linking, codegen/
  stack/resources, complete-model state/logits/adapters/cancellation and matched
  complete-request benefit. Include the independent integrated temporal GDN
  controls. Preserve the closed local supervisor-start failure separately; it
  ran no Cargo commands or tests. Preserve matching helper/runtime source identities
  and a verified rollback while qualifying the selected execution path.
- [ ] Qualify [scoped recurrent RMS weights](docs/cpu-kernels.md#scoped-recurrent-rms-weights):
  native/portable and feature-disabled arithmetic/failure controls, graph
  recovery, strict ARM64 Release analysis/linking, codegen/resources,
  complete-model state/logits/adapters/cancellation and matched CPU-only request
  benefit before promoting the selected execution path.
- [ ] Qualify [CPU preparation candidates](docs/cpu-kernels.md#cpu-preparation-candidates),
  [dispatch and attention preparation](docs/cpu-kernels.md#dispatch-and-attention-preparation) and
  [tiled dispatch and adapters](docs/cpu-kernels.md#tiled-dispatch-and-adapter-preparation), retaining each
  path's independent integration, numerical, fallback and benefit gates.
- [ ] Qualify [exact rotation butterflies, BF16 row pairs and recurrent validation](docs/cpu-kernels.md#exact-rotation-butterflies-bf16-row-pairs-and-recurrent-validation):
  final-source native ARM64 strict Release analysis and linking, independent
  scalar-bit/failure and disabled-feature controls, ARM64 codegen/stack/resources,
  complete-model state/logit/adapter/cancellation parity and matching-source
  complete-request CPU benefit. Keep singleton exact-prefill convolution separate
  from ordinary decode. Preserve matching helper/runtime source identities
  and a verified rollback while qualifying the selected execution path.
- [ ] Qualify [grouped dispatch, half widening and exact LUT execution](docs/cpu-kernels.md#grouped-dispatch-hardware-half-widening-and-exact-lut-execution).
- [ ] Qualify [complete-row SDOT, fixed attention lanes and bounded GDN phases](docs/cpu-kernels.md#complete-row-sdot-fixed-attention-lanes-and-bounded-gdn-phases).
- [ ] Qualify [scoped PQ2 validation, tiled Q/K/V and temporal GDN](docs/cpu-kernels.md#scoped-pq2-validation-tiled-qkv-preparation-and-temporal-gdn).
- [ ] Qualify direct PTQ1-by-Q8_K arithmetic separately from original-PQ2 and
  scalar controls; preserve packed scales, transforms, block boundaries and FMA order.
- [ ] Qualify explicit opt-in exact LUT integration. Extend cross-token LUT
  consumers only when attribution identifies useful work; include preparation,
  scratch, scalar/NEON identity and complete-request cost.
- [ ] Bound selected CPU screens to at most four NEON layouts on three shapes
  and at most two exact LUT candidates before choosing further integration.
- [ ] Integrate further selected cross-token and 8/16/32-token tiled-prefill
  consumers, preserving transformed bases, causal state, bounded scratch and
  rollback before timing the complete request.
- [ ] Use signed constant floor division only inside a measured parent hotspot
  with unchanged rounding/rejection and charged setup/dispatch costs.
- [ ] After the selected CPU candidate's acceptance, review
  [ciot T2P weight repacking](docs/cpu-kernels.md#external-layout-and-microkernel-references)
  for a demonstrated projection bottleneck. Resolve reuse permission before
  copying code; charge load-time preparation, peak duplicate weights, cache/file
  identity and cancellation. Preserve all PQ2 codes, signed extremes, scales and
  floating order; do not repeat affine-hoisting work or assume the old tail-path
  traffic problem applies to current Prism.
- [ ] For a selected speech/Prism projection bottleneck, review
  [rten-gemm and KleidiAI packing/tile interfaces](docs/cpu-kernels.md#external-layout-and-microkernel-references)
  for focused Rust reuse/ports. Retain checked DOTPROD/I8MM dispatch, baseline
  fallback, bounded storage and independent arithmetic/state controls; source
  availability and upstream microbenchmarks supply no device benefit verdict.
- [ ] Capture a matched complete-request pair and record promotion or hold;
  update README measurements only after a freshly declared, source-bound capture
  with aggregate/per-arm power and cleanup observations. Surveyed source controls `#22`,
  `#26` and `#28` retain the owning compiled CI and X1/X2 device gates above; deferred CPU
  kernel candidates `#23`–`#25`, `#27` and `#29` stay gated behind baseline bottleneck
  attribution and per-kernel numerical/tail/resource acceptance.

## Adreno Vulkan

- [ ] Diagnose X2 `VK_ERROR_INCOMPATIBLE_DRIVER` and the separate GDI32 lookup
  error 126; establish their causes before further inference claims.
- [ ] After driver/access and packed-projection acceptance, implement and compare
  bounded fused and [decomposed attention](docs/adreno.md#decomposed-attention-candidate)
  in the Rust GPU backend. Preserve global normalization/FP16 probability rounding,
  causal state, exact stage dependencies, complete cache identity and explicit
  scratch/transfer admission; require path witnesses, independent stage/state
  controls and whole-request benefit before selecting either implementation.
- [ ] Qualify the opt-in native ARM64 OpenCL original-PQ2/Q8_K projection
  pilot under unchanged mitigations, including owning hosted analysis/linking,
  exact input/kernel/device identities, every-block scalar replay and terminal
  cleanup. Prepare a separately locked Vulkan candidate under the
  [Adreno contract](docs/adreno.md); neither path admits GPU serving yet.
  Implemented in source: `AdrenoFamily::{X1_85, X2_90}` and
  `OpenClKernelKind::{BlockDots, Gemv4Row, Gemv8Row, Prefill4Col, FwhtLocal1024}`
  via `opencl_pq2_kernel_source_for` and `opencl_pq2_projection_with_kernel` in
  `src/windows-process/src/native/opencl/projection.rs` (awaiting physical X1
  hardware parity and GPU serving qualification).
- [ ] Bind operation support, effective placement, fallback and transfer ledgers
  to selected graph shapes. Qualify PQ2/PTQ1 and no-/approved-LoRA independently,
  including full-state lifecycle and resources.
- [ ] Compare matched CPU, GPU offload and qualified Hexagon paths with complete
  startup, preparation, transfers, synchronization and request attribution.
- [ ] Optimize measured bottlenecks only, including packed kernels, reusable
  weights, cache/lifetime accounting and bounded scratch.
- [ ] Qualify full-device benefit, pressure/device-loss/cancellation recovery,
  rollback, resources and safety before selecting a GPU profile. Keep conditional Adreno
  GPU, async storage and DCVS/power-scaling gates from screening items `#39`–`#44`
  deferred behind driver/access resolution and matched host/device attribution.

## Hexagon NPU

- [ ] Qualify imported-secondary-arena sealing in the native resident path:
  refusal before writer callback, byte/handle/report mutation or queue submission;
  unused arenas stay dirty/unimported and preprepared views remain usable after
  import. Verify primary append-after-warm flushes and independent shape identity,
  native ARM64 strict Release analysis, owning compiled controls and signed-device
  coherency/lifecycle acceptance before device-profile promotion.
- [ ] Close applicable [signed-runtime/SDK/Windows ABI prerequisites](docs/hexagon-source-audit.md)
  and independent raw-byte replay before expanding a selected device workload.
- [ ] Qualify the implemented six-arena/168-projection `native-q2` placement:
  exact selected tensors, adapter/full-state parity, arena admission, load costs,
  fallback, persistent lifecycle and complete-request benefit under [Hexagon policy](docs/hexagon.md).
- [ ] Qualify the implemented `HTP v73` X1 composite placement and telemetry in
  physical X1 hardware runs: Prism `bonsai-hexagon-native` (`HtpArchProfile::v73_default`,
  partial secondary arenas), `bonsai-hexagon-runtime` (`X1_V73_MAX_NPU_LAYERS = 27`,
  `plan_x1_v73_three_processor_placement` partitioning `168` NPU `Q2_0` + `232`
  GPU `Q2_0` projections), `CompositeProjectionExecutor`, `CompositeTelemetry`,
  `HybridPrefillCoordinator`, Runner `bonsai-broker` `"hexagon-native-q2-partial"`
  (`v73`/`v81`) startup readiness admission, Runner `1|4|8|12` row-worker admission
  with Snapdragon X1 `8`-worker live host default, and Prism per-group
  `Transforms::project_group_into_with_executor` CPU sibling dispatch under partial
  executors under [Hexagon policy](docs/hexagon.md).
- [ ] Qualify the pinned model-thread initialization/readiness handshake
  and `startup_projection` facts, including failed initialization, CPU/external
  executors, actual resident counts/bytes and Runner's bounded observation
  parser. Preserve the legacy readiness contract for an explicit 0.1.27 rollback.
- [ ] Qualify existing full-FFN chains, sibling packets, multi-tile N=1 and
  tile-major reuse independently. Implement true N>1 prefill only with a supported
  signed solver, causal state, bounded VTCM/DMA and measured total-cost benefit.
- [ ] Integrate remaining RMSNorm/RoPE/attention/conv/GDN VTCM chains only where
  signed operation support, transformed bases, full-state ownership and measured
  benefit justify placement.
- [ ] Extend guarded-Q2/HMX projection controls to broader live activations,
  shapes, tokens, adapters, logits and complete state. Preserve original Q8_K and
  per-128 FMA, every-limb/plane comparisons, selected DSP identity, cleanup and
  source-bound numerical acceptance under [packed arithmetic](docs/hexagon.md#packed-arithmetic-and-graph-state).
- [ ] Qualify [bounded resident projection lookup](docs/hexagon.md#resident-identity-and-bounded-lookup):
  native-Q2/HMX shared-storage geometry, name-sensitive selection, stable handles,
  sibling reuse and complete metadata reservations; native/fallback controls,
  strict Release analysis/linking, signed execution, full-model state/adapters,
  cancellation/resources and matched complete-request benefit.
- [ ] Qualify persistent guarded-bridge session admission through the implemented
  explicit partial placement and whole-request reservation or a separately
  reviewed supported lifetime/fused design. Account for all prompt/output/attempted
  operations; resets must not replenish the terminal cap. Keep the default native-Q2
  persistent-session policy separate from guarded/model-probe controls.

### Promotion sequence

- [ ] **NPU-1 — Prepare the qualification packet.** Freeze Runner/Prism/compiler/
  executable, original-PQ2/adapter, X2 SKU, driver/runtime/catalog and DSP identities;
  close applicable prerequisites and retain independent raw inputs, outputs,
  hashes, oracles, cleanup codes and driver-owner lifetime policy.
- [ ] **NPU-2 — Qualify arithmetic and adapters.** Compare every offloaded result,
  final logits and complete state with the same-artifact CPU oracle for varied
  prompts/shapes/generated history, selected scalar/NEON/four-worker policy,
  loaded-zero identity and scale-1.3/change-history cases. PTQ1/Q8_0 stay separate.
- [ ] **NPU-3 — Qualify placement, resources and lifetime admission.** Bind actual
  DSP/VTCM/DMA/mapping/cache behavior to tensors; charge host/device resources
  outside Job commit. Reserve whole-request work and qualify the guarded bridge's
  24,000-attempt lifetime or a separately reviewed serving design before admission.
- [ ] **NPU-4 — Qualify persistent serving and failures.** Cover fresh/reused/
  edited prefixes, full context/output reservations, repeated requests,
  cancellation before/after submission, Stop/Retry/reset, queue saturation,
  unsupported shapes, pressure, allocation/mapping failures, loss and stale caches.
  Require complete committed-state rollback and verified teardown; preserve
  uncertain buffers until terminal completion or supervised exit.
- [ ] **NPU-5 — Establish end-to-end benefit.** Select measured bottlenecks and
  compare matched CPU/default settings with repeated complete answers, prefill/
  decode, resources, quality and comfort. Charge startup, transfers, RPC and
  synchronization; require the [promotion benefit](docs/validation.md#promotion-criteria)
  and eligible matched power conditions.
- [ ] **NPU-6 — Qualify the explicit application profile.** Release/pin supported
  Rust helpers and signed kernels; verify broker/launcher placement/readiness,
  authentication/counting/one-slot/cancellation, child-only environment, override
  rejection and isolation beyond one Job. Cover clean install, absent/changed
  drivers/runtime, upgrades, pressure and original-PQ2 CPU rollback.
- [ ] **NPU-7 — Independently review and promote the scoped release.** Score all
  correctness, adapter, lifecycle, resource, quality, benefit, packaging/isolation
  and rollback gates against one settled source/release/device tuple. Retain
  numerical/human review, supported subsets and no-go cases; requalify after
  relevant identity changes.
- [ ] **NPU-8 — Qualify later expansion separately.** Add common-evidence
  request-boundary CPU/GPU/NPU selection only after backend/resource/fallback
  acceptance. Qualify HMX, PTQ1, X1 and Adreno independently of an explicit X2
  partial-HVX profile.
- [ ] Complete sustained qualification only in an eligible environment where
  setup, every arm, power, cleanup and retention fit the original allowance;
  leave it pending when the laptop budget cannot support it. Keep conditional Hexagon
  NPU, Adreno GPU, async storage lifetime and DCVS/power-scaling gates (`#39`–`#44`)
  deferred unless predeclared workload evidence justifies reopening them under `NPU-1`–`NPU-8`.

## August to November 2026 upstream screening follow-up

Track remaining qualification, conditional candidates and housekeeping from the
51-item screening (`PrismML-Eng/llama.cpp@6447906bf96c1a023980d3453fb59daa6af3bb23`,
`ggml-org/llama.cpp@a956a852c9c8bc1fb07206364838ddef299178c0` and
`vllm-project/vllm@5e51c0ca1041168de7aa9e096982aebaeebabd38`) in the
[screening ledger](docs/research-register.md#august-to-november-2026-screening-ledger).
Reference audit `#1` (`ClampedL2` GDN Q/K normalization) is closed.

### Repin, CI and device qualification of landed screening controls

- [x] Publish a Foundation release bundle from reviewed `prism-inference` `main`
  (`0.1.30`, `89df9aaff9e580d66679f2befe94d471bcf4fc7f`, including
  `129fae8..c0893f8`) and repin Runner past `Prism 0.1.28` under the
  [distribution guide](docs/releases.md) so installed and daily-use builds consume
  the new runtime checkpoints, GDN guards, memory ledger and worker cancellation
  (`#2`–`#7`, `#9`–`#13`, `#15`, `#19`–`#20`, `#22`, `#25`–`#26`, `#28`,
  `#30`–`#34`, `#36`, `#45`–`#51`).
- [ ] Complete hosted `source-checks` and `foundation` compiler, strict Release
  analyzer and full regression-suite verdicts in both repositories under
  [promotion criteria](docs/validation.md#promotion-criteria); local merge checks
  ran scoped Clippy and targeted `--exact` tests only.
- [ ] Qualify landed controls on actual weights and X1/X2 hardware within
  predeclared 5-to-15-minute AC/battery-logged allowances:
  - [ ] `#8`: Exercise live long-prefill HTTP Stop/disconnect,
    `Worker::settled_within` and `speech-tts` cooperative `CancelHook` cancellation.
  - [ ] `#14`: Compare `SwiftExecutionShape` and `SwiftExecutionMemoryLedger`
    conservative commit charges against measured Windows Job-object commit across
    M1–M3 (`8K`–`256K`).
  - [ ] `#30`–`#34`, `#36`: Measure complete-cycle suffix speculation
    (`SpeculativeStats` proposed/verified/accepted histograms, `replay_steps` and
    `VerificationLimits` stop/cap truncation) against single-token decode before
    enabling speculation by default.
  - [ ] `#49`–`#51`: Qualify live `bonsai-chat` streaming Markdown prefix retention,
    incremental TTS segmentation and `bonsai-tools` calibration/probe receipts.

### Conditional screening items deferred pending bottleneck or prerequisite evidence

#### Memory, loading and approximate cache

- [ ] **#16 — Aligned loading and upload staging reuse.** Process largest tensors
  first, reuse bounded aligned staging and shrink excess capacity once accelerator
  upload or cold-load staging demonstrates a measured bottleneck under the
  [memory contract](docs/swift-memory.md).
- [ ] **#17 — Optional Windows unbuffered direct I/O.** Prototype an aligned
  `FILE_FLAG_NO_BUFFERING` handle with logical seek semantics, CNG digest continuity
  and explicit buffered fallback only if cold model loading is I/O-bound and beats
  the existing mapped path.
- [ ] **#18 — Owned tensor-range mapping and pinning.** Merge owned tensor byte
  ranges and exclude unrelated spans instead of mapping `first..last` when
  shared-weight or accelerator pinning requires it, preserving Windows handle and
  reparse-point identity guards.
- [ ] **#21 — Calibrated K-cache mean centering.** Evaluate subtracting a
  per-layer/head/channel key mean on rotated `K4` only after `K1` baseline
  attribution, with separate center-vector storage/identity accounting and held-out
  answer parity under [quantized KV](docs/quantized-kv.md#qualification-contract).

#### CPU kernel tuning

- [ ] **#23 — Weight-stationary multirow decode.** Evaluate reusing quantized
  activation blocks across multiple output rows on X1/X2, measuring register
  pressure and L1/L2 cache behavior while preserving per-block reduction order.
  Implemented in source (physical X1 throughput, parity and power qualification
  open): Prism `16`-row `PQ2`/`Q8_0`/`PTQ1` SIMD tiles, X1 `3`-head/`2`-head
  grouped attention and `[token][head]` F16-packed `4`-head decode,
  `bonsai-speech-tts` `4`-row `Q8_0`/`Q4_K`/`Q6_K` SIMD tiles in
  `src/speech-tts/src/engine/packed.rs`, `bonsai-speech-asr` bit-exact non-FMA
  `1x4`/`4x8` `F32`/`Q8_0` SIMD tiles, 4-frame `mixed_fft_4` frontend, batched
  encoder/causal prefill and `row_workers` in `src/speech-asr/src/`, X1 `(8, 12)`
  TTS / `(8, 10)` ASR thread budgets / `SpeculationPolicy` /
  `"runner-x1-cpu-evaluation-plan"` (`d8-p12`) in `cpu-policy` / `bonsai-tools`,
  Prism server `--cpu-family x1|x2`, `--spec-type none|suffix`, and
  `--draft-max 1..=8`, Runner `1|4|8|12` row-worker admission with Snapdragon X1
  `8`-worker live host default, Prism per-group CPU sibling dispatch under partial
  executors, Prism tiled (`ProjectionTile`) / head-major 4-query parallel
  `SwiftVisionEncoder` (with parallel `LayerNorm` & bicubic resize), and
  `bonsai-chat` scanline RGBA box downscaling / chunked BGRX normalization.
- [ ] **#24 — Token-count projection crossover tuning.** Benchmark separate 1-token
  decode, short verification block (`2`–`9`) and larger prefill tile crossovers with
  odd-tail retention, keeping any rounding change behind an explicit profile gate.
- [ ] **#25 — Remaining shared activation-preparation consumers.** Extend shared
  transform-and-quantize preparation to remaining recurrent sibling projections
  sharing identical input, basis, width and quantizer.
- [ ] **#27 — Fused GDN `conv1d + SiLU + ClampedL2` pass.** Evaluate fusing causal
  3-tap convolution, `SiLU` and `ClampedL2` Q/K normalization alongside joined BF16
  alpha/beta to cut intermediate traversals while preserving exact `ClampedL2`
  numerics. Implemented in source in Prism (`bonsai-neon-dot` / graph kernels;
  physical X1 hardware measurements and parity qualification remain open).
- [ ] **#29 — Consumer-layout attention probabilities and tail balancing.** Emit FP16
  attention probabilities directly in consumer layout and split oversized final
  head/token tasks without changing the FP16 boundary or reduction order under
  [CPU kernels](docs/cpu-kernels.md#cross-generation-cpu-promotion).

#### Speculative decoding extensions

- [ ] **#35 — Shared target embeddings and LM head for learned drafters.** Share
  transformed embeddings/head with strict tokenizer, basis, scale, adapter and
  feature-order identity checks only after a compatible published Swift drafter is
  available under
  [inference algorithms](docs/inference-algorithms.md#hybrid-cache-reuse-and-speculative-state).
- [ ] **#37 — Checkpointable delayed recurrent updates (`ReplaySSM`).** Include ring
  cursors, pending inputs and materialization boundaries in state identity only if
  measured state-copy traffic justifies `KVBuffer`/`SpecLA`-style delayed recurrence.
- [ ] **#38 — Transactional grammar speculation.** Bind grammar state advancement and
  rejection-sampling masks into the atomic KV/GDN/convolution/sampler rollback
  transaction before allowing speculative decoding alongside grammar constraints.

#### Adreno Vulkan, Hexagon NPU and power policy

- [ ] **#39 — Native packed Vulkan (`PTQ1` / integer-dot) kernels.** Gate behind
  `VK_ERROR_INCOMPATIBLE_DRIVER` diagnosis and Qualcomm matrix-conversion capability
  checks with explicit fallback under [Adreno](docs/adreno.md).
- [ ] **#40 — Resource-pressure fused versus decomposed GPU attention.** Select
  row/head tile shapes from register and local-memory pressure to prevent spills and
  Windows GPU TDRs while preserving global normalization and FP16 probabilities.
- [ ] **#41 — Hexagon `PQ2`/`PTQ1` tile packing, VTCM sizing and IOVA placement.**
  Tailor host repacking and mapping placement to actual queried VTCM capacity and
  Swift's `PQ2`/`PTQ1` layouts under [Hexagon](docs/hexagon.md).
- [ ] **#42 — Pipelined DMA, HVX preparation and HMX chunkwise GDN.** Gate under an
  explicit separate numerical profile (HMX FP16 intermediates do not preserve the
  FP32 GDN contract) with a small-token CPU/HVX fallback.
- [ ] **#43 — Device-side explicit async storage fences.** Extend implicit-storage
  dependency tracking and terminal-completion transfer batching to Adreno/Hexagon
  async arenas (CPU `GraphAllocator` multi-hop view/alias liveness is already landed).
- [ ] **#44 — Sustained versus burst accelerator DCVS/HMX power policy.** Compare
  sustained versus burst clock/power modes on X1/X2 using complete-answer latency,
  energy/charge delta and thermal comfort rather than peak microbenchmark clocks.

## Survey ideas outside the current laptop backlog

- [ ] **F1 — Conditional research.** After a useful verifier and compatible
  published auxiliary exist, screen trained parallel/phrase proposals with
  admitted native resources. After a useful
  fixed chain, consider scheduling, rejected-suffix conditioning or overlap only
  with required target features and independent hardware capacity under the
  [research scope](docs/research-register.md#research-catalog).
- [ ] Reopen exact full-precision weight codecs only for a suitable measured
  bottleneck; exact FP16 KV codecs only for material selected-context traffic;
  transfer/inactive-session codecs only for a selected capacity or transfer need.
- [ ] Reopen approximate checkpoint omission only if exact reuse and suffix
  recomputation cannot meet a demonstrated need. Wide trees, SVD caches and sparse
  indexes need a measured bottleneck and bounded native benefit.
- [ ] Consider [learned corpus retrieval](docs/research-register.md#conditional-learned-corpus-retrieval)
  only after embeddings are justified and vector search is a measured bottleneck;
  qualify passage relevance, complete-task cost, resource bounds and fallback.
- [ ] After portable-update acceptance, design a separate signed-metadata trust
  extension using the [tough/TUF reference](docs/research-register.md#reliability-and-distribution-references).
  Specify publisher key custody/rotation, trust bootstrap, expiry/clock and offline
  behavior, rollback/freeze rejection and recovery before changing release
  publication or the installed updater. Current checksums remain within GitHub's
  trust boundary; this research does not establish signing or authenticity.

## Future Qwen MoE migration

- [ ] After Qwen 4 and the corresponding Bonsai upgrade are available,
  authenticate actual architecture/artifacts and establish native Rust support.
- [ ] Evaluate justified expert capacity/tiering and download-space preflight
  under the [conditional MoE design](docs/model-selection.md#conditional-future-bonsai-moe-design),
  then qualify state/resources, quality, Stop/Retry, cleanup, native application
  behavior and practical benefit with explicit Swift rollback.

## Deferred Prism migration

- [ ] Reopen broader Prism work only for a Swift dependency or chosen milestone;
  close reference provenance/backend, geometry/performance and Windows-source
  inventories under [conformance](docs/prism-upstream-tests.md).
- [ ] Replace remaining numerical/codec operations, callbacks, backward graphs,
  importance encoders and owned `qb-*` APIs when selected.
- [ ] Replace selected chat/parser/template/tokenizer and capability/server
  conversions with complete independent source contracts.
- [ ] Replace selected acquisition/CLI/model/session/architecture/state/concurrency
  APIs with independently qualified native implementations.
- [ ] Close selected DSpark, LoRA conversion/merge/inference, grammar and safe
  public ABI work, preserving approved corrections and compile-only scope.

## Deferred dependency experiments

- [ ] After sequential-decoder and ASR acceptance, compare optional Earshot
  speech-boundary ranking against the existing RMS chooser under the
  [sample-conservation contract](docs/streaming-resources.md#candidate-asr-boundary-selection). Preserve every PCM
  sample, the 30-second recognizer limit and current decoding path; charge detector
  state/work, transcript boundary accuracy, cancellation and complete import cost
  within one predeclared workload before any adoption.
- [ ] Re-evaluate [Wild linker](https://github.com/wild-linker/wild) only after
  an upstream release provides full Windows support, including native
  `aarch64-pc-windows-msvc` linking. Verify MSVC/PE-COFF, static CRT and debug-symbol
  compatibility, then compare final-link and end-to-end build times against the
  current MSVC/LLD paths under [source, power and budget controls](docs/validation.md).
  Require correctness, packaging acceptance and measured build-time benefit before adoption.
- [ ] Reopen MXC supervisor, wgpu backend removal or archive splitting only for
  demonstrated need and support/device/fallback/build-size acceptance under the
  [dependency audit](docs/dependency-audit.md).
- [ ] Integrate candidate speech SIMD projections, parallel projections,
  four-frame streaming and FastConformer transpose into serving before model
  qualification (source support implemented for `bonsai-speech-tts` and
  `bonsai-speech-asr` `--threads` / X1 cluster-aware budgets, bit-exact non-FMA
  SIMD kernels, 4-frame frontend and batched causal prefill/encoder; model and
  device qualification remain open); qualify direct `memchr` transport scans
  independently.
- [ ] Qualify [SIMD-JSON tape reuse](docs/dependency-audit.md#current-candidate-and-pending-gate)
  in owning main-only CI: Unicode/duplicates/unknowns/shapes/bounds/canonical bytes,
  at least 5% lower equal-weight geometric-mean time and no case over 5% slower in
  both cold/reused screens, including scales, before changing the dependency choice.

## Optional agents

- [ ] After creative-chat qualification, evaluate OpenCode, then Aider, through
  a bounded authenticated inference bridge.
- [ ] Qualify MXC ProcessContainer isolation for agents/plugins/hooks/descendants
  with `allowDaclMutation:false`, file/reparse/handle/subprocess/network bypass
  checks and no unsandboxed fallback. Qualify a separate-kernel VM tier separately.
