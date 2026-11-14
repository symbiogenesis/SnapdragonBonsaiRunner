# Daily-use readiness and qualification limits

This record supports the open C1/L1/R2, A1/R1, D1 and Q1 gates in
[PLAN.md](../../PLAN.md). It records source corrections and a failed local
verification workload on 2026-11-14; it is not an application acceptance receipt.

## Existing hosted blockers

- Runner [Foundation 43628943863](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43628943863)
  failed its planner's cache-environment fixture at published `a0873e69`.
  Later local cache-fixture corrections need published-source qualification.
- Prism [Foundation 43580578857](https://github.com/symbiogenesis/prism-inference/actions/runs/43580578857)
  at `27f6848f4d61a7be22c8898838c985b7bbb81eba` failed three Q8 sibling
  controls with `InvalidWeightScale`. Their fixture incorrectly passed zero
  scales to the stricter wire packer; native signed-zero acceptance and strict
  wire rejection remain separate. The correction changes tests, not arithmetic.
- Prism [Source checks 43580578896](https://github.com/symbiogenesis/prism-inference/actions/runs/43580578896)
  reported missing exported PowerShell help and incorrect workflow error-output
  typing. The source corrections require owning analyzer acceptance.

## Local source checks

Runner workflow startup policy and all 16 focused startup fixtures passed.
The owned-language guard, the exact cache fixture
`mixed_case_names_have_same_build_identity`, and actionlint 1.7.12 with five
syntax fixtures passed. Prism's pinned workflow preflight passed its 11 negative
controls and process exit controls. PSScriptAnalyzer 1.25.0 was unavailable.

The native Release workload selected two `bonsai-windows-process` environment
controls and nine `bonsai-tools` controls: two startup summary/receipt checks,
descendant cancellation and surviving-child rejection, three speech packet/schema/
clip-identity checks, partial-download recovery and pinned automatic setup.
It used `bonsai-ci rust-check --offline`, an external maintenance target and
exact `--test-case` selections with `--locked --release --all-features --lib`.
Compiler: `nightly-2026-11-12-aarch64-pc-windows-msvc`, rustc
`3ef0c22cdb86ac42dc6b8f0f1ec36857daa6f2a1`.

Rust sources were at Runner `de874c2a` during this workload. The environment
selection exhausted its 50-second command allowance while compiling dependencies;
no test ran and the native supervisor reported stopped descendants. Four tools
selections then exited 101 during compilation with E0277 at
`speech_evaluation/evaluate/tests.rs:225`: sha2 0.11's digest array does not
implement `LowerHex`. The runner began its fifth selection before its remaining
tree was explicitly terminated. No selected Rust test passed; unexecuted work
remains pending. Commit `bc379197` corrects the test's independent bytewise SHA
encoding but was not compiled in this closed workload. It inherits no verdict.

The predeclared aggregate allowance was five minutes, with a three-to-four-minute
estimate using the existing cache. Execution was stopped early at the actionable
compile failure. The total through process termination, selected log cleanup,
summary retention and the final power observation was 274 seconds; this clock is
closed. Native read-only power endpoints observed AC disconnected,
battery saver off and charge 91% before work, then AC disconnected, saver off and
charge 90% after process termination and evidence cleanup. Equal endpoints do not establish continuous
stability. There were no comparison arms, model loads, inference, audio playback
or performance claims. The failed run is not retried or continued as acceptance.
The native timeout and explicit tree termination outcomes are recorded above;
they do not prove application/model cleanup. Raw build chatter and duplicate
compiler logs are unnecessary for this diagnosis; this summary retains the
actionable error and skipped-work observations.

## Remaining acceptance

Publish and qualify one reviewed Runner/Prism source tuple before actual-model
work. A1/R1 still require the exact-source analyzer/link/package receipts and
actual native UI Stop, acknowledged cleanup, Reload, explicit Retry, preserved
drafts/history and rollback. D1 still needs real Schannel/proxy and clean-machine
setup/update acceptance. Q1 needs complete captured answers/clips and independent
human review; fixed speech-packet capture support supplies no listening verdict.

The child-environment snapshot guard closes one credential-admission race.
At that source, PSEC host-loopback still reached other local services, same-user
interference remained possible, and private scratch/log storage had no hard
storage bound. Broker-only transport and complete restricted inference authority
remain open; the follow-up below scopes the new parent-owned log cap.

## Follow-up source fixes and owner handoff

The reviewed follow-up candidate adds five source corrections:

- Local focused Rust checks stop after their first failed selection and retain
  later selections as pending. Hosted ordinary verdict coverage stays intact.
- Timed artifact intake shares its deadline with cached hashing, recovery,
  publication and cleanup; a cache hit cannot bypass an expired allowance.
- Preparation cleanup has one explicit attempt, with cleanup failure terminal.
- Contained-process destruction cannot renew a previously attempted stop's wait.
- Sandboxed HTTP workers inherit stdout/stderr pipe writers instead of disk
  handles. Trusted parent drains retain at most 16 MiB per stream, report capture
  failures and share the existing cleanup allowance. Offline decoder pipes and
  `NUL` stderr stay separate. These are source changes, not native enforcement
  or measured overhead verdicts.

One new source-check workload at Runner
`cad8b6484fa30021523899082021621ba3b239c5` declared five minutes total, estimated
three to four minutes with the existing cache, and reserved sixty seconds for
cleanup, retention and power. It started at `2026-11-14T20:03:23.0421081Z`.
The owned-language guard passed (exit 0). Selected-package `cargo fmt --check`
failed (exit 1) on an unwrapped assertion in `startup_download.rs`; execution
stopped before any Cargo compilation or Rust test. Pinned formatting subsequently
corrected that source. The workload was not rerun and supplies no compiled verdict.
The cached native supervisor was not invoked; its SHA-256 was
`1825d30584b84678b72439f49e4b9cee2772e33cb1a0a224795ffceaa9156d02`.
No model, native sandbox, UI, audio, benchmark or comparison arm ran.

All seven selected Rust cases remain pending in their owning Foundation lanes:

- `native::cleanup::tests::explicit_cleanup_failure_never_renews_the_drop_allowance`
- `native::output::tests::sandbox_capture_retains_exact_prefix_and_rejects_first_excess_byte`
- `native::output::tests::sandbox_capture_distinguishes_read_and_disk_exhaustion_failures`
- `native::output::tests::sandbox_capture_observes_panicked_drain_before_worker_exit`
- `command::checks::tests::local_focused_failure_stops_and_retains_pending_selections`
- `startup_download::tests::download_timeout_bounds_verified_cache_hits_and_preserves_unfinished_cleanup`
- `startup_download::tests::download_timeout_stops_before_retry_or_publication_after_transfer`

UI compilation and
`startup::tests::preparation_cleanup_failure_is_terminal_after_one_stop_attempt`
also remain pending. Three ignored ownership controls cover stdout flood, stderr
flood and normal-exit log draining. They are not added to the existing fixed
six-case sandbox workload; source-bound execution admission for those additional
controls remains open.

The paired Prism source is `ea753599bf221e2c320e595d628be23c9b5e3afa`.
Runner's unchanged 0.1.28 runtime remains bound to
`99671be7ab2be01e406a11b0c3d6caede398d96f`; these identities are distinct.
The owner chose to operate CI. Publish reviewed local `main`, then use the
[source and acceptance handoff](../daily-use-review.md#source-and-acceptance-handoff)
for Source checks (including main-branch dependency auditing), Foundation and
hosted-only native acceptance preparation. Require the expected source/compiler,
strict-analyzer, production-link, workflow and inventory receipts before device work.
Existing application, distribution and complete-answer/listening packets still
need actual-system captures and independent human review. Scratch/PSEC-private
temp quotas, broker-only transport and same-user isolation remain open.

Native power endpoints observed AC disconnected, battery saver off and charge
81% before and immediately after the stopped command (2.804 seconds). After
selected summary retention and scratch cleanup, the final native observation was
AC disconnected, battery saver off and charge 80%. UTC elapsed through cleanup,
final power and the record correction was at most 130 seconds; the command's monotonic continuation was
unavailable after it exited. Equal endpoints prove no uninterrupted stability.
No owned compiler/test process was started. The single scratch summary was
consolidated here and removed; no raw build output or duplicate receipt is retained.
