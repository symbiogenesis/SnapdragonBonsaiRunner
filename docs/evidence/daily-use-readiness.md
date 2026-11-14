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
PSEC host-loopback still reaches other local services, same-user interference
remains possible, and private scratch/log storage has no disk quota. Broker-only
transport and complete restricted inference authority remain open.
