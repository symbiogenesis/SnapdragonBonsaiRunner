# CI failure audit

## Latest ten-run follow-up

`gh run list --status failure --limit 10` on 2026-11-14 returned the following
**ten runs across all workflows**: five Foundation, three Hexagon and two Windows
Release runs, spanning 18:51 through 21:33 UTC. Cancelled runs are outside this
sample. All failed jobs were inspected; Release run 43628943271 includes both
failed attempts. Six runs overlap the earlier audit below. Four newer runs expose
different failures after the earlier cache and download-handle corrections.
The starting checkout was `dcb66cb9`; concurrent, unrelated work through
`79a520bb` was preserved during integration.

| Run | Workflow / source | Confirmed root cause and prevention |
| --- | --- | --- |
| [43643423171](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43643423171) | Foundation / `94c25987` | Linux maintenance lib-test compilation rejected five unused Windows-only helpers/fields. Match platform selection to consumers and exercise portable accounting/verdict/identity policy in its existing tests. The aggregate rejection follows the planner failure. |
| [43643423275](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43643423275) | Hexagon / `94c25987` | Public ABI intake referenced nonexistent `hexagon-public-qaic-41_171_859_264`; the archive is `hexagon-public-qaic-41171859264`. Correct the path and validate all seven pinned retained inputs before downloads or compiler work. |
| [43641937848](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43641937848) | Foundation / `0d1f2c7f` | Same five Linux compilation diagnostics as 43643423171. Earlier cache-environment fixtures passed; this is a new failure reached afterward. |
| [43641937940](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43641937940) | Hexagon / `0d1f2c7f` | Same retained-source pathname defect as 43643423275. Both ABI artifacts contain `commands: []` and only the two downloaded public headers, placing failure before compiler selection. |
| [43633113374](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43633113374) | Foundation / `3d3ffe77` | F2 below: fixture removed an environment variable with `$null`, which PowerShell 7.5+ treated as an empty value. Preserve absent versus empty values with actual-null deletion. |
| [43631697910](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43631697910) | Windows Release / `8aca8b02` | R4 below: digit grouping of opaque test IDs failed Clippy after successful formatting. Remove preference lint groups from required verdicts; retain behavior and explicit safety checks. |
| [43631695104](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43631695104) | Foundation / `8aca8b02` | Same F2 cache fixture failure. No product-check verdict was reached. |
| [43631695096](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43631695096) | Hexagon / `8aca8b02` | H2 and H4 below: oversized dependency preparation exceeded 180 seconds; public input identity intake overlapped its writable download handle. Narrow dependency owners and close the writer before guarded reads. Both source corrections already existed. |
| [43628943271](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43628943271) | Windows Release / `a0873e69` | R3 below: valid temporary draft URL was rejected after creation, then again on attempt 2 reread. Admit that URL only for the exact draft; keep source, metadata and asset identity checks. Correction already existed. |
| [43628943863](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43628943863) | Foundation / `a0873e69` | Same F2 cache fixture failure. |

The malformed pathname first appears in migration commit `03c9f155`; its parent
PowerShell control has the correct spelling. There is no evidence establishing
that a later cosmetic-lint repair caused this path defect. The new Foundation
diagnostics concern `Version::next_for_test`, `Bridge::elapsed_seconds`,
`apply_budget`, `carries_workload_verdict` and `Workload::{name,scope}`. The repair
does not disable compiler warnings or broadly allow dead code.

### Process corrections and remaining limits

Clippy's correctness, suspicious-code and performance groups remain required,
along with explicit safety and ownership checks. Style, complexity, pedantic,
nursery and Cargo preference groups are explicitly allowed even under
`-D warnings`. This avoids replacing each new cosmetic failure with another
individual exception. Compiler warnings, Rustfmt, workflow syntax, pin integrity
and behavioral tests retain their existing responsibilities.

The outgoing-tip checker now selects Rust formatting when root or nested
`rustfmt.toml` or `.rustfmt.toml` changes. The focused selector also checks that
unrelated TOML and example filenames do not select formatting. Existing
immutable-tip intake, trusted checking policy, prepared-helper identities,
per-ref checks and failure closure remain in force.

Release already checks the latest successful exact-SHA main-push Foundation and
Source checks before fan-out and again before publication. Its production-only
builds use a different profile from Foundation's incremental builds; analyzers
and tests are not repeated. Packaging and publication reuse the same-run checked
maintenance tools. The review found no new Release gate defect; the tooling guide
now corrects its stale claim that Release TTS repeats the full check suite.

Remote inspection found no active rules on `main` and no branch protection. The
only repository ruleset, `26259738`, is disabled and covers deletion and force
pushes, without required checks. Local hooks prevent ordinary unchecked pushes
when configured, but cannot enforce GitHub acceptance against disabled hooks or
administrative credentials.

A concrete remote option preserves direct-main delivery: first check the exact
candidate commit on another branch; then require its `Source sanity` status
before fast-forwarding reviewed `main`. A main-only active ruleset would have no
bypass actors, disallow deletion and non-fast-forward updates, and require that
check. It needs a candidate workflow before activation: the existing main-only
triggers cannot supply a pre-push verdict for a new commit. GitHub Actions app
binding alone does not identify a trusted workflow; strong enforcement also needs
a separately controlled check issuer and session credentials without repository
administration. See GitHub's [required-check rule contract](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset).
No remote rules, credentials, pushes or CI runs were changed in this follow-up.

### Follow-up verification and retention

The fixed workload was declared before execution: two copied-policy Clippy
controls, five exact Rust regressions, scoped formatting, the owned-language
guard and cleanup. The ceiling was 420 seconds with a conservative 360-second
estimate because two test targets needed compilation and no compatible test
executables had been found. It ran from 2026-11-14 21:51:34 through 21:54:41 UTC,
including setup, retained-result intake and cleanup: **187 seconds total**.
Native test invocations had narrower aggregate bounds of 100 and 230 seconds;
each Clippy arm had 30 seconds. No clock was restarted or failed Rust test retried.

AC was connected, charge 66% and charging before and after the workload.
The cosmetic arm's before/after observations at 21:51:44–45 UTC and defect arm's
before/after observations at 21:51:45 UTC separately recorded the same conditions.
Battery saver was unknown throughout. These correctness controls make no
performance comparison; endpoint observations do not prove uninterrupted power
stability.

The native supervisor was an invocation-private digest-verified copy bound to
reviewed source `79a520bbe92a4af2c971422b381f44ca3bcc95eb`, executable SHA-256
`02c44eb41de5ba9d44063601deb025079605974a5b95f9958802562b7b262a84`.
The selected tests consumed the integrated code through `e9971f1e` plus the
format-only assertion correction. They used pinned
`nightly-2026-11-12-aarch64-pc-windows-msvc`, rustc `1.102.0-nightly`
(`3ef0c22cdb86ac42dc6b8f0f1ec36857daa6f2a1`) and PowerShell 7.7.0.
Both native receipts recorded lockfile SHA-256
`1765bb78a66af5030fee89f7a9f5f982809a281ad3a029b807016c6c099839b0`,
registry SHA-256 `b5efe1ecf8cec95674a7cc5d99cd5768b81e59b19b96ed971bc6bcbf27ae6ab3`
and workspace-manifest SHA-256
`76ddbf7923a8b0ed3324a0298ff69087004fd3f09e9d91ad0eda0e287a06f6c9`.

| Selection | Result |
| --- | --- |
| `command::push_sanity::tests::push_sanity_formats_changed_root_and_nested_rustfmt_configuration` | Passed: one exact test, zero failures/ignored. |
| `acceptance_driver::tests::bridge_charges_every_fraction_without_restarting_deadlines` | Passed: one exact test, zero failures/ignored. |
| `acceptance_driver::tests::profile_and_markers_require_exact_fixed_controls` | Passed: one exact test, zero failures/ignored. |
| `sandbox_qualification::tests::sandbox_log_admission_keeps_three_controls_inside_the_original_deadline` | Passed: one exact test, zero failures/ignored. |
| `hexagon_tools::public_control::windows::tests::reviewed_sources_require_complete_exact_archive` | Passed: seven pinned sources, missing-root and changed-digest controls; one exact test, zero failures/ignored. |
| Copied actual workspace lint tables, dependency-free Clippy fixture, `--locked --release --offline --all-features --all-targets -- -D warnings` | Cosmetic arm passed. Deliberate `value / value` arm exited 101 and specifically reported `clippy::eq_op`, as required. |
| `pwsh -NoLogo -NoProfile -File tools/ci/check_owned_languages.ps1` | Passed. |
| `cargo +nightly-2026-11-12 fmt --manifest-path Cargo.toml --package bonsai-ci --package bonsai-tools -- --check` | Initially exited 1 for two new ABI-test assertions. Applied exactly the reported wrapping; final `--package bonsai-tools` formatting check passed. |

No complete suite, model trial, benchmark, CI dispatch or push was selected.
Linux compilation and complete owning-lane hosted acceptance remain pending in
C1; local Windows tests do not establish that verdict. The checker preparation
must match the final committed inputs before a future push; these focused checks
are not an outgoing-tip push verdict.

The two ABI artifact scratch directories, copied-policy scratch, private helper,
temporary driver, compiler outputs, native working directories and raw successful
logs/receipts were removed. This concise summary is the retained diagnosis and
focused-check result; no independent replay consumer requires those temporary
files. Existing required numerical/ABI replay inputs remain untouched.

## Earlier 30-run sample

The earlier audit recorded **30 failed workflow runs**, across all workflows, returned
by `gh run list --status failure --limit 30` on 2026-11-14. It is 30 runs in total,
not 30 per workflow: 16 Foundation, 10 Hexagon and four Windows Release runs.
The sample spans 2026-11-13 22:31 through 2026-11-14 19:37 UTC. Cancelled runs are
outside this failure sample. Run 43628943271 has two failed attempts; both were
inspected. Failures in multiple jobs of one run are listed together.

The reviewed local starting revision was `d195bc778d0e17df0e90555abfac826af5af90fa`,
27 commits ahead of the local `origin/main` at
`3d3ffe77a39c8afbe9edf30f28b0d8ca8cd7e5d8`. Many historical corrections already
existed locally. A local correction is not a successful hosted rerun.

## Earlier sample: every run

Run links identify the full source revision, attempt and retained logs. Cause
codes below identify the originating job/step and correction; aggregate rejection
after a failed planner is a consequence, not another root cause.

| Run | Workflow | Source | Causes |
| --- | --- | --- | --- |
| [43633113374](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43633113374) | Foundation | `3d3ffe77` | F2 |
| [43631697910](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43631697910) | Windows Release | `8aca8b02` | R4 |
| [43631695104](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43631695104) | Foundation | `8aca8b02` | F2 |
| [43631695096](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43631695096) | Hexagon | `8aca8b02` | H2, H4 |
| [43628943863](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43628943863) | Foundation | `a0873e69` | F2 |
| [43628943271](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43628943271) | Windows Release | `a0873e69` | R3, attempts 1 and 2 |
| [43580820777](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43580820777) | Foundation | `d3fe70d5` | F2 |
| [43580820773](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43580820773) | Hexagon | `d3fe70d5` | H2, H4 |
| [43580820113](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43580820113) | Windows Release | `d3fe70d5` | R2a–R2e |
| [43580406374](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43580406374) | Hexagon | `b008b2d7` | H3, H4 |
| [43580406370](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43580406370) | Foundation | `b008b2d7` | F2 |
| [43579574922](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43579574922) | Foundation | `da2311c1` | F2 |
| [43579574822](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43579574822) | Hexagon | `da2311c1` | H3, H4 |
| [43578257461](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43578257461) | Foundation | `a05c11de` | F2 |
| [43577726760](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43577726760) | Foundation | `3546ba73` | F2 |
| [43577726709](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43577726709) | Windows Release | `3546ba73` | R1 |
| [43577726696](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43577726696) | Hexagon | `3546ba73` | H3 |
| [43576602927](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43576602927) | Foundation | `250c3e9b` | F2 |
| [43575640226](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43575640226) | Foundation | `ca1c99ab` | F2 |
| [43575640207](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43575640207) | Hexagon | `ca1c99ab` | H3 |
| [43574528631](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43574528631) | Foundation | `1fc86278` | F2 |
| [43574528645](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43574528645) | Hexagon | `1fc86278` | H2 |
| [43565674892](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43565674892) | Hexagon | `ba6db31d` | H2 |
| [43565674757](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43565674757) | Foundation | `ba6db31d` | F2 |
| [43562543809](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43562543809) | Foundation | `7c09d422` | F2 |
| [43526969258](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43526969258) | Hexagon | `90e6e876` | H1 |
| [43526969076](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43526969076) | Foundation | `90e6e876` | F1 |
| [43523437861](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43523437861) | Hexagon | `feca21a1` | H1 |
| [43523438099](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43523438099) | Foundation | `feca21a1` | F1 |
| [43519935649](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/43519935649) | Foundation | `83077b76` | F1 |

## Earlier causes and prevention

| Code | Confirmed cause and source correction | Prevention and limits |
| --- | --- | --- |
| F1 | `changes` cache-input fixture used `RustFlags` and `RUSTFLAGS` in PowerShell's case-insensitive hash literal; parsing rejected duplicate keys. Existing `1b486bf9` uses an ordinal map. | Keep case-sensitive environment fixtures. Name the failed fixture before rethrowing its original error. |
| F2 | `mixed_case_names_have_same_build_identity` attempted removal with `$null`; PowerShell 7.5+ bound that call to an empty string, leaving a hashed override. Existing `2f20e3ab` uses actual-null deletion. The step was initially bundled under `Check CI selection and gate regressions`, then split into `Check CI orchestration fixtures`. | New focused fixture preserves and restores both absent and deliberately empty values; do not normalize distinct inputs into a false cache match. The Foundation aggregate correctly rejected the failed planner. No product check verdict was reached. |
| H1 | `controls` / `Prepare source-bound…`: `dispatch-build.txt` reports that the root exited with one surviving owned Job member; strict teardown stopped it and rejected the run. Neither child identity nor root exit code was retained in these historical records. | Existing `e74ab4c8`/`c1b18b80` use pinned LLD to avoid persistent MSVC linker-service descendants. Attribution of these particular historical children to a PDB server is an inference, not an established identity. Keep strict teardown; no basis exists to waive it as a false positive or claim this historical uncertainty eliminated. |
| H2 | Same `controls` preparation exceeded its fixed 180-second dependency-build bound. Latest capture was 197,898 stdout bytes and 9,956 stderr bytes, below the output bound; compilation reached Windows process/MXC dependencies without completing. It built the entire tools library merely to obtain three externs; Windows SHA-256 was also only a dev dependency there. | Build CPU policy for serde/JSON, then only locked SHA-256 using its normal dependency context in the CI manifest. Both commands share the original 180 seconds. Preserve pinned host/target linking and independent numerical/source controls. The focused fixture makes both large owners unbuildable and requires the actual dependency artifacts. Hosted cold-build duration remains pending. |
| H3 | `ranking-controls` / `Validate comparative…`: `cargo fmt -p bonsai-cpu-policy` rejected array wrapping at `hardware/tests.rs:27`, after all eight ranking tests passed. Existing `d3fe70d5` repaired formatting. | Keep the package-scoped formatter once in its owning lane; pre-push Rust formatting catches this before publication. This was a format failure, not failed ranking arithmetic. |
| H4 | `public-native-abi` / `Compare real public queue declarations…`: Win32 sharing violation (`os error 32`). `Control::receive` retained its writable download handle while restrictive read-only identity intake denied write sharing. Latest provenance had `commands: []`, placing failure before compilation. | Sync and close the writer before guarded identity intake, then verify length/hash again. A network-free Windows fixture checks identity, duplicate refusal and cleanup. Do not weaken the immutable-read sharing contract or retry the same deterministic failure. |
| R1 | `validate` release primitive probe merged compiler/rustup stderr into successful JSON stdout; JSON parsing saw an `info:` prefix. Existing `f39ae0bc` separates stderr; `4ee2037d` pins bootstrap. | Keep machine-readable stdout separate from diagnostics. Package/publish now consume the same-run checked helper bundle instead of bootstrapping it again. |
| R2a | `maintenance` tests: seven failures shared a stale numerical capture source (`1429a337…`) while the runtime lock selected `99671be7…`. An expected allowance-error assertion was masked by that earlier identity failure. Existing `56feb066` updates the capture pin. | Preserve source identity admission and its focused controls. A mismatch is actionable, not a reason to weaken numerical evidence. |
| R2b | `core` test `invalid_scale_closes_schema_connection_before_backend_and_preserves_service` expected empty EOF although the service correctly returned a framed 400 `invalid_chat_request`. Existing `a78e1936` corrects the expectation. | Assert the documented protocol response; retain the independent no-backend-launch/service-preservation checks. |
| R2c | `ui`: eleven failed controls covered transcript reflow/viewport anchors, continuous repaint effects, recovery initialization and missing font/glyph initialization. Existing `143eb46e`, `a0873e69`, `9737da04` address these source/fixture defects. | Preserve UI behavioral tests and shared font setup in the owning Foundation lane. Do not suppress failed UI behavior as cosmetic lint. Actual UI/device acceptance remains separate. |
| R2d | `ui` Clippy rejected optional const promotion, unit-expression punctuation and a deliberate `expect_err`; maintenance rejected a redundant clone. Existing `63da734e` and ownership cleanup corrected the source. | Allow optional const promotion and unit punctuation globally, alongside digit grouping below; preserve panic/ownership and correctness controls. No blanket analyzer waiver was added. |
| R2e | `tts` notice collection bootstrapped Cargo outside the checkout with default rustc 1.99.1, below the workspace's 1.101 requirement. Existing `4ee2037d` pins the dated toolchain in the release adapter and notice collector. | Pin the compiler explicitly across working-directory changes; reuse existing native outputs where possible. Notice completeness remains required. |
| R3 | `publish`: attempt 1 created a draft but confirmation rejected GitHub's valid temporary `untagged-` draft URL; attempt 2 rejected the same draft on reread. Existing `f61a2242` permits that temporary URL for the exact draft while preserving canonical published URLs and metadata/asset checks. | Reuse and independently inspect an existing same-run draft; do not repeat ambiguous mutations. Preserve source, tag, description, flags and byte identities. Both failed attempts remain part of the history. |
| R4 | `maintenance` Clippy rejected grouping of opaque numeric fixture IDs (`unreadable_literal`); `3d3ffe77` reformatted them. | Allow this cosmetic lint in the workspace. The root check never demonstrated incorrect program behavior. |

## Earlier structural corrections

Release previously started expensive checks even while Foundation's planner for
the same SHA had already failed. It now requires the latest successful exact-SHA
main-push Foundation and Source checks, verifies attempt-specific paginated jobs
and mandatory gate steps, and checks again before publication. Older successes,
ancestors, skipped mandatory jobs, bundle-only work and changing rerun attempts
cannot substitute. Release then builds its five production lanes without
repeating Foundation's analyzers/tests, and shares its source/digest-bound native
tools with packaging and publication. Independent release primitives, notices,
packaging and executable smoke checks remain required.

Both prerequisite workflows now emit a verdict for every main push. Existing
selectors retain their scope: unrelated changes do not select expensive jobs,
unless earlier code still lacks a successful baseline. This avoids a permanently
missing exact-SHA verdict for code-only or documentation-only release commits.

The tracked pre-push guard materializes outgoing branch/tag tips with a separate
Git index and applies basic source checks before an ordinary push. Agent
instructions prohibit bypassing it. Only changed PowerShell syntax and existing
workflow/format/policy checks are selected, plus named controls for known fragile
adapters. It adds no general style linter or full local suite. Local hooks cannot
technically prevent a privileged operator from disabling them; absolute remote
enforcement requires GitHub rules plus restricted push/bypass credentials. No
remote rules, pushes, CI dispatches or new runs were performed in this audit.

## Earlier local validation and retention

The predeclared workload started 2026-11-14 19:52:30 UTC with a 720-second ceiling
and conservative 600-second estimate. Two Rust tool crates plus focused fixtures,
source checks, integration and cleanup justified extending the default five
minutes. AC was disconnected, charge was 86%, and battery saver was unknown at
the start. No models, full suites, benchmarks or performance comparisons were
selected. Equal endpoint observations cannot prove uninterrupted power stability.

The last aggregate power observation was 2026-11-14 20:04:42 UTC: AC disconnected,
80% charge, battery saver unknown. The observed wall-clock interval was about
12 minutes 12 seconds, **12 seconds beyond the declared limit** and below the
absolute 15-minute ceiling. The final check and its cleanup/observation ran late;
this is an allowance overrun, not a qualified timing result. No diagnostic clock
was restarted and no further diagnostics were run.
Intermediate native push-check snapshots observed battery saver off at 82% and
81% charge; these do not fill the unknown aggregate endpoint saver observations.

| Selected work | Result and scope |
| --- | --- |
| Four cache-environment cases, two Source routing/trigger cases, startup guard, one startup fixture, owned-language guard | Nine focused commands passed in the Foundation worktree. |
| Integrated workflow syntax plus `same_repository_action_and_script_text_pass` | All 13 manifests and the named fixture passed before snapshot-specific changes. |
| Native release-prerequisite fixtures | Four passed: exact identity/latest failure, complete gate/attempt binding, pagination/legitimate skips and rerun race. |
| Native outgoing-tip fixtures | Four passed after fixing Windows verbatim paths, including a real temporary-index snapshot with conflicting staged/unstaged repairs. |
| Public-input writer handoff | The one network-free Windows fixture passed. |
| Projection dependency/linker handoff | Initial selection failed because Cargo pruned the hash package from the selected owner's context; corrected two-context source passed the named fixture. |
| Strict Release Clippy, CI and tools library/test targets | Passed after resolving the native path arithmetic and removing nonsemantic blockers. This preceded the final CMD/config/preparation source follow-ups; their final analyzer verdict remains pending. |
| Complete local push adapter | Earlier failures exposed direct packaged-PowerShell launch restrictions, then snapshot actionlint config discovery. The owned-CMD correction compiled and reached the workflow syntax stage, where custom runner labels were incorrectly rejected. No passing complete-adapter verdict is claimed. |

The final source makes actionlint's config explicit, adds its named snapshot
regression, handles final output bounds and historical fixture compatibility,
and changes ordinary pushes to require an already prepared source/digest-bound
helper. The last change removes an unbounded hidden Cargo bootstrap from a push;
explicit preparation remains a caller-supervised source build. These source-only
follow-ups were reviewed after the allowance closed and were **not rerun**.
Preparation now binds an explicit Windows ARM64 target/output, closing a source-
review finding where a Cargo target override could leave an older host binary.
Named adapter self-checks use the committed checking policy; they do not qualify
non-HEAD candidate adapter behavior. All outgoing snapshots still receive static
checks. Candidate-root adapter controls remain a scoped follow-up in C1.
No helper preparation receipt for the final source was created. A push therefore
fails closed until its prerequisite is prepared and the remaining checks pass.
Owning hosted acceptance stays in [C1](../../PLAN.md).

Failed logs and only specifically selected nonblank artifact files were inspected
in temporary storage. The per-run source identities, errors, losing outcomes and
limits above are the durable diagnosis; no complete artifact ZIP, copied source
tree, successful build chatter or placeholder receipt belongs in this record.
This summary does not replace any independent numerical/ABI replay inputs already
required by the repository's existing evidence consumers.
