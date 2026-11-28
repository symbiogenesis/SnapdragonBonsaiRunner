# Windows distribution

Runner is available as a
[Windows ARM64 prerelease](https://github.com/symbiogenesis/SnapdragonBonsaiRunner/releases).
Build a portable executable with the
[development guide](development-guide.md#build-a-single-executable).
The format, setup and update behavior below still require the acceptance work
in [PLAN.md](../PLAN.md).

## Download and open

The distribution format is one portable executable for Snapdragon X and X2
laptops. A publication contains three assets:

| Asset | Purpose |
| --- | --- |
| `SwiftBonsai2Runner-<version>-windows-arm64.exe` | Five application tools, package metadata and notices |
| `SHA256SUMS.txt` | Executable and manifest checksums |
| `release-manifest.json` | Version, source, build identities and executable digest |

Verify the executable against the same publication's checksums, then open it.
Only the `.exe` needs deploying. It opens the native UI without a terminal,
checkout, compiler, installer or separate Microsoft runtime installation.
Checksums establish byte integrity within GitHub's trust boundary; the executable
is unsigned.

The launcher verifies and extracts its package into
`%LOCALAPPDATA%\SnapdragonBonsai\application-bundles\<first-24-manifest-sha256-characters>\`
and verifies cached files before reuse. Process ownership follows
[Windows sandboxing](sandboxing.md).

## First launch and in-app setup

Use serviced Windows 11 ARM64 with the required
[containment capabilities](sandboxing.md). Startup selects context and memory
from available RAM, reserving 4 GiB for Windows and other apps. Unknown or
insufficient memory stops startup; see [context sizing](swift-memory.md#memory-selected-serving-context).

The app downloads the pinned Swift weights (about 6.71 GiB), adapter, inference
runtime and speech models under their [license terms](licensing.md). Text chat
can start before speech preparation finishes. Microphone capture requires consent.

**Pause setup** retains partial downloads; **Retry setup** or reopening resumes
preparation. Verified files are reused. Corrupt model files are replaced only
after the replacement verifies. Unsafe paths, unreadable files and unrelated
runtime files require attention. Repair needs storage for both copies.
Setup details and **Copy diagnostics** identify the failed step and data folder.
Speech has a separate **Retry speech setup** action in **Settings > Voice**.

Preparation never silently substitutes an engine or model. Explicit engine,
representation and rollback options are in the
[development guide](development-guide.md#native-chat-app).

## Update and remove

Portable builds check GitHub 10 seconds after startup and every 15 minutes at below-normal
background priority; **Settings > Application** also offers a manual check (which probes
past stale local offline cooldowns while still honoring server rate limits). Discovery
selects the highest numeric version, including prereleases, from a bounded catalog,
excluding drafts and older versions. An incomplete catalog or invalid newest release fails
without falling back. Conditional requests reuse unchanged metadata; failures honor
server backoff and never count as successful checks.

Settings shows the current version, update status (with live download phase and byte
progress, or deferred close-install status) and one relevant action without opening another
settings area. The app downloads and verifies the asset digest, manifest, checksums and
tag/source identity in the background, pre-extracts the verified portable bundle for fast
restart, and prunes obsolete staged updates and bundle caches best-effort. A quiet nudge
offers **Restart to update** and **Later** while typing and generation continue, and shows
a compact inline status strip while stopping to update. The Settings badge also shows that
an update is ready.

Choosing **Restart to update** from the nudge or Settings opens one confirmation
popup. **Restart and update** closes, replaces and reopens the app with its original
options and data directory; **Cancel** leaves it running without authorizing an
installation. **Later** dismisses the nudge for that verified release during the
session, keeps the Settings badge and installs on close without reopening. No
confirmation appears automatically when a download finishes. Closing before
verification finishes exits without replacement.

Replacement requires confirmed process teardown and revalidated publication.
Download or replacement failure preserves the existing executable. If restart
fails, reopen manually; read-only locations require manual replacement. Source
and preview launches do not check for updates.

Manual replacement also works: close the app, replace the verified executable
and reopen with the same data directory. To remove it, delete the executable and
optionally its package cache. Models, conversations and preferences remain in the
data directory until explicitly deleted. Developer
[checked-directory packages](chat-packages.md) have separate activation controls.

## Publish a release

Follow push authorization and CI ownership under [AGENTS.md](../AGENTS.md).
When the user requests a release, review source, publish `main`, create and push
the matching tag at that commit, then dispatch `release.yml` with
`prerelease=true`. An omitted `tag` uses `workspace.package.version` from
`Cargo.toml`; an explicit tag must match it. The publication body links to the
README at that immutable source revision. No per-version notes are required.

Before dispatch, Foundation and Source checks must have completed successfully
for that exact main-push commit. Both workflows record a verdict on every main
push; their selectors still avoid unrelated expensive work. Follow CI ownership
when inspecting those runs. Release does not dispatch, wait for or retry them.
Its native `release-prerequisites` gate checks repository, workflow, exact SHA,
main-push event, latest attempt and complete paginated job inventory, including
the successful Foundation planner/aggregate and Source lock-verification step.
Missing, pending, failed, cancelled, skipped required or bundle-only results
block Release before its build fan-out. The gate repeats before publication.

Foundation owns formatting, analyzers, documentation and tests. Windows Release
consumes that verdict and builds five production lanes on hosted Windows ARM64,
without repeating those checks. Both workflows use the same non-incremental
`[profile.release]` codegen. Packaging and publication reuse the maintenance
lane's same-run, revision/digest-verified native tools instead of rebuilding them.
Release retains its own primitive/argument controls, source checks, notices,
packaging and executable smoke checks. It verifies source-bound receipts and notices,
assembles the executable, and checks its GUI header, extraction and identities.
Missing notice material stops publication. These checks do not establish
clean-machine, actual-model, quality, performance or isolation acceptance.

Only the publication job writes release content. It uploads the exact three-asset
inventory to a draft and verifies remote assets and source before publication.
The tag must identify the checked commit. Existing publications are never replaced.

Successful draft creation is confirmed by an independent API read of the returned
release ID, without relying on immediate catalog visibility. Draft admission
allows GitHub's temporary `untagged-` URL within the same repository while checking
the exact ID, tag, source, description, flags and asset identities. Published
releases must use the canonical tag URL.

To recover interrupted publication, rerun the failed job from the same run:

```powershell
gh run rerun RUN_ID --failed --repo symbiogenesis/SnapdragonBonsaiRunner
```

Recovery reuses retained packaging assets and requires matching source, description,
title, prerelease setting and asset identities. It uploads missing assets,
verifies existing ones and resolves ambiguous responses through readback. A
completed publication is verified without writes, even if `main` has advanced.
Draft mutation still requires the original current `main`. Unknown assets,
changed metadata or bytes, and absent or changed tags stop recovery. A new
dispatch refuses an existing release; rebuilding or deleting a draft is not a
recovery path.
If a source correction is needed after a draft was created, retain that failed
outcome and prepare a new version from the corrected source.
