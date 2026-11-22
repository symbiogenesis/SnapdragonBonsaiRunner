# Retained evidence

Keep inputs and records required by the consumers below under the
[retention policy](../validation.md#evidence-retention).
Historical records may name retired Python tools or external Python sources.
Preserve those identities as inert evidence; use the current Rust/PowerShell
consumers below without installing or executing the recorded Python tools.

| Retained material | Current consumer |
| --- | --- |
| [CI failure audit](ci-failure-audit.md) | Root causes for the latest 30 failed runs, targeted prevention and pending hosted acceptance |
| [Prism 0.1.32 delivery](prism-0.1.32-release.md) | Current runtime/source lock, original release receipt, registry delivery, bounded X2 broker-pipe trial and independent lock-update replay |
| [Prism 0.1.31 delivery](prism-0.1.31-release.md) | Prior 0.1.31 runtime/source lock, original release receipt, registry delivery and lock-update replay |
| [Prism 0.1.30 delivery](prism-0.1.30-release.md) | Prior 0.1.30 runtime/source lock, original release receipt, registry delivery and lock-update replay |
| [Prism 0.1.28 delivery](prism-0.1.28-release.md) | Prior 0.1.28 runtime/source lock, original release receipt and 0.1.29 registry delivery |
| [Daily-use readiness](daily-use-readiness.md) | Actionable qualification failures, bounded local-check outcome and open daily-use/isolation acceptance gates |
| `cpp-retirement-model-free.json` | [TTS decode-graph fixture provenance](../../src/speech-tts/tests/fixtures/decode-graph-v1.provenance.json) |
| `weight-patch-sanity/` | [Initial four-fact adapter-cost smoke](weight-patch-sanity.md): independent answer, accounting, resource, supervision and power review |
| `lora-weight-edit/` | [Direct LoRA weight edit](lora-weight-edit.md): independent edit derivation, candidate digest, complete factual, storage, supervision and power review |
| [DaisyRose task-vector audit](daisyrose-task-vector-audit.md) | Closed Gate 1 (`DR1`) no-go summary for raw BF16 task-vector LoRA extraction (`E(32) = 14.26% < 50%`) |
| [DaisyRose spectral-denoising audit](daisyrose-spectral-denoising-audit.md) | Closed Track A (`DR-A1`) empirical Rademacher-null + Gavish–Donoho summary (`saturated-high-rank-tail`, `81.98%` bulk, Track A standalone `DR-A2` no-go) |
| [DaisyRose multi-rank wiring & bounded vet](daisyrose-multirank-wiring-vet.md) | Source-bound Prism + Runner multi-rank wiring commits, local verification counts, bounded 4-layer F16 export/verify and 27B `scale = 0.0` parity vet (live `:0.5` rollback arm skipped; repin/lock/CI open) |
| [DaisyRose DR-E1/DR-G2 screen & AC vets](daisyrose-dr-e1-dr-g2-screen.md) | Full 64-layer `DR-E1` extraction, `91298a4c` matched-AC 64-layer parity/rollback/`3.16%`-overhead vet pass (`<= 10%`), and closed incomplete battery/`28d47fe`/`91298a4c` `DR-G2` captures awaiting awake 4-category screen & owner blinded review |
| `hexagon-public-qaic-41171859264/` | [Signed-Q2 source lock](../../config/hexagon-signed-q2-candidate.lock.json), public-wire review and [Windows ABI replay inputs](../../tools/hexagon/windows_queue_abi.ps1) |
| `hexagon-public-qaic-41169481882/provenance.json` | Supplemental template notices referenced by the current [QAIC provenance](hexagon-public-qaic-41171859264/provenance.json) |
| Hexagon operation, traffic, session-budget and K-tiling ledgers | [Static policy verifiers](../../tools/hexagon/README.md) |
| Hexagon signed-package, projection-binding, FD/session and advance-control records | [Experiment lock](../../config/hexagon-experiment.lock.json), [signed-Q2 lock](../../config/hexagon-signed-q2-candidate.lock.json) and their archive manifest |
| `hexagon-adapter-session-20261103/model-device-request.json` | Request template consumed by [model-control preparation](../../tools/hexagon/prepare_model_control.ps1) |

Preserve required replay bytes, notices and integrity records. Supplemental
QAIC provenance supplies template notices, not missing payloads or an independent
replay. An observation qualifies only its recorded source, tools and workload.
