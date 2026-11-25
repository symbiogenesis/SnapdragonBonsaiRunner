# Open math conjectures and 2026 proof implications

This reference evaluates the October–November 2026 foundational mathematics
and complexity-theory proof releases from Anthropic, OpenAI and Google
(Cogentic) strictly for the **Swift Bonsai 2 inference engine** (`prism-inference`
dense ternary `qwen35` hybrid Gated DeltaNet + full attention on native Windows
ARM64 Snapdragon X1/X2, plus native Rust speech FFT and projection kernels).
Harness, UI, Windows sandboxing and cryptographic update/TLS implications are
explicitly out of scope here. [PLAN.md](../PLAN.md) owns backlog items; the
[research register](research-register.md#mathematical-results-and-proof-obligations)
(`T13`–`T16`) holds reusable proof obligations.

## Scope and verification discipline

Every 2026 release below must be separated by both **verification tier** and
**finite-size Snapdragon applicability**:

- **Lean 4 + independent kernel / Mathlib checks (`anthropics/formal-math`, Alpöge RH/GRH):**
  `anthropics/formal-math` (`3sum-apsp/`, `eth/`, `vp-vnp/`) packages
  self-contained Lake projects checked with `lean4export` and the independent
  `nanoda` kernel; Anthropic explicitly notes that no human mathematician has
  yet audited `vp-vnp/Statement.lean`'s formalization mapping. Levent Alpöge's
  17.4M-line Lean 4 proof of the Riemann Hypothesis and Generalized Riemann
  Hypothesis (`leanprover-community/riemann-hypothesis`, Nov 20, 2026) compiled
  and passed Mathlib-canonical checks by the Lean FRO and Mathlib Initiative.
- **Raw AI manuscripts with full or partial Lean 4 formalizations (`openai/math`):**
  OpenAI labels `openai/math` (Oct 6, Oct 26, Nov 23 and Nov 24, 2026 drops) as
  raw research data pending peer review. Several entries carry `lean/docs/<NNN>.md`
  formalizations, while narrower scope boundaries apply in places (for instance,
  Family 379 division-gate circuit lower bounds and unformalized auxiliary modules).
- **Autonomous multi-agent preprints (`google-research/cogentic_research_papers`):**
  Released Oct 30–Nov 25, 2026 by Google Research / DeepMind's Gemini 4 Cogentic
  harness (`arXiv:2609.40324`), with Lean 4 formalizations announced as
  forthcoming companion submissions and no human peer review yet.

**Inference-engine bottom line:** None of the October–November 2026 proofs
unlocks an immediate drop-in Snapdragon X1/X2 kernel speedup or RAM reduction.
Batch-one Swift decode is DRAM-bandwidth-bound `GEMV` governed by
[T1](research-register.md#t1-packing-limits-and-the-difference-between-sparse-and-small)
and [T2](research-register.md#t2-traffic-arithmetic-and-the-whole-request-ceiling),
and the asymptotic matmul, output-sparse thin-matrix, FFT, ETH and
space-simulation crossovers sit astronomically beyond Swift's dimensions
(`K = 5,120`, `N_ffn = 17,408`, `D = 256`, `N <= 262,144`). Instead, these
proofs contribute **one immediately usable mathematical certificate (`T15`
Crouzeix numerical-range bound for non-normal GDN Neumann truncations)**,
**sharp theoretical ceilings and state-capacity floors (`T13`, `T14`, `T16`)**
that prevent chasing impossible dense/FFT/state shortcuts, and **conditional
future references** (such as `OpenAI 110` for large-`k` tiered MoE/KV paging).

## Inference-engine triage

| Proof / source (date) | Formalization status | Mathematical claim | Engine subsystem | Inference-engine verdict |
| --- | --- | --- | --- | --- |
| OpenAI Families `107` & `379` (`openai/math`, Oct & Nov 24, 2026) | Partial Lean (`lean/docs/107.md`, `379.md`); peer review open | Square matmul exponent `omega = 9/4` over every field (upper bound + `>= n^(9/4)/2` nonscalar mul/div gates); rectangular `n x n^a x n` in `n^(2+o(1))` for `a > 0.465` (`char 0`) | Projections & tiled prefill | **Hard ceiling; no immediate kernel** ([T13](research-register.md#t13-asymptotic-matmul-output-sparse-thin-products-and-circuit-bounds)) |
| Anthropic / Alman–Vassilevska Williams (`arXiv:2610.06783`, `anthropics/formal-math/3sum-apsp/`, Oct 2026) | Lean 4 + `nanoda` verified | `O(n^1.9992)` 3SUM and `O(n^2.99942)` `(min,+)`-product / APSP via Claude-discovered output-sparse thin matmul (`N x D` by `D x N`, `D <= N^(1/18)`, `N^2/sqrt(D)` outputs in `O(N^2 / D^0.063)`) | Attention `QK^T` & NariQ Viterbi trellis DP | **Refutes conditional barriers; galactic crossover (`N >= 2^144` for `D = 256`)** ([T13](research-register.md#t13-asymptotic-matmul-output-sparse-thin-products-and-circuit-bounds)) |
| OpenAI Family `135` (`openai/math`, Oct 2026) | Preprint (`CONTENTS.md#135`) | `n^Theta(sqrt(n))` homogeneous depth-5 lower bound for iterated `n x n` matrix multiplication (`char 0`) | Multi-layer residual & GDN chains | **Structural lower bound against collapsing deep layers** ([T13](research-register.md#t13-asymptotic-matmul-output-sparse-thin-products-and-circuit-bounds)) |
| Anthropic (`anthropics/formal-math/vp-vnp/`, Nov 23, 2026) & OpenAI Family `379` (Nov 24, 2026) | Lean 4 + `nanoda` (Anthropic; human `Statement.lean` audit open); Lean (`379.md`) | `VP != VNP` over every field (permanent outside `char 2`, Hamiltonian cycle, 2D grid state sums, border complexity; not Boolean `P != NP`) | Exact global tensor-network / partition-function contraction | **Negative structural bound reinforcing autoregressive factorization** ([T13](research-register.md#t13-asymptotic-matmul-output-sparse-thin-products-and-circuit-bounds)) |
| OpenAI Families `130`, `109` & `379` (`openai/math`, Oct & Nov 24, 2026) | Lean (`lean/docs/130.md`, `109.md`, `379.md`); peer review open | Finite abelian Fourier transforms in `N (log N)^(1/2+o(1))` binary linear gates (explicit `O(n (log n)^(1 - 10^-13))` DFT); integer mul in `n (1 + log n)^(1/2+o(1))` | `1024`-pt activation FWHT, `256`-pt `rsq4` KV WHT, `512`-pt speech FFT | **Asymptotic only; unstable/galactic vs. pinned binary32 butterfly** ([T14](research-register.md#t14-sub-n-log-n-finite-abelian-transforms-versus-fixed-size-rounding-stability)) |
| OpenAI Family `325` (`openai/math`, Oct 2026) | Lean (`lean/docs/325.md`); peer review open | Complete Crouzeix conjecture: `||P[A]|| <= 2 sup_{z in W(A)} ||P(z)||` with sharp constant `2` for Hilbert-space operators | GDN chunkwise Neumann / multiplication-only inverse approximations (`arXiv:2606.06034`, `#42` HMX) | **Active computable pre-implementation error certificate** ([T15](research-register.md#t15-sharp-non-normal-gdn-polynomial-truncation-bounds-via-crouzeix)) |
| OpenAI Family `140` (`openai/math`, Oct 2026) | Lean (`lean/docs/140.md`); peer review open | One-pass noiseless Gaussian regression with `A d^2` persistent bits needs `Omega_A(d log(1/eps))` samples | GDN `O(d^2)` recurrent matrix state (`149.625 MiB` FP32) & `SPARE` compression | **Active state-capacity floor reinforcing FP32 GDN protection** ([T16](research-register.md#t16-recurrent-state-capacity-dimension-reduction-paging-and-lp-and-packing-limits)) |
| OpenAI Families `094` & `098` (`openai/math`, Oct 2026) | Lean (`lean/docs/094.md`, `098.md`); peer review open | `L_p` (`p != 2`) distorted `D > 1` embedding in `d = n^o(1)` vs. exact `D = 1` bound `Theta(n^2)`; compact Hilbert counterexamples to bi-Lipschitz reduction | Lossless FP16 KV vs. approximate `rsq4`/`rsq4k3v` KV | **Formalizes why lossless vs. approximate KV profiles are distinct** ([T16](research-register.md#t16-recurrent-state-capacity-dimension-reduction-paging-and-lp-and-packing-limits)) |
| OpenAI Family `110` (`openai/math`, Oct 2026) | Lean (`lean/docs/110.md`); peer review open | Randomized `k`-server competitive ratio `O(log^2(k+1))` on arbitrary metrics | Prefix checkpoints (`k <= 8`) & future tiered KV / MoE expert paging (`k >> 8`) | **No gain at `k <= 8`; conditional reference for future large-`k` MoE/tiered paging** ([T16](research-register.md#t16-recurrent-state-capacity-dimension-reduction-paging-and-lp-and-packing-limits)) |
| OpenAI Family `137` (`openai/math`, Oct 2026) | Lean (`lean/docs/137.md`); peer review open | Deterministic 1-tape time-`T` simulation in `O(T^(2/5) log^C T)` space with unrestricted simulation time | Activation / recurrent checkpointing & rematerialization | **Non-viable for interactive latency (unbounded recompute cost)** ([T16](research-register.md#t16-recurrent-state-capacity-dimension-reduction-paging-and-lp-and-packing-limits)) |
| OpenAI Families `380`, `118` & `102` (`openai/math`, Oct & Nov 24, 2026) | Lean (`lean/docs/380.md`, `118.md`, `102.md`); peer review open | Strongly polynomial LP & convex QP (Smale 9th) + superpolynomial simple-polytope diameter (`380`); unbounded bin-packing configuration-LP gaps (`118`); Unique Games Conjecture (`102`) | Offline `weight-patch` LP / `DR-W1` ridge-QP & Hexagon VTCM arena packing (`#41`) | **Offline convex exactness (retains PDHG/Cholesky); confirms discrete NPU arena packing cannot rely on LP relaxation** ([T16](research-register.md#t16-recurrent-state-capacity-dimension-reduction-paging-and-lp-and-packing-limits)) |
| Google Cogentic (`cogentic_research_papers`, Nov 18 & 25, 2026) | Preprints; Lean 4 formalizations forthcoming, unreviewed | Log-Rank square-root barrier broken (`D^cc(M) <= C' r^(1/2 - 5e-6)`, `C = 2^38`, `R_0 = ceil(e^(4e6))`); VC sample compression `O(d^2 log^2 d)` | Ternary weight block coding (`PTQ1_0`) & small calibration / gate-scorer sample bounds | **Asymptotic only; does not beat `PTQ1_0` (`T1`) or relax `U1`/`DR-G2` held-out gates** ([T16](research-register.md#t16-recurrent-state-capacity-dimension-reduction-paging-and-lp-and-packing-limits)) |
| Anthropic RH/GRH (Nov 20, 2026), OpenAI `003`, Cogentic factoring/dlog (Nov 25, 2026), Navier–Stokes (OpenAI Sep 8 / `376`, Alpöge–Buckmaster Nov 20), `ETH` (OpenAI `378`, Anthropic `eth/`), OpenAI `103`/`124`/`139`/`239`/`265`, Cogentic games/graphs | Mixed (Alpöge RH/GRH Lean FRO/Mathlib checked; `eth/` Lean 4 + `nanoda`; others Lean or preprints) | Analytic number theory, PDE blowup, `2^(n/(log n)^(1/2+o(1)))` `#SAT`, `L = RL = BPL`, 3-machine unit scheduling, oracle log-concave sampling, random sign singularity, 2D gapped PEPS, bimatrix Nash / prophets / edge reconstruction | None | **Explicitly closed non-engine proofs** ([see below](#explicitly-closed-non-engine-proofs)) |

## Matrix multiplication, rectangular prefill and output-sparse attention

- **Square and rectangular matmul ceiling (`OpenAI 107` / `Family 379`, `T13`):**
  OpenAI's October–November 2026 preprints place the square matrix multiplication
  exponent at `omega = 9/4 = 2.25` over every field (matching a claimed
  `n^(9/4)/2` nonscalar multiplication/division gate lower bound) and give
  characteristic-zero `n^(2+o(1))` rectangular multiplication for inner
  dimension `n^a` (`a > 0.465`). This closes the theoretical `omega = 2`
  conjecture, yet yields no Snapdragon kernel:
  1. Batch-one Swift decode is `M = 1` vector-matrix streaming (`GEMV`) limited
     by DRAM bandwidth `B` over packed `PQ2_0`/`PTQ1_0` weights (`T2`), not
     square `O(n^3)` FLOPs.
  2. Even in `8/16/32`-token CPU prefill or `16`-token Adreno GPU prefill,
     Swift's projection widths (`5,120`, `17,408`) are orders of magnitude below
     Strassen-family and Euclidean tensor-degeneration crossover scales, and
     arbitrary real/complex tensor degenerations destroy exact `Q8_K`/`Q8_0`
     integer dot bounds (`T3`) and binary32 reduction order (`T8`).
- **Output-sparse thin rectangular matmul (`3SUM`/`APSP`, `arXiv:2610.06783`, `T13`):**
  The Claude-discovered subroutine verified in `anthropics/formal-math/3sum-apsp/`
  multiplies `N x D` by `D x N` matrices over bounded integers when `D <= N^(1/18)`,
  computing up to `N^2 / sqrt(D)` selected entries in `O(N^2 / D^0.063)` time.
  While this removes conditional `3SUM`/`APSP` lower bounds for exact
  `(min,+)`-products (and theoretically touches sparse attention `QK^T` selection
  or `NariQ` 16-band Viterbi DP), Swift has fixed head dimension `D = 256`,
  which requires context length `N >= 256^18 = 2^144` just to enter the `D <= N^(1/18)`
  regime—far above `N <= 262,144`. Unconditional I/O bounds ([T4](research-register.md#t4-attention-io-optimality-has-a-computation-model))
  still govern attention; practical Snapdragon gains continue to come from
  `6:1` grouped GQA KV reuse, rotated `rsq4` KV compression and conditional
  `GateSight` eviction.
- **Depth-5 iterated matmul and `VP != VNP` (`OpenAI 135`, `Anthropic vp-vnp`, `OpenAI 379`, `T13`):**
  The `n^Theta(sqrt(n))` depth-5 lower bound for iterated `n x n` matrix
  products and the separation `VP != VNP` over every field (via 2D grid state
  sums, Hamiltonian cycle and permanent lower bounds) confirm that multi-layer
  residual/GDN compositions cannot be algebraically collapsed into shallow
  circuits and exact global tensor-network partition sums remain superpolynomial
  even in border complexity, reinforcing token-by-token autoregressive
  factorization with bounded speculative verification (`T6`). Note that `VP != VNP`
  separates algebraic circuits and does not settle Boolean `P != NP`.

## Finite abelian Fourier transforms and bit arithmetic

`OpenAI 130` (`Anthropic`/`OpenAI` `Family 379`) proves that finite abelian
Fourier transforms of order `N` require `N (log N)^(1/2+o(1))` binary linear
gates (with an explicit `O(n (log n)^(1 - 10^-13))` exact-complex DFT), and
`OpenAI 109`/`379` gives `n (1 + log n)^(1/2+o(1))` multitape integer
multiplication (`T14`).

For Swift's inference engine, the transform sizes are small and fixed:

- `1024`-point signed Walsh–Hadamard activation rotation (`4,423,680` shared
  rotation/sign/metadata bytes in [swift-memory](swift-memory.md#other-allocation-bounds)),
- `256`-point three-pass Walsh–Hadamard KV rotation in [quantized KV](quantized-kv.md#scope-and-algorithm),
- `512`-point real FFT / RFFT in native Rust speech (`bonsai-speech-tts` / `bonsai-speech-asr`).

At `N <= 1,024` (`log2 N <= 10`), `(log N)^(-10^-13)` saves virtually zero
operations while non-butterfly factorizations with unrestricted scalar weights
break pinned binary32 floating-point order (`T8`), NEON vector locality, and the
existing `0 ULP` / `ARITHMETIC_IDENTITY` contracts. Keep the pinned butterfly
and `1024^2` rotation contracts unchanged.

## Gated DeltaNet non-normal stability and recurrent state capacity

Two October 2026 results directly sharpen Swift's Gated DeltaNet (GDN) analysis
across its 48 recurrent layers:

1. **Computable truncation certificate (`OpenAI 325`, Complete Crouzeix Conjecture, `T15`):**
   `OpenAI 325` establishes `||P[A]|| <= 2 sup_{z in W(A)} ||P(z)||` for every
   bounded Hilbert-space operator `A` and polynomial `P`, where `W(A)` is the
   numerical range (field of values) and the constant `2` is sharp across all
   dimensions. In chunkwise GDN (`T5`), the strict lower-triangular chunk
   transition `L` (with `L^c = 0`) has spectral radius `0`, which previously left
   low-order truncated Neumann inverse approximations (`(I - L)^(-1) = sum_{j=0}^{c-1} L^j`,
   such as [Qualcomm's multiplication-only GDN inverse](https://arxiv.org/abs/2606.06034)
   or a future `#42` HMX FP16 chunkwise candidate) without an eigenvalue-based
   error bound because `L` is non-normal. Applying Crouzeix to the truncation
   residual polynomial `R_m(L) = (I - L)^(-1) - sum_{j=0}^{m} L^j = L^(m+1) (I - L)^(-1)`
   bounds operator-norm truncation error solely from the numerical range `W(L)`
   (computable per chunk via `2x2`/power-iteration Hermitian projections of
   `(e^(i*theta) L + e^(-i*theta) L^T)/2`), turning `T5`'s qualitative warning
   into an explicit pre-implementation acceptance gate.
2. **Information-theoretic `O(d^2)` state floor (`OpenAI 140`, `T16`):**
   `OpenAI 140` proves that a one-pass learner with `A d^2` persistent bits
   needs `Omega_A(d log(1/eps))` noiseless Gaussian samples to recover a
   `d`-dimensional direction within angular error `eps`. Because GDN maintains
   an `O(d^2)` associative key-value matrix (`S_t`, `149.625 MiB` FP32 across
   Swift's 48 layers), compressing state below `Theta(d^2)` bits (`SPARE`-style
   low-rank state reduction) hits an information-theoretic recall wall over long
   contexts, reinforcing [T10](research-register.md#t10-recurrent-quantization-error-depends-on-the-trajectory)
   and Swift's policy of keeping GDN matrix state uncompressed in FP32.

## KV compression, dimension reduction, online paging and space-time

- **Exact vs. approximate KV separation (`OpenAI 094`, `OpenAI 098`, `T16`):**
  `OpenAI 094` shows `L_p` (`p != 2`) allows `n^o(1)`-dimension embeddings for
  distortion `D > 1` while exact `D = 1` embeddings force `Theta(n^2)` dimension;
  `OpenAI 098` constructs compact doubling subsets of Hilbert space with no
  bi-Lipschitz finite-dimensional embedding. Applied to Swift's 16
  full-attention layers, this formalizes why lossless `F16` KV cannot be
  replaced by low-dimensional sketches without distortion, whereas accepting
  bounded distortion (`rsq4` / `rsq4k3v`) unlocks large memory savings behind
  separate quality/state gates ([quantized KV](quantized-kv.md)).
- **Online `k`-server paging (`OpenAI 110`, `T16`):**
  Randomized `k`-server achieves `O(log^2(k+1))` competitiveness on arbitrary
  metrics, including weighted/non-uniform eviction costs. For Swift's current
  `MAX_REUSABLE_BRANCH_CHECKPOINTS = 8` (`k <= 8`), `Runtime`'s deterministic
  `Marconi/GDSF` utility eviction already tracks exact token-prefill work vs.
  checkpoint bytes with zero randomization ambiguity. `OpenAI 110` becomes
  conditionally relevant only if future Qwen MoE tiering or multi-tier KV
  offload manages large slot counts (`k >> 8`) across non-uniform DRAM/NVMe
  latencies.
- **Sublinear space-time simulation (`OpenAI 137`, `T16`):**
  `O(T^(2/5) log^C T)` space simulation of time `T` trades exponentially/polynomially
  higher recomputation time for smaller space. On latency-bound interactive
  Swift inference (`10`–`20+ tok/s`), rematerializing dropped KV/GDN state in
  place of retaining `132` B/head `rsq4` KV is non-viable; explicit
  `BranchCheckpoints` (`1..=8`) and `PromptEndAndLatest` reuse remain the right
  tradeoff.
- **Sample compression and log-rank (`Cogentic VC`, `Cogentic Log-Rank`, `T16`):**
  Cogentic's `O(d^2 log^2 d)` VC sample compression applies to binary concept
  classes and does not relax empirical `U1`/`DR-G2` held-out splits for 27B
  generative calibration. Cogentic's Log-Rank improvement (`C = 2^38`, threshold
  `R_0 = ceil(e^(4e6))`) is strictly asymptotic and does not improve on
  `PTQ1_0`'s byte-aligned 26-byte/128-trit bound ([T1](research-register.md#t1-packing-limits-and-the-difference-between-sparse-and-small))
  for Swift's `32.76154%`-zero ternary tensors.

## Offline pure-Rust weight-space optimization and NPU arena packing

- **Strongly polynomial LP/QP (`OpenAI 380`, `T16`):**
  `OpenAI 380` gives a deterministic strongly polynomial algorithm for linear
  programming and convex quadratic programming (answering Smale's 9th problem)
  while disproving the polynomial Hirsch conjecture for simple polytope walks.
  For offline pure-Rust weight editing in `bonsai-tools` (`weight-patch` sparse
  LP and `DR-W1` Tikhonov ridge QP), this provides theoretical exactness
  reassurance, while practical dense `17,408`-wide optimization continues to use
  PDHG and closed-form Cholesky/ridge solvers which run in seconds without
  symbolic pivot-tree overhead.
- **Unbounded configuration-LP gaps and `UGC` (`OpenAI 118`, `OpenAI 102`, `T16`):**
  `OpenAI 118` shows bin-packing configuration LPs can have unbounded additive
  integrality gaps (and constant-additive approximation is NP-hard even for
  `<= 5` items per bin), and `OpenAI 102` confirms `UGC` hardness beyond
  Goemans–Williamson / `2`-factor cut/cover bounds. This reinforces keeping
  Hexagon NPU multi-arena VTCM/FastRPC placement (`#41`, `<= 480 MiB` arenas,
  `plan_x1_v73_three_processor_placement`) explicit, deterministic and
  hardware-verified rather than relying on LP-relaxation rounding.

## Explicitly closed non-engine proofs

The remaining high-profile October–November 2026 proofs establish **no**
inference-engine acceleration or memory-reduction pathway and are closed here:

- **Riemann Hypothesis / GRH & number theory** (`leanprover-community/riemann-hypothesis`,
  Nov 20, 2026; `OpenAI 003` Quasi-RH `Re s > 7/8`; Cogentic `integer_factorization`
  `L_N[1/3, 105]`, `deterministic_factoring_l34`, `discrete_logarithm`, Nov 25,
  2026): Prime-distribution and factoring/dlog bounds do not affect dense
  floating/integer tensor algebra, quantized KV or GDN recurrence; any signature/TLS
  implications belong outside the inference engine.
- **Navier–Stokes blowup & fluid Turing completeness** (OpenAI Sep 8, 2026
  forced 3D blowup; `OpenAI 376` forced Navier–Stokes universal computation;
  Alpöge–Buckmaster Nov 20, 2026 blowup preprint): Continuum fluid PDE
  singularities and forced-PDE universality have no bearing on discrete
  transformer/GDN inference kernels.
- **Disproof of `ETH`** (`OpenAI 378` Oct 26, 2026; `anthropics/formal-math/eth/`
  Oct 2026): Deterministic `2^(n / (log n)^(1/2+o(1)))` `#SAT` is subexponential,
  not polynomial (`P != NP` remains open), with crossover constants far outside
  tractable `n`. `T1`, `T2` and `T4` are unconditional, while `GrammarGss` (`#38`)
  and `U2`/`DR-W1` `MCKP` are already polynomial/microsecond in Rust.
- **Other complexity, probability, physics & game-theory results** (`OpenAI 103`
  `L = RL = BPL`, `OpenAI 124` 3-machine unit scheduling, `OpenAI 139`
  continuous log-concave oracle sampling, `OpenAI 239` symmetric random sign
  matrix singularity, `OpenAI 265` 2D gapped PEPS area law; Cogentic
  `bimatrix_nash_equilibrium`, `ce_convex_games`, `fictitious_play_convergence`,
  `subadditive_prophet_inequalities`, `edge_reconstruction`):
  `rsq4` WHT signs are already deterministic (`103`); `NativeRowPool` uses `8`/`12`
  workers with variable-cost tiled tasks rather than `3` unit-length machines (`124`);
  Swift sampling (`T6`) is discrete `248,320`-way categorical and `T10` SR
  requires unbiased draws (`139`); Swift's ternary weights are trained dense
  `PQ2_0`, not i.i.d. symmetric sign matrices or 2D gapped lattice Hamiltonians
  (`239`, `265`); and game-theoretic equilibria / prophet inequalities / graph
  reconstruction (`Cogentic`) target mechanism design rather than single-request
  token generation.

## Reopening criteria for Snapdragon X1 and X2

Do not schedule implementation or device-trial budget for any result in this
document unless all three criteria are met:

1. **Finite-size crossover:** An explicit algorithm with small constant factors
   beats the current `prism-inference` baseline at Swift's actual dimensions
   (`K = 5,120`, `N_ffn = 17,408`, `D = 256`, `N <= 262,144`) in pure Rust.
2. **Numerical & state identity:** Preserves bit-exact `PQ2`/`PTQ1` integer dot
   bounds (`T3`), declared binary32/FP16 rounding or tolerances (`T8`, `T12`),
   and atomic KV/GDN/convolution/sampler rollback (`T5`, `T15`), or enters as a
   separately qualified profile.
3. **Matched-power device benefit:** Passes [promotion criteria](validation.md#promotion-criteria)
   and [owner-device budgets](../AGENTS.md#owner-device-execution-budgets) on
   physical Snapdragon X1/X2 hardware.
