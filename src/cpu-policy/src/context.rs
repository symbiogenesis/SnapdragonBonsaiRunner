//! Pure Swift context sizing from a prelaunch available-physical-memory snapshot.
//!
//! This is conservative admission arithmetic, not measured peak memory. Callers
//! recheck the resulting requirement immediately before starting the owned Job.

/// Swift's verified training context; available memory may select less.
pub const MODEL_CONTEXT_TOKENS: usize = 262_144;
/// Swift vocabulary entry count used for speculative verification logits bounds.
pub const SWIFT_VOCAB_SIZE: usize = 248_320;
/// Minimum selected-profile state/checkpoint reservation (`1 GiB`).
pub const STATE_MEMORY_FLOOR_BYTES: u64 = 1024 * 1024 * 1024;
/// Physical RAM retained for Windows and other applications.
pub const HOST_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Selection-only slack for startup allocations between observation and child admission.
/// The final admission check still requires the Job allowance plus the 4 GiB host reserve.
pub const STARTUP_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
/// Existing Swift allowance excluding its reference 8K FP16 KV payload.
///
/// This retains 11 GiB for weights, recurrence, adapters, runtime and headroom;
/// it is conservative source accounting, not a measured peak.
pub const FIXED_RUNTIME_BYTES: u64 = 11 * 1024 * 1024 * 1024;
/// FP16 attention KV payload per token across Swift's 16 attention layers.
pub const KV_BYTES_PER_TOKEN: u64 = 65_536;
/// Maximum publisher score rows: the fixed 64-token microbatch and 24 heads.
pub const MAX_CONTEXT_SCORE_ROWS: u64 = 64 * 24;
/// Live and rollback KV, publisher attention scratch and bounded token metadata.
/// Native sequential attention uses less scratch; no cache-saving assumption is made.
pub const CONTEXT_BYTES_PER_TOKEN: u64 = 2 * KV_BYTES_PER_TOKEN + MAX_CONTEXT_SCORE_ROWS * 4 + 128;
/// One graph's KV, sequential native attention scores and token/page metadata.
pub const STATE_BYTES_PER_TOKEN: u64 = KV_BYTES_PER_TOKEN + 24 * 4 + 128;
/// K4/V4 records, including their four-byte scales, across Swift's attention layers.
pub const RSQ4_KV_BYTES_PER_TOKEN: u64 = 16_896;
/// `Rotated4FittedValues` K/V records, including block scales, across Swift's attention layers.
pub const ROTATED4_FITTED_VALUES_KV_BYTES_PER_TOKEN: u64 = 16_896;
/// `Rotated4FittedKv` (`rsq4kvf`) K/V records, using identical 132-byte layout with version-3 co-optimized K & V scales.
pub const ROTATED4_FITTED_KV_BYTES_PER_TOKEN: u64 = 16_896;
/// K4/V3 records, including their four-byte scales, across Swift's attention layers.
pub const RSQ4K3V_KV_BYTES_PER_TOKEN: u64 = 14_848;
/// The runtime's reviewed total graph-state allocation ceiling.
pub const STATE_LIMIT_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Swift's F32 recurrent/convolution state independent of attention context.
pub const RECURRENT_BYTES: u64 = 156_893_184;
/// Geometry-derived state/checkpoint ceiling for the reviewed full-context K4/V4 profile.
/// Requires a separately reviewed runtime; ordinary K4/V4 retains its 4 GiB cap.
pub const RSQ4_FULL_STATE_LIMIT_BYTES: u64 =
    262_144_u64 * (RSQ4_KV_BYTES_PER_TOKEN + 96 + 128) + RECURRENT_BYTES;
/// Context selection granularity, divisible by the runtime's 64-token cache pages.
pub const CONTEXT_PAGE_TOKENS: usize = 256;
/// Number of tokens sealed into each immutable copy-on-write KV cache page.
pub const KV_COW_PAGE_TOKENS: usize = 64;
/// Resident PQ2 projection bytes offloaded to Hexagon NPU (`blk.0..=26`, 168 projections)
/// eligible for host CPU weight deduplication when CPU tiled prefill is disabled.
pub const HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES: u64 = 2_888_663_040;
/// Maximum reusable multi-branch recurrent checkpoint slots admitted per session.
pub const MAX_REUSABLE_BRANCH_CHECKPOINTS: u8 = 8;
/// Swift's largest projection activation/output width (FFN intermediate width).
pub const SWIFT_MAX_LAYER_WIDTH: usize = 17_408;
/// Bounded task-slice and error-descriptor reserve for parallel tiled prefill.
pub const SWIFT_ROW_TASK_SCRATCH_BYTES: u64 = 16 * 1024;
/// Preserve room for the largest reply, template margin and a positive prompt.
pub const MIN_CONTEXT_TOKENS: usize =
    (4_096_usize + 64).div_ceil(CONTEXT_PAGE_TOKENS) * CONTEXT_PAGE_TOKENS;

fn tile_scratch_fixed_and_per_ctx(
    tokens: usize,
    kv_format: SwiftKvCacheFormat,
) -> Option<(u64, u64)> {
    if !matches!(tokens, 8 | 16 | 32) {
        return None;
    }
    let t = u64::try_from(tokens).ok()?;
    let score_heads = match kv_format {
        SwiftKvCacheFormat::F16 => 6_u64,
        SwiftKvCacheFormat::Rsq4
        | SwiftKvCacheFormat::Rsq4K3v
        | SwiftKvCacheFormat::Rotated4FittedValues
        | SwiftKvCacheFormat::Rotated4FittedKv => 1_u64,
    };
    let row_bytes = t.checked_mul(u64::try_from(SWIFT_MAX_LAYER_WIDTH).ok()?.checked_mul(80)?)?;
    let meta_bytes = t.checked_mul(2_048)?;
    let fixed = row_bytes
        .checked_add(meta_bytes)?
        .checked_add(64 * 1024)?
        .checked_add(SWIFT_ROW_TASK_SCRATCH_BYTES)?;
    let per_ctx = t.checked_mul(score_heads * 4)?;
    Some((fixed, per_ctx))
}

/// Conservative additional scratch reservation (`bytes`) required by Prism's
/// `Qwen35::tiled_prefill_scratch_bytes` for an explicit `8`, `16`, or `32`-token
/// CPU tiled prefill configuration at `context_tokens` and `kv_format`.
///
/// Returns `None` for unsupported tile sizes (`!(8|16|32)`), `context_tokens == 0`,
/// or `u64` arithmetic overflow.
#[must_use]
pub fn swift_tiled_prefill_scratch_bytes(
    tokens: usize,
    context_tokens: usize,
    kv_format: SwiftKvCacheFormat,
) -> Option<u64> {
    if context_tokens == 0 {
        return None;
    }
    let (fixed, per_ctx) = tile_scratch_fixed_and_per_ctx(tokens, kv_format)?;
    let ctx = u64::try_from(context_tokens).ok()?;
    let scores = ctx.checked_mul(per_ctx)?;
    fixed.checked_add(scores)
}

/// Conservative selected-profile state/checkpoint budget (`bytes`) for
/// `context_tokens` and `kv_format`, matching Prism's `selected_state_budget`.
#[must_use]
pub fn swift_selected_state_budget_bytes(
    context_tokens: usize,
    kv_format: SwiftKvCacheFormat,
) -> Option<u64> {
    if context_tokens == 0 {
        return None;
    }
    let per_token = kv_format.kv_bytes_per_token().checked_add(96 + 128)?;
    let total = u64::try_from(context_tokens)
        .ok()?
        .checked_mul(per_token)?
        .checked_add(RECURRENT_BYTES)?;
    Some(total.max(STATE_MEMORY_FLOOR_BYTES))
}

/// Conservative logits verification scratch (`bytes`) required by Prism's
/// `Qwen35::tiled_verification_scratch_bytes` for an explicit `8`, `16`, or `32`-token
/// CPU tiled prefill configuration.
#[must_use]
pub fn swift_tiled_verification_scratch_bytes(prefill_tokens: usize) -> Option<u64> {
    if !matches!(prefill_tokens, 8 | 16 | 32) {
        return None;
    }
    let rows = u64::try_from(prefill_tokens.min(8)).ok()?;
    let vocab = u64::try_from(SWIFT_VOCAB_SIZE).ok()?;
    let row_vecs = rows.checked_mul(24)?;
    rows.checked_mul(vocab)?
        .checked_mul(4)?
        .checked_add(row_vecs)
}

/// Immutable cache and execution layout used to admit one Swift model lifetime.
///
/// Native profiles require a separately reviewed, source-bound runtime release.
/// The publisher retains its original FP16 cache and microbatch score allowance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwiftContextProfile {
    /// Publisher FP16 cache with 64-token microbatch attention scratch.
    PublisherF16,
    /// Native FP16 cache with sequential attention scratch.
    NativeF16,
    /// Native K4/V4 cache with sequential attention scratch.
    NativeRsq4,
    /// Explicit full-model K4/V4 capacity with its separately reviewed state bound.
    NativeRsq4Full,
    /// Explicit native K4/V3 capacity profile with sequential attention scratch.
    NativeRsq4K3v,
}

impl SwiftContextProfile {
    /// Attention payload per token, including packed record scales.
    #[must_use]
    pub const fn kv_bytes_per_token(self) -> u64 {
        match self {
            Self::PublisherF16 | Self::NativeF16 => KV_BYTES_PER_TOKEN,
            Self::NativeRsq4 | Self::NativeRsq4Full => RSQ4_KV_BYTES_PER_TOKEN,
            Self::NativeRsq4K3v => RSQ4K3V_KV_BYTES_PER_TOKEN,
        }
    }

    /// Two complete caches, selected score scratch and bounded token metadata.
    ///
    /// Shared immutable pages do not reduce this general divergent-history bound.
    #[must_use]
    pub const fn context_bytes_per_token(self) -> u64 {
        let score_rows = match self {
            Self::PublisherF16 => MAX_CONTEXT_SCORE_ROWS,
            Self::NativeF16 | Self::NativeRsq4 | Self::NativeRsq4Full | Self::NativeRsq4K3v => 24,
        };
        2 * self.kv_bytes_per_token() + score_rows * 4 + 128
    }

    /// One live graph's cache, sequential scores and token/page bookkeeping.
    #[must_use]
    pub const fn state_bytes_per_token(self) -> u64 {
        self.kv_bytes_per_token() + 24 * 4 + 128
    }

    /// Immutable graph/checkpoint ceiling selected with the reviewed runtime profile.
    #[must_use]
    pub const fn state_limit_bytes(self) -> u64 {
        match self {
            Self::NativeRsq4Full => RSQ4_FULL_STATE_LIMIT_BYTES,
            _ => STATE_LIMIT_BYTES,
        }
    }
}

/// Concrete attention KV representation selected for an execution shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwiftKvCacheFormat {
    /// Unpacked FP16 attention KV records (`65_536` bytes/token).
    F16,
    /// Native K4/V4 block-quantized KV records (`16_896` bytes/token).
    Rsq4,
    /// Native K4/V3 asymmetric KV records (`14_848` bytes/token).
    Rsq4K3v,
    /// Hadamard-rotated 4-bit fitted-values KV records (`16_896` bytes/token).
    Rotated4FittedValues,
    /// Hadamard-rotated 4-bit co-optimized key and value KV records (`16_896` bytes/token).
    Rotated4FittedKv,
}

impl SwiftKvCacheFormat {
    /// Attention KV payload per token, including per-block scales.
    #[must_use]
    pub const fn kv_bytes_per_token(self) -> u64 {
        match self {
            Self::F16 => KV_BYTES_PER_TOKEN,
            Self::Rsq4 => RSQ4_KV_BYTES_PER_TOKEN,
            Self::Rsq4K3v => RSQ4K3V_KV_BYTES_PER_TOKEN,
            Self::Rotated4FittedValues => ROTATED4_FITTED_VALUES_KV_BYTES_PER_TOKEN,
            Self::Rotated4FittedKv => ROTATED4_FITTED_KV_BYTES_PER_TOKEN,
        }
    }

    /// Map an admission profile to its base attention KV representation.
    #[must_use]
    pub const fn from_profile(profile: SwiftContextProfile) -> Self {
        match profile {
            SwiftContextProfile::PublisherF16 | SwiftContextProfile::NativeF16 => Self::F16,
            SwiftContextProfile::NativeRsq4 | SwiftContextProfile::NativeRsq4Full => Self::Rsq4,
            SwiftContextProfile::NativeRsq4K3v => Self::Rsq4K3v,
        }
    }
}

/// Model weight packing representation used to calibrate host weight footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwiftWeightPacking {
    /// Pinned source PQ2.0 mixed-precision packing (`7_206_168_928` bytes).
    Pq2_0,
    /// Derived repacked PTQ1.0 weights (`5_946_648_928` bytes; `1_259_520_000` bytes smaller).
    Ptq1_0,
    /// Opt-in `NariQ` `2.5bpw` trellis weights (`8_443_119_712` bytes).
    NariQ2_5Bpw,
    /// Opt-in `NariQ` `3.0bpw` trellis weights (`10_123_626_592` bytes).
    NariQ3_0Bpw,
    /// Opt-in `NariQ` `2.0bpw` trellis weights (`6_753_954_304` bytes).
    NariQ2_0Bpw,
    /// Opt-in `Ternary Bonsai 2.5` `PQ2_0` weights (`7_253_354_848` bytes).
    Bonsai25Pq2,
    /// Opt-in `Ternary Bonsai 2.5` `PTQ1_0` weights (`5_993_834_848` bytes).
    Bonsai25Ptq1,
}

impl SwiftWeightPacking {
    /// Converts a selected opt-in [`crate::NariqVariant`] into its host packing policy.
    #[must_use]
    pub const fn from_nariq(variant: crate::NariqVariant) -> Self {
        match variant {
            crate::NariqVariant::Bpw2_5 => Self::NariQ2_5Bpw,
            crate::NariqVariant::Bpw3_0 => Self::NariQ3_0Bpw,
            crate::NariqVariant::Bpw2_0 => Self::NariQ2_0Bpw,
        }
    }

    /// Converts a selected opt-in [`crate::Bonsai25Variant`] into its host packing policy.
    #[must_use]
    pub const fn from_bonsai25(variant: crate::Bonsai25Variant) -> Self {
        match variant {
            crate::Bonsai25Variant::Pq2 => Self::Bonsai25Pq2,
            crate::Bonsai25Variant::Ptq1 => Self::Bonsai25Ptq1,
        }
    }

    /// Exact pinned GGUF weight byte length for this packing format.
    #[must_use]
    pub const fn weight_bytes(self) -> u64 {
        match self {
            Self::Pq2_0 => crate::SWIFT_WEIGHT_BYTES,
            Self::Ptq1_0 => crate::PTQ1_WEIGHT_BYTES,
            Self::NariQ2_5Bpw => crate::NariqVariant::Bpw2_5.weight_bytes(),
            Self::NariQ3_0Bpw => crate::NariqVariant::Bpw3_0.weight_bytes(),
            Self::NariQ2_0Bpw => crate::NariqVariant::Bpw2_0.weight_bytes(),
            Self::Bonsai25Pq2 => crate::Bonsai25Variant::Pq2.weight_bytes(),
            Self::Bonsai25Ptq1 => crate::Bonsai25Variant::Ptq1.weight_bytes(),
        }
    }

    /// Fixed runtime byte allowance for calibrated CoW admission.
    #[must_use]
    pub const fn calibrated_fixed_runtime_bytes(self) -> u64 {
        match self {
            Self::Pq2_0 => FIXED_RUNTIME_BYTES,
            Self::Ptq1_0 => {
                FIXED_RUNTIME_BYTES - (crate::SWIFT_WEIGHT_BYTES - crate::PTQ1_WEIGHT_BYTES)
            }
            Self::NariQ2_5Bpw | Self::NariQ3_0Bpw | Self::NariQ2_0Bpw => {
                (FIXED_RUNTIME_BYTES - crate::SWIFT_WEIGHT_BYTES)
                    + self.weight_bytes()
                    + crate::NARIQ_TRELLIS_SCRATCH_HEADROOM_BYTES
            }
            Self::Bonsai25Pq2 | Self::Bonsai25Ptq1 => {
                (FIXED_RUNTIME_BYTES - crate::SWIFT_WEIGHT_BYTES) + self.weight_bytes()
            }
        }
    }
}

/// Explicit worst-case runtime, checkpoint, scratch, and accelerator shape for
/// Swift admission accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwiftExecutionShape {
    /// Attention KV format charged per token per live KV copy.
    pub kv_format: SwiftKvCacheFormat,
    /// Model weights and fixed runtime allowance charged inside the Job commit limit.
    pub weights_and_fixed_runtime_bytes: u64,
    /// OS-evictable read-only `MapViewOfFile` weight section bytes credited against
    /// private Windows Job commit while remaining charged in physical RAM admission.
    pub evictable_mapped_weight_bytes: u64,
    /// Live active recurrent/convolution state copies (`RECURRENT_BYTES` each).
    pub active_recurrent_copies: u8,
    /// Live rollback recurrent/convolution checkpoint copies (`RECURRENT_BYTES` each).
    pub rollback_recurrent_copies: u8,
    /// Live prompt-reuse recurrent/convolution checkpoint copies (`RECURRENT_BYTES` each).
    pub reuse_recurrent_copies: u8,
    /// Live KV cache copies retained concurrently across active, rollback, or reuse histories.
    pub live_kv_copies: u8,
    /// Attention score row allowance per token (`24` sequential or `MAX_CONTEXT_SCORE_ROWS` batched).
    pub score_rows_per_token: u64,
    /// Token and page table bookkeeping bytes charged per token.
    pub per_token_metadata_bytes: u64,
    /// Fixed scratch budget for single-token sequential decode.
    pub decode_scratch_bytes: u64,
    /// Fixed scratch budget for tiled prompt prefill.
    pub tile_prefill_scratch_bytes: u64,
    /// Fixed verification scratch for speculative decoding.
    pub speculative_verify_scratch_bytes: u64,
    /// Multi-row target logits scratch required during speculative verification.
    pub multi_row_logits_bytes: u64,
    /// Recurrent state replay scratch during delayed verification/rollback.
    pub replay_scratch_bytes: u64,
    /// Optional accelerator resident arena tracked outside Windows Job private commit.
    pub accelerator_resident_arena_bytes: u64,
    /// Optional accelerator transfer staging buffers tracked outside Windows Job private commit.
    pub accelerator_staging_bytes: u64,
    /// Allocator fragmentation margin included inside the Windows Job commit limit.
    pub allocator_fragmentation_margin_bytes: u64,
}

impl SwiftExecutionShape {
    /// Conservative execution shape matching an existing [`SwiftContextProfile`],
    /// explicitly accounting for active, rollback, and reuse recurrent copies.
    #[must_use]
    pub const fn for_profile(profile: SwiftContextProfile) -> Self {
        let score_rows_per_token = match profile {
            SwiftContextProfile::PublisherF16 => MAX_CONTEXT_SCORE_ROWS,
            SwiftContextProfile::NativeF16
            | SwiftContextProfile::NativeRsq4
            | SwiftContextProfile::NativeRsq4Full
            | SwiftContextProfile::NativeRsq4K3v => 24,
        };
        Self {
            kv_format: SwiftKvCacheFormat::from_profile(profile),
            weights_and_fixed_runtime_bytes: FIXED_RUNTIME_BYTES,
            evictable_mapped_weight_bytes: 0,
            active_recurrent_copies: 1,
            rollback_recurrent_copies: 1,
            reuse_recurrent_copies: 1,
            live_kv_copies: 2,
            score_rows_per_token,
            per_token_metadata_bytes: 128,
            decode_scratch_bytes: 0,
            tile_prefill_scratch_bytes: 0,
            speculative_verify_scratch_bytes: 0,
            multi_row_logits_bytes: 0,
            replay_scratch_bytes: 0,
            accelerator_resident_arena_bytes: 0,
            accelerator_staging_bytes: 0,
            allocator_fragmentation_margin_bytes: 0,
        }
    }

    /// Calibrate [`Self::weights_and_fixed_runtime_bytes`] for `packing` relative
    /// to [`FIXED_RUNTIME_BYTES`].
    #[must_use]
    pub const fn with_weight_packing_calibrated(mut self, packing: SwiftWeightPacking) -> Self {
        self.weights_and_fixed_runtime_bytes = packing.calibrated_fixed_runtime_bytes();
        self
    }

    /// Credit `mapped_weight_bytes` of OS-evictable read-only `MapViewOfFile` weight
    /// section pages out of private Windows Job commit while keeping
    /// [`SwiftMemoryBreakdown::total_footprint_bytes`] honest for physical RAM admission.
    ///
    /// Returns `None` if `mapped_weight_bytes > self.weights_and_fixed_runtime_bytes`.
    #[must_use]
    pub fn with_mapped_weights_commit_credit(mut self, mapped_weight_bytes: u64) -> Option<Self> {
        if mapped_weight_bytes > self.weights_and_fixed_runtime_bytes {
            return None;
        }
        self.evictable_mapped_weight_bytes = mapped_weight_bytes;
        Some(self)
    }

    /// Release `released_host_pq2_bytes` of Hexagon-resident `PQ2_0` projection
    /// weights from host CPU memory while adding `resident_arena_bytes` to
    /// [`Self::accelerator_resident_arena_bytes`].
    ///
    /// Returns `None` when CPU tiled prefill scratch is nonzero (since CPU tiled
    /// prefill retains host weights), when `released_host_pq2_bytes` exceeds
    /// [`HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES`] or [`Self::weights_and_fixed_runtime_bytes`],
    /// when remaining host weight allowance would fall below
    /// [`Self::evictable_mapped_weight_bytes`], or on `u64` arithmetic overflow.
    #[must_use]
    pub fn with_hexagon_host_deduplication(
        mut self,
        resident_arena_bytes: u64,
        released_host_pq2_bytes: u64,
    ) -> Option<Self> {
        if self.tile_prefill_scratch_bytes != 0
            || released_host_pq2_bytes > HEXAGON_RESIDENT_PQ2_HOST_DEDUP_BYTES
        {
            return None;
        }
        let remaining_weights = self
            .weights_and_fixed_runtime_bytes
            .checked_sub(released_host_pq2_bytes)?;
        if remaining_weights < self.evictable_mapped_weight_bytes {
            return None;
        }
        self.weights_and_fixed_runtime_bytes = remaining_weights;
        self.accelerator_resident_arena_bytes = self
            .accelerator_resident_arena_bytes
            .checked_add(resident_arena_bytes)?;
        Some(self)
    }

    /// Reserve `branch_slots` (`0..=8`) prompt-reuse recurrent/convolution checkpoint
    /// slots (`RECURRENT_BYTES` per slot).
    ///
    /// Returns `None` when `branch_slots > MAX_REUSABLE_BRANCH_CHECKPOINTS`.
    #[must_use]
    pub const fn with_branch_reuse_checkpoints(mut self, branch_slots: u8) -> Option<Self> {
        if branch_slots > MAX_REUSABLE_BRANCH_CHECKPOINTS {
            return None;
        }
        self.reuse_recurrent_copies = branch_slots;
        Some(self)
    }

    /// Set [`Self::live_kv_copies`] to `1` for append-only copy-on-write KV admission.
    #[must_use]
    pub const fn with_append_only_cow_kv_admission(mut self) -> Self {
        self.live_kv_copies = 1;
        self
    }

    /// Add `budget.total_bytes()` for an active multi-rank DaisyRose adapter into
    /// [`Self::weights_and_fixed_runtime_bytes`], or return `None` on overflow.
    #[must_use]
    pub fn with_multi_rank_adapter(mut self, budget: MultiRankAdapterBudget) -> Option<Self> {
        let total = budget.total_bytes()?;
        self.weights_and_fixed_runtime_bytes =
            self.weights_and_fixed_runtime_bytes.checked_add(total)?;
        Some(self)
    }

    /// Total live recurrent/convolution checkpoint bytes across active, rollback,
    /// and reuse copies (`156_893_184` bytes per copy), or `None` on overflow.
    #[must_use]
    pub fn recurrent_checkpoint_bytes(&self) -> Option<u64> {
        let copies = u64::from(self.active_recurrent_copies)
            .checked_add(u64::from(self.rollback_recurrent_copies))?
            .checked_add(u64::from(self.reuse_recurrent_copies))?;
        copies.checked_mul(RECURRENT_BYTES)
    }

    /// Live KV cache, score scratch, and metadata bytes charged per admitted token.
    #[must_use]
    pub fn kv_bytes_per_token_total(&self) -> Option<u64> {
        let kv = u64::from(self.live_kv_copies).checked_mul(self.kv_format.kv_bytes_per_token())?;
        let scores = self.score_rows_per_token.checked_mul(4)?;
        kv.checked_add(scores)?
            .checked_add(self.per_token_metadata_bytes)
    }

    /// Worst-case execution scratch bytes:
    /// `max(decode_scratch, tile_prefill_scratch, speculative_verify_scratch + multi_row_logits + replay_scratch)`.
    #[must_use]
    pub fn execution_scratch_bytes(&self) -> Option<u64> {
        let speculative_total = self
            .speculative_verify_scratch_bytes
            .checked_add(self.multi_row_logits_bytes)?
            .checked_add(self.replay_scratch_bytes)?;
        Some(
            self.decode_scratch_bytes
                .max(self.tile_prefill_scratch_bytes)
                .max(speculative_total),
        )
    }

    /// Attach Prism's exact `CPU` tiled-prefill scratch reserve for `tokens` (`8`,
    /// `16`, or `32`) at `context_tokens` using this shape's `kv_format`.
    #[must_use]
    pub fn with_tiled_prefill_tokens(
        mut self,
        tokens: usize,
        context_tokens: usize,
    ) -> Option<Self> {
        self.tile_prefill_scratch_bytes =
            swift_tiled_prefill_scratch_bytes(tokens, context_tokens, self.kv_format)?;
        Some(self)
    }

    /// Physical copy-on-write `KV` cache and per-token bookkeeping bytes when
    /// sealed `64`-token `Arc` pages are shared across `live_kv_copies`, keeping
    /// at most one unsealed tail page per additional history.
    #[must_use]
    pub fn physical_cow_kv_bytes(&self, context_tokens: usize) -> Option<u64> {
        if context_tokens == 0 {
            return Some(0);
        }
        let ctx = u64::try_from(context_tokens).ok()?;
        let bookkeeping = ctx.checked_mul(
            self.score_rows_per_token
                .checked_mul(4)?
                .checked_add(self.per_token_metadata_bytes)?,
        )?;
        let kv_per_token = self.kv_format.kv_bytes_per_token();
        if self.live_kv_copies == 0 {
            return Some(bookkeeping);
        }
        let extra_copies = u64::from(self.live_kv_copies - 1);
        let tail_tokens = u64::try_from(context_tokens.min(KV_COW_PAGE_TOKENS)).ok()?;
        let shared_kv = ctx.checked_mul(kv_per_token)?;
        let unsealed_tails = extra_copies
            .checked_mul(tail_tokens)?
            .checked_mul(kv_per_token)?;
        shared_kv
            .checked_add(unsealed_tails)?
            .checked_add(bookkeeping)
    }

    /// Itemize all worst-case memory categories for `context_tokens`.
    #[must_use]
    pub fn breakdown(&self, context_tokens: usize) -> Option<SwiftMemoryBreakdown> {
        let tokens = u64::try_from(context_tokens).ok()?;
        let per_token = self.kv_bytes_per_token_total()?;
        let live_kv_bytes = tokens.checked_mul(per_token)?;
        let physical_cow_kv_bytes = self.physical_cow_kv_bytes(context_tokens)?;
        let recurrent_checkpoint_bytes = self.recurrent_checkpoint_bytes()?;
        let execution_scratch_bytes = self.execution_scratch_bytes()?;
        Some(SwiftMemoryBreakdown {
            weights_and_fixed_runtime_bytes: self.weights_and_fixed_runtime_bytes,
            evictable_mapped_weight_bytes: self.evictable_mapped_weight_bytes,
            recurrent_checkpoint_bytes,
            live_kv_bytes,
            physical_cow_kv_bytes,
            execution_scratch_bytes,
            accelerator_resident_arena_bytes: self.accelerator_resident_arena_bytes,
            accelerator_staging_bytes: self.accelerator_staging_bytes,
            allocator_fragmentation_margin_bytes: self.allocator_fragmentation_margin_bytes,
        })
    }
}

/// Itemized worst-case execution memory ledger across CPU Job commit and
/// unified-memory accelerator reservations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwiftMemoryBreakdown {
    /// Model weights and fixed runtime allowance (`a`).
    pub weights_and_fixed_runtime_bytes: u64,
    /// OS-evictable read-only `MapViewOfFile` weight section bytes credited against
    /// private Windows Job commit while remaining included in physical RAM admission.
    pub evictable_mapped_weight_bytes: u64,
    /// Live recurrent/convolution checkpoint copies across active, rollback, and reuse (`b`).
    pub recurrent_checkpoint_bytes: u64,
    /// Live KV cache copies and per-token bookkeeping for the selected context (`c`).
    pub live_kv_bytes: u64,
    /// Copy-on-write shared-page `KV` footprint across active and rollback histories.
    pub physical_cow_kv_bytes: u64,
    /// Worst-case execution scratch across decode, tile prefill, and speculative verify/replay (`d`).
    pub execution_scratch_bytes: u64,
    /// Optional accelerator resident arena outside Windows Job private commit (`e`).
    pub accelerator_resident_arena_bytes: u64,
    /// Optional accelerator transfer staging bytes outside Windows Job private commit (`e`).
    pub accelerator_staging_bytes: u64,
    /// Allocator fragmentation margin included inside the Windows Job commit limit (`f`).
    pub allocator_fragmentation_margin_bytes: u64,
}

impl SwiftMemoryBreakdown {
    /// Host private memory subject to the Windows Job commit ceiling after
    /// crediting `evictable_mapped_bytes` of OS-evictable read-only `MapViewOfFile`
    /// weight section pages.
    ///
    /// Returns `None` if `evictable_mapped_bytes > self.weights_and_fixed_runtime_bytes`
    /// or on arithmetic overflow. Physical RAM admission via
    /// [`Self::total_footprint_bytes`] continues to charge full weights.
    #[must_use]
    pub fn private_job_commit_with_mapped_credit(
        &self,
        evictable_mapped_bytes: u64,
    ) -> Option<u64> {
        self.weights_and_fixed_runtime_bytes
            .checked_sub(evictable_mapped_bytes)?
            .checked_add(self.recurrent_checkpoint_bytes)?
            .checked_add(self.live_kv_bytes)?
            .checked_add(self.execution_scratch_bytes)?
            .checked_add(self.allocator_fragmentation_margin_bytes)
    }

    /// Complete host private memory subject to the Windows Job commit ceiling
    /// (`weights - evictable_mapped + fixed runtime + recurrent copies + live KV + scratch + fragmentation`),
    /// excluding external accelerator resident arena, staging allocations, and
    /// OS-evictable mapped weight pages.
    #[must_use]
    pub fn job_commit_bytes(&self) -> Option<u64> {
        self.private_job_commit_with_mapped_credit(self.evictable_mapped_weight_bytes)
    }

    /// Expected steady-state physical footprint when immutable `64`-token `KV`
    /// pages are shared via `Arc` copy-on-write rather than duplicated across
    /// active and rollback histories, excluding [`HOST_RESERVE_BYTES`].
    #[must_use]
    pub fn physical_cow_footprint_bytes(&self) -> Option<u64> {
        self.weights_and_fixed_runtime_bytes
            .checked_add(self.recurrent_checkpoint_bytes)?
            .checked_add(self.physical_cow_kv_bytes)?
            .checked_add(self.execution_scratch_bytes)?
            .checked_add(self.allocator_fragmentation_margin_bytes)?
            .checked_add(self.accelerator_bytes()?)
    }

    /// Total optional accelerator resident arena and staging bytes tracked
    /// outside the Windows Job private commit limit.
    #[must_use]
    pub fn accelerator_bytes(&self) -> Option<u64> {
        self.accelerator_resident_arena_bytes
            .checked_add(self.accelerator_staging_bytes)
    }

    /// Combined physical RAM ledger across host weights/commit and unified
    /// accelerator reservations, excluding [`HOST_RESERVE_BYTES`].
    #[must_use]
    pub fn total_footprint_bytes(&self) -> Option<u64> {
        self.job_commit_bytes()?
            .checked_add(self.evictable_mapped_weight_bytes)?
            .checked_add(self.accelerator_bytes()?)
    }
}

/// A context and its complete conservative Job/admission requirements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwiftContextPlan {
    /// Fixed prompt, response and template reservation budget for one model lifetime.
    pub context_tokens: usize,
    /// Complete owned-Job memory ceiling including fixed runtime headroom.
    pub memory_limit_bytes: usize,
    /// Required available physical RAM, including the separate host reserve.
    pub required_available_bytes: u64,
}

impl SwiftContextPlan {
    /// Select the largest page-aligned context fitting physical RAM and graph state.
    ///
    /// This compatibility entry point preserves publisher FP16/microbatch sizing.
    /// Native source consumers must select their immutable layout with
    /// [`Self::for_profile`] after verifying the matching runtime contract.
    ///
    /// # Errors
    /// Returns a value-free error when the minimum usable context cannot fit or
    /// checked arithmetic cannot represent the selected complete allocation.
    pub fn for_available_memory(available_bytes: u64) -> Result<Self, String> {
        Self::for_profile(available_bytes, SwiftContextProfile::PublisherF16)
    }

    /// Select the largest page-aligned context for an explicit cache/execution layout.
    ///
    /// The two-cache process charge preserves rollback while an unrelated history
    /// is computed. Optional tiled prefill or speculation require their own additive
    /// reservations and are not admitted by these sequential native profiles.
    ///
    /// # Errors
    /// Rejects insufficient memory or an unrepresentable complete allocation.
    pub fn for_profile(available_bytes: u64, profile: SwiftContextProfile) -> Result<Self, String> {
        let process_bytes = profile.context_bytes_per_token();
        let context_tokens = admit_context_tokens(
            available_bytes,
            profile,
            FIXED_RUNTIME_BYTES,
            process_bytes,
            MODEL_CONTEXT_TOKENS,
        )?;
        let context_bytes = u64::try_from(context_tokens)
            .ok()
            .and_then(|tokens| tokens.checked_mul(process_bytes))
            .ok_or_else(insufficient)?;
        let memory_limit = FIXED_RUNTIME_BYTES
            .checked_add(context_bytes)
            .ok_or_else(insufficient)?;
        let required_available_bytes = memory_limit
            .checked_add(HOST_RESERVE_BYTES)
            .ok_or_else(insufficient)?;
        Ok(Self {
            context_tokens,
            memory_limit_bytes: usize::try_from(memory_limit).map_err(|_| insufficient())?,
            required_available_bytes,
        })
    }

    /// Select the largest page-aligned context admitted by both the base
    /// `profile` curve and an explicit [`SwiftExecutionShape`] memory ledger.
    ///
    /// Guarantees `context_tokens` never exceeds [`Self::for_profile`] under
    /// the same `available_bytes`, sizes `memory_limit_bytes` strictly from
    /// [`SwiftMemoryBreakdown::job_commit_bytes`], and charges both Job commit
    /// and external accelerator allocations against `available_bytes`.
    ///
    /// # Errors
    /// Rejects insufficient memory, zero per-token footprint, or arithmetic
    /// overflow while assembling the breakdown.
    pub fn for_execution_shape(
        available_bytes: u64,
        profile: SwiftContextProfile,
        shape: &SwiftExecutionShape,
    ) -> Result<(Self, SwiftMemoryBreakdown), String> {
        let baseline = Self::for_profile(available_bytes, profile)?;
        let zero_context = shape.breakdown(0).ok_or_else(insufficient)?;
        let fixed_physical = zero_context
            .total_footprint_bytes()
            .ok_or_else(insufficient)?;
        let per_token = shape.kv_bytes_per_token_total().ok_or_else(insufficient)?;
        let context_tokens = admit_context_tokens(
            available_bytes,
            profile,
            fixed_physical,
            per_token,
            baseline.context_tokens,
        )?;
        let breakdown = shape.breakdown(context_tokens).ok_or_else(insufficient)?;
        let job_commit = breakdown.job_commit_bytes().ok_or_else(insufficient)?;
        let required_available_bytes = breakdown
            .total_footprint_bytes()
            .and_then(|bytes| bytes.checked_add(HOST_RESERVE_BYTES))
            .ok_or_else(insufficient)?;
        let plan = Self {
            context_tokens,
            memory_limit_bytes: usize::try_from(job_commit).map_err(|_| insufficient())?,
            required_available_bytes,
        };
        Ok((plan, breakdown))
    }

    /// Select the largest page-aligned context admitted under copy-on-write `KV`
    /// accounting for an explicit [`SwiftExecutionShape`].
    ///
    /// Charges [`SwiftExecutionShape::physical_cow_kv_bytes`] (`1` shared copy plus
    /// `<= 64`-token unsealed tail per extra live KV copy) alongside recurrent,
    /// scratch, fragmentation, weight, and accelerator footprints while still
    /// enforcing [`HOST_RESERVE_BYTES`], [`STARTUP_RESERVE_BYTES`], and
    /// `profile.state_limit_bytes()`.
    ///
    /// # Errors
    /// Rejects zero live KV copies, invalid mapped commit credit, insufficient
    /// memory, or arithmetic overflow.
    pub fn for_append_only_cow_shape(
        available_bytes: u64,
        profile: SwiftContextProfile,
        shape: &SwiftExecutionShape,
    ) -> Result<Self, String> {
        if shape.live_kv_copies == 0 {
            return Err(insufficient());
        }
        let zero_context = shape.breakdown(0).ok_or_else(insufficient)?;
        // Enforce mapped credit bounds (`evictable_mapped <= weights`) even though
        // physical RAM admission continues to charge full host weights.
        let _ = zero_context.job_commit_bytes().ok_or_else(insufficient)?;
        let fixed_physical = zero_context
            .physical_cow_footprint_bytes()
            .ok_or_else(insufficient)?;
        let kv_per_token = shape.kv_format.kv_bytes_per_token();
        let bookkeeping_per_token = shape
            .score_rows_per_token
            .checked_mul(4)
            .and_then(|scores| scores.checked_add(shape.per_token_metadata_bytes))
            .ok_or_else(insufficient)?;
        let per_cow_token = kv_per_token
            .checked_add(bookkeeping_per_token)
            .ok_or_else(insufficient)?;
        let extra_copies = u64::from(shape.live_kv_copies - 1);
        let cow_tail_tokens = u64::try_from(KV_COW_PAGE_TOKENS).map_err(|_| insufficient())?;
        let unsealed_tail_bytes = extra_copies
            .checked_mul(cow_tail_tokens)
            .and_then(|tokens| tokens.checked_mul(kv_per_token))
            .ok_or_else(insufficient)?;
        let fixed_cow = fixed_physical
            .checked_add(unsealed_tail_bytes)
            .ok_or_else(insufficient)?;
        let context_tokens = admit_context_tokens(
            available_bytes,
            profile,
            fixed_cow,
            per_cow_token,
            MODEL_CONTEXT_TOKENS,
        )?;
        let breakdown = shape.breakdown(context_tokens).ok_or_else(insufficient)?;
        let physical_cow = breakdown
            .physical_cow_footprint_bytes()
            .ok_or_else(insufficient)?;
        let accelerator = breakdown.accelerator_bytes().ok_or_else(insufficient)?;
        let cow_job_commit = physical_cow
            .checked_sub(accelerator)
            .and_then(|host| host.checked_sub(breakdown.evictable_mapped_weight_bytes))
            .ok_or_else(insufficient)?;
        let required_available_bytes = physical_cow
            .checked_add(HOST_RESERVE_BYTES)
            .ok_or_else(insufficient)?;
        Ok(Self {
            context_tokens,
            memory_limit_bytes: usize::try_from(cow_job_commit).map_err(|_| insufficient())?,
            required_available_bytes,
        })
    }

    /// Derive a context plan for `profile`, reserving [`crate::MMPROJ_RESERVE_BYTES`]
    /// when `request_vision` is true and fitting within `available_bytes`.
    ///
    /// If `available_bytes` admits text-only Swift but cannot accommodate the vision
    /// projector reserve, this cleanly falls back to `(text_only_plan, false)`.
    ///
    /// # Errors
    /// Returns an error if even text-only Swift cannot fit within `available_bytes`.
    pub fn for_profile_with_vision(
        available_bytes: u64,
        profile: SwiftContextProfile,
        request_vision: bool,
    ) -> Result<(Self, bool), String> {
        let (plan, vision, _) =
            Self::for_profile_with_options(available_bytes, profile, request_vision, None)?;
        Ok((plan, vision))
    }

    /// Derive a context plan for `profile`, adding an optional `CPU` tiled prefill
    /// scratch reservation (`8`, `16`, or `32` tokens) within `available_bytes`.
    ///
    /// Returns `(plan, Some(prefill_scratch_bytes))` when `prefill_tokens` is
    /// `Some`, guaranteeing `plan.memory_limit_bytes - prefill_scratch_bytes` equals
    /// the exact paired native `--context-tokens` and base `--memory-limit-bytes`
    /// allowance checked by Prism.
    ///
    /// # Errors
    /// Rejects unsupported profiles (`PublisherF16`), unsupported tile counts,
    /// insufficient memory, or overflow.
    pub fn for_profile_with_prefill(
        available_bytes: u64,
        profile: SwiftContextProfile,
        prefill_tokens: Option<usize>,
    ) -> Result<(Self, Option<usize>), String> {
        let Some(tile_tokens) = prefill_tokens else {
            return Ok((Self::for_profile(available_bytes, profile)?, None));
        };
        if profile == SwiftContextProfile::PublisherF16 {
            return Err(insufficient());
        }
        let kv_format = SwiftKvCacheFormat::from_profile(profile);
        let (fixed_tile, score_per_ctx_token) =
            tile_scratch_fixed_and_per_ctx(tile_tokens, kv_format).ok_or_else(insufficient)?;
        let fixed_bytes = FIXED_RUNTIME_BYTES
            .checked_add(fixed_tile)
            .ok_or_else(insufficient)?;
        let total_per_token = profile
            .context_bytes_per_token()
            .checked_add(score_per_ctx_token)
            .ok_or_else(insufficient)?;
        let context_tokens = admit_context_tokens(
            available_bytes,
            profile,
            fixed_bytes,
            total_per_token,
            MODEL_CONTEXT_TOKENS,
        )?;
        let scratch_u64 = swift_tiled_prefill_scratch_bytes(tile_tokens, context_tokens, kv_format)
            .ok_or_else(insufficient)?;
        let context_bytes = u64::try_from(context_tokens)
            .ok()
            .and_then(|count| count.checked_mul(profile.context_bytes_per_token()))
            .ok_or_else(insufficient)?;
        let memory_limit = FIXED_RUNTIME_BYTES
            .checked_add(context_bytes)
            .and_then(|bytes| bytes.checked_add(scratch_u64))
            .ok_or_else(insufficient)?;
        let required_available_bytes = memory_limit
            .checked_add(HOST_RESERVE_BYTES)
            .ok_or_else(insufficient)?;
        let scratch_bytes = usize::try_from(scratch_u64).map_err(|_| insufficient())?;
        Ok((
            Self {
                context_tokens,
                memory_limit_bytes: usize::try_from(memory_limit).map_err(|_| insufficient())?,
                required_available_bytes,
            },
            Some(scratch_bytes),
        ))
    }

    /// Derive a context plan for `profile`, reserving [`crate::MMPROJ_RESERVE_BYTES`]
    /// when `request_vision` is true and fitting within `available_bytes`, and
    /// optionally reserving `CPU` tiled prefill scratch for `prefill_tokens`.
    ///
    /// If `available_bytes` admits text-only execution but cannot accommodate the
    /// vision projector reserve, this cleanly falls back to
    /// `(text_only_plan, false, scratch)`.
    ///
    /// # Errors
    /// Returns an error when even text-only execution cannot fit within `available_bytes`
    /// or when `prefill_tokens` selects an invalid tile count or non-native profile.
    pub fn for_profile_with_options(
        available_bytes: u64,
        profile: SwiftContextProfile,
        request_vision: bool,
        prefill_tokens: Option<usize>,
    ) -> Result<(Self, bool, Option<usize>), String> {
        with_optional_vision(
            available_bytes,
            request_vision,
            |avail| Self::for_profile_with_prefill(avail, profile, prefill_tokens),
            |_| Ok(()),
        )
    }

    /// Derive a context plan and itemized memory breakdown for `shape`, reserving
    /// [`crate::MMPROJ_RESERVE_BYTES`] when `request_vision` is true and fits.
    ///
    /// Falls back to `(text_only_plan, breakdown, false)` when available RAM fits
    /// text-only execution but cannot admit the vision projector reserve.
    ///
    /// # Errors
    /// Returns an error if even text-only Swift cannot fit within `available_bytes`.
    pub fn for_execution_shape_with_vision(
        available_bytes: u64,
        profile: SwiftContextProfile,
        shape: &SwiftExecutionShape,
        request_vision: bool,
    ) -> Result<(Self, SwiftMemoryBreakdown, bool), String> {
        let (plan, vision, breakdown) = with_optional_vision(
            available_bytes,
            request_vision,
            |avail| Self::for_execution_shape(avail, profile, shape),
            |breakdown| {
                breakdown.weights_and_fixed_runtime_bytes = breakdown
                    .weights_and_fixed_runtime_bytes
                    .checked_add(crate::MMPROJ_RESERVE_BYTES)
                    .ok_or_else(insufficient)?;
                Ok(())
            },
        )?;
        Ok((plan, breakdown, vision))
    }
}

/// Sparse prompt-reuse checkpoint selection for calibrated CoW admission.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SwiftCalibratedReuse {
    /// No prompt-reuse checkpoints retained beyond live and rollback state.
    #[default]
    Disabled,
    /// Retain `1..=8` sparse latest-branch checkpoints.
    Latest {
        /// Number of retained sparse checkpoints (`1..=8`).
        checkpoints: u8,
        /// Optional explicit checkpoint byte override validated against `checkpoints * state_budget`.
        checkpoint_bytes: Option<usize>,
    },
    /// Retain prompt-end checkpoint plus `1..=8` sparse latest-branch checkpoints.
    PromptEndAndLatest {
        /// Number of retained latest checkpoints (`1..=8`), adding 1 prompt-end slot (`2..=9` total).
        checkpoints: u8,
        /// Optional explicit checkpoint byte override validated against `checkpoints * state_budget`.
        checkpoint_bytes: Option<usize>,
    },
}

/// Session execution shape inputs for calibrated CoW context planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SwiftCalibratedOptions {
    /// Weight packing representation calibrating host fixed bytes and mapped credit.
    pub weight_packing: SwiftWeightPacking,
    /// Whether verified multimodal vision (`mmproj`) should be admitted if RAM allows.
    pub request_vision: bool,
    /// Optional CPU tiled prefill token batch (`8`, `16`, or `32`).
    pub prefill_tokens: Option<usize>,
    /// Whether chunkwise WY `F32` GDN prefill is enabled (requires FP16 KV and prefill tiles).
    pub chunkwise_gdn_wy_f32: bool,
    /// Optional suffix speculative draft token cap (`1..=8`; incompatible with vision and WY GDN).
    pub speculation_tokens: Option<u8>,
    /// Prompt-reuse checkpoint mode and sparse slot budget.
    pub prompt_reuse: SwiftCalibratedReuse,
}

/// Conservative memory reserve for an active stackable multi-rank `LoRA` adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiRankAdapterBudget {
    /// Resident factor storage bytes of the verified adapter.
    pub payload_bytes: u64,
    /// Temporary runtime scratch headroom charged on top of payload storage.
    pub scratch_headroom_bytes: u64,
}

impl MultiRankAdapterBudget {
    /// Conservative budget estimate for an `NpuFriendly` adapter of `max_rank` (`1..=64`).
    ///
    /// Charges `4_194_304` payload bytes per rank for `F16` adapters and `8_388_608`
    /// payload bytes per rank for `F32` adapters, plus [`crate::DAISYROSE_SCRATCH_HEADROOM_BYTES`].
    #[must_use]
    pub const fn for_npu_friendly_rank(max_rank: u16, is_f32: bool) -> Option<Self> {
        if max_rank == 0 || max_rank > 64 {
            return None;
        }
        let bytes_per_rank = if is_f32 {
            match crate::DAISYROSE_NPU_FRIENDLY_BYTES_PER_RANK_F16.checked_mul(2) {
                Some(bytes) => bytes,
                None => return None,
            }
        } else {
            crate::DAISYROSE_NPU_FRIENDLY_BYTES_PER_RANK_F16
        };
        let mut payload_bytes = 0_u64;
        let mut remaining = max_rank;
        while remaining > 0 {
            payload_bytes = match payload_bytes.checked_add(bytes_per_rank) {
                Some(sum) => sum,
                None => return None,
            };
            remaining -= 1;
        }
        Self::from_verified_bytes(payload_bytes)
    }

    /// Conservative budget estimate for an `NpuFriendly` `F16` adapter of `max_rank` (`1..=64`).
    #[must_use]
    pub const fn for_npu_friendly_f16(max_rank: u16) -> Option<Self> {
        Self::for_npu_friendly_rank(max_rank, false)
    }

    /// Conservative budget bound from a verified GGUF file's exact byte size.
    #[must_use]
    pub const fn from_verified_bytes(bytes: u64) -> Option<Self> {
        if bytes == 0 {
            return None;
        }
        Some(Self {
            payload_bytes: bytes,
            scratch_headroom_bytes: crate::DAISYROSE_SCRATCH_HEADROOM_BYTES,
        })
    }

    /// Total adapter payload and scratch reserve in bytes.
    #[must_use]
    pub const fn total_bytes(self) -> Option<u64> {
        self.payload_bytes.checked_add(self.scratch_headroom_bytes)
    }
}

/// Calibrated representation- and CoW-aware Swift admission result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SwiftCalibratedPlan {
    /// Selected context, private Windows Job commit ceiling, and physical admission threshold.
    pub context_plan: SwiftContextPlan,
    /// Physical CoW process working-set ceiling (`physical_cow [+ MMPROJ_RESERVE_BYTES]`).
    pub working_set_limit_bytes: usize,
    /// Whether multimodal vision projector headroom was admitted.
    pub vision_admitted: bool,
    /// Additive CPU tiled prefill scratch bytes when `prefill_tokens` is `Some`.
    pub prefill_scratch_bytes: Option<usize>,
    /// Paired `(verification_scratch_bytes, speculative_checkpoint_bytes)` when speculating.
    pub speculation_reserves: Option<(usize, usize)>,
}

fn validate_calibrated_reuse(
    reuse: SwiftCalibratedReuse,
) -> Result<(u64, u64, Option<usize>), String> {
    let (checkpoints, checkpoint_bytes, extra_slot) = match reuse {
        SwiftCalibratedReuse::Disabled => return Ok((0, 0, None)),
        SwiftCalibratedReuse::Latest {
            checkpoints,
            checkpoint_bytes,
        } => (checkpoints, checkpoint_bytes, 0_u64),
        SwiftCalibratedReuse::PromptEndAndLatest {
            checkpoints,
            checkpoint_bytes,
        } => (checkpoints, checkpoint_bytes, 1_u64),
    };
    if !(1..=MAX_REUSABLE_BRANCH_CHECKPOINTS).contains(&checkpoints) || checkpoint_bytes == Some(0)
    {
        return Err(insufficient());
    }
    let min_slots = u64::from(checkpoints);
    let slots = min_slots.checked_add(extra_slot).ok_or_else(insufficient)?;
    Ok((slots, min_slots, checkpoint_bytes))
}

fn eval_calibrated_cow_at(
    available_bytes: u64,
    profile: SwiftContextProfile,
    options: SwiftCalibratedOptions,
    context_tokens: usize,
    reuse_slots: u64,
    min_checkpoint_slots: u64,
    explicit_checkpoint_bytes: Option<usize>,
    include_mmproj: bool,
) -> Option<SwiftCalibratedPlan> {
    let kv_format = SwiftKvCacheFormat::from_profile(profile);
    let ctx = u64::try_from(context_tokens).ok()?;
    let state_bytes = ctx
        .checked_mul(profile.state_bytes_per_token())?
        .checked_add(RECURRENT_BYTES)?;
    if state_bytes > profile.state_limit_bytes() {
        return None;
    }

    let state_budget = swift_selected_state_budget_bytes(context_tokens, kv_format)?;
    if let Some(explicit) = explicit_checkpoint_bytes {
        let explicit_u64 = u64::try_from(explicit).ok()?;
        let min_state_charge = min_checkpoint_slots.checked_mul(state_budget)?;
        if explicit_u64 < min_state_charge {
            return None;
        }
    }

    let fixed_bytes = options.weight_packing.calibrated_fixed_runtime_bytes();
    let base_recurrent = 2_u64
        .checked_add(u64::from(options.speculation_tokens.is_some()))?
        .checked_mul(RECURRENT_BYTES)?;
    let min_reuse_bytes = reuse_slots.checked_mul(RECURRENT_BYTES)?;
    let reuse_recurrent = explicit_checkpoint_bytes.map_or(Some(min_reuse_bytes), |explicit| {
        u64::try_from(explicit)
            .ok()
            .map(|bytes| bytes.max(min_reuse_bytes))
    })?;
    let recurrent_bytes = base_recurrent.checked_add(reuse_recurrent)?;

    let kv_bytes = kv_format.kv_bytes_per_token();
    let cow_kv = ctx
        .checked_mul(kv_bytes.checked_add(96 + 128)?)?
        .checked_add(
            u64::try_from(context_tokens.min(KV_COW_PAGE_TOKENS))
                .ok()?
                .checked_mul(kv_bytes)?,
        )?;

    let prefill_scratch_u64 = match options.prefill_tokens {
        Some(tokens) => Some(swift_tiled_prefill_scratch_bytes(
            tokens,
            context_tokens,
            kv_format,
        )?),
        None => None,
    };
    let verify_scratch_u64 = if options.speculation_tokens.is_some() {
        let prefill = options.prefill_tokens?;
        Some(swift_tiled_verification_scratch_bytes(prefill)?)
    } else {
        None
    };
    let total_scratch = prefill_scratch_u64
        .unwrap_or(0)
        .checked_add(verify_scratch_u64.unwrap_or(0))?;

    let physical_cow = fixed_bytes
        .checked_add(recurrent_bytes)?
        .checked_add(cow_kv)?
        .checked_add(total_scratch)?;
    let mmproj_reserve = if include_mmproj {
        crate::MMPROJ_RESERVE_BYTES
    } else {
        0
    };
    let working_set_u64 = physical_cow.checked_add(mmproj_reserve)?;
    let admission_threshold = working_set_u64
        .checked_add(HOST_RESERVE_BYTES)?
        .checked_add(STARTUP_RESERVE_BYTES)?;
    if admission_threshold > available_bytes {
        return None;
    }

    let private_job_commit = working_set_u64.checked_sub(options.weight_packing.weight_bytes())?;
    let required_available_bytes = working_set_u64.checked_add(HOST_RESERVE_BYTES)?;
    let prefill_scratch_bytes = match prefill_scratch_u64 {
        Some(bytes) => Some(usize::try_from(bytes).ok()?),
        None => None,
    };
    let speculation_reserves = match verify_scratch_u64 {
        Some(verify) => Some((
            usize::try_from(verify).ok()?,
            usize::try_from(state_budget).ok()?,
        )),
        None => None,
    };

    Some(SwiftCalibratedPlan {
        context_plan: SwiftContextPlan {
            context_tokens,
            memory_limit_bytes: usize::try_from(private_job_commit).ok()?,
            required_available_bytes,
        },
        working_set_limit_bytes: usize::try_from(working_set_u64).ok()?,
        vision_admitted: include_mmproj,
        prefill_scratch_bytes,
        speculation_reserves,
    })
}

/// Input parameters and constructor for calibrated CoW context planning.
impl SwiftContextPlan {
    /// Select the largest page-aligned context admitted under Prism's calibrated
    /// representation- and CoW-aware physical policy while crediting OS-evictable
    /// read-only mapped weight pages out of the private Windows Job commit ceiling.
    ///
    /// Falls back cleanly to text-only execution (`vision_admitted = false`) when
    /// available RAM fits text-only execution without [`crate::MMPROJ_RESERVE_BYTES`],
    /// or when CPU speculation is active (since speculation and mmproj are mutually
    /// exclusive).
    ///
    /// # Errors
    /// Rejects non-native profiles, invalid tile/speculation/GDN combinations,
    /// undersized explicit checkpoint reserves, insufficient RAM, or overflow.
    pub fn for_calibrated_execution(
        available_bytes: u64,
        profile: SwiftContextProfile,
        options: SwiftCalibratedOptions,
    ) -> Result<SwiftCalibratedPlan, String> {
        if profile == SwiftContextProfile::PublisherF16 {
            return Err(insufficient());
        }
        let kv_format = SwiftKvCacheFormat::from_profile(profile);
        if let Some(tokens) = options.prefill_tokens
            && !matches!(tokens, 8 | 16 | 32)
        {
            return Err(insufficient());
        }
        if options.chunkwise_gdn_wy_f32
            && (options.prefill_tokens.is_none()
                || kv_format != SwiftKvCacheFormat::F16
                || options.speculation_tokens.is_some())
        {
            return Err(insufficient());
        }
        if let Some(spec_tokens) = options.speculation_tokens
            && (!(1..=8).contains(&spec_tokens)
                || options.prefill_tokens.is_none()
                || options.chunkwise_gdn_wy_f32)
        {
            return Err(insufficient());
        }
        let (reuse_slots, min_checkpoint_slots, explicit_checkpoint_bytes) =
            validate_calibrated_reuse(options.prompt_reuse)?;

        let max_state_tokens_u64 = profile
            .state_limit_bytes()
            .checked_sub(RECURRENT_BYTES)
            .ok_or_else(insufficient)?
            / profile.state_bytes_per_token();
        let max_state_tokens = usize::try_from(max_state_tokens_u64)
            .unwrap_or(MODEL_CONTEXT_TOKENS)
            .min(MODEL_CONTEXT_TOKENS);
        let min_pages = MIN_CONTEXT_TOKENS / CONTEXT_PAGE_TOKENS;
        let max_pages = max_state_tokens / CONTEXT_PAGE_TOKENS;
        if min_pages > max_pages {
            return Err(insufficient());
        }

        let search = |include_mmproj: bool| -> Option<SwiftCalibratedPlan> {
            let eval_pages = |pages: usize| {
                eval_calibrated_cow_at(
                    available_bytes,
                    profile,
                    options,
                    pages * CONTEXT_PAGE_TOKENS,
                    reuse_slots,
                    min_checkpoint_slots,
                    explicit_checkpoint_bytes,
                    include_mmproj,
                )
            };
            let mut best = eval_pages(min_pages)?;
            let mut lo = min_pages + 1;
            let mut hi = max_pages;
            while lo <= hi {
                let mid = lo + (hi - lo) / 2;
                if let Some(plan) = eval_pages(mid) {
                    best = plan;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            Some(best)
        };

        let try_vision = options.request_vision && options.speculation_tokens.is_none();
        if try_vision && let Some(plan) = search(true) {
            return Ok(plan);
        }
        search(false).ok_or_else(insufficient)
    }

    /// Derive a calibrated CoW context plan while charging `extra_adapter_bytes` for
    /// an active stackable multi-rank adapter on top of base weights, vision, and caches.
    ///
    /// # Errors
    /// Returns an error if `available_bytes` cannot cover `extra_adapter_bytes` plus the
    /// minimum calibrated execution profile, or on arithmetic overflow.
    pub fn for_calibrated_execution_with_adapter_reserve(
        available_bytes: u64,
        profile: SwiftContextProfile,
        options: SwiftCalibratedOptions,
        extra_adapter_bytes: u64,
    ) -> Result<SwiftCalibratedPlan, String> {
        if extra_adapter_bytes == 0 {
            return Self::for_calibrated_execution(available_bytes, profile, options);
        }
        let remaining = available_bytes
            .checked_sub(extra_adapter_bytes)
            .ok_or_else(insufficient)?;
        let mut plan = Self::for_calibrated_execution(remaining, profile, options)?;
        let extra_usize = usize::try_from(extra_adapter_bytes).map_err(|_| insufficient())?;
        plan.context_plan.memory_limit_bytes = plan
            .context_plan
            .memory_limit_bytes
            .checked_add(extra_usize)
            .ok_or_else(insufficient)?;
        plan.context_plan.required_available_bytes = plan
            .context_plan
            .required_available_bytes
            .checked_add(extra_adapter_bytes)
            .ok_or_else(insufficient)?;
        plan.working_set_limit_bytes = plan
            .working_set_limit_bytes
            .checked_add(extra_usize)
            .ok_or_else(insufficient)?;
        Ok(plan)
    }
}

fn admit_context_tokens(
    available_bytes: u64,
    profile: SwiftContextProfile,
    fixed_bytes: u64,
    per_token: u64,
    max_tokens: usize,
) -> Result<usize, String> {
    if per_token == 0 {
        return Err(insufficient());
    }
    let allocation = available_bytes
        .checked_sub(HOST_RESERVE_BYTES)
        .and_then(|bytes| bytes.checked_sub(STARTUP_RESERVE_BYTES))
        .and_then(|bytes| bytes.checked_sub(fixed_bytes))
        .ok_or_else(insufficient)?;
    let state_allocation = profile
        .state_limit_bytes()
        .checked_sub(RECURRENT_BYTES)
        .ok_or_else(insufficient)?;
    let tokens = (allocation / per_token).min(state_allocation / profile.state_bytes_per_token());
    let tokens = usize::try_from(tokens)
        .map_err(|_| insufficient())?
        .min(max_tokens)
        .min(MODEL_CONTEXT_TOKENS);
    let context_tokens = tokens / CONTEXT_PAGE_TOKENS * CONTEXT_PAGE_TOKENS;
    if context_tokens < MIN_CONTEXT_TOKENS {
        return Err(insufficient());
    }
    Ok(context_tokens)
}

fn with_optional_vision<T>(
    available_bytes: u64,
    request_vision: bool,
    plan_for: impl Fn(u64) -> Result<(SwiftContextPlan, T), String>,
    add_mmproj_to_extra: impl FnOnce(&mut T) -> Result<(), String>,
) -> Result<(SwiftContextPlan, bool, T), String> {
    if request_vision
        && let Some(remaining) = available_bytes.checked_sub(crate::MMPROJ_RESERVE_BYTES)
        && let Ok((mut plan, mut extra)) = plan_for(remaining)
    {
        apply_mmproj_reserve_to_plan(&mut plan)?;
        add_mmproj_to_extra(&mut extra)?;
        return Ok((plan, true, extra));
    }
    let (plan, extra) = plan_for(available_bytes)?;
    Ok((plan, false, extra))
}

fn apply_mmproj_reserve_to_plan(plan: &mut SwiftContextPlan) -> Result<(), String> {
    let reserve_usize = usize::try_from(crate::MMPROJ_RESERVE_BYTES).map_err(|_| insufficient())?;
    plan.memory_limit_bytes = plan
        .memory_limit_bytes
        .checked_add(reserve_usize)
        .ok_or_else(insufficient)?;
    plan.required_available_bytes = plan
        .required_available_bytes
        .checked_add(crate::MMPROJ_RESERVE_BYTES)
        .ok_or_else(insufficient)?;
    Ok(())
}

fn insufficient() -> String {
    "Available memory cannot admit Swift's context and reserved runtime resources.".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn sizing_preserves_reserves_and_rejects_less_than_one_minimum_context()
    -> Result<(), Box<dyn Error>> {
        let minimum = HOST_RESERVE_BYTES
            + STARTUP_RESERVE_BYTES
            + FIXED_RUNTIME_BYTES
            + u64::try_from(MIN_CONTEXT_TOKENS)? * CONTEXT_BYTES_PER_TOKEN;
        assert!(SwiftContextPlan::for_available_memory(minimum - 1).is_err());
        let plan = SwiftContextPlan::for_available_memory(minimum)?;
        assert_eq!(plan.context_tokens, MIN_CONTEXT_TOKENS);
        assert_eq!(
            plan.required_available_bytes + STARTUP_RESERVE_BYTES,
            minimum
        );
        assert_eq!(
            plan.required_available_bytes - u64::try_from(plan.memory_limit_bytes)?,
            HOST_RESERVE_BYTES
        );
        for unavailable in [0, HOST_RESERVE_BYTES, FIXED_RUNTIME_BYTES] {
            assert!(SwiftContextPlan::for_available_memory(unavailable).is_err());
        }
        Ok(())
    }

    #[test]
    fn more_memory_selects_more_context_until_model_or_runtime_state_cap()
    -> Result<(), Box<dyn Error>> {
        let gib = 1024 * 1024 * 1024;
        let small = SwiftContextPlan::for_available_memory(16 * gib)?;
        let large = SwiftContextPlan::for_available_memory(18 * gib)?;
        let unlimited = SwiftContextPlan::for_available_memory(u64::MAX)?;
        assert!(small.context_tokens < 8_192);
        assert!(large.context_tokens > small.context_tokens);
        assert_eq!(unlimited.context_tokens, 62_720);
        assert!(unlimited.context_tokens <= MODEL_CONTEXT_TOKENS);
        let state_bytes =
            u64::try_from(unlimited.context_tokens)? * STATE_BYTES_PER_TOKEN + RECURRENT_BYTES;
        assert!(state_bytes <= STATE_LIMIT_BYTES);
        for (available, plan) in [(16 * gib, small), (18 * gib, large)] {
            assert!(plan.required_available_bytes <= available);
            assert!(available - plan.required_available_bytes >= STARTUP_RESERVE_BYTES);
            assert!(
                available - plan.required_available_bytes - STARTUP_RESERVE_BYTES
                    < 256 * CONTEXT_BYTES_PER_TOKEN
            );
        }
        Ok(())
    }

    #[test]
    fn native_cache_profiles_select_capacity_without_spending_host_or_startup_reserves()
    -> Result<(), Box<dyn Error>> {
        use SwiftContextProfile::{NativeF16, NativeRsq4, PublisherF16};
        let gib = 1024 * 1024 * 1024;
        assert_eq!(PublisherF16.context_bytes_per_token(), 137_344);
        assert_eq!(NativeF16.context_bytes_per_token(), 131_296);
        assert_eq!(NativeRsq4.context_bytes_per_token(), 34_016);
        assert_eq!(NativeRsq4.state_bytes_per_token(), 17_120);
        let publisher = SwiftContextPlan::for_profile(20 * gib, PublisherF16)?;
        assert_eq!(publisher.context_tokens, 37_120);
        assert_eq!(publisher, SwiftContextPlan::for_available_memory(20 * gib)?);
        for (available, expected) in [(18 * gib, 86_784), (20 * gib, 149_760), (24 * gib, 241_664)]
        {
            let plan = SwiftContextPlan::for_profile(available, NativeRsq4)?;
            assert_eq!(plan.context_tokens, expected);
            assert!(plan.required_available_bytes + STARTUP_RESERVE_BYTES <= available);
            assert_eq!(
                u64::try_from(plan.memory_limit_bytes)? + HOST_RESERVE_BYTES,
                plan.required_available_bytes
            );
            let tokens = u64::try_from(plan.context_tokens)?;
            assert!(
                tokens * NativeRsq4.state_bytes_per_token() + RECURRENT_BYTES <= STATE_LIMIT_BYTES
            );
        }
        for (profile, ceiling) in [(NativeF16, 62_720), (NativeRsq4, 241_664)] {
            let plan = SwiftContextPlan::for_profile(u64::MAX, profile)?;
            assert_eq!(plan.context_tokens, ceiling);
            let next_page = u64::try_from(plan.context_tokens + CONTEXT_PAGE_TOKENS)?;
            assert!(
                next_page * profile.state_bytes_per_token() + RECURRENT_BYTES > STATE_LIMIT_BYTES
            );
        }
        Ok(())
    }

    #[test]
    fn explicit_k4v3_capacity_preserves_two_caches_reserves_and_the_model_ceiling()
    -> Result<(), Box<dyn Error>> {
        let profile = SwiftContextProfile::NativeRsq4K3v;
        assert_eq!(profile.kv_bytes_per_token(), 14_848);
        assert_eq!(profile.context_bytes_per_token(), 29_920);
        assert_eq!(profile.state_bytes_per_token(), 15_072);
        let gib = 1024 * 1024 * 1024;
        for (available, expected) in [(18 * gib, 98_560), (20 * gib, 170_240), (24 * gib, 262_144)]
        {
            let plan = SwiftContextPlan::for_profile(available, profile)?;
            assert_eq!(plan.context_tokens, expected);
            assert!(plan.required_available_bytes + STARTUP_RESERVE_BYTES <= available);
            assert_eq!(
                u64::try_from(plan.memory_limit_bytes)? + HOST_RESERVE_BYTES,
                plan.required_available_bytes
            );
            assert!(
                u64::try_from(plan.context_tokens)? * profile.state_bytes_per_token()
                    + RECURRENT_BYTES
                    <= STATE_LIMIT_BYTES
            );
            if plan.context_tokens < MODEL_CONTEXT_TOKENS {
                let next_page = u64::try_from(plan.context_tokens + CONTEXT_PAGE_TOKENS)?;
                assert!(
                    FIXED_RUNTIME_BYTES
                        + next_page * profile.context_bytes_per_token()
                        + HOST_RESERVE_BYTES
                        + STARTUP_RESERVE_BYTES
                        > available
                );
            }
        }
        assert_eq!(
            SwiftContextPlan::for_profile(u64::MAX, profile)?.context_tokens,
            MODEL_CONTEXT_TOKENS
        );
        Ok(())
    }

    #[test]
    fn full_k4v4_candidate_reaches_model_limit_without_reducing_process_or_host_charges()
    -> Result<(), Box<dyn Error>> {
        use SwiftContextProfile::{NativeRsq4, NativeRsq4Full};
        assert_eq!(RSQ4_FULL_STATE_LIMIT_BYTES, 4_644_798_464);
        assert_eq!(NativeRsq4.state_limit_bytes(), STATE_LIMIT_BYTES);
        assert_eq!(
            NativeRsq4Full.state_limit_bytes(),
            RSQ4_FULL_STATE_LIMIT_BYTES
        );
        assert_eq!(
            NativeRsq4Full.context_bytes_per_token(),
            NativeRsq4.context_bytes_per_token()
        );
        assert_eq!(
            NativeRsq4Full.state_bytes_per_token(),
            NativeRsq4.state_bytes_per_token()
        );
        let full_budget = HOST_RESERVE_BYTES
            + STARTUP_RESERVE_BYTES
            + FIXED_RUNTIME_BYTES
            + u64::try_from(MODEL_CONTEXT_TOKENS)? * NativeRsq4Full.context_bytes_per_token();
        let below = SwiftContextPlan::for_profile(full_budget - 1, NativeRsq4Full)?;
        assert_eq!(
            below.context_tokens,
            MODEL_CONTEXT_TOKENS - CONTEXT_PAGE_TOKENS
        );
        for available in [full_budget, 24 * 1024 * 1024 * 1024, u64::MAX] {
            let full = SwiftContextPlan::for_profile(available, NativeRsq4Full)?;
            assert_eq!(full.context_tokens, MODEL_CONTEXT_TOKENS);
            assert_eq!(
                full.required_available_bytes + STARTUP_RESERVE_BYTES,
                full_budget
            );
            assert_eq!(
                SwiftContextPlan::for_profile(available, NativeRsq4)?.context_tokens,
                241_664
            );
        }
        let next_page = u64::try_from(MODEL_CONTEXT_TOKENS + CONTEXT_PAGE_TOKENS)?;
        assert!(
            next_page * NativeRsq4Full.state_bytes_per_token() + RECURRENT_BYTES
                > NativeRsq4Full.state_limit_bytes()
        );
        Ok(())
    }

    #[test]
    fn vision_context_planning_reserves_mmproj_or_falls_back_cleanly() -> Result<(), Box<dyn Error>>
    {
        let profile = SwiftContextProfile::NativeRsq4;
        let minimum = HOST_RESERVE_BYTES
            + STARTUP_RESERVE_BYTES
            + FIXED_RUNTIME_BYTES
            + u64::try_from(MIN_CONTEXT_TOKENS)? * profile.context_bytes_per_token();
        let (fallback, vision_on_tight_ram) =
            SwiftContextPlan::for_profile_with_vision(minimum, profile, true)?;
        assert!(!vision_on_tight_ram);
        assert_eq!(fallback, SwiftContextPlan::for_profile(minimum, profile)?);

        let with_vision_budget = minimum + crate::MMPROJ_RESERVE_BYTES;
        let (vision_plan, vision_admitted) =
            SwiftContextPlan::for_profile_with_vision(with_vision_budget, profile, true)?;
        assert!(vision_admitted);
        assert_eq!(vision_plan.context_tokens, MIN_CONTEXT_TOKENS);
        assert_eq!(
            vision_plan.required_available_bytes - fallback.required_available_bytes,
            crate::MMPROJ_RESERVE_BYTES
        );
        assert_eq!(
            u64::try_from(vision_plan.memory_limit_bytes - fallback.memory_limit_bytes)?,
            crate::MMPROJ_RESERVE_BYTES
        );

        let shape = SwiftExecutionShape::for_profile(profile);
        let shape_min_footprint = shape
            .breakdown(MIN_CONTEXT_TOKENS)
            .and_then(|breakdown| breakdown.total_footprint_bytes())
            .ok_or("missing minimum shape footprint")?;
        let shape_base_budget =
            minimum.max(HOST_RESERVE_BYTES + STARTUP_RESERVE_BYTES + shape_min_footprint);
        let (shape_fallback, _) =
            SwiftContextPlan::for_execution_shape(shape_base_budget, profile, &shape)?;
        let (shape_plan, breakdown, shape_vision) =
            SwiftContextPlan::for_execution_shape_with_vision(
                shape_base_budget + crate::MMPROJ_RESERVE_BYTES,
                profile,
                &shape,
                true,
            )?;
        assert!(shape_vision);
        assert_eq!(
            shape_plan.required_available_bytes - shape_fallback.required_available_bytes,
            crate::MMPROJ_RESERVE_BYTES
        );
        assert_eq!(
            breakdown
                .total_footprint_bytes()
                .map(|b| b + HOST_RESERVE_BYTES),
            Some(shape_plan.required_available_bytes)
        );
        assert_eq!(
            breakdown.job_commit_bytes(),
            Some(u64::try_from(shape_plan.memory_limit_bytes)?)
        );
        Ok(())
    }

    #[test]
    fn every_cache_profile_rejects_one_byte_below_its_minimum_launch_budget()
    -> Result<(), Box<dyn Error>> {
        for profile in [
            SwiftContextProfile::PublisherF16,
            SwiftContextProfile::NativeF16,
            SwiftContextProfile::NativeRsq4,
            SwiftContextProfile::NativeRsq4Full,
            SwiftContextProfile::NativeRsq4K3v,
        ] {
            let minimum = HOST_RESERVE_BYTES
                + STARTUP_RESERVE_BYTES
                + FIXED_RUNTIME_BYTES
                + u64::try_from(MIN_CONTEXT_TOKENS)? * profile.context_bytes_per_token();
            assert!(SwiftContextPlan::for_profile(minimum - 1, profile).is_err());
            let plan = SwiftContextPlan::for_profile(minimum, profile)?;
            assert_eq!(plan.context_tokens, MIN_CONTEXT_TOKENS);
            assert_eq!(
                plan.required_available_bytes + STARTUP_RESERVE_BYTES,
                minimum
            );
            assert!(SwiftContextPlan::for_profile(0, profile).is_err());
        }
        Ok(())
    }

    #[test]
    fn itemized_execution_shape_accounts_all_categories_and_preserves_conservative_curve()
    -> Result<(), Box<dyn Error>> {
        assert_eq!(
            SwiftKvCacheFormat::Rotated4FittedValues.kv_bytes_per_token(),
            16_896
        );
        assert_eq!(
            SwiftKvCacheFormat::Rotated4FittedKv.kv_bytes_per_token(),
            16_896
        );
        let gib = 1024 * 1024 * 1024;
        let baseline = SwiftContextPlan::for_profile(20 * gib, SwiftContextProfile::NativeRsq4)?;
        let mut shape = SwiftExecutionShape::for_profile(SwiftContextProfile::NativeRsq4);
        shape.kv_format = SwiftKvCacheFormat::Rotated4FittedValues;
        shape.decode_scratch_bytes = 8 * 1024 * 1024;
        shape.tile_prefill_scratch_bytes = 32 * 1024 * 1024;
        shape.speculative_verify_scratch_bytes = 24 * 1024 * 1024;
        shape.multi_row_logits_bytes = 16 * 1024 * 1024;
        shape.replay_scratch_bytes = 20 * 1024 * 1024;
        shape.accelerator_resident_arena_bytes = 512 * 1024 * 1024;
        shape.accelerator_staging_bytes = 64 * 1024 * 1024;
        shape.allocator_fragmentation_margin_bytes = 48 * 1024 * 1024;

        let (plan, breakdown) = SwiftContextPlan::for_execution_shape(
            20 * gib,
            SwiftContextProfile::NativeRsq4,
            &shape,
        )?;
        assert!(plan.context_tokens <= baseline.context_tokens);
        assert_eq!(
            breakdown.weights_and_fixed_runtime_bytes,
            FIXED_RUNTIME_BYTES
        );
        assert_eq!(breakdown.recurrent_checkpoint_bytes, 3 * RECURRENT_BYTES);
        assert_eq!(
            breakdown.execution_scratch_bytes,
            (24 + 16 + 20) * 1024 * 1024
        );
        assert_eq!(
            breakdown.accelerator_bytes(),
            Some((512 + 64) * 1024 * 1024)
        );
        assert_eq!(
            breakdown.allocator_fragmentation_margin_bytes,
            48 * 1024 * 1024
        );
        assert_eq!(
            u64::try_from(plan.memory_limit_bytes)?,
            breakdown.job_commit_bytes().ok_or("overflow")?
        );
        assert_eq!(
            plan.required_available_bytes,
            breakdown
                .job_commit_bytes()
                .and_then(|job| job.checked_add(breakdown.accelerator_bytes()?))
                .and_then(|total| total.checked_add(HOST_RESERVE_BYTES))
                .ok_or("overflow")?
        );
        assert!(plan.required_available_bytes + STARTUP_RESERVE_BYTES <= 20 * gib);

        // Smaller speculative branch falls back to tile prefill scratch when larger.
        shape.replay_scratch_bytes = 0;
        shape.speculative_verify_scratch_bytes = 4 * 1024 * 1024;
        shape.multi_row_logits_bytes = 4 * 1024 * 1024;
        assert_eq!(shape.execution_scratch_bytes(), Some(32 * 1024 * 1024));

        // Larger scratch or accelerator reservation never increases admitted context.
        let mut larger = shape;
        larger.accelerator_resident_arena_bytes += gib;
        let (tighter, _) = SwiftContextPlan::for_execution_shape(
            20 * gib,
            SwiftContextProfile::NativeRsq4,
            &larger,
        )?;
        assert!(tighter.context_tokens <= plan.context_tokens);
        Ok(())
    }

    #[test]
    fn tiled_prefill_scratch_and_cow_kv_accounting_preserve_exact_contracts()
    -> Result<(), Box<dyn Error>> {
        let gib = 1024 * 1024 * 1024_u64;
        assert_eq!(
            swift_tiled_prefill_scratch_bytes(8, 8_192, SwiftKvCacheFormat::Rsq4),
            Some(11_501_568)
        );
        assert_eq!(
            swift_tiled_prefill_scratch_bytes(8, 262_144, SwiftKvCacheFormat::Rsq4),
            Some(19_628_032)
        );
        assert_eq!(
            swift_tiled_prefill_scratch_bytes(16, 62_720, SwiftKvCacheFormat::F16),
            Some(46_481_408)
        );
        for invalid in [0, 1, 4, 7, 15, 64] {
            assert_eq!(
                swift_tiled_prefill_scratch_bytes(invalid, 8_192, SwiftKvCacheFormat::F16),
                None
            );
        }
        assert_eq!(
            swift_tiled_prefill_scratch_bytes(8, 0, SwiftKvCacheFormat::F16),
            None
        );

        for profile in [
            SwiftContextProfile::NativeF16,
            SwiftContextProfile::NativeRsq4,
            SwiftContextProfile::NativeRsq4Full,
            SwiftContextProfile::NativeRsq4K3v,
        ] {
            for tile in [8_usize, 16, 32] {
                let (plan, vision, Some(scratch)) = SwiftContextPlan::for_profile_with_options(
                    24 * gib,
                    profile,
                    true,
                    Some(tile),
                )?
                else {
                    return Err("expected prefill scratch bytes".into());
                };
                assert!(vision);
                let expected = FIXED_RUNTIME_BYTES
                    + crate::MMPROJ_RESERVE_BYTES
                    + u64::try_from(plan.context_tokens)? * profile.context_bytes_per_token()
                    + u64::try_from(scratch)?;
                assert_eq!(u64::try_from(plan.memory_limit_bytes)?, expected);
                assert!(plan.required_available_bytes + STARTUP_RESERVE_BYTES <= 24 * gib);
            }
        }
        assert!(
            SwiftContextPlan::for_profile_with_prefill(
                24 * gib,
                SwiftContextProfile::PublisherF16,
                Some(8)
            )
            .is_err()
        );

        let shape = SwiftExecutionShape::for_profile(SwiftContextProfile::NativeRsq4Full)
            .with_tiled_prefill_tokens(8, 262_144)
            .ok_or("missing tiled prefill shape")?;
        assert_eq!(shape.tile_prefill_scratch_bytes, 19_628_032);
        let breakdown = shape.breakdown(262_144).ok_or("missing breakdown")?;
        let worst = breakdown.total_footprint_bytes().ok_or("overflow")?;
        let cow = breakdown.physical_cow_footprint_bytes().ok_or("overflow")?;
        assert!(cow < worst);
        // Sharing 262_144 - 64 immutable KV tokens saves 262_080 * 16_896 bytes (~4.12 GiB).
        assert_eq!(worst - cow, 262_080 * RSQ4_KV_BYTES_PER_TOKEN);
        Ok(())
    }

    #[test]
    fn multi_rank_adapter_budget_accounts_for_shape_and_calibrated_cow_plan()
    -> Result<(), Box<dyn Error>> {
        let budget =
            MultiRankAdapterBudget::for_npu_friendly_f16(16).ok_or("valid rank 16 budget")?;
        assert_eq!(budget.payload_bytes, 16 * 4_194_304);
        assert_eq!(
            budget.scratch_headroom_bytes,
            crate::DAISYROSE_SCRATCH_HEADROOM_BYTES
        );
        assert!(MultiRankAdapterBudget::for_npu_friendly_f16(0).is_none());
        assert!(MultiRankAdapterBudget::for_npu_friendly_f16(65).is_none());
        assert!(MultiRankAdapterBudget::from_verified_bytes(0).is_none());

        let base_shape = SwiftExecutionShape::for_profile(SwiftContextProfile::NativeRsq4);
        let stacked_shape = base_shape
            .with_multi_rank_adapter(budget)
            .ok_or("stacked shape")?;
        assert_eq!(
            stacked_shape.weights_and_fixed_runtime_bytes
                - base_shape.weights_and_fixed_runtime_bytes,
            budget.total_bytes().ok_or("budget total")?
        );

        let gib = 1024 * 1024 * 1024_u64;
        let opts = SwiftCalibratedOptions {
            weight_packing: SwiftWeightPacking::Pq2_0,
            request_vision: false,
            prefill_tokens: Some(8),
            chunkwise_gdn_wy_f32: false,
            speculation_tokens: None,
            prompt_reuse: SwiftCalibratedReuse::Disabled,
        };
        let extra = MultiRankAdapterBudget::from_verified_bytes(40 * 1024 * 1024)
            .and_then(MultiRankAdapterBudget::total_bytes)
            .ok_or("extra total")?;
        let base_plan = SwiftContextPlan::for_calibrated_execution(
            24 * gib,
            SwiftContextProfile::NativeRsq4,
            opts,
        )?;
        let stacked_plan = SwiftContextPlan::for_calibrated_execution_with_adapter_reserve(
            24 * gib,
            SwiftContextProfile::NativeRsq4,
            opts,
            extra,
        )?;
        assert!(stacked_plan.context_plan.context_tokens <= base_plan.context_plan.context_tokens);
        assert!(
            stacked_plan.context_plan.required_available_bytes + STARTUP_RESERVE_BYTES <= 24 * gib
        );
        Ok(())
    }

    #[test]
    fn npu_friendly_rank_budget_supports_f16_and_f32_bounds() -> Result<(), Box<dyn Error>> {
        for rank in [1_u16, 16, 64] {
            let f16_budget =
                MultiRankAdapterBudget::for_npu_friendly_rank(rank, false).ok_or("f16 rank")?;
            assert_eq!(f16_budget.payload_bytes, u64::from(rank) * 4_194_304);
            assert_eq!(
                f16_budget.scratch_headroom_bytes,
                crate::DAISYROSE_SCRATCH_HEADROOM_BYTES
            );
            assert_eq!(
                Some(f16_budget),
                MultiRankAdapterBudget::for_npu_friendly_f16(rank)
            );
            assert_eq!(
                f16_budget.total_bytes(),
                Some(u64::from(rank) * 4_194_304 + crate::DAISYROSE_SCRATCH_HEADROOM_BYTES)
            );

            let f32_budget =
                MultiRankAdapterBudget::for_npu_friendly_rank(rank, true).ok_or("f32 rank")?;
            assert_eq!(f32_budget.payload_bytes, u64::from(rank) * 8_388_608);
            assert_eq!(
                f32_budget.scratch_headroom_bytes,
                crate::DAISYROSE_SCRATCH_HEADROOM_BYTES
            );
            assert_eq!(
                f32_budget.total_bytes(),
                Some(u64::from(rank) * 8_388_608 + crate::DAISYROSE_SCRATCH_HEADROOM_BYTES)
            );
        }

        for invalid_rank in [0_u16, 65, u16::MAX] {
            assert!(MultiRankAdapterBudget::for_npu_friendly_rank(invalid_rank, false).is_none());
            assert!(MultiRankAdapterBudget::for_npu_friendly_rank(invalid_rank, true).is_none());
        }
        Ok(())
    }

    #[test]
    fn nariq_weight_packing_calibration_accounts_for_artifact_sizes_and_headroom()
    -> Result<(), Box<dyn Error>> {
        for (variant, expected_packing) in [
            (crate::NariqVariant::Bpw2_5, SwiftWeightPacking::NariQ2_5Bpw),
            (crate::NariqVariant::Bpw3_0, SwiftWeightPacking::NariQ3_0Bpw),
            (crate::NariqVariant::Bpw2_0, SwiftWeightPacking::NariQ2_0Bpw),
        ] {
            let packing = SwiftWeightPacking::from_nariq(variant);
            assert_eq!(packing, expected_packing);
            assert_eq!(packing.weight_bytes(), variant.weight_bytes());
            let expected_fixed = (FIXED_RUNTIME_BYTES - crate::SWIFT_WEIGHT_BYTES)
                + variant.weight_bytes()
                + crate::NARIQ_TRELLIS_SCRATCH_HEADROOM_BYTES;
            assert_eq!(packing.calibrated_fixed_runtime_bytes(), expected_fixed);
            let shape = SwiftExecutionShape::for_profile(SwiftContextProfile::NativeRsq4Full)
                .with_weight_packing_calibrated(packing);
            assert_eq!(shape.weights_and_fixed_runtime_bytes, expected_fixed);
        }
        Ok(())
    }

    #[test]
    fn bonsai25_weight_packing_calibration_tracks_exact_artifact_sizes_without_trellis_headroom()
    -> Result<(), Box<dyn Error>> {
        for (variant, expected_packing) in [
            (crate::Bonsai25Variant::Pq2, SwiftWeightPacking::Bonsai25Pq2),
            (
                crate::Bonsai25Variant::Ptq1,
                SwiftWeightPacking::Bonsai25Ptq1,
            ),
        ] {
            let packing = SwiftWeightPacking::from_bonsai25(variant);
            assert_eq!(packing, expected_packing);
            assert_eq!(packing.weight_bytes(), variant.weight_bytes());
            let expected_fixed =
                (FIXED_RUNTIME_BYTES - crate::SWIFT_WEIGHT_BYTES) + variant.weight_bytes();
            assert_eq!(packing.calibrated_fixed_runtime_bytes(), expected_fixed);
            let shape = SwiftExecutionShape::for_profile(SwiftContextProfile::NativeRsq4Full)
                .with_weight_packing_calibrated(packing);
            assert_eq!(shape.weights_and_fixed_runtime_bytes, expected_fixed);
        }
        Ok(())
    }
}
