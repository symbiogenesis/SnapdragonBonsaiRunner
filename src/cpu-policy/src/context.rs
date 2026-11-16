//! Pure Swift context sizing from a prelaunch available-physical-memory snapshot.
//!
//! This is conservative admission arithmetic, not measured peak memory. Callers
//! recheck the resulting requirement immediately before starting the owned Job.

/// Swift's verified training context; available memory may select less.
pub const MODEL_CONTEXT_TOKENS: usize = 262_144;
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
        | SwiftKvCacheFormat::Rotated4FittedValues => 1_u64,
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

/// Explicit worst-case runtime, checkpoint, scratch, and accelerator shape for
/// Swift admission accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwiftExecutionShape {
    /// Attention KV format charged per token per live KV copy.
    pub kv_format: SwiftKvCacheFormat,
    /// Model weights and fixed runtime allowance charged inside the Job commit limit.
    pub weights_and_fixed_runtime_bytes: u64,
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
    /// Complete host private memory subject to the Windows Job commit ceiling
    /// (`weights + fixed runtime + recurrent copies + live KV + scratch + fragmentation`),
    /// excluding external accelerator resident arena and staging allocations.
    #[must_use]
    pub fn job_commit_bytes(&self) -> Option<u64> {
        self.weights_and_fixed_runtime_bytes
            .checked_add(self.recurrent_checkpoint_bytes)?
            .checked_add(self.live_kv_bytes)?
            .checked_add(self.execution_scratch_bytes)?
            .checked_add(self.allocator_fragmentation_margin_bytes)
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

    /// Combined physical RAM ledger across Windows Job commit and unified
    /// accelerator reservations, excluding [`HOST_RESERVE_BYTES`].
    #[must_use]
    pub fn total_footprint_bytes(&self) -> Option<u64> {
        self.job_commit_bytes()?
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
        let allocation = available_bytes
            .checked_sub(HOST_RESERVE_BYTES)
            .and_then(|bytes| bytes.checked_sub(STARTUP_RESERVE_BYTES))
            .and_then(|bytes| bytes.checked_sub(FIXED_RUNTIME_BYTES))
            .ok_or_else(insufficient)?;
        let state_allocation = profile
            .state_limit_bytes()
            .checked_sub(RECURRENT_BYTES)
            .ok_or_else(insufficient)?;
        let process_bytes = profile.context_bytes_per_token();
        let state_bytes = profile.state_bytes_per_token();
        let tokens = (allocation / process_bytes).min(state_allocation / state_bytes);
        let tokens = usize::try_from(tokens)
            .map_err(|_| insufficient())?
            .min(MODEL_CONTEXT_TOKENS);
        let context_tokens = tokens / CONTEXT_PAGE_TOKENS * CONTEXT_PAGE_TOKENS;
        if context_tokens < MIN_CONTEXT_TOKENS {
            return Err(insufficient());
        }
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
        let per_token = shape
            .kv_bytes_per_token_total()
            .filter(|bytes| *bytes > 0)
            .ok_or_else(insufficient)?;
        let allocation = available_bytes
            .checked_sub(HOST_RESERVE_BYTES)
            .and_then(|bytes| bytes.checked_sub(STARTUP_RESERVE_BYTES))
            .and_then(|bytes| bytes.checked_sub(fixed_physical))
            .ok_or_else(insufficient)?;
        let state_allocation = profile
            .state_limit_bytes()
            .checked_sub(RECURRENT_BYTES)
            .ok_or_else(insufficient)?;
        let tokens =
            (allocation / per_token).min(state_allocation / profile.state_bytes_per_token());
        let tokens = usize::try_from(tokens)
            .map_err(|_| insufficient())?
            .min(baseline.context_tokens)
            .min(MODEL_CONTEXT_TOKENS);
        let context_tokens = tokens / CONTEXT_PAGE_TOKENS * CONTEXT_PAGE_TOKENS;
        if context_tokens < MIN_CONTEXT_TOKENS {
            return Err(insufficient());
        }
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
        if !request_vision {
            return Ok((Self::for_profile(available_bytes, profile)?, false));
        }
        if let Some(remaining) = available_bytes.checked_sub(crate::MMPROJ_RESERVE_BYTES)
            && let Ok(mut plan) = Self::for_profile(remaining, profile)
        {
            apply_mmproj_reserve_to_plan(&mut plan)?;
            return Ok((plan, true));
        }
        Ok((Self::for_profile(available_bytes, profile)?, false))
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
        let allocation = available_bytes
            .checked_sub(HOST_RESERVE_BYTES)
            .and_then(|bytes| bytes.checked_sub(STARTUP_RESERVE_BYTES))
            .and_then(|bytes| bytes.checked_sub(FIXED_RUNTIME_BYTES))
            .and_then(|bytes| bytes.checked_sub(fixed_tile))
            .ok_or_else(insufficient)?;
        let state_allocation = profile
            .state_limit_bytes()
            .checked_sub(RECURRENT_BYTES)
            .ok_or_else(insufficient)?;
        let total_per_token = profile
            .context_bytes_per_token()
            .checked_add(score_per_ctx_token)
            .ok_or_else(insufficient)?;
        let tokens =
            (allocation / total_per_token).min(state_allocation / profile.state_bytes_per_token());
        let tokens = usize::try_from(tokens)
            .map_err(|_| insufficient())?
            .min(MODEL_CONTEXT_TOKENS);
        let context_tokens = tokens / CONTEXT_PAGE_TOKENS * CONTEXT_PAGE_TOKENS;
        if context_tokens < MIN_CONTEXT_TOKENS {
            return Err(insufficient());
        }
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
        if !request_vision {
            let (plan, scratch) =
                Self::for_profile_with_prefill(available_bytes, profile, prefill_tokens)?;
            return Ok((plan, false, scratch));
        }
        if let Some(remaining) = available_bytes.checked_sub(crate::MMPROJ_RESERVE_BYTES)
            && let Ok((mut plan, scratch)) =
                Self::for_profile_with_prefill(remaining, profile, prefill_tokens)
        {
            apply_mmproj_reserve_to_plan(&mut plan)?;
            return Ok((plan, true, scratch));
        }
        let (plan, scratch) =
            Self::for_profile_with_prefill(available_bytes, profile, prefill_tokens)?;
        Ok((plan, false, scratch))
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
        if !request_vision {
            let (plan, breakdown) = Self::for_execution_shape(available_bytes, profile, shape)?;
            return Ok((plan, breakdown, false));
        }
        if let Some(remaining) = available_bytes.checked_sub(crate::MMPROJ_RESERVE_BYTES)
            && let Ok((mut plan, mut breakdown)) =
                Self::for_execution_shape(remaining, profile, shape)
        {
            apply_mmproj_reserve_to_plan(&mut plan)?;
            breakdown.weights_and_fixed_runtime_bytes = breakdown
                .weights_and_fixed_runtime_bytes
                .checked_add(crate::MMPROJ_RESERVE_BYTES)
                .ok_or_else(insufficient)?;
            return Ok((plan, breakdown, true));
        }
        let (plan, breakdown) = Self::for_execution_shape(available_bytes, profile, shape)?;
        Ok((plan, breakdown, false))
    }
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
}
