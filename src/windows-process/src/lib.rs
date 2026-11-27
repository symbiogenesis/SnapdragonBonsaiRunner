//! Windows 10+ process ownership using an unnamed, noninheritable Job object.
//!
//! A suspended root must enter its Job before its first instruction; dropping its
//! owner kills the complete tree. This enforces lifecycle and committed-memory
//! limits. Explicit sandboxed launches additionally require native Windows PSEC
//! filesystem, network and desktop containment; no weaker fallback is used.
#![cfg(windows)]

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::io;
use std::os::windows::ffi::OsStrExt as _;
use std::os::windows::fs::MetadataExt as _;
use std::os::windows::io::AsSocket;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

// All Win32 calls and their pointer/handle contracts are confined here.
#[allow(
    unsafe_code,
    reason = "Keep audited Win32 calls in this native boundary."
)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "Explicit parent-only visibility keeps the audited boundary private."
)]
mod native;

#[cfg(feature = "quant-kernels")]
pub mod conv_kernels;
#[cfg(feature = "quant-kernels")]
pub mod quant_kernels;

/// Internal CLI support for `bonsai-push-recovery`.
#[forbid(unsafe_code)]
#[doc(hidden)]
pub mod push_recovery;

/// Read-only memory-mapped file view backed by an owned Win32 section object.
pub struct MappedFile(native::mmap::NativeMappedFile);

impl MappedFile {
    /// Map an ordinary non-empty file read-only into memory and prefetch its pages.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::from_file(&file)
    }

    /// Map an open read-only file handle into memory and prefetch its pages.
    pub fn from_file(file: &std::fs::File) -> io::Result<Self> {
        native::mmap::map_readonly_file(file).map(Self)
    }

    /// Map an open read-only file handle into memory without calling `PrefetchVirtualMemory`.
    ///
    /// Prefer this when pages were already brought into the OS file cache (for example,
    /// right after SHA-256 verification) or when sections such as vocabulary embeddings are
    /// accessed sparsely by row.
    pub fn from_file_without_prefetch(file: &std::fs::File) -> io::Result<Self> {
        native::mmap::map_readonly_file_with_prefetch(file, false).map(Self)
    }

    /// Borrow the mapped bytes as an immutable slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    /// Return the mapped byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.as_slice().len()
    }

    /// Return whether the mapped slice is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.as_slice().is_empty()
    }
}

impl AsRef<[u8]> for MappedFile {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

/// Persistent spin-then-park worker pool using the calling thread as worker 0.
pub struct RowWorkerPool(native::row_pool::NativeRowWorkerPool);

impl RowWorkerPool {
    /// Create a worker pool with `worker_count` total participants (including caller).
    pub fn new(worker_count: usize, name_prefix: &str) -> io::Result<Self> {
        native::row_pool::NativeRowWorkerPool::new(worker_count, name_prefix).map(Self)
    }

    /// Total worker count participating in work partitions.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.0.worker_count()
    }

    /// Partition `out` across up to `active_workers` along `alignment`-element boundaries.
    pub fn run_aligned<F>(
        &self,
        out: &mut [f32],
        active_workers: usize,
        alignment: usize,
        compute: &F,
    ) where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        self.0.run_aligned(out, active_workers, alignment, compute);
    }

    /// Partition `out` across up to `active_workers` along `alignment`-element boundaries
    /// for arbitrary `Send` element types.
    pub fn run_slice_aligned<T: Send, F>(
        &self,
        out: &mut [T],
        active_workers: usize,
        alignment: usize,
        compute: &F,
    ) where
        F: Fn(usize, &mut [T]) + Sync,
    {
        self.0
            .run_slice_aligned(out, active_workers, alignment, compute);
    }

    /// Fallible counterpart to [`Self::run_aligned`], joining every chunk and returning
    /// the smallest-`offset` error deterministically when any slice fails.
    ///
    /// # Errors
    /// Propagates the smallest-`offset` I/O error returned by `compute`.
    pub fn try_run_aligned<F>(
        &self,
        out: &mut [f32],
        active_workers: usize,
        alignment: usize,
        compute: &F,
    ) -> io::Result<()>
    where
        F: Fn(usize, &mut [f32]) -> io::Result<()> + Sync,
    {
        self.0
            .try_run_aligned(out, active_workers, alignment, compute)
    }

    /// Partition `out` across up to `active_workers` with single-element granularity.
    pub fn run<F>(&self, out: &mut [f32], active_workers: usize, compute: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        self.0.run_aligned(out, active_workers, 1, compute);
    }
}

/// Reuses the calling terminal for a graphical executable's command-line mode.
///
/// Redirected standard output is preserved. No console is ever allocated; an
/// absent parent console leaves ordinary redirected diagnostics unchanged.
pub fn attach_parent_console() {
    native::presentation::attach_parent_console();
}

/// Displays a fatal graphical startup failure using a native Windows dialog.
///
/// The fixed-size dialog offers a scrollable preview and an explicit button to
/// copy the complete error. The synchronous call retains no caller pointers and
/// creates no console. If Windows cannot create the dialog, a bounded native
/// message box reports that the detailed controls are unavailable.
pub fn show_startup_error(message: &str) {
    native::presentation::show_error(message);
}

/// Atomically renames an open file over an existing destination on its volume.
///
/// Retain the verified source handle with delete access and without write or
/// delete sharing throughout this call. The source is selected by that handle,
/// so replacement never reopens an attacker-replaceable source pathname.
/// The caller validates and retains the destination's directory identity.
///
/// # Errors
/// Rejects relative or invalid destination names and failed Windows replacement,
/// including missing delete access, sharing conflicts and cross-volume moves.
pub fn replace_open_file(file: &std::fs::File, target: &Path) -> io::Result<()> {
    native::replacement::replace(file, target)
}

/// Returns this process's creation FILETIME for retained supervision identity.
///
/// # Errors
/// Rejects a failed process-handle or process-time query.
pub fn current_process_creation_time() -> io::Result<u64> {
    native::maintenance::current_creation_time()
}

/// Returns the Windows creation FILETIME for an explicitly selected process.
///
/// # Errors
/// Rejects PID zero and failed process/query operations.
pub fn process_creation_file_time(id: u32) -> io::Result<u64> {
    native::process_creation_time(id)
}

/// Observes the direct caller PID and Windows creation FILETIME.
///
/// The parent's creation must precede this process, preventing PID reuse from
/// admitting a replacement caller. The supervisor must retain the returned
/// identity with `ParentProcess::open` before beginning owned work.
///
/// # Errors
/// Rejects inaccessible, exited or replaced callers and failed process snapshots.
pub fn parent_process_identity() -> io::Result<(u32, u64)> {
    native::caller::identity()
}

/// A verified Authenticode leaf signer from the Windows trust provider.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthenticodeSigner {
    /// Signer certificate common name; trust does not grant device admission.
    pub common_name: String,
    /// Lowercase SHA-1 certificate thumbprint, distinct from the file SHA-256.
    pub thumbprint: String,
}

/// Verifies Authenticode against cached Windows trust and returns its leaf signer.
///
/// The caller retains an immutable file guard. No UI or network retrieval occurs.
/// Run this potentially provider-blocking operation in a deadline-owned worker.
///
/// # Errors
/// Rejects failed trust/revocation checks, missing signers and malformed names.
pub fn verify_authenticode(file: &std::fs::File, path: &Path) -> io::Result<AuthenticodeSigner> {
    native::signature::verify(file, path)
}

/// Lists matching module paths from a bounded read-only snapshot of a live process.
///
/// Paths identify observed modules, not independently verified mapped bytes.
///
/// # Errors
/// Rejects invalid selections, failed snapshots and oversized inventories.
pub fn process_module_paths(id: u32, basename: &str) -> io::Result<Vec<PathBuf>> {
    native::signature::modules(id, basename)
}

/// One present `SetupAPI` device, with optional read-only driver registry metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInventoryEntry {
    /// Device description, unknown when the property cannot be read.
    pub name: Option<String>,
    /// Device manufacturer, unknown when the property cannot be read.
    pub manufacturer: Option<String>,
    /// Present device instance identifier.
    pub device_id: String,
    /// Driver registry version; this does not establish package identity.
    pub driver_version: Option<String>,
    /// Driver registry date; this does not establish package identity.
    pub driver_date: Option<String>,
    /// Driver registry INF basename, unknown when unavailable.
    pub inf_name: Option<String>,
}

/// Inventories up to sixteen present devices in one explicit `SetupAPI` class.
///
/// Supports `ComputeAccelerator`, `Display` and `Media`; never loads a driver or
/// verifies a deployment signature. Optional unavailable properties stay unknown.
///
/// # Errors
/// Rejects unsupported classes, failed native snapshots and oversized inventories.
pub fn device_inventory(class: &str) -> io::Result<Vec<DeviceInventoryEntry>> {
    native::devices::inventory(class)
}

/// Returns the current UTC wall clock in the canonical round-trip receipt format.
#[must_use]
pub fn utc_now() -> String {
    native::maintenance::utc_now()
}

/// Requires a dedicated directory on a recognized local Windows drive.
///
/// # Errors
/// Rejects drive roots, relative, UNC, device and unavailable/network drives.
pub fn require_local_data_drive(path: &Path) -> io::Result<()> {
    native::maintenance::local_drive(path)
}

/// Returns bytes available to this caller on the selected directory's filesystem.
///
/// # Errors
/// Rejects an invalid path or failed Windows free-space query.
pub fn available_disk_bytes(path: &Path) -> io::Result<u64> {
    native::maintenance::available_disk_bytes(path)
}

/// Returns the shared Windows performance counter and its ticks-per-second rate.
///
/// # Errors
/// Rejects failed or invalid native counter queries.
pub fn monotonic_ticks() -> io::Result<(u64, u64)> {
    native::maintenance::monotonic_ticks()
}

/// Checks a bounded process snapshot for any explicitly selected executable name.
///
/// # Errors
/// Rejects invalid selections, oversized inventories and failed native enumeration.
pub fn has_running_process(names: &[&str]) -> io::Result<bool> {
    native::maintenance::has_running_process(names)
}

/// Bounded failure context retained inside a Vulkan query's `io::Error`.
///
/// Downcast `io::Error::get_ref()` to this type to distinguish failed discovery
/// stages without parsing driver messages or inferring unsupported hardware.
#[cfg(feature = "vulkan")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VulkanQueryFailure {
    /// Fixed query-stage label; never a driver-provided string or file path.
    pub stage: &'static str,
    /// Raw `VkResult` when a Vulkan entry point returned one.
    pub vulkan_result: Option<i32>,
    /// Loader API version already observed before failure, if available.
    pub loader_api_version: Option<u32>,
    /// Requested application API version, if instance creation was attempted.
    pub application_api_version: Option<u32>,
    /// Optional current-process module observations around instance creation.
    pub instance_modules: Option<VulkanInstanceModules>,
    /// Independent count-only graphics query after failed instance creation.
    pub graphics_adapter_count: Option<WindowsGraphicsAdapterCount>,
}

/// One count-only `D3DKMTEnumAdapters2` observation in the current worker.
///
/// No adapter array or handles are requested. A failed call leaves its output
/// count undefined; it cannot prove that the computer has no graphics devices.
#[cfg(feature = "vulkan")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowsGraphicsAdapterCount {
    /// Fixed stage: existing module, exported entry point, or count query.
    pub stage: &'static str,
    /// Immediate Win32 lookup error, unknown when no lookup failed.
    /// A module lookup error does not establish that the module is absent.
    pub windows_error: Option<u32>,
    /// Raw `NTSTATUS` from the count query, unknown before the call.
    pub ntstatus: Option<i32>,
    /// Reported count only when the query returned `STATUS_SUCCESS`.
    pub adapter_count: Option<u32>,
}

#[cfg(feature = "vulkan")]
impl std::fmt::Display for VulkanQueryFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Vulkan inventory failed at {}", self.stage)?;
        if let Some(result) = self.vulkan_result {
            write!(formatter, " (VkResult {result})")?;
        }
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
impl std::error::Error for VulkanQueryFailure {}

/// Model-free observations from the native system Vulkan loader.
#[cfg(feature = "vulkan")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VulkanInventory {
    /// Loader-reported packed Vulkan API version, without vendor decoding.
    pub loader_api_version: u32,
    /// Declared application/instance API version, capped at Vulkan 1.3.
    pub application_api_version: u32,
    /// Current-process module observations immediately around instance creation.
    pub instance_modules: VulkanInstanceModules,
    /// Enumerated adapters, including explicitly identified software devices.
    pub adapters: Vec<VulkanAdapter>,
}

/// Bounded module-name snapshots around the single Vulkan instance-creation call.
///
/// Basename snapshots cannot identify attempted or unloaded DLLs, distinguish
/// files with equal basenames, or establish driver dependencies. Optional paths
/// are for a separate bounded file observation; they do not identify mapped bytes.
#[cfg(feature = "vulkan")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VulkanInstanceModules {
    /// Sorted unique basenames before instance creation, or unknown on failure.
    pub before: Option<Vec<String>>,
    /// Sorted unique basenames after instance creation returns, or unknown.
    pub after: Option<Vec<String>>,
    /// At most sixteen post-instance loader/new-module paths for bounded auditing.
    /// Paths may contain private directories and must not enter diagnostic JSON.
    pub file_candidates: Option<Vec<VulkanModuleFileCandidate>>,
}

/// One current-process module whose on-disk file may be observed separately.
#[cfg(feature = "vulkan")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VulkanModuleFileCandidate {
    /// Observed basename, without directories.
    pub name: String,
    /// Current module path, unknown if its bounded native query failed.
    /// This is not proof of the identity of the already mapped image.
    pub path: Option<PathBuf>,
}

/// One supported Vulkan `VK_KHR_cooperative_matrix` multiply-accumulate shape.
#[cfg(feature = "vulkan")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanCooperativeMatrixShape {
    /// Matrix `M` dimension.
    pub m_size: u32,
    /// Matrix `N` dimension.
    pub n_size: u32,
    /// Matrix `K` dimension.
    pub k_size: u32,
    /// Raw `VkComponentTypeKHR` value for operand `A`.
    pub a_type_raw: i32,
    /// Raw `VkComponentTypeKHR` value for operand `B`.
    pub b_type_raw: i32,
    /// Raw `VkComponentTypeKHR` value for accumulator `C`.
    pub c_type_raw: i32,
    /// Raw `VkComponentTypeKHR` value for result matrix `D`.
    pub result_type_raw: i32,
    /// Whether saturating accumulation is supported for this shape.
    pub saturating_accumulation: bool,
    /// Raw `VkScopeKHR` execution scope.
    pub scope_raw: i32,
}

/// Bounded Vulkan physical-device compute limits relevant to kernel tiling.
#[cfg(feature = "vulkan")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanComputeLimits {
    /// Maximum total workgroup shared memory in bytes (`maxComputeSharedMemorySize`).
    pub max_compute_shared_memory_size: u32,
    /// Maximum local workgroup invocations (`maxComputeWorkGroupInvocations`).
    pub max_compute_work_group_invocations: u32,
    /// Maximum local workgroup extent `(x, y, z)` (`maxComputeWorkGroupSize`).
    pub max_compute_work_group_size: [u32; 3],
    /// Maximum single storage-buffer binding range in bytes (`maxStorageBufferRange`).
    pub max_storage_buffer_range: u32,
    /// Maximum simultaneous device memory allocations (`maxMemoryAllocationCount`).
    pub max_memory_allocation_count: u32,
}

/// One Vulkan physical device; observations do not qualify an inference backend.
#[cfg(feature = "vulkan")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VulkanAdapter {
    /// Driver-reported device name, not an independently established `SoC` SKU.
    pub name: String,
    /// Numeric Vulkan vendor identifier.
    pub vendor_id: u32,
    /// Numeric Vulkan device identifier.
    pub device_id: u32,
    /// Vendor-specific raw driver version; never decoded as a Vulkan version.
    pub driver_version: u32,
    /// Device-supported packed Vulkan API version.
    pub api_version: u32,
    /// Stable Vulkan device-type label.
    pub device_type: &'static str,
    /// Whether the API identifies an integrated or discrete hardware GPU.
    pub hardware_gpu: bool,
    /// Compute-capable queue families and their queue counts.
    pub compute_queues: Vec<(u32, u32)>,
    /// Advertised device extension names, sorted for stable inspection.
    pub extensions: Vec<String>,
    /// Heap sizes and device-local flags; these are capacities, not free budgets.
    pub memory_heaps: Vec<(u64, bool)>,
    /// Supported half-precision shader arithmetic, unknown before Vulkan 1.1 queries.
    pub shader_float16: Option<bool>,
    /// Supported integer-dot shader feature, unknown before Vulkan 1.1 queries.
    pub shader_integer_dot_product: Option<bool>,
    /// Whether packed signed 4x8-bit integer dot product is hardware-accelerated.
    pub integer_dot_4x8_packed_signed_accelerated: Option<bool>,
    /// Supported `VK_KHR_cooperative_matrix` shapes when advertised and enumerated.
    pub cooperative_matrix_properties: Option<Vec<VulkanCooperativeMatrixShape>>,
    /// Whether `VK_QCOM_cooperative_matrix_conversion` is advertised.
    pub qcom_cooperative_matrix_conversion: bool,
    /// Per-heap `(budget_bytes, usage_bytes)` from `VK_EXT_memory_budget`, when advertised.
    pub memory_budget_heaps: Option<Vec<(u64, u64)>>,
    /// Physical device compute and storage limits from `VkPhysicalDeviceLimits`.
    pub compute_limits: VulkanComputeLimits,
    /// Subgroup width when Vulkan 1.1 properties are available.
    pub subgroup_size: Option<u32>,
    /// Raw subgroup operation flags when Vulkan 1.1 properties are available.
    pub subgroup_operations: Option<u32>,
    /// Raw shader-stage flags restricting the reported subgroup operations.
    pub subgroup_stages: Option<u32>,
}

/// Query the native system Vulkan loader without creating a logical device.
///
/// Requires a native ARM64 process and rejects Vulkan environment overrides.
/// Enumeration storage is bounded. Driver calls themselves have no timeout;
/// callers must isolate this operation in a deadline-owned child process, as
/// the maintenance `host-gpu` command does. No kernels, surfaces or windows are
/// created, and no device memory is allocated.
///
/// # Errors
/// Returns unsupported target, missing loader, malformed/oversized native data
/// or Vulkan failures. Installed loader/ICD trust is not established by a query.
/// Known stages carry a downcastable [`VulkanQueryFailure`]; a failed query does
/// not establish that the physical device lacks Vulkan support.
#[cfg(feature = "vulkan")]
pub fn vulkan_inventory() -> io::Result<VulkanInventory> {
    native::vulkan::query()
}

/// Model-free observations from the absolute Windows system `OpenCL` loader.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenClInventory {
    /// Selected loader path; this is not a signature or dependency trust verdict.
    pub loader_path: PathBuf,
    /// Enumerated `OpenCL` platforms, each containing only GPU observations.
    pub platforms: Vec<OpenClPlatform>,
}

/// One system `OpenCL` platform, without a context, queue or executable program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenClPlatform {
    /// Driver-reported platform name.
    pub name: String,
    /// Driver-reported vendor name.
    pub vendor: String,
    /// `OpenCL` platform version string; not parsed as a kernel qualification.
    pub version: String,
    /// Enumerated GPU devices; CPU devices are not requested.
    pub devices: Vec<OpenClDevice>,
}

/// Read-only capabilities of one `OpenCL` GPU; no inference backend is qualified.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenClDevice {
    /// Driver-reported GPU name, not an independently verified silicon SKU.
    pub name: String,
    /// Driver-reported vendor name.
    pub vendor: String,
    /// Raw Khronos/PCI vendor identifier.
    pub vendor_id: u32,
    /// Driver-reported `OpenCL` device version.
    pub version: String,
    /// Driver-reported driver version.
    pub driver_version: String,
    /// `OpenCL` C language version reported by the compiler.
    pub opencl_c_version: String,
    /// Whether the driver reports the device currently available.
    pub available: bool,
    /// Whether the driver reports an online `OpenCL` C compiler.
    pub compiler_available: bool,
    /// Device address width in bits.
    pub address_bits: u32,
    /// Reported global memory capacity, not a live free-memory budget.
    pub global_memory_bytes: u64,
    /// Maximum single memory-object allocation, not guaranteed allocatable bytes.
    pub maximum_allocation_bytes: u64,
    /// Reported local memory capacity per compute unit.
    pub local_memory_bytes: u64,
    /// Advertised extension names in sorted order.
    pub extensions: Vec<String>,
}

/// Enumerate native ARM64 GPU capabilities using the absolute system `OpenCL` loader.
///
/// Queries create no context, queue, program, kernel or device allocation. Storage
/// is bounded to eight platforms, sixteen total GPUs and 16 KiB per string.
/// Driver calls need an externally owned deadline, as in `host-opencl`.
///
/// # Errors
/// Rejects emulated/non-ARM64 queries, `OpenCL` environment overrides, missing
/// loaders, failed calls and malformed/oversized driver data. Loading the system
/// implementation does not establish its signature or dependency provenance.
pub fn opencl_inventory() -> io::Result<OpenClInventory> {
    native::opencl::query()
}

/// Exact operator-selected `OpenCL` platform, GPU and driver for the fixed pilot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenClProjectionSelection {
    /// Exact platform name, without a first-device or alternate-driver fallback.
    pub platform_name: String,
    /// Exact reported Adreno X2 device name; no silicon qualification is implied.
    pub device_name: String,
    /// Exact driver version reported by the selected device.
    pub driver_version: String,
    /// Exact reported vendor identifier.
    pub vendor_id: u32,
}

/// Completed per-block integer dots from the standalone original-PQ2 pilot.
#[derive(Debug)]
pub struct OpenClPq2Projection {
    /// System loader pathname for a separate bounded file-identity observation.
    pub loader_path: PathBuf,
    /// Driver-reported selected device, without a backend-qualification claim.
    pub device: OpenClDevice,
    /// One signed integer dot per original 128-weight block, in row-major order.
    pub integer_dots: Vec<i32>,
    /// Native release results for kernel, program, three buffers, queue, context.
    pub cleanup_codes: [i32; 7],
}

/// Supported Qualcomm Adreno GPU families on native Windows ARM64 targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdrenoFamily {
    /// Adreno X1-85 GPU on Snapdragon X Elite (`3` slices, `6` Dual CUs, `1536` FP32 ALUs).
    X1_85,
    /// Adreno X2-90 GPU on Snapdragon X2 Elite Extreme.
    X2_90,
}

impl AdrenoFamily {
    /// Recognize Adreno X1-85 or X2-90 GPU device names without enabling automatic selection.
    #[must_use]
    pub fn from_device_name(name: &str) -> Option<Self> {
        if name.is_empty() || name.len() > 512 || name.chars().any(char::is_control) {
            return None;
        }
        let lower = name.to_ascii_lowercase();
        if !lower.contains("adreno") {
            return None;
        }
        let is_x1 = (lower.contains("x1-85") || lower.contains("x1")) && !lower.contains("x2");
        let is_x2 = (lower.contains("x2-90") || lower.contains("x2")) && !lower.contains("x1");
        match (is_x1, is_x2) {
            (true, false) => Some(Self::X1_85),
            (false, true) => Some(Self::X2_90),
            _ => None,
        }
    }
}

/// Pure `OpenCL` C 1.2 integer kernel variant for Adreno `Q2_0`/`Q8_K` and `FWHT` execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenClKernelKind {
    /// Single-block 128-symbol scalar `PQ2`/`Q8_K` dot product (`pq2_q8k_block_dots`).
    BlockDots,
    /// Four-row batched decode GEMV reusing loaded `Q8_K` activations (`pq2_q8k_gemv_4row`).
    Gemv4Row,
    /// Eight-row batched decode GEMV for large feed-forward shapes (`pq2_q8k_gemv_8row`).
    Gemv8Row,
    /// Four-token batched prefill column GEMV reusing decoded `PQ2` weights (`pq2_q8k_prefill_4col`).
    Prefill4Col,
    /// Local-memory 1024-point radix-2 integer Walsh-Hadamard transform (`fwht_local_1024`).
    FwhtLocal1024,
}

impl OpenClKernelKind {
    /// Exact `OpenCL` kernel entry-point symbol for this variant.
    #[must_use]
    pub const fn entry_point(self) -> &'static str {
        match self {
            Self::BlockDots => "pq2_q8k_block_dots",
            Self::Gemv4Row => "pq2_q8k_gemv_4row",
            Self::Gemv8Row => "pq2_q8k_gemv_8row",
            Self::Prefill4Col => "pq2_q8k_prefill_4col",
            Self::FwhtLocal1024 => "fwht_local_1024",
        }
    }
}

/// Fixed `OpenCL` C 1.2 source whose digest must be reviewed before pilot execution.
#[must_use]
pub const fn opencl_pq2_kernel_source() -> &'static str {
    native::opencl::projection::SOURCE
}

/// Pure `OpenCL` C 1.2 source for a specific Adreno `Q2_0`/`Q8_K` or `FWHT` kernel variant.
#[must_use]
pub const fn opencl_pq2_kernel_source_for(kind: OpenClKernelKind) -> &'static str {
    native::opencl::projection::source_for(kind)
}

/// Execute one fixed K=5120, M=17408 `PQ2/Q8_K` integer projection on Adreno X2.
///
/// The original 34-byte PQ2 blocks remain unchanged. `codes` contains 5120
/// already-prepared signed `Q8_K` codes represented as two's-complement bytes.
/// Original half scales, `Q8_K` scales and ordered FMA reduction stay with the
/// caller. This opens a context, compiles the fixed `OpenCL` C 1.2 kernel and
/// allocates at most 26,465,280 explicit device-buffer bytes. Driver/compiler
/// resources are additional and have no verified native memory budget.
///
/// Call only inside an externally deadline-owned child with the ordinary worker
/// mitigations. Driver calls have no intrinsic timeout. A failed terminal fence
/// retains every uncertain owner until supervised process exit. This primitive
/// confers no model, adapter, state, performance or application qualification.
///
/// # Errors
/// Rejects non-ARM64 execution, environment overrides, wrong fixed input lengths,
/// ambiguous/changed/unavailable device identity, unsupported reported allocation
/// limits, failed program build, submission or terminal readback.
pub fn opencl_pq2_projection(
    selection: &OpenClProjectionSelection,
    weights: &[u8],
    codes: &[u8],
) -> io::Result<OpenClPq2Projection> {
    native::opencl::projection::run(selection, weights, codes)
}

/// Execute a parameterized `OpenCL` C 1.2 `PQ2/Q8_K` or `FWHT` kernel on a selected Adreno GPU.
///
/// # Errors
/// Rejects invalid geometry/buffer bounds (`> 64 MiB` per buffer), environment overrides,
/// non-ARM64 hosts, mismatched [`AdrenoFamily`], unavailable devices, failed program builds,
/// submission failures, or terminal readback errors.
pub fn opencl_pq2_projection_with_kernel(
    selection: &OpenClProjectionSelection,
    family: AdrenoFamily,
    kind: OpenClKernelKind,
    rows: usize,
    columns: usize,
    tokens: usize,
    weights: &[u8],
    codes: &[u8],
) -> io::Result<OpenClPq2Projection> {
    native::opencl::projection::run_with_kernel(
        selection, family, kind, rows, columns, tokens, weights, codes,
    )
}

/// One observed Windows CPU Set; processor indices are relative to `group`.
///
/// Efficiency and scheduling classes are raw operating-system observations,
/// not Snapdragon core names or evidence that a particular thread count wins.
/// Parking and allocation are changing state rather than compatibility identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent operating-system CPU Set flags are retained without inventing a policy state machine."
)]
pub struct CpuSet {
    /// Windows CPU Set identifier, valid for the current system observation.
    pub id: u32,
    /// Processor group containing this CPU Set's home processor.
    pub group: u16,
    /// Home logical processor within its group.
    pub logical_processor_index: u8,
    /// Group-relative core index; shared by hardware threads on one core.
    pub core_index: u8,
    /// Group-relative shared last-level cache index.
    pub last_level_cache_index: u8,
    /// Group-relative NUMA node index.
    pub numa_node_index: u8,
    /// Windows efficiency class; larger values identify faster, less efficient cores.
    pub efficiency_class: u8,
    /// Raw Windows scheduling class, without an inferred performance policy.
    pub scheduling_class: u8,
    /// Whether Windows currently reports the CPU Set as parked.
    pub parked: bool,
    /// Whether Windows reports the CPU Set as allocated.
    pub allocated: bool,
    /// Whether the CPU Set is allocated to this querying process.
    pub allocated_to_target_process: bool,
    /// Whether Windows reports the CPU Set as real-time.
    pub real_time: bool,
}

/// Reads bounded, current-process Windows CPU Set topology without changing affinity.
///
/// The returned records are sorted by processor group and logical processor.
/// Unknown future record types are skipped. Observations do not establish ISA,
/// runtime eligibility, process affinity restrictions or a qualified CPU profile.
/// See [the Windows API contract](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getsystemcpusetinformation).
///
/// # Errors
/// Returns native query failures, `Unsupported` when no known CPU Set records
/// exist, or `InvalidData` for malformed, duplicate or oversized records. A
/// changing required buffer size is retried at most three times.
pub fn cpu_sets() -> io::Result<Vec<CpuSet>> {
    native::cpu::query()
}

/// Windows-reported availability of native ARM64 packed-kernel instructions.
///
/// A positive flag is an OS observation, not kernel correctness or performance
/// qualification. A negative flag means unsupported **or not detected** by the
/// Windows HAL; it does not prove that the silicon lacks the instruction set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Arm64Isa {
    /// `PF_ARM_NEON_INSTRUCTIONS_AVAILABLE` (19).
    pub advsimd: bool,
    /// `PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE` (43), including SDOT/UDOT.
    pub dotprod: bool,
    /// `PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE` (66), NEON integer matrix multiply.
    pub i8mm: bool,
}

/// Queries native ARM64 instruction availability without executing optional ISA.
///
/// Uses [the Windows processor-feature API](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent).
/// I8MM detection requires Windows 11 24H2 / Server 2025 or later; earlier HALs
/// can return false even on capable hardware. This does not change dispatch,
/// infer support from a processor name, or inspect an emulated process's ISA.
///
/// # Errors
/// Returns `Unsupported` outside an ARM64 build; emulated x86/x64 callers cannot
/// produce a native ARM64 eligibility observation.
pub fn arm64_isa() -> io::Result<Arm64Isa> {
    native::isa::query()
}

/// Stable observed host facts for partitioning host-native compiler outputs.
///
/// This is build-cache identity, not a runtime compatibility or ISA qualification
/// verdict. It deliberately excludes changing CPU availability and power state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeBuildHost {
    /// Bounded CPU description from the native hardware registry.
    pub processor_name: String,
    /// CPU architecture, family, model and revision reported by Windows.
    pub processor_identifier: String,
    /// CPU vendor reported by the native hardware registry.
    pub vendor_identifier: String,
    /// Actual running kernel major, minor and build numbers from `RtlGetVersion`.
    pub windows_version: [u32; 3],
    /// Sorted available Windows processor-feature IDs in the bounded range 0..256.
    ///
    /// Includes optional ARM features queried by the pinned native GGML build;
    /// negative observations mean unsupported or not detected by the Windows HAL.
    pub processor_features: Vec<u32>,
}

/// Observes native Windows ARM64 compiler-host facts without executing optional ISA.
///
/// Reads fixed hardware-registry values, the actual kernel version and bounded
/// `IsProcessorFeaturePresent` flags. No affinity, power or compiler policy changes.
///
/// # Errors
/// Returns `Unsupported` outside an ARM64 process, native query failures, or
/// `InvalidData` when mandatory identity values are absent, malformed or oversized.
pub fn native_build_host() -> io::Result<NativeBuildHost> {
    native::host::query()
}

/// Native system manufacturer and product name from the fixed BIOS registry key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputerIdentity {
    /// Firmware-reported system manufacturer, without serial or user identity.
    pub manufacturer: String,
    /// Firmware-reported system product name.
    pub model: String,
}

/// Returns bounded firmware-reported manufacturer/model strings without WMI.
///
/// # Errors
/// Rejects unavailable, oversized, malformed or empty registry strings.
pub fn computer_identity() -> io::Result<ComputerIdentity> {
    native::host::computer_identity()
}

/// Current Windows power-source and battery-saver snapshot.
///
/// These changing status values are neither energy telemetry nor immutable
/// hardware identity. Equal snapshots do not rule out intervening changes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PowerStatus {
    /// Whether AC power is connected; `None` preserves Windows' unknown sentinel.
    pub ac_online: Option<bool>,
    /// Whether Windows currently reports battery saver as enabled.
    pub battery_saver: bool,
    /// Reported remaining battery percentage; `None` preserves unknown status.
    pub battery_percent: Option<u8>,
}

/// Reads a power-status snapshot without changing power policy or device settings.
///
/// See [the Windows power-status contract](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-system_power_status).
/// No energy consumption, thermal condition or effective performance mode is inferred.
///
/// # Errors
/// Returns native query failures or `InvalidData` for undocumented field values.
/// Documented unknown AC and battery-percentage sentinels remain `None`.
pub fn power_status() -> io::Result<PowerStatus> {
    native::power::query()
}

/// Checked Windows effective power mode, using notification contract version two.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EffectivePowerMode {
    /// Battery saver, also named energy saver high savings in newer SDKs.
    BatterySaver,
    /// Better battery, also named energy saver standard in newer SDKs.
    BetterBattery,
    /// Balanced performance.
    Balanced,
    /// High performance under the legacy high-performance overlay.
    HighPerformance,
    /// Maximum performance.
    MaxPerformance,
    /// Game-mode power policy.
    GameMode,
    /// Windows mixed-reality power policy.
    MixedReality,
}

impl EffectivePowerMode {
    /// Stable observation label; no energy or performance outcome is inferred.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BatterySaver => "battery-saver",
            Self::BetterBattery => "better-battery",
            Self::Balanced => "balanced",
            Self::HighPerformance => "high-performance",
            Self::MaxPerformance => "max-performance",
            Self::GameMode => "game-mode",
            Self::MixedReality => "mixed-reality",
        }
    }
}

/// Sequential observations of Windows power scheme and effective power mode.
///
/// These are changing policy conditions, not an immutable compatibility identity.
/// A scheme identifier does not capture edits to individual power settings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PowerPolicy {
    /// Active scheme GUID in lowercase canonical hyphenated notation.
    pub active_scheme: String,
    /// Latest observed mode from a temporary version-two notification registration.
    pub effective_mode: EffectivePowerMode,
}

/// Observes active power scheme and effective mode without changing settings.
///
/// Uses [PowerGetActiveScheme](https://learn.microsoft.com/en-us/windows/win32/api/powersetting/nf-powersetting-powergetactivescheme)
/// and the initial current-value notification from
/// [PowerRegisterForEffectivePowerModeNotifications](https://learn.microsoft.com/en-us/windows/win32/api/powersetting/nf-powersetting-powerregisterforeffectivepowermodenotifications).
/// The registration is removed before returning a successful snapshot. No polling,
/// user power-mode names, undocumented overlay API or energy estimate is used.
/// Waiting for the initial asynchronous notification is bounded to one second;
/// native cleanup then waits for the short callback to finish.
///
/// # Errors
/// Returns native failures, a missing initial notification, an unknown mode or
/// a malformed scheme. A failed notification cleanup disables future mode queries
/// in this process rather than accumulating registrations or reusing stale data.
pub fn power_policy() -> io::Result<PowerPolicy> {
    native::power_policy::query()
}

/// Thread-local RAII guard inhibiting Windows idle system sleep on the acquiring thread.
///
/// Calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` on acquisition
/// and restores `SetThreadExecutionState(ES_CONTINUOUS)` on drop, without forcing the
/// display awake or preventing explicit user/lid sleep. Marked `!Send` and `!Sync`
/// because Windows thread execution state is tracked per OS thread and is not
/// reference-counted across overlapping guards on the same thread.
#[must_use]
pub struct SystemAwakeGuard(native::power::NativeSystemAwakeGuard);

impl SystemAwakeGuard {
    /// Inhibits idle system sleep for the calling thread until the returned guard drops.
    ///
    /// # Errors
    /// Returns the last Win32 error when `SetThreadExecutionState` returns zero.
    pub fn acquire() -> io::Result<Self> {
        native::power::NativeSystemAwakeGuard::acquire().map(Self)
    }
}

/// Explicit launch inputs; no shell or executable search path is used.
#[derive(Debug)]
pub struct ProcessSpec {
    /// Expanded absolute path of the executable.
    pub executable: PathBuf,
    /// Arguments encoded with the Windows C runtime argv convention.
    pub arguments: Vec<OsString>,
    /// Expanded absolute native working directory.
    pub working_directory: PathBuf,
    /// Expanded absolute path of a new output evidence file.
    pub stdout: PathBuf,
    /// Expanded absolute path of a new error evidence file.
    pub stderr: PathBuf,
    /// Maximum job-wide committed memory, including descendants.
    pub memory_limit_bytes: usize,
    /// Maximum simultaneously active processes, including the root.
    pub active_process_limit: u32,
}

/// Least-authority native worker filesystem grants.
///
/// Networking denies external ingress and egress, with a per-container host
/// loopback exception for the authenticated local HTTP service. Loopback access
/// is not limited to the broker. Required Windows runtime directories are added
/// by the native boundary; caller profile and data directories are not.
#[cfg(feature = "containment")]
#[derive(Debug)]
pub struct SandboxPolicy {
    /// Exact existing model, credential, executable and runtime library files.
    pub readonly_paths: Vec<PathBuf>,
    /// Existing private worker scratch, also used as its working directory.
    /// PSEC may substitute its own private per-container TEMP/TMP directory.
    pub writable_directory: PathBuf,
}

/// Immutable network authority selected before a contained worker starts.
#[cfg(feature = "containment")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SandboxNetwork {
    /// Existing authenticated HTTP workers may communicate with host services.
    /// This grant is broader than broker-only access.
    HostLoopback,
    /// No network capability, host peer or loopback exception is granted.
    Offline,
}

/// Maximum number of explicit read-only sections inherited by a broker-pipe worker.
#[cfg(feature = "containment")]
pub const MAX_INHERITED_SECTIONS: usize = 8;

/// Inherited `PAGE_READONLY` Windows section handle granted to a contained broker-pipe child.
#[cfg(feature = "containment")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InheritedSectionGrant {
    /// Zero-based caller-assigned section slot matching the input slice index.
    pub slot: u32,
    /// Raw inherited Windows `HANDLE` value valid in the child process.
    pub child_handle: u64,
    /// Nonzero byte length of the regular file backed by the section.
    pub bytes: u64,
}

/// Maximum aggregate private storage permitted across a contained worker's scratch
/// directory and Windows-managed AppContainer storage folder (64 MiB).
pub const MAX_WORKER_PRIVATE_STORAGE_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum filesystem entries permitted across a contained worker's scratch
/// directory and Windows-managed AppContainer storage folder.
pub const MAX_WORKER_PRIVATE_STORAGE_ENTRIES: usize = 1_024;

const MAX_STORAGE_DEPTH: usize = 16;
const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;

/// Requires native PSEC 1.1, host-loopback ingress and security-environment creation.
///
/// This never changes host ACLs, firewall settings or installed Windows features.
/// It does not launch a process or establish actual-model compatibility.
/// Recognized ambient credential variable names reject with a value-free error;
/// other inherited entries remain intact, so unknown secrets remain a residual risk.
///
/// # Errors
/// Returns `Unsupported` when the native contract is unavailable, or a native
/// failure when the host cannot create the required restricted environment, or
/// `InvalidInput` for recognized ambient credential variables.
#[cfg(feature = "containment")]
pub fn native_sandbox_available() -> io::Result<()> {
    validate_sandbox_environment()?;
    native::containment::available(SandboxNetwork::HostLoopback)
}

/// Requires native PSEC 1.1 offline containment without host loopback or parent environment filtering.
///
/// Used by broker-pipe inference workers whose environment is constructed from a minimal
/// six-variable whitelist and whose communication uses parent-owned anonymous pipes.
///
/// # Errors
/// Returns `Unsupported` when the native contract is unavailable, or a native failure
/// when the host cannot create an offline restricted security environment.
#[cfg(feature = "containment")]
pub fn native_sandbox_offline_available() -> io::Result<()> {
    native::containment::available(SandboxNetwork::Offline)
}

/// Reports whether Windows currently advertises a pending update or servicing restart.
///
/// Checks the read-only `Component Based Servicing\RebootPending` and
/// `WindowsUpdate\Auto Update\RebootRequired` keys under `HKLM`. This is an
/// advisory diagnostic only; unavailable keys report `false` and never relax
/// containment or admission requirements.
#[must_use]
pub fn windows_reboot_pending() -> bool {
    native::host::reboot_pending()
}

/// Startup compiler/tool inputs with a child-only inherited-environment overlay.
///
/// The unnamed Job owns the complete tree without applying model-specific
/// memory or process-count limits. Descendants cannot escape this ownership.
#[derive(Debug)]
pub struct ToolProcessSpec {
    /// Expanded absolute executable path; no executable search is performed.
    pub executable: PathBuf,
    /// Windows CRT arguments, or the explicit `cmd.exe /d /c` startup grammar.
    pub arguments: Vec<OsString>,
    /// Expanded absolute native working directory.
    pub working_directory: PathBuf,
    /// Expanded absolute path of a new standard-output evidence file.
    pub stdout: PathBuf,
    /// Expanded absolute path of a new standard-error evidence file.
    pub stderr: PathBuf,
    /// Child-only overrides; Windows system/profile variables cannot be replaced.
    pub environment: BTreeMap<String, String>,
}

/// Current root-process counters, distinct from the job-wide commit limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessMemory {
    /// Root-process private committed bytes.
    pub private_bytes: usize,
    /// Root-process resident working-set bytes.
    pub working_set_bytes: usize,
    /// Operating-system lifetime peak working set, including mapped residency.
    pub peak_working_set_bytes: usize,
}

/// Owns a complete native process tree; dropping it kills remaining processes.
pub struct OwnedProcess(native::NativeProcess);

#[derive(Debug)]
struct UnconfirmedStartCleanup(io::Error);

impl std::fmt::Display for UnconfirmedStartCleanup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, formatter)
    }
}

impl std::error::Error for UnconfirmedStartCleanup {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

fn unconfirmed_start_cleanup(error: io::Error) -> io::Error {
    io::Error::new(error.kind(), UnconfirmedStartCleanup(error))
}

/// Identifies a launch failure whose already-created worker exit is unconfirmed.
///
/// The launch attempted exact-root termination before returning this error, but
/// could not verify completion. Retain granted file identities and private
/// storage and close further admission until application exit. Error kinds such
/// as `InvalidInput` or `Unsupported` alone cannot establish cleanup status.
/// The original error kind, display and source remain available for diagnosis.
#[must_use]
pub fn process_start_cleanup_unconfirmed(error: &io::Error) -> bool {
    error
        .get_ref()
        .is_some_and(<dyn std::error::Error + Send + Sync>::is::<UnconfirmedStartCleanup>)
}

/// Retained identity of the caller whose lifetime bounds a supervised operation.
pub struct ParentProcess(native::ParentProcess);

impl ParentProcess {
    /// Opens a noninheritable synchronization handle and checks creation identity.
    ///
    /// `creation_file_time` is the caller's Windows UTC creation FILETIME, in
    /// 100-nanosecond ticks since 1601. Checking it prevents accepting a reused PID.
    ///
    /// # Errors
    /// Rejects inaccessible, exited or mismatched processes, including PID zero.
    pub fn open(id: u32, creation_file_time: u64) -> io::Result<Self> {
        native::ParentProcess::open(id, creation_file_time).map(Self)
    }

    /// Confirms that the retained caller process has not exited.
    ///
    /// # Errors
    /// Returns a failed native wait or `BrokenPipe` after caller exit.
    pub fn ensure_alive(&self) -> io::Result<()> {
        self.0.ensure_alive()
    }
}

impl OwnedProcess {
    /// Starts a natively contained worker with verified Job and mitigation attributes.
    ///
    /// The exact worker retains the existing PID, memory, process-count and
    /// lifetime contracts. Writable application storage is private scratch;
    /// Windows also supplies its own per-container private temporary directory.
    /// OS/profile environment entries are inherited; TEMP/TMP request scratch
    /// but PSEC can replace them with that OS-managed temporary directory.
    /// All policy and capability failures reject without an uncontained retry.
    /// Parent-owned pipe readers retain at most 16 MiB for each standard stream;
    /// excess output or a failed capture terminates the exact worker. The worker
    /// inherits no seekable log-file handle. This does not quota its writable
    /// application scratch or Windows-managed private temporary directory.
    /// Native launches require a verified one-process Job before the first
    /// instruction. PSEC may omit ordinary creation attributes, so suspended
    /// Job assignment is repaired explicitly and native terminate-on-close is
    /// also required. There is no retry without containment or resource limits.
    ///
    /// # Errors
    /// Returns invalid launch/grant paths, reparse paths, overlapping read/write
    /// grants, unavailable native PSEC support or failed native creation.
    #[cfg(feature = "containment")]
    pub fn start_sandboxed(spec: &ProcessSpec, policy: &SandboxPolicy) -> io::Result<Self> {
        Self::validate_sandboxed(spec, std::env::vars_os().map(|(name, _)| name))?;
        let command = command_line(&spec.executable, &spec.arguments)?;
        native::NativeProcess::start_sandboxed(spec, command, policy).map(Self)
    }

    /// Starts a contained worker with bounded anonymous-pipe standard streams.
    ///
    /// Returns the owner, parent stdin writer and parent stdout reader. Only the
    /// worker endpoints and a write-only `NUL` stderr handle enter the inherited
    /// handle list. `spec.stdout` and `spec.stderr` remain validated reserved
    /// names and no files are created there. The binary protocol must carry
    /// bounded errors; an untrusted worker cannot fill a stderr log on disk.
    /// Each pipe requests a 64 KiB kernel buffer; the protocol must independently
    /// bound messages and apply backpressure.
    ///
    /// The caller must supervise blocking pipe operations off its UI thread and
    /// stop the worker before joining a stalled reader. Dropping the parent
    /// writer signals input EOF; dropping or stopping the owner kills the worker
    /// and releases its output endpoint. Offline launch does not require the
    /// host-loopback feature. No policy or resource-limit fallback is attempted.
    ///
    /// # Errors
    /// Returns the same admission errors as [`Self::start_sandboxed`], plus failed
    /// pipe creation or handle-inheritance configuration.
    #[cfg(feature = "containment")]
    pub fn start_sandboxed_piped(
        spec: &ProcessSpec,
        policy: &SandboxPolicy,
        network: SandboxNetwork,
    ) -> io::Result<(Self, std::fs::File, std::fs::File)> {
        Self::validate_sandboxed(spec, std::env::vars_os().map(|(name, _)| name))?;
        let command = command_line(&spec.executable, &spec.arguments)?;
        let (process, input, output) =
            native::NativeProcess::start_sandboxed_piped(spec, command, policy, network)?;
        Ok((Self(process), input, output))
    }

    /// Starts an offline piped worker without inheriting caller environment entries.
    ///
    /// The requested child environment contains only `SystemRoot`, `windir` and
    /// `SystemDrive` derived from the native Windows directory, plus private
    /// scratch `TEMP`/`TMP` and `LOCALAPPDATA`. Windows PSEC requires the latter
    /// name to be present; it receives the private scratch path, not the user's
    /// profile. It supplies no `PATH`, caller profile paths, application options
    /// or arbitrary inherited values. Windows PSEC may supply its own
    /// private temporary-storage environment. The parent environment is unchanged.
    /// Caller credential and backend-override names do not reject this launch:
    /// no caller entries are inherited, and the constructed child names are
    /// independently admitted. Existing filesystem, Job, memory, handle and cleanup
    /// requirements from [`Self::start_sandboxed_piped`] remain in force. There is
    /// no loopback grant.
    ///
    /// # Errors
    /// Returns the same path, policy, resource, handle and native launch errors as
    /// [`Self::start_sandboxed_piped`], or native Windows-directory lookup and
    /// minimal child-environment admission/encoding failures.
    #[cfg(feature = "containment")]
    pub fn start_sandboxed_piped_minimal(
        spec: &ProcessSpec,
        policy: &SandboxPolicy,
    ) -> io::Result<(Self, std::fs::File, std::fs::File)> {
        Self::validate_sandboxed(spec, std::iter::empty())?;
        let command = command_line(&spec.executable, &spec.arguments)?;
        let (process, input, output) =
            native::NativeProcess::start_sandboxed_piped_minimal(spec, command, policy)?;
        Ok((Self(process), input, output))
    }

    /// Starts an offline broker-pipe inference worker with minimal environment and bounded `stderr`.
    ///
    /// Returns the owned worker plus parent stdin writer and parent stdout reader pipes.
    /// The worker enters an offline `AppContainer` (`SandboxNetwork::Offline`), inherits only
    /// `stdin`, `stdout`, and a bounded parent-drained `stderr` capture (`spec.stderr`, 16 MiB)
    /// while creating a 0-byte `spec.stdout` sentinel, and receives only the minimal
    /// six-variable Windows environment (`SystemRoot`, `windir`, `SystemDrive`, `LOCALAPPDATA`,
    /// `TEMP`, `TMP`). Before resuming the worker thread, native verification confirms
    /// one-process Job membership, `Win32k` denial, `TokenIsAppContainer != 0`, and resolves the
    /// OS AppContainer folder path for private-storage accounting alongside
    /// `policy.writable_directory`.
    ///
    /// # Errors
    /// Returns invalid launch/grant paths, reparse paths, overlapping read/write
    /// grants, unavailable offline native PSEC support, AppContainer verification failure,
    /// or failed native creation.
    #[cfg(feature = "containment")]
    pub fn start_sandboxed_broker_pipe(
        spec: &ProcessSpec,
        policy: &SandboxPolicy,
    ) -> io::Result<(Self, std::fs::File, std::fs::File)> {
        let (process, input, output, _grants) =
            Self::start_sandboxed_broker_pipe_with_sections(spec, policy, &[])?;
        Ok((process, input, output))
    }

    /// Starts an offline broker-pipe inference worker with explicit inherited `PAGE_READONLY` sections.
    ///
    /// Creates one unnamed `PAGE_READONLY` section per supplied regular non-empty file,
    /// duplicates each handle into an inheritable descriptor attenuated strictly to
    /// `SECTION_MAP_READ | SECTION_QUERY`, passes those section handles alongside
    /// `[stdin, stdout, stderr]` in `PROC_THREAD_ATTRIBUTE_HANDLE_LIST`, and closes the parent's
    /// inheritable section handles before returning `(process, parent_stdin, parent_stdout, grants)`.
    ///
    /// # Errors
    /// Returns `InvalidInput` if more than [`MAX_INHERITED_SECTIONS`] files are supplied, any
    /// file is empty or not a regular non-reparse file, or launch/policy validation fails;
    /// returns native section creation, containment, or AppContainer verification errors otherwise.
    #[cfg(feature = "containment")]
    pub fn start_sandboxed_broker_pipe_with_sections(
        spec: &ProcessSpec,
        policy: &SandboxPolicy,
        sections: &[&std::fs::File],
    ) -> io::Result<(
        Self,
        std::fs::File,
        std::fs::File,
        Vec<InheritedSectionGrant>,
    )> {
        if sections.len() > MAX_INHERITED_SECTIONS {
            return Err(invalid(
                "broker-pipe worker accepts at most eight inherited read-only sections",
            ));
        }
        Self::validate_sandboxed(spec, std::iter::empty())?;
        let command = command_line(&spec.executable, &spec.arguments)?;
        let (process, input, output, grants) =
            native::NativeProcess::start_sandboxed_broker_pipe_with_sections(
                spec, command, policy, sections,
            )?;
        Ok((Self(process), input, output, grants))
    }

    #[cfg(feature = "containment")]
    fn validate_sandboxed(
        spec: &ProcessSpec,
        inherited_names: impl IntoIterator<Item = OsString>,
    ) -> io::Result<()> {
        validate_process_spec(spec)?;
        if spec.active_process_limit != 1 {
            return Err(invalid(
                "native sandbox requires a single-process worker policy",
            ));
        }
        validate_sandboxed_environment_names(inherited_names)
    }

    /// Atomically owns a startup compiler/tool tree and then resumes its root.
    ///
    /// The complete Unicode environment is inherited and merged case-insensitively
    /// with the child-only overlay. OS/profile entries remain unchanged. Only the
    /// three evidence streams are inherited as handles, and the window is hidden.
    /// Closing the owner kills descendants even if the root already exited.
    ///
    /// # Errors
    /// Rejects invalid paths, malformed arguments or environment entries, protected
    /// system/profile overrides, existing evidence files, or native launch failures.
    pub fn start_tool(spec: &ToolProcessSpec) -> io::Result<Self> {
        Self::start_tool_inner(spec, native::ToolPriority::Inherited, None)
    }

    /// Owns a background startup tool tree at below-normal CPU priority.
    ///
    /// Applies the priority limit to the complete Job before its root starts,
    /// including compiler descendants and nested owned tools. The caller and
    /// its other Jobs retain their scheduling policy. This changes CPU priority
    /// only; it does not promise disk bandwidth or I/O priority isolation.
    /// All path, environment, handle and teardown rules match [`Self::start_tool`].
    ///
    /// # Errors
    /// Returns the same validation and launch failures as [`Self::start_tool`],
    /// including failure to establish the required Job priority limit.
    pub fn start_background_tool(spec: &ToolProcessSpec) -> io::Result<Self> {
        Self::start_tool_inner(spec, native::ToolPriority::Background, None)
    }

    /// Starts a tool with explicit child-only environment removals.
    ///
    /// Uses the same atomic Job ownership as `start_tool`. Windows system/profile
    /// variables are protected; the parent environment is never changed.
    ///
    /// # Errors
    /// Rejects invalid removals and every path, Job or creation failure from `start_tool`.
    pub fn start_tool_with_removed(spec: &ToolProcessSpec, removed: &[String]) -> io::Result<Self> {
        Self::start_tool_inner(spec, native::ToolPriority::Inherited, Some(removed))
    }

    fn start_tool_inner(
        spec: &ToolProcessSpec,
        priority: native::ToolPriority,
        removed: Option<&[String]>,
    ) -> io::Result<Self> {
        for path in [
            &spec.executable,
            &spec.working_directory,
            &spec.stdout,
            &spec.stderr,
        ] {
            validate_path(path)?;
        }
        if !spec.executable.is_file() || !spec.working_directory.is_dir() {
            return Err(invalid("executable or working directory does not exist"));
        }
        let command = tool_command_line(&spec.executable, &spec.arguments)?;
        let environment = match removed {
            Some(removed) => Some(native::tool_environment_with_removed(
                &spec.environment,
                removed,
            )?),
            None => native::tool_environment(&spec.environment)?,
        };
        native::NativeProcess::start_tool(spec, command, environment.as_deref(), priority).map(Self)
    }

    /// Creates evidence files, atomically assigns a suspended root, then resumes.
    ///
    /// The full OS/profile environment is inherited without modification.
    /// Model/backend override variables are rejected before creating any files.
    /// Headless CPU workers deny Win32k calls, legacy extension points and
    /// remote/low-integrity executable images, with heap termination enabled.
    /// These creation-time mitigations preserve nested Jobs but do not restrict
    /// ordinary filesystem access, sockets or the caller's security identity.
    ///
    /// # Errors
    /// Rejects ambiguous paths, invalid limits/arguments, inherited backend
    /// overrides, existing evidence files, or any failed Windows operation,
    /// including unsupported required process-creation mitigations.
    pub fn start(spec: &ProcessSpec) -> io::Result<Self> {
        validate_process(spec)?;
        let command = command_line(&spec.executable, &spec.arguments)?;
        native::NativeProcess::start_with_profile(
            spec,
            command,
            native::MitigationProfile::Headless,
        )
        .map(Self)
    }

    /// Starts a bounded, model-free `--version` or `--help` inspection.
    ///
    /// The caller verifies and retains the inspected executable and dependencies.
    /// Exactly one fixed inspection argument and a one-process limit are required.
    /// Memory, Job ownership, child-process denial, evidence and inherited-environment
    /// validation match worker launches. Heap termination, extension-point denial
    /// and image-load restrictions remain enabled. Win32k calls remain available
    /// so legacy runtime inspection can initialize imported Windows shell DLLs.
    /// This is an explicit inspection policy, never a fallback for model workers.
    ///
    /// # Errors
    /// Rejects arguments other than a single `--version` or `--help`, process
    /// limits other than one, and every invalid worker launch input or native failure.
    pub fn start_inspection(spec: &ProcessSpec) -> io::Result<Self> {
        validate_process(spec)?;
        if spec.active_process_limit != 1
            || !matches!(spec.arguments.as_slice(), [argument] if argument == "--version" || argument == "--help")
        {
            return Err(invalid(
                "inspection requires one --version or --help argument and a single-process limit",
            ));
        }
        let command = command_line(&spec.executable, &spec.arguments)?;
        native::NativeProcess::start_with_profile(
            spec,
            command,
            native::MitigationProfile::Inspection,
        )
        .map(Self)
    }

    /// Starts a bounded graphics discovery or standalone pilot probe.
    ///
    /// Preserves the single-process Job limit, committed memory limit,
    /// heap termination, extension-point denial, image-load restrictions, and
    /// inherited environment validation of worker launches, while leaving Win32k
    /// system calls available so the Windows Vulkan loader and `OpenCL` WDDM
    /// adapters can enumerate via `gdi32.dll!D3DKMTEnumAdapters2`. Restricted
    /// strictly to allowlisted graphics subcommands with `--request PATH`.
    ///
    /// # Errors
    /// Rejects non-unit process limits, unallowlisted arguments, ambiguous
    /// request paths, and every invalid worker launch input or native failure.
    pub fn start_graphics_probe(spec: &ProcessSpec) -> io::Result<Self> {
        validate_process(spec)?;
        let [subcommand, option, request_path] = spec.arguments.as_slice() else {
            return Err(invalid(
                "graphics probe requires [subcommand, --request, PATH] and a single-process limit",
            ));
        };
        if spec.active_process_limit != 1
            || !matches!(
                subcommand.to_str(),
                Some("host-gpu-query" | "host-opencl-query" | "opencl-projection-worker")
            )
            || option != "--request"
            || request_path.is_empty()
        {
            return Err(invalid(
                "graphics probe requires a single-process limit and an allowlisted graphics probe command",
            ));
        }
        validate_path(Path::new(request_path))?;
        let command = command_line(&spec.executable, &spec.arguments)?;
        native::NativeProcess::start_with_profile(
            spec,
            command,
            native::MitigationProfile::GraphicsProbe,
        )
        .map(Self)
    }

    /// Returns the root Windows process identifier.
    #[must_use]
    pub const fn id(&self) -> u32 {
        self.0.id()
    }

    /// Polls root and contained-log completion without waiting.
    ///
    /// Exit code 259 is handled correctly. A sandboxed HTTP worker's status is
    /// returned only after both bounded captures finish and are joined.
    ///
    /// # Errors
    /// Returns failed Windows wait/query operations or contained log failures,
    /// with the observed native exit code or an explicit unknown observation.
    pub fn try_wait(&self) -> io::Result<Option<u32>> {
        self.0.try_wait()
    }

    /// Waits within one bounded interval for root exit and contained log capture.
    ///
    /// A sandboxed HTTP worker also requires completed, joined standard-stream
    /// capture before returning its exit status. Binary-pipe workers have no log
    /// captures. Captured bytes and the worker's native exit code stay unchanged.
    ///
    /// # Errors
    /// Rejects intervals at or above the native infinite-wait sentinel, failed
    /// waits, or sandbox log overflow/read/write failures. A log failure does not
    /// establish cleanup; the caller must still explicitly stop the owner.
    pub fn wait_for_exit(&self, timeout: Duration) -> io::Result<Option<u32>> {
        self.0.wait_for_exit(timeout)
    }

    /// Terminates the entire tree and waits within a positive timeout.
    ///
    /// # Errors
    /// Returns invalid zero timeouts, Windows failures, or `TimedOut` if active
    /// processes or contained log captures remain after the deadline. A known log
    /// failure does not prevent confirming process cleanup. Dropping still closes
    /// the Job and native containment without renewing an explicitly attempted
    /// stop's wait.
    pub fn stop(&self, timeout: Duration) -> io::Result<()> {
        self.0.stop(timeout)
    }

    /// Returns the current number of live members in this owned Job.
    ///
    /// A completed root may still have descendants; callers must retain ownership
    /// until this count reaches zero or complete explicit tree termination.
    ///
    /// # Errors
    /// Returns failed Windows Job accounting operations.
    pub fn active_processes(&self) -> io::Result<u32> {
        self.0.active_processes()
    }

    /// Captures bounded synthetic-fixture Job counters and member identities.
    ///
    /// Available only with `test-probe`. Returns independent accounting, raw
    /// assigned/listed/PID fields, retained member PIDs and snapshot parent PIDs
    /// and executable basenames; no paths, arguments or environment values.
    /// Later query failures remain inline and never erase earlier observations.
    /// Snapshot metadata is diagnostic, not admission.
    ///
    /// Native-query, 64-member and snapshot-bound failures are retained inline.
    #[cfg(feature = "test-probe")]
    #[must_use]
    pub fn test_job_accounting(&self) -> String {
        self.0.test_job_accounting()
    }

    /// Returns the observed peak job-wide committed bytes.
    /// Windows may include denied allocation requests in this diagnostic counter;
    /// it is not a measurement of successfully resident or accepted allocations.
    ///
    /// # Errors
    /// Returns failed Windows job-accounting operations.
    pub fn peak_job_memory_bytes(&self) -> io::Result<usize> {
        self.0.peak_job_memory_bytes()
    }

    /// Returns current root private commitment, working set and lifetime peak residency.
    ///
    /// # Errors
    /// Returns failed Windows process-memory queries.
    pub fn memory_usage(&self) -> io::Result<ProcessMemory> {
        self.0.memory_usage()
    }

    /// Measures aggregate bytes and entries across this worker's private storage roots.
    ///
    /// Includes the configured `writable_directory` and, when present on a natively
    /// contained worker, its kernel-resolved OS AppContainer storage folder. Traversal is
    /// iterative, bounded to depth 16, and rejects reparse points without following links.
    ///
    /// # Errors
    /// Rejects non-directory roots, reparse points, traversal exceeding depth 16, or
    /// unexpected filesystem errors while scanning private storage.
    pub fn private_storage_usage(&self) -> io::Result<(u64, usize)> {
        measure_private_roots(self.0.private_roots(), u64::MAX, usize::MAX)
    }

    /// Checks this worker's private scratch and OS AppContainer storage against caps.
    ///
    /// Returns the observed `(bytes, entries)` pair when within `max_bytes` and
    /// `max_entries`. Traversal is iterative, ignores missing roots as zero usage, and
    /// fails closed on any reparse point or excessive directory nesting.
    ///
    /// # Errors
    /// Rejects exceeded byte or entry caps (`InvalidData` / `FileTooLarge`), reparse
    /// points, depth over 16, or non-directory roots.
    pub fn check_private_storage_budget(
        &self,
        max_bytes: u64,
        max_entries: usize,
    ) -> io::Result<(u64, usize)> {
        measure_private_roots(self.0.private_roots(), max_bytes, max_entries)
    }
}

/// Checks an explicit list of private worker roots against byte and entry limits.
///
/// # Errors
/// Rejects exceeded byte or entry caps (`InvalidInput`), reparse points
/// (`PermissionDenied`), depth over 16, or non-directory roots.
pub fn check_private_storage_roots(
    roots: &[PathBuf],
    max_bytes: u64,
    max_entries: usize,
) -> io::Result<(u64, usize)> {
    measure_private_roots(roots, max_bytes, max_entries)
}

fn normalize_root_component(component: Component<'_>) -> Option<String> {
    match component {
        Component::Prefix(prefix) => {
            Some(prefix.as_os_str().to_string_lossy().to_ascii_lowercase())
        }
        Component::RootDir => Some("\\".to_owned()),
        Component::Normal(part) => Some(part.to_string_lossy().to_ascii_lowercase()),
        Component::CurDir | Component::ParentDir => None,
    }
}

fn lexical_root_contains(parent: &Path, child: &Path) -> bool {
    let Some(parent_parts) = parent
        .components()
        .map(normalize_root_component)
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let Some(child_parts) = child
        .components()
        .map(normalize_root_component)
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    child_parts.len() >= parent_parts.len() && child_parts[..parent_parts.len()] == parent_parts[..]
}

fn validate_non_reparse_dir(path: &Path) -> io::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
        Ok(meta) => {
            if (meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT) != 0 || !meta.is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "private storage root must be an ordinary non-reparse directory",
                ));
            }
            Ok(true)
        }
    }
}

fn deduplicated_roots(roots: &[PathBuf]) -> io::Result<Vec<PathBuf>> {
    let mut active = Vec::with_capacity(roots.len());
    for root in roots {
        if !validate_non_reparse_dir(root)? {
            continue;
        }
        if active
            .iter()
            .any(|existing: &PathBuf| lexical_root_contains(existing, root))
        {
            continue;
        }
        active.retain(|existing| !lexical_root_contains(root, existing));
        active.push(root.clone());
    }
    Ok(active)
}

fn measure_private_roots(
    roots: &[PathBuf],
    max_bytes: u64,
    max_entries: usize,
) -> io::Result<(u64, usize)> {
    let scan_roots = deduplicated_roots(roots)?;
    let mut total_bytes = 0_u64;
    let mut total_entries = 0_usize;
    let mut stack = Vec::new();
    for root in scan_roots {
        stack.push((root, 0_usize));
        while let Some((dir, depth)) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            for entry_result in entries {
                let entry = match entry_result {
                    Ok(entry) => entry,
                    Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                    Err(error) => return Err(error),
                };
                let path = entry.path();
                let meta = match std::fs::symlink_metadata(&path) {
                    Ok(meta) => meta,
                    Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                    Err(error) => return Err(error),
                };
                if (meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT) != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "private storage must not contain symbolic links or reparse points",
                    ));
                }
                total_entries = total_entries
                    .checked_add(1)
                    .filter(|count| *count <= max_entries)
                    .ok_or_else(|| invalid("private storage entry count exceeded budget"))?;
                if meta.is_dir() {
                    let next_depth = depth
                        .checked_add(1)
                        .filter(|value| *value <= MAX_STORAGE_DEPTH)
                        .ok_or_else(|| invalid("private storage directory depth exceeded bound"))?;
                    stack.push((path, next_depth));
                } else {
                    total_bytes = total_bytes
                        .checked_add(meta.len())
                        .filter(|bytes| *bytes <= max_bytes)
                        .ok_or_else(|| invalid("private storage byte usage exceeded budget"))?;
                }
            }
        }
    }
    Ok((total_bytes, total_entries))
}

fn validate_process(spec: &ProcessSpec) -> io::Result<()> {
    validate_process_spec(spec)?;
    for (name, _) in std::env::vars_os() {
        validate_backend_environment_name(&name)?;
    }
    Ok(())
}

fn validate_process_spec(spec: &ProcessSpec) -> io::Result<()> {
    for path in [
        &spec.executable,
        &spec.working_directory,
        &spec.stdout,
        &spec.stderr,
    ] {
        validate_path(path)?;
    }
    if !spec.executable.is_file() || !spec.working_directory.is_dir() {
        return Err(invalid("executable or working directory does not exist"));
    }
    if spec.memory_limit_bytes == 0 || spec.active_process_limit == 0 {
        return Err(invalid(
            "job memory and active-process limits must be positive",
        ));
    }
    Ok(())
}

fn validate_backend_environment_name(name: &OsStr) -> io::Result<()> {
    if forbidden_environment_name(name) {
        return Err(invalid("inherited environment contains a backend override"));
    }
    Ok(())
}

#[cfg(feature = "containment")]
fn validate_sandboxed_environment_names(
    names: impl IntoIterator<Item = OsString>,
) -> io::Result<()> {
    for name in names {
        validate_backend_environment_name(&name)?;
        validate_sandbox_environment_name(&name)?;
    }
    Ok(())
}

/// Returns currently available physical memory; this is an observation only.
///
/// # Errors
/// Returns a failed Windows memory-status query.
pub fn available_memory_bytes() -> io::Result<u64> {
    native::available_memory_bytes()
}

/// Native physical and effective commit observations from `GlobalMemoryStatusEx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryStatus {
    /// SMBIOS-reported installed physical memory, if the native query succeeds.
    pub installed_physical_bytes: Option<u64>,
    /// OS-usable physical memory in bytes.
    pub total_physical_bytes: u64,
    /// Physical memory currently available to the system in bytes.
    pub available_physical_bytes: u64,
    /// Effective commit limit: the smaller current-process or system limit.
    pub commit_limit_bytes: u64,
    /// Commit bytes currently available to this process.
    pub available_commit_bytes: u64,
}

/// Returns a native memory snapshot without loading models or changing policy.
///
/// # Errors
/// Rejects a failed Windows memory query.
pub fn memory_status() -> io::Result<MemoryStatus> {
    native::memory_status()
}

/// Finds the process listening specifically on IPv4 `127.0.0.1:port`.
///
/// Wildcard and nonloopback listeners are not accepted. An ambiguous duplicate
/// listener is an error. This observation does not reserve or authenticate a port.
///
/// # Errors
/// Returns invalid port zero, malformed/oversized tables, ambiguity, or Windows
/// TCP-table query failures.
pub fn listener_process_id(port: u16) -> io::Result<Option<u32>> {
    if port == 0 {
        return Err(invalid("listener port must be nonzero"));
    }
    native::listener_process_id(port)
}

/// Requires exclusive Windows ownership of a socket's bound address.
///
/// Call this before binding a server socket or reserving a backend port. Without
/// exclusivity, some conflicting `SO_REUSEADDR` binds can share an
/// address and make delivery indeterminate. The socket stays borrowed;
/// this function does not bind it or take ownership of its handle.
///
/// # Errors
/// Returns the Winsock error if `SO_EXCLUSIVEADDRUSE` cannot be enabled.
pub fn require_exclusive_address(socket: &impl AsSocket) -> io::Result<()> {
    native::require_exclusive_address(socket)
}

/// Binds a numeric IPv4 loopback (`127.0.0.1`) TCP listener with `SO_EXCLUSIVEADDRUSE`
/// and the requested listen backlog using direct `WinSock` calls.
///
/// # Errors
/// Rejects negative backlogs or failed Winsock initialization, socket creation,
/// exclusive-address configuration, binding, or listening.
pub fn bind_exclusive_loopback_listener(
    port: u16,
    backlog: i32,
) -> io::Result<std::net::TcpListener> {
    if backlog <= 0 {
        return Err(invalid("listener backlog must be positive"));
    }
    native::bind_exclusive_loopback_listener(port, backlog)
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn starts_with_ignore_ascii_case(bytes: &[u8], prefix: &[u8]) -> bool {
    bytes
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

#[cfg(feature = "containment")]
fn ends_with_ignore_ascii_case(bytes: &[u8], suffix: &[u8]) -> bool {
    bytes.len().checked_sub(suffix.len()).is_some_and(|start| {
        bytes
            .get(start..)
            .is_some_and(|tail| tail.eq_ignore_ascii_case(suffix))
    })
}

fn forbidden_environment_name(name: &OsStr) -> bool {
    let bytes = name.as_encoded_bytes();
    [
        b"LLAMA_".as_slice(),
        b"LLAMACPP_".as_slice(),
        b"GGML_".as_slice(),
        b"AIP_".as_slice(),
        b"MTMD_".as_slice(),
    ]
    .iter()
    .any(|prefix| starts_with_ignore_ascii_case(bytes, prefix))
        || [
            b"HF_TOKEN".as_slice(),
            b"HF_ENDPOINT".as_slice(),
            b"MODEL_ENDPOINT".as_slice(),
        ]
        .iter()
        .any(|exact| bytes.eq_ignore_ascii_case(exact))
}

/// Checks ambient credential variable names without creating a native environment.
///
/// This cheap preflight never changes the caller's environment and is not proof
/// of PSEC availability or enforcement. Sandboxed launches that inherit caller
/// entries repeat this check and admit their complete native policy before
/// creating the worker. Minimal-environment launches instead admit only their
/// constructed child entries. Unknown inherited variable names can still contain
/// secrets.
///
/// # Errors
/// Returns `InvalidInput` when a recognized credential-shaped name is present.
#[cfg(feature = "containment")]
pub fn validate_sandbox_environment() -> io::Result<()> {
    std::env::vars_os().try_for_each(|(name, _)| validate_sandbox_environment_name(&name))
}

#[cfg(feature = "containment")]
fn validate_sandbox_environment_name(name: &OsStr) -> io::Result<()> {
    if credential_environment_name(name) {
        return Err(invalid(
            "sandbox launch rejects inherited credential environment variables",
        ));
    }
    Ok(())
}

#[cfg(feature = "containment")]
fn credential_environment_name(name: &OsStr) -> bool {
    let bytes = name.as_encoded_bytes();
    [
        b"_TOKEN".as_slice(),
        b"_SECRET".as_slice(),
        b"_PASSWORD".as_slice(),
        b"_API_KEY".as_slice(),
        b"_PRIVATE_KEY".as_slice(),
        b"_ACCESS_KEY".as_slice(),
        b"_CONNECTION_STRING".as_slice(),
        b"_CONNECTIONSTRING".as_slice(),
        b"_DATABASE_URL".as_slice(),
        b"_REDIS_URL".as_slice(),
    ]
    .iter()
    .any(|suffix| ends_with_ignore_ascii_case(bytes, suffix))
        || [
            b"TOKEN".as_slice(),
            b"SECRET".as_slice(),
            b"PASSWORD".as_slice(),
            b"API_KEY".as_slice(),
            b"PRIVATE_KEY".as_slice(),
            b"ACCESS_KEY".as_slice(),
            b"AWS_ACCESS_KEY_ID".as_slice(),
            b"AZURE_OPENAI_KEY".as_slice(),
            b"AZURE_STORAGE_KEY".as_slice(),
            b"AZURE_STORAGE_CONNECTION_STRING".as_slice(),
            b"GOOGLE_APPLICATION_CREDENTIALS".as_slice(),
            b"CONNECTION_STRING".as_slice(),
            b"CONNECTIONSTRING".as_slice(),
            b"DATABASE_URL".as_slice(),
            b"REDIS_URL".as_slice(),
        ]
        .iter()
        .any(|exact| bytes.eq_ignore_ascii_case(exact))
        || [
            b"CONNECTIONSTRINGS__".as_slice(),
            b"SQLCONNSTR_".as_slice(),
            b"SQLAZURECONNSTR_".as_slice(),
            b"MYSQLCONNSTR_".as_slice(),
            b"CUSTOMCONNSTR_".as_slice(),
            b"POSTGRESQLCONNSTR_".as_slice(),
        ]
        .iter()
        .any(|prefix| starts_with_ignore_ascii_case(bytes, prefix))
}

fn validate_path(path: &Path) -> io::Result<()> {
    // Inspect raw segments as Path::components normalizes embedded `.` segments.
    let text: Vec<u16> = path.as_os_str().encode_wide().collect();
    let ambiguous = text.split(|unit| *unit == 92 || *unit == 47).any(|part| {
        part == [46]
            || part == [46, 46]
            || part.last().is_some_and(|last| *last == 46 || *last == 32)
    });
    if !path.is_absolute() || text.contains(&0) || text.contains(&37) || ambiguous {
        return Err(invalid(
            "native paths must be expanded absolute paths without ambiguous segments",
        ));
    }
    // Device/verbatim paths can bypass ordinary Win32 file-name semantics.
    if !matches!(path.components().next(), Some(Component::Prefix(prefix)) if matches!(prefix.kind(), std::path::Prefix::Disk(_) | std::path::Prefix::UNC(_, _)))
    {
        return Err(invalid("native paths must use a drive or UNC prefix"));
    }
    for component in path.components() {
        if let Component::Normal(name) = component {
            // CreateFileW also opens DOS devices and NTFS alternate streams.
            // Evidence paths must name ordinary files, even when fully qualified.
            if name
                .encode_wide()
                .any(|unit| unit < 32 || matches!(unit, 34 | 42 | 58 | 60 | 62 | 63 | 124))
                || reserved_device(name)
            {
                return Err(invalid(
                    "native paths must name ordinary files and directories",
                ));
            }
        }
    }
    Ok(())
}

fn reserved_device(name: &OsStr) -> bool {
    let mut base = name
        .as_encoded_bytes()
        .split(|&byte| byte == b'.')
        .next()
        .unwrap_or_default();
    while let Some(trimmed) = base.strip_suffix(b" ") {
        base = trimmed;
    }
    [
        b"CON".as_slice(),
        b"PRN".as_slice(),
        b"AUX".as_slice(),
        b"NUL".as_slice(),
        b"CONIN$".as_slice(),
        b"CONOUT$".as_slice(),
    ]
    .iter()
    .any(|reserved| base.eq_ignore_ascii_case(reserved))
        || (starts_with_ignore_ascii_case(base, b"COM")
            || starts_with_ignore_ascii_case(base, b"LPT"))
            && base.get(3..).is_some_and(|suffix| {
                matches!(
                    suffix,
                    [b'1'..=b'9'] | [0xc2, 0xb9] | [0xc2, 0xb2] | [0xc2, 0xb3]
                )
            })
}

fn command_line(executable: &Path, arguments: &[OsString]) -> io::Result<Vec<u16>> {
    let mut output = Vec::new();
    for value in
        std::iter::once(executable.as_os_str()).chain(arguments.iter().map(OsString::as_os_str))
    {
        if !output.is_empty() {
            output.push(32);
        }
        quote(value, &mut output)?;
        if output.len() >= 32_767 {
            return Err(invalid("Windows command line exceeds 32767 UTF-16 units"));
        }
    }
    output.push(0);
    Ok(output)
}

fn tool_command_line(executable: &Path, arguments: &[OsString]) -> io::Result<Vec<u16>> {
    if !executable
        .file_name()
        .is_some_and(|name| name.eq_ignore_ascii_case("cmd.exe"))
    {
        return command_line(executable, arguments);
    }
    let [disable_auto_run, execute, script] = arguments else {
        return Err(invalid("unsupported command interpreter arguments"));
    };
    if disable_auto_run != "/d" || execute != "/c" {
        return Err(invalid("command interpreter startup must disable AutoRun"));
    }
    let script = script
        .to_str()
        .ok_or_else(|| invalid("command interpreter input must be Unicode"))?;
    if script.contains(['\0', '\r', '\n']) {
        return Err(invalid("unsupported command interpreter input"));
    }
    let command = if Path::new(script).is_absolute() {
        validate_path(Path::new(script))?;
        if script.contains('"') {
            return Err(invalid("unsupported environment batch path"));
        }
        format!(" /d /v:off /s /c \"\"{}\"\"", script.replace('/', "\\"))
    } else {
        format!(" /d /v:off /c {script}")
    };
    let mut output = Vec::new();
    quote(executable.as_os_str(), &mut output)?;
    output.extend(command.encode_utf16());
    if output.len() >= 32_767 {
        return Err(invalid("Windows command line exceeds 32767 UTF-16 units"));
    }
    output.push(0);
    Ok(output)
}

fn quote(value: &OsStr, output: &mut Vec<u16>) -> io::Result<()> {
    output.push(34);
    let mut slashes = 0;
    for unit in value.encode_wide() {
        match unit {
            0 => return Err(invalid("arguments cannot contain NUL")),
            92 => {
                slashes += 1;
                continue;
            }
            34 => output.extend(std::iter::repeat_n(92, slashes * 2 + 1)),
            _ => output.extend(std::iter::repeat_n(92, slashes)),
        }
        output.push(unit);
        slashes = 0;
    }
    output.extend(std::iter::repeat_n(92, slashes * 2));
    output.push(34);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "containment")]
    #[test]
    fn minimal_worker_admission_excludes_ambient_credentials_and_backend_overrides()
    -> io::Result<()> {
        // The minimal path passes no inherited names. Ordinary paths must still
        // reject every poisoned entry, without changing global process state.
        validate_sandboxed_environment_names(std::iter::empty())?;
        for name in [
            "OPENAI_API_KEY",
            "HF_TOKEN",
            "LLAMA_ARG_THREADS",
            "MODEL_ENDPOINT",
        ] {
            assert!(validate_sandboxed_environment_names([OsString::from(name)]).is_err());
        }
        validate_sandboxed_environment_names(
            [
                "SystemRoot",
                "windir",
                "SystemDrive",
                "LOCALAPPDATA",
                "TEMP",
                "TMP",
            ]
            .map(OsString::from),
        )
    }

    #[test]
    fn windows_argument_quoting() -> io::Result<()> {
        for (input, expected) in [
            ("", "\"\""),
            ("a b", "\"a b\""),
            ("a\\", "\"a\\\\\""),
            ("a\"b", "\"a\\\"b\""),
        ] {
            let mut output = Vec::new();
            quote(OsStr::new(input), &mut output)?;
            assert_eq!(String::from_utf16_lossy(&output), expected);
        }
        assert!(
            command_line(
                Path::new("C:\\probe.exe"),
                &[OsString::from("x".repeat(32_767))]
            )
            .is_err()
        );
        assert!(command_line(Path::new("C:\\probe.exe"), &[OsString::from("a\0b")]).is_err());
        Ok(())
    }

    #[test]
    fn path_and_environment_policy() {
        for path in [
            "relative",
            "C:relative",
            "C:\\a\\..\\b",
            "C:\\a\\.\\b",
            "C:\\a.\\b",
            "C:\\%TEMP%\\b",
            "\\\\?\\C:\\a",
        ] {
            assert!(validate_path(Path::new(path)).is_err(), "{path}");
        }
        assert!(validate_path(Path::new("C:\\native cwd\\probe.exe")).is_ok());
        for name in ["llama_arg", "GGML_CUDA", "HF_TOKEN", "model_endpoint"] {
            assert!(forbidden_environment_name(OsStr::new(name)));
        }
        for name in [
            "SystemDrive",
            "SystemRoot",
            "windir",
            "ProgramData",
            "LOCALAPPDATA",
            "APPDATA",
            "PATH",
            "HF_HOME",
        ] {
            assert!(!forbidden_environment_name(OsStr::new(name)));
        }
    }

    #[test]
    fn native_paths_reject_devices_streams_and_reserved_characters() {
        for name in [
            "NUL",
            "nul.txt",
            "NUL .txt",
            "Con",
            "PRN.log",
            "AUX",
            "CONIN$",
            "CONOUT$",
            "COM1",
            "com9.txt",
            "LPT1",
            "lpt9.log",
            "COM\u{b9}",
            "com\u{b2}.txt",
            "COM\u{b3}",
            "LPT\u{b9}",
            "lpt\u{b2}.txt",
            "LPT\u{b3}",
            "file.log:stream",
            "file.log::$DATA",
            "wild?card",
            "wild*card",
            "quoted\"name",
            "pipe|name",
            "less<name",
            "greater>name",
            "control\u{1f}name",
        ] {
            assert!(
                validate_path(&Path::new("C:\\evidence").join(name)).is_err(),
                "{name}"
            );
            assert!(
                validate_path(&Path::new("C:\\evidence").join(name).join("output.log")).is_err(),
                "{name}"
            );
        }
        for name in [
            "console.log",
            "nulled.txt",
            "COM10",
            "LPT10",
            ".config",
            "ordinary name",
        ] {
            assert!(
                validate_path(&Path::new("C:\\evidence").join(name)).is_ok(),
                "{name}"
            );
        }
    }
}
#[cfg(test)]
#[test]
fn failed_start_cleanup_marker_preserves_kind_display_and_source() {
    for kind in [
        io::ErrorKind::TimedOut,
        io::ErrorKind::Other,
        io::ErrorKind::InvalidInput,
        io::ErrorKind::Unsupported,
    ] {
        let original = io::Error::new(kind, "native cleanup failure");
        assert!(!process_start_cleanup_unconfirmed(&original));
        let marked = unconfirmed_start_cleanup(original);
        assert!(process_start_cleanup_unconfirmed(&marked));
        assert_eq!(marked.kind(), kind);
        assert_eq!(marked.to_string(), "native cleanup failure");
        assert_eq!(
            marked
                .get_ref()
                .and_then(|source| source.source())
                .map(ToString::to_string)
                .as_deref(),
            Some("native cleanup failure")
        );
    }
}

#[cfg(all(test, feature = "containment"))]
#[test]
fn sandbox_environment_classifies_credential_names_without_changing_environment() {
    for name in [
        "OPENAI_API_KEY",
        "openai_api_key",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_KEY",
        "AZURE_STORAGE_CONNECTION_STRING",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "service_TOKEN",
        "service_SECRET",
        "service_PASSWORD",
        "service_API_KEY",
        "service_PRIVATE_KEY",
        "service_ACCESS_KEY",
        "TOKEN",
        "PASSWORD",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "DATABASE_URL",
        "REDIS_URL",
        "APP_DATABASE_URL",
        "APP_REDIS_URL",
        "APP_CONNECTION_STRING",
        "APP_CONNECTIONSTRING",
        "ConnectionStrings__DefaultConnection",
        "SQLCONNSTR_Default",
        "SQLAZURECONNSTR_Default",
        "MYSQLCONNSTR_Default",
        "CUSTOMCONNSTR_Default",
        "POSTGRESQLCONNSTR_Default",
    ] {
        assert!(credential_environment_name(OsStr::new(name)), "{name}");
    }
    for name in [
        "SystemDrive",
        "SystemRoot",
        "windir",
        "ProgramData",
        "LOCALAPPDATA",
        "APPDATA",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "PATH",
        "TEMP",
        "TMP",
        "ComSpec",
        "ProgramFiles",
        "HF_HOME",
        "TOKENIZERS_PARALLELISM",
        "PASSWORD_POLICY",
    ] {
        assert!(!credential_environment_name(OsStr::new(name)), "{name}");
    }
}

#[cfg(test)]
mod storage_budget_tests {
    use super::*;
    use std::fs;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempTree(PathBuf);

    impl Drop for TempTree {
        fn drop(&mut self) {
            if let Ok(entries) = fs::read_dir(&self.0) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Ok(meta) = fs::symlink_metadata(&path)
                        && (meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT) != 0
                    {
                        let _ = fs::remove_dir(&path);
                    }
                }
            }
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn temp_tree(label: &str) -> io::Result<TempTree> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "Bonsai-StorageBudget-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(TempTree(path))
    }

    #[test]
    fn private_storage_budget_tracks_missing_nested_bytes_entries_depth_and_reparse()
    -> io::Result<()> {
        let missing = std::env::temp_dir().join("Bonsai-Nonexistent-Storage-Root-For-Budget-Test");
        assert_eq!(measure_private_roots(&[missing], 1024, 16)?, (0, 0));

        let tree = temp_tree("caps")?;
        let root = &tree.0;
        let nested = root.join("sub");
        fs::create_dir(&nested)?;
        fs::write(nested.join("a.bin"), vec![1_u8; 64])?;
        // Duplicate or ancestor-contained roots must not double count entries or bytes.
        assert_eq!(
            measure_private_roots(&[root.clone(), nested.clone(), root.clone()], 128, 8)?,
            (64, 2)
        );

        // Byte cap rejection
        let err = measure_private_roots(std::slice::from_ref(root), 32, 8).err();
        assert_eq!(
            err.as_ref().map(io::Error::kind),
            Some(io::ErrorKind::InvalidInput)
        );

        // Entry count rejection
        let err = measure_private_roots(std::slice::from_ref(root), 1024, 1).err();
        assert_eq!(
            err.as_ref().map(io::Error::kind),
            Some(io::ErrorKind::InvalidInput)
        );

        // Depth > MAX_STORAGE_DEPTH rejection
        let deep = temp_tree("deep")?;
        let mut cursor = deep.0.clone();
        for level in 0..=MAX_STORAGE_DEPTH {
            cursor = cursor.join(format!("d{level}"));
            fs::create_dir(&cursor)?;
        }
        let err = measure_private_roots(std::slice::from_ref(&deep.0), 1024, 1024).err();
        assert_eq!(
            err.as_ref().map(io::Error::kind),
            Some(io::ErrorKind::InvalidInput)
        );

        // Directory junction (reparse point) rejection without following into target
        let target = temp_tree("target")?;
        fs::write(target.0.join("secret.txt"), b"outside")?;
        let junction_parent = temp_tree("junction")?;
        let junction_path = junction_parent.0.join("link");
        let comspec =
            std::env::var_os("ComSpec").ok_or_else(|| io::Error::other("missing ComSpec"))?;
        let status = Command::new(comspec)
            .arg("/d")
            .arg("/c")
            .arg("mklink")
            .arg("/J")
            .arg(&junction_path)
            .arg(&target.0)
            .status()?;
        assert!(status.success(), "mklink /J failed in test fixture");
        let err = measure_private_roots(std::slice::from_ref(&junction_parent.0), 1024, 16).err();
        assert_eq!(
            err.as_ref().map(io::Error::kind),
            Some(io::ErrorKind::PermissionDenied)
        );
        Ok(())
    }

    #[test]
    fn listener_process_id_reports_owned_loopback_socket_and_closed_port() -> io::Result<()> {
        assert_eq!(
            listener_process_id(0).err().map(|err| err.kind()),
            Some(io::ErrorKind::InvalidInput)
        );
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
        let port = listener.local_addr()?.port();
        assert_eq!(listener_process_id(port)?, Some(std::process::id()));
        drop(listener);
        assert_eq!(listener_process_id(port)?, None);
        Ok(())
    }

    #[test]
    fn mapped_file_maps_regular_file_and_rejects_empty_and_directories() -> io::Result<()> {
        let tree = temp_tree("mmap")?;
        let data_path = tree.0.join("weights.bin");
        let payload: Vec<u8> = (0..=255_u8).cycle().take(4096 + 17).collect();
        fs::write(&data_path, &payload)?;
        let mapped = MappedFile::open(&data_path)?;
        assert_eq!(mapped.len(), payload.len());
        assert!(!mapped.is_empty());
        assert_eq!(mapped.as_slice(), payload.as_slice());

        let opened = fs::File::open(&data_path)?;
        let no_prefetch = MappedFile::from_file_without_prefetch(&opened)?;
        assert_eq!(no_prefetch.as_slice(), payload.as_slice());

        let empty_path = tree.0.join("empty.bin");
        fs::write(&empty_path, b"")?;
        assert!(MappedFile::open(&empty_path).is_err());
        let empty_file = fs::File::open(&empty_path)?;
        assert!(MappedFile::from_file_without_prefetch(&empty_file).is_err());

        let dir_handle = fs::File::open(&tree.0);
        if let Ok(file) = dir_handle {
            assert!(MappedFile::from_file(&file).is_err());
        }
        Ok(())
    }

    #[test]
    fn row_worker_pool_executes_aligned_and_nested_tasks_deterministically() -> io::Result<()> {
        let pool = RowWorkerPool::new(4, "test-row-pool")?;
        assert_eq!(pool.worker_count(), 4);
        let mut out = vec![0.0_f32; 65];
        pool.run_aligned(&mut out, 4, 8, &|offset, slice| {
            for (idx, slot) in slice.iter_mut().enumerate() {
                let global = offset + idx;
                *slot = u16::try_from(global).map_or(0.0, f32::from);
            }
        });
        for (idx, &val) in out.iter().enumerate() {
            assert_eq!(val, u16::try_from(idx).map_or(0.0, f32::from));
        }

        // Verify nested pool invocation runs inline on the worker without deadlock.
        let mut outer = vec![0.0_f32; 16];
        pool.run_aligned(&mut outer, 4, 4, &|offset, slice| {
            let mut inner = vec![0.0_f32; slice.len()];
            pool.run(&mut inner, 4, &|inner_off, inner_slice| {
                for (i, slot) in inner_slice.iter_mut().enumerate() {
                    *slot = u16::try_from(offset + inner_off + i).map_or(0.0, f32::from) + 1.0;
                }
            });
            slice.copy_from_slice(&inner);
        });
        for (idx, &val) in outer.iter().enumerate() {
            assert_eq!(val, u16::try_from(idx).map_or(0.0, f32::from) + 1.0);
        }

        let mut fallible = vec![0.0_f32; 32];
        pool.try_run_aligned(&mut fallible, 4, 8, &|offset, slice| {
            for (idx, slot) in slice.iter_mut().enumerate() {
                *slot = u16::try_from(offset + idx).map_or(0.0, f32::from);
            }
            Ok(())
        })?;
        for (idx, &val) in fallible.iter().enumerate() {
            assert_eq!(val, u16::try_from(idx).map_or(0.0, f32::from));
        }

        let err = pool.try_run_aligned(&mut fallible, 4, 8, &|offset, _slice| {
            if offset >= 8 {
                return Err(io::Error::other(format!("fail-{offset}")));
            }
            Ok(())
        });
        assert_eq!(err.err().map(|e| e.to_string()), Some("fail-8".to_owned()));

        #[derive(Debug, Default, PartialEq, Eq)]
        struct Item {
            index: usize,
            label: String,
        }
        let mut items: Vec<Item> = (0..19).map(|_| Item::default()).collect();
        pool.run_slice_aligned(&mut items, 4, 3, &|offset, slice| {
            for (i, item) in slice.iter_mut().enumerate() {
                let g = offset + i;
                item.index = g;
                item.label = format!("item-{g}");
            }
        });
        for (idx, item) in items.iter().enumerate() {
            assert_eq!(item.index, idx);
            assert_eq!(item.label, format!("item-{idx}"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod maintenance_tests {
    use super::*;

    #[test]
    fn current_identity_retains_the_running_process() -> io::Result<()> {
        let created = current_process_creation_time()?;
        assert_ne!(created, 0);
        ParentProcess::open(std::process::id(), created)?.ensure_alive()?;
        assert!(ParentProcess::open(std::process::id(), created + 1).is_err());
        Ok(())
    }

    #[test]
    fn observations_preserve_canonical_time_and_reject_nonlocal_storage() -> io::Result<()> {
        let stamp = utc_now();
        assert_eq!(stamp.len(), 28);
        for (index, expected) in [
            (4, b'-'),
            (7, b'-'),
            (10, b'T'),
            (13, b':'),
            (16, b':'),
            (19, b'.'),
            (27, b'Z'),
        ] {
            assert_eq!(stamp.as_bytes()[index], expected);
        }
        for path in [
            "relative",
            "C:relative",
            "C:\\",
            "C:\\\\",
            "\\\\server.invalid\\share",
            "//?/C:/Bonsai",
        ] {
            assert!(require_local_data_drive(Path::new(path)).is_err());
        }
        require_local_data_drive(&std::env::temp_dir())?;
        assert!(available_disk_bytes(&std::env::temp_dir())? > 0);
        assert!(available_disk_bytes(Path::new("C:\\nul\0tail")).is_err());
        Ok(())
    }

    #[test]
    fn shared_clock_and_process_snapshot_observe_the_live_test_process() -> io::Result<()> {
        let (first, frequency) = monotonic_ticks()?;
        let (second, next_frequency) = monotonic_ticks()?;
        assert!(frequency > 0 && second >= first);
        assert_eq!(frequency, next_frequency);
        let executable = std::env::current_exe()?;
        let name = executable
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| io::Error::other("Missing test executable name."))?;
        assert!(has_running_process(&[name])?);
        assert!(has_running_process(&[""]).is_err());
        Ok(())
    }
}
