//! Bounded read-only Vulkan queries. The caller owns the driver-call deadline.

use std::ffi::{CStr, FromBytesUntilNulError, OsString};
use std::io;
use std::os::windows::ffi::OsStringExt as _;
use std::path::PathBuf;
use std::ptr::{null, null_mut};

use ash::vk::{self, Handle as _};

use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::GetProcAddress;

use super::library::SystemLibrary;
use crate::{
    VulkanAdapter, VulkanInstanceModules, VulkanInventory, VulkanModuleFileCandidate,
    VulkanQueryFailure, WindowsGraphicsAdapterCount,
};

const MAX_ADAPTERS: usize = 16;
const MAX_QUEUES: usize = 64;
const MAX_EXTENSIONS: usize = 1024;
const MAX_MODULES: usize = 256;
const MAX_MODULE_NAME: usize = 260;

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "unsupported Vulkan inventory")
}

fn failure(
    stage: &'static str,
    vulkan_result: Option<i32>,
    loader_api_version: Option<u32>,
    application_api_version: Option<u32>,
    kind: io::ErrorKind,
) -> io::Error {
    io::Error::new(
        kind,
        VulkanQueryFailure {
            stage,
            vulkan_result,
            loader_api_version,
            application_api_version,
            instance_modules: None,
            graphics_adapter_count: None,
        },
    )
}

fn module_name(buffer: &[u16], count: usize) -> Option<String> {
    // A full buffer may be truncated. Reject malformed UTF-16 and names which
    // could expose paths, embed controls, or collapse distinct observations.
    if count == 0 || count >= buffer.len() {
        return None;
    }
    let name = String::from_utf16(buffer.get(..count)?).ok()?;
    if name
        .chars()
        .any(|character| character.is_control() || matches!(character, '/' | '\\' | ':'))
    {
        return None;
    }
    Some(name.to_ascii_lowercase())
}

fn file_candidate_indices<T>(before: Option<&[String]>, modules: &[(String, T)]) -> Vec<usize> {
    modules
        .iter()
        .enumerate()
        .filter(|(_, (name, _))| name == "vulkan-1.dll")
        .chain(modules.iter().enumerate().filter(|(_, (name, _))| {
            name != "vulkan-1.dll"
                && before.is_some_and(|before| before.binary_search(name).is_err())
        }))
        .take(16)
        .map(|(index, _)| index)
        .collect()
}

fn module_snapshot(
    before: Option<&[String]>,
    capture_paths: bool,
) -> Option<(Vec<String>, Vec<VulkanModuleFileCandidate>)> {
    use windows_sys::Win32::System::{
        ProcessStatus::{K32EnumProcessModules, K32GetModuleBaseNameW, K32GetModuleFileNameExW},
        Threading::GetCurrentProcess,
    };

    let mut modules = [null_mut(); MAX_MODULES];
    let bytes = u32::try_from(size_of_val(&modules)).ok()?;
    let mut needed = 0_u32;
    // SAFETY: This pseudo-handle denotes only the current child, needs no close,
    // and is never exposed. Both output buffers remain writable for the call.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: The initialized module array has exactly the advertised byte size;
    // enumeration writes no more than that size and retains no storage pointers.
    if unsafe { K32EnumProcessModules(process, modules.as_mut_ptr(), bytes, &raw mut needed) } == 0
        || needed == 0
        || needed > bytes
        || !usize::try_from(needed)
            .ok()?
            .is_multiple_of(size_of_val(&modules[0]))
    {
        return None;
    }
    let count = usize::try_from(needed).ok()? / size_of_val(&modules[0]);
    let mut names = Vec::with_capacity(count);
    for module in modules.get(..count)? {
        if module.is_null() {
            return None;
        }
        let mut buffer = [0_u16; MAX_MODULE_NAME];
        // SAFETY: Module handles are opaque values from this process snapshot;
        // Win32 validates them. Unload races fail the entire optional snapshot.
        // The writable name buffer has the advertised UTF-16 element count.
        let length = unsafe { K32GetModuleBaseNameW(process, *module, buffer.as_mut_ptr(), 260) };
        names.push((
            module_name(&buffer, usize::try_from(length).ok()?)?,
            *module,
        ));
    }
    // The loader is audited first; remaining candidates are newly observed
    // basenames. Unknown before-snapshots permit only the known system loader.
    names.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let mut candidates = Vec::new();
    if capture_paths {
        for index in file_candidate_indices(before, &names) {
            let (name, module) = &names[index];
            let mut stack = [0_u16; 260];
            // SAFETY: The process/opaque module handle came from this snapshot;
            // Win32 validates unload races. The buffer has its advertised size.
            let length =
                unsafe { K32GetModuleFileNameExW(process, *module, stack.as_mut_ptr(), 260) };
            let path = usize::try_from(length).ok().and_then(|length| {
                let path = if length > 0 && length < stack.len() {
                    Some(PathBuf::from(OsString::from_wide(&stack[..length])))
                } else if length >= stack.len() - 1 {
                    let mut heap = vec![0_u16; 32768];
                    // SAFETY: Same live snapshot module handle and bounded 32768-unit heap buffer.
                    let full = unsafe {
                        K32GetModuleFileNameExW(process, *module, heap.as_mut_ptr(), 32768)
                    };
                    usize::try_from(full)
                        .ok()
                        .filter(|full| *full > 0 && *full < heap.len())
                        .map(|full| PathBuf::from(OsString::from_wide(&heap[..full])))
                } else {
                    None
                };
                path.filter(|path| path.is_absolute())
            });
            candidates.push(VulkanModuleFileCandidate {
                name: name.clone(),
                path,
            });
        }
    }
    let mut basenames: Vec<_> = names.into_iter().map(|(name, _)| name).collect();
    basenames.dedup();
    Some((basenames, candidates))
}

fn check(result: vk::Result) -> io::Result<()> {
    if result == vk::Result::SUCCESS {
        Ok(())
    } else {
        Err(failure(
            "device-query",
            Some(result.as_raw()),
            None,
            None,
            io::ErrorKind::Other,
        ))
    }
}

fn text(c_str: Result<&CStr, FromBytesUntilNulError>) -> io::Result<String> {
    let text = c_str
        .map_err(|_| invalid())?
        .to_str()
        .map_err(|_| invalid())?;
    if text.chars().any(char::is_control) {
        return Err(invalid());
    }
    Ok(text.to_owned())
}

struct Loader {
    entry: ash::Entry,
    _library: SystemLibrary,
}

impl Loader {
    fn load() -> io::Result<Self> {
        let library =
            SystemLibrary::load_system32("vulkan-1.dll", "system Vulkan loader unavailable")?;
        let proc_addr = library
            .symbol(c"vkGetInstanceProcAddr")
            .map_err(|_| io::Error::other("system Vulkan loader unavailable"))?;
        // SAFETY: `vkGetInstanceProcAddr` has the exact Khronos Vulkan 1.0 ABI.
        let get_instance_proc_addr = unsafe {
            std::mem::transmute::<unsafe extern "system" fn() -> isize, vk::PFN_vkGetInstanceProcAddr>(
                proc_addr,
            )
        };
        // SAFETY: Global loader command lookup uses a null VkInstance and NUL-terminated name.
        if unsafe { get_instance_proc_addr(vk::Instance::null(), c"vkCreateInstance".as_ptr()) }
            .is_none()
        {
            return Err(io::Error::other("system Vulkan loader unavailable"));
        }
        // SAFETY: `get_instance_proc_addr` has been resolved from the live
        // `vulkan-1.dll`, `vkCreateInstance` is verified non-null, and `library`
        // outlives `entry`.
        let entry = unsafe {
            ash::Entry::from_static_fn(ash::StaticFn {
                get_instance_proc_addr,
            })
        };
        Ok(Self {
            entry,
            _library: library,
        })
    }

    fn instance_version(&self) -> Result<u32, i32> {
        // SAFETY: `try_enumerate_instance_version` probes `vkEnumerateInstanceVersion`
        // with a null instance and falls back to `Ok(None)` on Vulkan 1.0 loaders.
        match unsafe { self.entry.try_enumerate_instance_version() } {
            Ok(Some(version)) => Ok(version),
            Ok(None) => Ok(vk::API_VERSION_1_0),
            Err(status) => Err(status.as_raw()),
        }
    }

    fn has_instance_symbol(&self, instance: vk::Instance, name: &CStr) -> io::Result<()> {
        let get_instance_proc_addr = self.entry.static_fn().get_instance_proc_addr;
        // SAFETY: Valid live VkInstance and NUL-terminated Vulkan command name.
        let symbol = unsafe { get_instance_proc_addr(instance, name.as_ptr()) };
        symbol.ok_or_else(invalid).map(|_| ())
    }
}

struct Instance(ash::Instance);

const fn graphics_count_result(status: i32, count: u32) -> WindowsGraphicsAdapterCount {
    WindowsGraphicsAdapterCount {
        stage: "count-query",
        windows_error: None,
        ntstatus: Some(status),
        // Microsoft specifies that NumAdapters is undefined on error. Retain
        // no output count unless the call returned exactly STATUS_SUCCESS.
        adapter_count: if status == 0 { Some(count) } else { None },
    }
}

fn graphics_adapter_count() -> WindowsGraphicsAdapterCount {
    use windows_sys::{
        Wdk::Graphics::Direct3D::D3DKMT_ENUMADAPTERS2,
        Win32::{Foundation::GetLastError, System::LibraryLoader::GetModuleHandleExW},
    };

    type Count = unsafe extern "system" fn(*mut D3DKMT_ENUMADAPTERS2) -> i32;
    struct ExistingModule(HMODULE);
    impl Drop for ExistingModule {
        fn drop(&mut self) {
            // SAFETY: GetModuleHandleExW acquired exactly this one reference;
            // the count function pointer is no longer used when it is released.
            let _ = unsafe { FreeLibrary(self.0) };
        }
    }

    let mut report = WindowsGraphicsAdapterCount {
        stage: "existing-gdi32-module",
        windows_error: None,
        ntstatus: None,
        adapter_count: None,
    };
    let mut handle = null_mut();
    // SAFETY: The fixed name is NUL terminated and the handle output is live.
    // This takes a reference only to an existing module, without loading one.
    if unsafe { GetModuleHandleExW(0, windows_sys::w!("gdi32.dll"), &raw mut handle) } == 0 {
        // SAFETY: Capture this thread's lookup error immediately, before any
        // other native call can replace it. A failed lookup does not prove absence.
        report.windows_error = Some(unsafe { GetLastError() });
        return report;
    }
    let module = ExistingModule(handle);
    report.stage = "enum-adapters-entry-point";
    // SAFETY: The acquired reference keeps the existing module live; the fixed
    // symbol is NUL terminated and used only before that reference is released.
    let Some(function) =
        (unsafe { GetProcAddress(module.0, c"D3DKMTEnumAdapters2".as_ptr().cast()) })
    else {
        // SAFETY: Capture the export lookup's error before releasing the module
        // reference or making any other native call.
        report.windows_error = Some(unsafe { GetLastError() });
        return report;
    };
    // SAFETY: The fixed Windows export has this documented system ABI, using
    // the generated SDK structure and the NTSTATUS i32 return representation.
    let count =
        unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, Count>(function) };
    let mut request = D3DKMT_ENUMADAPTERS2::default();
    // SAFETY: This live writable structure has a null pAdapters field: only
    // the supported count is requested, without an adapter array or handles.
    // One call is made under the same owning child deadline and mitigations.
    let status = unsafe { count(&raw mut request) };
    graphics_count_result(status, request.NumAdapters)
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns the instance, creates no children,
        // and is dropped before the Loader providing its function pointers.
        unsafe { self.0.destroy_instance(None) };
    }
}

fn create_instance(
    loader: &Loader,
    loader_api: u32,
    application_api: u32,
) -> io::Result<(Instance, VulkanInstanceModules)> {
    let application = vk::ApplicationInfo::default()
        .application_name(c"bonsai-inventory")
        .api_version(application_api);
    let create = vk::InstanceCreateInfo::default().application_info(&application);
    let before = module_snapshot(None, false).map(|(names, _)| names);
    let mut raw_instance = vk::Instance::null();
    let create_fn = loader.entry.fp_v1_0().create_instance;
    // SAFETY: Initialized create structures have live pointers; no layers or
    // extensions are requested. The instance is destroyed before its Loader.
    let status = unsafe { create_fn(&raw const create, null(), &raw mut raw_instance) };
    let after = module_snapshot(before.as_deref(), true);
    let (after, file_candidates) = after.map_or((None, None), |(names, candidates)| {
        (Some(names), Some(candidates))
    });
    let modules = VulkanInstanceModules {
        before,
        after,
        file_candidates,
    };
    if status != vk::Result::SUCCESS || raw_instance.is_null() {
        return Err(io::Error::other(VulkanQueryFailure {
            stage: "instance-creation",
            vulkan_result: Some(status.as_raw()),
            loader_api_version: Some(loader_api),
            application_api_version: Some(application_api),
            instance_modules: Some(modules),
            graphics_adapter_count: Some(graphics_adapter_count()),
        }));
    }
    loader.has_instance_symbol(raw_instance, c"vkDestroyInstance")?;
    // SAFETY: `raw_instance` is a live owned VkInstance and `vkDestroyInstance`
    // was verified non-null; remaining required symbols are verified before use.
    let instance = Instance(unsafe { ash::Instance::load(loader.entry.static_fn(), raw_instance) });
    for name in [
        c"vkEnumeratePhysicalDevices",
        c"vkGetPhysicalDeviceProperties",
        c"vkEnumerateDeviceExtensionProperties",
        c"vkGetPhysicalDeviceQueueFamilyProperties",
        c"vkGetPhysicalDeviceMemoryProperties",
    ] {
        loader.has_instance_symbol(instance.0.handle(), name)?;
    }
    if application_api >= vk::API_VERSION_1_1 {
        for name in [
            c"vkGetPhysicalDeviceFeatures2",
            c"vkGetPhysicalDeviceProperties2",
        ] {
            loader.has_instance_symbol(instance.0.handle(), name)?;
        }
    }
    Ok((instance, modules))
}

const fn device_type(kind: vk::PhysicalDeviceType) -> (&'static str, bool) {
    match kind {
        vk::PhysicalDeviceType::INTEGRATED_GPU => ("integrated-gpu", true),
        vk::PhysicalDeviceType::DISCRETE_GPU => ("discrete-gpu", true),
        vk::PhysicalDeviceType::VIRTUAL_GPU => ("virtual-gpu", false),
        vk::PhysicalDeviceType::CPU => ("cpu", false),
        _ => ("other", false),
    }
}

#[derive(Default)]
struct Features {
    float16: Option<bool>,
    integer_dot: Option<bool>,
    subgroup_size: Option<u32>,
    subgroup_operations: Option<u32>,
    subgroup_stages: Option<u32>,
}

fn feature_structures(
    application_api: u32,
    device_api: u32,
    extensions: &[String],
) -> Option<(bool, bool)> {
    let effective_api = application_api.min(device_api);
    (effective_api >= vk::API_VERSION_1_1).then(|| {
        (
            effective_api >= vk::API_VERSION_1_2
                || extensions
                    .iter()
                    .any(|name| name == "VK_KHR_shader_float16_int8"),
            effective_api >= vk::API_VERSION_1_3
                || extensions
                    .iter()
                    .any(|name| name == "VK_KHR_shader_integer_dot_product"),
        )
    })
}

fn features(
    instance: &Instance,
    device: vk::PhysicalDevice,
    api: u32,
    extensions: &[String],
    application_api: u32,
) -> Features {
    let Some((supports_half, supports_dot)) = feature_structures(application_api, api, extensions)
    else {
        return Features::default();
    };
    let mut half = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2::default();
    if supports_half {
        features = features.push_next(&mut half);
    }
    if supports_dot {
        features = features.push_next(&mut dot);
    }
    // SAFETY: Vulkan 1.1 query with a valid live device and verified 1.1 entry
    // point. Only supported core or advertised extension structures are chained.
    unsafe {
        instance
            .0
            .get_physical_device_features2(device, &mut features)
    };
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    // SAFETY: Both structures are supported core 1.1 structures and live
    // throughout the read-only properties query against this valid device.
    unsafe {
        instance
            .0
            .get_physical_device_properties2(device, &mut properties2)
    };
    Features {
        float16: supports_half.then_some(half.shader_float16 != 0),
        integer_dot: supports_dot.then_some(dot.shader_integer_dot_product != 0),
        subgroup_size: Some(subgroup.subgroup_size),
        subgroup_operations: Some(subgroup.supported_operations.as_raw()),
        subgroup_stages: Some(subgroup.supported_stages.as_raw()),
    }
}

fn adapter(
    instance: &Instance,
    device: vk::PhysicalDevice,
    application_api: u32,
) -> io::Result<VulkanAdapter> {
    // SAFETY: device was enumerated from this live instance; the read-only
    // query initializes `properties` and retains no borrowed storage.
    let properties = unsafe { instance.0.get_physical_device_properties(device) };
    let mut extensions = vec![vk::ExtensionProperties::default(); MAX_EXTENSIONS];
    let mut count = u32::try_from(extensions.len()).map_err(|_| invalid())?;
    let enumerate_extensions = instance.0.fp_v1_0().enumerate_device_extension_properties;
    // SAFETY: Valid instance/device, no layer filter, initialized output storage
    // for count entries. The driver may only write that many entries.
    check(unsafe {
        enumerate_extensions(device, null(), &raw mut count, extensions.as_mut_ptr())
    })?;
    let count = usize::try_from(count).map_err(|_| invalid())?;
    if count > extensions.len() {
        return Err(invalid());
    }
    extensions.truncate(count);
    let mut extensions: Vec<_> = extensions
        .iter()
        .map(|extension| text(extension.extension_name_as_c_str()))
        .collect::<io::Result<_>>()?;
    extensions.sort_unstable();

    let mut queues = [vk::QueueFamilyProperties::default(); MAX_QUEUES];
    let mut queue_count = 0;
    let get_queue_families = instance
        .0
        .fp_v1_0()
        .get_physical_device_queue_family_properties;
    // SAFETY: Valid device and writable count; null storage requests only count.
    unsafe { get_queue_families(device, &raw mut queue_count, null_mut()) };
    if usize::try_from(queue_count).map_err(|_| invalid())? > queues.len() {
        return Err(invalid());
    }
    // SAFETY: The bounded initialized output array accommodates the queried
    // count. Vulkan writes at most the input count and retains no pointers.
    unsafe { get_queue_families(device, &raw mut queue_count, queues.as_mut_ptr()) };
    let queue_count = usize::try_from(queue_count).map_err(|_| invalid())?;
    let compute_queues = queues
        .get(..queue_count)
        .ok_or_else(invalid)?
        .iter()
        .enumerate()
        .filter(|(_, queue)| queue.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|(index, queue)| u32::try_from(index).map(|index| (index, queue.queue_count)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid())?;

    // SAFETY: Valid device from this live instance. Returned fixed-size memory
    // properties are fully initialized by Vulkan and do not borrow driver data.
    let memory = unsafe { instance.0.get_physical_device_memory_properties(device) };
    let heaps = usize::try_from(memory.memory_heap_count).map_err(|_| invalid())?;
    let memory_heaps = memory
        .memory_heaps
        .get(..heaps)
        .ok_or_else(invalid)?
        .iter()
        .map(|heap| {
            (
                heap.size,
                heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
            )
        })
        .collect();

    let features = features(
        instance,
        device,
        properties.api_version,
        &extensions,
        application_api,
    );
    let (device_type, hardware_gpu) = device_type(properties.device_type);
    Ok(VulkanAdapter {
        name: text(properties.device_name_as_c_str())?,
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        driver_version: properties.driver_version,
        api_version: properties.api_version,
        device_type,
        hardware_gpu,
        compute_queues,
        extensions,
        memory_heaps,
        shader_float16: features.float16,
        shader_integer_dot_product: features.integer_dot,
        subgroup_size: features.subgroup_size,
        subgroup_operations: features.subgroup_operations,
        subgroup_stages: features.subgroup_stages,
    })
}

pub(super) fn query() -> io::Result<VulkanInventory> {
    if !cfg!(target_arch = "aarch64")
        || std::env::vars_os().any(|(name, _)| {
            name.as_encoded_bytes()
                .get(..3)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"VK_"))
        })
    {
        return Err(failure(
            "eligibility",
            None,
            None,
            None,
            io::ErrorKind::Unsupported,
        ));
    }
    let loader =
        Loader::load().map_err(|error| failure("system-loader", None, None, None, error.kind()))?;
    let loader_api_version = loader.instance_version().map_err(|code| {
        failure(
            "loader-version",
            Some(code),
            None,
            None,
            io::ErrorKind::Other,
        )
    })?;
    let application_api = loader_api_version.min(vk::API_VERSION_1_3);
    let (instance, instance_modules) =
        create_instance(&loader, loader_api_version, application_api)?;
    let mut devices = [vk::PhysicalDevice::null(); MAX_ADAPTERS];
    let mut count = u32::try_from(devices.len()).map_err(|_| invalid())?;
    let handle = instance.0.handle();
    let enumerate_devices = instance.0.fp_v1_0().enumerate_physical_devices;
    // SAFETY: Valid instance and initialized storage for the advertised count.
    // SUCCESS is required; INCOMPLETE rejects rather than growing without bound.
    let result = unsafe { enumerate_devices(handle, &raw mut count, devices.as_mut_ptr()) };
    if result != vk::Result::SUCCESS {
        return Err(failure(
            "physical-device-enumeration",
            Some(result.as_raw()),
            Some(loader_api_version),
            Some(application_api),
            io::ErrorKind::Other,
        ));
    }
    let count = usize::try_from(count).map_err(|_| invalid())?;
    let adapters = devices
        .get(..count)
        .ok_or_else(invalid)?
        .iter()
        .map(|device| {
            adapter(&instance, *device, application_api).map_err(|error| {
                let result = error
                    .get_ref()
                    .and_then(|error| error.downcast_ref::<VulkanQueryFailure>())
                    .and_then(|failure| failure.vulkan_result);
                failure(
                    "adapter-properties",
                    result,
                    Some(loader_api_version),
                    Some(application_api),
                    error.kind(),
                )
            })
        })
        .collect::<io::Result<_>>()?;
    Ok(VulkanInventory {
        loader_api_version,
        application_api_version: application_api,
        instance_modules,
        adapters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_file_candidates_prioritize_loader_and_bound_new_images() {
        let mut modules: Vec<_> = (0..32)
            .map(|index| (format!("driver-{index:02}.dll"), ()))
            .collect();
        modules.push(("vulkan-1.dll".into(), ()));
        let before = vec!["driver-00.dll".into(), "vulkan-1.dll".into()];
        let selected = file_candidate_indices(Some(&before), &modules);
        assert_eq!(selected.len(), 16);
        assert_eq!(selected[0], 32);
        assert!(!selected.contains(&0));
        assert_eq!(file_candidate_indices(None, &modules), vec![32]);
        assert_eq!(
            file_candidate_indices::<()>(Some(&[]), &[]),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn module_snapshots_reject_truncation_paths_controls_and_invalid_utf16() {
        assert_eq!(
            module_name(&[65, 46, 68, 76, 76, 0], 5),
            Some("a.dll".into())
        );
        for buffer in [
            vec![0],
            vec![65],
            vec![0xD800, 0],
            vec![10, 0],
            vec![92, 0],
            vec![47, 0],
            vec![58, 0],
        ] {
            assert!(module_name(&buffer, buffer.len().saturating_sub(1)).is_none());
        }
        assert!(module_name(&[65, 66], 2).is_none());
        assert!(module_name(&[65, 0], 3).is_none());
    }

    #[test]
    fn vulkan_failure_payload_preserves_raw_results_and_unknown_versions() -> io::Result<()> {
        let error = check(vk::Result::ERROR_INCOMPATIBLE_DRIVER)
            .err()
            .ok_or_else(invalid)?;
        let context = error
            .get_ref()
            .and_then(|error| error.downcast_ref::<VulkanQueryFailure>())
            .ok_or_else(invalid)?;
        assert_eq!(context.stage, "device-query");
        assert_eq!(context.vulkan_result, Some(-9));
        assert_eq!(context.loader_api_version, None);
        assert_eq!(context.application_api_version, None);
        assert_eq!(context.graphics_adapter_count, None);
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert_eq!(
            error.to_string(),
            "Vulkan inventory failed at device-query (VkResult -9)"
        );
        Ok(())
    }

    #[test]
    fn graphics_count_failure_never_exposes_undefined_adapter_count() {
        let success = graphics_count_result(0, 2);
        assert_eq!(success.ntstatus, Some(0));
        assert_eq!(success.windows_error, None);
        assert_eq!(success.adapter_count, Some(2));
        assert_eq!(graphics_count_result(0, 0).adapter_count, Some(0));
        for status in [-1, i32::MIN, 1] {
            let failure = graphics_count_result(status, u32::MAX);
            assert_eq!(failure.stage, "count-query");
            assert_eq!(failure.windows_error, None);
            assert_eq!(failure.ntstatus, Some(status));
            assert_eq!(failure.adapter_count, None);
        }
    }

    #[test]
    fn vulkan_names_and_device_types_preserve_unqualified_boundaries() -> io::Result<()> {
        assert!(check(vk::Result::SUCCESS).is_ok());
        assert!(check(vk::Result::INCOMPLETE).is_err());
        assert!(check(vk::Result::ERROR_INCOMPATIBLE_DRIVER).is_err());
        assert_eq!(text(CStr::from_bytes_until_nul(b"A\0B"))?, "A");
        assert!(text(CStr::from_bytes_until_nul(b"AB")).is_err());
        assert!(text(CStr::from_bytes_until_nul(b"\n\0")).is_err());
        assert_eq!(device_type(vk::PhysicalDeviceType::CPU), ("cpu", false));
        assert_eq!(
            device_type(vk::PhysicalDeviceType::VIRTUAL_GPU),
            ("virtual-gpu", false)
        );
        assert_eq!(
            device_type(vk::PhysicalDeviceType::INTEGRATED_GPU),
            ("integrated-gpu", true)
        );
        Ok(())
    }

    #[test]
    fn vulkan_feature_structures_require_application_and_device_eligibility() {
        assert_eq!(
            feature_structures(vk::API_VERSION_1_1, vk::API_VERSION_1_3, &[]),
            Some((false, false))
        );
        assert_eq!(
            feature_structures(vk::API_VERSION_1_3, vk::API_VERSION_1_1, &[]),
            Some((false, false))
        );
        assert_eq!(
            feature_structures(vk::API_VERSION_1_3, vk::API_VERSION_1_3, &[]),
            Some((true, true))
        );
        assert_eq!(
            feature_structures(vk::API_VERSION_1_0, vk::API_VERSION_1_3, &[]),
            None
        );
        assert_eq!(
            feature_structures(
                vk::API_VERSION_1_1,
                vk::API_VERSION_1_1,
                &[
                    "VK_KHR_shader_float16_int8".into(),
                    "VK_KHR_shader_integer_dot_product".into()
                ]
            ),
            Some((true, true))
        );
    }
}
