//! Bounded `OpenCL` 1.x information queries; callers own the driver-call deadline.

use std::ffi::{CStr, c_void};
use std::io;
use std::path::PathBuf;

use super::library::SystemLibrary;
use crate::{OpenClDevice, OpenClInventory, OpenClPlatform};

pub(crate) mod projection;

const MAX_PLATFORMS: usize = 8;
const MAX_DEVICES: usize = 16;
const MAX_TEXT: usize = 16 * 1024;
const GPU: u64 = 1 << 2;
const PLATFORM_NOT_FOUND: i32 = -1001;
const DEVICE_NOT_FOUND: i32 = -1;

type Id = *mut c_void;
type Platforms = unsafe extern "system" fn(u32, *mut Id, *mut u32) -> i32;
type PlatformInfo = unsafe extern "system" fn(Id, u32, usize, *mut c_void, *mut usize) -> i32;
type Devices = unsafe extern "system" fn(Id, u64, u32, *mut Id, *mut u32) -> i32;
type DeviceInfo = unsafe extern "system" fn(Id, u32, usize, *mut c_void, *mut usize) -> i32;

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "unsupported OpenCL inventory")
}

fn check(code: i32) -> io::Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(io::Error::other(format!("OpenCL query failed ({code})")))
    }
}

fn bounded_count(count: u32, maximum: usize) -> io::Result<usize> {
    usize::try_from(count)
        .ok()
        .filter(|count| *count <= maximum)
        .ok_or_else(invalid)
}

fn text(bytes: &[u8], written: usize) -> io::Result<String> {
    let bytes = bytes.get(..written).ok_or_else(invalid)?;
    let value = CStr::from_bytes_with_nul(bytes)
        .map_err(|_| invalid())?
        .to_str()
        .map_err(|_| invalid())?;
    if value.chars().any(char::is_control) {
        return Err(invalid());
    }
    Ok(value.to_owned())
}

fn extensions(value: &str) -> io::Result<Vec<String>> {
    let mut names = Vec::new();
    for name in value.split_ascii_whitespace() {
        if names.len() == 512 || name.len() > 255 || !name.is_ascii() {
            return Err(invalid());
        }
        names.push(name.to_owned());
    }
    names.sort_unstable();
    if names.array_windows().any(|[a, b]| a == b) {
        return Err(invalid());
    }
    Ok(names)
}

struct Api {
    platforms: Platforms,
    platform_info: PlatformInfo,
    devices: Devices,
    device_info: DeviceInfo,
    path: PathBuf,
    _library: SystemLibrary,
}

impl Api {
    fn load() -> io::Result<Self> {
        let library =
            SystemLibrary::load_system32("OpenCL.dll", "system OpenCL loader unavailable")?;
        let path = library.path().to_path_buf();
        Ok(Self {
            platforms: super::library::load_symbol!(
                library,
                c"clGetPlatformIDs",
                Platforms,
                |_| { invalid() }
            ),
            platform_info: super::library::load_symbol!(
                library,
                c"clGetPlatformInfo",
                PlatformInfo,
                |_| invalid()
            ),
            devices: super::library::load_symbol!(library, c"clGetDeviceIDs", Devices, |_| {
                invalid()
            }),
            device_info: super::library::load_symbol!(
                library,
                c"clGetDeviceInfo",
                DeviceInfo,
                |_| invalid()
            ),
            path,
            _library: library,
        })
    }

    fn string(&self, id: Id, field: u32, platform: bool) -> io::Result<String> {
        let mut storage = [0_u8; MAX_TEXT];
        let mut written = 0;
        let query = if platform {
            self.platform_info
        } else {
            self.device_info
        };
        // SAFETY: The id belongs to this still-loaded implementation. OpenCL
        // string fields write bytes, bounded by storage; written has native size.
        check(unsafe {
            query(
                id,
                field,
                storage.len(),
                storage.as_mut_ptr().cast(),
                &raw mut written,
            )
        })?;
        text(&storage, written)
    }

    fn integer<const N: usize>(&self, id: Id, field: u32) -> io::Result<[u8; N]> {
        // Eight-byte alignment covers every queried cl_uint/cl_ulong. A typed
        // u64 supplies aligned native output without casting a byte allocation.
        let mut storage = 0_u64;
        let mut written = 0;
        if N > size_of::<u64>() {
            return Err(invalid());
        }
        // SAFETY: The enumerated id and loader remain live. The aligned output
        // has eight bytes and the field's exact advertised size is N (4 or 8).
        check(unsafe {
            (self.device_info)(id, field, N, (&raw mut storage).cast(), &raw mut written)
        })?;
        if written != N {
            return Err(invalid());
        }
        storage.to_ne_bytes()[..N].try_into().map_err(|_| invalid())
    }

    fn u32(&self, id: Id, field: u32) -> io::Result<u32> {
        self.integer(id, field).map(u32::from_ne_bytes)
    }

    fn u64(&self, id: Id, field: u32) -> io::Result<u64> {
        self.integer(id, field).map(u64::from_ne_bytes)
    }

    fn boolean(&self, id: Id, field: u32) -> io::Result<bool> {
        bool::try_from(self.u32(id, field)?).map_err(|_| invalid())
    }

    fn device(&self, id: Id) -> io::Result<OpenClDevice> {
        if id.is_null() || self.u64(id, 0x1000)? != GPU {
            return Err(invalid());
        }
        let address_bits = self.u32(id, 0x100D)?;
        if !matches!(address_bits, 32 | 64) {
            return Err(invalid());
        }
        Ok(OpenClDevice {
            name: self.string(id, 0x102B, false)?,
            vendor: self.string(id, 0x102C, false)?,
            vendor_id: self.u32(id, 0x1001)?,
            version: self.string(id, 0x102F, false)?,
            driver_version: self.string(id, 0x102D, false)?,
            opencl_c_version: self.string(id, 0x103D, false)?,
            available: self.boolean(id, 0x1027)?,
            compiler_available: self.boolean(id, 0x1028)?,
            address_bits,
            global_memory_bytes: self.u64(id, 0x101F)?,
            maximum_allocation_bytes: self.u64(id, 0x1010)?,
            local_memory_bytes: self.u64(id, 0x1023)?,
            extensions: extensions(&self.string(id, 0x1030, false)?)?,
        })
    }

    fn platform(&self, id: Id, remaining: usize) -> io::Result<OpenClPlatform> {
        if id.is_null() {
            return Err(invalid());
        }
        let mut devices = [std::ptr::null_mut(); MAX_DEVICES];
        let mut count = 0;
        // SAFETY: Valid enumerated platform, fixed initialized output for sixteen
        // device IDs and a native u32 count. Only GPU devices are requested.
        let code = unsafe { (self.devices)(id, GPU, 16, devices.as_mut_ptr(), &raw mut count) };
        let count = if code == DEVICE_NOT_FOUND {
            0
        } else {
            check(code)?;
            bounded_count(count, remaining)?
        };
        Ok(OpenClPlatform {
            name: self.string(id, 0x0902, true)?,
            vendor: self.string(id, 0x0903, true)?,
            version: self.string(id, 0x0901, true)?,
            devices: devices[..count]
                .iter()
                .map(|id| self.device(*id))
                .collect::<io::Result<_>>()?,
        })
    }
}

fn validate_environment() -> io::Result<()> {
    if !cfg!(target_arch = "aarch64")
        || std::env::vars_os().any(|(name, _)| {
            let bytes = name.as_encoded_bytes();
            [b"OCL_".as_slice(), b"OPENCL_".as_slice(), b"CL_".as_slice()]
                .iter()
                .any(|prefix| {
                    bytes
                        .get(..prefix.len())
                        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
                })
        })
    {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native ARM64 OpenCL access rejects environment overrides",
        ));
    }
    Ok(())
}

pub(crate) fn query() -> io::Result<OpenClInventory> {
    validate_environment()?;
    let api = Api::load()?;
    let mut platforms = [std::ptr::null_mut(); MAX_PLATFORMS];
    let mut count = 0;
    // SAFETY: The loader remains owned; initialized output accommodates eight
    // platform IDs and a native count. OpenCL reports the total available count.
    let code = unsafe { (api.platforms)(8, platforms.as_mut_ptr(), &raw mut count) };
    if code == PLATFORM_NOT_FOUND {
        return Ok(OpenClInventory {
            loader_path: api.path,
            platforms: Vec::new(),
        });
    }
    check(code)?;
    let count = bounded_count(count, MAX_PLATFORMS)?;
    let mut remaining = MAX_DEVICES;
    let platforms = platforms[..count]
        .iter()
        .map(|id| {
            let platform = api.platform(*id, remaining)?;
            remaining -= platform.devices.len();
            Ok(platform)
        })
        .collect::<io::Result<_>>()?;
    Ok(OpenClInventory {
        loader_path: api.path,
        platforms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opencl_inventory_rejects_unbounded_or_malformed_native_properties() -> io::Result<()> {
        assert_eq!(bounded_count(16, 16)?, 16);
        assert!(bounded_count(17, 16).is_err());
        assert_eq!(text(b"Adreno\0", 7)?, "Adreno");
        for (bytes, written) in [
            (&b"\0"[..], 0),
            (b"name", 4),
            (b"a\0b\0", 4),
            (b"\xFF\0", 2),
            (b"a\0", 3),
        ] {
            assert!(text(bytes, written).is_err());
        }
        assert_eq!(extensions("cl_z cl_a")?, ["cl_a", "cl_z"]);
        assert!(extensions("cl_a cl_a").is_err());
        assert!(extensions(&"x".repeat(256)).is_err());
        assert!(
            extensions(
                &(0..513)
                    .map(|index| format!("cl_{index}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
            .is_err()
        );
        assert!(check(-30).is_err());
        Ok(())
    }
}
