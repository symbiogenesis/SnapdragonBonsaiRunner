//! Read-only, bounded `SetupAPI` inventories; presence does not establish signed deployment.

use std::io;
use std::ptr;
use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    DICS_FLAG_GLOBAL, DIGCF_PRESENT, DIREG_DRV, GUID_DEVCLASS_COMPUTEACCELERATOR,
    GUID_DEVCLASS_DISPLAY, GUID_DEVCLASS_MEDIA, HDEVINFO, SP_DEVINFO_DATA, SPDRP_DEVICEDESC,
    SPDRP_MFG, SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInfo, SetupDiGetClassDevsW,
    SetupDiGetDeviceInstanceIdW, SetupDiGetDeviceRegistryPropertyW, SetupDiOpenDevRegKey,
};
use windows_sys::Win32::Foundation::{ERROR_NO_MORE_ITEMS, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Registry::{KEY_READ, REG_SZ, RRF_RT_REG_SZ, RegGetValueW};

struct Set(HDEVINFO);
impl Drop for Set {
    fn drop(&mut self) {
        // SAFETY: The SetupAPI set is owned exactly once and outlives every use.
        let _ = unsafe { SetupDiDestroyDeviceInfoList(self.0) };
    }
}
fn text(buffer: &[u16]) -> io::Result<String> {
    let units = super::utf16_z(buffer, "Unterminated device property.")?;
    let text =
        String::from_utf16(units).map_err(|_| super::invalid("Invalid device property UTF-16."))?;
    if text.chars().any(char::is_control) {
        return Err(super::invalid(
            "Device property contains control characters.",
        ));
    }
    Ok(text)
}
fn decode_utf16_bytes(buffer: &[u16], bytes: u32) -> Option<String> {
    if bytes < 2 || !bytes.is_multiple_of(2) {
        return None;
    }
    let units = usize::try_from(bytes / 2)
        .ok()
        .filter(|units| *units <= buffer.len())?;
    text(&buffer[..units]).ok()
}
fn property(set: &Set, entry: &SP_DEVINFO_DATA, property: u32) -> Option<String> {
    let mut buffer = [0_u16; 2048];
    let mut kind = 0;
    let mut required = 0_u32;
    // SAFETY: The live set/entry are borrowed and the property destination is
    // writable for exactly the bounded UTF-16 array's byte size.
    let read = unsafe {
        SetupDiGetDeviceRegistryPropertyW(
            set.0,
            entry,
            property,
            &raw mut kind,
            buffer.as_mut_ptr().cast(),
            4096,
            &raw mut required,
        )
    };
    if read == 0 || kind != REG_SZ {
        return None;
    }
    decode_utf16_bytes(&buffer, required)
}
fn registry(key: &super::RegistryKey, name: &str) -> Option<String> {
    let name = super::wide(name);
    let mut buffer = [0_u16; 2048];
    let mut size = 4096_u32;
    // SAFETY: The owned key and terminated value name remain live; the writable
    // destination has the declared byte capacity. RegGetValue ensures termination.
    let status = unsafe {
        RegGetValueW(
            key.0,
            ptr::null(),
            name.as_ptr(),
            RRF_RT_REG_SZ,
            ptr::null_mut(),
            buffer.as_mut_ptr().cast(),
            &raw mut size,
        )
    };
    if status != 0 {
        return None;
    }
    decode_utf16_bytes(&buffer, size)
}

pub(crate) fn inventory(class: &str) -> io::Result<Vec<crate::DeviceInventoryEntry>> {
    let guid = match class {
        "ComputeAccelerator" => GUID_DEVCLASS_COMPUTEACCELERATOR,
        "Display" => GUID_DEVCLASS_DISPLAY,
        "Media" => GUID_DEVCLASS_MEDIA,
        _ => return Err(super::invalid("Unsupported read-only device class.")),
    };
    // SAFETY: The static GUID is read-only and no optional enumerator/window is supplied.
    let raw = unsafe {
        SetupDiGetClassDevsW(&raw const guid, ptr::null(), ptr::null_mut(), DIGCF_PRESENT)
    };
    if raw == -1 {
        return Err(io::Error::last_os_error());
    }
    let set = Set(raw);
    let mut devices = Vec::new();
    for index in 0..=16 {
        let mut entry = SP_DEVINFO_DATA {
            cbSize: super::win_size::<SP_DEVINFO_DATA>(),
            ..SP_DEVINFO_DATA::default()
        };
        // SAFETY: set is live and entry has its initialized exact structure size.
        if unsafe { SetupDiEnumDeviceInfo(set.0, index, &raw mut entry) } == 0 {
            let error = io::Error::last_os_error();
            return if error.raw_os_error() == i32::try_from(ERROR_NO_MORE_ITEMS).ok() {
                Ok(devices)
            } else {
                Err(error)
            };
        }
        if index == 16 {
            return Err(super::invalid(
                "Device inventory exceeds sixteen present entries.",
            ));
        }
        let mut id = [0_u16; 2048];
        // SAFETY: The live set/entry and writable bounded UTF-16 output remain allocated.
        super::check(unsafe {
            SetupDiGetDeviceInstanceIdW(
                set.0,
                &raw const entry,
                id.as_mut_ptr(),
                2048,
                ptr::null_mut(),
            )
        })?;
        // SAFETY: The selected live device's driver key is opened read-only.
        let raw_key = unsafe {
            SetupDiOpenDevRegKey(
                set.0,
                &raw const entry,
                DICS_FLAG_GLOBAL,
                0,
                DIREG_DRV,
                KEY_READ,
            )
        };
        let key = (raw_key != INVALID_HANDLE_VALUE && !raw_key.is_null())
            .then(|| super::RegistryKey(raw_key));
        devices.push(crate::DeviceInventoryEntry {
            name: property(&set, &entry, SPDRP_DEVICEDESC),
            manufacturer: property(&set, &entry, SPDRP_MFG),
            device_id: text(&id)?,
            driver_version: key.as_ref().and_then(|key| registry(key, "DriverVersion")),
            driver_date: key.as_ref().and_then(|key| registry(key, "DriverDate")),
            inf_name: key.as_ref().and_then(|key| registry(key, "InfPath")),
        });
    }
    Err(super::invalid("Device inventory bound was exceeded."))
}

#[cfg(test)]
mod tests {
    use super::text;

    #[test]
    fn device_text_rejects_control_characters_and_invalid_utf16() {
        assert_eq!(
            text(&[u16::from(b'Q'), u16::from(b'u'), 0]).ok().as_deref(),
            Some("Qu")
        );
        for control in [0x0001_u16, 0x000A, 0x000D, 0x001B, 0x007F] {
            assert!(text(&[u16::from(b'A'), control, u16::from(b'B'), 0]).is_err());
        }
        assert!(text(&[u16::from(b'A'), u16::from(b'B')]).is_err());
        assert!(text(&[0xD800_u16, 0]).is_err());
    }
}
