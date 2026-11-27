//! Exact-version MXC PSEC composition with the runner's atomic Windows Job.
//!
//! MXC 1.0.1's supported spawn facade cannot accept our Job or resource limits.
//! These explicitly pinned doc-hidden compatibility exports are intentionally
//! confined here. Review this boundary and its fixed `FlatBuffer` policy whenever
//! either dependency changes; never use SDK dispatch, audit or fallback tiers.

#![deny(
    clippy::non_send_fields_in_send_ty,
    reason = "This module audits native handle thread safety independently of broad Clippy category defaults."
)]

use std::collections::{BTreeMap, HashSet};
use std::ffi::OsString;
use std::io;
use std::os::windows::ffi::OsStrExt as _;
use std::os::windows::fs::MetadataExt as _;
use std::path::{Component, Path, PathBuf, Prefix};

use mxc_sdk::process_container_common::{
    ProcessSecurityEnvironment, ProcessSecurityEnvironmentApi,
    base_container_runner::BaseContainerRunner,
};
use mxc_sdk::process_security_environment_spec::process_security_environment_layout::{
    EndpointPolicyT, FilterAction, IngressPolicyT, NetworkPolicyT, ProcessSecurityEnvironmentT,
    SchemaVersionT, finish_process_security_environment_buffer,
};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
use windows_sys::Win32::System::SystemInformation::GetWindowsDirectoryW;

use crate::{ProcessSpec, SandboxNetwork, SandboxPolicy, invalid, validate_path};

// PROCESS_SECURITY_ENVIRONMENT_FLAG_TERMINATE_ON_CLOSE from processmodel.h.
// Keep native enforcement owned even when brokered creation bypasses Job
// inheritance. Never cancel this flag or substitute it for process/resource caps.
// https://learn.microsoft.com/windows/win32/api/processmodel/ne-processmodel-process_security_environment_flags
const TERMINATE_ON_CLOSE: u32 = 0x1;

// The pinned Swift runtime contains 63 files before the model, backend key and
// optional adapter are added. Keep a finite policy budget large enough for that
// complete verified bundle; every grant still names one validated existing file.
const MAX_READONLY_GRANTS: usize = 128;

/// Owns only a native PSEC handle; creation and policy remain immutable thereafter.
pub(super) struct Environment(ProcessSecurityEnvironment);

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "The upstream owner wraps a process-wide opaque Windows handle; the safety contract is documented here."
)]
// SAFETY: PSEC handles are process-wide native handles, not thread-affine COM
// objects. Moving the unique owner does not transfer or duplicate ownership.
// The upstream close function and resident system DLL remain live process-wide.
unsafe impl Send for Environment {}
// SAFETY: Shared access exposes only a borrowed, immutable handle for process
// creation. Native close needs exclusive ownership through Drop; it cannot race
// any safe borrow, and this wrapper never mutates the environment after creation.
unsafe impl Sync for Environment {}

impl Environment {
    pub(super) fn raw(&self) -> HANDLE {
        self.0.raw().0
    }

    pub(super) fn create(
        spec: &ProcessSpec,
        policy: &SandboxPolicy,
        network: SandboxNetwork,
    ) -> io::Result<Self> {
        let readonly = validate_policy(spec, policy)?;
        let api = available_api(network)?;
        let bytes = policy_bytes(
            readonly,
            Some(path_text(&policy.writable_directory)?),
            network,
        );
        api.create(&bytes, TERMINATE_ON_CLOSE)
            .map(Self)
            .map_err(native_error)
    }
}

pub(crate) fn available(network: SandboxNetwork) -> io::Result<()> {
    let api = available_api(network)?;
    // API exports alone do not prove an enabled, usable native implementation.
    // Creating the same network/UI posture exercises actual PSEC policy admission
    // without a child, credentials, filesystem grants or host ACL changes.
    let environment = api
        .create(&policy_bytes(Vec::new(), None, network), TERMINATE_ON_CLOSE)
        .map_err(native_error)?;
    drop(environment);
    Ok(())
}

fn available_api(network: SandboxNetwork) -> io::Result<ProcessSecurityEnvironmentApi> {
    let api = ProcessSecurityEnvironmentApi::load().map_err(native_error)?;
    if !api.supports_version(1, 1).map_err(native_error)? {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native sandbox requires PSEC 1.1 support",
        ));
    }
    if network == SandboxNetwork::HostLoopback
        && !BaseContainerRunner::supports_ingress_host_loopback_allow()
    {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native sandbox requires PSEC 1.1 and host-loopback ingress support",
        ));
    }
    Ok(api)
}

fn native_error(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        format!("native PSEC containment: {error}"),
    )
}

pub(super) fn child_environment(policy: &SandboxPolicy) -> io::Result<Vec<u16>> {
    child_environment_from(policy, std::env::vars_os())
}

pub(super) fn minimal_child_environment(policy: &SandboxPolicy) -> io::Result<Vec<u16>> {
    minimal_environment_from(policy, &windows_directory()?)
}

fn minimal_environment_from(policy: &SandboxPolicy, windows: &Path) -> io::Result<Vec<u16>> {
    let Some(Component::Prefix(prefix)) = windows.components().next() else {
        return Err(invalid("minimal worker requires a local Windows directory"));
    };
    let Prefix::Disk(drive) = prefix.kind() else {
        return Err(invalid(
            "minimal worker requires a drive-letter Windows directory",
        ));
    };
    let windows = path_text(windows)?;
    let scratch = path_text(&policy.writable_directory)?;
    // No inherited entry is consulted, including entries with unknown secret
    // names. Runtime paths come from Windows rather than caller-controlled
    // SystemRoot/windir values. PSEC requires LOCALAPPDATA to be present even
    // for an offline worker (mxc-sdk 1.0.1 launch_diagnostics.rs). Bind that
    // presence-only requirement to private scratch, never the caller's profile.
    // The ordinary worker environment is unchanged.
    let entries = [
        (OsString::from("SystemRoot"), OsString::from(&windows)),
        (OsString::from("windir"), OsString::from(windows)),
        (
            OsString::from("SystemDrive"),
            OsString::from(format!("{}:", char::from(drive))),
        ),
        (OsString::from("LOCALAPPDATA"), OsString::from(&scratch)),
        (OsString::from("TEMP"), OsString::from(&scratch)),
        (OsString::from("TMP"), OsString::from(scratch)),
    ];
    // Admit the exact child names, not caller entries that will never reach it.
    crate::validate_sandboxed_environment_names(entries.iter().map(|(name, _)| name.clone()))?;
    super::environment::build(entries, &BTreeMap::new())
}

fn child_environment_from(
    policy: &SandboxPolicy,
    inherited: impl IntoIterator<Item = (OsString, OsString)>,
) -> io::Result<Vec<u16>> {
    // Admit exactly the snapshot that CreateProcessW will receive. A separate
    // caller preflight cannot cover environment changes before worker creation.
    let inherited: Vec<_> = inherited.into_iter().collect();
    for (name, _) in &inherited {
        crate::validate_sandbox_environment_name(name)?;
    }
    let scratch = path_text(&policy.writable_directory)?;
    super::environment::build(
        inherited,
        &BTreeMap::from([
            ("TEMP".to_owned(), scratch.clone()),
            ("TMP".to_owned(), scratch),
        ]),
    )
}

fn validate_policy(spec: &ProcessSpec, policy: &SandboxPolicy) -> io::Result<Vec<String>> {
    if policy.readonly_paths.is_empty() || policy.readonly_paths.len() > MAX_READONLY_GRANTS {
        return Err(invalid(
            "native sandbox requires 1..128 explicit read-only grants",
        ));
    }
    let mut checked_ancestors = HashSet::new();
    let writable_meta =
        validate_local_path_cached(&policy.writable_directory, &mut checked_ancestors)?;
    let writable_ancestors_u16 = encode_ancestors_utf16(&policy.writable_directory)?;
    let writable_u16 = writable_ancestors_u16
        .first()
        .ok_or_else(|| invalid("sandbox working directory must be its existing private scratch"))?;
    let working_u16 = encode_utf16_path(&spec.working_directory)?;
    if !writable_meta.is_dir() || !same_utf16_path(&working_u16, writable_u16)? {
        return Err(invalid(
            "sandbox working directory must be its existing private scratch",
        ));
    }
    let mut scratch = Vec::with_capacity(260);
    for output in [&spec.stdout, &spec.stderr] {
        if within_utf16_root(output, writable_u16, &mut scratch)? {
            return Err(invalid(
                "sandbox evidence must remain outside worker scratch",
            ));
        }
    }
    let os_runtime = runtime_paths_cached(&mut checked_ancestors)?;
    let mut readonly: Vec<String> = os_runtime.iter().map(|(text, _)| text.clone()).collect();
    let user_profile_ancestors = std::env::var_os("USERPROFILE")
        .map(|profile| encode_ancestors_utf16(Path::new(&profile)))
        .transpose()?;
    let executable_u16 = encode_utf16_path(&spec.executable)?;
    let mut executable_granted = false;
    let mut path_u16 = Vec::with_capacity(260);
    for path in &policy.readonly_paths {
        let leaf_meta = validate_local_path_cached(path, &mut checked_ancestors)?;
        if !leaf_meta.is_file() {
            return Err(invalid(
                "sandbox read-only grants must name exact existing files",
            ));
        }
        encode_utf16_into(path, &mut path_u16)?;
        if within_utf16_root(path, writable_u16, &mut scratch)?
            || within_encoded_ancestors(&path_u16, &writable_ancestors_u16)?
        {
            return Err(invalid(
                "sandbox read-only and writable grants must not overlap",
            ));
        }
        if let Some(ref profile_ancestors) = user_profile_ancestors
            && within_encoded_ancestors(&path_u16, profile_ancestors)?
        {
            return Err(invalid(
                "sandbox cannot grant an entire user profile or its ancestors",
            ));
        }
        if same_utf16_path(&executable_u16, &path_u16)? {
            executable_granted = true;
        }
        readonly.push(path_text(path)?);
    }
    if !executable_granted {
        return Err(invalid(
            "sandbox executable requires an explicit read-only grant",
        ));
    }
    for (_, os_dir_u16) in &os_runtime {
        if within_encoded_ancestors(os_dir_u16, &writable_ancestors_u16)? {
            return Err(invalid(
                "sandbox scratch cannot overlap an OS runtime grant",
            ));
        }
    }
    readonly.sort_unstable();
    readonly.dedup();
    Ok(readonly)
}

fn validate_local_path(path: &Path) -> io::Result<()> {
    let mut checked = HashSet::new();
    validate_local_path_cached(path, &mut checked).map(|_| ())
}

fn validate_local_path_cached(
    path: &Path,
    checked_ancestors: &mut HashSet<PathBuf>,
) -> io::Result<std::fs::Metadata> {
    validate_path(path)?;
    if !matches!(path.components().next(), Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_)))
        || path.file_name().is_none()
        || path_text(path)?
            .get(2..)
            .is_some_and(|tail| tail.contains(':'))
    {
        return Err(invalid(
            "sandbox grants require non-root local paths without streams",
        ));
    }
    let leaf_meta = std::fs::symlink_metadata(path)?;
    if leaf_meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(invalid("sandbox paths must not contain reparse points"));
    }
    for ancestor in path.ancestors().skip(1) {
        if checked_ancestors.contains(ancestor) {
            break;
        }
        let metadata = std::fs::symlink_metadata(ancestor)?;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(invalid("sandbox paths must not contain reparse points"));
        }
        checked_ancestors.insert(ancestor.to_path_buf());
    }
    Ok(leaf_meta)
}

fn path_text(path: &Path) -> io::Result<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid("sandbox paths must be Unicode"))
}

fn encode_utf16_path(path: &Path) -> io::Result<Vec<u16>> {
    let units: Vec<u16> = path.as_os_str().encode_wide().collect();
    i32::try_from(units.len()).map_err(|_| invalid("sandbox path is too long"))?;
    Ok(units)
}

fn encode_utf16_into(path: &Path, buf: &mut Vec<u16>) -> io::Result<()> {
    buf.clear();
    buf.extend(path.as_os_str().encode_wide());
    i32::try_from(buf.len()).map_err(|_| invalid("sandbox path is too long"))?;
    Ok(())
}

fn same_utf16_path(left: &[u16], right: &[u16]) -> io::Result<bool> {
    if left.len() != right.len() {
        return Ok(false);
    }
    if left == right {
        return Ok(true);
    }
    super::compare_ordinal_ignore_case(left, right, "sandbox path is too long")
        .map(|ordering| ordering.is_eq())
}

fn encode_ancestors_utf16(path: &Path) -> io::Result<Vec<Vec<u16>>> {
    path.ancestors().map(encode_utf16_path).collect()
}

fn within_utf16_root(path: &Path, root_u16: &[u16], scratch: &mut Vec<u16>) -> io::Result<bool> {
    for ancestor in path.ancestors() {
        encode_utf16_into(ancestor, scratch)?;
        if same_utf16_path(scratch, root_u16)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn within_encoded_ancestors(target_u16: &[u16], ancestors_u16: &[Vec<u16>]) -> io::Result<bool> {
    for ancestor_u16 in ancestors_u16 {
        if same_utf16_path(ancestor_u16, target_u16)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn windows_directory() -> io::Result<PathBuf> {
    let windows = super::query_system_dir("Windows directory length overflow", |buf, cap| {
        // SAFETY: Caller provides writable UTF-16 buffer of `cap` elements.
        unsafe { GetWindowsDirectoryW(buf, cap) }
    })?;
    validate_local_path(&windows)?;
    Ok(windows)
}

fn windows_directory_cached(checked_ancestors: &mut HashSet<PathBuf>) -> io::Result<PathBuf> {
    let windows = super::query_system_dir("Windows directory length overflow", |buf, cap| {
        // SAFETY: Caller provides writable UTF-16 buffer of `cap` elements.
        unsafe { GetWindowsDirectoryW(buf, cap) }
    })?;
    let meta = validate_local_path_cached(&windows, checked_ancestors)?;
    if meta.is_dir() {
        checked_ancestors.insert(windows.clone());
    }
    Ok(windows)
}

fn runtime_paths_cached(checked_ancestors: &mut HashSet<PathBuf>) -> io::Result<Vec<(String, Vec<u16>)>> {
    let windows = windows_directory_cached(checked_ancestors)?;
    let mut paths = Vec::new();
    for name in ["System32", "WinSxS", "SysWOW64"] {
        let path = windows.join(name);
        match validate_local_path_cached(&path, checked_ancestors) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
            Ok(meta) => {
                if meta.is_dir() {
                    paths.push((path_text(&path)?, encode_utf16_path(&path)?));
                }
            }
        }
    }
    if paths.is_empty() {
        return Err(invalid("Windows runtime directories are unavailable"));
    }
    Ok(paths)
}

fn policy_bytes(
    readonly: Vec<String>,
    scratch: Option<String>,
    authority: SandboxNetwork,
) -> Vec<u8> {
    let mut egress = EndpointPolicyT::default();
    egress.default_action = FilterAction::deny;
    let mut ingress = IngressPolicyT::default();
    ingress.default_action = FilterAction::deny;
    ingress.host_loopback = match authority {
        SandboxNetwork::HostLoopback => FilterAction::allow,
        SandboxNetwork::Offline => FilterAction::deny,
    };
    let mut network = NetworkPolicyT::default();
    network.egress = Some(Box::new(egress));
    network.ingress = Some(Box::new(ingress));
    // Host loopback is explicitly bidirectional and includes other host-owned
    // service addresses. Native tests show that neither removing the capability/
    // peer nor explicit all-address egress deny rules narrow that local scope.
    // No internet/private-network capability or external allow rule is granted.
    if authority == SandboxNetwork::HostLoopback {
        network.allowed_appcontainer_peer = Some("MXC-Loopback".to_owned());
    }
    let mut spec = ProcessSecurityEnvironmentT::default();
    spec.version = SchemaVersionT { major: 1, minor: 1 };
    if authority == SandboxNetwork::HostLoopback {
        spec.capabilities = Some("networkLoopback".to_owned());
    }
    spec.disallow_win32k_system_calls = true;
    // PSEC enforces these desktop restrictions without nesting a Job UI limit.
    spec.ui_restrictions = 0x03ff;
    spec.fs_read_only = (!readonly.is_empty()).then_some(readonly);
    spec.fs_read_write = scratch.map(|path| vec![path]);
    spec.network_policy = Some(Box::new(network));
    let mut builder = flatbuffers::FlatBufferBuilder::with_capacity(1024);
    let policy = spec.pack(&mut builder);
    finish_process_security_environment_buffer(&mut builder, policy);
    builder.finished_data().to_vec()
}

#[cfg(test)]
mod tests {
    use std::{fs, time::SystemTime};

    use super::*;

    #[test]
    fn minimal_child_environment_contains_required_psec_names_and_only_private_storage()
    -> io::Result<()> {
        let policy = SandboxPolicy {
            readonly_paths: Vec::new(),
            writable_directory: PathBuf::from("R:\\private\\scratch"),
        };
        let block = minimal_environment_from(&policy, Path::new("R:\\Windows"))?;
        let entries: Vec<_> = block
            .split(|unit| *unit == 0)
            .filter(|entry| !entry.is_empty())
            .collect();
        assert_eq!(entries.len(), 6);
        for expected in [
            "SystemRoot=R:\\Windows",
            "windir=R:\\Windows",
            "SystemDrive=R:",
            "LOCALAPPDATA=R:\\private\\scratch",
            "TEMP=R:\\private\\scratch",
            "TMP=R:\\private\\scratch",
        ] {
            assert!(entries.contains(&expected.encode_utf16().collect::<Vec<_>>().as_slice()));
        }
        assert!(block.ends_with(&[0, 0]));
        Ok(())
    }

    #[test]
    fn child_environment_admits_the_retained_snapshot_and_rejects_late_credentials()
    -> io::Result<()> {
        let policy = SandboxPolicy {
            readonly_paths: Vec::new(),
            writable_directory: PathBuf::from("C:\\private\\scratch"),
        };
        let inherited = vec![
            (OsString::from("SystemRoot"), OsString::from("C:\\Windows")),
            (OsString::from("USERPROFILE"), OsString::from("C:\\profile")),
            (OsString::from("TEMP"), OsString::from("C:\\previous-temp")),
            (OsString::from("TMP"), OsString::from("C:\\previous-temp")),
        ];
        let block = child_environment_from(&policy, inherited.clone())?;
        for expected in [
            "SystemRoot=C:\\Windows",
            "USERPROFILE=C:\\profile",
            "TEMP=C:\\private\\scratch",
            "TMP=C:\\private\\scratch",
        ] {
            assert!(
                block.split(|unit| *unit == 0).any(|entry| {
                    entry == expected.encode_utf16().collect::<Vec<_>>().as_slice()
                })
            );
        }
        // A clean earlier snapshot must not authorize a changed child snapshot.
        // Inject it as data: this control never mutates the process environment.
        for name in [
            "OPENAI_API_KEY",
            "gItHuB_ToKeN",
            "ConnectionStrings__Default",
        ] {
            let mut changed = inherited.clone();
            changed.push((OsString::from(name), OsString::from("credential-fixture")));
            let error = child_environment_from(&policy, changed)
                .err()
                .ok_or_else(|| io::Error::other("credential-bearing snapshot admitted"))?;
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(
                error.to_string(),
                "sandbox launch rejects inherited credential environment variables"
            );
        }
        Ok(())
    }

    #[test]
    fn offline_policy_omits_network_authority_and_preserves_headless_restrictions() -> io::Result<()>
    {
        use mxc_sdk::process_security_environment_spec::process_security_environment_layout::root_as_process_security_environment;

        for authority in [SandboxNetwork::Offline, SandboxNetwork::HostLoopback] {
            let bytes = policy_bytes(
                vec!["C:\\approved\\worker.exe".to_owned()],
                Some("C:\\private\\scratch".to_owned()),
                authority,
            );
            let policy = root_as_process_security_environment(&bytes)
                .map_err(io::Error::other)?
                .unpack();
            assert!(policy.disallow_win32k_system_calls);
            assert_eq!(policy.ui_restrictions, 0x03ff);
            assert_eq!(
                policy.fs_read_only,
                Some(vec!["C:\\approved\\worker.exe".to_owned()])
            );
            assert_eq!(
                policy.fs_read_write,
                Some(vec!["C:\\private\\scratch".to_owned()])
            );
            let network = policy
                .network_policy
                .ok_or_else(|| invalid("missing network policy"))?;
            let ingress = network
                .ingress
                .ok_or_else(|| invalid("missing ingress policy"))?;
            let egress = network
                .egress
                .ok_or_else(|| invalid("missing egress policy"))?;
            assert_eq!(egress.default_action, FilterAction::deny);
            assert_eq!(ingress.default_action, FilterAction::deny);
            match authority {
                SandboxNetwork::Offline => {
                    assert_eq!(ingress.host_loopback, FilterAction::deny);
                    assert!(network.allowed_appcontainer_peer.is_none());
                    assert!(policy.capabilities.is_none());
                }
                SandboxNetwork::HostLoopback => {
                    assert_eq!(ingress.host_loopback, FilterAction::allow);
                    assert_eq!(
                        network.allowed_appcontainer_peer.as_deref(),
                        Some("MXC-Loopback")
                    );
                    assert_eq!(policy.capabilities.as_deref(), Some("networkLoopback"));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn verified_runtime_grant_budget_admits_swift_bundle_and_rejects_overflow() -> io::Result<()> {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "Bonsai-SandboxGrantBudget-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&directory)?;
        let scratch = directory.join("worker");
        fs::create_dir(&scratch)?;
        let specification = ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: Vec::new(),
            working_directory: scratch.clone(),
            stdout: directory.join("backend.stdout"),
            stderr: directory.join("backend.stderr"),
            memory_limit_bytes: 256 * 1024 * 1024,
            active_process_limit: 1,
        };
        let mut policy = SandboxPolicy {
            readonly_paths: vec![specification.executable.clone()],
            writable_directory: scratch,
        };
        // Model the complete 63-file pinned runtime plus model, key and adapter.
        // Distinct real files exercise ancestry, overlap and file-kind validation.
        for index in 1..=MAX_READONLY_GRANTS {
            let file = directory.join(format!("verified-{index}.bin"));
            fs::write(&file, b"verified fixture")?;
            policy.readonly_paths.push(file);
            if matches!(policy.readonly_paths.len(), 66 | MAX_READONLY_GRANTS) {
                validate_policy(&specification, &policy)?;
            }
        }
        let error = validate_policy(&specification, &policy)
            .err()
            .ok_or_else(|| io::Error::other("over-budget sandbox grants accepted"))?;
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(!specification.stdout.exists());
        assert!(!specification.stderr.exists());
        Ok(())
    }

    #[test]
    fn cached_utf16_containment_checks_handle_case_ancestry_and_overlap() -> io::Result<()> {
        let a_u16 = encode_utf16_path(Path::new("C:\\Alpha\\Beta"))?;
        let b_u16 = encode_utf16_path(Path::new("c:\\alpha\\beta"))?;
        let c_u16 = encode_utf16_path(Path::new("C:\\Alpha\\Beta\\Gamma"))?;
        assert!(same_utf16_path(&a_u16, &b_u16)?);
        assert!(!same_utf16_path(&a_u16, &c_u16)?);

        let anc_u16 = encode_ancestors_utf16(Path::new("C:\\Alpha\\Beta\\Gamma"))?;
        assert!(within_encoded_ancestors(&b_u16, &anc_u16)?);
        let mut scratch = Vec::new();
        assert!(within_utf16_root(
            Path::new("C:\\alpha\\beta\\file.bin"),
            &a_u16,
            &mut scratch
        )?);
        assert!(!within_utf16_root(
            Path::new("C:\\alpha\\other\\file.bin"),
            &a_u16,
            &mut scratch
        )?);
        Ok(())
    }
}
